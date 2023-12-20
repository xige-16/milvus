// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package indexnode

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/cockroachdb/errors"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/indexpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/indexcgowrapper"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/hardware"
	"github.com/milvus-io/milvus/pkg/util/indexparamcheck"
	"github.com/milvus-io/milvus/pkg/util/indexparams"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
)

var (
	errCancel      = fmt.Errorf("canceled")
	diskUsageRatio = 4.0
)

type Blob = storage.Blob

type taskInfo struct {
	cancel              context.CancelFunc
	state               commonpb.IndexState
	fileKeys            []string
	serializedSize      uint64
	failReason          string
	currentIndexVersion int32
	indexStoreVersion   int64

	// task statistics
	statistic *indexpb.JobInfo
}

type task interface {
	Ctx() context.Context
	Name() string
	Prepare(context.Context) error
	LoadData(context.Context) error
	BuildIndex(context.Context) error
	SaveIndexFiles(context.Context) error
	OnEnqueue(context.Context) error
	SetState(state commonpb.IndexState, failReason string)
	GetState() commonpb.IndexState
	Reset()
}

type indexBuildTaskV2 struct {
	*indexBuildTask
}

func (it *indexBuildTaskV2) parseParams(ctx context.Context) error {
	it.collectionID = it.req.CollectionID
	it.partitionID = it.req.PartitionID
	it.segmentID = it.req.SegmentID
	it.fieldType = it.req.FieldType
	it.fieldID = it.req.FieldID
	it.fieldName = it.req.FieldName
	return nil
}

func (it *indexBuildTaskV2) BuildIndex(ctx context.Context) error {
	err := it.parseParams(ctx)
	if err != nil {
		log.Ctx(ctx).Warn("parse field meta from binlog failed", zap.Error(err))
		return err
	}

	indexType := it.newIndexParams[common.IndexTypeKey]
	if indexType == indexparamcheck.IndexDISKANN {
		// check index node support disk index
		if !Params.IndexNodeCfg.EnableDisk.GetAsBool() {
			log.Ctx(ctx).Warn("IndexNode don't support build disk index",
				zap.String("index type", it.newIndexParams[common.IndexTypeKey]),
				zap.Bool("enable disk", Params.IndexNodeCfg.EnableDisk.GetAsBool()))
			return merr.WrapErrIndexNotSupported("disk index")
		}

		// check load size and size of field data
		localUsedSize, err := indexcgowrapper.GetLocalUsedSize(paramtable.Get().LocalStorageCfg.Path.GetValue())
		if err != nil {
			log.Ctx(ctx).Warn("IndexNode get local used size failed")
			return err
		}
		fieldDataSize, err := estimateFieldDataSize(it.statistic.Dim, it.req.GetNumRows(), it.fieldType)
		if err != nil {
			log.Ctx(ctx).Warn("IndexNode get local used size failed")
			return err
		}
		usedLocalSizeWhenBuild := int64(float64(fieldDataSize)*diskUsageRatio) + localUsedSize
		maxUsedLocalSize := int64(Params.IndexNodeCfg.DiskCapacityLimit.GetAsFloat() * Params.IndexNodeCfg.MaxDiskUsagePercentage.GetAsFloat())

		if usedLocalSizeWhenBuild > maxUsedLocalSize {
			log.Ctx(ctx).Warn("IndexNode don't has enough disk size to build disk ann index",
				zap.Int64("usedLocalSizeWhenBuild", usedLocalSizeWhenBuild),
				zap.Int64("maxUsedLocalSize", maxUsedLocalSize))
			return merr.WrapErrServiceDiskLimitExceeded(float32(usedLocalSizeWhenBuild), float32(maxUsedLocalSize))
		}

		err = indexparams.SetDiskIndexBuildParams(it.newIndexParams, int64(fieldDataSize))
		if err != nil {
			log.Ctx(ctx).Warn("failed to fill disk index params", zap.Error(err))
			return err
		}
	}

	var buildIndexInfo *indexcgowrapper.BuildIndexInfo
	buildIndexInfo, err = indexcgowrapper.NewBuildIndexInfo(it.req.GetStorageConfig())
	defer indexcgowrapper.DeleteBuildIndexInfo(buildIndexInfo)
	if err != nil {
		log.Ctx(ctx).Warn("create build index info failed", zap.Error(err))
		return err
	}
	err = buildIndexInfo.AppendFieldMetaInfoV2(it.collectionID, it.partitionID, it.segmentID, it.fieldID, it.fieldType, it.fieldName, it.req.Dim)
	if err != nil {
		log.Ctx(ctx).Warn("append field meta failed", zap.Error(err))
		return err
	}

	err = buildIndexInfo.AppendIndexMetaInfo(it.req.IndexID, it.req.BuildID, it.req.IndexVersion)
	if err != nil {
		log.Ctx(ctx).Warn("append index meta failed", zap.Error(err))
		return err
	}

	err = buildIndexInfo.AppendBuildIndexParam(it.newIndexParams)
	if err != nil {
		log.Ctx(ctx).Warn("append index params failed", zap.Error(err))
		return err
	}

	err = buildIndexInfo.AppendIndexStorageInfo(it.req.StorePath, it.req.IndexStorePath, it.req.StoreVersion)
	if err != nil {
		log.Ctx(ctx).Warn("append storage info failed", zap.Error(err))
		return err
	}

	jsonIndexParams, err := json.Marshal(it.newIndexParams)
	if err != nil {
		log.Ctx(ctx).Error("failed to json marshal index params", zap.Error(err))
		return err
	}

	log.Ctx(ctx).Info("index params are ready",
		zap.Int64("buildID", it.BuildID),
		zap.String("index params", string(jsonIndexParams)))

	err = buildIndexInfo.AppendBuildTypeParam(it.newTypeParams)
	if err != nil {
		log.Ctx(ctx).Warn("append type params failed", zap.Error(err))
		return err
	}

	it.index, err = indexcgowrapper.CreateIndexV2(ctx, buildIndexInfo)
	if err != nil {
		if it.index != nil && it.index.CleanLocalData() != nil {
			log.Ctx(ctx).Error("failed to clean cached data on disk after build index failed",
				zap.Int64("buildID", it.BuildID),
				zap.Int64("index version", it.req.GetIndexVersion()))
		}
		log.Ctx(ctx).Error("failed to build index", zap.Error(err))
		return err
	}

	buildIndexLatency := it.tr.RecordSpan()
	metrics.IndexNodeKnowhereBuildIndexLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(float64(buildIndexLatency.Milliseconds()))

	log.Ctx(ctx).Info("Successfully build index", zap.Int64("buildID", it.BuildID), zap.Int64("Collection", it.collectionID), zap.Int64("SegmentID", it.segmentID))
	return nil
}

func (it *indexBuildTaskV2) SaveIndexFiles(ctx context.Context) error {
	gcIndex := func() {
		if err := it.index.Delete(); err != nil {
			log.Ctx(ctx).Error("IndexNode indexBuildTask Execute CIndexDelete failed", zap.Error(err))
		}
	}
	version, err := it.index.UpLoadV2()
	if err != nil {
		log.Ctx(ctx).Error("failed to upload index", zap.Error(err))
		gcIndex()
		return err
	}

	encodeIndexFileDur := it.tr.Record("index serialize and upload done")
	metrics.IndexNodeEncodeIndexFileLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(encodeIndexFileDur.Seconds())

	// early release index for gc, and we can ensure that Delete is idempotent.
	gcIndex()

	// use serialized size before encoding
	it.serializedSize = 0
	saveFileKeys := make([]string, 0)

	it.statistic.EndTime = time.Now().UnixMicro()
	it.node.storeIndexFilesAndStatisticV2(it.ClusterID, it.BuildID, saveFileKeys, it.serializedSize, &it.statistic, it.currentIndexVersion, version)
	log.Ctx(ctx).Debug("save index files done", zap.Strings("IndexFiles", saveFileKeys))
	saveIndexFileDur := it.tr.RecordSpan()
	metrics.IndexNodeSaveIndexFileLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(saveIndexFileDur.Seconds())
	it.tr.Elapse("index building all done")
	log.Ctx(ctx).Info("Successfully save index files", zap.Int64("buildID", it.BuildID), zap.Int64("Collection", it.collectionID),
		zap.Int64("partition", it.partitionID), zap.Int64("SegmentId", it.segmentID))
	return nil
}

// IndexBuildTask is used to record the information of the index tasks.
type indexBuildTask struct {
	ident  string
	cancel context.CancelFunc
	ctx    context.Context

	cm                  storage.ChunkManager
	index               indexcgowrapper.CodecIndex
	savePaths           []string
	req                 *indexpb.CreateJobRequest
	currentIndexVersion int32
	BuildID             UniqueID
	nodeID              UniqueID
	ClusterID           string
	collectionID        UniqueID
	partitionID         UniqueID
	segmentID           UniqueID
	fieldID             UniqueID
	fieldName           string
	fieldType           schemapb.DataType
	fieldData           storage.FieldData
	indexBlobs          []*storage.Blob
	newTypeParams       map[string]string
	newIndexParams      map[string]string
	serializedSize      uint64
	tr                  *timerecord.TimeRecorder
	queueDur            time.Duration
	statistic           indexpb.JobInfo
	node                *IndexNode
}

func (it *indexBuildTask) Reset() {
	it.ident = ""
	it.cancel = nil
	it.ctx = nil
	it.cm = nil
	it.index = nil
	it.savePaths = nil
	it.req = nil
	it.fieldData = nil
	it.indexBlobs = nil
	it.newTypeParams = nil
	it.newIndexParams = nil
	it.tr = nil
	it.node = nil
}

// Ctx is the context of index tasks.
func (it *indexBuildTask) Ctx() context.Context {
	return it.ctx
}

// Name is the name of task to build index.
func (it *indexBuildTask) Name() string {
	return it.ident
}

func (it *indexBuildTask) SetState(state commonpb.IndexState, failReason string) {
	it.node.storeTaskState(it.ClusterID, it.BuildID, state, failReason)
}

func (it *indexBuildTask) GetState() commonpb.IndexState {
	return it.node.loadTaskState(it.ClusterID, it.BuildID)
}

// OnEnqueue enqueues indexing tasks.
func (it *indexBuildTask) OnEnqueue(ctx context.Context) error {
	it.queueDur = 0
	it.tr.RecordSpan()
	it.statistic.StartTime = time.Now().UnixMicro()
	it.statistic.PodID = it.node.GetNodeID()
	log.Ctx(ctx).Info("IndexNode IndexBuilderTask Enqueue", zap.Int64("buildID", it.BuildID), zap.Int64("segmentID", it.segmentID))
	return nil
}

func (it *indexBuildTask) Prepare(ctx context.Context) error {
	it.queueDur = it.tr.RecordSpan()
	log.Ctx(ctx).Info("Begin to prepare indexBuildTask", zap.Int64("buildID", it.BuildID),
		zap.Int64("Collection", it.collectionID), zap.Int64("SegmentID", it.segmentID))
	typeParams := make(map[string]string)
	indexParams := make(map[string]string)

	// type params can be removed
	for _, kvPair := range it.req.GetTypeParams() {
		key, value := kvPair.GetKey(), kvPair.GetValue()
		typeParams[key] = value
		indexParams[key] = value
	}

	for _, kvPair := range it.req.GetIndexParams() {
		key, value := kvPair.GetKey(), kvPair.GetValue()
		indexParams[key] = value
	}
	it.newTypeParams = typeParams
	it.newIndexParams = indexParams
	it.statistic.IndexParams = it.req.GetIndexParams()
	// ugly codes to get dimension
	if dimStr, ok := typeParams[common.DimKey]; ok {
		var err error
		it.statistic.Dim, err = strconv.ParseInt(dimStr, 10, 64)
		if err != nil {
			log.Ctx(ctx).Error("parse dimesion failed", zap.Error(err))
			// ignore error
		}
	}
	log.Ctx(ctx).Info("Successfully prepare indexBuildTask", zap.Int64("buildID", it.BuildID),
		zap.Int64("Collection", it.collectionID), zap.Int64("SegmentID", it.segmentID))
	return nil
}

func (it *indexBuildTask) LoadData(ctx context.Context) error {
	getValueByPath := func(path string) ([]byte, error) {
		data, err := it.cm.Read(ctx, path)
		if err != nil {
			if errors.Is(err, merr.ErrIoKeyNotFound) {
				return nil, err
			}
			return nil, err
		}
		return data, nil
	}
	getBlobByPath := func(path string) (*Blob, error) {
		value, err := getValueByPath(path)
		if err != nil {
			return nil, err
		}
		return &Blob{
			Key:   path,
			Value: value,
		}, nil
	}

	toLoadDataPaths := it.req.GetDataPaths()
	keys := make([]string, len(toLoadDataPaths))
	blobs := make([]*Blob, len(toLoadDataPaths))

	loadKey := func(idx int) error {
		keys[idx] = toLoadDataPaths[idx]
		blob, err := getBlobByPath(toLoadDataPaths[idx])
		if err != nil {
			return err
		}
		blobs[idx] = blob
		return nil
	}
	// Use hardware.GetCPUNum() instead of hardware.GetCPUNum()
	// to respect CPU quota of container/pod
	// gomaxproc will be set by `automaxproc`, passing 0 will just retrieve the value
	err := funcutil.ProcessFuncParallel(len(toLoadDataPaths), hardware.GetCPUNum(), loadKey, "loadKey")
	if err != nil {
		log.Ctx(ctx).Warn("loadKey failed", zap.Error(err))
		return err
	}

	loadFieldDataLatency := it.tr.CtxRecord(ctx, "load field data done")
	metrics.IndexNodeLoadFieldLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(loadFieldDataLatency.Seconds())

	err = it.decodeBlobs(ctx, blobs)
	if err != nil {
		log.Ctx(ctx).Info("failed to decode blobs", zap.Int64("buildID", it.BuildID),
			zap.Int64("Collection", it.collectionID), zap.Int64("SegmentID", it.segmentID), zap.Error(err))
	} else {
		log.Ctx(ctx).Info("Successfully load data", zap.Int64("buildID", it.BuildID),
			zap.Int64("Collection", it.collectionID), zap.Int64("SegmentID", it.segmentID))
	}
	blobs = nil
	debug.FreeOSMemory()
	return err
}

func (it *indexBuildTask) BuildIndex(ctx context.Context) error {
	err := it.parseFieldMetaFromBinlog(ctx)
	if err != nil {
		log.Ctx(ctx).Warn("parse field meta from binlog failed", zap.Error(err))
		return err
	}

	indexType := it.newIndexParams[common.IndexTypeKey]
	if indexType == indexparamcheck.IndexDISKANN {
		// check index node support disk index
		if !Params.IndexNodeCfg.EnableDisk.GetAsBool() {
			log.Ctx(ctx).Warn("IndexNode don't support build disk index",
				zap.String("index type", it.newIndexParams[common.IndexTypeKey]),
				zap.Bool("enable disk", Params.IndexNodeCfg.EnableDisk.GetAsBool()))
			return errors.New("index node don't support build disk index")
		}

		// check load size and size of field data
		localUsedSize, err := indexcgowrapper.GetLocalUsedSize(paramtable.Get().LocalStorageCfg.Path.GetValue())
		if err != nil {
			log.Ctx(ctx).Warn("IndexNode get local used size failed")
			return err
		}
		fieldDataSize, err := estimateFieldDataSize(it.statistic.Dim, it.req.GetNumRows(), it.fieldType)
		if err != nil {
			log.Ctx(ctx).Warn("IndexNode get local used size failed")
			return err
		}
		usedLocalSizeWhenBuild := int64(float64(fieldDataSize)*diskUsageRatio) + localUsedSize
		maxUsedLocalSize := int64(Params.IndexNodeCfg.DiskCapacityLimit.GetAsFloat() * Params.IndexNodeCfg.MaxDiskUsagePercentage.GetAsFloat())

		if usedLocalSizeWhenBuild > maxUsedLocalSize {
			log.Ctx(ctx).Warn("IndexNode don't has enough disk size to build disk ann index",
				zap.Int64("usedLocalSizeWhenBuild", usedLocalSizeWhenBuild),
				zap.Int64("maxUsedLocalSize", maxUsedLocalSize))
			return errors.New("index node don't has enough disk size to build disk ann index")
		}

		err = indexparams.SetDiskIndexBuildParams(it.newIndexParams, int64(fieldDataSize))
		if err != nil {
			log.Ctx(ctx).Warn("failed to fill disk index params", zap.Error(err))
			return err
		}
	}

	var buildIndexInfo *indexcgowrapper.BuildIndexInfo
	buildIndexInfo, err = indexcgowrapper.NewBuildIndexInfo(it.req.GetStorageConfig())
	defer indexcgowrapper.DeleteBuildIndexInfo(buildIndexInfo)
	if err != nil {
		log.Ctx(ctx).Warn("create build index info failed", zap.Error(err))
		return err
	}
	err = buildIndexInfo.AppendFieldMetaInfo(it.collectionID, it.partitionID, it.segmentID, it.fieldID, it.fieldType)
	if err != nil {
		log.Ctx(ctx).Warn("append field meta failed", zap.Error(err))
		return err
	}

	err = buildIndexInfo.AppendIndexMetaInfo(it.req.IndexID, it.req.BuildID, it.req.IndexVersion)
	if err != nil {
		log.Ctx(ctx).Warn("append index meta failed", zap.Error(err))
		return err
	}

	err = buildIndexInfo.AppendBuildIndexParam(it.newIndexParams)
	if err != nil {
		log.Ctx(ctx).Warn("append index params failed", zap.Error(err))
		return err
	}

	jsonIndexParams, err := json.Marshal(it.newIndexParams)
	if err != nil {
		log.Ctx(ctx).Error("failed to json marshal index params", zap.Error(err))
		return err
	}

	log.Ctx(ctx).Info("index params are ready",
		zap.Int64("buildID", it.BuildID),
		zap.String("index params", string(jsonIndexParams)))

	err = buildIndexInfo.AppendBuildTypeParam(it.newTypeParams)
	if err != nil {
		log.Ctx(ctx).Warn("append type params failed", zap.Error(err))
		return err
	}

	for _, path := range it.req.GetDataPaths() {
		err = buildIndexInfo.AppendInsertFile(path)
		if err != nil {
			log.Ctx(ctx).Warn("append insert binlog path failed", zap.Error(err))
			return err
		}
	}

	it.currentIndexVersion = getCurrentIndexVersion(it.req.GetCurrentIndexVersion())
	if err := buildIndexInfo.AppendIndexEngineVersion(it.currentIndexVersion); err != nil {
		log.Ctx(ctx).Warn("append index engine version failed", zap.Error(err))
		return err
	}

	it.index, err = indexcgowrapper.CreateIndex(ctx, buildIndexInfo)
	if err != nil {
		if it.index != nil && it.index.CleanLocalData() != nil {
			log.Ctx(ctx).Error("failed to clean cached data on disk after build index failed",
				zap.Int64("buildID", it.BuildID),
				zap.Int64("index version", it.req.GetIndexVersion()))
		}
		log.Ctx(ctx).Error("failed to build index", zap.Error(err))
		return err
	}

	buildIndexLatency := it.tr.RecordSpan()
	metrics.IndexNodeKnowhereBuildIndexLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(buildIndexLatency.Seconds())

	log.Ctx(ctx).Info("Successfully build index", zap.Int64("buildID", it.BuildID), zap.Int64("Collection", it.collectionID), zap.Int64("SegmentID", it.segmentID), zap.Int32("currentIndexVersion", it.currentIndexVersion))
	return nil
}

func (it *indexBuildTask) SaveIndexFiles(ctx context.Context) error {
	gcIndex := func() {
		if err := it.index.Delete(); err != nil {
			log.Ctx(ctx).Error("IndexNode indexBuildTask Execute CIndexDelete failed", zap.Error(err))
		}
	}
	indexFilePath2Size, err := it.index.UpLoad()
	if err != nil {
		log.Ctx(ctx).Error("failed to upload index", zap.Error(err))
		gcIndex()
		return err
	}
	encodeIndexFileDur := it.tr.Record("index serialize and upload done")
	metrics.IndexNodeEncodeIndexFileLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(encodeIndexFileDur.Seconds())

	// early release index for gc, and we can ensure that Delete is idempotent.
	gcIndex()

	// use serialized size before encoding
	it.serializedSize = 0
	saveFileKeys := make([]string, 0)
	for filePath, fileSize := range indexFilePath2Size {
		it.serializedSize += uint64(fileSize)
		parts := strings.Split(filePath, "/")
		fileKey := parts[len(parts)-1]
		saveFileKeys = append(saveFileKeys, fileKey)
	}

	it.statistic.EndTime = time.Now().UnixMicro()
	it.node.storeIndexFilesAndStatistic(it.ClusterID, it.BuildID, saveFileKeys, it.serializedSize, &it.statistic, it.currentIndexVersion)
	log.Ctx(ctx).Debug("save index files done", zap.Strings("IndexFiles", saveFileKeys))
	saveIndexFileDur := it.tr.RecordSpan()
	metrics.IndexNodeSaveIndexFileLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(saveIndexFileDur.Seconds())
	it.tr.Elapse("index building all done")
	log.Ctx(ctx).Info("Successfully save index files", zap.Int64("buildID", it.BuildID), zap.Int64("Collection", it.collectionID),
		zap.Int64("partition", it.partitionID), zap.Int64("SegmentId", it.segmentID))
	return nil
}

func (it *indexBuildTask) parseFieldMetaFromBinlog(ctx context.Context) error {
	toLoadDataPaths := it.req.GetDataPaths()
	if len(toLoadDataPaths) == 0 {
		return merr.WrapErrParameterInvalidMsg("data insert path must be not empty")
	}
	data, err := it.cm.Read(ctx, toLoadDataPaths[0])
	if err != nil {
		if errors.Is(err, merr.ErrIoKeyNotFound) {
			return err
		}
		return err
	}

	var insertCodec storage.InsertCodec
	collectionID, partitionID, segmentID, insertData, err := insertCodec.DeserializeAll([]*Blob{{Key: toLoadDataPaths[0], Value: data}})
	if err != nil {
		return err
	}
	if len(insertData.Data) != 1 {
		return merr.WrapErrParameterInvalidMsg("we expect only one field in deserialized insert data")
	}

	it.collectionID = collectionID
	it.partitionID = partitionID
	it.segmentID = segmentID
	for fID, value := range insertData.Data {
		it.fieldType = indexcgowrapper.GenDataset(value).DType
		it.fieldID = fID
		break
	}

	return nil
}

func (it *indexBuildTask) decodeBlobs(ctx context.Context, blobs []*storage.Blob) error {
	var insertCodec storage.InsertCodec
	collectionID, partitionID, segmentID, insertData, err2 := insertCodec.DeserializeAll(blobs)
	if err2 != nil {
		return err2
	}
	metrics.IndexNodeDecodeFieldLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(it.tr.RecordSpan().Seconds())

	if len(insertData.Data) != 1 {
		return merr.WrapErrParameterInvalidMsg("we expect only one field in deserialized insert data")
	}
	it.collectionID = collectionID
	it.partitionID = partitionID
	it.segmentID = segmentID

	deserializeDur := it.tr.RecordSpan()

	log.Ctx(ctx).Info("IndexNode deserialize data success",
		zap.Int64("index id", it.req.IndexID),
		zap.String("index name", it.req.IndexName),
		zap.Int64("collectionID", it.collectionID),
		zap.Int64("partitionID", it.partitionID),
		zap.Int64("segmentID", it.segmentID),
		zap.Duration("deserialize duration", deserializeDur))

	// we can ensure that there blobs are in one Field
	var data storage.FieldData
	var fieldID storage.FieldID
	for fID, value := range insertData.Data {
		data = value
		fieldID = fID
		break
	}
	it.statistic.NumRows = int64(data.RowNum())
	it.fieldID = fieldID
	it.fieldData = data
	return nil
}
