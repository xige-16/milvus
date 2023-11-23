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

package datanode

import (
	"context"
	"fmt"
	"path"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/datanode/allocator"
	"github.com/milvus-io/milvus/internal/datanode/broker"
	"github.com/milvus-io/milvus/internal/datanode/metacache"
	"github.com/milvus-io/milvus/internal/datanode/syncmgr"
	"github.com/milvus-io/milvus/internal/datanode/writebuffer"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/flowgraph"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/mq/msgdispatcher"
	"github.com/milvus-io/milvus/pkg/mq/msgstream"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

// dataSyncService controls a flowgraph for a specific collection
type dataSyncService struct {
	ctx          context.Context
	cancelFn     context.CancelFunc
	metacache    metacache.MetaCache
	opID         int64
	collectionID UniqueID // collection id of vchan for which this data sync service serves
	vchannelName string

	// TODO: should be equal to paramtable.GetNodeID(), but intergrationtest has 1 paramtable for a minicluster, the NodeID
	// varies, will cause savebinglogpath check fail. So we pass ServerID into dataSyncService to aviod it failure.
	serverID UniqueID

	fg *flowgraph.TimeTickedFlowGraph // internal flowgraph processes insert/delta messages

	broker  broker.Broker
	syncMgr syncmgr.SyncManager

	flushCh          chan flushMsg
	resendTTCh       chan resendTTMsg    // chan to ask for resending DataNode time tick message.
	timetickSender   *timeTickSender     // reference to timeTickSender
	compactor        *compactionExecutor // reference to compaction executor
	flushingSegCache *Cache              // a guarding cache stores currently flushing segment ids

	clearSignal  chan<- string       // signal channel to notify flowgraph close for collection/partition drop msg consumed
	idAllocator  allocator.Allocator // id/timestamp allocator
	msFactory    msgstream.Factory
	dispClient   msgdispatcher.Client
	chunkManager storage.ChunkManager

	stopOnce sync.Once
}

type nodeConfig struct {
	msFactory    msgstream.Factory // msgStream factory
	collectionID UniqueID
	vChannelName string
	metacache    metacache.MetaCache
	allocator    allocator.Allocator
	serverID     UniqueID
}

// start the flow graph in dataSyncService
func (dsService *dataSyncService) start() {
	if dsService.fg != nil {
		log.Info("dataSyncService starting flow graph", zap.Int64("collectionID", dsService.collectionID),
			zap.String("vChanName", dsService.vchannelName))
		dsService.fg.Start()
	} else {
		log.Warn("dataSyncService starting flow graph is nil", zap.Int64("collectionID", dsService.collectionID),
			zap.String("vChanName", dsService.vchannelName))
	}
}

func (dsService *dataSyncService) GracefullyClose() {
	if dsService.fg != nil {
		log.Info("dataSyncService gracefully closing flowgraph")
		dsService.fg.SetCloseMethod(flowgraph.CloseGracefully)
		dsService.close()
	}
}

func (dsService *dataSyncService) close() {
	dsService.stopOnce.Do(func() {
		log := log.Ctx(dsService.ctx).With(
			zap.Int64("collectionID", dsService.collectionID),
			zap.String("vChanName", dsService.vchannelName),
		)
		if dsService.fg != nil {
			log.Info("dataSyncService closing flowgraph")
			dsService.dispClient.Deregister(dsService.vchannelName)
			dsService.fg.Close()
			log.Info("dataSyncService flowgraph closed")
		}

		dsService.cancelFn()

		log.Info("dataSyncService closed")
	})
}

func getMetaCacheWithTickler(initCtx context.Context, node *DataNode, info *datapb.ChannelWatchInfo, tickler *tickler, unflushed, flushed []*datapb.SegmentInfo) (metacache.MetaCache, error) {
	tickler.setTotal(int32(len(unflushed) + len(flushed)))
	return initMetaCache(initCtx, node.chunkManager, info, tickler, unflushed, flushed)
}

func getMetaCacheWithEtcdTickler(initCtx context.Context, node *DataNode, info *datapb.ChannelWatchInfo, tickler *etcdTickler, unflushed, flushed []*datapb.SegmentInfo) (metacache.MetaCache, error) {
	tickler.watch()
	defer tickler.stop()

	return initMetaCache(initCtx, node.chunkManager, info, tickler, unflushed, flushed)
}

func initMetaCache(initCtx context.Context, chunkManager storage.ChunkManager, info *datapb.ChannelWatchInfo, tickler interface{ inc() }, unflushed, flushed []*datapb.SegmentInfo) (metacache.MetaCache, error) {
	recoverTs := info.GetVchan().GetSeekPosition().GetTimestamp()

	// tickler will update addSegment progress to watchInfo
	futures := make([]*conc.Future[any], 0, len(unflushed)+len(flushed))
	segmentPks := typeutil.NewConcurrentMap[int64, []*storage.PkStatistics]()

	loadSegmentStats := func(segType string, segments []*datapb.SegmentInfo) {
		for _, item := range segments {
			log.Info("recover segments from checkpoints",
				zap.String("vChannelName", item.GetInsertChannel()),
				zap.Int64("segmentID", item.GetID()),
				zap.Int64("numRows", item.GetNumOfRows()),
				zap.String("segmentType", segType),
			)
			segment := item

			future := getOrCreateIOPool().Submit(func() (any, error) {
				stats, err := loadStats(initCtx, chunkManager, info.GetSchema(), segment.GetID(), segment.GetCollectionID(), segment.GetStatslogs(), recoverTs)
				if err != nil {
					return nil, err
				}
				segmentPks.Insert(segment.GetID(), stats)
				tickler.inc()

				return struct{}{}, nil
			})

			futures = append(futures, future)
		}
	}

	loadSegmentStats("growing", unflushed)
	loadSegmentStats("sealed", flushed)

	// use fetched segment info
	info.Vchan.FlushedSegments = flushed
	info.Vchan.UnflushedSegments = unflushed

	if err := conc.AwaitAll(futures...); err != nil {
		return nil, err
	}

	// return channel, nil
	metacache := metacache.NewMetaCache(info, func(segment *datapb.SegmentInfo) *metacache.BloomFilterSet {
		entries, _ := segmentPks.Get(segment.GetID())
		return metacache.NewBloomFilterSet(entries...)
	})

	return metacache, nil
}

func loadStats(ctx context.Context, chunkManager storage.ChunkManager, schema *schemapb.CollectionSchema, segmentID int64, collectionID int64, statsBinlogs []*datapb.FieldBinlog, ts Timestamp) ([]*storage.PkStatistics, error) {
	startTs := time.Now()
	log := log.With(zap.Int64("segmentID", segmentID))
	log.Info("begin to init pk bloom filter", zap.Int("statsBinLogsLen", len(statsBinlogs)))

	// get pkfield id
	pkField := int64(-1)
	for _, field := range schema.Fields {
		if field.IsPrimaryKey {
			pkField = field.FieldID
			break
		}
	}

	// filter stats binlog files which is pk field stats log
	bloomFilterFiles := []string{}
	logType := storage.DefaultStatsType

	for _, binlog := range statsBinlogs {
		if binlog.FieldID != pkField {
			continue
		}
	Loop:
		for _, log := range binlog.GetBinlogs() {
			_, logidx := path.Split(log.GetLogPath())
			// if special status log exist
			// only load one file
			switch logidx {
			case storage.CompoundStatsType.LogIdx():
				bloomFilterFiles = []string{log.GetLogPath()}
				logType = storage.CompoundStatsType
				break Loop
			default:
				bloomFilterFiles = append(bloomFilterFiles, log.GetLogPath())
			}
		}
	}

	// no stats log to parse, initialize a new BF
	if len(bloomFilterFiles) == 0 {
		log.Warn("no stats files to load")
		return nil, nil
	}

	// read historical PK filter
	values, err := chunkManager.MultiRead(ctx, bloomFilterFiles)
	if err != nil {
		log.Warn("failed to load bloom filter files", zap.Error(err))
		return nil, err
	}
	blobs := make([]*Blob, 0)
	for i := 0; i < len(values); i++ {
		blobs = append(blobs, &Blob{Value: values[i]})
	}

	var stats []*storage.PrimaryKeyStats
	if logType == storage.CompoundStatsType {
		stats, err = storage.DeserializeStatsList(blobs[0])
		if err != nil {
			log.Warn("failed to deserialize stats list", zap.Error(err))
			return nil, err
		}
	} else {
		stats, err = storage.DeserializeStats(blobs)
		if err != nil {
			log.Warn("failed to deserialize stats", zap.Error(err))
			return nil, err
		}
	}

	var size uint
	result := make([]*storage.PkStatistics, 0, len(stats))
	for _, stat := range stats {
		pkStat := &storage.PkStatistics{
			PkFilter: stat.BF,
			MinPK:    stat.MinPk,
			MaxPK:    stat.MaxPk,
		}
		size += stat.BF.Cap()
		result = append(result, pkStat)
	}

	log.Info("Successfully load pk stats", zap.Any("time", time.Since(startTs)), zap.Uint("size", size))
	return result, nil
}

func getServiceWithChannel(initCtx context.Context, node *DataNode, info *datapb.ChannelWatchInfo, metacache metacache.MetaCache, unflushed, flushed []*datapb.SegmentInfo) (*dataSyncService, error) {
	var (
		channelName  = info.GetVchan().GetChannelName()
		collectionID = info.GetVchan().GetCollectionID()
	)

	config := &nodeConfig{
		msFactory: node.factory,
		allocator: node.allocator,

		collectionID: collectionID,
		vChannelName: channelName,
		metacache:    metacache,
		serverID:     node.session.ServerID,
	}

	var (
		flushCh    = make(chan flushMsg, 100)
		resendTTCh = make(chan resendTTMsg, 100)
	)

	node.writeBufferManager.Register(channelName, metacache, writebuffer.WithMetaWriter(syncmgr.BrokerMetaWriter(node.broker)), writebuffer.WithIDAllocator(node.allocator))
	ctx, cancel := context.WithCancel(node.ctx)
	ds := &dataSyncService{
		ctx:        ctx,
		cancelFn:   cancel,
		flushCh:    flushCh,
		resendTTCh: resendTTCh,
		opID:       info.GetOpID(),

		dispClient: node.dispClient,
		msFactory:  node.factory,
		broker:     node.broker,

		idAllocator:  config.allocator,
		metacache:    config.metacache,
		collectionID: config.collectionID,
		vchannelName: config.vChannelName,
		serverID:     config.serverID,

		flushingSegCache: node.segmentCache,
		clearSignal:      node.clearSignal,
		chunkManager:     node.chunkManager,
		compactor:        node.compactionExecutor,
		timetickSender:   node.timeTickSender,
		syncMgr:          node.syncMgr,

		fg: nil,
	}

	// init flowgraph
	fg := flowgraph.NewTimeTickedFlowGraph(node.ctx)
	dmStreamNode, err := newDmInputNode(initCtx, node.dispClient, info.GetVchan().GetSeekPosition(), config)
	if err != nil {
		return nil, err
	}

	ddNode, err := newDDNode(
		node.ctx,
		collectionID,
		channelName,
		info.GetVchan().GetDroppedSegmentIds(),
		flushed,
		unflushed,
		node.compactionExecutor,
	)
	if err != nil {
		return nil, err
	}

	var updater statsUpdater
	if Params.DataNodeCfg.DataNodeTimeTickByRPC.GetAsBool() {
		updater = ds.timetickSender
	} else {
		m, err := config.msFactory.NewMsgStream(ctx)
		if err != nil {
			return nil, err
		}

		m.AsProducer([]string{Params.CommonCfg.DataCoordTimeTick.GetValue()})
		metrics.DataNodeNumProducers.WithLabelValues(fmt.Sprint(paramtable.GetNodeID())).Inc()
		log.Info("datanode AsProducer", zap.String("TimeTickChannelName", Params.CommonCfg.DataCoordTimeTick.GetValue()))

		m.EnableProduce(true)

		updater = newMqStatsUpdater(config, m)
	}

	writeNode := newWriteNode(node.ctx, node.writeBufferManager, updater, config)

	ttNode, err := newTTNode(config, node.broker, node.writeBufferManager)
	if err != nil {
		return nil, err
	}

	if err := fg.AssembleNodes(dmStreamNode, ddNode, writeNode, ttNode); err != nil {
		return nil, err
	}
	ds.fg = fg

	return ds, nil
}

// newServiceWithEtcdTickler gets a dataSyncService, but flowgraphs are not running
// initCtx is used to init the dataSyncService only, if initCtx.Canceled or initCtx.Timeout
// newServiceWithEtcdTickler stops and returns the initCtx.Err()
func newServiceWithEtcdTickler(initCtx context.Context, node *DataNode, info *datapb.ChannelWatchInfo, tickler *etcdTickler) (*dataSyncService, error) {
	// recover segment checkpoints
	unflushedSegmentInfos, err := node.broker.GetSegmentInfo(initCtx, info.GetVchan().GetUnflushedSegmentIds())
	if err != nil {
		return nil, err
	}
	flushedSegmentInfos, err := node.broker.GetSegmentInfo(initCtx, info.GetVchan().GetFlushedSegmentIds())
	if err != nil {
		return nil, err
	}

	// init channel meta
	metaCache, err := getMetaCacheWithEtcdTickler(initCtx, node, info, tickler, unflushedSegmentInfos, flushedSegmentInfos)
	if err != nil {
		return nil, err
	}

	return getServiceWithChannel(initCtx, node, info, metaCache, unflushedSegmentInfos, flushedSegmentInfos)
}

// newDataSyncService gets a dataSyncService, but flowgraphs are not running
// initCtx is used to init the dataSyncService only, if initCtx.Canceled or initCtx.Timeout
// newDataSyncService stops and returns the initCtx.Err()
// NOTE: compactiable for event manager
func newDataSyncService(initCtx context.Context, node *DataNode, info *datapb.ChannelWatchInfo, tickler *tickler) (*dataSyncService, error) {
	// recover segment checkpoints
	unflushedSegmentInfos, err := node.broker.GetSegmentInfo(initCtx, info.GetVchan().GetUnflushedSegmentIds())
	if err != nil {
		return nil, err
	}
	flushedSegmentInfos, err := node.broker.GetSegmentInfo(initCtx, info.GetVchan().GetFlushedSegmentIds())
	if err != nil {
		return nil, err
	}

	// init metaCache meta
	metaCache, err := getMetaCacheWithTickler(initCtx, node, info, tickler, unflushedSegmentInfos, flushedSegmentInfos)
	if err != nil {
		return nil, err
	}

	return getServiceWithChannel(initCtx, node, info, metaCache, unflushedSegmentInfos, flushedSegmentInfos)
}
