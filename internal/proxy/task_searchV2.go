package proxy

import (
	"context"
	"fmt"
	"github.com/cockroachdb/errors"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"
	"sort"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

const (
	SearchTaskV2Name = "SearchTaskV2"
)

type searchV2Task struct {
	Condition
	//*internalpb.SearchRequest
	ctx context.Context

	result  *milvuspb.SearchResults
	request *milvuspb.HybridSearchRequest

	tr      *timerecord.TimeRecorder
	schema  *schemapb.CollectionSchema
	requery bool

	userOutputFields []string

	qc   types.QueryCoordClient
	node types.ProxyComponent
	lb   LBPolicy

	collectionName string
	collectionID   UniqueID

	multipleRecallResults *typeutil.ConcurrentSet[*milvuspb.SearchResults]
	reScorers             []reScorer
}

func (t *searchV2Task) PreExecute(ctx context.Context) error {
	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "Proxy-SearchV2-PreExecute")
	defer sp.End()

	if len(t.request.Requests) > 1024 {
		return errors.New("maximum of ann search requests is 1024")
	}
	collectionName := t.request.CollectionName
	t.collectionName = collectionName
	collID, err := globalMetaCache.GetCollectionID(ctx, t.request.GetDbName(), collectionName)
	if err != nil {
		return err
	}
	t.collectionID = collID

	log := log.Ctx(ctx).With(zap.Int64("collID", collID), zap.String("collName", collectionName))
	t.schema, err = globalMetaCache.GetCollectionSchema(ctx, t.request.GetDbName(), collectionName)
	if err != nil {
		log.Warn("get collection schema failed", zap.Error(err))
		return err
	}

	partitionKeyMode, err := isPartitionKeyMode(ctx, t.request.GetDbName(), collectionName)
	if err != nil {
		log.Warn("is partition key mode failed", zap.Error(err))
		return err
	}
	if partitionKeyMode && len(t.request.GetPartitionNames()) != 0 {
		return errors.New("not support manually specifying the partition names if partition key mode is used")
	}

	t.request.OutputFields, t.userOutputFields, err = translateOutputFields(t.request.OutputFields, t.schema, false)
	if err != nil {
		log.Warn("translate output fields failed", zap.Error(err))
		return err
	}
	log.Debug("translate output fields",
		zap.Strings("output fields", t.request.GetOutputFields()))

	if len(t.request.OutputFields) > 0 {
		t.requery = true
	}

	t.reScorers, err = NewReScorer(t.request.GetRequests(), t.request.GetRankParams())
	if err != nil {
		log.Warn("generate reScorer failed", zap.Error(err))
		return err
	}

	log.Debug("searchV2 PreExecute done.",
		zap.Uint64("guarantee_ts", t.request.GetGuaranteeTimestamp()),
		zap.Bool("use_default_consistency", t.request.GetUseDefaultConsistency()),
		zap.Any("consistency level", t.request.GetConsistencyLevel()))

	return nil
}

func (t *searchV2Task) Execute(ctx context.Context) error {
	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "Proxy-SearchV2-Execute")
	defer sp.End()

	tr := timerecord.NewTimeRecorder(fmt.Sprintf("proxy execute searchV2 %d", t.ID()))
	defer tr.CtxElapse(ctx, "done")

	for index, req := range t.request.Requests {
		nq, err := getNq(req)
		if err != nil {
			log.Warn("failed to get nq", zap.Error(err))
			return err
		}
		if nq > 1 {
			err = merr.WrapErrParameterInvalid("1", fmt.Sprint(nq), "nq should not be greater than 1")
			log.Warn(err.Error())
			return err
		}

		req.TravelTimestamp = t.request.GetTravelTimestamp()
		req.GuaranteeTimestamp = t.request.GetGuaranteeTimestamp()
		req.NotReturnAllMeta = t.request.GetNotReturnAllMeta()
		req.ConsistencyLevel = t.request.GetConsistencyLevel()
		req.UseDefaultConsistency = t.request.GetUseDefaultConsistency()
		req.OutputFields = nil

		result, err := t.node.Search(ctx, req)
		if err != nil {
			return err
		}
		if result.GetStatus().GetErrorCode() == commonpb.ErrorCode_NotShardLeader {
			log.Warn("QueryNode is not shardLeader")
			return errInvalidShardLeaders
		}
		if result.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
			log.Warn("QueryNode search result error",
				zap.String("reason", result.GetStatus().GetReason()))
			return merr.Error(result.GetStatus())
		}

		t.reScorers[index].reScore(result)
		t.multipleRecallResults.Insert(result)
	}

	log.Debug("SearchV2 Execute done.",
		zap.Int64("collection", t.collectionID))
	return nil
}

func (t *searchV2Task) PostExecute(ctx context.Context) error {
	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "Proxy-SearchV2-PostExecute")
	defer sp.End()

	tr := timerecord.NewTimeRecorder("searchV2Task PostExecute")
	defer func() {
		tr.CtxElapse(ctx, "done")
	}()

	primaryFieldSchema, err := typeutil.GetPrimaryFieldSchema(t.schema)
	if err != nil {
		log.Warn("failed to get primary field schema", zap.Error(err))
		return err
	}

	queryParams, err := parseQueryParams(t.request.GetRankParams())
	if err != nil {
		return err
	}

	t.result, err = rankSearchResultData(ctx, t.multipleRecallResults.Collect(), 1, queryParams.limit+queryParams.offset, primaryFieldSchema.GetDataType(), queryParams.offset)
	if err != nil {
		return err
	}

	t.result.CollectionName = t.collectionName
	t.fillInFieldInfo()

	if t.requery {
		err := t.Requery()
		if err != nil {
			log.Warn("failed to requery", zap.Error(err))
			return err
		}
	}
	t.result.Results.OutputFields = t.userOutputFields

	log.Debug("SearchV2 post execute done",
		zap.Int64("collection", t.collectionID))
	return nil
}

func (t *searchV2Task) Requery() error {
	queryReq := &milvuspb.QueryRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_Retrieve,
		},
		DbName:                t.request.GetDbName(),
		CollectionName:        t.request.GetCollectionName(),
		Expr:                  "",
		OutputFields:          t.request.GetOutputFields(),
		PartitionNames:        t.request.GetPartitionNames(),
		GuaranteeTimestamp:    t.request.GetGuaranteeTimestamp(),
		TravelTimestamp:       t.request.GetTravelTimestamp(),
		NotReturnAllMeta:      t.request.GetNotReturnAllMeta(),
		ConsistencyLevel:      t.request.GetConsistencyLevel(),
		UseDefaultConsistency: t.request.GetUseDefaultConsistency(),
	}

	return doRequery(t.ctx, t.collectionID, t.node, t.schema, queryReq, t.result)
}

func rankSearchResultData(ctx context.Context, searchResults []*milvuspb.SearchResults, nq int64, topk int64, pkType schemapb.DataType, offset int64) (*milvuspb.SearchResults, error) {
	tr := timerecord.NewTimeRecorder("rankSearchResultData")
	defer func() {
		tr.CtxElapse(ctx, "done")
	}()

	limit := topk - offset
	log.Ctx(ctx).Debug("rankSearchResultData",
		zap.Int("len(searchResults)", len(searchResults)),
		zap.Int64("nq", nq),
		zap.Int64("offset", offset),
		zap.Int64("limit", limit))

	ret := &milvuspb.SearchResults{
		Status: merr.Success(),
		Results: &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       topk,
			FieldsData: make([]*schemapb.FieldData, 0),
			Scores:     []float32{},
			Ids:        &schemapb.IDs{},
			Topks:      []int64{},
		},
	}

	switch pkType {
	case schemapb.DataType_Int64:
		ret.GetResults().Ids.IdField = &schemapb.IDs_IntId{
			IntId: &schemapb.LongArray{
				Data: make([]int64, 0),
			},
		}
	case schemapb.DataType_VarChar:
		ret.GetResults().Ids.IdField = &schemapb.IDs_StrId{
			StrId: &schemapb.StringArray{
				Data: make([]string, 0),
			},
		}
	default:
		return nil, errors.New("unsupported pk type")
	}

	accumulatedScores := make([]map[interface{}]float32, nq)
	for i := int64(0); i < nq; i++ {
		accumulatedScores[i] = make(map[interface{}]float32)
	}

	for _, result := range searchResults {
		scores := result.GetResults().GetScores()
		start := int64(0)
		for i := int64(0); i < nq; i++ {
			realTopk := result.GetResults().Topks[i]
			for j := start; j < start+realTopk; j++ {
				id := typeutil.GetPK(result.GetResults().GetIds(), j)
				accumulatedScores[i][id] += scores[j]
			}
			start += realTopk
		}
	}

	for i := int64(0); i < nq; i++ {
		idSet := accumulatedScores[i]
		keys := make([]interface{}, 0)
		for key := range idSet {
			keys = append(keys, key)
		}

		if int64(len(keys)) <= offset {
			ret.Results.Topks = append(ret.Results.Topks, 0)
			continue
		}

		sort.Slice(keys, func(i, j int) bool {
			return idSet[keys[i]] >= idSet[keys[j]]
		})

		if int64(len(keys)) > topk {
			keys = keys[:topk]
		}
		ret.Results.Topks = append(ret.Results.Topks, int64(len(keys)))
		for _, key := range keys {
			typeutil.AppendPKs(ret.Results.Ids, key)
			ret.Results.Scores = append(ret.Results.Scores, idSet[key])
		}
	}

	ret.Results.TopK = limit
	return ret, nil
}

func (t *searchV2Task) fillInFieldInfo() {
	if len(t.request.OutputFields) != 0 && len(t.result.Results.FieldsData) != 0 {
		for i, name := range t.request.OutputFields {
			for _, field := range t.schema.Fields {
				if t.result.Results.FieldsData[i] != nil && field.Name == name {
					t.result.Results.FieldsData[i].FieldName = field.Name
					t.result.Results.FieldsData[i].FieldId = field.FieldID
					t.result.Results.FieldsData[i].Type = field.DataType
					t.result.Results.FieldsData[i].IsDynamic = field.IsDynamic
				}
			}
		}
	}
}

func (t *searchV2Task) TraceCtx() context.Context {
	return t.ctx
}

func (t *searchV2Task) ID() UniqueID {
	return t.request.Base.MsgID
}

func (t *searchV2Task) SetID(uid UniqueID) {
	t.request.Base.MsgID = uid
}

func (t *searchV2Task) Name() string {
	return SearchTaskV2Name
}

func (t *searchV2Task) Type() commonpb.MsgType {
	return t.request.Base.MsgType
}

func (t *searchV2Task) BeginTs() Timestamp {
	return t.request.Base.Timestamp
}

func (t *searchV2Task) EndTs() Timestamp {
	return t.request.Base.Timestamp
}

func (t *searchV2Task) SetTs(ts Timestamp) {
	t.request.Base.Timestamp = ts
}

func (t *searchV2Task) OnEnqueue() error {
	t.request.Base = commonpbutil.NewMsgBase()
	t.request.Base.MsgType = commonpb.MsgType_Search
	t.request.Base.SourceID = paramtable.GetNodeID()
	return nil
}
