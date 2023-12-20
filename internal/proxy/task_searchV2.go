package proxy

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"strconv"

	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
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

type searchTaskV2 struct {
	Condition
	//*internalpb.SearchRequest
	ctx context.Context

	result  *milvuspb.SearchResults
	request *milvuspb.SearchRequestV2

	tr      *timerecord.TimeRecorder
	schema  *schemapb.CollectionSchema
	requery bool

	userOutputFields []string

	qc   types.QueryCoordClient
	node types.ProxyComponent
	lb   LBPolicy

	collectionName string
	collectionID   UniqueID

	multipleRecallTasks map[string]*milvuspb.SearchResults
	limit               int64
}

func (t *searchTaskV2) PreExecute(ctx context.Context) error {
	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "Proxy-SearchV2-PreExecute")
	defer sp.End()

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

	limitStr, err := funcutil.GetAttrByKeyFromRepeatedKV(LimitKey, t.request.GetRankParams())
	if err != nil {
		return errors.New(LimitKey + " not found in search_params")
	}
	t.limit, err = strconv.ParseInt(limitStr, 0, 64)
	if err != nil {
		return fmt.Errorf("%s [%s] is invalid", LimitKey, limitStr)
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

	log.Debug("searchV2 PreExecute done.",
		zap.Uint64("guarantee_ts", t.request.GetGuaranteeTimestamp()),
		zap.Bool("use_default_consistency", t.request.GetUseDefaultConsistency()),
		zap.Any("consistency level", t.request.GetConsistencyLevel()))

	return nil
}

func (t *searchTaskV2) Execute(ctx context.Context) error {
	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "Proxy-SearchV2-Execute")
	defer sp.End()

	tr := timerecord.NewTimeRecorder(fmt.Sprintf("proxy execute searchV2 %d", t.ID()))
	defer tr.CtxElapse(ctx, "done")

	for _, req := range t.request.Requests {
		nq, err := getNq(req)
		if err != nil {
			log.Warn("failed to get nq", zap.Error(err))
			return err
		}
		if nq > 1 {
			log.Warn("nq should be less than or equal to 1")
			return errors.New("nq should be less than or equal to 1")
		}
		annsField, err := funcutil.GetAttrByKeyFromRepeatedKV(AnnsFieldKey, req.GetSearchParams())
		if err != nil || len(annsField) == 0 {
			if enableMultipleVectorFields {
				return errors.New(AnnsFieldKey + " not found in search_params")
			}
			vecFieldSchema, err2 := typeutil.GetVectorFieldSchema(t.schema)
			if err2 != nil {
				return errors.New(AnnsFieldKey + " not found in schema")
			}
			annsField = vecFieldSchema.Name
		}

		req.TravelTimestamp = t.request.GetTravelTimestamp()
		req.GuaranteeTimestamp = t.request.GetGuaranteeTimestamp()
		req.NotReturnAllMeta = t.request.GetNotReturnAllMeta()
		req.ConsistencyLevel = t.request.GetConsistencyLevel()
		req.UseDefaultConsistency = t.request.GetUseDefaultConsistency()

		result, err := t.node.Search(ctx, req)
		if err != nil {
			log.Debug("fail to search on field", zap.String("field", annsField))
			return err
		}
		if result.GetStatus().GetErrorCode() == commonpb.ErrorCode_NotShardLeader {
			log.Warn("QueryNode is not shardLeader")
			return errInvalidShardLeaders
		}
		if result.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
			log.Warn("QueryNode search result error",
				zap.String("reason", result.GetStatus().GetReason()))
			return errors.Wrapf(merr.Error(result.GetStatus()), "fail to search on field %s", annsField)
		}

		t.multipleRecallTasks[annsField] = result
	}

	log.Debug("SearchV2 Execute done.",
		zap.Int64("collection", t.collectionID))
	return nil
}

func (t *searchTaskV2) PostExecute(ctx context.Context) error {
	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "Proxy-Search-PostExecute")
	defer sp.End()

	tr := timerecord.NewTimeRecorder("searchTaskV2 PostExecute")
	defer func() {
		tr.CtxElapse(ctx, "done")
	}()

	for _, value := range t.multipleRecallTasks {
		t.result = value
		break
	}

	t.result.Results.TopK = t.limit

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

	log.Debug("Search post execute done",
		zap.Int64("collection", t.collectionID))
	return nil
}

func (t *searchTaskV2) Requery() error {
	pkField, err := typeutil.GetPrimaryFieldSchema(t.schema)
	if err != nil {
		return err
	}
	ids := t.result.GetResults().GetIds()
	plan := planparserv2.CreateRequeryPlan(pkField, ids)

	queryReq := &milvuspb.QueryRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_Retrieve,
		},
		DbName:             t.request.GetDbName(),
		CollectionName:     t.request.GetCollectionName(),
		Expr:               "",
		OutputFields:       t.request.GetOutputFields(),
		PartitionNames:     t.request.GetPartitionNames(),
		GuaranteeTimestamp: t.request.GetGuaranteeTimestamp(),
		//QueryParams:        t.request.GetSearchParams(),
	}
	qt := &queryTask{
		ctx:       t.ctx,
		Condition: NewTaskCondition(t.ctx),
		RetrieveRequest: &internalpb.RetrieveRequest{
			Base: commonpbutil.NewMsgBase(
				commonpbutil.WithMsgType(commonpb.MsgType_Retrieve),
				commonpbutil.WithSourceID(paramtable.GetNodeID()),
			),
			ReqID: paramtable.GetNodeID(),
		},
		request: queryReq,
		plan:    plan,
		qc:      t.node.(*Proxy).queryCoord,
		lb:      t.node.(*Proxy).lbPolicy,
	}
	queryResult, err := t.node.(*Proxy).query(t.ctx, qt)
	if err != nil {
		return err
	}
	if queryResult.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
		return merr.Error(queryResult.GetStatus())
	}
	// Reorganize Results. The order of query result ids will be altered and differ from queried ids.
	// We should reorganize query results to keep the order of original queried ids. For example:
	// ===========================================
	//  3  2  5  4  1  (query ids)
	//       ||
	//       || (query)
	//       \/
	//  4  3  5  1  2  (result ids)
	// v4 v3 v5 v1 v2  (result vectors)
	//       ||
	//       || (reorganize)
	//       \/
	//  3  2  5  4  1  (result ids)
	// v3 v2 v5 v4 v1  (result vectors)
	// ===========================================
	pkFieldData, err := typeutil.GetPrimaryFieldData(queryResult.GetFieldsData(), pkField)
	if err != nil {
		return err
	}
	offsets := make(map[any]int)
	for i := 0; i < typeutil.GetPKSize(pkFieldData); i++ {
		pk := typeutil.GetData(pkFieldData, i)
		offsets[pk] = i
	}

	t.result.Results.FieldsData = make([]*schemapb.FieldData, len(queryResult.GetFieldsData()))
	for i := 0; i < typeutil.GetSizeOfIDs(ids); i++ {
		id := typeutil.GetPK(ids, int64(i))
		if _, ok := offsets[id]; !ok {
			return fmt.Errorf("incomplete query result, missing id %s, len(searchIDs) = %d, len(queryIDs) = %d, collection=%d",
				id, typeutil.GetSizeOfIDs(ids), len(offsets), t.collectionID)
		}
		typeutil.AppendFieldData(t.result.Results.FieldsData, queryResult.GetFieldsData(), int64(offsets[id]))
	}

	// filter id field out if it is not specified as output
	t.result.Results.FieldsData = lo.Filter(t.result.Results.FieldsData, func(fieldData *schemapb.FieldData, i int) bool {
		return lo.Contains(t.request.GetOutputFields(), fieldData.GetFieldName())
	})

	return nil
}

func (t *searchTaskV2) fillInFieldInfo() {
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

func (t *searchTaskV2) TraceCtx() context.Context {
	return t.ctx
}

func (t *searchTaskV2) ID() UniqueID {
	return t.request.Base.MsgID
}

func (t *searchTaskV2) SetID(uid UniqueID) {
	t.request.Base.MsgID = uid
}

func (t *searchTaskV2) Name() string {
	return SearchTaskV2Name
}

func (t *searchTaskV2) Type() commonpb.MsgType {
	return t.request.Base.MsgType
}

func (t *searchTaskV2) BeginTs() Timestamp {
	return t.request.Base.Timestamp
}

func (t *searchTaskV2) EndTs() Timestamp {
	return t.request.Base.Timestamp
}

func (t *searchTaskV2) SetTs(ts Timestamp) {
	t.request.Base.Timestamp = ts
}

func (t *searchTaskV2) OnEnqueue() error {
	t.request.Base = commonpbutil.NewMsgBase()
	t.request.Base.MsgType = commonpb.MsgType_Search
	t.request.Base.SourceID = paramtable.GetNodeID()
	return nil
}
