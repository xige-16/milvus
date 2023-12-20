package proxy

//
//import (
//	"context"
//	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
//	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
//	"github.com/milvus-io/milvus/internal/proto/internalpb"
//	"github.com/milvus-io/milvus/internal/types"
//	"github.com/milvus-io/milvus/pkg/log"
//	"github.com/milvus-io/milvus/pkg/util/timerecord"
//	"github.com/milvus-io/milvus/pkg/util/typeutil"
//	"go.opentelemetry.io/otel"
//	"go.uber.org/zap"
//)
//
//type rankTask struct {
//	Condition
//	ctx context.Context
//
//	multipleRecallResults map[UniqueID]*milvuspb.SearchResults
//	request               *milvuspb.SearchRequestV2
//
//	tr      *timerecord.TimeRecorder
//	schema  *schemapb.CollectionSchema
//	requery bool
//
//	userOutputFields []string
//
//	offset    int64
//	resultBuf *typeutil.ConcurrentSet[*internalpb.SearchResults]
//
//	qc   types.QueryCoordClient
//	node types.ProxyComponent
//	lb   LBPolicy
//
//	collectionName string
//	collectionID   UniqueID
//}
//
//func (t *rankTask) PreExecute(ctx context.Context) error {
//	ctx, sp := otel.Tracer(typeutil.ProxyRole).Start(ctx, "Proxy-SearchV2-rank-PreExecute")
//	defer sp.End()
//
//	var err error
//	t.request.OutputFields, t.userOutputFields, err = translateOutputFields(t.request.OutputFields, t.schema, false)
//	if err != nil {
//		log.Warn("translate output fields failed", zap.Error(err))
//		return err
//	}
//
//	if len(t.request.OutputFields) > 0 {
//		t.requery = true
//	}
//	log.Debug("translate output fields",
//		zap.Strings("output fields", t.request.GetOutputFields()))
//
//	log.Debug("searchV2-rank PreExecute done.",
//		zap.Uint64("guarantee_ts", t.request.GetGuaranteeTimestamp()),
//		zap.Bool("use_default_consistency", t.request.GetUseDefaultConsistency()),
//		zap.Any("consistency level", t.request.GetConsistencyLevel()))
//
//	return nil
//}
//
//func (t *rankTask) Execute(ctx context.Context) error {
//
//}
//
//func (t *rankTask) PostExecute(ctx context.Context) error {
//
//}
