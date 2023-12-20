package rank

//
//import (
//	"github.com/cockroachdb/errors"
//
//	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
//	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
//	"github.com/milvus-io/milvus/internal/proto/planpb"
//	"github.com/milvus-io/milvus/pkg/util/funcutil"
//)
//
//type rankType int
//
//const (
//	invalidRankType  rankType = iota // invalidRankType   = 0
//	rrfRankType                      // rrfRankType = 1
//	weightedRankType                 // weightedRankType = 2
//	udfExprRankType                  // udfExprRankType = 3
//)
//
//var (
//	rankTypeMap = map[string]rankType{
//		"invalid":  invalidRankType,
//		"rrf":      rrfRankType,
//		"weighted": weightedRankType,
//		"expr":     udfExprRankType,
//	}
//)
//
//func createRankPlan(rankParamsPairStr []*commonpb.KeyValuePair) (*planpb.PlanNode, error) {
//	rankTypeStr, err := funcutil.GetAttrByKeyFromRepeatedKV(RankTypeKey, rankParamsPairStr)
//	if err != nil {
//		return nil, errors.New(RankTypeKey + " not found in rank_params")
//	}
//
//	if _, ok := rankTypeMap[rankTypeStr]; !ok {
//		return nil, errors.Errorf("unsupported rank type %s", rankTypeStr)
//	}
//
//	plan := &planpb.PlanNode{}
//
//	switch rankTypeMap[rankTypeStr] {
//	case rrfRankType:
//
//	case weightedRankType:
//
//	default:
//		return nil, errors.Errorf("unsupported rank type %s", rankTypeStr)
//	}
//
//	return plan, nil
//}
//
//type Ranker interface {
//	PreCheck(inputs map[UniqueID]*milvuspb.SearchResults) error
//	Norm(inputs map[UniqueID]*milvuspb.SearchResults) error
//	Rank(inputs map[UniqueID]*milvuspb.SearchResults) (*milvuspb.SearchResults, error)
//}
//
//type RRFRanker struct {
//	k        int
//	fieldIDs []UniqueID
//}
//
//func (rr *RRFRanker) PreCheck(inputs map[UniqueID]*milvuspb.SearchResults) error {
//	for _, fieldID := range rr.fieldIDs {
//		if _, ok := inputs[fieldID]; !ok {
//			return errors.Errorf("miss field %d search result", fieldID)
//		}
//	}
//
//	return nil
//}
//
//func (rr *RRFRanker) Norm(inputs map[UniqueID]*milvuspb.SearchResults) error {
//	return nil
//}
//
//func (rr *RRFRanker) Rank(inputs map[UniqueID]*milvuspb.SearchResults) (result *milvuspb.SearchResults, err error) {
//
//}
//
//type WeightedRanker struct {
//	weights  []float64
//	fieldIDs []UniqueID
//}
//
//func (wr *WeightedRanker) PreCheck(inputs map[UniqueID]*milvuspb.SearchResults) error {
//	if len(wr.weights) != len(wr.fieldIDs) {
//		return errors.New("inconsistent length on weights and fieldIDs")
//	}
//	for _, fieldID := range wr.fieldIDs {
//		if _, ok := inputs[fieldID]; !ok {
//			return errors.Errorf("miss field %d search result", fieldID)
//		}
//	}
//
//	return nil
//}
//
//func (wr *WeightedRanker) Norm(inputs map[UniqueID]*milvuspb.SearchResults) error {
//	return nil
//}
//
//func (wr *WeightedRanker) Rank(inputs map[UniqueID]*milvuspb.SearchResults) (result *milvuspb.SearchResults, err error) {
//
//}
//
//type rankNode interface {
//	exec() error
//}
//
//type normPolicy func(inputs []*milvuspb.SearchResults) error
//type rankPolicy func(inputs []*milvuspb.SearchResults) (*milvuspb.SearchResults, error)
//
//type execNode struct {
//}
