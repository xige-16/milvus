package proxy

import (
	"encoding/json"
	"fmt"
	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"strconv"
)

type rankType int

const (
	invalidRankType  rankType = iota // invalidRankType   = 0
	rrfRankType                      // rrfRankType = 1
	weightedRankType                 // weightedRankType = 2
	udfExprRankType                  // udfExprRankType = 3
)

var (
	rankTypeMap = map[string]rankType{
		"invalid":  invalidRankType,
		"rrf":      rrfRankType,
		"weighted": weightedRankType,
		"expr":     udfExprRankType,
	}
)

type reScorer interface {
	name() string
	reScore(input *milvuspb.SearchResults)
}

type baseScorer struct {
	scorerName   string
	roundDecimal int64
}

func (bs *baseScorer) name() string {
	return bs.scorerName
}

type rrfScorer struct {
	baseScorer
	k int
}

func (rs *rrfScorer) reScore(input *milvuspb.SearchResults) {

}

type weightedScorer struct {
	baseScorer
	weights []float64
}

func (ws *weightedScorer) reScore(input *milvuspb.SearchResults) {

}

func NewReScorer(reqs []*milvuspb.SearchRequest, rankParams []*commonpb.KeyValuePair) ([]reScorer, error) {
	rankTypeStr, err := funcutil.GetAttrByKeyFromRepeatedKV(RankTypeKey, rankParams)
	if err != nil {
		return nil, errors.New(RankTypeKey + " not found in rank_params")
	}

	if _, ok := rankTypeMap[rankTypeStr]; !ok {
		return nil, errors.Errorf("unsupported rank type %s", rankTypeStr)
	}

	roundDecimalStr, err := funcutil.GetAttrByKeyFromRepeatedKV(RoundDecimalKey, rankParams)
	if err != nil {
		roundDecimalStr = "-1"
	}

	roundDecimal, err := strconv.ParseInt(roundDecimalStr, 0, 64)
	if err != nil {
		return nil, fmt.Errorf("%s [%s] is invalid, should be -1 or an integer in range [0, 6]", RoundDecimalKey, roundDecimalStr)
	}

	if roundDecimal != -1 && (roundDecimal > 6 || roundDecimal < 0) {
		return nil, fmt.Errorf("%s [%s] is invalid, should be -1 or an integer in range [0, 6]", RoundDecimalKey, roundDecimalStr)
	}

	paramStr, err := funcutil.GetAttrByKeyFromRepeatedKV(RankParamsKey, rankParams)
	if err != nil {
		return nil, errors.New(RankParamsKey + " not found in rank_params")
	}

	var params map[string]interface{}
	err = json.Unmarshal([]byte(paramStr), &params)
	if err != nil {
		return nil, err
	}

	res := make([]reScorer, len(reqs))
	switch rankTypeMap[rankTypeStr] {
	case rrfRankType:
		for i := range reqs {
			k, ok := params[RRFParamsKey].(int)
			if !ok {
				return nil, errors.New(RRFParamsKey + " not found in rank_params")
			}
			res[i] = &rrfScorer{
				baseScorer: baseScorer{
					scorerName:   "rrf",
					roundDecimal: roundDecimal,
				},
				k: k,
			}
		}
	case weightedRankType:
		for i := range reqs {
			weights, ok := params[WeightsParamsKey].([]float64)
			if len(reqs) != len(weights) {
				return nil, merr.WrapErrParameterInvalid(fmt.Sprint(len(reqs)), fmt.Sprint(len(weights)), "weights mismatch with ann search requests")
			}
			if !ok {
				return nil, errors.New(WeightsParamsKey + " not found in rank_params")
			}
			res[i] = &weightedScorer{
				baseScorer: baseScorer{
					scorerName:   "weighted",
					roundDecimal: roundDecimal,
				},
				weights: weights,
			}
		}
	default:
		return nil, errors.Errorf("unsupported rank type %s", rankTypeStr)
	}

	return res, nil
}
