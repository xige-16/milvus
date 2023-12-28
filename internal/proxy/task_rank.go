package proxy

import (
	"encoding/json"
	"fmt"
	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"go.uber.org/zap"
	"reflect"
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
	k float32
}

func (rs *rrfScorer) reScore(input *milvuspb.SearchResults) {
	for i := range input.Results.Scores {
		input.Results.Scores[i] = 1 / (rs.k + float32(i+1))
	}
}

type weightedScorer struct {
	baseScorer
	weight float32
}

func (ws *weightedScorer) reScore(input *milvuspb.SearchResults) {
	for i, score := range input.Results.Scores {
		input.Results.Scores[i] = ws.weight * score
	}
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
	log.Info("rank params", zap.Any("params", params), zap.String("param str", paramStr))

	res := make([]reScorer, len(reqs))
	switch rankTypeMap[rankTypeStr] {
	case rrfRankType:
		k, ok := params[RRFParamsKey].(float64)
		if !ok {
			return nil, errors.New(RRFParamsKey + " not found in rank_params")
		}
		log.Info("rrf params", zap.Float64("k", k))
		for i := range reqs {
			res[i] = &rrfScorer{
				baseScorer: baseScorer{
					scorerName:   "rrf",
					roundDecimal: roundDecimal,
				},
				k: float32(k),
			}
		}
	case weightedRankType:
		if _, ok := params[WeightsParamsKey]; !ok {
			return nil, errors.New(WeightsParamsKey + " not found in rank_params")
		}
		weights := make([]float32, 0)
		switch reflect.TypeOf(params[WeightsParamsKey]).Kind() {
		case reflect.Slice:
			rs := reflect.ValueOf(params[WeightsParamsKey])

			for i := 0; i < rs.Len(); i++ {
				weights = append(weights, float32(rs.Index(i).Interface().(float64)))
			}
		default:
			return nil, errors.New("The weights param should be an array")
		}

		log.Info("weights params", zap.Any("weights", weights))
		//weights, ok := params[WeightsParamsKey].([]float32)
		if len(reqs) != len(weights) {
			return nil, merr.WrapErrParameterInvalid(fmt.Sprint(len(reqs)), fmt.Sprint(len(weights)), "weights mismatch with ann search requests")
		}
		for i := range reqs {
			res[i] = &weightedScorer{
				baseScorer: baseScorer{
					scorerName:   "weighted",
					roundDecimal: roundDecimal,
				},
				weight: weights[i],
			}
		}
	default:
		return nil, errors.Errorf("unsupported rank type %s", rankTypeStr)
	}

	return res, nil
}
