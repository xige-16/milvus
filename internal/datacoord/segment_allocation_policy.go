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

package datacoord

import (
	"errors"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"sort"
	"time"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

type estimatedSegLimit struct {
	minNumRow   int64
	maxDataSize int64
}

type calUpperLimitPolicy func(schema *schemapb.CollectionSchema) (*estimatedSegLimit, error)

func calBySchemaPolicy(schema *schemapb.CollectionSchema) (*estimatedSegLimit, error) {
	if schema == nil {
		return nil, errors.New("nil schema")
	}
	maxSizePerRecord, err := typeutil.EstimateSizePerRecord(schema)
	if err != nil {
		return nil, err
	}
	// check zero value, preventing panicking
	if maxSizePerRecord == 0 {
		return nil, errors.New("zero size record schema found")
	}
	threshold := Params.DataCoordCfg.SegmentMaxSize * 1024 * 1024
	return &estimatedSegLimit{
		minNumRow:   int64(threshold / float64(maxSizePerRecord)),
		maxDataSize: int64(threshold),
	}, nil
}

// AllocatePolicy helper function definition to allocate Segment space
type AllocatePolicy func(segments []*SegmentInfo, segmentIDReq *datapb.SegmentIDRequest,
	segmentLimit *estimatedSegLimit) ([]*Allocation, []*Allocation)

// AllocatePolicyV1 v1 policy simple allocation policy using Greedy Algorithm
func AllocatePolicyV1(segments []*SegmentInfo, segmentIDReq *datapb.SegmentIDRequest,
	segmentLimit *estimatedSegLimit) ([]*Allocation, []*Allocation) {
	count := int64(segmentIDReq.Count)
	estimatedCountPerSegment := segmentLimit.minNumRow
	newSegmentAllocations := make([]*Allocation, 0)
	existedSegmentAllocations := make([]*Allocation, 0)
	// create new segment if count >= max num
	for count >= estimatedCountPerSegment {
		allocation := getAllocation(estimatedCountPerSegment)
		newSegmentAllocations = append(newSegmentAllocations, allocation)
		count -= estimatedCountPerSegment
	}

	// allocate space for remaining count
	if count == 0 {
		return newSegmentAllocations, existedSegmentAllocations
	}
	for _, segment := range segments {
		var allocSize int64
		for _, allocation := range segment.allocations {
			allocSize += allocation.NumOfRows
		}
		free := segment.GetMaxRowNum() - segment.GetNumOfRows() - allocSize
		if free < count {
			continue
		}
		allocation := getAllocation(count)
		allocation.SegmentID = segment.GetID()
		existedSegmentAllocations = append(existedSegmentAllocations, allocation)
		return newSegmentAllocations, existedSegmentAllocations
	}

	// allocate new segment for remaining count
	allocation := getAllocation(count)
	newSegmentAllocations = append(newSegmentAllocations, allocation)
	return newSegmentAllocations, existedSegmentAllocations
}

// AllocatePolicyV2 v2 policy simple allocation policy by insert data size in segmentIDReq
func AllocatePolicyV2(segments []*SegmentInfo, segmentIDReq *datapb.SegmentIDRequest,
	segmentLimit *estimatedSegLimit) ([]*Allocation, []*Allocation) {
	reqDataSize := segmentIDReq.GetDataSize()
	maxSizePerSegment := uint64(segmentLimit.maxDataSize)

	newSegmentAllocations := make([]*Allocation, 0)
	existedSegmentAllocations := make([]*Allocation, 0)

	for reqDataSize >= maxSizePerSegment {
		allocation := getSizeAllocation(maxSizePerSegment)
		newSegmentAllocations = append(newSegmentAllocations, allocation)
		reqDataSize -= maxSizePerSegment
	}

	// allocate space for remaining data size
	if reqDataSize == 0 {
		return newSegmentAllocations, existedSegmentAllocations
	}
	for _, segment := range segments {
		var allocSize uint64
		for _, allocation := range segment.allocations {
			allocSize += allocation.DataSize
		}
		free := maxSizePerSegment - segment.GetNumOfRows() - allocSize
		if free < reqDataSize {
			continue
		}
		allocation := getSizeAllocation(reqDataSize)
		allocation.SegmentID = segment.GetID()
		existedSegmentAllocations = append(existedSegmentAllocations, allocation)
		return newSegmentAllocations, existedSegmentAllocations
	}

	// allocate new segment for remaining data size
	allocation := getSizeAllocation(reqDataSize)
	newSegmentAllocations = append(newSegmentAllocations, allocation)
	return newSegmentAllocations, existedSegmentAllocations
}

// segmentSealPolicy seal policy applies to segment
type segmentSealPolicy func(segment *SegmentInfo, ts Timestamp) bool

// getSegmentCapacityPolicy get segmentSealPolicy with segment size factor policy
func getSegmentCapacityPolicy(sizeFactor float64) segmentSealPolicy {
	return func(segment *SegmentInfo, ts Timestamp) bool {
		var allocSize int64
		for _, allocation := range segment.allocations {
			allocSize += allocation.NumOfRows
		}
		return float64(segment.currRows) >= sizeFactor*float64(segment.GetMaxRowNum())
	}
}

// getLastExpiresLifetimePolicy get segmentSealPolicy with lifetime limit compares ts - segment.lastExpireTime
func sealByLifetimePolicy(lifetime time.Duration) segmentSealPolicy {
	return func(segment *SegmentInfo, ts Timestamp) bool {
		pts, _ := tsoutil.ParseTS(ts)
		epts, _ := tsoutil.ParseTS(segment.GetLastExpireTime())
		d := pts.Sub(epts)
		return d >= lifetime
	}
}

// channelSealPolicy seal policy applies to channel
type channelSealPolicy func(string, []*SegmentInfo, Timestamp) []*SegmentInfo

// getChannelCapacityPolicy get channelSealPolicy with channel segment capacity policy
func getChannelOpenSegCapacityPolicy(limit int) channelSealPolicy {
	return func(channel string, segs []*SegmentInfo, ts Timestamp) []*SegmentInfo {
		if len(segs) <= limit {
			return []*SegmentInfo{}
		}
		sortSegmentsByLastExpires(segs)
		offLen := len(segs) - limit
		if offLen > len(segs) {
			offLen = len(segs)
		}
		return segs[0:offLen]
	}
}

// sortSegStatusByLastExpires sort segmentStatus with lastExpireTime ascending order
func sortSegmentsByLastExpires(segs []*SegmentInfo) {
	sort.Slice(segs, func(i, j int) bool {
		return segs[i].LastExpireTime < segs[j].LastExpireTime
	})
}

type flushPolicy func(segment *SegmentInfo, t Timestamp) bool

const flushInterval = 2 * time.Second

func flushPolicyV1(segment *SegmentInfo, t Timestamp) bool {
	return segment.GetState() == commonpb.SegmentState_Sealed &&
		segment.GetLastExpireTime() <= t &&
		time.Since(segment.lastFlushTime) >= flushInterval &&
		segment.currRows != 0
}
