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
	"math/rand"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	allocator2 "github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/datanode/allocator"
	"github.com/milvus-io/milvus/internal/datanode/broker"
	"github.com/milvus-io/milvus/internal/datanode/metacache"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/importutil"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/etcd"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
)

type DataNodeServicesSuite struct {
	suite.Suite

	broker  *broker.MockBroker
	node    *DataNode
	etcdCli *clientv3.Client
	ctx     context.Context
	cancel  context.CancelFunc
}

func TestDataNodeServicesSuite(t *testing.T) {
	suite.Run(t, new(DataNodeServicesSuite))
}

func (s *DataNodeServicesSuite) SetupSuite() {
	importutil.ReportImportAttempts = 1

	s.ctx, s.cancel = context.WithCancel(context.Background())
	etcdCli, err := etcd.GetEtcdClient(
		Params.EtcdCfg.UseEmbedEtcd.GetAsBool(),
		Params.EtcdCfg.EtcdUseSSL.GetAsBool(),
		Params.EtcdCfg.Endpoints.GetAsStrings(),
		Params.EtcdCfg.EtcdTLSCert.GetValue(),
		Params.EtcdCfg.EtcdTLSKey.GetValue(),
		Params.EtcdCfg.EtcdTLSCACert.GetValue(),
		Params.EtcdCfg.EtcdTLSMinVersion.GetValue())
	s.Require().NoError(err)
	s.etcdCli = etcdCli
}

func (s *DataNodeServicesSuite) SetupTest() {
	s.node = newIDLEDataNodeMock(s.ctx, schemapb.DataType_Int64)
	s.node.SetEtcdClient(s.etcdCli)

	err := s.node.Init()
	s.Require().NoError(err)

	alloc := allocator.NewMockAllocator(s.T())
	alloc.EXPECT().Start().Return(nil).Maybe()
	alloc.EXPECT().Close().Maybe()
	alloc.EXPECT().GetIDAlloactor().Return(&allocator2.IDAllocator{}).Maybe()
	alloc.EXPECT().Alloc(mock.Anything).Call.Return(int64(22222),
		func(count uint32) int64 {
			return int64(22222 + count)
		}, nil).Maybe()
	s.node.allocator = alloc

	meta := NewMetaFactory().GetCollectionMeta(1, "collection", schemapb.DataType_Int64)
	broker := broker.NewMockBroker(s.T())
	broker.EXPECT().GetSegmentInfo(mock.Anything, mock.Anything).
		Return([]*datapb.SegmentInfo{}, nil).Maybe()
	broker.EXPECT().DescribeCollection(mock.Anything, mock.Anything, mock.Anything).
		Return(&milvuspb.DescribeCollectionResponse{
			Status:    merr.Status(nil),
			Schema:    meta.GetSchema(),
			ShardsNum: common.DefaultShardsNum,
		}, nil).Maybe()
	broker.EXPECT().ReportTimeTick(mock.Anything, mock.Anything).Return(nil).Maybe()
	broker.EXPECT().SaveBinlogPaths(mock.Anything, mock.Anything).Return(nil).Maybe()
	broker.EXPECT().UpdateChannelCheckpoint(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
	broker.EXPECT().AllocTimestamp(mock.Anything, mock.Anything).Call.Return(tsoutil.ComposeTSByTime(time.Now(), 0),
		func(_ context.Context, num uint32) uint32 { return num }, nil).Maybe()

	s.broker = broker
	s.node.broker = broker

	err = s.node.Start()
	s.Require().NoError(err)

	s.node.chunkManager = storage.NewLocalChunkManager(storage.RootPath("/tmp/milvus_test/datanode"))
	paramtable.SetNodeID(1)
}

func (s *DataNodeServicesSuite) TearDownTest() {
	if s.broker != nil {
		s.broker.AssertExpectations(s.T())
		s.broker = nil
	}

	if s.node != nil {
		s.node.Stop()
		s.node = nil
	}
}

func (s *DataNodeServicesSuite) TearDownSuite() {
	s.cancel()
	err := s.etcdCli.Close()
	s.Require().NoError(err)
}

func (s *DataNodeServicesSuite) TestNotInUseAPIs() {
	s.Run("WatchDmChannels", func() {
		status, err := s.node.WatchDmChannels(s.ctx, &datapb.WatchDmChannelsRequest{})
		s.Assert().NoError(err)
		s.Assert().True(merr.Ok(status))
	})
	s.Run("GetTimeTickChannel", func() {
		_, err := s.node.GetTimeTickChannel(s.ctx, nil)
		s.Assert().NoError(err)
	})

	s.Run("GetStatisticsChannel", func() {
		_, err := s.node.GetStatisticsChannel(s.ctx, nil)
		s.Assert().NoError(err)
	})
}

func (s *DataNodeServicesSuite) TestGetComponentStates() {
	resp, err := s.node.GetComponentStates(s.ctx, nil)
	s.Assert().NoError(err)
	s.Assert().True(merr.Ok(resp.GetStatus()))
	s.Assert().Equal(common.NotRegisteredID, resp.State.NodeID)

	s.node.SetSession(&sessionutil.Session{})
	s.node.session.UpdateRegistered(true)
	resp, err = s.node.GetComponentStates(context.Background(), nil)
	s.Assert().NoError(err)
	s.Assert().True(merr.Ok(resp.GetStatus()))
}

func (s *DataNodeServicesSuite) TestGetCompactionState() {
	s.Run("success", func() {
		s.node.compactionExecutor.executing.Insert(int64(3), newMockCompactor(true))
		s.node.compactionExecutor.executing.Insert(int64(2), newMockCompactor(true))
		s.node.compactionExecutor.completed.Insert(int64(1), &datapb.CompactionPlanResult{
			PlanID: 1,
			State:  commonpb.CompactionState_Completed,
			Segments: []*datapb.CompactionSegment{
				{SegmentID: 10},
			},
		})
		stat, err := s.node.GetCompactionState(s.ctx, nil)
		s.Assert().NoError(err)
		s.Assert().Equal(3, len(stat.GetResults()))

		var mu sync.RWMutex
		cnt := 0
		for _, v := range stat.GetResults() {
			if v.GetState() == commonpb.CompactionState_Completed {
				mu.Lock()
				cnt++
				mu.Unlock()
			}
		}
		mu.Lock()
		s.Assert().Equal(1, cnt)
		mu.Unlock()

		s.Assert().Equal(1, s.node.compactionExecutor.completed.Len())
	})

	s.Run("unhealthy", func() {
		node := &DataNode{}
		node.UpdateStateCode(commonpb.StateCode_Abnormal)
		resp, _ := node.GetCompactionState(s.ctx, nil)
		s.Assert().Equal(merr.Code(merr.ErrServiceNotReady), resp.GetStatus().GetCode())
	})
}

func (s *DataNodeServicesSuite) TestCompaction() {
	dmChannelName := "by-dev-rootcoord-dml_0_100v0"
	schema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.TimeStampField, Name: common.TimeStampFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.StartOfUserFieldID, DataType: schemapb.DataType_Int64, IsPrimaryKey: true, Name: "pk"},
			{FieldID: common.StartOfUserFieldID + 1, DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{
				{Key: common.DimKey, Value: "128"},
			}},
		},
	}
	flushedSegmentID := int64(100)
	growingSegmentID := int64(101)

	vchan := &datapb.VchannelInfo{
		CollectionID:        1,
		ChannelName:         dmChannelName,
		UnflushedSegmentIds: []int64{},
		FlushedSegmentIds:   []int64{},
	}

	err := s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, vchan, schema, genTestTickler())
	s.Require().NoError(err)

	fgservice, ok := s.node.flowgraphManager.GetFlowgraphService(dmChannelName)
	s.Require().True(ok)

	metaCache := metacache.NewMockMetaCache(s.T())
	metaCache.EXPECT().Collection().Return(1).Maybe()
	metaCache.EXPECT().Schema().Return(schema).Maybe()
	s.node.writeBufferManager.Register(dmChannelName, metaCache, nil)

	fgservice.metacache.AddSegment(&datapb.SegmentInfo{
		ID:            flushedSegmentID,
		CollectionID:  1,
		PartitionID:   2,
		StartPosition: &msgpb.MsgPosition{},
	}, func(_ *datapb.SegmentInfo) *metacache.BloomFilterSet { return metacache.NewBloomFilterSet() })
	fgservice.metacache.AddSegment(&datapb.SegmentInfo{
		ID:            growingSegmentID,
		CollectionID:  1,
		PartitionID:   2,
		StartPosition: &msgpb.MsgPosition{},
	}, func(_ *datapb.SegmentInfo) *metacache.BloomFilterSet { return metacache.NewBloomFilterSet() })
	s.Run("service_not_ready", func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		node := &DataNode{}
		node.UpdateStateCode(commonpb.StateCode_Abnormal)
		req := &datapb.CompactionPlan{
			PlanID:  1000,
			Channel: dmChannelName,
		}

		resp, err := node.Compaction(ctx, req)
		s.NoError(err)
		s.False(merr.Ok(resp))
	})

	s.Run("channel_not_match", func() {
		node := s.node
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		req := &datapb.CompactionPlan{
			PlanID:  1000,
			Channel: dmChannelName + "other",
		}

		resp, err := node.Compaction(ctx, req)
		s.NoError(err)
		s.False(merr.Ok(resp))
	})

	s.Run("channel_dropped", func() {
		node := s.node
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		node.compactionExecutor.dropped.Insert(dmChannelName)
		defer node.compactionExecutor.dropped.Remove(dmChannelName)

		req := &datapb.CompactionPlan{
			PlanID:  1000,
			Channel: dmChannelName,
		}

		resp, err := node.Compaction(ctx, req)
		s.NoError(err)
		s.False(merr.Ok(resp))
	})

	s.Run("compact_growing_segment", func() {
		node := s.node
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		req := &datapb.CompactionPlan{
			PlanID:  1000,
			Channel: dmChannelName,
			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
				{SegmentID: 102, Level: datapb.SegmentLevel_L0},
				{SegmentID: growingSegmentID, Level: datapb.SegmentLevel_L1},
			},
		}

		resp, err := node.Compaction(ctx, req)
		s.NoError(err)
		s.False(merr.Ok(resp))
	})
}

func (s *DataNodeServicesSuite) TestFlushSegments() {
	dmChannelName := "fake-by-dev-rootcoord-dml-channel-test-FlushSegments"
	schema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.TimeStampField, Name: common.TimeStampFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.StartOfUserFieldID, DataType: schemapb.DataType_Int64, IsPrimaryKey: true, Name: "pk"},
			{FieldID: common.StartOfUserFieldID + 1, DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{
				{Key: common.DimKey, Value: "128"},
			}},
		},
	}
	segmentID := int64(100)

	vchan := &datapb.VchannelInfo{
		CollectionID:        1,
		ChannelName:         dmChannelName,
		UnflushedSegmentIds: []int64{},
		FlushedSegmentIds:   []int64{},
	}

	err := s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, vchan, schema, genTestTickler())
	s.Require().NoError(err)

	fgservice, ok := s.node.flowgraphManager.GetFlowgraphService(dmChannelName)
	s.Require().True(ok)

	metaCache := metacache.NewMockMetaCache(s.T())
	metaCache.EXPECT().Collection().Return(1).Maybe()
	metaCache.EXPECT().Schema().Return(schema).Maybe()
	s.node.writeBufferManager.Register(dmChannelName, metaCache, nil)

	fgservice.metacache.AddSegment(&datapb.SegmentInfo{
		ID:            segmentID,
		CollectionID:  1,
		PartitionID:   2,
		StartPosition: &msgpb.MsgPosition{},
	}, func(_ *datapb.SegmentInfo) *metacache.BloomFilterSet { return metacache.NewBloomFilterSet() })

	s.Run("service_not_ready", func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		node := &DataNode{}
		node.UpdateStateCode(commonpb.StateCode_Abnormal)
		req := &datapb.FlushSegmentsRequest{
			Base: &commonpb.MsgBase{
				TargetID: s.node.GetSession().ServerID,
			},
			DbID:         0,
			CollectionID: 1,
			SegmentIDs:   []int64{0},
		}

		resp, err := node.FlushSegments(ctx, req)
		s.NoError(err)
		s.False(merr.Ok(resp))
	})

	s.Run("node_id_not_match", func() {
		node := s.node
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		req := &datapb.FlushSegmentsRequest{
			Base: &commonpb.MsgBase{
				TargetID: s.node.GetSession().ServerID + 1,
			},
			DbID:         0,
			CollectionID: 1,
			SegmentIDs:   []int64{0},
		}

		resp, err := node.FlushSegments(ctx, req)
		s.NoError(err)
		s.False(merr.Ok(resp))
	})

	s.Run("channel_not_found", func() {
		node := s.node
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		req := &datapb.FlushSegmentsRequest{
			Base: &commonpb.MsgBase{
				TargetID: s.node.GetSession().ServerID,
			},
			DbID:         0,
			CollectionID: 1,
			SegmentIDs:   []int64{segmentID},
		}

		resp, err := node.FlushSegments(ctx, req)
		s.NoError(err)
		s.False(merr.Ok(resp))
	})

	s.Run("normal_flush", func() {
		node := s.node
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		req := &datapb.FlushSegmentsRequest{
			Base: &commonpb.MsgBase{
				TargetID: s.node.GetSession().ServerID,
			},
			DbID:         0,
			CollectionID: 1,
			SegmentIDs:   []int64{segmentID},
			ChannelName:  dmChannelName,
		}

		resp, err := node.FlushSegments(ctx, req)
		s.NoError(err)
		s.True(merr.Ok(resp))
	})
}

func (s *DataNodeServicesSuite) TestShowConfigurations() {
	pattern := "datanode.Port"
	req := &internalpb.ShowConfigurationsRequest{
		Base: &commonpb.MsgBase{
			MsgType: commonpb.MsgType_WatchQueryChannels,
			MsgID:   rand.Int63(),
		},
		Pattern: pattern,
	}

	// test closed server
	node := &DataNode{}
	node.SetSession(&sessionutil.Session{SessionRaw: sessionutil.SessionRaw{ServerID: 1}})
	node.stateCode.Store(commonpb.StateCode_Abnormal)

	resp, err := node.ShowConfigurations(s.ctx, req)
	s.Assert().NoError(err)
	s.Assert().False(merr.Ok(resp.GetStatus()))

	node.stateCode.Store(commonpb.StateCode_Healthy)
	resp, err = node.ShowConfigurations(s.ctx, req)
	s.Assert().NoError(err)
	s.Assert().True(merr.Ok(resp.GetStatus()))
	s.Assert().Equal(1, len(resp.Configuations))
	s.Assert().Equal("datanode.port", resp.Configuations[0].Key)
}

func (s *DataNodeServicesSuite) TestGetMetrics() {
	node := &DataNode{}
	node.SetSession(&sessionutil.Session{SessionRaw: sessionutil.SessionRaw{ServerID: 1}})
	node.flowgraphManager = newFlowgraphManager()
	// server is closed
	node.stateCode.Store(commonpb.StateCode_Abnormal)
	resp, err := node.GetMetrics(s.ctx, &milvuspb.GetMetricsRequest{})
	s.Assert().NoError(err)
	s.Assert().False(merr.Ok(resp.GetStatus()))

	node.stateCode.Store(commonpb.StateCode_Healthy)

	// failed to parse metric type
	invalidRequest := "invalid request"
	resp, err = node.GetMetrics(s.ctx, &milvuspb.GetMetricsRequest{
		Request: invalidRequest,
	})
	s.Assert().NoError(err)
	s.Assert().False(merr.Ok(resp.GetStatus()))

	// unsupported metric type
	unsupportedMetricType := "unsupported"
	req, err := metricsinfo.ConstructRequestByMetricType(unsupportedMetricType)
	s.Assert().NoError(err)
	resp, err = node.GetMetrics(s.ctx, req)
	s.Assert().NoError(err)
	s.Assert().False(merr.Ok(resp.GetStatus()))

	// normal case
	req, err = metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
	s.Assert().NoError(err)
	resp, err = node.GetMetrics(node.ctx, req)
	s.Assert().NoError(err)
	s.Assert().True(merr.Ok(resp.GetStatus()))
	log.Info("Test DataNode.GetMetrics",
		zap.String("name", resp.ComponentName),
		zap.String("response", resp.Response))
}

func (s *DataNodeServicesSuite) TestImport() {
	s.node.rootCoord = &RootCoordFactory{
		collectionID: 100,
		pkType:       schemapb.DataType_Int64,
	}

	content := []byte(`{
		"rows":[
			{"bool_field": true, "int8_field": 10, "int16_field": 101, "int32_field": 1001, "int64_field": 10001, "float32_field": 3.14, "float64_field": 1.56, "varChar_field": "hello world", "binary_vector_field": [254, 0, 254, 0], "float_vector_field": [1.1, 1.2]},
			{"bool_field": false, "int8_field": 11, "int16_field": 102, "int32_field": 1002, "int64_field": 10002, "float32_field": 3.15, "float64_field": 2.56, "varChar_field": "hello world", "binary_vector_field": [253, 0, 253, 0], "float_vector_field": [2.1, 2.2]},
			{"bool_field": true, "int8_field": 12, "int16_field": 103, "int32_field": 1003, "int64_field": 10003, "float32_field": 3.16, "float64_field": 3.56, "varChar_field": "hello world", "binary_vector_field": [252, 0, 252, 0], "float_vector_field": [3.1, 3.2]},
			{"bool_field": false, "int8_field": 13, "int16_field": 104, "int32_field": 1004, "int64_field": 10004, "float32_field": 3.17, "float64_field": 4.56, "varChar_field": "hello world", "binary_vector_field": [251, 0, 251, 0], "float_vector_field": [4.1, 4.2]},
			{"bool_field": true, "int8_field": 14, "int16_field": 105, "int32_field": 1005, "int64_field": 10005, "float32_field": 3.18, "float64_field": 5.56, "varChar_field": "hello world", "binary_vector_field": [250, 0, 250, 0], "float_vector_field": [5.1, 5.2]}
		]
		}`)
	filePath := filepath.Join(s.node.chunkManager.RootPath(), "rows_1.json")
	err := s.node.chunkManager.Write(s.ctx, filePath, content)
	s.Require().NoError(err)

	s.node.reportImportRetryTimes = 1 // save test time cost from 440s to 180s
	s.Run("test normal", func() {
		defer func() {
			s.TearDownTest()
		}()
		chName1 := "fake-by-dev-rootcoord-dml-testimport-1"
		chName2 := "fake-by-dev-rootcoord-dml-testimport-2"
		err := s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, &datapb.VchannelInfo{
			CollectionID:        100,
			ChannelName:         chName1,
			UnflushedSegmentIds: []int64{},
			FlushedSegmentIds:   []int64{},
		}, nil, genTestTickler())
		s.Require().Nil(err)
		err = s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, &datapb.VchannelInfo{
			CollectionID:        100,
			ChannelName:         chName2,
			UnflushedSegmentIds: []int64{},
			FlushedSegmentIds:   []int64{},
		}, nil, genTestTickler())
		s.Require().Nil(err)

		_, ok := s.node.flowgraphManager.GetFlowgraphService(chName1)
		s.Require().True(ok)
		_, ok = s.node.flowgraphManager.GetFlowgraphService(chName2)
		s.Require().True(ok)

		req := &datapb.ImportTaskRequest{
			ImportTask: &datapb.ImportTask{
				CollectionId: 100,
				PartitionId:  100,
				ChannelNames: []string{chName1, chName2},
				Files:        []string{filePath},
				RowBased:     true,
			},
		}

		s.broker.EXPECT().ReportImport(mock.Anything, mock.Anything).Return(nil)
		s.broker.EXPECT().UpdateSegmentStatistics(mock.Anything, mock.Anything).Return(nil)
		s.broker.EXPECT().AssignSegmentID(mock.Anything, mock.Anything).
			Return([]int64{10001}, nil)
		s.broker.EXPECT().SaveImportSegment(mock.Anything, mock.Anything).Return(nil)

		s.node.Import(s.ctx, req)

		stat, err := s.node.Import(context.WithValue(s.ctx, ctxKey{}, ""), req)
		s.Assert().NoError(err)
		s.Assert().True(merr.Ok(stat))
		s.Assert().Equal("", stat.GetReason())

		reqWithoutPartition := &datapb.ImportTaskRequest{
			ImportTask: &datapb.ImportTask{
				CollectionId: 100,
				ChannelNames: []string{chName1, chName2},
				Files:        []string{filePath},
				RowBased:     true,
			},
		}
		stat2, err := s.node.Import(context.WithValue(s.ctx, ctxKey{}, ""), reqWithoutPartition)
		s.Assert().NoError(err)
		s.Assert().False(merr.Ok(stat2))
	})

	s.Run("Test Import bad flow graph", func() {
		s.SetupTest()
		defer func() {
			s.TearDownTest()
		}()
		chName1 := "fake-by-dev-rootcoord-dml-testimport-1-badflowgraph"
		chName2 := "fake-by-dev-rootcoord-dml-testimport-2-badflowgraph"
		err := s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, &datapb.VchannelInfo{
			CollectionID:        100,
			ChannelName:         chName1,
			UnflushedSegmentIds: []int64{},
			FlushedSegmentIds:   []int64{},
		}, nil, genTestTickler())
		s.Require().Nil(err)
		err = s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, &datapb.VchannelInfo{
			CollectionID:        999, // wrong collection ID.
			ChannelName:         chName2,
			UnflushedSegmentIds: []int64{},
			FlushedSegmentIds:   []int64{},
		}, nil, genTestTickler())
		s.Require().Nil(err)

		_, ok := s.node.flowgraphManager.GetFlowgraphService(chName1)
		s.Require().True(ok)
		_, ok = s.node.flowgraphManager.GetFlowgraphService(chName2)
		s.Require().True(ok)

		s.broker.EXPECT().UpdateSegmentStatistics(mock.Anything, mock.Anything).Return(nil)
		s.broker.EXPECT().ReportImport(mock.Anything, mock.Anything).Return(nil)
		s.broker.EXPECT().AssignSegmentID(mock.Anything, mock.Anything).
			Return([]int64{10001}, nil)
		s.broker.EXPECT().SaveImportSegment(mock.Anything, mock.Anything).Return(nil)

		req := &datapb.ImportTaskRequest{
			ImportTask: &datapb.ImportTask{
				CollectionId: 100,
				PartitionId:  100,
				ChannelNames: []string{chName1, chName2},
				Files:        []string{filePath},
				RowBased:     true,
			},
		}
		stat, err := s.node.Import(context.WithValue(s.ctx, ctxKey{}, ""), req)
		s.Assert().NoError(err)
		s.Assert().True(merr.Ok(stat))
		s.Assert().Equal("", stat.GetReason())
	})
	s.Run("test_Import_report_import_error", func() {
		s.SetupTest()
		s.node.reportImportRetryTimes = 1
		defer func() {
			s.TearDownTest()
		}()

		s.broker.EXPECT().AssignSegmentID(mock.Anything, mock.Anything).
			Return([]int64{10001}, nil)
		s.broker.EXPECT().ReportImport(mock.Anything, mock.Anything).Return(errors.New("mocked"))
		s.broker.EXPECT().UpdateSegmentStatistics(mock.Anything, mock.Anything).Return(nil)
		s.broker.EXPECT().SaveImportSegment(mock.Anything, mock.Anything).Return(nil)

		req := &datapb.ImportTaskRequest{
			ImportTask: &datapb.ImportTask{
				CollectionId: 100,
				PartitionId:  100,
				ChannelNames: []string{"ch1", "ch2"},
				Files:        []string{filePath},
				RowBased:     true,
			},
		}
		stat, err := s.node.Import(s.ctx, req)
		s.Assert().NoError(err)
		s.Assert().False(merr.Ok(stat))
	})

	s.Run("test_import_error", func() {
		s.SetupTest()
		defer func() {
			s.TearDownTest()
		}()
		s.broker.ExpectedCalls = nil
		s.broker.EXPECT().DescribeCollection(mock.Anything, mock.Anything, mock.Anything).
			Return(&milvuspb.DescribeCollectionResponse{
				Status: merr.Status(merr.WrapErrCollectionNotFound("collection")),
			}, nil)
		s.broker.EXPECT().GetSegmentInfo(mock.Anything, mock.Anything).
			Return([]*datapb.SegmentInfo{}, nil).Maybe()
		s.broker.EXPECT().ReportTimeTick(mock.Anything, mock.Anything).Return(nil).Maybe()
		s.broker.EXPECT().SaveBinlogPaths(mock.Anything, mock.Anything).Return(nil).Maybe()
		s.broker.EXPECT().UpdateChannelCheckpoint(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.broker.EXPECT().AllocTimestamp(mock.Anything, mock.Anything).Call.Return(tsoutil.ComposeTSByTime(time.Now(), 0),
			func(_ context.Context, num uint32) uint32 { return num }, nil).Maybe()

		s.broker.EXPECT().ReportImport(mock.Anything, mock.Anything).Return(nil)
		req := &datapb.ImportTaskRequest{
			ImportTask: &datapb.ImportTask{
				CollectionId: 100,
				PartitionId:  100,
			},
		}
		stat, err := s.node.Import(context.WithValue(s.ctx, ctxKey{}, ""), req)
		s.Assert().NoError(err)
		s.Assert().False(merr.Ok(stat))

		stat, err = s.node.Import(context.WithValue(s.ctx, ctxKey{}, returnError), req)
		s.Assert().NoError(err)
		s.Assert().False(merr.Ok(stat))

		s.node.stateCode.Store(commonpb.StateCode_Abnormal)
		stat, err = s.node.Import(context.WithValue(s.ctx, ctxKey{}, ""), req)
		s.Assert().NoError(err)
		s.Assert().False(merr.Ok(stat))
	})
}

func (s *DataNodeServicesSuite) TestAddImportSegment() {
	schema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.TimeStampField, Name: common.TimeStampFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.StartOfUserFieldID, DataType: schemapb.DataType_Int64, IsPrimaryKey: true, Name: "pk"},
			{FieldID: common.StartOfUserFieldID + 1, DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{
				{Key: common.DimKey, Value: "128"},
			}},
		},
	}
	s.Run("test AddSegment", func() {
		s.node.rootCoord = &RootCoordFactory{
			collectionID: 100,
			pkType:       schemapb.DataType_Int64,
		}

		chName1 := "fake-by-dev-rootcoord-dml-testaddsegment-1"
		chName2 := "fake-by-dev-rootcoord-dml-testaddsegment-2"
		err := s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, &datapb.VchannelInfo{
			CollectionID:        100,
			ChannelName:         chName1,
			UnflushedSegmentIds: []int64{},
			FlushedSegmentIds:   []int64{},
		}, schema, genTestTickler())
		s.Require().NoError(err)
		err = s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, &datapb.VchannelInfo{
			CollectionID:        100,
			ChannelName:         chName2,
			UnflushedSegmentIds: []int64{},
			FlushedSegmentIds:   []int64{},
		}, schema, genTestTickler())
		s.Require().NoError(err)

		_, ok := s.node.flowgraphManager.GetFlowgraphService(chName1)
		s.Assert().True(ok)
		_, ok = s.node.flowgraphManager.GetFlowgraphService(chName2)
		s.Assert().True(ok)

		resp, err := s.node.AddImportSegment(context.WithValue(s.ctx, ctxKey{}, ""), &datapb.AddImportSegmentRequest{
			SegmentId:    100,
			CollectionId: 100,
			PartitionId:  100,
			ChannelName:  chName1,
			RowNum:       500,
		})
		s.Assert().NoError(err)
		s.Assert().True(merr.Ok(resp.GetStatus()))
		s.Assert().Equal("", resp.GetStatus().GetReason())
		s.Assert().NotEqual(nil, resp.GetChannelPos())

		getFlowGraphServiceAttempts = 3
		resp, err = s.node.AddImportSegment(context.WithValue(s.ctx, ctxKey{}, ""), &datapb.AddImportSegmentRequest{
			SegmentId:    100,
			CollectionId: 100,
			PartitionId:  100,
			ChannelName:  "bad-ch-name",
			RowNum:       500,
		})
		s.Assert().NoError(err)
		// TODO ASSERT COMBINE ERROR
		s.Assert().False(merr.Ok(resp.GetStatus()))
		// s.Assert().Equal(merr.Code(merr.ErrChannelNotFound), stat.GetStatus().GetCode())
	})
}

func (s *DataNodeServicesSuite) TestSyncSegments() {
	chanName := "fake-by-dev-rootcoord-dml-test-syncsegments-1"
	schema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: common.RowIDField, Name: common.RowIDFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.TimeStampField, Name: common.TimeStampFieldName, DataType: schemapb.DataType_Int64},
			{FieldID: common.StartOfUserFieldID, DataType: schemapb.DataType_Int64, IsPrimaryKey: true, Name: "pk"},
			{FieldID: common.StartOfUserFieldID + 1, DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{
				{Key: common.DimKey, Value: "128"},
			}},
		},
	}

	err := s.node.flowgraphManager.AddandStartWithEtcdTickler(s.node, &datapb.VchannelInfo{
		CollectionID:        1,
		ChannelName:         chanName,
		UnflushedSegmentIds: []int64{},
		FlushedSegmentIds:   []int64{100, 200, 300},
	}, schema, genTestTickler())
	s.Require().NoError(err)
	fg, ok := s.node.flowgraphManager.GetFlowgraphService(chanName)
	s.Assert().True(ok)

	fg.metacache.AddSegment(&datapb.SegmentInfo{ID: 100, CollectionID: 1, State: commonpb.SegmentState_Flushed}, EmptyBfsFactory)
	fg.metacache.AddSegment(&datapb.SegmentInfo{ID: 101, CollectionID: 1, State: commonpb.SegmentState_Flushed}, EmptyBfsFactory)
	fg.metacache.AddSegment(&datapb.SegmentInfo{ID: 200, CollectionID: 1, State: commonpb.SegmentState_Flushed}, EmptyBfsFactory)
	fg.metacache.AddSegment(&datapb.SegmentInfo{ID: 201, CollectionID: 1, State: commonpb.SegmentState_Flushed}, EmptyBfsFactory)
	fg.metacache.AddSegment(&datapb.SegmentInfo{ID: 300, CollectionID: 1, State: commonpb.SegmentState_Flushed}, EmptyBfsFactory)

	s.Run("invalid compacted from", func() {
		req := &datapb.SyncSegmentsRequest{
			CompactedTo: 400,
			NumOfRows:   100,
		}

		req.CompactedFrom = []UniqueID{}
		status, err := s.node.SyncSegments(s.ctx, req)
		s.Assert().NoError(err)
		s.Assert().False(merr.Ok(status))

		req.CompactedFrom = []UniqueID{101, 201}
		status, err = s.node.SyncSegments(s.ctx, req)
		s.Assert().NoError(err)
		s.Assert().False(merr.Ok(status))
	})

	s.Run("valid request numRows>0", func() {
		req := &datapb.SyncSegmentsRequest{
			CompactedFrom: []UniqueID{100, 200, 101, 201},
			CompactedTo:   102,
			NumOfRows:     100,
			ChannelName:   chanName,
			CollectionId:  1,
		}
		status, err := s.node.SyncSegments(s.ctx, req)
		s.Assert().NoError(err)
		s.Assert().True(merr.Ok(status))

		_, result := fg.metacache.GetSegmentByID(req.GetCompactedTo(), metacache.WithSegmentState(commonpb.SegmentState_Flushed))
		s.True(result)
		for _, compactFrom := range req.GetCompactedFrom() {
			seg, result := fg.metacache.GetSegmentByID(compactFrom, metacache.WithSegmentState(commonpb.SegmentState_Flushed))
			s.True(result)
			s.Equal(req.CompactedTo, seg.CompactTo())
		}

		status, err = s.node.SyncSegments(s.ctx, req)
		s.Assert().NoError(err)
		s.Assert().True(merr.Ok(status))
	})

	s.Run("without_channel_meta", func() {
		fg.metacache.UpdateSegments(metacache.UpdateState(commonpb.SegmentState_Flushed),
			metacache.WithSegmentIDs(100, 200, 300))

		req := &datapb.SyncSegmentsRequest{
			CompactedFrom: []int64{100, 200},
			CompactedTo:   101,
			NumOfRows:     0,
		}
		status, err := s.node.SyncSegments(s.ctx, req)
		s.Assert().NoError(err)
		s.Assert().False(merr.Ok(status))
	})

	s.Run("valid_request_with_meta_num=0", func() {
		fg.metacache.UpdateSegments(metacache.UpdateState(commonpb.SegmentState_Flushed),
			metacache.WithSegmentIDs(100, 200, 300))

		req := &datapb.SyncSegmentsRequest{
			CompactedFrom: []int64{100, 200},
			CompactedTo:   301,
			NumOfRows:     0,
			ChannelName:   chanName,
			CollectionId:  1,
		}
		status, err := s.node.SyncSegments(s.ctx, req)
		s.Assert().NoError(err)
		s.Assert().True(merr.Ok(status))

		seg, result := fg.metacache.GetSegmentByID(100, metacache.WithSegmentState(commonpb.SegmentState_Flushed))
		s.True(result)
		s.Equal(metacache.NullSegment, seg.CompactTo())
		seg, result = fg.metacache.GetSegmentByID(200, metacache.WithSegmentState(commonpb.SegmentState_Flushed))
		s.True(result)
		s.Equal(metacache.NullSegment, seg.CompactTo())
		_, result = fg.metacache.GetSegmentByID(301, metacache.WithSegmentState(commonpb.SegmentState_Flushed))
		s.False(result)
	})
}

func (s *DataNodeServicesSuite) TestResendSegmentStats() {
	req := &datapb.ResendSegmentStatsRequest{
		Base: &commonpb.MsgBase{},
	}

	resp, err := s.node.ResendSegmentStats(s.ctx, req)
	s.Assert().NoError(err, "empty call, no error")
	s.Assert().True(merr.Ok(resp.GetStatus()), "empty call, status shall be OK")
}

/*
func (s *DataNodeServicesSuite) TestFlushChannels() {
	dmChannelName := "fake-by-dev-rootcoord-dml-channel-TestFlushChannels"

	vChan := &datapb.VchannelInfo{
		CollectionID:        1,
		ChannelName:         dmChannelName,
		UnflushedSegmentIds: []int64{},
		FlushedSegmentIds:   []int64{},
	}

	err := s.node.flowgraphManager.addAndStartWithEtcdTickler(s.node, vChan, nil, genTestTickler())
	s.Require().NoError(err)

	fgService, ok := s.node.flowgraphManager.getFlowgraphService(dmChannelName)
	s.Require().True(ok)

	flushTs := Timestamp(100)

	req := &datapb.FlushChannelsRequest{
		Base: &commonpb.MsgBase{
			TargetID: s.node.GetSession().ServerID,
		},
		FlushTs:  flushTs,
		Channels: []string{dmChannelName},
	}

	status, err := s.node.FlushChannels(s.ctx, req)
	s.Assert().NoError(err)
	s.Assert().True(merr.Ok(status))

	s.Assert().True(fgService.channel.getFlushTs() == flushTs)
}*/

func (s *DataNodeServicesSuite) TestRPCWatch() {
	ctx := context.Background()
	status, err := s.node.NotifyChannelOperation(ctx, nil)
	s.NoError(err)
	s.NotNil(status)

	resp, err := s.node.CheckChannelOperationProgress(ctx, nil)
	s.NoError(err)
	s.NotNil(resp)
}
