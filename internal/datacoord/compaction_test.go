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
	"context"
	"sync"
	"testing"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	mockkv "github.com/milvus-io/milvus/internal/kv/mocks"
	"github.com/milvus-io/milvus/internal/metastore/kv/datacoord"
	"github.com/milvus-io/milvus/internal/mocks"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

func Test_compactionPlanHandler_execCompactionPlan(t *testing.T) {
	type fields struct {
		plans            map[int64]*compactionTask
		sessions         *SessionManager
		chManager        *ChannelManager
		allocatorFactory func() allocator
	}
	type args struct {
		signal *compactionSignal
		plan   *datapb.CompactionPlan
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
		err     error
	}{
		{
			"test exec compaction",
			fields{
				plans: map[int64]*compactionTask{},
				sessions: &SessionManager{
					sessions: struct {
						sync.RWMutex
						data map[int64]*Session
					}{
						data: map[int64]*Session{
							1: {client: &mockDataNodeClient{ch: make(chan interface{}, 1)}},
						},
					},
				},
				chManager: &ChannelManager{
					store: &ChannelStore{
						channelsInfo: map[int64]*NodeChannelInfo{
							1: {NodeID: 1, Channels: []RWChannel{&channelMeta{Name: "ch1"}}},
						},
					},
				},
				allocatorFactory: func() allocator { return newMockAllocator() },
			},
			args{
				signal: &compactionSignal{id: 100},
				plan:   &datapb.CompactionPlan{PlanID: 1, Channel: "ch1", Type: datapb.CompactionType_MergeCompaction},
			},
			false,
			nil,
		},
		{
			"test exec compaction failed",
			fields{
				plans: map[int64]*compactionTask{},
				chManager: &ChannelManager{
					store: &ChannelStore{
						channelsInfo: map[int64]*NodeChannelInfo{
							1:        {NodeID: 1, Channels: []RWChannel{}},
							bufferID: {NodeID: bufferID, Channels: []RWChannel{}},
						},
					},
				},
				allocatorFactory: func() allocator { return newMockAllocator() },
			},
			args{
				signal: &compactionSignal{id: 100},
				plan:   &datapb.CompactionPlan{PlanID: 1, Channel: "ch1", Type: datapb.CompactionType_MergeCompaction},
			},
			true,
			errChannelNotWatched,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &compactionPlanHandler{
				plans:     tt.fields.plans,
				sessions:  tt.fields.sessions,
				chManager: tt.fields.chManager,
				allocator: tt.fields.allocatorFactory(),
				scheduler: NewCompactionScheduler(),
			}
			Params.Save(Params.DataCoordCfg.CompactionCheckIntervalInSeconds.Key, "1")
			c.start()
			err := c.execCompactionPlan(tt.args.signal, tt.args.plan)
			require.ErrorIs(t, tt.err, err)

			task := c.getCompaction(tt.args.plan.PlanID)
			if !tt.wantErr {
				assert.Equal(t, tt.args.plan, task.plan)
				assert.Equal(t, tt.args.signal, task.triggerInfo)
				assert.Equal(t, 1, c.scheduler.getExecutingTaskNum())
			} else {
				assert.Eventually(t,
					func() bool {
						c.scheduler.mu.RLock()
						defer c.scheduler.mu.RUnlock()
						return c.scheduler.getExecutingTaskNum() == 0 && len(c.scheduler.parallelTasks[1]) == 0
					},
					5*time.Second, 100*time.Millisecond)
			}
			c.stop()
		})
	}
}

func Test_compactionPlanHandler_execWithParallels(t *testing.T) {
	mockDataNode := &mocks.MockDataNodeClient{}
	paramtable.Get().Save(Params.DataCoordCfg.CompactionCheckIntervalInSeconds.Key, "0.001")
	defer paramtable.Get().Reset(Params.DataCoordCfg.CompactionCheckIntervalInSeconds.Key)
	c := &compactionPlanHandler{
		plans: map[int64]*compactionTask{},
		sessions: &SessionManager{
			sessions: struct {
				sync.RWMutex
				data map[int64]*Session
			}{
				data: map[int64]*Session{
					1: {client: mockDataNode},
				},
			},
		},
		chManager: &ChannelManager{
			store: &ChannelStore{
				channelsInfo: map[int64]*NodeChannelInfo{
					1: {NodeID: 1, Channels: []RWChannel{&channelMeta{Name: "ch1"}}},
				},
			},
		},
		allocator: newMockAllocator(),
		scheduler: NewCompactionScheduler(),
	}

	signal := &compactionSignal{id: 100}
	plan1 := &datapb.CompactionPlan{PlanID: 1, Channel: "ch1", Type: datapb.CompactionType_MergeCompaction}
	plan2 := &datapb.CompactionPlan{PlanID: 2, Channel: "ch1", Type: datapb.CompactionType_MergeCompaction}
	plan3 := &datapb.CompactionPlan{PlanID: 3, Channel: "ch1", Type: datapb.CompactionType_MergeCompaction}

	var mut sync.RWMutex
	called := 0

	mockDataNode.EXPECT().Compaction(mock.Anything, mock.Anything, mock.Anything).
		Run(func(ctx context.Context, req *datapb.CompactionPlan, opts ...grpc.CallOption) {
			mut.Lock()
			defer mut.Unlock()
			called++
		}).Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil).Times(2)

	err := c.execCompactionPlan(signal, plan1)
	require.NoError(t, err)
	err = c.execCompactionPlan(signal, plan2)
	require.NoError(t, err)
	err = c.execCompactionPlan(signal, plan3)
	require.NoError(t, err)

	assert.Equal(t, 3, c.scheduler.getExecutingTaskNum())

	// parallel for the same node are 2
	tasks := c.scheduler.schedule()
	assert.Equal(t, 1, len(tasks))
	assert.Equal(t, int64(1), tasks[0].plan.PlanID)
	assert.Equal(t, int64(1), tasks[0].dataNodeID)
	c.notifyTasks(tasks)

	tasks = c.scheduler.schedule()
	assert.Equal(t, 1, len(tasks))
	assert.Equal(t, int64(2), tasks[0].plan.PlanID)
	assert.Equal(t, int64(1), tasks[0].dataNodeID)
	c.notifyTasks(tasks)

	// wait for compaction called
	assert.Eventually(t, func() bool {
		mut.RLock()
		defer mut.RUnlock()
		return called == 2
	}, 3*time.Second, time.Millisecond*100)

	tasks = c.scheduler.schedule()
	assert.Equal(t, 0, len(tasks))
}

func getInsertLogPath(rootPath string, segmentID typeutil.UniqueID) string {
	return metautil.BuildInsertLogPath(rootPath, 10, 100, segmentID, 1000, 10000)
}

func getStatsLogPath(rootPath string, segmentID typeutil.UniqueID) string {
	return metautil.BuildStatsLogPath(rootPath, 10, 100, segmentID, 1000, 10000)
}

func getDeltaLogPath(rootPath string, segmentID typeutil.UniqueID) string {
	return metautil.BuildDeltaLogPath(rootPath, 10, 100, segmentID, 10000)
}

func TestCompactionPlanHandler_handleMergeCompactionResult(t *testing.T) {
	mockDataNode := &mocks.MockDataNodeClient{}
	call := mockDataNode.EXPECT().SyncSegments(mock.Anything, mock.Anything, mock.Anything).
		Run(func(ctx context.Context, req *datapb.SyncSegmentsRequest, opts ...grpc.CallOption) {}).
		Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

	dataNodeID := UniqueID(111)

	seg1 := &datapb.SegmentInfo{
		ID:        1,
		Binlogs:   []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log1", 1))},
		Statslogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log2", 1))},
		Deltalogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log3", 1))},
	}

	seg2 := &datapb.SegmentInfo{
		ID:        2,
		Binlogs:   []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log4", 2))},
		Statslogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log5", 2))},
		Deltalogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log6", 2))},
	}

	plan := &datapb.CompactionPlan{
		PlanID: 1,
		SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
			{
				SegmentID:           seg1.ID,
				FieldBinlogs:        seg1.GetBinlogs(),
				Field2StatslogPaths: seg1.GetStatslogs(),
				Deltalogs:           seg1.GetDeltalogs(),
			},
			{
				SegmentID:           seg2.ID,
				FieldBinlogs:        seg2.GetBinlogs(),
				Field2StatslogPaths: seg2.GetStatslogs(),
				Deltalogs:           seg2.GetDeltalogs(),
			},
		},
		Type: datapb.CompactionType_MergeCompaction,
	}

	sessions := &SessionManager{
		sessions: struct {
			sync.RWMutex
			data map[int64]*Session
		}{
			data: map[int64]*Session{
				dataNodeID: {client: mockDataNode},
			},
		},
	}

	task := &compactionTask{
		triggerInfo: &compactionSignal{id: 1},
		state:       executing,
		plan:        plan,
		dataNodeID:  dataNodeID,
	}

	plans := map[int64]*compactionTask{1: task}

	metakv := mockkv.NewMetaKv(t)
	metakv.EXPECT().Save(mock.Anything, mock.Anything).Return(errors.New("failed")).Maybe()
	metakv.EXPECT().MultiSave(mock.Anything).Return(errors.New("failed")).Maybe()
	metakv.EXPECT().HasPrefix(mock.Anything).Return(false, nil).Maybe()
	errMeta := &meta{
		catalog: &datacoord.Catalog{MetaKv: metakv},
		segments: &SegmentsInfo{
			map[int64]*SegmentInfo{
				seg1.ID: {SegmentInfo: seg1},
				seg2.ID: {SegmentInfo: seg2},
			},
		},
	}

	meta := &meta{
		catalog: &datacoord.Catalog{MetaKv: NewMetaMemoryKV()},
		segments: &SegmentsInfo{
			map[int64]*SegmentInfo{
				seg1.ID: {SegmentInfo: seg1},
				seg2.ID: {SegmentInfo: seg2},
			},
		},
	}

	c := &compactionPlanHandler{
		plans:    plans,
		sessions: sessions,
		meta:     meta,
	}

	c2 := &compactionPlanHandler{
		plans:    plans,
		sessions: sessions,
		meta:     errMeta,
	}

	compactionResult := &datapb.CompactionPlanResult{
		PlanID: 1,
		Segments: []*datapb.CompactionSegment{
			{
				SegmentID:           3,
				NumOfRows:           15,
				InsertLogs:          []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log301", 3))},
				Field2StatslogPaths: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log302", 3))},
				Deltalogs:           []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log303", 3))},
			},
		},
	}

	compactionResult2 := &datapb.CompactionPlanResult{
		PlanID: 1,
		Segments: []*datapb.CompactionSegment{
			{
				SegmentID:           3,
				NumOfRows:           0,
				InsertLogs:          []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log301", 3))},
				Field2StatslogPaths: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log302", 3))},
				Deltalogs:           []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log303", 3))},
			},
		},
	}

	has, err := c.meta.HasSegments([]UniqueID{1, 2})
	require.NoError(t, err)
	require.True(t, has)

	has, err = c.meta.HasSegments([]UniqueID{3})
	require.Error(t, err)
	require.False(t, has)

	err = c.handleMergeCompactionResult(plan, compactionResult)
	assert.NoError(t, err)

	err = c.handleMergeCompactionResult(plan, compactionResult2)
	assert.NoError(t, err)

	err = c2.handleMergeCompactionResult(plan, compactionResult2)
	assert.Error(t, err)

	has, err = c.meta.HasSegments([]UniqueID{1, 2, 3})
	require.NoError(t, err)
	require.True(t, has)

	call.Unset()
	mockDataNode.EXPECT().SyncSegments(mock.Anything, mock.Anything, mock.Anything).
		Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_UnexpectedError}, nil)
	err = c.handleMergeCompactionResult(plan, compactionResult2)
	assert.Error(t, err)
}

func TestCompactionPlanHandler_completeCompaction(t *testing.T) {
	t.Run("test not exists compaction task", func(t *testing.T) {
		c := &compactionPlanHandler{
			plans: map[int64]*compactionTask{1: {}},
		}
		err := c.completeCompaction(&datapb.CompactionPlanResult{PlanID: 2})
		assert.Error(t, err)
	})
	t.Run("test completed compaction task", func(t *testing.T) {
		c := &compactionPlanHandler{
			plans: map[int64]*compactionTask{1: {state: completed}},
		}
		err := c.completeCompaction(&datapb.CompactionPlanResult{PlanID: 1})
		assert.Error(t, err)
	})

	t.Run("test complete merge compaction task", func(t *testing.T) {
		mockDataNode := &mocks.MockDataNodeClient{}
		mockDataNode.EXPECT().SyncSegments(mock.Anything, mock.Anything, mock.Anything).
			Run(func(ctx context.Context, req *datapb.SyncSegmentsRequest, opts ...grpc.CallOption) {}).
			Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		dataNodeID := UniqueID(111)

		seg1 := &datapb.SegmentInfo{
			ID:        1,
			Binlogs:   []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log1", 1))},
			Statslogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log2", 1))},
			Deltalogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log3", 1))},
		}

		seg2 := &datapb.SegmentInfo{
			ID:        2,
			Binlogs:   []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log4", 2))},
			Statslogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log5", 2))},
			Deltalogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log6", 2))},
		}

		plan := &datapb.CompactionPlan{
			PlanID: 1,
			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
				{
					SegmentID:           seg1.ID,
					FieldBinlogs:        seg1.GetBinlogs(),
					Field2StatslogPaths: seg1.GetStatslogs(),
					Deltalogs:           seg1.GetDeltalogs(),
				},
				{
					SegmentID:           seg2.ID,
					FieldBinlogs:        seg2.GetBinlogs(),
					Field2StatslogPaths: seg2.GetStatslogs(),
					Deltalogs:           seg2.GetDeltalogs(),
				},
			},
			Type: datapb.CompactionType_MergeCompaction,
		}

		sessions := &SessionManager{
			sessions: struct {
				sync.RWMutex
				data map[int64]*Session
			}{
				data: map[int64]*Session{
					dataNodeID: {client: mockDataNode},
				},
			},
		}

		task := &compactionTask{
			triggerInfo: &compactionSignal{id: 1},
			state:       executing,
			plan:        plan,
			dataNodeID:  dataNodeID,
		}

		plans := map[int64]*compactionTask{1: task}

		meta := &meta{
			catalog: &datacoord.Catalog{MetaKv: NewMetaMemoryKV()},
			segments: &SegmentsInfo{
				map[int64]*SegmentInfo{
					seg1.ID: {SegmentInfo: seg1},
					seg2.ID: {SegmentInfo: seg2},
				},
			},
		}
		compactionResult := datapb.CompactionPlanResult{
			PlanID: 1,
			Segments: []*datapb.CompactionSegment{
				{
					SegmentID:           3,
					NumOfRows:           15,
					InsertLogs:          []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log301", 3))},
					Field2StatslogPaths: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log302", 3))},
					Deltalogs:           []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log303", 3))},
				},
			},
		}

		c := &compactionPlanHandler{
			plans:     plans,
			sessions:  sessions,
			meta:      meta,
			scheduler: NewCompactionScheduler(),
		}

		err := c.completeCompaction(&compactionResult)
		assert.NoError(t, err)
	})

	t.Run("test empty result merge compaction task", func(t *testing.T) {
		mockDataNode := &mocks.MockDataNodeClient{}
		mockDataNode.EXPECT().SyncSegments(mock.Anything, mock.Anything, mock.Anything).
			Run(func(ctx context.Context, req *datapb.SyncSegmentsRequest, opts ...grpc.CallOption) {}).
			Return(&commonpb.Status{ErrorCode: commonpb.ErrorCode_Success}, nil)

		dataNodeID := UniqueID(111)

		seg1 := &datapb.SegmentInfo{
			ID:        1,
			Binlogs:   []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log1", 1))},
			Statslogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log2", 1))},
			Deltalogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log3", 1))},
		}

		seg2 := &datapb.SegmentInfo{
			ID:        2,
			Binlogs:   []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log4", 2))},
			Statslogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log5", 2))},
			Deltalogs: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log6", 2))},
		}

		plan := &datapb.CompactionPlan{
			PlanID: 1,
			SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
				{
					SegmentID:           seg1.ID,
					FieldBinlogs:        seg1.GetBinlogs(),
					Field2StatslogPaths: seg1.GetStatslogs(),
					Deltalogs:           seg1.GetDeltalogs(),
				},
				{
					SegmentID:           seg2.ID,
					FieldBinlogs:        seg2.GetBinlogs(),
					Field2StatslogPaths: seg2.GetStatslogs(),
					Deltalogs:           seg2.GetDeltalogs(),
				},
			},
			Type: datapb.CompactionType_MergeCompaction,
		}

		sessions := &SessionManager{
			sessions: struct {
				sync.RWMutex
				data map[int64]*Session
			}{
				data: map[int64]*Session{
					dataNodeID: {client: mockDataNode},
				},
			},
		}

		task := &compactionTask{
			triggerInfo: &compactionSignal{id: 1},
			state:       executing,
			plan:        plan,
			dataNodeID:  dataNodeID,
		}

		plans := map[int64]*compactionTask{1: task}

		meta := &meta{
			catalog: &datacoord.Catalog{MetaKv: NewMetaMemoryKV()},
			segments: &SegmentsInfo{
				map[int64]*SegmentInfo{
					seg1.ID: {SegmentInfo: seg1},
					seg2.ID: {SegmentInfo: seg2},
				},
			},
		}

		meta.AddSegment(context.TODO(), NewSegmentInfo(seg1))
		meta.AddSegment(context.TODO(), NewSegmentInfo(seg2))

		segments := meta.GetAllSegmentsUnsafe()
		assert.Equal(t, len(segments), 2)
		compactionResult := datapb.CompactionPlanResult{
			PlanID: 1,
			Segments: []*datapb.CompactionSegment{
				{
					SegmentID:           3,
					NumOfRows:           0,
					InsertLogs:          []*datapb.FieldBinlog{getFieldBinlogPaths(101, getInsertLogPath("log301", 3))},
					Field2StatslogPaths: []*datapb.FieldBinlog{getFieldBinlogPaths(101, getStatsLogPath("log302", 3))},
					Deltalogs:           []*datapb.FieldBinlog{getFieldBinlogPaths(101, getDeltaLogPath("log303", 3))},
				},
			},
		}

		c := &compactionPlanHandler{
			plans:     plans,
			sessions:  sessions,
			meta:      meta,
			scheduler: NewCompactionScheduler(),
		}

		err := c.completeCompaction(&compactionResult)
		assert.NoError(t, err)

		segments = meta.GetAllSegmentsUnsafe()
		assert.Equal(t, len(segments), 3)

		for _, segment := range segments {
			assert.True(t, segment.State == commonpb.SegmentState_Dropped)
		}
	})
}

func Test_compactionPlanHandler_getCompaction(t *testing.T) {
	type fields struct {
		plans    map[int64]*compactionTask
		sessions *SessionManager
	}
	type args struct {
		planID int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *compactionTask
	}{
		{
			"test get non existed task",
			fields{plans: map[int64]*compactionTask{}},
			args{planID: 1},
			nil,
		},
		{
			"test get existed task",
			fields{
				plans: map[int64]*compactionTask{1: {
					state: executing,
				}},
			},
			args{planID: 1},
			&compactionTask{
				state: executing,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &compactionPlanHandler{
				plans:    tt.fields.plans,
				sessions: tt.fields.sessions,
			}
			got := c.getCompaction(tt.args.planID)
			assert.EqualValues(t, tt.want, got)
		})
	}
}

func Test_compactionPlanHandler_updateCompaction(t *testing.T) {
	type fields struct {
		plans    map[int64]*compactionTask
		sessions *SessionManager
		meta     *meta
	}
	type args struct {
		ts Timestamp
	}

	ts := time.Now()
	tests := []struct {
		name      string
		fields    fields
		args      args
		wantErr   bool
		timeout   []int64
		failed    []int64
		unexpired []int64
	}{
		{
			"test update compaction task",
			fields{
				plans: map[int64]*compactionTask{
					1: {
						state:      executing,
						dataNodeID: 1,
						plan: &datapb.CompactionPlan{
							PlanID:           1,
							StartTime:        tsoutil.ComposeTS(ts.UnixNano()/int64(time.Millisecond), 0),
							TimeoutInSeconds: 10,
							SegmentBinlogs: []*datapb.CompactionSegmentBinlogs{
								{SegmentID: 1},
							},
						},
					},
					2: {
						state:      executing,
						dataNodeID: 2,
						plan: &datapb.CompactionPlan{
							PlanID:           2,
							StartTime:        tsoutil.ComposeTS(ts.UnixNano()/int64(time.Millisecond), 0),
							TimeoutInSeconds: 1,
						},
					},
					3: {
						state:      executing,
						dataNodeID: 2,
						plan: &datapb.CompactionPlan{
							PlanID:           3,
							StartTime:        tsoutil.ComposeTS(ts.UnixNano()/int64(time.Millisecond), 0),
							TimeoutInSeconds: 1,
						},
					},
					4: {
						state:      executing,
						dataNodeID: 2,
						plan: &datapb.CompactionPlan{
							PlanID:           4,
							StartTime:        tsoutil.ComposeTS(ts.UnixNano()/int64(time.Millisecond), 0) - 200*1000,
							TimeoutInSeconds: 1,
						},
					},
					5: { // timeout and failed
						state:      timeout,
						dataNodeID: 2,
						plan: &datapb.CompactionPlan{
							PlanID:           5,
							StartTime:        tsoutil.ComposeTS(ts.UnixNano()/int64(time.Millisecond), 0) - 200*1000,
							TimeoutInSeconds: 1,
						},
					},
					6: { // timeout and executing
						state:      timeout,
						dataNodeID: 2,
						plan: &datapb.CompactionPlan{
							PlanID:           6,
							StartTime:        tsoutil.ComposeTS(ts.UnixNano()/int64(time.Millisecond), 0) - 200*1000,
							TimeoutInSeconds: 1,
						},
					},
				},
				meta: &meta{
					segments: &SegmentsInfo{
						map[int64]*SegmentInfo{
							1: {SegmentInfo: &datapb.SegmentInfo{ID: 1}},
						},
					},
				},
				sessions: &SessionManager{
					sessions: struct {
						sync.RWMutex
						data map[int64]*Session
					}{
						data: map[int64]*Session{
							2: {client: &mockDataNodeClient{
								compactionStateResp: &datapb.CompactionStateResponse{
									Results: []*datapb.CompactionPlanResult{
										{PlanID: 1, State: commonpb.CompactionState_Executing},
										{PlanID: 3, State: commonpb.CompactionState_Completed, Segments: []*datapb.CompactionSegment{{PlanID: 3}}},
										{PlanID: 4, State: commonpb.CompactionState_Executing},
										{PlanID: 6, State: commonpb.CompactionState_Executing},
									},
								},
							}},
						},
					},
				},
			},
			args{ts: tsoutil.ComposeTS(ts.Add(5*time.Second).UnixNano()/int64(time.Millisecond), 0)},
			false,
			[]int64{4, 6},
			[]int64{2, 5},
			[]int64{1, 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &compactionPlanHandler{
				plans:     tt.fields.plans,
				sessions:  tt.fields.sessions,
				meta:      tt.fields.meta,
				scheduler: NewCompactionScheduler(),
			}

			err := c.updateCompaction(tt.args.ts)
			assert.Equal(t, tt.wantErr, err != nil)

			for _, id := range tt.timeout {
				task := c.getCompaction(id)
				assert.Equal(t, timeout, task.state)
			}

			for _, id := range tt.failed {
				task := c.getCompaction(id)
				assert.Equal(t, failed, task.state)
			}

			for _, id := range tt.unexpired {
				task := c.getCompaction(id)
				assert.NotEqual(t, failed, task.state)
			}

			c.scheduler.mu.Lock()
			assert.Equal(t, 0, len(c.scheduler.parallelTasks[2]))
			c.scheduler.mu.Unlock()
		})
	}
}

func Test_newCompactionPlanHandler(t *testing.T) {
	type args struct {
		sessions  *SessionManager
		cm        *ChannelManager
		meta      *meta
		allocator allocator
	}
	tests := []struct {
		name string
		args args
		want *compactionPlanHandler
	}{
		{
			"test new handler",
			args{
				&SessionManager{},
				&ChannelManager{},
				&meta{},
				newMockAllocator(),
			},
			&compactionPlanHandler{
				plans:     map[int64]*compactionTask{},
				sessions:  &SessionManager{},
				chManager: &ChannelManager{},
				meta:      &meta{},
				allocator: newMockAllocator(),
				scheduler: NewCompactionScheduler(),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := newCompactionPlanHandler(tt.args.sessions, tt.args.cm, tt.args.meta, tt.args.allocator)
			assert.EqualValues(t, tt.want, got)
		})
	}
}

func Test_getCompactionTasksBySignalID(t *testing.T) {
	type fields struct {
		plans map[int64]*compactionTask
	}
	type args struct {
		signalID int64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   []*compactionTask
	}{
		{
			"test get compaction tasks",
			fields{
				plans: map[int64]*compactionTask{
					1: {
						triggerInfo: &compactionSignal{id: 1},
						state:       executing,
					},
					2: {
						triggerInfo: &compactionSignal{id: 1},
						state:       completed,
					},
					3: {
						triggerInfo: &compactionSignal{id: 1},
						state:       failed,
					},
				},
			},
			args{1},
			[]*compactionTask{
				{
					triggerInfo: &compactionSignal{id: 1},
					state:       executing,
				},
				{
					triggerInfo: &compactionSignal{id: 1},
					state:       completed,
				},
				{
					triggerInfo: &compactionSignal{id: 1},
					state:       failed,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &compactionPlanHandler{
				plans: tt.fields.plans,
			}
			got := h.getCompactionTasksBySignalID(tt.args.signalID)
			assert.ElementsMatch(t, tt.want, got)
		})
	}
}

func getFieldBinlogPaths(id int64, paths ...string) *datapb.FieldBinlog {
	l := &datapb.FieldBinlog{
		FieldID: id,
		Binlogs: make([]*datapb.Binlog, 0, len(paths)),
	}
	for _, path := range paths {
		l.Binlogs = append(l.Binlogs, &datapb.Binlog{LogPath: path})
	}
	return l
}

func getFieldBinlogPathsWithEntry(id int64, entry int64, paths ...string) *datapb.FieldBinlog {
	l := &datapb.FieldBinlog{
		FieldID: id,
		Binlogs: make([]*datapb.Binlog, 0, len(paths)),
	}
	for _, path := range paths {
		l.Binlogs = append(l.Binlogs, &datapb.Binlog{LogPath: path, EntriesNum: entry})
	}
	return l
}
