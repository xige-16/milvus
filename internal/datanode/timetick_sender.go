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
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus/internal/datanode/broker"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/util/retry"
)

// timeTickSender is to merge channel states updated by flow graph node and send to datacoord periodically
// timeTickSender hold a SegmentStats time sequence cache for each channel,
// after send succeeds will clean the cache earlier than the sended timestamp
type timeTickSender struct {
	nodeID int64
	broker broker.Broker

	wg         sync.WaitGroup
	cancelFunc context.CancelFunc

	options []retry.Option

	mu                  sync.Mutex
	channelStatesCaches map[string]*segmentStatesSequence // string -> *segmentStatesSequence
}

// data struct only used in timeTickSender
type segmentStatesSequence struct {
	data map[uint64][]*commonpb.SegmentStats // ts -> segmentStats
}

func newTimeTickSender(broker broker.Broker, nodeID int64, opts ...retry.Option) *timeTickSender {
	return &timeTickSender{
		nodeID:              nodeID,
		broker:              broker,
		channelStatesCaches: make(map[string]*segmentStatesSequence, 0),
		options:             opts,
	}
}

func (m *timeTickSender) start() {
	m.wg.Add(1)
	ctx, cancel := context.WithCancel(context.Background())
	m.cancelFunc = cancel
	go func() {
		defer m.wg.Done()
		m.work(ctx)
	}()
}

func (m *timeTickSender) Stop() {
	if m.cancelFunc != nil {
		m.cancelFunc()
		m.wg.Wait()
	}
}

func (m *timeTickSender) work(ctx context.Context) {
	ticker := time.NewTicker(Params.DataNodeCfg.DataNodeTimeTickInterval.GetAsDuration(time.Millisecond))
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			log.Info("timeTickSender context done")
			return
		case <-ticker.C:
			m.sendReport(ctx)
		}
	}
}

func (m *timeTickSender) update(channelName string, timestamp uint64, segmentStats []*commonpb.SegmentStats) {
	m.mu.Lock()
	defer m.mu.Unlock()
	channelStates, ok := m.channelStatesCaches[channelName]
	if !ok {
		channelStates = &segmentStatesSequence{
			data: make(map[uint64][]*commonpb.SegmentStats, 0),
		}
	}
	channelStates.data[timestamp] = segmentStats
	m.channelStatesCaches[channelName] = channelStates
}

func (m *timeTickSender) mergeDatanodeTtMsg() ([]*msgpb.DataNodeTtMsg, map[string]uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var msgs []*msgpb.DataNodeTtMsg
	sendedLastTss := make(map[string]uint64, 0)

	for channelName, channelSegmentStates := range m.channelStatesCaches {
		var lastTs uint64
		segNumRows := make(map[int64]int64, 0)
		for ts, segmentStates := range channelSegmentStates.data {
			if ts > lastTs {
				lastTs = ts
			}
			// merge the same segments into one
			for _, segmentStat := range segmentStates {
				if v, ok := segNumRows[segmentStat.GetSegmentID()]; ok {
					// numRows is supposed to keep growing
					if segmentStat.GetNumRows() > v {
						segNumRows[segmentStat.GetSegmentID()] = segmentStat.GetNumRows()
					}
				} else {
					segNumRows[segmentStat.GetSegmentID()] = segmentStat.GetNumRows()
				}
			}
		}
		toSendSegmentStats := make([]*commonpb.SegmentStats, 0)
		for id, numRows := range segNumRows {
			toSendSegmentStats = append(toSendSegmentStats, &commonpb.SegmentStats{
				SegmentID: id,
				NumRows:   numRows,
			})
		}
		msgs = append(msgs, &msgpb.DataNodeTtMsg{
			Base: commonpbutil.NewMsgBase(
				commonpbutil.WithMsgType(commonpb.MsgType_DataNodeTt),
				commonpbutil.WithSourceID(m.nodeID),
			),
			ChannelName:   channelName,
			Timestamp:     lastTs,
			SegmentsStats: toSendSegmentStats,
		})
		sendedLastTss[channelName] = lastTs
	}

	return msgs, sendedLastTss
}

func (m *timeTickSender) cleanStatesCache(sendedLastTss map[string]uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	sizeBeforeClean := len(m.channelStatesCaches)
	log := log.With(zap.Any("sendedLastTss", sendedLastTss), zap.Int("sizeBeforeClean", sizeBeforeClean))
	for channelName, sendedLastTs := range sendedLastTss {
		channelCache, ok := m.channelStatesCaches[channelName]
		if ok {
			for ts := range channelCache.data {
				if ts <= sendedLastTs {
					delete(channelCache.data, ts)
				}
			}
			m.channelStatesCaches[channelName] = channelCache
		}
		if len(channelCache.data) == 0 {
			delete(m.channelStatesCaches, channelName)
		}
	}
	log.RatedDebug(30, "timeTickSender channelStatesCaches", zap.Int("sizeAfterClean", len(m.channelStatesCaches)))
}

func (m *timeTickSender) sendReport(ctx context.Context) error {
	toSendMsgs, sendLastTss := m.mergeDatanodeTtMsg()
	log.RatedDebug(30, "timeTickSender send datanode timetick message", zap.Any("toSendMsgs", toSendMsgs), zap.Any("sendLastTss", sendLastTss))
	err := retry.Do(ctx, func() error {
		return m.broker.ReportTimeTick(ctx, toSendMsgs)
	}, m.options...)
	if err != nil {
		log.Error("ReportDataNodeTtMsgs fail after retry", zap.Error(err))
		return err
	}
	m.cleanStatesCache(sendLastTss)
	return nil
}
