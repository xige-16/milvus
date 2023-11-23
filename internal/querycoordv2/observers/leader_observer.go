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

package observers

import (
	"context"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/utils"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
)

const (
	interval   = 1 * time.Second
	RPCTimeout = 3 * time.Second
)

// LeaderObserver is to sync the distribution with leader
type LeaderObserver struct {
	wg      sync.WaitGroup
	cancel  context.CancelFunc
	dist    *meta.DistributionManager
	meta    *meta.Meta
	target  *meta.TargetManager
	broker  meta.Broker
	cluster session.Cluster
	nodeMgr *session.NodeManager

	dispatcher *taskDispatcher[int64]

	stopOnce sync.Once
}

func (o *LeaderObserver) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	o.cancel = cancel

	o.dispatcher.Start()

	o.wg.Add(1)
	go func() {
		defer o.wg.Done()
		o.schedule(ctx)
	}()
}

func (o *LeaderObserver) Stop() {
	o.stopOnce.Do(func() {
		if o.cancel != nil {
			o.cancel()
		}
		o.wg.Wait()

		o.dispatcher.Stop()
	})
}

func (o *LeaderObserver) schedule(ctx context.Context) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			log.Info("stop leader observer")
			return

		case <-ticker.C:
			o.observe(ctx)
		}
	}
}

func (o *LeaderObserver) observe(ctx context.Context) {
	o.observeSegmentsDist(ctx)
}

func (o *LeaderObserver) readyToObserve(collectionID int64) bool {
	metaExist := (o.meta.GetCollection(collectionID) != nil)
	targetExist := o.target.IsNextTargetExist(collectionID) || o.target.IsCurrentTargetExist(collectionID)

	return metaExist && targetExist
}

func (o *LeaderObserver) observeSegmentsDist(ctx context.Context) {
	collectionIDs := o.meta.CollectionManager.GetAll()
	for _, cid := range collectionIDs {
		if o.readyToObserve(cid) {
			o.dispatcher.AddTask(cid)
		}
	}
}

func (o *LeaderObserver) observeCollection(ctx context.Context, collection int64) {
	replicas := o.meta.ReplicaManager.GetByCollection(collection)
	for _, replica := range replicas {
		leaders := o.dist.ChannelDistManager.GetShardLeadersByReplica(replica)
		for ch, leaderID := range leaders {
			if ok, _ := o.nodeMgr.IsStoppingNode(leaderID); ok {
				// no need to correct leader's view which is loaded on stopping node
				continue
			}

			leaderView := o.dist.LeaderViewManager.GetLeaderShardView(leaderID, ch)
			if leaderView == nil {
				continue
			}
			dists := o.dist.SegmentDistManager.GetByShardWithReplica(ch, replica)

			actions := o.findNeedLoadedSegments(leaderView, dists)
			actions = append(actions, o.findNeedRemovedSegments(leaderView, dists)...)
			o.sync(ctx, replica.GetID(), leaderView, actions)
		}
	}
}

func (o *LeaderObserver) findNeedLoadedSegments(leaderView *meta.LeaderView, dists []*meta.Segment) []*querypb.SyncAction {
	ret := make([]*querypb.SyncAction, 0)
	dists = utils.FindMaxVersionSegments(dists)
	for _, s := range dists {
		version, ok := leaderView.Segments[s.GetID()]
		currentTarget := o.target.GetSealedSegment(s.CollectionID, s.GetID(), meta.CurrentTarget)
		existInCurrentTarget := currentTarget != nil
		existInNextTarget := o.target.GetSealedSegment(s.CollectionID, s.GetID(), meta.NextTarget) != nil

		if !existInCurrentTarget && !existInNextTarget {
			continue
		}

		if !ok || version.GetVersion() < s.Version { // Leader misses this segment
			ctx := context.Background()
			resp, err := o.broker.GetSegmentInfo(ctx, s.GetID())
			if err != nil || len(resp.GetInfos()) == 0 {
				log.Warn("failed to get segment info from DataCoord", zap.Error(err))
				continue
			}
			loadInfo := utils.PackSegmentLoadInfo(resp, nil)

			log.Debug("leader observer append a segment to set",
				zap.Int64("collectionID", leaderView.CollectionID),
				zap.String("channel", leaderView.Channel),
				zap.Int64("leaderViewID", leaderView.ID),
				zap.Int64("segmentID", s.GetID()),
				zap.Int64("nodeID", s.Node))
			ret = append(ret, &querypb.SyncAction{
				Type:        querypb.SyncType_Set,
				PartitionID: s.GetPartitionID(),
				SegmentID:   s.GetID(),
				NodeID:      s.Node,
				Version:     s.Version,
				Info:        loadInfo,
			})
		}
	}
	return ret
}

func (o *LeaderObserver) findNeedRemovedSegments(leaderView *meta.LeaderView, dists []*meta.Segment) []*querypb.SyncAction {
	ret := make([]*querypb.SyncAction, 0)
	distMap := make(map[int64]struct{})
	for _, s := range dists {
		distMap[s.GetID()] = struct{}{}
	}
	for sid, s := range leaderView.Segments {
		_, ok := distMap[sid]
		existInCurrentTarget := o.target.GetSealedSegment(leaderView.CollectionID, sid, meta.CurrentTarget) != nil
		existInNextTarget := o.target.GetSealedSegment(leaderView.CollectionID, sid, meta.NextTarget) != nil
		if ok || existInCurrentTarget || existInNextTarget {
			continue
		}
		log.Debug("leader observer append a segment to remove",
			zap.Int64("collectionID", leaderView.CollectionID),
			zap.String("channel", leaderView.Channel),
			zap.Int64("leaderViewID", leaderView.ID),
			zap.Int64("segmentID", sid),
			zap.Int64("nodeID", s.NodeID))
		ret = append(ret, &querypb.SyncAction{
			Type:      querypb.SyncType_Remove,
			SegmentID: sid,
			NodeID:    s.NodeID,
		})
	}
	return ret
}

func (o *LeaderObserver) sync(ctx context.Context, replicaID int64, leaderView *meta.LeaderView, diffs []*querypb.SyncAction) bool {
	if len(diffs) == 0 {
		return true
	}

	log := log.With(
		zap.Int64("leaderID", leaderView.ID),
		zap.Int64("collectionID", leaderView.CollectionID),
		zap.String("channel", leaderView.Channel),
	)

	collectionInfo, err := o.broker.DescribeCollection(ctx, leaderView.CollectionID)
	if err != nil {
		log.Warn("failed to get collection info", zap.Error(err))
		return false
	}
	partitions, err := utils.GetPartitions(o.meta.CollectionManager, leaderView.CollectionID)
	if err != nil {
		log.Warn("failed to get partitions", zap.Error(err))
		return false
	}

	req := &querypb.SyncDistributionRequest{
		Base: commonpbutil.NewMsgBase(
			commonpbutil.WithMsgType(commonpb.MsgType_SyncDistribution),
		),
		CollectionID: leaderView.CollectionID,
		ReplicaID:    replicaID,
		Channel:      leaderView.Channel,
		Actions:      diffs,
		Schema:       collectionInfo.GetSchema(),
		LoadMeta: &querypb.LoadMetaInfo{
			LoadType:     o.meta.GetLoadType(leaderView.CollectionID),
			CollectionID: leaderView.CollectionID,
			PartitionIDs: partitions,
		},
		Version: time.Now().UnixNano(),
	}
	ctx, cancel := context.WithTimeout(ctx, paramtable.Get().QueryCoordCfg.SegmentTaskTimeout.GetAsDuration(time.Millisecond))
	defer cancel()
	resp, err := o.cluster.SyncDistribution(ctx, leaderView.ID, req)
	if err != nil {
		log.Warn("failed to sync distribution", zap.Error(err))
		return false
	}

	if resp.ErrorCode != commonpb.ErrorCode_Success {
		log.Warn("failed to sync distribution", zap.String("reason", resp.GetReason()))
		return false
	}

	return true
}

func NewLeaderObserver(
	dist *meta.DistributionManager,
	meta *meta.Meta,
	targetMgr *meta.TargetManager,
	broker meta.Broker,
	cluster session.Cluster,
	nodeMgr *session.NodeManager,
) *LeaderObserver {
	ob := &LeaderObserver{
		dist:    dist,
		meta:    meta,
		target:  targetMgr,
		broker:  broker,
		cluster: cluster,
		nodeMgr: nodeMgr,
	}

	dispatcher := newTaskDispatcher[int64](ob.observeCollection)
	ob.dispatcher = dispatcher

	return ob
}
