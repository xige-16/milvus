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

package rootcoord

import (
	"context"
	"fmt"
	"sync"

	"github.com/cockroachdb/errors"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	grpcproxyclient "github.com/milvus-io/milvus/internal/distributed/proxy/client"
	"github.com/milvus-io/milvus/internal/proto/proxypb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/sessionutil"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type proxyCreator func(ctx context.Context, addr string, nodeID int64) (types.ProxyClient, error)

func DefaultProxyCreator(ctx context.Context, addr string, nodeID int64) (types.ProxyClient, error) {
	cli, err := grpcproxyclient.NewClient(ctx, addr, nodeID)
	if err != nil {
		return nil, err
	}
	return cli, nil
}

type proxyClientManager struct {
	creator     proxyCreator
	proxyClient *typeutil.ConcurrentMap[int64, types.ProxyClient]
	helper      proxyClientManagerHelper
}

type proxyClientManagerHelper struct {
	afterConnect func()
}

var defaultClientManagerHelper = proxyClientManagerHelper{
	afterConnect: func() {},
}

func newProxyClientManager(creator proxyCreator) *proxyClientManager {
	return &proxyClientManager{
		creator:     creator,
		proxyClient: typeutil.NewConcurrentMap[int64, types.ProxyClient](),
		helper:      defaultClientManagerHelper,
	}
}

func (p *proxyClientManager) AddProxyClients(sessions []*sessionutil.Session) {
	for _, session := range sessions {
		p.AddProxyClient(session)
	}
}

func (p *proxyClientManager) GetProxyClients() *typeutil.ConcurrentMap[int64, types.ProxyClient] {
	return p.proxyClient
}

func (p *proxyClientManager) AddProxyClient(session *sessionutil.Session) {
	_, ok := p.proxyClient.Get(session.ServerID)
	if ok {
		return
	}

	p.connect(session)
	p.updateProxyNumMetric()
}

// GetProxyCount returns number of proxy clients.
func (p *proxyClientManager) GetProxyCount() int {
	return p.proxyClient.Len()
}

// mutex.Lock is required before calling this method.
func (p *proxyClientManager) updateProxyNumMetric() {
	metrics.RootCoordProxyCounter.WithLabelValues().Set(float64(p.proxyClient.Len()))
}

func (p *proxyClientManager) connect(session *sessionutil.Session) {
	pc, err := p.creator(context.Background(), session.Address, session.ServerID)
	if err != nil {
		log.Warn("failed to create proxy client", zap.String("address", session.Address), zap.Int64("serverID", session.ServerID), zap.Error(err))
		return
	}

	_, ok := p.proxyClient.GetOrInsert(session.GetServerID(), pc)
	if ok {
		pc.Close()
		return
	}
	log.Info("succeed to create proxy client", zap.String("address", session.Address), zap.Int64("serverID", session.ServerID))
	p.helper.afterConnect()
}

func (p *proxyClientManager) DelProxyClient(s *sessionutil.Session) {
	cli, ok := p.proxyClient.GetAndRemove(s.GetServerID())
	if ok {
		cli.Close()
	}

	p.updateProxyNumMetric()
	log.Info("remove proxy client", zap.String("proxy address", s.Address), zap.Int64("proxy id", s.ServerID))
}

func (p *proxyClientManager) InvalidateCollectionMetaCache(ctx context.Context, request *proxypb.InvalidateCollMetaCacheRequest, opts ...expireCacheOpt) error {
	c := defaultExpireCacheConfig()
	for _, opt := range opts {
		opt(&c)
	}
	c.apply(request)

	if p.proxyClient.Len() == 0 {
		log.Warn("proxy client is empty, InvalidateCollectionMetaCache will not send to any client")
		return nil
	}

	group := &errgroup.Group{}
	p.proxyClient.Range(func(key int64, value types.ProxyClient) bool {
		k, v := key, value
		group.Go(func() error {
			sta, err := v.InvalidateCollectionMetaCache(ctx, request)
			if err != nil {
				if errors.Is(err, merr.ErrNodeNotFound) {
					log.Warn("InvalidateCollectionMetaCache failed due to proxy service not found", zap.Error(err))
					return nil
				}
				return fmt.Errorf("InvalidateCollectionMetaCache failed, proxyID = %d, err = %s", k, err)
			}
			if sta.ErrorCode != commonpb.ErrorCode_Success {
				return fmt.Errorf("InvalidateCollectionMetaCache failed, proxyID = %d, err = %s", k, sta.Reason)
			}
			return nil
		})
		return true
	})
	return group.Wait()
}

// InvalidateCredentialCache TODO: too many codes similar to InvalidateCollectionMetaCache.
func (p *proxyClientManager) InvalidateCredentialCache(ctx context.Context, request *proxypb.InvalidateCredCacheRequest) error {
	if p.proxyClient.Len() == 0 {
		log.Warn("proxy client is empty, InvalidateCredentialCache will not send to any client")
		return nil
	}

	group := &errgroup.Group{}
	p.proxyClient.Range(func(key int64, value types.ProxyClient) bool {
		k, v := key, value
		group.Go(func() error {
			sta, err := v.InvalidateCredentialCache(ctx, request)
			if err != nil {
				return fmt.Errorf("InvalidateCredentialCache failed, proxyID = %d, err = %s", k, err)
			}
			if sta.ErrorCode != commonpb.ErrorCode_Success {
				return fmt.Errorf("InvalidateCredentialCache failed, proxyID = %d, err = %s", k, sta.Reason)
			}
			return nil
		})
		return true
	})

	return group.Wait()
}

// UpdateCredentialCache TODO: too many codes similar to InvalidateCollectionMetaCache.
func (p *proxyClientManager) UpdateCredentialCache(ctx context.Context, request *proxypb.UpdateCredCacheRequest) error {
	if p.proxyClient.Len() == 0 {
		log.Warn("proxy client is empty, UpdateCredentialCache will not send to any client")
		return nil
	}

	group := &errgroup.Group{}
	p.proxyClient.Range(func(key int64, value types.ProxyClient) bool {
		k, v := key, value
		group.Go(func() error {
			sta, err := v.UpdateCredentialCache(ctx, request)
			if err != nil {
				return fmt.Errorf("UpdateCredentialCache failed, proxyID = %d, err = %s", k, err)
			}
			if sta.ErrorCode != commonpb.ErrorCode_Success {
				return fmt.Errorf("UpdateCredentialCache failed, proxyID = %d, err = %s", k, sta.Reason)
			}
			return nil
		})
		return true
	})
	return group.Wait()
}

// RefreshPolicyInfoCache TODO: too many codes similar to InvalidateCollectionMetaCache.
func (p *proxyClientManager) RefreshPolicyInfoCache(ctx context.Context, req *proxypb.RefreshPolicyInfoCacheRequest) error {
	if p.proxyClient.Len() == 0 {
		log.Warn("proxy client is empty, RefreshPrivilegeInfoCache will not send to any client")
		return nil
	}

	group := &errgroup.Group{}
	p.proxyClient.Range(func(key int64, value types.ProxyClient) bool {
		k, v := key, value
		group.Go(func() error {
			status, err := v.RefreshPolicyInfoCache(ctx, req)
			if err != nil {
				return fmt.Errorf("RefreshPolicyInfoCache failed, proxyID = %d, err = %s", k, err)
			}
			if status.GetErrorCode() != commonpb.ErrorCode_Success {
				return merr.Error(status)
			}
			return nil
		})
		return true
	})
	return group.Wait()
}

// GetProxyMetrics sends requests to proxies to get metrics.
func (p *proxyClientManager) GetProxyMetrics(ctx context.Context) ([]*milvuspb.GetMetricsResponse, error) {
	if p.proxyClient.Len() == 0 {
		log.Warn("proxy client is empty, GetMetrics will not send to any client")
		return nil, nil
	}

	req, err := metricsinfo.ConstructRequestByMetricType(metricsinfo.SystemInfoMetrics)
	if err != nil {
		return nil, err
	}

	group := &errgroup.Group{}
	var metricRspsMu sync.Mutex
	metricRsps := make([]*milvuspb.GetMetricsResponse, 0)
	p.proxyClient.Range(func(key int64, value types.ProxyClient) bool {
		k, v := key, value
		group.Go(func() error {
			rsp, err := v.GetProxyMetrics(ctx, req)
			if err != nil {
				return fmt.Errorf("GetMetrics failed, proxyID = %d, err = %s", k, err)
			}
			if rsp.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
				return fmt.Errorf("GetMetrics failed, proxyID = %d, err = %s", k, rsp.GetStatus().GetReason())
			}
			metricRspsMu.Lock()
			metricRsps = append(metricRsps, rsp)
			metricRspsMu.Unlock()
			return nil
		})
		return true
	})
	err = group.Wait()
	if err != nil {
		return nil, err
	}
	return metricRsps, nil
}

// SetRates notifies Proxy to limit rates of requests.
func (p *proxyClientManager) SetRates(ctx context.Context, request *proxypb.SetRatesRequest) error {
	if p.proxyClient.Len() == 0 {
		log.Warn("proxy client is empty, SetRates will not send to any client")
		return nil
	}

	group := &errgroup.Group{}
	p.proxyClient.Range(func(key int64, value types.ProxyClient) bool {
		k, v := key, value
		group.Go(func() error {
			sta, err := v.SetRates(ctx, request)
			if err != nil {
				return fmt.Errorf("SetRates failed, proxyID = %d, err = %s", k, err)
			}
			if sta.GetErrorCode() != commonpb.ErrorCode_Success {
				return fmt.Errorf("SetRates failed, proxyID = %d, err = %s", k, sta.Reason)
			}
			return nil
		})
		return true
	})
	return group.Wait()
}
