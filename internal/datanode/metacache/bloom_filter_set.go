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

package metacache

import (
	"sync"

	"github.com/bits-and-blooms/bloom/v3"

	"github.com/milvus-io/milvus/internal/storage"
)

type BloomFilterSet struct {
	mut     sync.Mutex
	current *storage.PkStatistics
	history []*storage.PkStatistics
}

func NewBloomFilterSet(historyEntries ...*storage.PkStatistics) *BloomFilterSet {
	return &BloomFilterSet{
		history: historyEntries,
	}
}

func (bfs *BloomFilterSet) PkExists(pk storage.PrimaryKey) bool {
	bfs.mut.Lock()
	defer bfs.mut.Unlock()
	if bfs.current != nil && bfs.current.PkExist(pk) {
		return true
	}

	for _, bf := range bfs.history {
		if bf.PkExist(pk) {
			return true
		}
	}
	return false
}

func (bfs *BloomFilterSet) UpdatePKRange(ids storage.FieldData) error {
	bfs.mut.Lock()
	defer bfs.mut.Unlock()

	if bfs.current == nil {
		bfs.current = &storage.PkStatistics{
			PkFilter: bloom.NewWithEstimates(storage.BloomFilterSize, storage.MaxBloomFalsePositive),
		}
	}

	return bfs.current.UpdatePKRange(ids)
}

func (bfs *BloomFilterSet) Roll() {
	bfs.mut.Lock()
	defer bfs.mut.Unlock()

	if bfs.current != nil {
		bfs.history = append(bfs.history, bfs.current)
		bfs.current = nil
	}
}

func (bfs *BloomFilterSet) GetHistory() []*storage.PkStatistics {
	bfs.mut.Lock()
	defer bfs.mut.Unlock()

	return bfs.history
}
