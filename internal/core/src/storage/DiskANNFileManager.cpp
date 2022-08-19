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

#include "storage/DiskANNFileManager.h"

namespace milvus::storage {

DiskANNFileManagerImpl::DiskANNFileManagerImpl(int64_t collectionId,
                                               int64_t partiitionId,
                                               int64_t segmentId,
                                               const ChunkManager& chunk_manager){};

DiskANNFileManagerImpl::DiskANNFileManagerImpl(){};

std::string
DiskANNFileManagerImpl::GetRemoteObjectName(const std::string& localfile) {
    return std::to_string(index_build_id_) + "/" + std::to_string(0) + "/" + std::to_string(partition_id_) + "/" +
           std::to_string(segment_id_) + localfile;
}

}  // namespace milvus::storage