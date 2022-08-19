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

#pragma once

#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "storage/ChunkManager.h"
#include "knowhere/common/FileManager.h"

using knowhere::FileManager;

namespace milvus::storage {

class DiskANNFileManagerImpl : public FileManager {
 public:
    explicit DiskANNFileManagerImpl(int64_t collectionId,
                                    int64_t partiitionId,
                                    int64_t segmentId,
                                    const ChunkManager& chunk_manager);

    explicit DiskANNFileManagerImpl();

    ~DiskANNFileManagerImpl() = default;

    bool
    LoadFile(const std::string& filename) noexcept {
        return true;
    }

    bool
    AddFile(const std::string& filename) noexcept {
        return true;
    }

    std::optional<bool>
    IsExisted(const std::string& filename) noexcept {
        return true;
    }

    bool
    RemoveFile(const std::string& filename) noexcept {
        return true;
    }

 public:
    virtual std::string
    GetName() const {
        return "DiskANNFileManagerImpl";
    }
    std::string
    GetRemoteObjectName(const std::string& localfile);

    std::string
    GetLocalFilePrefix() const {
        // TODO ::
        return "/tmp/" + std::to_string(index_build_id_);
    }

 private:
    int64_t collection_id_;
    int64_t partition_id_;
    int64_t segment_id_;
    int64_t field_id_;
    int64_t index_build_id_;
    mutable std::shared_mutex mutex_;
    // this map record (remote_object, local_file) pair
    std::map<std::string, std::string> file_map_;
    //  LocalChunkManagerSPtr local_chunk_manager_;
    //  RemoteChunkManagerSPtr remote_chunk_manager_;
};

}  // namespace milvus::storage