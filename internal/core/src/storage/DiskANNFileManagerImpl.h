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

//#include "FileManager.h"
#include "storage/IndexData.h"
#include "knowhere/common/FileManager.h"

namespace knowhere {

class DiskANNFileManagerImpl : public knowhere::FileManager {
 public:
    explicit DiskANNFileManagerImpl(const milvus::storage::FieldDataMeta& field_mata,
                                    const milvus::storage::IndexMeta& index_meta);

    virtual ~DiskANNFileManagerImpl();

    virtual bool
    LoadFile(const std::string& filename) noexcept;

    virtual bool
    AddFile(const std::string& filename) noexcept;

    virtual std::optional<bool>
    IsExisted(const std::string& filename) noexcept;

    virtual bool
    RemoveFile(const std::string& filename) noexcept;

 public:
    virtual std::string
    GetName() const {
        return "DiskANNFileManagerImpl";
    }

    std::string
    GetRemoteObjectPrefix();

    std::string
    GetLocalObjectPrefix();

    void
    SetLocalPaths(std::vector<std::string> paths) {
        local_paths_ = paths;
    }

    std::vector<std::string>
    GetRemotePaths() const {
        return remote_paths_;
    }

 private:
    int64_t
    GetIndexBuildId() {
        return index_meta_.build_id;
    }

    std::string
    GetFileName(const std::string& localfile);

 private:
    // collection meta
    milvus::storage::FieldDataMeta field_meta_;

    // index meta
    milvus::storage::IndexMeta index_meta_;

    // local file path (abs path)
    std::vector<std::string> local_paths_;

    // remote file path
    std::vector<std::string> remote_paths_;
};

}  // namespace knowhere
