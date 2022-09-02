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

#include <string>
#include <mutex>
#include <shared_mutex>

#include "common/type_c.h"
#include "config/ConfigChunkManager.h"
#include "index/Index.h"
#include "index/ScalarIndex.h"
#include "index/VectorIndex.h"
#include "index/IndexInfo.h"
#include "storage/LocalChunkManager.h"
#include "storage/MinioChunkManager.h"
#include "storage/Types.h"
//#include "knowhere/common/FileManager.h"
#include "storage/FileManager.h"

namespace milvus::Index {

class IndexFactory {
 public:
    IndexFactory() = default;
    IndexFactory(const IndexFactory&) = delete;
    IndexFactory
    operator=(const IndexFactory&) = delete;

 public:
    static IndexFactory&
    GetInstance() {
        // thread-safe enough after c++ 11
        static IndexFactory instance;

        return instance;
    }

    IndexBasePtr
    CreateIndex(const CreateIndexInfo& create_index_info, storage::FileManagerImplPtr file_manager);

    IndexBasePtr
    CreateVectorIndex(const CreateIndexInfo& create_index_info, storage::FileManagerImplPtr file_manager);

    IndexBasePtr
    CreateScalarIndex(const CreateIndexInfo& create_index_info);

    //    IndexBasePtr
    //    CreateIndex(DataType dtype, const IndexType& index_type, const IndexMode& index_mode = IndexMode::MODE_CPU);
 private:
    template <typename T>
    ScalarIndexPtr<T>
    CreateScalarIndex(const IndexType& index_type);
};

}  // namespace milvus::Index

#include "index/IndexFactory-inl.h"
