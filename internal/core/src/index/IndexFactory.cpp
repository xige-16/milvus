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

#include "index/IndexFactory.h"
#include "index/VectorMemIndex.h"
#include "index/VectorDIskAnnIndex.h"
#include "index/Utils.h"
#include "index/Meta.h"

#include "exceptions/EasyAssert.h"

namespace milvus::Index {

IndexBasePtr
IndexFactory::CreateIndex(const BuildIndexInfo& build_index_info) {
    if (datatype_is_vector(build_index_info.field_type)) {
        return CreateVectorIndex(build_index_info);
    }

    return CreateScalarIndex(build_index_info);
}

IndexBasePtr
IndexFactory::CreateScalarIndex(const BuildIndexInfo& build_index_info) {
    auto data_type = build_index_info.field_type;
    auto index_type = build_index_info.index_type;

    switch (data_type) {
        // create scalar index
        case DataType::BOOL:
            return CreateScalarIndex<bool>(index_type);
        case DataType::INT8:
            return CreateScalarIndex<int8_t>(index_type);
        case DataType::INT16:
            return CreateScalarIndex<int16_t>(index_type);
        case DataType::INT32:
            return CreateScalarIndex<int32_t>(index_type);
        case DataType::INT64:
            return CreateScalarIndex<int64_t>(index_type);
        case DataType::FLOAT:
            return CreateScalarIndex<float>(index_type);
        case DataType::DOUBLE:
            return CreateScalarIndex<double>(index_type);

            // create string index
        case DataType::STRING:
        case DataType::VARCHAR:
            return CreateScalarIndex<std::string>(index_type);
        default:
            throw std::invalid_argument(std::string("invalid data type to build index: ") +
                                        std::to_string(int(data_type)));
    }
}

IndexBasePtr
IndexFactory::CreateVectorIndex(const BuildIndexInfo& build_index_info) {
    auto data_type = build_index_info.field_type;
    auto index_type = build_index_info.index_type;
    auto metric_type = build_index_info.metric_type;
    auto index_mode = build_index_info.index_mode;

    // create disk index
    if (is_in_disk_list(index_type)) {
        switch (data_type) {
            case DataType::VECTOR_FLOAT: {
                auto file_manager = std::make_shared<knowhere::DiskANNFileManagerImpl>(
                    GetFieldMetaFromBuildIndexInfo(build_index_info), GetIndexMetaFromBuildIndexInfo(build_index_info));
                return std::make_shared<VectorDiskAnnIndex<float>>(index_type, metric_type, index_mode, file_manager);
            }
            default:
                throw std::invalid_argument(std::string("invalid data type to build disk index: ") +
                                            std::to_string(int(data_type)));
        }
    }

    // create mem index
    return std::make_shared<VectorMemIndex>(index_type, metric_type, index_mode);
}

}  // namespace milvus::Index
