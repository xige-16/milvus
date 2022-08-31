// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "common/CDataType.h"
#include "index/IndexFactory.h"
#include "segcore/load_index_c.h"
#include "common/FieldMeta.h"
#include "common/Utils.h"
#include "pb/index_cgo_msg.pb.h"
#include "index/Meta.h"

CStatus
NewLoadIndexInfo(CLoadIndexInfo* c_load_index_info) {
    try {
        auto load_index_info = std::make_unique<milvus::Index::LoadIndexInfo>();
        *c_load_index_info = load_index_info.release();
        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

void
DeleteLoadIndexInfo(CLoadIndexInfo c_load_index_info) {
    auto info = (milvus::Index::LoadIndexInfo*)c_load_index_info;
    delete info;
}

CStatus
AppendIndexParam(CLoadIndexInfo c_load_index_info, const char* c_index_key, const char* c_index_value) {
    try {
        auto load_index_info = (milvus::Index::LoadIndexInfo*)c_load_index_info;
        std::string index_key(c_index_key);
        std::string index_value(c_index_value);
        load_index_info->index_params[index_key] = index_value;

        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

CStatus
AppendFieldInfo(CLoadIndexInfo c_load_index_info,
                int64_t collection_id,
                int64_t partition_id,
                int64_t segment_id,
                int64_t field_id,
                enum CDataType field_type) {
    try {
        auto load_index_info = (milvus::Index::LoadIndexInfo*)c_load_index_info;
        load_index_info->collection_id = collection_id;
        load_index_info->partition_id = partition_id;
        load_index_info->segment_id = segment_id;
        load_index_info->field_id = field_id;
        load_index_info->field_type = milvus::DataType(field_type);

        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

CStatus
appendVecIndex(CLoadIndexInfo c_load_index_info, CBinarySet c_binary_set) {
    try {
        auto load_index_info = (milvus::Index::LoadIndexInfo*)c_load_index_info;
        auto binary_set = (knowhere::BinarySet*)c_binary_set;
        auto& index_params = load_index_info->index_params;

        // get index type
        AssertInfo(index_params.find("index_type") != index_params.end(), "index type is empty");
        auto index_type = index_params.at("index_type");

        // get metric type
        AssertInfo(index_params.find("metric_type") != index_params.end(), "metric type is empty");
        auto metric_type = index_params.at("metric_type");

        // TODO :: get index build id from index file path
        milvus::Index::BuildIndexInfo build_index_info{load_index_info->collection_id,
                                                       load_index_info->partition_id,
                                                       load_index_info->segment_id,
                                                       load_index_info->field_id,
                                                       load_index_info->field_type,
                                                       load_index_info->index_version,
                                                       load_index_info->index_id,
                                                       load_index_info->index_build_id,
                                                       index_type,
                                                       metric_type,
                                                       milvus::IndexMode::MODE_CPU,
                                                       load_index_info->index_params};

        milvus::Config config;
        config["index_files"] = load_index_info->index_files;

        load_index_info->index = milvus::Index::IndexFactory::GetInstance().CreateVectorIndex(build_index_info);
        load_index_info->index->Load(*binary_set, config);
        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

CStatus
appendScalarIndex(CLoadIndexInfo c_load_index_info, CBinarySet c_binary_set) {
    try {
        auto load_index_info = (milvus::Index::LoadIndexInfo*)c_load_index_info;
        auto field_type = load_index_info->field_type;
        auto binary_set = (knowhere::BinarySet*)c_binary_set;
        auto& index_params = load_index_info->index_params;
        bool find_index_type = index_params.count("index_type") > 0 ? true : false;
        AssertInfo(find_index_type == true, "Can't find index type in index_params");

        milvus::Index::BuildIndexInfo build_index_info;
        build_index_info.index_params = index_params;
        build_index_info.field_type = milvus::DataType(field_type);
        load_index_info->index = milvus::Index::IndexFactory::GetInstance().CreateScalarIndex(build_index_info);
        load_index_info->index->Load(*binary_set);
        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

CStatus
AppendIndex(CLoadIndexInfo c_load_index_info, CBinarySet c_binary_set) {
    auto load_index_info = (milvus::Index::LoadIndexInfo*)c_load_index_info;
    auto field_type = load_index_info->field_type;
    if (milvus::datatype_is_vector(field_type)) {
        return appendVecIndex(c_load_index_info, c_binary_set);
    }
    return appendScalarIndex(c_load_index_info, c_binary_set);
}

CStatus
AppendIndexFilePath(CLoadIndexInfo c_load_index_info, const char* c_file_path) {
    try {
        auto load_index_info = (milvus::Index::LoadIndexInfo*)c_load_index_info;
        std::string index_file_path(c_file_path);
        load_index_info->index_files.emplace_back(index_file_path);

        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}

CStatus
AppendIndexInfo(
    CLoadIndexInfo c_load_index_info, int64_t index_id, int64_t build_id, int64_t version, const char* c_index_params) {
    try {
        auto load_index_info = (milvus::Index::LoadIndexInfo*)c_load_index_info;
        load_index_info->index_id = index_id;
        load_index_info->index_build_id = build_id;
        load_index_info->index_version = version;
        milvus::proto::indexcgo::IndexParams index_params;
        milvus::ParseFromString(index_params, c_index_params);

        for (auto i = 0; i < index_params.params().size(); i++) {
            auto& param = index_params.params(i);
            load_index_info->index_params[param.key()] = param.value();
        }

        auto status = CStatus();
        status.error_code = Success;
        status.error_msg = "";
        return status;
    } catch (std::exception& e) {
        auto status = CStatus();
        status.error_code = UnexpectedError;
        status.error_msg = strdup(e.what());
        return status;
    }
}
