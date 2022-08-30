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

#include "indexbuilder/ScalarIndexCreator.h"
#include "index/IndexFactory.h"
#include "index/IndexInfo.h"
#include "index/Meta.h"

#include <string>

namespace milvus::indexbuilder {

ScalarIndexCreator::ScalarIndexCreator(DataType dtype, const char* type_params, const char* index_params)
    : dtype_(dtype) {
    // TODO: move parse-related logic to a common interface.
    milvus::ParseFromString(type_params_, std::string(type_params));
    milvus::ParseFromString(index_params_, std::string(index_params));

    for (auto i = 0; i < type_params_.params_size(); ++i) {
        const auto& param = type_params_.params(i);
        config_[param.key()] = param.value();
    }

    for (auto i = 0; i < index_params_.params_size(); ++i) {
        const auto& param = index_params_.params(i);
        config_[param.key()] = param.value();
    }

    milvus::Index::BuildIndexInfo index_info;
    index_info.field_type = dtype_;
    index_info.index_params[milvus::Index::INDEX_TYPE] = index_type();
    index_ = Index::IndexFactory::GetInstance().CreateIndex(index_info);
}

ScalarIndexCreator::ScalarIndexCreator(DataType data_type,
                                       const std::map<std::string, std::string> type_params,
                                       const std::map<std::string, std::string> index_params)
    : dtype_(data_type) {
    for (auto& param : type_params) {
        config_[param.first] = param.second;
    }

    for (auto& param : index_params) {
        config_[param.first] = param.second;
    }

    milvus::Index::BuildIndexInfo index_info;
    index_info.field_type = dtype_;
    index_info.index_params[milvus::Index::INDEX_TYPE] = index_type();
    index_ = Index::IndexFactory::GetInstance().CreateIndex(index_info);
}

void
ScalarIndexCreator::Build(const knowhere::DatasetPtr& dataset) {
    index_->BuildWithDataset(dataset);
}

knowhere::BinarySet
ScalarIndexCreator::Serialize() {
    return index_->Serialize(config_);
}

void
ScalarIndexCreator::Load(const knowhere::BinarySet& binary_set) {
    index_->Load(binary_set);
}

std::string
ScalarIndexCreator::index_type() {
    // TODO
    return "sort";
}

}  // namespace milvus::indexbuilder
