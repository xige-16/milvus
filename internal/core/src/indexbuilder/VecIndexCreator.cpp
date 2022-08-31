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

#include <map>

#include "exceptions/EasyAssert.h"
#include "indexbuilder/VecIndexCreator.h"
#include "index/Utils.h"
#include "index/IndexFactory.h"
#include "knowhere/index/VecIndex.h"

namespace milvus::indexbuilder {

VecIndexCreator::VecIndexCreator(DataType data_type,
                                 const char* serialized_type_params,
                                 const char* serialized_index_params)
    : data_type_(data_type) {
    milvus::ParseFromString(type_params_, std::string(serialized_type_params));
    milvus::ParseFromString(index_params_, std::string(serialized_index_params));

    for (auto i = 0; i < type_params_.params_size(); ++i) {
        const auto& param = type_params_.params(i);
        config_[param.key()] = param.value();
    }

    for (auto i = 0; i < index_params_.params_size(); ++i) {
        const auto& param = index_params_.params(i);
        config_[param.key()] = param.value();
    }

    index_ = Index::IndexFactory::GetInstance().CreateIndex(get_build_index_info());
    AssertInfo(index_ != nullptr, "[VecIndexCreator]Index is null after create index");
}

int64_t
VecIndexCreator::dim() {
    return Index::GetDimFromConfig(config_);
}

void
VecIndexCreator::Build(const milvus::DatasetPtr& dataset) {
    index_->BuildWithDataset(dataset, config_);
}

milvus::BinarySet
VecIndexCreator::Serialize() {
    return index_->Serialize(config_);
}

void
VecIndexCreator::Load(const milvus::BinarySet& binary_set) {
    index_->Load(binary_set, config_);
}

std::unique_ptr<SearchResult>
VecIndexCreator::Query(const milvus::DatasetPtr& dataset, const SearchInfo& search_info, const BitsetView& bitset) {
    auto vector_index = dynamic_cast<Index::VectorIndex*>(index_.get());
    return vector_index->Query(dataset, search_info, bitset);
}

Index::BuildIndexInfo
VecIndexCreator::get_build_index_info() const {
    Index::BuildIndexInfo index_info;
    // set collection id
    auto collection_id = Index::GetValueFromConfig<std::string>(config_, Index::COLLECTION_ID);
    if (collection_id.has_value()) {
        index_info.collection_id = std::stol(collection_id.value());
    }
    // set partition id
    auto partition_id = Index::GetValueFromConfig<std::string>(config_, Index::PARTITION_ID);
    if (partition_id.has_value()) {
        index_info.partition_id = std::stol(partition_id.value());
    }
    // set segment id
    auto segment_id = Index::GetValueFromConfig<std::string>(config_, Index::SEGMENT_ID);
    if (segment_id.has_value()) {
        index_info.segment_id = std::stol(segment_id.value());
    }
    // set field id
    auto field_id = Index::GetValueFromConfig<std::string>(config_, Index::FIELD_ID);
    if (field_id.has_value()) {
        index_info.field_id = std::stol(field_id.value());
    }
    // set index version
    auto index_version = Index::GetValueFromConfig<std::string>(config_, Index::INDEX_VERSION);
    if (index_version.has_value()) {
        index_info.index_version = std::stol(index_version.value());
    }
    // set index id
    auto index_id = Index::GetValueFromConfig<std::string>(config_, Index::INDEX_ID);
    if (index_id.has_value()) {
        index_info.index_id = std::stol(index_id.value());
    }
    // set index build id
    auto index_build_id = Index::GetValueFromConfig<std::string>(config_, Index::INDEX_BUILD_ID);
    if (index_build_id.has_value()) {
        index_info.index_build_id = std::stol(index_build_id.value());
    }

    index_info.field_type = data_type_;

    // set index type
    index_info.index_type = Index::GetIndexTypeFromConfig(config_);
    // set metric type
    index_info.metric_type = Index::GetMetricTypeFromConfig(config_);
    // set index mode
    index_info.index_mode = Index::GetIndexModeFromConfig(config_);

    return index_info;
}

}  // namespace milvus::indexbuilder
