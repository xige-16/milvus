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
#include "index/utils.h"
#include "indexbuilder/helper.h"
#include "index/IndexFactory.h"
#include "knowhere/index/VecIndex.h"
#include "config/ConfigKnowhere.h"
#include "pb/index_cgo_msg.pb.h"

namespace milvus::indexbuilder {

VecIndexCreator::VecIndexCreator(DataType data_type, const char* serialized_type_params, const char* serialized_index_params): data_type_(data_type) {
    Helper::ParseFromString(type_params_, std::string(serialized_type_params));
    Helper::ParseFromString(index_params_, std::string(serialized_index_params));

    for (auto i = 0; i < type_params_.params_size(); ++i) {
        const auto& param = type_params_.params(i);
        config_[param.key()] = param.value();
    }

    for (auto i = 0; i < index_params_.params_size(); ++i) {
        const auto& param = index_params_.params(i);
        config_[param.key()] = param.value();
    }

    auto build_index_info = get_build_index_info();
    index_ = Index::IndexFactory::GetInstance().CreateIndex(*build_index_info.get());
    AssertInfo(index_ != nullptr, "[VecIndexCreator]Index is null after create index");
}

template <typename T>
std::optional<T>
VecIndexCreator::get_config_by_name(const std::string& name) const {
    if (config_.contains(name)) {
        return knowhere::GetValueFromConfig<T>(config_, name);
    }
    return std::nullopt;
}

int64_t
VecIndexCreator::dim() {
    auto dimension = get_config_by_name<int64_t>(knowhere::meta::DIM);
    AssertInfo(dimension.has_value(), "[VecIndexCreator]Dimension doesn't have value");
    return (dimension.value());
}

void
VecIndexCreator::Build(const knowhere::DatasetPtr& dataset) {
    index_->BuildWithDataset(dataset, config_);
}

knowhere::BinarySet
VecIndexCreator::Serialize() {
    return index_->Serialize(config_);
}

void
VecIndexCreator::Load(const knowhere::BinarySet& binary_set) {
    index_->Load(binary_set, config_);
}

std::unique_ptr<SearchResult>
VecIndexCreator::Query(const knowhere::DatasetPtr& dataset, const SearchInfo& search_info, const BitsetView& bitset) {
    auto vector_index = dynamic_cast<Index::VectorIndex*>(index_.get());
    return vector_index->Query(dataset, search_info, bitset);
}

std::string
VecIndexCreator::get_index_type() const {
    // return index_->index_type();
    // knowhere bug here
    // the index_type of all ivf-based index will change to ivf flat after loaded
    auto type = get_config_by_name<std::string>("index_type");
    return type.has_value() ? type.value() : std::string(knowhere::IndexEnum::INDEX_FAISS_IVFPQ);
}

std::string
VecIndexCreator::get_metric_type() const {
    auto type = get_config_by_name<std::string>(knowhere::meta::METRIC_TYPE);
    if (type.has_value()) {
        return type.value();
    } else {
        auto index_type = get_index_type();
        if (Index::is_in_bin_list(index_type)) {
            return std::string(knowhere::metric::JACCARD);
        } else {
            return std::string(knowhere::metric::L2);
        }
    }
}

std::string
VecIndexCreator::get_index_mode() const {
    auto mode = get_config_by_name<std::string>("index_mode");
    return mode.has_value() ? mode.value() : "CPU";
}

std::unique_ptr<Index::BuildIndexInfo>
VecIndexCreator::get_build_index_info() const {
    auto index_info = std::make_unique<Index::BuildIndexInfo>();
    index_info->field_type = data_type_;
    index_info->index_params[Index::INDEX_TYPE] = get_index_type();
    index_info->index_params[Index::INDEX_MODE] = get_index_mode();
    index_info->index_params[Index::METRIC_TYPE] = get_metric_type();

    // ugly, but compatible with previous in-memory index logic
    if (config_.contains(Index::COLLECTION_ID)) {
        index_info->collection_id = get_config_by_name<int64_t>(Index::COLLECTION_ID).value();
    }

    if (config_.contains(Index::PARTITION_ID)) {
        index_info->partition_id = get_config_by_name<int64_t>(Index::PARTITION_ID).value();
    }

    if (config_.contains(Index::SEGMENT_ID)) {
        index_info->segment_id = get_config_by_name<int64_t>(Index::SEGMENT_ID).value();
    }

    if (config_.contains(Index::FIELD_ID)) {
        index_info->field_id = get_config_by_name<int64_t>(Index::FIELD_ID).value();
    }

    if (config_.contains(Index::INDEX_BUILD_ID)) {
        index_info->index_build_id = get_config_by_name<int64_t>(Index::INDEX_BUILD_ID).value();
    }

    if (config_.contains(Index::INDEX_ID)) {
        index_info->index_id = get_config_by_name<int64_t>(Index::INDEX_ID).value();
    }

    if (config_.contains(Index::INDEX_VERSION)) {
        index_info->index_version = get_config_by_name<int64_t>(Index::INDEX_VERSION).value();
    }
}



}  // namespace milvus::indexbuilder
