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

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "indexbuilder/IndexCreatorBase.h"
#include "index/VectorIndex.h"
#include "index/IndexInfo.h"
#include "pb/index_cgo_msg.pb.h"

namespace milvus::indexbuilder {

// TODO: better to distinguish binary vec & float vec.
class VecIndexCreator : public IndexCreatorBase {
 public:
    explicit VecIndexCreator(DataType dtype, const char* serialized_type_params, const char* serialized_index_params);

    void
    Build(const knowhere::DatasetPtr& dataset) override;

    knowhere::BinarySet
    Serialize() override;

    void
    Load(const knowhere::BinarySet& binary_set) override;

    int64_t
    dim();

    std::unique_ptr<SearchResult>
    Query(const knowhere::DatasetPtr& dataset, const SearchInfo& search_info, const BitsetView& bitset);

 private:
  std::string
  get_index_type() const;

  std::string
  get_metric_type() const;

  std::string
  get_index_mode() const;

  std::unique_ptr<Index::BuildIndexInfo>
  get_build_index_info() const;

    template <typename T>
    std::optional<T>
    get_config_by_name(const std::string& name) const;

 private:
    milvus::Index::IndexBasePtr index_ = nullptr;
  proto::indexcgo::TypeParams type_params_;
  proto::indexcgo::IndexParams index_params_;
    knowhere::Config config_;
    DataType data_type_;
};

}  // namespace milvus::indexbuilder
