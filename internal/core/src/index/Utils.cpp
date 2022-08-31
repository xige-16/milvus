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

#include <algorithm>
#include <tuple>
#include <vector>
#include <functional>

#include "index/Utils.h"
#include "index/Meta.h"
#include "pb/index_cgo_msg.pb.h"
#include <google/protobuf/text_format.h>
#include "exceptions/EasyAssert.h"

namespace milvus::Index {

size_t
get_file_size(int fd) {
    struct stat s;
    fstat(fd, &s);
    return s.st_size;
}

std::vector<IndexType>
NM_List() {
    static std::vector<IndexType> ret{
        knowhere::IndexEnum::INDEX_FAISS_IVFFLAT,
    };
    return ret;
}

std::vector<IndexType>
BIN_List() {
    static std::vector<IndexType> ret{
        knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP,
        knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT,
    };
    return ret;
}

std::vector<IndexType>
DISK_LIST() {
    static std::vector<IndexType> ret{
        knowhere::IndexEnum::INDEX_DISKANN,
    };
    return ret;
}

std::vector<std::tuple<IndexType, MetricType>>
unsupported_index_combinations() {
    static std::vector<std::tuple<IndexType, MetricType>> ret{
        std::make_tuple(knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, knowhere::metric::L2),
    };
    return ret;
}

template <typename T>
bool
is_in_list(const T& t, std::function<std::vector<T>()> list_func) {
    auto l = list_func();
    return std::find(l.begin(), l.end(), t) != l.end();
}

bool
is_in_bin_list(const IndexType& index_type) {
    return is_in_list<IndexType>(index_type, BIN_List);
}

bool
is_in_nm_list(const IndexType& index_type) {
    return is_in_list<IndexType>(index_type, NM_List);
}

bool
is_in_disk_list(const IndexType& index_type) {
    return is_in_list<IndexType>(index_type, DISK_LIST);
}

bool
is_unsupported(const IndexType& index_type, const MetricType& metric_type) {
    return is_in_list<std::tuple<IndexType, MetricType>>(std::make_tuple(index_type, metric_type),
                                                         unsupported_index_combinations);
}

bool
CheckKeyInConfig(const Config& cfg, const std::string& key) {
    return cfg.contains(key);
}

IndexMode
GetIndexMode(const Config& config) {
    static std::map<std::string, knowhere::IndexMode> mode_map = {
        {"CPU", knowhere::IndexMode::MODE_CPU},
        {"GPU", knowhere::IndexMode::MODE_GPU},
    };
    auto mode = GetValueFromConfig<std::string>(config, INDEX_MODE);
    return mode.has_value() ? mode_map[mode.value()] : knowhere::IndexMode::MODE_CPU;
}

void
ParseFromString(google::protobuf::Message& params, const std::string& str) {
    auto ok = google::protobuf::TextFormat::ParseFromString(str, &params);
    AssertInfo(ok, "failed to parse params from string");
}

int64_t
GetDimFromConfig(const Config& config) {
    auto dimension = GetValueFromConfig<std::string>(config, "dim");
    AssertInfo(dimension.has_value(), "dimension not exist in config");
    return (std::stoi(dimension.value()));
}

std::string
GetMetricTypeFromConfig(const Config& config) {
    auto metric_type = GetValueFromConfig<std::string>(config, "metric_type");
    AssertInfo(metric_type.has_value(), "metric_type not exist in config");
    return metric_type.value();
}

std::string
GetIndexTypeFromConfig(const Config& config) {
    auto index_type = GetValueFromConfig<std::string>(config, "index_type");
    AssertInfo(index_type.has_value(), "index_type not exist in config");
    return index_type.value();
}

IndexMode
GetIndexModeFromConfig(const Config& config) {
    static std::map<std::string, knowhere::IndexMode> mode_map = {
        {"CPU", knowhere::IndexMode::MODE_CPU},
        {"GPU", knowhere::IndexMode::MODE_GPU},
    };
    auto mode = GetValueFromConfig<std::string>(config, INDEX_MODE);
    return mode.has_value() ? mode_map[mode.value()] : knowhere::IndexMode::MODE_CPU;
}

storage::IndexMeta
GetIndexMetaFromBuildIndexInfo(const BuildIndexInfo& build_index_info) {
    storage::IndexMeta index_meta{build_index_info.segment_id, build_index_info.field_id,
                                  build_index_info.index_build_id, build_index_info.index_version};

    return std::move(index_meta);
}

storage::FieldDataMeta
GetFieldMetaFromBuildIndexInfo(const BuildIndexInfo& build_index_info) {
    storage::FieldDataMeta meta{build_index_info.collection_id, build_index_info.partition_id,
                                build_index_info.segment_id, build_index_info.field_id};
    return std::move(meta);
}

}  // namespace milvus::Index