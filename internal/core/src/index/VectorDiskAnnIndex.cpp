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

#include "index/VectorDIskAnnIndex.h"
#include "index/Meta.h"
#include "index/Utils.h"

#include "storage/LocalChunkManager.h"
#include "config/ConfigKnowhere.h"
#include "storage/Util.h"
#include "common/Utils.h"

namespace milvus::Index {

template <typename T>
VectorDiskAnnIndex<T>::VectorDiskAnnIndex(const IndexType& index_type,
                                          const MetricType& metric_type,
                                          const IndexMode& index_mode,
                                          std::shared_ptr<knowhere::DiskANNFileManagerImpl> file_manager)
    : VectorIndex(index_type, index_mode, metric_type), file_manager_(file_manager) {
    auto& local_chunk_manager = storage::LocalChunkManager::GetInstance();
    local_chunk_manager.CreateDir(file_manager_->GetLocalObjectPrefix());
    index_ =
        std::make_unique<knowhere::IndexDiskANN<T>>(file_manager->GetLocalObjectPrefix(), metric_type, file_manager);
}

template <typename T>
void
VectorDiskAnnIndex<T>::Load(const BinarySet& binary_set /* not used */, const Config& config) {
    auto prepare_config = parse_prepare_config(const_cast<Config&>(config));
    knowhere::Config cfg;
    knowhere::DiskANNPrepareConfig::Set(cfg, prepare_config);

    auto index_files = GetValueFromConfig<std::vector<std::string>>(config, "index_files");
    AssertInfo(index_files.has_value(), "index file paths is empty when load disk ann index data");
    file_manager_->CacheIndexToDisk(index_files.value());
    index_->Prepare(cfg);
}

template <typename T>
void
VectorDiskAnnIndex<T>::BuildWithDataset(const DatasetPtr& dataset, const Config& config) {
    auto build_config = parse_build_config(const_cast<Config&>(config));
    auto segment_id = GetValueFromConfig<std::string>(config, Index::SEGMENT_ID);
    AssertInfo(segment_id.has_value(), "segment id not exist");
    auto field_id = GetValueFromConfig<std::string>(config, Index::FIELD_ID);
    AssertInfo(field_id.has_value(), "field id not exist");
    auto local_data_path =
        storage::GenRawDataPathPrefix(std::stol(segment_id.value()), std::stol(field_id.value())) + "raw_data";
    build_config.data_path = local_data_path;

    auto& local_chunk_manager = storage::LocalChunkManager::GetInstance();
    auto num = uint32_t(milvus::GetDatasetRows(dataset));
    auto dim = uint32_t(milvus::GetDatasetDim(dataset));
    auto data_size = num * dim * sizeof(float);
    auto raw_data = const_cast<void*>(milvus::GetDatasetTensor(dataset));

    auto total_size = sizeof(num) + sizeof(dim) + data_size;
    std::vector<uint8_t> disk_raw_data(total_size);
    int offset = 0;
    memcpy(disk_raw_data.data() + offset, &num, sizeof(num));
    offset += sizeof(num);
    memcpy(disk_raw_data.data() + offset, &dim, sizeof(dim));
    offset += sizeof(dim);
    memcpy(disk_raw_data.data() + offset, raw_data, data_size);

    // for now, disk ann only support float vector
    local_chunk_manager.Write(local_data_path, disk_raw_data.data(), total_size);

    knowhere::Config cfg;
    knowhere::DiskANNBuildConfig::Set(cfg, build_config);

    index_->BuildAll(nullptr, cfg);

    local_chunk_manager.RemoveDir(storage::GetLocalRawDataPathPrefixWithBuildID(std::stol(segment_id.value())));
}

template <typename T>
std::unique_ptr<SearchResult>
VectorDiskAnnIndex<T>::Query(const DatasetPtr dataset, const SearchInfo& search_info, const BitsetView& bitset) {
    auto query_config = parse_query_config(const_cast<Config&>(search_info.search_params_));
    AssertInfo(GetMetricType() == search_info.metric_type_,
               "Metric type of field index isn't the same with search info");
    auto num_queries = milvus::GetDatasetRows(dataset);
    auto topk = search_info.topk_;

    query_config.k = topk;

    // TODO:: ugly
    if (query_config.search_list_size <= topk) {
        query_config.search_list_size = topk + 1;
    }
    if (query_config.search_list_size >= 65535 * 3) {
        query_config.search_list_size = 65535 * 3 - 1;
    }

    AssertInfo(query_config.search_list_size > topk && query_config.search_list_size < 65535 * 3,
               "search_list should be greater then topk and less than 65535 * 3");
    knowhere::Config cfg;
    knowhere::DiskANNQueryConfig::Set(cfg, query_config);

    auto final = index_->Query(dataset, cfg, bitset);
    auto ids = milvus::GetDatasetIDs(final);
    float* distances = (float*)milvus::GetDatasetDistance(final);

    auto round_decimal = search_info.round_decimal_;
    auto total_num = num_queries * topk;

    if (round_decimal != -1) {
        const float multiplier = pow(10.0, round_decimal);
        for (int i = 0; i < total_num; i++) {
            distances[i] = round(distances[i] * multiplier) / multiplier;
        }
    }
    auto result = std::make_unique<SearchResult>();
    result->seg_offsets_.resize(total_num);
    result->distances_.resize(total_num);
    result->total_nq_ = num_queries;
    result->unity_topK_ = topk;

    std::copy_n(ids, total_num, result->seg_offsets_.data());
    std::copy_n(distances, total_num, result->distances_.data());

    return result;
}

template <typename T>
knowhere::DiskANNBuildConfig
VectorDiskAnnIndex<T>::parse_build_config(Config& config) {
    // parse config from string type to valid value
    parse_config(config);

    // set disk ann build config
    knowhere::DiskANNBuildConfig build_config;
    // set data path
    //    auto data_path = GetValueFromConfig<std::string>(config, DISK_ANN_RAW_DATA_PATH);
    //    AssertInfo(data_path.has_value(), "param " + std::string(DISK_ANN_RAW_DATA_PATH) + "is empty");
    //    build_config.data_path = data_path.value();

    // set max degree
    auto max_degree = GetValueFromConfig<uint32_t>(config, DISK_ANN_MAX_DEGREE);
    AssertInfo(max_degree.has_value(), "param " + std::string(DISK_ANN_MAX_DEGREE) + "is empty");
    build_config.max_degree = max_degree.value();

    // set build list
    auto search_list_size = GetValueFromConfig<uint32_t>(config, DISK_ANN_BUILD_LIST);
    AssertInfo(search_list_size.has_value(), "param " + std::string(DISK_ANN_BUILD_LIST) + "is empty");
    build_config.search_list_size = search_list_size.value();

    // set search dram budget
    auto search_dram_budget_gb = GetValueFromConfig<float>(config, DISK_ANN_SEARCH_DRAM_BUDGET);
    AssertInfo(search_dram_budget_gb.has_value(), "param " + std::string(DISK_ANN_SEARCH_DRAM_BUDGET) + "is empty");
    build_config.search_dram_budget_gb = search_dram_budget_gb.value();

    // set build dram budget
    auto build_dram_budget_gb = GetValueFromConfig<float>(config, DISK_ANN_BUILD_DRAM_BUDGET);
    AssertInfo(build_dram_budget_gb.has_value(), "param " + std::string(DISK_ANN_BUILD_DRAM_BUDGET) + "is empty");
    build_config.build_dram_budget_gb = build_dram_budget_gb.value();

    // set num build thread
    auto num_threads = GetValueFromConfig<uint32_t>(config, DISK_ANN_BUILD_THREAD_NUM);
    AssertInfo(num_threads.has_value(), "param " + std::string(DISK_ANN_BUILD_THREAD_NUM) + "is empty");
    build_config.num_threads = num_threads.value();

    // set pq bytes
    auto pq_disk_bytes = GetValueFromConfig<uint32_t>(config, DISK_ANN_PQ_BYTES);
    AssertInfo(pq_disk_bytes.has_value(), "param " + std::string(DISK_ANN_PQ_BYTES) + "is empty");
    build_config.disk_pq_dims = pq_disk_bytes.value();

    return build_config;
}

template <typename T>
knowhere::DiskANNPrepareConfig
VectorDiskAnnIndex<T>::parse_prepare_config(Config& config) {
    // parse config from string type to valid value
    parse_config(config);

    knowhere::DiskANNPrepareConfig prepare_config;
    prepare_config.warm_up = false;
    prepare_config.use_bfs_cache = false;

    // set prepare thread num
    auto num_threads = GetValueFromConfig<uint32_t>(config, DISK_ANN_PREPARE_THREAD_NUM);
    AssertInfo(num_threads.has_value(), "param " + std::string(DISK_ANN_PREPARE_THREAD_NUM) + "is empty");
    prepare_config.num_threads = num_threads.value();

    // set prepare cached node
    auto num_nodes_to_cache = GetValueFromConfig<uint32_t>(config, DISK_ANN_PREPARE_NODES_CACHED);
    AssertInfo(num_nodes_to_cache.has_value(), "param " + std::string(DISK_ANN_PREPARE_NODES_CACHED) + "is empty");
    prepare_config.num_nodes_to_cache = num_nodes_to_cache.value();

    return prepare_config;
}

template <typename T>
knowhere::DiskANNQueryConfig
VectorDiskAnnIndex<T>::parse_query_config(Config& config) {
    // parse config from string type to valid value
    parse_config(config);

    knowhere::DiskANNQueryConfig query_config;

    // set topk
    auto topK = GetValueFromConfig<uint32_t>(config, knowhere::meta::TOPK);
    if (topK.has_value()) {
        query_config.k = topK.value();
    }

    // set search list
    auto search_list_size = GetValueFromConfig<uint32_t>(config, DISK_ANN_QUERY_LIST);
    AssertInfo(search_list_size.has_value(), "param " + std::string(DISK_ANN_QUERY_LIST) + "is empty");
    query_config.search_list_size = search_list_size.value();

    // set beamwidth
    auto beamwidth = GetValueFromConfig<uint32_t>(config, DISK_ANN_QUERY_BEAMWIDTH);
    AssertInfo(beamwidth.has_value(), "param " + std::string(DISK_ANN_QUERY_BEAMWIDTH) + "is empty");
    query_config.beamwidth = beamwidth.value();

    return query_config;
}

template <typename T>
void
VectorDiskAnnIndex<T>::parse_config(Config& config) {
    auto stoi_closure = [](const std::string& s) -> uint32_t { return std::stoi(s); };
    auto stof_closure = [](const std::string& s) -> float { return std::stof(s); };

    /***************************** meta *******************************/
    CheckParameter<int>(config, knowhere::meta::SLICE_SIZE, stoi_closure,
                        std::optional{config::KnowhereGetIndexSliceSize()});
    CheckParameter<int>(config, knowhere::meta::DIM, stoi_closure, std::nullopt);
    CheckParameter<int>(config, knowhere::meta::TOPK, stoi_closure, std::nullopt);

    /************************** DiskAnn build Params ************************/
    CheckParameter<int>(config, DISK_ANN_MAX_DEGREE, stoi_closure, std::optional{48});
    CheckParameter<int>(config, DISK_ANN_BUILD_LIST, stoi_closure, std::optional{128});
    CheckParameter<float>(config, DISK_ANN_SEARCH_DRAM_BUDGET, stof_closure, std::optional{0.03});
    CheckParameter<float>(config, DISK_ANN_BUILD_DRAM_BUDGET, stof_closure, std::optional{32});
    CheckParameter<int>(config, DISK_ANN_BUILD_THREAD_NUM, stoi_closure, std::optional{8});
    CheckParameter<int>(config, DISK_ANN_PQ_BYTES, stoi_closure, std::optional{0});

    /************************** DiskAnn prepare Params ************************/
    CheckParameter<int>(config, DISK_ANN_PREPARE_THREAD_NUM, stoi_closure, std::optional{8});
    //    CheckParameter<int>(config, DISK_ANN_PREPARE_NODES_CACHED, stoi_closure, std::optional{10000});
    CheckParameter<int>(config, DISK_ANN_PREPARE_NODES_CACHED, stoi_closure, std::optional{1});

    /************************** DiskAnn query Params ************************/
    CheckParameter<int>(config, DISK_ANN_QUERY_LIST, stoi_closure, std::nullopt);
    CheckParameter<int>(config, DISK_ANN_QUERY_BEAMWIDTH, stoi_closure, std::optional{16});
}

template class VectorDiskAnnIndex<float>;

}  // namespace milvus::Index