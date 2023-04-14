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
#include <boost/filesystem.hpp>

#include "common/Common.h"
#include "common/Slice.h"
#include "log/Log.h"
#include "config/ConfigKnowhere.h"
#include "storage/DiskFileManagerImpl.h"
#include "storage/LocalChunkManager.h"
#include "storage/MinioChunkManager.h"
#include "storage/Exception.h"
#include "storage/IndexData.h"
#include "storage/Util.h"
#include "storage/ThreadPool.h"

namespace milvus::storage {

DiskFileManagerImpl::DiskFileManagerImpl(const FieldDataMeta& field_mata,
                                         const IndexMeta& index_meta,
                                         const StorageConfig& storage_config)
    : FileManagerImpl(field_mata, index_meta) {
    rcm_ = std::make_unique<MinioChunkManager>(storage_config);
}

DiskFileManagerImpl::DiskFileManagerImpl(const FieldDataMeta& field_mata,
                                         const IndexMeta& index_meta,
                                         RemoteChunkManagerPtr remote_chunk_manager)
    : FileManagerImpl(field_mata, index_meta) {
    rcm_ = remote_chunk_manager;
}

DiskFileManagerImpl::~DiskFileManagerImpl() {
    auto& local_chunk_manager = LocalChunkManager::GetInstance();
    local_chunk_manager.RemoveDir(GetLocalIndexPathPrefixWithBuildID(index_meta_.build_id));
}

bool
DiskFileManagerImpl::LoadFile(const std::string& file) noexcept {
    return true;
}

std::string
DiskFileManagerImpl::GetRemoteIndexPath(const std::string& file_name, int64_t slice_num) const {
    auto remote_prefix = GetRemoteIndexObjectPrefix();
    return remote_prefix + "/" + file_name + "_" + std::to_string(slice_num);
}

bool
DiskFileManagerImpl::AddFile(const std::string& file) noexcept {
    auto& local_chunk_manager = LocalChunkManager::GetInstance();
    FILEMANAGER_TRY
    if (!local_chunk_manager.Exist(file)) {
        LOG_SEGCORE_ERROR_C << "local file: " << file << " does not exist ";
        return false;
    }

    // record local file path
    local_paths_.emplace_back(file);

    auto fileName = GetFileName(file);
    auto fileSize = local_chunk_manager.Size(file);

    std::vector<std::string> batch_remote_files;
    std::vector<int64_t> remote_file_sizes;
    std::vector<int64_t> local_file_offsets;

    int slice_num = 0;
    auto parallel_degree = uint64_t(DEFAULT_INDEX_MAX_MEMORY_LIMIT / INDEX_FILE_SLICE_SIZE);
    for (int64_t offset = 0; offset < fileSize; slice_num++) {
        if (batch_remote_files.size() >= parallel_degree) {
            AddBatchIndexFiles(file, local_file_offsets, batch_remote_files, remote_file_sizes);
            batch_remote_files.clear();
            remote_file_sizes.clear();
            local_file_offsets.clear();
        }

        auto batch_size = std::min(INDEX_FILE_SLICE_SIZE, int64_t(fileSize) - offset);
        batch_remote_files.emplace_back(GetRemoteIndexPath(fileName, slice_num));
        remote_file_sizes.emplace_back(batch_size);
        local_file_offsets.emplace_back(offset);

        offset += batch_size;
    }

    if (batch_remote_files.size() > 0) {
        AddBatchIndexFiles(file, local_file_offsets, batch_remote_files, remote_file_sizes);
    }
    FILEMANAGER_CATCH
    FILEMANAGER_END

    return true;
}  // namespace knowhere

void
DiskFileManagerImpl::AddBatchIndexFiles(const std::string& local_file_name,
                                        const std::vector<int64_t>& local_file_offsets,
                                        const std::vector<std::string>& remote_files,
                                        const std::vector<int64_t>& remote_file_sizes) {
    auto& local_chunk_manager = LocalChunkManager::GetInstance();
    auto& pool = ThreadPool::GetInstance();

    auto LoadIndexFromDisk = [&](const std::string& file, const int64_t offset,
                                 const int64_t data_size) -> std::shared_ptr<uint8_t[]> {
        auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[data_size]);
        local_chunk_manager.Read(file, offset, buf.get(), data_size);
        return buf;
    };

    std::vector<std::future<std::shared_ptr<uint8_t[]>>> futures;
    AssertInfo(local_file_offsets.size() == remote_files.size(), "inconsistent size of offset slices with file slices");
    AssertInfo(remote_files.size() == remote_file_sizes.size(), "inconsistent size of file slices with size slices");

    for (int64_t i = 0; i < remote_files.size(); ++i) {
        futures.push_back(pool.Submit(LoadIndexFromDisk, local_file_name, local_file_offsets[i], remote_file_sizes[i]));
    }

    std::vector<std::shared_ptr<uint8_t[]>> index_datas;  // hold index data util upload index file done
    std::vector<const uint8_t*> data_slices;
    for (auto& future : futures) {
        auto res = future.get();
        index_datas.emplace_back(res);
        data_slices.emplace_back(res.get());
    }

    auto res = PutIndexData(rcm_.get(), data_slices, remote_file_sizes, remote_files, field_meta_, index_meta_);
    for (auto iter = res.begin(); iter != res.end(); ++iter) {
        remote_paths_to_size_[iter->first] = iter->second;
    }
}

void
DiskFileManagerImpl::CacheIndexToDisk(std::vector<std::string> remote_files) {
    auto& local_chunk_manager = LocalChunkManager::GetInstance();

    std::map<std::string, std::vector<int>> index_slices;
    for (auto& file_path : remote_files) {
        auto pos = file_path.find_last_of("_");
        index_slices[file_path.substr(0, pos)].emplace_back(std::stoi(file_path.substr(pos + 1)));
    }

    for (auto& slices : index_slices) {
        std::sort(slices.second.begin(), slices.second.end());
    }

    auto EstimateParalleDegree = [&](const std::string& file) -> uint64_t {
        auto fileSize = rcm_->Size(file);
        return uint64_t(DEFAULT_INDEX_MAX_MEMORY_LIMIT / fileSize);
    };

    for (auto& slices : index_slices) {
        auto prefix = slices.first;
        auto local_index_file_name = GetLocalIndexObjectPrefix() + prefix.substr(prefix.find_last_of("/") + 1);
        local_chunk_manager.CreateFile(local_index_file_name);
        int64_t offset = 0;
        std::vector<std::string> batch_remote_files;
        uint64_t max_parallel_degree = INT_MAX;
        for (auto iter = slices.second.begin(); iter != slices.second.end(); iter++) {
            if (batch_remote_files.size() >= max_parallel_degree) {
                auto next_offset = CacheBatchIndexFilesToDisk(batch_remote_files, local_index_file_name, offset);
                offset = next_offset;
                batch_remote_files.clear();
            }
            auto origin_file = prefix + "_" + std::to_string(*iter);
            if (batch_remote_files.size() == 0) {
                // Use first file size as average size to estimate
                max_parallel_degree = EstimateParalleDegree(origin_file);
            }
            batch_remote_files.push_back(origin_file);
        }
        if (batch_remote_files.size() > 0) {
            auto next_offset = CacheBatchIndexFilesToDisk(batch_remote_files, local_index_file_name, offset);
            offset = next_offset;
            batch_remote_files.clear();
        }
        local_paths_.emplace_back(local_index_file_name);
    }
}

uint64_t
DiskFileManagerImpl::CacheBatchIndexFilesToDisk(const std::vector<std::string>& remote_files,
                                                const std::string& local_file_name,
                                                uint64_t local_file_init_offfset) {
    auto& local_chunk_manager = LocalChunkManager::GetInstance();
    auto index_datas = GetObjectData(rcm_.get(), remote_files);
    int batch_size = remote_files.size();
    AssertInfo(index_datas.size() == batch_size, "inconsistent file num and index data num!");

    uint64_t offset = local_file_init_offfset;
    for (int i = 0; i < batch_size; ++i) {
        auto index_data = index_datas[i];
        auto index_size = index_data->Size();
        local_chunk_manager.Write(local_file_name, offset,
                                  reinterpret_cast<uint8_t*>(const_cast<void*>(index_data->Data())), index_size);
        offset += index_size;
    }

    return offset;
}

std::string
DiskFileManagerImpl::CacheRawDataToDisk(std::vector<std::string> remote_files) {
    std::sort(remote_files.begin(), remote_files.end(), [](const std::string& a, const std::string& b) {
        return std::stol(a.substr(a.find_last_of("/") + 1)) < std::stol(b.substr(b.find_last_of("/") + 1));
    });

    auto field_datas = GetObjectData(rcm_.get(), remote_files);
    int batch_size = remote_files.size();
    AssertInfo(field_datas.size() == batch_size, "inconsistent file num and raw data num!");

    uint32_t num_rows = 0;
    uint32_t dim = 0;
    for (int i = 0; i < batch_size; ++i) {
        auto field_data = field_datas[i];
        auto num = field_data->get_num_rows();
        num_rows += uint32_t(num);
        AssertInfo(dim == 0 || dim == field_data->get_dim(), "inconsistent dim value in multi binlogs!");
        dim = field_data->get_dim();
    }

    auto segment_id = GetFieldDataMeta().segment_id;
    auto field_id = GetFieldDataMeta().field_id;
    auto local_data_path = storage::GenFieldRawDataPathPrefix(segment_id, field_id) + "raw_data";
    auto& local_chunk_manager = LocalChunkManager::GetInstance();
    local_chunk_manager.CreateFile(local_data_path);

    int64_t write_offset = 0;
    local_chunk_manager.Write(local_data_path, write_offset, &num_rows, sizeof(num_rows));
    write_offset += sizeof(num_rows);
    local_chunk_manager.Write(local_data_path, write_offset, &dim, sizeof(dim));
    write_offset += sizeof(dim);

    for (int i = 0; i < batch_size; ++i) {
        auto field_data = field_datas[i];
        auto data_size = field_data->get_num_rows() * dim * sizeof(float);
        local_chunk_manager.Write(local_data_path, write_offset, const_cast<void*>(field_data->Data()), data_size);
        write_offset += data_size;
    }

    return local_data_path;
}

std::string
DiskFileManagerImpl::GetFileName(const std::string& localfile) {
    boost::filesystem::path localPath(localfile);
    return localPath.filename().string();
}

std::string
DiskFileManagerImpl::GetLocalIndexObjectPrefix() {
    return GenLocalIndexPathPrefix(index_meta_.build_id, index_meta_.index_version);
}

std::string
DiskFileManagerImpl::GetLocalRawDataObjectPrefix() {
    return GenFieldRawDataPathPrefix(field_meta_.segment_id, field_meta_.field_id);
}

bool
DiskFileManagerImpl::RemoveFile(const std::string& file) noexcept {
    // TODO: implement this interface
    return false;
}

std::optional<bool>
DiskFileManagerImpl::IsExisted(const std::string& file) noexcept {
    bool isExist = false;
    auto& local_chunk_manager = LocalChunkManager::GetInstance();
    try {
        isExist = local_chunk_manager.Exist(file);
    } catch (LocalChunkManagerException& e) {
        // LOG_SEGCORE_DEBUG_ << "LocalChunkManagerException:"
        //                   << e.what();
        return std::nullopt;
    } catch (std::exception& e) {
        // LOG_SEGCORE_DEBUG_ << "Exception:" << e.what();
        return std::nullopt;
    }
    return isExist;
}

}  // namespace milvus::storage
