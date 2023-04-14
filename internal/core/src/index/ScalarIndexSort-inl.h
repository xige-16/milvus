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
#include <memory>
#include <utility>
#include <pb/schema.pb.h>
#include <vector>
#include <map>
#include <string>
#include "knowhere/common/Log.h"
#include "Meta.h"
#include "common/Utils.h"
#include "common/Slice.h"
#include "index/Utils.h"

namespace milvus::index {

template <typename T>
inline ScalarIndexSort<T>::ScalarIndexSort(storage::FileManagerImplPtr file_manager) : is_built_(false), data_() {
    if (file_manager != nullptr) {
        file_manager_ = std::dynamic_pointer_cast<storage::MemFileManagerImpl>(file_manager);
    }
}

template <typename T>
inline void
ScalarIndexSort<T>::Build(const Config& config) {
    if (is_built_)
        return;
    auto insert_files = GetValueFromConfig<std::vector<std::string>>(config, "insert_files");
    AssertInfo(insert_files.has_value(), "insert file paths is empty when build index");
    auto field_datas = file_manager_->CacheRawDataToMemory(insert_files.value());

    int64_t total_num_rows = 0;
    for (auto data : field_datas) {
        total_num_rows += data->get_num_rows();
    }
    if (total_num_rows == 0) {
        // todo: throw an exception
        throw std::invalid_argument("ScalarIndexSort cannot build null values!");
    }

    data_.reserve(total_num_rows);
    int64_t offset = 0;
    for (auto data : field_datas) {
        auto slice_num = data->get_num_rows();
        for (size_t i = 0; i < slice_num; ++i) {
            auto value = reinterpret_cast<const T*>(data->RawValue(i));
            data_.emplace_back(IndexStructure(*value, offset));
            offset++;
        }
    }

    std::sort(data_.begin(), data_.end());
    idx_to_offsets_.resize(total_num_rows);
    for (size_t i = 0; i < total_num_rows; ++i) {
        idx_to_offsets_[data_[i].idx_] = i;
    }
    is_built_ = true;
}

template <typename T>
inline void
ScalarIndexSort<T>::Build(size_t n, const T* values) {
    if (is_built_)
        return;
    if (n == 0) {
        // todo: throw an exception
        throw std::invalid_argument("ScalarIndexSort cannot build null values!");
    }
    data_.reserve(n);
    idx_to_offsets_.resize(n);
    T* p = const_cast<T*>(values);
    for (size_t i = 0; i < n; ++i) {
        data_.emplace_back(IndexStructure(*p++, i));
    }
    std::sort(data_.begin(), data_.end());
    for (size_t i = 0; i < data_.size(); ++i) {
        idx_to_offsets_[data_[i].idx_] = i;
    }
    is_built_ = true;
}

template <typename T>
inline BinarySet
ScalarIndexSort<T>::SerializeWithoutDisassemble(const Config& config) {
    AssertInfo(is_built_, "index has not been built");

    auto index_data_size = data_.size() * sizeof(IndexStructure<T>);
    std::shared_ptr<uint8_t[]> index_data(new uint8_t[index_data_size]);
    memcpy(index_data.get(), data_.data(), index_data_size);

    std::shared_ptr<uint8_t[]> index_length(new uint8_t[sizeof(size_t)]);
    auto index_size = data_.size();
    memcpy(index_length.get(), &index_size, sizeof(size_t));

    BinarySet res_set;
    res_set.Append("index_data", index_data, index_data_size);
    res_set.Append("index_length", index_length, sizeof(size_t));

    return res_set;
}

template <typename T>
inline BinarySet
ScalarIndexSort<T>::Serialize(const Config& config) {
    BinarySet res_set = SerializeWithoutDisassemble(config);
    milvus::Disassemble(res_set);

    return res_set;
}

template <typename T>
inline BinarySet
ScalarIndexSort<T>::Upload(const Config& config) {
    auto binary_set = SerializeWithoutDisassemble(config);
    auto sliced_binary_set = ShallowDisassemble(binary_set);
    file_manager_->AddFile(sliced_binary_set);

    auto remote_paths_to_size = file_manager_->GetRemotePathsToFileSize();
    BinarySet ret;
    for (auto& file : remote_paths_to_size) {
        auto abs_file_path = file.first;
        ret.Append(abs_file_path.substr(abs_file_path.find_last_of("/") + 1), nullptr, file.second);
    }

    return ret;
}

template <typename T>
inline void
ScalarIndexSort<T>::Load(const BinarySet& index_binary, const Config& config) {
    size_t index_size;
    milvus::Assemble(const_cast<BinarySet&>(index_binary));
    auto index_length = index_binary.GetByName("index_length");
    memcpy(&index_size, index_length->data.get(), (size_t)index_length->size);

    auto index_data = index_binary.GetByName("index_data");
    data_.resize(index_size);
    idx_to_offsets_.resize(index_size);
    memcpy(data_.data(), index_data->data.get(), (size_t)index_data->size);
    for (size_t i = 0; i < data_.size(); ++i) {
        idx_to_offsets_[data_[i].idx_] = i;
    }
    is_built_ = true;
}

template <typename T>
void
ScalarIndexSort<T>::AssembleAndLoadIndexDatas(const std::map<std::string, storage::FieldDataPtr>& index_datas) {
    size_t index_size;
    std::string index_length_key = "index_length";
    AssertInfo(index_datas.find(index_length_key) != index_datas.end(), "lost index length file");
    auto index_length_data = index_datas.at(index_length_key);
    memcpy(&index_size, index_length_data->Data(), (size_t)index_length_data->Size());

    std::string index_data_key = "index_data";
    bool index_data_sliced = false;
    data_.resize(index_size);
    if (index_datas.find(INDEX_FILE_SLICE_META) != index_datas.end()) {
        auto slice_meta = index_datas.at(INDEX_FILE_SLICE_META);
        Config meta_data = Config::parse(std::string(static_cast<const char*>(slice_meta->Data()), slice_meta->Size()));

        for (auto& item : meta_data[META]) {
            std::string prefix = item[NAME];
            if (prefix == index_data_key) {
                int slice_num = item[SLICE_NUM];
                auto total_len = static_cast<size_t>(item[TOTAL_LEN]);

                int64_t offset = 0;
                for (auto i = 0; i < slice_num; ++i) {
                    std::string file_name = GenSlicedFileName(prefix, i);
                    AssertInfo(index_datas.find(file_name) != index_datas.end(), "lost index slice data");
                    auto data = index_datas.at(file_name);
                    memcpy(reinterpret_cast<uint8_t*>(data_.data()) + offset, data->Data(), (size_t)data->Size());
                    offset += data->Size();
                }
                AssertInfo(total_len == offset, "index len is inconsistent after disassemble and assemble");
                index_data_sliced = true;
                break;
            }

            continue;
        }
    }

    if (!index_data_sliced) {
        AssertInfo(index_datas.find(index_data_key) != index_datas.end(), "lost index data file");
        auto data = index_datas.at(index_data_key);
        memcpy(data_.data(), data->Data(), (size_t)data->Size());
    }

    AssertInfo(data_.size() == index_size, "empty data when assemble");
    idx_to_offsets_.resize(index_size);
    for (size_t i = 0; i < data_.size(); ++i) {
        idx_to_offsets_[data_[i].idx_] = i;
    }
    is_built_ = true;
}

template <typename T>
inline void
ScalarIndexSort<T>::Load(const Config& config) {
    auto index_files = GetValueFromConfig<std::vector<std::string>>(config, "index_files");
    AssertInfo(index_files.has_value(), "index file paths is empty when load disk ann index");
    auto index_datas = file_manager_->LoadIndexToMemory(index_files.value());
    AssembleAndLoadIndexDatas(index_datas);
}

template <typename T>
inline const TargetBitmapPtr
ScalarIndexSort<T>::In(const size_t n, const T* values) {
    AssertInfo(is_built_, "index has not been built");
    TargetBitmapPtr bitset = std::make_unique<TargetBitmap>(data_.size());
    for (size_t i = 0; i < n; ++i) {
        auto lb = std::lower_bound(data_.begin(), data_.end(), IndexStructure<T>(*(values + i)));
        auto ub = std::upper_bound(data_.begin(), data_.end(), IndexStructure<T>(*(values + i)));
        for (; lb < ub; ++lb) {
            if (lb->a_ != *(values + i)) {
                std::cout << "error happens in ScalarIndexSort<T>::In, experted value is: " << *(values + i)
                          << ", but real value is: " << lb->a_;
            }
            bitset->set(lb->idx_);
        }
    }
    return bitset;
}

template <typename T>
inline const TargetBitmapPtr
ScalarIndexSort<T>::NotIn(const size_t n, const T* values) {
    AssertInfo(is_built_, "index has not been built");
    TargetBitmapPtr bitset = std::make_unique<TargetBitmap>(data_.size());
    bitset->set();
    for (size_t i = 0; i < n; ++i) {
        auto lb = std::lower_bound(data_.begin(), data_.end(), IndexStructure<T>(*(values + i)));
        auto ub = std::upper_bound(data_.begin(), data_.end(), IndexStructure<T>(*(values + i)));
        for (; lb < ub; ++lb) {
            if (lb->a_ != *(values + i)) {
                std::cout << "error happens in ScalarIndexSort<T>::NotIn, experted value is: " << *(values + i)
                          << ", but real value is: " << lb->a_;
            }
            bitset->reset(lb->idx_);
        }
    }
    return bitset;
}

template <typename T>
inline const TargetBitmapPtr
ScalarIndexSort<T>::Range(const T value, const OpType op) {
    AssertInfo(is_built_, "index has not been built");
    TargetBitmapPtr bitset = std::make_unique<TargetBitmap>(data_.size());
    auto lb = data_.begin();
    auto ub = data_.end();
    switch (op) {
        case OpType::LessThan:
            ub = std::lower_bound(data_.begin(), data_.end(), IndexStructure<T>(value));
            break;
        case OpType::LessEqual:
            ub = std::upper_bound(data_.begin(), data_.end(), IndexStructure<T>(value));
            break;
        case OpType::GreaterThan:
            lb = std::upper_bound(data_.begin(), data_.end(), IndexStructure<T>(value));
            break;
        case OpType::GreaterEqual:
            lb = std::lower_bound(data_.begin(), data_.end(), IndexStructure<T>(value));
            break;
        default:
            throw std::invalid_argument(std::string("Invalid OperatorType: ") + std::to_string((int)op) + "!");
    }
    for (; lb < ub; ++lb) {
        bitset->set(lb->idx_);
    }
    return bitset;
}

template <typename T>
inline const TargetBitmapPtr
ScalarIndexSort<T>::Range(T lower_bound_value, bool lb_inclusive, T upper_bound_value, bool ub_inclusive) {
    AssertInfo(is_built_, "index has not been built");
    TargetBitmapPtr bitset = std::make_unique<TargetBitmap>(data_.size());
    if (lower_bound_value > upper_bound_value ||
        (lower_bound_value == upper_bound_value && !(lb_inclusive && ub_inclusive))) {
        return bitset;
    }
    auto lb = data_.begin();
    auto ub = data_.end();
    if (lb_inclusive) {
        lb = std::lower_bound(data_.begin(), data_.end(), IndexStructure<T>(lower_bound_value));
    } else {
        lb = std::upper_bound(data_.begin(), data_.end(), IndexStructure<T>(lower_bound_value));
    }
    if (ub_inclusive) {
        ub = std::upper_bound(data_.begin(), data_.end(), IndexStructure<T>(upper_bound_value));
    } else {
        ub = std::lower_bound(data_.begin(), data_.end(), IndexStructure<T>(upper_bound_value));
    }
    for (; lb < ub; ++lb) {
        bitset->set(lb->idx_);
    }
    return bitset;
}

template <typename T>
inline T
ScalarIndexSort<T>::Reverse_Lookup(size_t idx) const {
    AssertInfo(idx < idx_to_offsets_.size(), "out of range of total count");
    AssertInfo(is_built_, "index has not been built");

    auto offset = idx_to_offsets_[idx];
    return data_[offset].a_;
}
}  // namespace milvus::index
