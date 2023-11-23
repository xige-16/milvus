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
#include <unordered_map>

#include "common/Types.h"
#include "log/Log.h"
#include "mmap/Column.h"

namespace milvus {

using Metrics = std::
    variant<int8_t, int16_t, int32_t, int64_t, float, double, std::string_view>;

template <typename T>
using MetricsDataType =
    std::conditional_t<std::is_same_v<T, std::string>, std::string_view, T>;

struct FieldChunkMetrics {
    Metrics min_;
    Metrics max_;
    bool hasValue_;

    FieldChunkMetrics() : hasValue_(false){};
};

class SkipIndex {
 public:
    SkipIndex() {
        fieldChunkMetrics_ = std::unordered_map<
            FieldId,
            std::unordered_map<int64_t, FieldChunkMetrics>>();
    }

    template <typename T>
    bool
    CanSkipUnaryRange(FieldId field_id,
                      int64_t chunk_id,
                      OpType op_type,
                      const T& val) const {
        auto& field_chunk_metrics = GetFieldChunkMetrics(field_id, chunk_id);
        if (MinMaxUnaryFilter<T>(field_chunk_metrics, op_type, val)) {
            return true;
        }
        //further more filters for skip, like ngram filter, bf and so on
        return false;
    }

    template <typename T>
    bool
    CanSkipBinaryRange(FieldId field_id,
                       int64_t chunk_id,
                       const T& lower_val,
                       const T& upper_val,
                       bool lower_inclusive,
                       bool upper_inclusive) const {
        auto& field_chunk_metrics = GetFieldChunkMetrics(field_id, chunk_id);
        if (MinMaxBinaryFilter<T>(field_chunk_metrics,
                                  lower_val,
                                  upper_val,
                                  lower_inclusive,
                                  upper_inclusive)) {
            return true;
        }
        return false;
    }

    void
    LoadPrimitive(milvus::FieldId field_id,
                  int64_t chunk_id,
                  milvus::DataType data_type,
                  const void* chunk_data,
                  int64_t count);

    void
    LoadString(milvus::FieldId field_id,
               int64_t chunk_id,
               const milvus::VariableColumn<std::string>& var_column);

 private:
    const FieldChunkMetrics&
    GetFieldChunkMetrics(FieldId field_id, int chunk_id) const;

    template <typename T>
    struct IsAllowedType {
        static constexpr bool isAllowedType =
            std::is_integral<T>::value || std::is_floating_point<T>::value ||
            std::is_same<T, std::string>::value ||
            std::is_same<T, std::string_view>::value;
        static constexpr bool isDisabledType =
            std::is_same<T, milvus::Json>::value ||
            std::is_same<T, bool>::value;
        static constexpr bool value = isAllowedType && !isDisabledType;
    };

    template <typename T>
    std::pair<MetricsDataType<T>, MetricsDataType<T>>
    GetMinMax(const FieldChunkMetrics& field_chunk_metrics) const {
        MetricsDataType<T> lower_bound;
        MetricsDataType<T> upper_bound;
        try {
            lower_bound =
                std::get<MetricsDataType<T>>(field_chunk_metrics.min_);
            upper_bound =
                std::get<MetricsDataType<T>>(field_chunk_metrics.max_);
        } catch (const std::bad_variant_access&) {
            return {};
        }
        return {lower_bound, upper_bound};
    }

    template <typename T>
    std::enable_if_t<SkipIndex::IsAllowedType<T>::value, bool>
    MinMaxUnaryFilter(const FieldChunkMetrics& field_chunk_metrics,
                      OpType op_type,
                      const T& val) const {
        if (!field_chunk_metrics.hasValue_) {
            return false;
        }
        std::pair<MetricsDataType<T>, MetricsDataType<T>> minMax =
            GetMinMax<T>(field_chunk_metrics);
        if (minMax.first == MetricsDataType<T>() ||
            minMax.second == MetricsDataType<T>()) {
            return false;
        }
        return RangeShouldSkip<T>(val, minMax.first, minMax.second, op_type);
    }

    template <typename T>
    std::enable_if_t<!SkipIndex::IsAllowedType<T>::value, bool>
    MinMaxUnaryFilter(const FieldChunkMetrics& field_chunk_metrics,
                      OpType op_type,
                      const T& val) const {
        return false;
    }

    template <typename T>
    std::enable_if_t<SkipIndex::IsAllowedType<T>::value, bool>
    MinMaxBinaryFilter(const FieldChunkMetrics& field_chunk_metrics,
                       const T& lower_val,
                       const T& upper_val,
                       bool lower_inclusive,
                       bool upper_inclusive) const {
        if (!field_chunk_metrics.hasValue_) {
            return false;
        }
        std::pair<MetricsDataType<T>, MetricsDataType<T>> minMax =
            GetMinMax<T>(field_chunk_metrics);
        if (minMax.first == MetricsDataType<T>() ||
            minMax.second == MetricsDataType<T>()) {
            return false;
        }
        bool should_skip = false;
        MetricsDataType<T> lower_bound = minMax.first;
        MetricsDataType<T> upper_bound = minMax.second;
        if (lower_inclusive && upper_inclusive) {
            should_skip =
                (lower_val > upper_bound) || (upper_val < lower_bound);
        } else if (lower_inclusive && !upper_inclusive) {
            should_skip =
                (lower_val > upper_bound) || (upper_val <= lower_bound);
        } else if (!lower_inclusive && upper_inclusive) {
            should_skip =
                (lower_val >= upper_bound) || (upper_val < lower_bound);
        } else {
            should_skip =
                (lower_val >= upper_bound) || (upper_val <= lower_bound);
        }
        return should_skip;
    }

    template <typename T>
    std::enable_if_t<!SkipIndex::IsAllowedType<T>::value, bool>
    MinMaxBinaryFilter(const FieldChunkMetrics& field_chunk_metrics,
                       const T& lower_val,
                       const T& upper_val,
                       bool lower_inclusive,
                       bool upper_inclusive) const {
        return false;
    }

    template <typename T>
    bool
    RangeShouldSkip(const T& value,
                    const MetricsDataType<T> lower_bound,
                    const MetricsDataType<T> upper_bound,
                    OpType op_type) const {
        bool should_skip = false;
        switch (op_type) {
            case OpType::Equal: {
                should_skip = value > upper_bound || value < lower_bound;
                break;
            }
            case OpType::LessThan: {
                should_skip = value <= lower_bound;
                break;
            }
            case OpType::LessEqual: {
                should_skip = value < lower_bound;
                break;
            }
            case OpType::GreaterThan: {
                should_skip = value >= upper_bound;
                break;
            }
            case OpType::GreaterEqual: {
                should_skip = value > upper_bound;
                break;
            }
            default: {
                should_skip = false;
            }
        }
        return should_skip;
    }

    template <typename T>
    std::pair<T, T>
    ProcessFieldMetrics(const T* data, int64_t count) {
        //double check to avoid crush
        if (data == nullptr || count == 0) {
            return {T(), T()};
        }
        T minValue = data[0];
        T maxValue = data[0];
        for (size_t i = 0; i < count; i++) {
            T value = data[i];
            if (value < minValue) {
                minValue = value;
            }
            if (value > maxValue) {
                maxValue = value;
            }
        }
        return {minValue, maxValue};
    }

 private:
    std::unordered_map<FieldId, std::unordered_map<int64_t, FieldChunkMetrics>>
        fieldChunkMetrics_;
};
}  // namespace milvus
