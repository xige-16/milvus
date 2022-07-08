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

#pragma once

#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <tbb/concurrent_unordered_map.h>
#include <boost/align/aligned_allocator.hpp>
#include <boost/container/vector.hpp>
#include <boost/dynamic_bitset.hpp>
#include <NamedType/named_type.hpp>
#include <variant>

#include "knowhere/common/MetricType.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "pb/plan.pb.h"

namespace milvus {

using idx_t = int64_t;
using offset_t = int32_t;
using date_t = int32_t;
using distance_t = float;

enum class DataType {
    NONE = 0,
    BOOL = 1,
    INT8 = 2,
    INT16 = 3,
    INT32 = 4,
    INT64 = 5,

    FLOAT = 10,
    DOUBLE = 11,

    STRING = 20,
    VARCHAR = 21,

    VECTOR_BINARY = 100,
    VECTOR_FLOAT = 101,
};

using Timestamp = uint64_t;  // TODO: use TiKV-like timestamp
constexpr auto MAX_TIMESTAMP = std::numeric_limits<Timestamp>::max();
constexpr auto MAX_ROW_COUNT = std::numeric_limits<idx_t>::max();

using OpType = proto::plan::OpType;
using ArithOpType = proto::plan::ArithOpType;
using ScalarArray = proto::schema::ScalarField;
using DataArray = proto::schema::FieldData;
using VectorArray = proto::schema::VectorField;
using IdArray = proto::schema::IDs;
using MetricType = faiss::MetricType;
using InsertData = proto::segcore::InsertRecord;
using PkType = std::variant<std::monostate, int64_t, std::string>;
//using Pk2OffsetType = tbb::concurrent_unordered_multimap<PkType, int64_t, std::hash<PkType>>;
using Pk2OffsetType = std::unordered_multimap<PkType, int64_t, std::hash<PkType>>;

MetricType
GetMetricType(const std::string& type);

std::string
MetricTypeToName(MetricType metric_type);

bool
IsPrimaryKeyDataType(DataType data_type);

// NOTE: dependent type
// used at meta-template programming
template <class...>
constexpr std::true_type always_true{};

template <class...>
constexpr std::false_type always_false{};

template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

namespace impl {
// hide identifier name to make auto-completion happy
struct FieldIdTag;
struct FieldNameTag;
struct FieldOffsetTag;
struct SegOffsetTag;
};  // namespace impl

using FieldId = fluent::NamedType<int64_t, impl::FieldIdTag, fluent::Comparable, fluent::Hashable>;
using FieldName = fluent::NamedType<std::string, impl::FieldNameTag, fluent::Comparable, fluent::Hashable>;
// using FieldOffset = fluent::NamedType<int64_t, impl::FieldOffsetTag, fluent::Comparable, fluent::Hashable>;
using SegOffset = fluent::NamedType<int64_t, impl::SegOffsetTag, fluent::Arithmetic>;

using BitsetType = boost::dynamic_bitset<>;
using BitsetTypePtr = std::shared_ptr<boost::dynamic_bitset<>>;
using BitsetTypeOpt = std::optional<BitsetType>;

template <typename Type>
using FixedVector = boost::container::vector<Type>;

const FieldId RowFieldID = FieldId(0);
const FieldId TimestampFieldID = FieldId(1);
}  // namespace milvus
