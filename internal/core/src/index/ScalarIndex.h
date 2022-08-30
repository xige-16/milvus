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

#include <map>
#include <memory>
#include <string>
#include <boost/dynamic_bitset.hpp>
#include "index/Index.h"
#include "common/Types.h"

namespace milvus::Index {

template <typename T>
class ScalarIndex : public IndexBase {
 public:
    virtual void
    Build(size_t n, const T* values) = 0;

    virtual const TargetBitmapPtr
    In(size_t n, const T* values) = 0;

    virtual const TargetBitmapPtr
    NotIn(size_t n, const T* values) = 0;

    virtual const TargetBitmapPtr
    Range(T value, OpType op) = 0;

    virtual const TargetBitmapPtr
    Range(T lower_bound_value, bool lb_inclusive, T upper_bound_value, bool ub_inclusive) = 0;

    virtual T
    Reverse_Lookup(size_t offset) const = 0;

    virtual const TargetBitmapPtr
    Query(const DatasetPtr& dataset);

    virtual int64_t
    Size() = 0;
};

template <typename T>
using ScalarIndexPtr = std::shared_ptr<ScalarIndex<T>>;

}  // namespace milvus::Index

#include "index/ScalarIndex-inl.h"
