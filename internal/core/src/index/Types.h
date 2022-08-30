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
#include <knowhere/common/BinarySet.h>
#include <knowhere/common/Dataset.h>
#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include <boost/dynamic_bitset.hpp>

namespace milvus::Index {

using BinarySet = knowhere::BinarySet;
using DatasetPtr = knowhere::DatasetPtr;

using Config = nlohmann::json;
using TargetBitmap = boost::dynamic_bitset<>;
using TargetBitmapPtr = std::unique_ptr<TargetBitmap>;

}  // namespace milvus::Index