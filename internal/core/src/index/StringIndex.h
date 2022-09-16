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

#include "index/ScalarIndex.h"
#include <string>
#include <memory>
#include <vector>
#include "index/Meta.h"
#include <pb/schema.pb.h>

namespace milvus::index {

class StringIndex : public ScalarIndex<std::string> {
 public:
    //    void
    //    Build(size_t n, const void* values, const Config& config = {}) override {
    //        proto::schema::StringArray arr;
    //        arr.ParseFromArray(values, n);
    //
    //        // TODO: optimize here. avoid memory copy.
    //        std::vector<std::string> vecs{arr.data().begin(), arr.data().end()};
    //        Build(arr.data().size(), vecs.data());
    //    }

    const TargetBitmapPtr
    Query(const DatasetPtr& dataset) override {
        auto op = dataset->Get<OpType>(OPERATOR_TYPE);
        if (op == OpType::PrefixMatch) {
            auto prefix = dataset->Get<std::string>(PREFIX_VALUE);
            return PrefixMatch(prefix);
        }
        return ScalarIndex<std::string>::Query(dataset);
    }

    virtual const TargetBitmapPtr
    PrefixMatch(std::string prefix) = 0;
};
using StringIndexPtr = std::unique_ptr<StringIndex>;
}  // namespace milvus::index
