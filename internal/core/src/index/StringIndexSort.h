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

#include <memory>
#include <vector>
#include <string>

#include "common/Utils.h"
#include "index/ScalarIndexSort.h"
#include "index/StringIndex.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

namespace milvus::Index {
// TODO: should inherit from StringIndex?
class StringIndexSort : public ScalarIndexSort<std::string> {
 public:
    void
    BuildWithDataset(const DatasetPtr& dataset, const Config& config = {}) {
        auto size = knowhere::GetDatasetRows(dataset);
        auto data = knowhere::GetDatasetTensor(dataset);
        proto::schema::StringArray arr;
        arr.ParseFromArray(data, size);

        {
            // TODO: optimize here. avoid memory copy.
            std::vector<std::string> vecs{arr.data().begin(), arr.data().end()};
            Build(arr.data().size(), vecs.data());
        }

        {
            // TODO: test this way.
            // auto strs = (const std::string*)arr.data().data();
            // Build(arr.data().size(), strs);
        }
    }

    const TargetBitmapPtr
    Query(const DatasetPtr& dataset) {
        auto op = dataset->Get<OpType>(OPERATOR_TYPE);
        if (op == OpType::PrefixMatch) {
            auto prefix = dataset->Get<std::string>(PREFIX_VALUE);
            return PrefixMatch(prefix);
        }
        return ScalarIndex<std::string>::Query(dataset);
    }

    const TargetBitmapPtr
    PrefixMatch(std::string prefix) {
        auto data = GetData();
        TargetBitmapPtr bitset = std::make_unique<TargetBitmap>(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            if (milvus::PrefixMatch(data[i].a_, prefix)) {
                bitset->set(data[i].idx_);
            }
        }
        return bitset;
    }
};
using StringIndexSortPtr = std::unique_ptr<StringIndexSort>;

inline StringIndexSortPtr
CreateStringIndexSort() {
    return std::make_unique<StringIndexSort>();
}
}  // namespace milvus::Index
