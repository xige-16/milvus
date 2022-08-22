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

#include <iostream>
#include <memory>

#include "arrow/api.h"
#include "storage/Types.h"
#include "storage/PayloadStream.h"

namespace milvus::storage {

using DataType = milvus::DataType;

class FieldData {
 public:
    explicit FieldData(const Payload& payload);

    explicit FieldData(std::shared_ptr<arrow::Array> raw_data, DataType data_type);

    explicit FieldData(const uint8_t* data, int length);

    ~FieldData() = default;

    DataType
    get_data_type() const {
        return data_type_;
    }

    bool
    get_bool_payload(int idx) const;

    void
    get_one_string_payload(int idx, char** cstr, int* str_size) const;

    // get the bytes stream of the arrow array data
    std::unique_ptr<Payload>
    get_payload() const;

    int
    get_payload_length() const {
        return array_->length();
    }

    int
    get_data_size() const;

    //    void
    //    append_data(uint8_t* values, int length);

    //  // read arrow data randomly
    //  const Payload&
    //  operator[](int64_t index) const;

 private:
    std::shared_ptr<arrow::Array> array_;
    DataType data_type_;
};

}  // namespace milvus::storage
