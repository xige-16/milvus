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

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <index/ScalarIndex.h>

#include "DeletedRecord.h"
#include "FieldIndexing.h"
#include "common/Schema.h"
#include "common/Span.h"
#include "common/SystemProperty.h"
#include "common/Types.h"
#include "common/LoadInfo.h"
#include "common/BitsetView.h"
#include "common/QueryResult.h"
#include "common/QueryInfo.h"
#include "query/Plan.h"
#include "query/PlanNode.h"
#include "pb/schema.pb.h"
#include "pb/segcore.pb.h"
#include "index/IndexInfo.h"
#include "index/SkipIndex.h"
#include "mmap/Column.h"

namespace milvus::segcore {

// common interface of SegmentSealed and SegmentGrowing used by C API
class SegmentInterface {
 public:
    virtual ~SegmentInterface() = default;

    virtual void
    FillPrimaryKeys(const query::Plan* plan, SearchResult& results) const = 0;

    virtual void
    FillTargetEntry(const query::Plan* plan, SearchResult& results) const = 0;

    virtual bool
    Contain(const PkType& pk) const = 0;

    virtual std::unique_ptr<SearchResult>
    Search(const query::Plan* Plan,
           const query::PlaceholderGroup* placeholder_group) const = 0;

    virtual std::unique_ptr<proto::segcore::RetrieveResults>
    Retrieve(const query::RetrievePlan* Plan,
             Timestamp timestamp,
             int64_t limit_size) const = 0;

    // TODO: memory use is not correct when load string or load string index
    virtual int64_t
    GetMemoryUsageInBytes() const = 0;

    virtual int64_t
    get_row_count() const = 0;

    virtual const Schema&
    get_schema() const = 0;

    virtual int64_t
    get_deleted_count() const = 0;

    virtual int64_t
    get_real_count() const = 0;

    virtual int64_t
    get_field_avg_size(FieldId field_id) const = 0;

    virtual void
    set_field_avg_size(FieldId field_id,
                       int64_t num_rows,
                       int64_t field_size) = 0;

    //  virtual int64_t
    //  PreDelete(int64_t size) = 0;

    virtual SegcoreError
    Delete(int64_t reserved_offset,
           int64_t size,
           const IdArray* pks,
           const Timestamp* timestamps) = 0;

    virtual void
    LoadDeletedRecord(const LoadDeletedRecordInfo& info) = 0;

    virtual void
    LoadFieldData(const LoadFieldDataInfo& info) = 0;

    virtual void
    LoadFieldDataV2(const LoadFieldDataInfo& info) = 0;

    virtual int64_t
    get_segment_id() const = 0;

    virtual SegmentType
    type() const = 0;

    virtual bool
    HasRawData(int64_t field_id) const = 0;
};

// internal API for DSL calculation
// only for implementation
class SegmentInternalInterface : public SegmentInterface {
 public:
    template <typename T>
    Span<T>
    chunk_data(FieldId field_id, int64_t chunk_id) const {
        return static_cast<Span<T>>(chunk_data_impl(field_id, chunk_id));
    }

    template <typename T>
    const index::ScalarIndex<T>&
    chunk_scalar_index(FieldId field_id, int64_t chunk_id) const {
        static_assert(IsScalar<T>);
        using IndexType = index::ScalarIndex<T>;
        auto base_ptr = chunk_index_impl(field_id, chunk_id);
        auto ptr = dynamic_cast<const IndexType*>(base_ptr);
        AssertInfo(ptr, "entry mismatch");
        return *ptr;
    }

    std::unique_ptr<SearchResult>
    Search(const query::Plan* Plan,
           const query::PlaceholderGroup* placeholder_group) const override;

    void
    FillPrimaryKeys(const query::Plan* plan,
                    SearchResult& results) const override;

    void
    FillTargetEntry(const query::Plan* plan,
                    SearchResult& results) const override;

    std::unique_ptr<proto::segcore::RetrieveResults>
    Retrieve(const query::RetrievePlan* Plan,
             Timestamp timestamp,
             int64_t limit_size) const override;

    virtual bool
    HasIndex(FieldId field_id) const = 0;

    virtual bool
    HasFieldData(FieldId field_id) const = 0;

    virtual std::string
    debug() const = 0;

    int64_t
    get_real_count() const override;

    int64_t
    get_field_avg_size(FieldId field_id) const override;

    void
    set_field_avg_size(FieldId field_id,
                       int64_t num_rows,
                       int64_t field_size) override;

    const SkipIndex&
    GetSkipIndex() const;

    void
    LoadPrimitiveSkipIndex(FieldId field_id,
                           int64_t chunk_id,
                           DataType data_type,
                           const void* chunk_data,
                           int64_t count);

    void
    LoadStringSkipIndex(FieldId field_id,
                        int64_t chunk_id,
                        const milvus::VariableColumn<std::string>& var_column);

 public:
    virtual void
    vector_search(SearchInfo& search_info,
                  const void* query_data,
                  int64_t query_count,
                  Timestamp timestamp,
                  const BitsetView& bitset,
                  SearchResult& output) const = 0;

    virtual void
    mask_with_delete(BitsetType& bitset,
                     int64_t ins_barrier,
                     Timestamp timestamp) const = 0;

    // count of chunk that has index available
    virtual int64_t
    num_chunk_index(FieldId field_id) const = 0;

    // count of chunk that has raw data
    virtual int64_t
    num_chunk_data(FieldId field_id) const = 0;

    virtual void
    mask_with_timestamps(BitsetType& bitset_chunk,
                         Timestamp timestamp) const = 0;

    // count of chunks
    virtual int64_t
    num_chunk() const = 0;

    // element size in each chunk
    virtual int64_t
    size_per_chunk() const = 0;

    virtual int64_t
    get_active_count(Timestamp ts) const = 0;

    virtual std::pair<std::unique_ptr<IdArray>, std::vector<SegOffset>>
    search_ids(const IdArray& id_array, Timestamp timestamp) const = 0;

    /**
     * Apply timestamp filtering on bitset, the query can't see an entity whose
     * timestamp is bigger than the timestamp of query.
     *
     * @param bitset The final bitset after scalar filtering and delta filtering,
     *  `false` means that the entity will be filtered out.
     * @param timestamp The timestamp of query.
     */
    void
    timestamp_filter(BitsetType& bitset, Timestamp timestamp) const;

    /**
     * Apply timestamp filtering on bitset, the query can't see an entity whose
     * timestamp is bigger than the timestamp of query. The passed offsets are
     * all candidate entities.
     *
     * @param bitset The final bitset after scalar filtering and delta filtering,
     *  `true` means that the entity will be filtered out.
     * @param offsets The segment offsets of all candidates.
     * @param timestamp The timestamp of query.
     */
    void
    timestamp_filter(BitsetType& bitset,
                     const std::vector<int64_t>& offsets,
                     Timestamp timestamp) const;

    /**
     * Sort all candidates in ascending order, and then return the limit smallest.
     * Bitset is used to check if the candidate will be filtered out. `false_filtered_out`
     * determines how to filter out candidates. If `false_filtered_out` is true, we will
     * filter all candidates whose related bit is false.
     *
     * @param limit
     * @param bitset
     * @param false_filtered_out
     * @return All candidates offsets.
     */
    virtual std::vector<OffsetMap::OffsetType>
    find_first(int64_t limit,
               const BitsetType& bitset,
               bool false_filtered_out) const = 0;

 protected:
    // internal API: return chunk_data in span
    virtual SpanBase
    chunk_data_impl(FieldId field_id, int64_t chunk_id) const = 0;

    // internal API: return chunk_index in span, support scalar index only
    virtual const index::IndexBase*
    chunk_index_impl(FieldId field_id, int64_t chunk_id) const = 0;

    // calculate output[i] = Vec[seg_offsets[i]}, where Vec binds to system_type
    virtual void
    bulk_subscript(SystemFieldType system_type,
                   const int64_t* seg_offsets,
                   int64_t count,
                   void* output) const = 0;

    // calculate output[i] = Vec[seg_offsets[i]}, where Vec binds to field_offset
    virtual std::unique_ptr<DataArray>
    bulk_subscript(FieldId field_id,
                   const int64_t* seg_offsets,
                   int64_t count) const = 0;

    virtual void
    check_search(const query::Plan* plan) const = 0;

    virtual const ConcurrentVector<Timestamp>&
    get_timestamps() const = 0;

 protected:
    mutable std::shared_mutex mutex_;
    // fieldID -> std::pair<num_rows, avg_size>
    std::unordered_map<FieldId, std::pair<int64_t, int64_t>>
        variable_fields_avg_size_;  // bytes;
    SkipIndex skipIndex_;
};

}  // namespace milvus::segcore
