/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_TIERED_ARRAY_INVERTED_LISTS_H
#define FAISS_TIERED_ARRAY_INVERTED_LISTS_H

#include <cstddef>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/invlists/InvertedLists.h>

namespace faiss {

struct TieredArrayInvertedLists : InvertedLists {
    int cxl_numa_node = 1;
    struct ListStorage {
        idx_t* ids = nullptr;
        uint8_t* codes = nullptr;
        size_t size = 0;
        size_t capacity = 0;
    };

    std::vector<ListStorage> lists;
    std::vector<MemoryTier> list_tiers;

    uint64_t bytes_promoted_epoch = 0;
    uint64_t bytes_demoted_epoch = 0;
    uint64_t bytes_promoted_total = 0;
    uint64_t bytes_demoted_total = 0;
    uint64_t lists_promoted_epoch = 0;
    uint64_t lists_demoted_epoch = 0;

    explicit TieredArrayInvertedLists(size_t nlist, size_t code_size);
    ~TieredArrayInvertedLists() override;

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids_in,
            const uint8_t* code) override;

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids_in,
            const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;
    void reset() override;

    MemoryTier get_list_tier(size_t list_no) const;
    void set_list_tier(size_t list_no, MemoryTier tier);
    void move_list_to_tier(size_t list_no, MemoryTier tier);

    uint64_t get_bytes_promoted_epoch() const;
    uint64_t get_bytes_demoted_epoch() const;
    uint64_t get_bytes_promoted_total() const;
    uint64_t get_bytes_demoted_total() const;
    uint64_t get_lists_promoted_epoch() const;
    uint64_t get_lists_demoted_epoch() const;
    void reset_migration_epoch_stats();

   private:
    idx_t* allocate_ids_buffer(size_t n, MemoryTier tier);
    uint8_t* allocate_codes_buffer(size_t nbytes, MemoryTier tier);

    void free_ids_buffer(idx_t* ptr, size_t n, MemoryTier tier);
    void free_codes_buffer(uint8_t* ptr, size_t nbytes, MemoryTier tier);

    void ensure_capacity(size_t list_no, size_t min_capacity);
    void relocate_list_storage(size_t list_no, MemoryTier dst_tier);
    void free_list_storage(size_t list_no);
};

} // namespace faiss

#endif
