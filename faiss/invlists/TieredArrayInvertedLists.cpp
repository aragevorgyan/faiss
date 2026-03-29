/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/TieredArrayInvertedLists.h>

#include <cstring>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

TieredArrayInvertedLists::TieredArrayInvertedLists(
        size_t nlist,
        size_t code_size)
        : InvertedLists(nlist, code_size),
          codes(nlist),
          ids(nlist),
          list_tiers(nlist, MemoryTier::DRAM) {}

size_t TieredArrayInvertedLists::list_size(size_t list_no) const {
    FAISS_THROW_IF_NOT(list_no < nlist);
    return ids[list_no].size();
}

const uint8_t* TieredArrayInvertedLists::get_codes(size_t list_no) const {
    FAISS_THROW_IF_NOT(list_no < nlist);
    return codes[list_no].empty() ? nullptr : codes[list_no].data();
}

const idx_t* TieredArrayInvertedLists::get_ids(size_t list_no) const {
    FAISS_THROW_IF_NOT(list_no < nlist);
    return ids[list_no].empty() ? nullptr : ids[list_no].data();
}

size_t TieredArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    size_t o = ids[list_no].size();
    ids[list_no].resize(o + n_entry);
    codes[list_no].resize((o + n_entry) * code_size);

    if (n_entry > 0) {
        memcpy(ids[list_no].data() + o, ids_in, sizeof(ids_in[0]) * n_entry);
        memcpy(
                codes[list_no].data() + o * code_size,
                code,
                code_size * n_entry);
    }

    return o;
}

void TieredArrayInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code) {
    FAISS_THROW_IF_NOT(list_no < nlist);
    FAISS_THROW_IF_NOT(offset + n_entry <= ids[list_no].size());

    if (n_entry > 0) {
        memcpy(
                ids[list_no].data() + offset,
                ids_in,
                sizeof(ids_in[0]) * n_entry);
        memcpy(
                codes[list_no].data() + offset * code_size,
                code,
                code_size * n_entry);
    }
}

void TieredArrayInvertedLists::resize(size_t list_no, size_t new_size) {
    FAISS_THROW_IF_NOT(list_no < nlist);
    ids[list_no].resize(new_size);
    codes[list_no].resize(new_size * code_size);
}

void TieredArrayInvertedLists::reset() {
    for (size_t i = 0; i < nlist; i++) {
        ids[i].clear();
        codes[i].clear();
        list_tiers[i] = MemoryTier::DRAM;
    }
}

MemoryTier TieredArrayInvertedLists::get_list_tier(size_t list_no) const {
    FAISS_THROW_IF_NOT(list_no < nlist);
    return list_tiers[list_no];
}

void TieredArrayInvertedLists::set_list_tier(size_t list_no, MemoryTier tier) {
    FAISS_THROW_IF_NOT(list_no < nlist);
    list_tiers[list_no] = tier;
}

void TieredArrayInvertedLists::move_list_to_tier(size_t list_no, MemoryTier tier) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    MemoryTier old_tier = list_tiers[list_no];
    if (old_tier == tier) {
        return;
    }

    relocate_list_storage(list_no);
    list_tiers[list_no] = tier;

    printf(
            "TieredArrayInvertedLists: moved list %zu from %d to %d (size=%zu)\n",
            list_no,
            int(old_tier),
            int(tier),
            ids[list_no].size());
}

void TieredArrayInvertedLists::relocate_list_storage(size_t list_no) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    const size_t sz = ids[list_no].size();
    FAISS_THROW_IF_NOT(codes[list_no].size() == sz * code_size);

    // allocate fresh storage
    std::vector<idx_t> new_ids(sz);
    std::vector<uint8_t> new_codes(sz * code_size);

    // copy ids
    if (sz > 0) {
        memcpy(new_ids.data(), ids[list_no].data(), sz * sizeof(idx_t));
        memcpy(
                new_codes.data(),
                codes[list_no].data(),
                sz * code_size * sizeof(uint8_t));
    }

    // swap new storage into place
    ids[list_no].swap(new_ids);
    codes[list_no].swap(new_codes);

    // old storage is released automatically when new_ids/new_codes go out of scope
}

} // namespace faiss