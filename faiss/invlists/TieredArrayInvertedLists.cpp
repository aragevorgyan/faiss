/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/TieredArrayInvertedLists.h>

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <numa.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

TieredArrayInvertedLists::TieredArrayInvertedLists(
        size_t nlist,
        size_t code_size)
        : InvertedLists(nlist, code_size),
          lists(nlist),
          list_tiers(nlist, MemoryTier::DRAM) {}

TieredArrayInvertedLists::~TieredArrayInvertedLists() {
    reset();
}

size_t TieredArrayInvertedLists::list_size(size_t list_no) const {
    FAISS_THROW_IF_NOT(list_no < nlist);
    return lists[list_no].size;
}

const uint8_t* TieredArrayInvertedLists::get_codes(size_t list_no) const {
    FAISS_THROW_IF_NOT(list_no < nlist);
    return lists[list_no].codes;
}

const idx_t* TieredArrayInvertedLists::get_ids(size_t list_no) const {
    FAISS_THROW_IF_NOT(list_no < nlist);
    return lists[list_no].ids;
}

idx_t* TieredArrayInvertedLists::allocate_ids_buffer(size_t n, MemoryTier tier) {
    if (n == 0) {
        return nullptr;
    }

    const size_t nbytes = n * sizeof(idx_t);

    if (tier == MemoryTier::CXL) {
        if (numa_available() != -1 && numa_num_configured_nodes() > 1) {
            void* p = numa_alloc_onnode(nbytes, cxl_numa_node);
            FAISS_THROW_IF_NOT_MSG(p != nullptr, "numa_alloc_onnode ids failed");

            std::printf("CXL alloc ids: n=%zu bytes=%zu node=%d ptr=%p\n",
                n, nbytes, cxl_numa_node, p);

            return static_cast<idx_t*>(p);
        } else {
            std::fprintf(stderr,
                    "Warning: NUMA/node1 unavailable for CXL ids allocation, falling back to malloc\n");
        }
    }

    void* p = std::malloc(nbytes);
    FAISS_THROW_IF_NOT_MSG(p != nullptr, "malloc ids failed");
    return static_cast<idx_t*>(p);
}

uint8_t* TieredArrayInvertedLists::allocate_codes_buffer(
        size_t nbytes,
        MemoryTier tier) {
    if (nbytes == 0) {
        return nullptr;
    }

    if (tier == MemoryTier::CXL) {
        if (numa_available() != -1 && numa_num_configured_nodes() > 1) {
            void* p = numa_alloc_onnode(nbytes, cxl_numa_node);
            FAISS_THROW_IF_NOT_MSG(p != nullptr, "numa_alloc_onnode codes failed");

            std::printf("CXL alloc codes: bytes=%zu node=%d ptr=%p\n",
                nbytes, cxl_numa_node, p);

            return static_cast<uint8_t*>(p);
        } else {
            std::fprintf(stderr,
                    "Warning: NUMA/node1 unavailable for CXL codes allocation, falling back to malloc\n");
        }
    }

    void* p = std::malloc(nbytes);
    FAISS_THROW_IF_NOT_MSG(p != nullptr, "malloc codes failed");
    return static_cast<uint8_t*>(p);
}

void TieredArrayInvertedLists::free_ids_buffer(
        idx_t* ptr,
        size_t n,
        MemoryTier tier) {
    if (!ptr) {
        return;
    }

    const size_t nbytes = n * sizeof(idx_t);

    if (tier == MemoryTier::CXL &&
        numa_available() != -1 &&
        numa_num_configured_nodes() > 1) {
        std::printf("CXL free ids: n=%zu bytes=%zu ptr=%p\n", n, nbytes, ptr);
        numa_free(ptr, nbytes);
        return;
    }

    std::free(ptr);
}

void TieredArrayInvertedLists::free_codes_buffer(
        uint8_t* ptr,
        size_t nbytes,
        MemoryTier tier) {
    if (!ptr) {
        return;
    }

    if (tier == MemoryTier::CXL &&
        numa_available() != -1 &&
        numa_num_configured_nodes() > 1) {
        std::printf("CXL free codes: bytes=%zu ptr=%p\n", nbytes, ptr);
        numa_free(ptr, nbytes);
        return;
    }

    std::free(ptr);
}

void TieredArrayInvertedLists::free_list_storage(size_t list_no) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    auto& lst = lists[list_no];
    MemoryTier tier = list_tiers[list_no];

    free_ids_buffer(lst.ids, lst.capacity, tier);
    free_codes_buffer(lst.codes, lst.capacity * code_size, tier);

    lst.ids = nullptr;
    lst.codes = nullptr;
    lst.size = 0;
    lst.capacity = 0;
}

void TieredArrayInvertedLists::ensure_capacity(size_t list_no, size_t min_capacity) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    auto& lst = lists[list_no];
    if (lst.capacity >= min_capacity) {
        return;
    }

    size_t new_capacity = lst.capacity == 0 ? min_capacity : lst.capacity;
    while (new_capacity < min_capacity) {
        new_capacity *= 2;
        if (new_capacity < min_capacity) {
            new_capacity = min_capacity;
            break;
        }
    }

    MemoryTier tier = list_tiers[list_no];

    idx_t* new_ids = allocate_ids_buffer(new_capacity, tier);
    uint8_t* new_codes = allocate_codes_buffer(new_capacity * code_size, tier);

    if (lst.size > 0) {
        std::memcpy(new_ids, lst.ids, lst.size * sizeof(idx_t));
        std::memcpy(new_codes, lst.codes, lst.size * code_size);
    }

    free_ids_buffer(lst.ids, lst.capacity, tier);
    free_codes_buffer(lst.codes, lst.capacity * code_size, tier);

    lst.ids = new_ids;
    lst.codes = new_codes;
    lst.capacity = new_capacity;
}

size_t TieredArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    auto& lst = lists[list_no];
    size_t old_size = lst.size;

    if (n_entry == 0) {
        return old_size;
    }

    ensure_capacity(list_no, old_size + n_entry);

    std::memcpy(lst.ids + old_size, ids_in, n_entry * sizeof(idx_t));
    std::memcpy(lst.codes + old_size * code_size, code, n_entry * code_size);

    lst.size += n_entry;
    return old_size;
}

void TieredArrayInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    auto& lst = lists[list_no];
    FAISS_THROW_IF_NOT(offset + n_entry <= lst.size);

    if (n_entry == 0) {
        return;
    }

    std::memcpy(lst.ids + offset, ids_in, n_entry * sizeof(idx_t));
    std::memcpy(lst.codes + offset * code_size, code, n_entry * code_size);
}

void TieredArrayInvertedLists::resize(size_t list_no, size_t new_size) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    auto& lst = lists[list_no];
    if (new_size > lst.capacity) {
        ensure_capacity(list_no, new_size);
    }

    lst.size = new_size;
}

void TieredArrayInvertedLists::reset() {
    for (size_t i = 0; i < nlist; i++) {
        free_list_storage(i);
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

void TieredArrayInvertedLists::relocate_list_storage(
        size_t list_no,
        MemoryTier dst_tier) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    auto& lst = lists[list_no];
    MemoryTier src_tier = list_tiers[list_no];

    if (src_tier == dst_tier) {
        return;
    }

    idx_t* new_ids = allocate_ids_buffer(lst.capacity, dst_tier);
    uint8_t* new_codes = allocate_codes_buffer(lst.capacity * code_size, dst_tier);

    if (lst.size > 0) {
        std::memcpy(new_ids, lst.ids, lst.size * sizeof(idx_t));
        std::memcpy(new_codes, lst.codes, lst.size * code_size);
    }

    free_ids_buffer(lst.ids, lst.capacity, src_tier);
    free_codes_buffer(lst.codes, lst.capacity * code_size, src_tier);

    lst.ids = new_ids;
    lst.codes = new_codes;
}

void TieredArrayInvertedLists::move_list_to_tier(size_t list_no, MemoryTier tier) {
    FAISS_THROW_IF_NOT(list_no < nlist);

    MemoryTier old_tier = list_tiers[list_no];
    if (old_tier == tier) {
        return;
    }

    relocate_list_storage(list_no, tier);
    list_tiers[list_no] = tier;

    printf(
            "TieredArrayInvertedLists: moved list %zu from %d to %d (size=%zu)\n",
            list_no,
            int(old_tier),
            int(tier),
            lists[list_no].size);
}
} // namespace faiss