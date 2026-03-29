#ifndef FAISS_TIERED_ARRAY_INVERTED_LISTS_H
#define FAISS_TIERED_ARRAY_INVERTED_LISTS_H

#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/invlists/InvertedLists.h>

namespace faiss {

struct TieredArrayInvertedLists : InvertedLists {
    // same storage model as ArrayInvertedLists
    std::vector<std::vector<uint8_t>> codes;
    std::vector<std::vector<idx_t>> ids;

    // physical/storage tier metadata per list
    std::vector<MemoryTier> list_tiers;

    explicit TieredArrayInvertedLists(size_t nlist, size_t code_size);

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

    // tier-aware helpers
    MemoryTier get_list_tier(size_t list_no) const;
    void set_list_tier(size_t list_no, MemoryTier tier);
    void move_list_to_tier(size_t list_no, MemoryTier tier);

    private:
    void relocate_list_storage(size_t list_no);
};

} // namespace faiss

#endif