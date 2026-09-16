#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/invlists/TieredArrayInvertedLists.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct FvecsData {
    int d = 0;
    int n = 0;
    std::vector<float> data;
};

FvecsData load_fvecs(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    int d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    if (!in || d <= 0) {
        throw std::runtime_error("Failed to read dimension from: " + filename);
    }

    in.seekg(0, std::ios::end);
    std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    const std::streamoff vec_size_bytes =
            static_cast<std::streamoff>((d + 1) * sizeof(float));
    if (file_size % vec_size_bytes != 0) {
        throw std::runtime_error("Invalid .fvecs file size: " + filename);
    }

    int n = static_cast<int>(file_size / vec_size_bytes);

    FvecsData result;
    result.d = d;
    result.n = n;
    result.data.resize(static_cast<size_t>(n) * d);

    for (int i = 0; i < n; i++) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!in || dim != d) {
            throw std::runtime_error("Invalid vector dimension in: " + filename);
        }

        in.read(reinterpret_cast<char*>(
                        result.data.data() + static_cast<size_t>(i) * d),
                static_cast<std::streamsize>(d * sizeof(float)));
        if (!in) {
            throw std::runtime_error(
                    "Failed while reading vector data from: " + filename);
        }
    }

    return result;
}

struct ListStat {
    size_t id = 0;
    size_t size = 0;
    uint64_t probe_epoch = 0;
    uint64_t scan_epoch = 0;
    uint64_t scanned_vectors_epoch = 0;
    uint64_t probe_total = 0;
    uint64_t scan_total = 0;
    uint64_t scanned_vectors_total = 0;
    faiss::MemoryTier tier = faiss::MemoryTier::DRAM;
};

static const char* tier_to_string(faiss::MemoryTier tier) {
    switch (tier) {
        case faiss::MemoryTier::DRAM:
            return "DRAM";
        case faiss::MemoryTier::CXL:
            return "CXL";
        default:
            return "UNKNOWN";
    }
}

int main(int argc, char* argv[]) {
    try {
	if (argc < 5) {
            std::printf(
                    "Usage: %s <nlist> <nprobe> <hot_lists> <k> [epoch_count] [data_dir]\n",
                    argv[0]);
            return 1;
        }

        const std::string data_dir = argc > 6 ? argv[6] : "sift";
        const std::string learn_file = data_dir + "/sift_learn.fvecs";
        const std::string base_file = data_dir + "/sift_base.fvecs";
        const std::string query_file = data_dir + "/sift_query.fvecs";

	const int nlist = std::atoi(argv[1]);
	const int nprobe_val = std::atoi(argv[2]);
	const size_t hot_lists_to_keep_in_dram = std::atol(argv[3]);
	const int k = std::atoi(argv[4]);
	const size_t epoch_count = argc > 5 ? std::atol(argv[5]) : 5;

        // -------------------------------------------------
        // Step 0: load SIFT data
        // -------------------------------------------------
        FvecsData learn = load_fvecs(learn_file);
        FvecsData base = load_fvecs(base_file);
        FvecsData query = load_fvecs(query_file);

        if (learn.d != base.d || base.d != query.d) {
            throw std::runtime_error("Dimension mismatch between SIFT files");
        }

        const int d = base.d;
        const int nlearn = learn.n;
        const int nb = base.n;
        const int nq = query.n;

        std::printf("Loaded SIFT data:\n");
        std::printf("  d      = %d\n", d);
        std::printf("  nlearn = %d\n", nlearn);
        std::printf("  nb     = %d\n", nb);
        std::printf("  nq     = %d\n", nq);

        // -------------------------------------------------
        // Step 1: create the coarse quantizer
        // -------------------------------------------------
        faiss::IndexFlatL2 quantizer(d);

        // -------------------------------------------------
        // Step 2: create IVF index
        // -------------------------------------------------
        faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);

        auto* tiered_invlists =
        new faiss::TieredArrayInvertedLists(nlist, index.code_size);
        index.replace_invlists(tiered_invlists, true);

        std::printf("Is trained: %d\n", index.is_trained);

        // -------------------------------------------------
        // Step 3: train index using sift_learn.fvecs
        // -------------------------------------------------
        index.train(nlearn, learn.data.data());

        std::printf("Is trained after training: %d\n", index.is_trained);

        // -------------------------------------------------
        // Step 4: add database vectors using sift_base.fvecs
        // -------------------------------------------------
        index.add(nb, base.data.data());

        std::printf("Total vectors indexed: %ld\n", index.ntotal);

        // -------------------------------------------------
        // Step 5: prepare search
        // -------------------------------------------------
        index.nprobe = nprobe_val;

        std::vector<float> distances(static_cast<size_t>(k) * nq);
        std::vector<faiss::idx_t> labels(static_cast<size_t>(k) * nq);

        // reset lifetime + epoch stats once before full experiment
        index.reset_list_stats();

        const size_t queries_per_epoch = static_cast<size_t>(nq) / epoch_count;
        std::vector<faiss::MemoryTier> prev_tiers;

        for (size_t epoch = 0; epoch < epoch_count; epoch++) {
            size_t q_begin = epoch * queries_per_epoch;
            size_t q_end = (epoch == epoch_count - 1)
                    ? static_cast<size_t>(nq)
                    : (epoch + 1) * queries_per_epoch;
            size_t q_count = q_end - q_begin;

            std::printf("\n==================================================\n");
            std::printf("Epoch %zu / %zu\n", epoch + 1, epoch_count);
            std::printf("Query range: [%zu, %zu)  count = %zu\n",
                    q_begin,
                    q_end,
                    q_count);

            // ---------------------------------------------
            // Reset epoch stats and search one epoch only
            // ---------------------------------------------
            index.reset_list_epoch_stats();
	    tiered_invlists->reset_migration_epoch_stats();

            index.search(
                    static_cast<faiss::idx_t>(q_count),
                    query.data.data() + q_begin * static_cast<size_t>(d),
                    k,
                    distances.data() + q_begin * static_cast<size_t>(k),
                    labels.data() + q_begin * static_cast<size_t>(k));

            // Print only first few queries of first epoch
            if (epoch == 0) {
                const size_t queries_to_print = std::min<size_t>(q_count, 10);
                for (size_t i = 0; i < queries_to_print; i++) {
                    size_t qi = q_begin + i;
                    std::printf("Query %zu:\n", qi);
                    for (int j = 0; j < k; j++) {
                        std::printf("  id=%ld  dist=%f\n",
                                labels[qi * static_cast<size_t>(k) + j],
                                distances[qi * static_cast<size_t>(k) + j]);
                    }
                }
            }

            // ---------------------------------------------
            // Fetch epoch + total stats
            // ---------------------------------------------
            std::vector<uint64_t> probe_epoch =
                    index.get_list_probe_count_epoch();
            std::vector<uint64_t> scan_epoch =
                    index.get_list_scan_count_epoch();
            std::vector<uint64_t> work_epoch =
                    index.get_list_scanned_vectors_epoch();

            std::vector<uint64_t> probe_total =
                    index.get_list_probe_count();
            std::vector<uint64_t> scan_total =
                    index.get_list_scan_count();
            std::vector<uint64_t> work_total =
                    index.get_list_scanned_vectors();

            // ---------------------------------------------
            // Build sortable per-list stats table
            // ---------------------------------------------
            std::vector<ListStat> stats;
            stats.reserve(probe_epoch.size());

            for (size_t i = 0; i < probe_epoch.size(); i++) {
                stats.push_back(ListStat{
                        i,
                        index.get_list_size(i),
                        probe_epoch[i],
                        scan_epoch[i],
                        work_epoch[i],
                        probe_total[i],
                        scan_total[i],
                        work_total[i],
                        index.get_list_tier(i)});
            }

            // ---------------------------------------------
            // Sort by epoch work (for reporting only)
            // ---------------------------------------------
            std::sort(
                    stats.begin(),
                    stats.end(),
                    [](const ListStat& a, const ListStat& b) {
                        return a.scanned_vectors_epoch >
                                b.scanned_vectors_epoch;
                    });

            // ---------------------------------------------
            // Recompute logical tiers inside FAISS
            // ---------------------------------------------
            size_t predicted_changes =
                    index.count_tier_changes_if_recomputed_from_epoch_stats(
                            hot_lists_to_keep_in_dram);

            index.recompute_tiers_from_epoch_stats(hot_lists_to_keep_in_dram);
            index.apply_static_tier_placement();

	    std::printf("\nActual migration (measured):\n");
	    std::printf("  promoted (CXL->DRAM): %llu lists, %llu bytes (%.2f MB)\n",
    		(unsigned long long)tiered_invlists->get_lists_promoted_epoch(),
    		(unsigned long long)tiered_invlists->get_bytes_promoted_epoch(),
    		tiered_invlists->get_bytes_promoted_epoch() / (1024.0 * 1024.0));
	    std::printf("  demoted  (DRAM->CXL): %llu lists, %llu bytes (%.2f MB)\n",
    		(unsigned long long)tiered_invlists->get_lists_demoted_epoch(),
    		(unsigned long long)tiered_invlists->get_bytes_demoted_epoch(),
    		tiered_invlists->get_bytes_demoted_epoch() / (1024.0 * 1024.0));
	    std::printf("  total moved: %llu bytes (%.2f MB)\n",
    		(unsigned long long)(tiered_invlists->get_bytes_promoted_epoch() +
                          tiered_invlists->get_bytes_demoted_epoch()),
    		(tiered_invlists->get_bytes_promoted_epoch() +
     		tiered_invlists->get_bytes_demoted_epoch()) / (1024.0 * 1024.0));

            std::vector<faiss::MemoryTier> curr_tiers = index.get_list_tiers();

            // refresh tiers in stats
            for (size_t i = 0; i < stats.size(); i++) {
                stats[i].tier = curr_tiers[stats[i].id];
            }

            size_t changed_tiers = predicted_changes;

            // optional consistency check against previous tiers
            if (!prev_tiers.empty() && prev_tiers.size() == curr_tiers.size()) {
                size_t observed_changes = 0;
                for (size_t i = 0; i < curr_tiers.size(); i++) {
                    if (curr_tiers[i] != prev_tiers[i]) {
                        observed_changes++;
                    }
                }

                if (observed_changes != changed_tiers) {
                    std::printf(
                            "Warning: predicted_changes=%zu but observed_changes=%zu\n",
                            changed_tiers,
                            observed_changes);
                    changed_tiers = observed_changes;
                }
            }

	    if (!prev_tiers.empty()) {
    		size_t intersection = 0;
    		size_t union_size = 0;

    	    	for (size_t i = 0; i < curr_tiers.size(); i++) {
        		bool was_dram = (prev_tiers[i] == faiss::MemoryTier::DRAM);
        		bool is_dram = (curr_tiers[i] == faiss::MemoryTier::DRAM);
        		if (was_dram && is_dram) intersection++;
        		if (was_dram || is_dram) union_size++;
    	    	}

    	    	double jaccard = union_size > 0 ? double(intersection) / double(union_size) : 1.0;

    	    	std::printf("\nHot set stability:\n");
   	    	std::printf("  Jaccard similarity with previous epoch: %.4f\n", jaccard);
	    }

            prev_tiers = curr_tiers;

            // ---------------------------------------------
            // Epoch totals
            // ---------------------------------------------
            uint64_t total_probes_epoch = 0;
            uint64_t total_scans_epoch = 0;
            uint64_t total_work_epoch = 0;

            for (const auto& s : stats) {
                total_probes_epoch += s.probe_epoch;
                total_scans_epoch += s.scan_epoch;
                total_work_epoch += s.scanned_vectors_epoch;
            }

            uint64_t dram_work_epoch = index.get_dram_scanned_vectors_epoch();
            uint64_t cxl_work_epoch = index.get_cxl_scanned_vectors_epoch();

            double dram_pct = total_work_epoch > 0
                    ? 100.0 * double(dram_work_epoch) / double(total_work_epoch)
                    : 0.0;
            double cxl_pct = total_work_epoch > 0
                    ? 100.0 * double(cxl_work_epoch) / double(total_work_epoch)
                    : 0.0;

            std::printf("\nPer-tier epoch work:\n");
            std::printf("  DRAM scanned vectors epoch = %llu (%.2f%%)\n",
                    static_cast<unsigned long long>(dram_work_epoch),
                    dram_pct);
            std::printf("  CXL  scanned vectors epoch = %llu (%.2f%%)\n",
                    static_cast<unsigned long long>(cxl_work_epoch),
                    cxl_pct);

            // ---------------------------------------------
            // Print top hottest lists
            // ---------------------------------------------
            std::printf("\nTop 10 hottest lists in this epoch:\n");
            std::printf(
                    "list_id  tier   size   probe_epoch  scan_epoch  scanned_vectors_epoch\n");
            for (size_t i = 0; i < 10 && i < stats.size(); i++) {
                const auto& s = stats[i];
                std::printf("%6zu  %-4s  %6zu  %11llu  %10llu  %21llu\n",
                        s.id,
                        tier_to_string(s.tier),
                        s.size,
                        static_cast<unsigned long long>(s.probe_epoch),
                        static_cast<unsigned long long>(s.scan_epoch),
                        static_cast<unsigned long long>(s.scanned_vectors_epoch));
            }

            // ---------------------------------------------
            // Print top coldest lists
            // ---------------------------------------------
            std::printf("\nTop 10 coldest lists in this epoch:\n");
            std::printf(
                    "list_id  tier   size   probe_epoch  scan_epoch  scanned_vectors_epoch\n");
            size_t start = stats.size() > 10 ? stats.size() - 10 : 0;
            for (size_t i = start; i < stats.size(); i++) {
                const auto& s = stats[i];
                std::printf("%6zu  %-4s  %6zu  %11llu  %10llu  %21llu\n",
                        s.id,
                        tier_to_string(s.tier),
                        s.size,
                        static_cast<unsigned long long>(s.probe_epoch),
                        static_cast<unsigned long long>(s.scan_epoch),
                        static_cast<unsigned long long>(s.scanned_vectors_epoch));
            }

            // ---------------------------------------------
            // Cumulative hotness report
            // ---------------------------------------------
            std::printf("\nCumulative epoch work concentration:\n");
            for (size_t cut : {size_t(10), size_t(20), size_t(30), size_t(40)}) {
                uint64_t running = 0;
                size_t actual = std::min(cut, stats.size());
                for (size_t i = 0; i < actual; i++) {
                    running += stats[i].scanned_vectors_epoch;
                }

                double pct = total_work_epoch > 0
                        ? (100.0 * double(running) / double(total_work_epoch))
                        : 0.0;

                std::printf(
                        "  top %2zu lists -> %12llu scanned_vectors_epoch (%.2f%%)\n",
                        actual,
                        static_cast<unsigned long long>(running),
                        pct);
            }

            // ---------------------------------------------
            // Tier summary
            // ---------------------------------------------
            size_t dram_count = 0;
            size_t cxl_count = 0;
            for (auto t : curr_tiers) {
                if (t == faiss::MemoryTier::DRAM) {
                    dram_count++;
                } else if (t == faiss::MemoryTier::CXL) {
                    cxl_count++;
                }
            }

            std::printf("\nTier summary:\n");
            std::printf("  DRAM lists         = %zu\n", dram_count);
            std::printf("  CXL lists          = %zu\n", cxl_count);
            std::printf("  tier changes       = %zu\n", changed_tiers);

            // ---------------------------------------------
            // Epoch totals
            // ---------------------------------------------
            std::printf("\nEpoch totals:\n");
            std::printf("  total probes epoch          = %llu\n",
                    static_cast<unsigned long long>(total_probes_epoch));
            std::printf("  total scans epoch           = %llu\n",
                    static_cast<unsigned long long>(total_scans_epoch));
            std::printf("  total scanned vectors epoch = %llu\n",
                    static_cast<unsigned long long>(total_work_epoch));
            std::printf("  expected probes epoch       = q_count * nprobe = %zu\n",
                    q_count * static_cast<size_t>(index.nprobe));
        }

        // -------------------------------------------------
        // Final lifetime totals
        // -------------------------------------------------
        std::vector<uint64_t> probe_total = index.get_list_probe_count();
        std::vector<uint64_t> scan_total = index.get_list_scan_count();
        std::vector<uint64_t> work_total = index.get_list_scanned_vectors();

        uint64_t total_probes_total = 0;
        uint64_t total_scans_total = 0;
        uint64_t total_work_total = 0;

        for (size_t i = 0; i < probe_total.size(); i++) {
            total_probes_total += probe_total[i];
            total_scans_total += scan_total[i];
            total_work_total += work_total[i];
        }

        std::printf("\n==================================================\n");
        std::printf("Final lifetime totals:\n");
        std::printf("  total probes total          = %llu\n",
                static_cast<unsigned long long>(total_probes_total));
        std::printf("  total scans total           = %llu\n",
                static_cast<unsigned long long>(total_scans_total));
        std::printf("  total scanned vectors total = %llu\n",
                static_cast<unsigned long long>(total_work_total));
        std::printf("  expected probes total       = nq * nprobe = %ld\n",
                nq * index.nprobe);

        std::printf("\nPer-tier lifetime work:\n");
        std::printf("  DRAM scanned vectors total = %llu\n",
                static_cast<unsigned long long>(index.get_dram_scanned_vectors_total()));
        std::printf("  CXL  scanned vectors total = %llu\n",
                static_cast<unsigned long long>(index.get_cxl_scanned_vectors_total()));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
