#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
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
    if (!in) throw std::runtime_error("Failed to open: " + filename);
    int d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    in.seekg(0, std::ios::end);
    std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    int n = static_cast<int>(file_size / ((d + 1) * sizeof(float)));
    FvecsData result;
    result.d = d;
    result.n = n;
    result.data.resize(static_cast<size_t>(n) * d);
    for (int i = 0; i < n; i++) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        in.read(reinterpret_cast<char*>(
            result.data.data() + static_cast<size_t>(i) * d),
            d * sizeof(float));
    }
    return result;
}

struct IvecsData {
    int d = 0;
    int n = 0;
    std::vector<int> data;
};

IvecsData load_ivecs(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open: " + filename);
    int d = 0;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    in.seekg(0, std::ios::end);
    std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    int n = static_cast<int>(file_size / ((d + 1) * sizeof(int)));
    IvecsData result;
    result.d = d;
    result.n = n;
    result.data.resize(static_cast<size_t>(n) * d);
    for (int i = 0; i < n; i++) {
        int dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int));
        in.read(reinterpret_cast<char*>(
            result.data.data() + static_cast<size_t>(i) * d),
            d * sizeof(int));
    }
    return result;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::printf("Usage: %s <nlist>\n", argv[0]);
            return 1;
        }

        const int nlist = std::atoi(argv[1]);

        FvecsData learn = load_fvecs("sift/sift_learn.fvecs");
        FvecsData base = load_fvecs("sift/sift_base.fvecs");
        FvecsData query = load_fvecs("sift/sift_query.fvecs");
        IvecsData gt = load_ivecs("sift/sift_groundtruth.ivecs");

        const int d = base.d;
        const int nq = query.n;
        const int recall_k = 10;

        std::printf("Dataset: %d base vectors, %d queries, dim=%d\n",
            base.n, nq, d);
        std::printf("Ground truth: %d queries x %d neighbors\n",
            gt.n, gt.d);

        // Build index
        faiss::IndexFlatL2 quantizer(d);
        faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);
        index.train(learn.n, learn.data.data());
        index.add(base.n, base.data.data());

        std::printf("\nnlist=%d, index built with %ld vectors\n\n",
            nlist, index.ntotal);

        // Allocate search results
        std::vector<float> distances(static_cast<size_t>(recall_k) * nq);
        std::vector<faiss::idx_t> labels(static_cast<size_t>(recall_k) * nq);

        // Sweep nprobe values
        std::printf("nprobe,recall@%d,lists_scanned_pct\n", recall_k);

        std::vector<int> nprobe_values = {
            1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50,
            60, 80, 100, 120, 150, 200, 250, 300, 400, 500
        };

        for (int test_nprobe : nprobe_values) {
            if (test_nprobe > nlist) break;

            index.nprobe = test_nprobe;
            index.search(nq, query.data.data(), recall_k,
                         distances.data(), labels.data());

            // Compute recall@k
            double recall = 0.0;
            for (int i = 0; i < nq; i++) {
                int hits = 0;
                for (int j = 0; j < recall_k; j++) {
                    faiss::idx_t returned_id = labels[
                        static_cast<size_t>(i) * recall_k + j];
                    for (int g = 0; g < recall_k && g < gt.d; g++) {
                        int gt_id = gt.data[
                            static_cast<size_t>(i) * gt.d + g];
                        if (returned_id == static_cast<faiss::idx_t>(gt_id)) {
                            hits++;
                            break;
                        }
                    }
                }
                recall += static_cast<double>(hits) / recall_k;
            }
            recall /= nq;

            std::printf("%4d,%.4f,%.2f%%\n",
                test_nprobe, recall,
                100.0 * test_nprobe / nlist);

            // Early stop if we've reached 99% recall
            if (recall > 0.99) break;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
