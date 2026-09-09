# cxl_test_app

Standalone test programs for the DRAM/CXL tiered inverted-list feature.

## Build

Build faiss normally first (see `INSTALL.md`), then compile a test app
against it, e.g. (run from the parent of the `faiss` checkout):

```
g++ -O2 faiss/cxl_test_app/ivf_flat_sift.cpp -o ivf_flat_sift \
    -I<path-to-faiss-checkout>/faiss \
    -L<path-to-faiss-checkout>/faiss/build/faiss \
    -lfaiss -lopenblas -fopenmp -lnuma
```

Swap in the `.cpp` file you want to build. `-lnuma` is required because
`TieredArrayInvertedLists` links against libnuma.

## Data

Download the SIFT datasets from http://corpus-texmex.irisa.fr/ and unzip
so that, relative to wherever you run the binary from:
- `sift/` contains `sift_learn.fvecs`, `sift_base.fvecs`, `sift_query.fvecs`,
  `sift_groundtruth.ivecs` (ANN_SIFT1M)
- `siftsmall/` contains the equivalent `siftsmall_*.fvecs` files (ANN_SIFT10K)

## Running

- `ivf_flat_sift <nlist> <nprobe> <hot_lists> <k> [epoch_count] [data_dir]` —
  tiered search over SIFT1M, reporting per-epoch/per-tier stats.
  `hot_lists` is the number of lists kept in DRAM, `k` is the number of
  nearest neighbors returned per query. `epoch_count` defaults to 5,
  `data_dir` defaults to `sift`.
- `ivf_flat_sift_small <nlist> <nprobe> <hot_lists> <k> [epoch_count] [data_dir]` —
  same experiment on the smaller SIFT10K dataset. Same params as
  `ivf_flat_sift`; `data_dir` defaults to `siftsmall`.
- `calibrate_nprobe <nlist>` — sweeps `nprobe` over SIFT1M and prints
  recall@10 vs. lists-scanned percentage, to help pick an `nprobe` value.
