#include <algorithm>
#include <cstdint>

// Exhaustive independent-permutation MinHash scan. No truth or prefilter input.
extern "C" void signature_scan(const uint64_t* query, const uint64_t* target,
                              int nq, int nt, int stride, int hashes,
                              int threads, int* output) {
  #pragma omp parallel for num_threads(threads) schedule(static)
  for (int t = 0; t < nt; ++t) {
    int best = 0;
    for (int q = 0; q < nq; ++q) {
      int equal = 0;
      #pragma omp simd reduction(+:equal)
      for (int h = 0; h < hashes; ++h)
        equal += query[(int64_t)q * stride + h] == target[(int64_t)t * stride + h];
      best = std::max(best, equal);
    }
    output[t] = best;
  }
}
