// Standalone SM70 Skinny kernel harness.
//
// Compiles the format policies + SIMT/QPN cores against plain CUDA (no torch)
// so the kernels can be validated numerically against a host FP64 reference
// and timed on a real V100 without building all of vLLM.
//
// Build twice, once per variant:
//   nvcc -O3 -arch=sm_70 -I base -DHARNESS_BASELINE=1 harness.cu -o
//   harness_base nvcc -O3 -arch=sm_70 -I new                       harness.cu
//   -o harness_new

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "sm70_skinny/formats/awq.cuh"
#include "sm70_skinny/formats/nvfp4.cuh"
#include "sm70_skinny/qpn.cuh"
#include "sm70_skinny/simt.cuh"

using namespace vllm::sm70_skinny;

#define CUDA_OK(expr)                                                     \
  do {                                                                    \
    cudaError_t status = (expr);                                          \
    if (status != cudaSuccess) {                                          \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(status), \
                  __FILE__, __LINE__);                                    \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// ---------------------------------------------------------------------------
// Host-side layout helpers (mirror the Python prepack)
// ---------------------------------------------------------------------------

static int qpn_col(int lane) {
  return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

static const int kQpnKOrder[16] = {0, 2,  4,  6,  1, 3,  5,  7,
                                   8, 10, 12, 14, 9, 11, 13, 15};

// codes[n][k] logical nibbles -> fragment-order [tiles][groups][32][8] bytes
static std::vector<uint8_t> qpn_prepack_codes(const std::vector<uint8_t>& nib,
                                              int n, int k) {
  const int tiles = n / 32, groups = k / 16;
  std::vector<uint8_t> out((size_t)tiles * groups * 32 * 8);
  for (int t = 0; t < tiles; ++t) {
    for (int g = 0; g < groups; ++g) {
      for (int lane = 0; lane < 32; ++lane) {
        const int row = t * 32 + qpn_col(lane);
        for (int b = 0; b < 8; ++b) {
          const int lo = nib[(size_t)row * k + g * 16 + kQpnKOrder[2 * b]];
          const int hi = nib[(size_t)row * k + g * 16 + kQpnKOrder[2 * b + 1]];
          out[((((size_t)t * groups + g) * 32) + lane) * 8 + b] =
              (uint8_t)(lo | (hi << 4));
        }
      }
    }
  }
  return out;
}

// meta[n][mgroups] -> [tiles][mgroups][32]
template <typename T>
static std::vector<T> qpn_prepack_meta(const std::vector<T>& meta, int n,
                                       int mgroups) {
  const int tiles = n / 32;
  std::vector<T> out((size_t)tiles * mgroups * 32);
  for (int t = 0; t < tiles; ++t) {
    for (int g = 0; g < mgroups; ++g) {
      for (int lane = 0; lane < 32; ++lane) {
        out[(((size_t)t * mgroups + g) * 32) + lane] =
            meta[(size_t)(t * 32 + qpn_col(lane)) * mgroups + g];
      }
    }
  }
  return out;
}

static float fp8e4m3_to_float_host(uint8_t v) {
  const int sign = (v & 0x80) ? -1 : 1;
  const int exp = (v >> 3) & 0xF;
  const int man = v & 0x7;
  if (exp == 0) return sign * std::ldexp((float)man, -9);
  return sign * std::ldexp(1.0f + man / 8.0f, exp - 7);
}

static const float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ---------------------------------------------------------------------------
// Launch shims - hide the API difference between baseline and new
// ---------------------------------------------------------------------------

// HARNESS_BASELINE  -> original simt + original qpn
// HARNESS_OLD_SIMT  -> original simt + new (split-K) qpn
// neither           -> new simt + new qpn
#if HARNESS_BASELINE
  #define HARNESS_OLD_SIMT 1
#endif

template <typename Policy, int M>
static void run_simt(typename Policy::Params params, const half* x, half* y,
                     int n, int k, bool two_rows, cudaStream_t stream) {
  constexpr int KC = 1024;
#if HARNESS_OLD_SIMT
  const dim3 grid(two_rows ? n / 16 : n / 8), block(256);
  const int smem = M * (KC / 2) * sizeof(half2);
  if (two_rows)
    simt_kernel<Policy, M, KC, 2>
        <<<grid, block, smem, stream>>>(params, x, y, n, k, nullptr, nullptr);
  else
    simt_kernel<Policy, M, KC, 1>
        <<<grid, block, smem, stream>>>(params, x, y, n, k, nullptr, nullptr);
#else
  const int smem = simt_shared_bytes(M, KC);
  launch_simt_kernel<Policy, M, KC>(params, x, y, n, k, two_rows, smem, stream);
#endif
}

template <typename Policy, int MT>
static void run_qpn(typename Policy::Params params, const half* x, half* y,
                    float* ws, int n, int k, int rows, int k_splits,
                    cudaStream_t stream) {
#if HARNESS_BASELINE
  (void)ws;
  (void)k_splits;
  qpn_kernel<Policy, MT>
      <<<dim3(n / 32), dim3(128), 0, stream>>>(params, x, y, n, k, rows);
#else
  launch_qpn_kernel<Policy, MT>(params, x, y, ws, n, k, rows, k_splits, stream);
#endif
}

static int choose_splits(int k, int n) {
#if HARNESS_BASELINE
  (void)k;
  (void)n;
  return 1;
#else
  // SKINNY_SPLITS=<n> forces the split factor for sweeps; 0 = use the policy.
  const char* forced = std::getenv("SKINNY_SPLITS");
  if (forced && std::atoi(forced) > 0) {
    const int want = std::atoi(forced);
    const int group_count = k / 16;
    // Fall back to 1 when the forced value would drop K16 groups.
    return (group_count % (4 * want) == 0) ? want : 1;
  }
  return qpn_choose_k_splits(k, n, kQpnTargetBlocks);
#endif
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

struct Result {
  double max_rel;
  double ms;
  int splits;
};

static double time_ms(const std::function<void(int)>& fn, int iters) {
  cudaEvent_t a, b;
  CUDA_OK(cudaEventCreate(&a));
  CUDA_OK(cudaEventCreate(&b));
  for (int i = 0; i < 5; ++i) fn(i);
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) fn(i);
  CUDA_OK(cudaEventRecord(b));
  CUDA_OK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_OK(cudaEventElapsedTime(&ms, a, b));
  CUDA_OK(cudaEventDestroy(a));
  CUDA_OK(cudaEventDestroy(b));
  return ms / iters;
}

// Reference is computed for a window of columns only; the full N x K host
// GEMM would dominate runtime and the fragment/tile mapping repeats every 32
// columns anyway. Checking both the first and last window catches tile-index
// errors as well as intra-tile permutation errors.
static const int kCheckCols = 256;

static double check_window(const std::vector<half>& got,
                           const std::vector<uint8_t>& nib,
                           const std::vector<half>& scales,
                           const std::vector<half>& biases,
                           const std::vector<half>& x, int m, int n, int k,
                           int groups, int col0, int cols) {
  double denom = 0.0, worst = 0.0;
  for (int row = 0; row < m; ++row) {
    for (int c = 0; c < cols; ++c) {
      const int col = col0 + c;
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        const double w =
            (double)nib[(size_t)col * k + kk] *
                (double)__half2float(scales[(size_t)col * groups + kk / 128]) +
            (double)__half2float(biases[(size_t)col * groups + kk / 128]);
        acc += (double)__half2float(x[(size_t)row * k + kk]) * w;
      }
      denom = std::max(denom, std::fabs(acc));
      worst = std::max(
          worst,
          std::fabs((double)__half2float(got[(size_t)row * n + col]) - acc));
    }
  }
  if (denom < 1e-9) denom = 1e-9;
  return worst / denom;
}

int main(int argc, char** argv) {
  int only_m = argc > 1 ? std::atoi(argv[1]) : 0;
  std::mt19937 rng(1234);

  struct Shape {
    const char* name;
    int n, k;
  };
  // Per-rank shapes for a Qwen-class 27B under TP4.
  const std::vector<Shape> shapes = {
      {"o_proj   ", 5120, 1280},
      {"qkv      ", 1792, 5120},
      {"down     ", 5120, 4352},
      {"gate_up  ", 8704, 5120},
  };

  std::printf("%-10s %-6s %-5s %-4s %10s %12s %8s\n", "shape", "kern", "M",
              "spl", "max_rel", "ms", "GB/s");

  for (const Shape& s : shapes) {
    const int n = s.n, k = s.k;
    const int groups = k / 128;

    // ---- build AWQ g128 data ----
    std::vector<uint8_t> nib((size_t)n * k);
    std::vector<half> scales((size_t)n * groups), biases((size_t)n * groups);
    std::uniform_int_distribution<int> qd(0, 15);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (size_t i = 0; i < nib.size(); ++i) nib[i] = (uint8_t)qd(rng);
    for (size_t i = 0; i < scales.size(); ++i) {
      const float sc = sd(rng);
      const float zp = (float)qd(rng);
      scales[i] = __float2half(sc);
      biases[i] = __float2half(-zp * sc);
    }
    std::vector<uint8_t> codes((size_t)n * k / 2);
    for (size_t i = 0; i < codes.size(); ++i)
      codes[i] = (uint8_t)(nib[2 * i] | (nib[2 * i + 1] << 4));

    // ---- activation ----
    const int max_m = 16;
    std::vector<half> x((size_t)max_m * k);
    std::uniform_real_distribution<float> xd(-0.05f, 0.05f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = __float2half(xd(rng));

    // ---- device buffers ----
    // Real decode walks every layer before returning to this one, so a layer's
    // weights are never still resident in the 6 MB L2 on the next step. A
    // benchmark that hammers one buffer would give small shapes free L2 hits
    // and wrongly reward (or punish) cache-policy changes such as __ldcs.
    // Rotate over enough distinct copies to blow past L2 instead.
    const size_t weight_bytes = codes.size() + scales.size() * 4;
    const int kRotate = (int)std::max<size_t>(
        2, (24u << 20) / std::max<size_t>(1, weight_bytes));
    std::vector<uint8_t*> d_codes(kRotate), d_qcodes(kRotate);
    std::vector<half*> d_scales(kRotate), d_biases(kRotate);
    std::vector<half*> d_qscales(kRotate), d_qbiases(kRotate);
    half *d_x, *d_y;
    float* d_ws;
    const auto qc = qpn_prepack_codes(nib, n, k);
    const auto qs = qpn_prepack_meta(scales, n, groups);
    const auto qb = qpn_prepack_meta(biases, n, groups);
    for (int r = 0; r < kRotate; ++r) {
      CUDA_OK(cudaMalloc(&d_codes[r], codes.size()));
      CUDA_OK(cudaMalloc(&d_qcodes[r], qc.size()));
      CUDA_OK(cudaMalloc(&d_scales[r], scales.size() * 2));
      CUDA_OK(cudaMalloc(&d_biases[r], biases.size() * 2));
      CUDA_OK(cudaMalloc(&d_qscales[r], qs.size() * 2));
      CUDA_OK(cudaMalloc(&d_qbiases[r], qb.size() * 2));
      CUDA_OK(cudaMemcpy(d_codes[r], codes.data(), codes.size(),
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_qcodes[r], qc.data(), qc.size(),
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_scales[r], scales.data(), scales.size() * 2,
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_biases[r], biases.data(), biases.size() * 2,
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_qscales[r], qs.data(), qs.size() * 2,
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_qbiases[r], qb.data(), qb.size() * 2,
                         cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaMalloc(&d_x, x.size() * 2));
    CUDA_OK(cudaMalloc(&d_y, (size_t)max_m * n * 2));
    CUDA_OK(cudaMalloc(&d_ws, (size_t)16 * max_m * n * 4));
    CUDA_OK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));

    auto sp_at = [&](int i) {
      return AwqG128SimtPolicy::Params{d_codes[i % kRotate],
                                       d_scales[i % kRotate],
                                       d_biases[i % kRotate], k};
    };
    auto qp_at = [&](int i) {
      return AwqG128QpnPolicy::Params{d_qcodes[i % kRotate],
                                      d_qscales[i % kRotate],
                                      d_qbiases[i % kRotate], k / 16};
    };

    const double weight_gb = (double)n * k * 0.53125 / 1e9;

    for (int m : {1, 2, 3, 4, 8, 16}) {
      if (only_m && m != only_m) continue;
      std::vector<half> got((size_t)m * n);
      std::function<void(int)> launch;
      const char* kern;
      int splits = 1;
      if (m <= 3) {
        kern = "simt";
        const bool two_rows = (k <= 2048) && (n % 16 == 0);
        if (m == 1)
          launch = [&](int i) {
            run_simt<AwqG128SimtPolicy, 1>(sp_at(i), d_x, d_y, n, k, two_rows,
                                           0);
          };
        else if (m == 2)
          launch = [&](int i) {
            run_simt<AwqG128SimtPolicy, 2>(sp_at(i), d_x, d_y, n, k, two_rows,
                                           0);
          };
        else
          launch = [&](int i) {
            run_simt<AwqG128SimtPolicy, 3>(sp_at(i), d_x, d_y, n, k, two_rows,
                                           0);
          };
      } else {
        kern = "qpn ";
        splits = choose_splits(k, n);
        if (m <= 8)
          launch = [&](int i) {
            run_qpn<AwqG128QpnPolicy, 1>(qp_at(i), d_x, d_y, d_ws, n, k, m,
                                         splits, 0);
          };
        else
          launch = [&](int i) {
            run_qpn<AwqG128QpnPolicy, 2>(qp_at(i), d_x, d_y, d_ws, n, k, m,
                                         splits, 0);
          };
      }

      launch(0);
      CUDA_OK(cudaDeviceSynchronize());
      CUDA_OK(cudaGetLastError());
      CUDA_OK(
          cudaMemcpy(got.data(), d_y, got.size() * 2, cudaMemcpyDeviceToHost));
      const int cols = std::min(kCheckCols, n);
      const double err = std::max(
          check_window(got, nib, scales, biases, x, m, n, k, groups, 0, cols),
          check_window(got, nib, scales, biases, x, m, n, k, groups, n - cols,
                       cols));
      const double ms = time_ms(launch, 200);
      std::printf("%-10s %-6s %-5d %-4d %10.3e %12.4f %8.1f\n", s.name, kern, m,
                  splits, err, ms, weight_gb / (ms / 1000.0));
      if (err > 3e-2) std::printf("   ^^ FAIL: exceeds 3e-2\n");
    }

    for (int r = 0; r < kRotate; ++r) {
      cudaFree(d_codes[r]);
      cudaFree(d_qcodes[r]);
      cudaFree(d_scales[r]);
      cudaFree(d_biases[r]);
      cudaFree(d_qscales[r]);
      cudaFree(d_qbiases[r]);
    }
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_ws);
  }
  return 0;
}
