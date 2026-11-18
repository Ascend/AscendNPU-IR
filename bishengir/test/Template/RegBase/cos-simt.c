// RUN: %host_cxx -std=c++17 -O2 -x c++ -D__DAV_C310__ \
// RUN:   -Wno-macro-redefined -Wno-return-type-c-linkage \
// RUN:   -Wno-unknown-attributes \
// RUN:   -I%bishengir_src_root -I%bishengir_src_root/lib/Template/include \
// RUN:   %s -o %t
// RUN: %t > %t.out
// RUN: FileCheck %s --match-full-lines < %t.out
//
// Host-side exercise of the RegBase SIMT cos template library. The device-only
// pieces (cce::async_invoke, thread/block ids, UB pointers) are stubbed so the
// real Cos.cpp body runs on the host, covering:
//   * the shared SIMT_TRIG_MAX_RANK / 2/pi table,
//   * the Cody-Waite and Payne-Hanek range-reduction branches, including NaN
//     and infinity,
//   * the rank-specialized _mlir_ciface_cos_fp32_simt_<rank>d entries for
//     ranks 1..5, including strided views and non-zero offsets.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#define __ubuf__
#define __gm__
#define __aicore__
#define __aiv__
#define __simt_vf__
#define __simt_callee__
#define LAUNCH_BOUND(x)

// Device scalar types; the SIMT body only needs float conversion.
struct half {
  float v = 0.0f;
  half() = default;
  half(float x) : v(x) {}
  operator float() const { return v; }
};
struct bfloat16_t {
  float v = 0.0f;
  bfloat16_t() = default;
  bfloat16_t(float x) : v(x) {}
  operator float() const { return v; }
};

// Vector types referenced by VecUtils.h, unused by Cos.cpp itself.
struct float4_e1m2_t {};
struct float4_e2m1_t {};
struct float8_e4m3_t {};
struct float8_e5m2_t {};
struct hifloat4x2_t {};
struct hifloat8_t {};
struct vector_f32 {};
struct vector_f16 {};
struct vector_bf16 {};
struct vector_f8e4m3 {};
struct vector_f8e5m2 {};
struct vector_2xvl_s64 {};
struct vector_2xvl_u64 {};
struct vector_s32 {};
struct vector_u32 {};
struct vector_s16 {};
struct vector_u16 {};
struct vector_s8 {};
struct vector_u8 {};

// Minimal SIMT runtime: a single "thread" walks the whole range, which keeps
// the generated kernels' loops equivalent to a sequential reference.
struct ThreadIdx {
  int64_t x;
  int64_t y;
  int64_t z;
};
static ThreadIdx threadIdx{0, 0, 0};
static ThreadIdx blockDim{1, 1, 1};

namespace cce {
struct dim3 {
  int x;
  int y;
  int z;
};
template <auto Fn, typename... Args> void async_invoke(dim3, Args... args) {
  Fn(args...);
}
} // namespace cce

static inline float npuir_test_fma(float a, float b, float c) {
  return std::fma(a, b, c);
}
static inline float npuir_test_fabsf(float x) { return std::fabs(x); }

#ifdef __fma
#undef __fma
#endif
#ifdef __fabsf
#undef __fabsf
#endif
#define __fma(a, b, c) npuir_test_fma((a), (b), (c))
#define __fabsf(x) npuir_test_fabsf(x)

#include "RegBase/SIMTTrigCommon.h"

#include "lib/Template/lib/RegBase/Vector/Cos.cpp"

static uint32_t gTable[320];
static int gFailures = 0;

static void report(bool ok, const char *name) {
  std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
  if (!ok)
    ++gFailures;
}

static double referenceCos(float x) {
  if (std::isnan(x) || std::isinf(x))
    return std::numeric_limits<double>::quiet_NaN();
  return std::cos(static_cast<double>(x));
}

static uint32_t tableChecksum() {
  uint32_t hash = 2166136261u;
  for (int i = 0; i < 320; ++i)
    for (int byte = 0; byte < 4; ++byte) {
      hash ^= (SIMT_TWO_OVER_PI_TABLE[i] >> (8 * byte)) & 0xffu;
      hash *= 16777619u;
    }
  return hash;
}

template <int Rank>
static void checkMemRef(const char *name, const float *src, const float *dst,
                        const int64_t *sizes, const int64_t *srcStrides,
                        const int64_t *dstStrides, int64_t offset,
                        int64_t total) {
  bool ok = true;
  for (int64_t i = 0; i < total; ++i) {
    int64_t linear = i;
    int64_t srcOffset = offset;
    int64_t dstOffset = offset;
    for (int dim = Rank - 1; dim >= 0; --dim) {
      int64_t index = linear % sizes[dim];
      linear /= sizes[dim];
      srcOffset += index * srcStrides[dim];
      dstOffset += index * dstStrides[dim];
    }
    constexpr double kTolerance = 1e-5;
    double ref = referenceCos(src[srcOffset]);
    ok &= std::fabs(static_cast<double>(dst[dstOffset]) - ref) <= kTolerance;
  }
  report(ok, name);
}

int cos_simt_host() {
  for (int i = 0; i < 320; ++i)
    gTable[i] = SIMT_TWO_OVER_PI_TABLE[i];

  std::printf("max-rank: %d\n", COS_SIMT_MAX_RANK);
  report(COS_SIMT_MAX_RANK == 5, "max-rank-default");
  std::printf("table-length: %d\n",
              static_cast<int>(sizeof(SIMT_TWO_OVER_PI_TABLE) /
                               sizeof(SIMT_TWO_OVER_PI_TABLE[0])));
  report(tableChecksum() == 0xddce4aa6u, "table-checksum");
  report(SIMT_TWO_OVER_PI_TABLE[129] == 0x00000001u, "table-first-limb");
  report(SIMT_TWO_OVER_PI_TABLE[319] == 0x9e21c820u, "table-tail");

  // Small and threshold angles: Cody-Waite path. The branch test is strict
  // (>), so the exact SIN_COS_THRESHOLD value stays on this path.
  const float small[] = {0.0f,       1e-8f,      0.25f,      0.5f,        1.0f,
                         1.5707963f, 3.1415927f, -1.0f,      -3.1415927f, 10.0f,
                         1000.0f,    71476.0f,   71476.0625f};
  bool constSmall = true;
  bool entrySmall = true;
  bool calleeSmall = true;
  for (float x : small) {
    double ref = referenceCos(x);
    float a = simt_cos<float>(x);
    float b = _mlir_ciface_simt_cos_float(x);
    float c = simt_cos_callee<float>(x, gTable);
    constSmall &= std::fabs(static_cast<double>(a) - ref) <= 1e-5;
    entrySmall &= std::fabs(static_cast<double>(b) - ref) <= 1e-5;
    calleeSmall &= std::fabs(static_cast<double>(c) - ref) <= 1e-5;
  }
  report(constSmall, "keep-const-small");
  report(entrySmall, "keep-entry-small");
  report(calleeSmall, "keep-callee-small");

  // Large angles: Payne-Hanek path (strictly above SIN_COS_THRESHOLD),
  // including the largest finite float. Only the table-driven callee and the
  // vector entries use the compiler-style reduction; the legacy scalar path
  // keeps its own embedded table and is not checked here.
  const float large[] = {71476.0703f, 1e6f,           -1e6f,
                         1e10f,       1e20f,          1e30f,
                         -1e30f,      3.40282347e38f, 123456789.0f};
  bool calleeLarge = true;
  for (float x : large) {
    double ref = referenceCos(x);
    float c = simt_cos_callee<float>(x, gTable);
    calleeLarge &= std::fabs(static_cast<double>(c) - ref) <= 1e-5;
  }
  report(calleeLarge, "keep-callee-large");

  // Volatile keeps the special values from being folded at compile time, so
  // the expo == 0xff branch really executes.
  volatile float positiveInfinity = INFINITY;
  volatile float negativeInfinity = -INFINITY;
  volatile float quietNan = NAN;
  report(std::isnan(simt_cos_callee<float>(positiveInfinity, gTable)),
         "callee-inf-is-nan");
  report(std::isnan(simt_cos_callee<float>(negativeInfinity, gTable)),
         "callee-negative-inf-is-nan");
  report(std::isnan(simt_cos_callee<float>(quietNan, gTable)),
         "callee-nan-is-nan");

  memref_t<int32_t, 1> table{reinterpret_cast<int32_t *>(gTable),
                             reinterpret_cast<int32_t *>(gTable),
                             0,
                             {320},
                             {1}};

  // Rank 1 fast path (contiguous).
  {
    float src[6] = {0.0f, 0.5f, 1.0f, 1e30f, -1e30f, -3.1415927f};
    float dst[6] = {};
    int64_t sizes[1] = {6};
    int64_t strides[1] = {1};
    memref_t<float, 1> s{src, src, 0, {6}, {1}};
    memref_t<float, 1> d{dst, dst, 0, {6}, {1}};
    _mlir_ciface_cos_fp32_simt_1d(&s, &table, &d);
    checkMemRef<1>("rank1", src, dst, sizes, strides, strides, 0, 6);
  }

  // Rank 2, contiguous.
  {
    float src[6] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
    float dst[6] = {};
    int64_t sizes[2] = {2, 3};
    int64_t strides[2] = {3, 1};
    memref_t<float, 2> s{src, src, 0, {2, 3}, {3, 1}};
    memref_t<float, 2> d{dst, dst, 0, {2, 3}, {3, 1}};
    _mlir_ciface_cos_fp32_simt_2d(&s, &table, &d);
    checkMemRef<2>("rank2", src, dst, sizes, strides, strides, 0, 6);
  }

  // Rank 2, transposed source and transposed destination views.
  {
    float src[6] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
    float dst[6] = {};
    int64_t sizes[2] = {2, 3};
    int64_t srcStrides[2] = {1, 2};
    int64_t dstStrides[2] = {3, 1};
    memref_t<float, 2> s{src, src, 0, {2, 3}, {1, 2}};
    memref_t<float, 2> d{dst, dst, 0, {2, 3}, {3, 1}};
    _mlir_ciface_cos_fp32_simt_2d(&s, &table, &d);
    checkMemRef<2>("rank2-strided", src, dst, sizes, srcStrides, dstStrides, 0,
                   6);
  }

  // Rank 3 with a non-zero base offset and a non-contiguous source.
  {
    float src[32] = {};
    float dst[32] = {};
    for (int i = 0; i < 32; ++i)
      src[i] = static_cast<float>(i) * 0.5f;
    int64_t sizes[3] = {2, 2, 2};
    int64_t srcStrides[3] = {4, 2, 1};
    int64_t dstStrides[3] = {4, 2, 1};
    memref_t<float, 3> s{src, src, 2, {2, 2, 2}, {4, 2, 1}};
    memref_t<float, 3> d{dst, dst, 2, {2, 2, 2}, {4, 2, 1}};
    _mlir_ciface_cos_fp32_simt_3d(&s, &table, &d);
    checkMemRef<3>("rank3-offset", src, dst, sizes, srcStrides, dstStrides, 2,
                   8);
  }

  // Rank 4, contiguous.
  {
    float src[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    float dst[8] = {};
    int64_t sizes[4] = {2, 2, 1, 2};
    int64_t strides[4] = {4, 2, 2, 1};
    memref_t<float, 4> s{src, src, 0, {2, 2, 1, 2}, {4, 2, 2, 1}};
    memref_t<float, 4> d{dst, dst, 0, {2, 2, 1, 2}, {4, 2, 2, 1}};
    _mlir_ciface_cos_fp32_simt_4d(&s, &table, &d);
    checkMemRef<4>("rank4", src, dst, sizes, strides, strides, 0, 8);
  }

  // Rank 5, the last rank with a template-library entry point.
  {
    float src[4] = {0.5f, 1e30f, -1e30f, 3.1415927f};
    float dst[4] = {};
    int64_t sizes[5] = {2, 1, 1, 1, 2};
    int64_t strides[5] = {2, 2, 2, 2, 1};
    memref_t<float, 5> s{src, src, 0, {2, 1, 1, 1, 2}, {2, 2, 2, 2, 1}};
    memref_t<float, 5> d{dst, dst, 0, {2, 1, 1, 1, 2}, {2, 2, 2, 2, 1}};
    _mlir_ciface_cos_fp32_simt_5d(&s, &table, &d);
    checkMemRef<5>("rank5", src, dst, sizes, strides, strides, 0, 4);
  }

  std::printf("failures: %d\n", gFailures);
  return gFailures == 0 ? 0 : 1;
}

int main() { return cos_simt_host(); }

// CHECK: max-rank: 5
// CHECK: max-rank-default: PASS
// CHECK: table-length: 320
// CHECK: table-checksum: PASS
// CHECK: table-first-limb: PASS
// CHECK: table-tail: PASS
// CHECK: keep-const-small: PASS
// CHECK: keep-entry-small: PASS
// CHECK: keep-callee-small: PASS
// CHECK: keep-callee-large: PASS
// CHECK: callee-inf-is-nan: PASS
// CHECK: callee-negative-inf-is-nan: PASS
// CHECK: callee-nan-is-nan: PASS
// CHECK: rank1: PASS
// CHECK: rank2: PASS
// CHECK: rank2-strided: PASS
// CHECK: rank3-offset: PASS
// CHECK: rank4: PASS
// CHECK: rank5: PASS
// CHECK: failures: 0
