// RUN: %host_cxx -std=c++17 -O2 -x c++ -D__DAV_C310__ \
// RUN:   -Wno-macro-redefined -Wno-return-type-c-linkage \
// RUN:   -Wno-unknown-attributes \
// RUN:   -I%bishengir_src_root -I%bishengir_src_root/lib/Template/include \
// RUN:   %s -o %t
// RUN: %t > %t.out
// RUN: FileCheck %s --match-full-lines < %t.out

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#define __ubuf__
#define __gm__
#define __aicore__
#define __aiv__
#define __simt_vf__
#define __simt_callee__
#define LAUNCH_BOUND(x)

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
#include "lib/Template/lib/RegBase/Vector/Sin.cpp"
#include "lib/Template/lib/RegBase/Vector/SinCos.cpp"

static uint32_t gTable[320];
static int gFailures = 0;

static void report(bool ok, const char *name) {
  std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
  if (!ok)
    ++gFailures;
}

static double referenceSin(float x) {
  if (std::isnan(x) || std::isinf(x))
    return std::numeric_limits<double>::quiet_NaN();
  return std::sin(static_cast<double>(x));
}

static double referenceCos(float x) {
  if (std::isnan(x) || std::isinf(x))
    return std::numeric_limits<double>::quiet_NaN();
  return std::cos(static_cast<double>(x));
}

// Bit-exact float comparison: the fused kernel is supposed to be the same
// arithmetic as the per-op kernels, so any ULP difference is a real finding
// (and a tolerance would hide exactly the kind of drift this test exists for).
// NaN is compared as "both NaN" because the payload is not part of the
// contract.
static bool sameBits(float lhs, float rhs) {
  if (std::isnan(lhs) && std::isnan(rhs))
    return true;
  uint32_t lhsBits = 0;
  uint32_t rhsBits = 0;
  std::memcpy(&lhsBits, &lhs, sizeof(lhsBits));
  std::memcpy(&rhsBits, &rhs, sizeof(rhsBits));
  return lhsBits == rhsBits;
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
static void checkMemRef(const char *name, const float *src, const float *dstSin,
                        const float *dstCos, const int64_t *sizes,
                        const int64_t *srcStrides, const int64_t *sinStrides,
                        const int64_t *cosStrides, int64_t offset,
                        int64_t total) {
  bool ok = true;
  for (int64_t i = 0; i < total; ++i) {
    int64_t linear = i;
    int64_t srcOffset = offset;
    int64_t sinOffset = offset;
    int64_t cosOffset = offset;
    for (int dim = Rank - 1; dim >= 0; --dim) {
      int64_t index = linear % sizes[dim];
      linear /= sizes[dim];
      srcOffset += index * srcStrides[dim];
      sinOffset += index * sinStrides[dim];
      cosOffset += index * cosStrides[dim];
    }
    constexpr double kTolerance = 1e-5;
    double refSin = referenceSin(src[srcOffset]);
    double refCos = referenceCos(src[srcOffset]);
    ok &= std::fabs(static_cast<double>(dstSin[sinOffset]) - refSin) <=
          kTolerance;
    ok &= std::fabs(static_cast<double>(dstCos[cosOffset]) - refCos) <=
          kTolerance;
  }
  report(ok, name);
}

int sincos_simt_host() {
  for (int i = 0; i < 320; ++i)
    gTable[i] = SIMT_TWO_OVER_PI_TABLE[i];

  std::printf("max-rank: %d\n", SINCOS_SIMT_MAX_RANK);
  report(SINCOS_SIMT_MAX_RANK == 5, "max-rank-default");
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
  bool smallOk = true;
  for (float x : small) {
    float s = 0.0f;
    float c = 0.0f;
    simt_sincos_callee<float>(x, gTable, &s, &c);
    double refSin = referenceSin(x);
    double refCos = referenceCos(x);
    smallOk &= std::fabs(static_cast<double>(s) - refSin) <= 1e-5;
    smallOk &= std::fabs(static_cast<double>(c) - refCos) <= 1e-5;
  }
  report(smallOk, "callee-small");

  // Large angles: Payne-Hanek path (strictly above SIN_COS_THRESHOLD),
  // including the largest finite float.
  const float large[] = {71476.0703f, 1e6f,           -1e6f,
                         1e10f,       1e20f,          1e30f,
                         -1e30f,      3.40282347e38f, 123456789.0f};
  bool largeOk = true;
  for (float x : large) {
    float s = 0.0f;
    float c = 0.0f;
    simt_sincos_callee<float>(x, gTable, &s, &c);
    double refSin = referenceSin(x);
    double refCos = referenceCos(x);
    largeOk &= std::fabs(static_cast<double>(s) - refSin) <= 1e-5;
    largeOk &= std::fabs(static_cast<double>(c) - refCos) <= 1e-5;
  }
  report(largeOk, "callee-large");

  // One shared range reduction must be a pure win, not a different numerical
  // result: the fused kernel has to return exactly the bits the per-op kernels
  // return for both branches. This is the invariant that lets the compiler
  // replace a math.sin/math.cos pair with a single sincos call.
  {
    bool sameOk = true;
    auto checkSame = [&](float x) {
      float s = 0.0f;
      float c = 0.0f;
      simt_sincos_callee<float>(x, gTable, &s, &c);
      sameOk &= sameBits(s, simt_sin_callee<float>(x, gTable));
      sameOk &= sameBits(c, simt_cos_callee<float>(x, gTable));
    };
    // Dense sweep entirely on the Cody-Waite side.
    for (int i = -2000; i <= 2000; ++i)
      checkSame(static_cast<float>(i) * 0.05f);
    // Geometric sweep across the Payne-Hanek threshold and far beyond it.
    for (int i = 0; i < 200; ++i) {
      const float magnitude =
          std::pow(10.0f, 1.0f + 0.145f * static_cast<float>(i));
      checkSame(magnitude);
      checkSame(-magnitude);
    }
    // Exactly at the branch boundary and one ULP past it, both signs.
    checkSame(SIN_COS_THRESHOLD);
    checkSame(std::nextafter(SIN_COS_THRESHOLD, 1e9f));
    checkSame(-SIN_COS_THRESHOLD);
    checkSame(-std::nextafter(SIN_COS_THRESHOLD, 1e9f));
    report(sameOk, "sincos-matches-sin-cos");
  }

  // Volatile keeps the special values from being folded at compile time, so
  // the expo == 0xff branch really executes.
  volatile float positiveInfinity = INFINITY;
  volatile float negativeInfinity = -INFINITY;
  volatile float quietNan = NAN;
  {
    float s = 0.0f;
    float c = 0.0f;
    simt_sincos_callee<float>(positiveInfinity, gTable, &s, &c);
    report(std::isnan(s) && std::isnan(c), "inf-is-nan");
  }
  {
    float s = 0.0f;
    float c = 0.0f;
    simt_sincos_callee<float>(negativeInfinity, gTable, &s, &c);
    report(std::isnan(s) && std::isnan(c), "negative-inf-is-nan");
  }
  {
    float s = 0.0f;
    float c = 0.0f;
    simt_sincos_callee<float>(quietNan, gTable, &s, &c);
    report(std::isnan(s) && std::isnan(c), "nan-is-nan");
  }

  memref_t<int32_t, 1> table{reinterpret_cast<int32_t *>(gTable),
                             reinterpret_cast<int32_t *>(gTable),
                             0,
                             {320},
                             {1}};

  // Rank 1 fast path (contiguous).
  {
    float src[6] = {0.0f, 0.5f, 1.0f, 1e30f, -1e30f, -3.1415927f};
    float dstSin[6] = {};
    float dstCos[6] = {};
    int64_t sizes[1] = {6};
    int64_t strides[1] = {1};
    memref_t<float, 1> s{src, src, 0, {6}, {1}};
    memref_t<float, 1> dS{dstSin, dstSin, 0, {6}, {1}};
    memref_t<float, 1> dC{dstCos, dstCos, 0, {6}, {1}};
    _mlir_ciface_sincos_fp32_simt_1d(&s, &table, &dS, &dC);
    checkMemRef<1>("rank1", src, dstSin, dstCos, sizes, strides, strides,
                   strides, 0, 6);
  }

  // Rank 1 with non-unit strides: the 1-D fast path takes the cheap route (no
  // index decomposition), so it has to read strides[0] instead of assuming the
  // view is contiguous.
  {
    float src[12] = {};
    float dstSin[12] = {};
    float dstCos[12] = {};
    for (int i = 0; i < 6; ++i)
      src[i * 2] = static_cast<float>(i) * 0.5f;
    int64_t sizes[1] = {6};
    int64_t srcStrides[1] = {2};
    int64_t strides[1] = {1};
    int64_t cosStrides[1] = {2};
    memref_t<float, 1> s{src, src, 0, {6}, {2}};
    memref_t<float, 1> dS{dstSin, dstSin, 0, {6}, {1}};
    memref_t<float, 1> dC{dstCos, dstCos, 0, {6}, {2}};
    _mlir_ciface_sincos_fp32_simt_1d(&s, &table, &dS, &dC);
    checkMemRef<1>("rank1-strided", src, dstSin, dstCos, sizes, srcStrides,
                   strides, cosStrides, 0, 6);
  }

  // Rank 2, contiguous.
  {
    float src[6] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
    float dstSin[6] = {};
    float dstCos[6] = {};
    int64_t sizes[2] = {2, 3};
    int64_t strides[2] = {3, 1};
    memref_t<float, 2> s{src, src, 0, {2, 3}, {3, 1}};
    memref_t<float, 2> dS{dstSin, dstSin, 0, {2, 3}, {3, 1}};
    memref_t<float, 2> dC{dstCos, dstCos, 0, {2, 3}, {3, 1}};
    _mlir_ciface_sincos_fp32_simt_2d(&s, &table, &dS, &dC);
    checkMemRef<2>("rank2", src, dstSin, dstCos, sizes, strides, strides,
                   strides, 0, 6);
  }

  // Rank 2, transposed source and independent destination views.
  {
    float src[6] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
    float dstSin[6] = {};
    float dstCos[6] = {};
    int64_t sizes[2] = {2, 3};
    int64_t srcStrides[2] = {1, 2};
    int64_t sinStrides[2] = {3, 1};
    int64_t cosStrides[2] = {1, 2};
    memref_t<float, 2> s{src, src, 0, {2, 3}, {1, 2}};
    memref_t<float, 2> dS{dstSin, dstSin, 0, {2, 3}, {3, 1}};
    memref_t<float, 2> dC{dstCos, dstCos, 0, {2, 3}, {1, 2}};
    _mlir_ciface_sincos_fp32_simt_2d(&s, &table, &dS, &dC);
    checkMemRef<2>("rank2-strided", src, dstSin, dstCos, sizes, srcStrides,
                   sinStrides, cosStrides, 0, 6);
  }

  // Rank 3 with a non-zero base offset and a non-contiguous source.
  {
    float src[32] = {};
    float dstSin[32] = {};
    float dstCos[32] = {};
    for (int i = 0; i < 32; ++i)
      src[i] = static_cast<float>(i) * 0.5f;
    int64_t sizes[3] = {2, 2, 2};
    int64_t strides[3] = {4, 2, 1};
    memref_t<float, 3> s{src, src, 2, {2, 2, 2}, {4, 2, 1}};
    memref_t<float, 3> dS{dstSin, dstSin, 2, {2, 2, 2}, {4, 2, 1}};
    memref_t<float, 3> dC{dstCos, dstCos, 2, {2, 2, 2}, {4, 2, 1}};
    _mlir_ciface_sincos_fp32_simt_3d(&s, &table, &dS, &dC);
    checkMemRef<3>("rank3-offset", src, dstSin, dstCos, sizes, strides, strides,
                   strides, 2, 8);
  }

  // Rank 4, contiguous.
  {
    float src[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    float dstSin[8] = {};
    float dstCos[8] = {};
    int64_t sizes[4] = {2, 2, 1, 2};
    int64_t strides[4] = {4, 2, 2, 1};
    memref_t<float, 4> s{src, src, 0, {2, 2, 1, 2}, {4, 2, 2, 1}};
    memref_t<float, 4> dS{dstSin, dstSin, 0, {2, 2, 1, 2}, {4, 2, 2, 1}};
    memref_t<float, 4> dC{dstCos, dstCos, 0, {2, 2, 1, 2}, {4, 2, 2, 1}};
    _mlir_ciface_sincos_fp32_simt_4d(&s, &table, &dS, &dC);
    checkMemRef<4>("rank4", src, dstSin, dstCos, sizes, strides, strides,
                   strides, 0, 8);
  }

  // Rank 5, the last rank with a template-library entry point.
  {
    float src[4] = {0.5f, 1e30f, -1e30f, 3.1415927f};
    float dstSin[4] = {};
    float dstCos[4] = {};
    int64_t sizes[5] = {2, 1, 1, 1, 2};
    int64_t strides[5] = {2, 2, 2, 2, 1};
    memref_t<float, 5> s{src, src, 0, {2, 1, 1, 1, 2}, {2, 2, 2, 2, 1}};
    memref_t<float, 5> dS{dstSin, dstSin, 0, {2, 1, 1, 1, 2}, {2, 2, 2, 2, 1}};
    memref_t<float, 5> dC{dstCos, dstCos, 0, {2, 1, 1, 1, 2}, {2, 2, 2, 2, 1}};
    _mlir_ciface_sincos_fp32_simt_5d(&s, &table, &dS, &dC);
    checkMemRef<5>("rank5", src, dstSin, dstCos, sizes, strides, strides,
                   strides, 0, 4);
  }

  std::printf("failures: %d\n", gFailures);
  return gFailures == 0 ? 0 : 1;
}

int main() { return sincos_simt_host(); }

// CHECK: max-rank: 5
// CHECK: max-rank-default: PASS
// CHECK: table-length: 320
// CHECK: table-checksum: PASS
// CHECK: table-first-limb: PASS
// CHECK: table-tail: PASS
// CHECK: callee-small: PASS
// CHECK: callee-large: PASS
// CHECK: sincos-matches-sin-cos: PASS
// CHECK: inf-is-nan: PASS
// CHECK: negative-inf-is-nan: PASS
// CHECK: nan-is-nan: PASS
// CHECK: rank1: PASS
// CHECK: rank1-strided: PASS
// CHECK: rank2: PASS
// CHECK: rank2-strided: PASS
// CHECK: rank3-offset: PASS
// CHECK: rank4: PASS
// CHECK: rank5: PASS
// CHECK: failures: 0
