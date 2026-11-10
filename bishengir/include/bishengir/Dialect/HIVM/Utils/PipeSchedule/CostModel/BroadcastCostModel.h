//===- BroadcastCostModel.h - development-stage vbrc cost model --*- C++
//-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Standalone cost model for hivm.hir.vbrc (broadcast).  Wired into the
// production vector path: getVectorGroupCost's vbrc branch recognizes the
// five patterns from the op metadata (MsDagBuilder resolves the REAL,
// unpadded output shape into VectorOpDesc::vbrcRealOutShape) and calls the
// queries below; forms outside the five patterns keep the generic vector
// fallback billing.  The broadcast kernel itself is still under
// development, so the tables live in this separate file (1019) instead of
// MSCostParams.h — move them there when broadcast stabilizes.
//
// Source: v2 report §3 (AIV-path trace source `vector_median`, µs), as
// summarized in
// work_flow_ai/1019-cost-model-fix1/[info]broadcast_model_info.md.
//
// Five patterns (the shapes a vbrc takes in CV kernels; recognition
// criteria from the same doc — expand_shape form, broadcast_dims, rank):
//   Pattern    vbrc in        vbrc out             dims
//   Inner      (d0,1)         (d0, pad(d1))        bd=[1], src is a column
//   Outer      (1,d1)         (pad(d0), d1)        bd=[0], src is a row
//   Middle     (d0,1,d2)      (d0, pad(d1), d2)    bd=[1], src is a plane
//   Scalar1d   scalar         pad(N)               no bd
//   Scalar2d   scalar         pad(d0) x pad(d1)    no bd
// where pad(x) = 2^ceil(log2(x)).  The models are driven by the REAL
// (unpadded) dimensions — d1/N are read from the trailing extract_slice
// or the output reinterpret_cast strides, never from the padded outs
// shape.  Pattern recognition and dim extraction are the CALLER's job;
// the queries below take already-resolved real dims.
//
// Model per pattern (latency in µs, a = per-pattern intercept):
//   Inner:    lat = a + S(d1)·d0          (segmentation variable d1)
//   Outer:    lat = a + S(d1)·d0          (segmentation variable d1)
//   Middle:   lat = a + S(u)·u,  u = d0·d1  (d2 does not enter the model)
//   Scalar1d: lat = a + S(N)·N
//   Scalar2d: lat = a + S(N)·N,  N = d0·d1
// The slope S is piecewise constant over the segmentation variable,
// fitted per dtype.  The calibrated segments are NOT contiguous: a value
// inside a GAP (prev.hi < v < next.lo) is fitted by the line segment
// connecting the adjacent segments' endpoints (prev.hi, S_prev) →
// (next.lo, S_next) — the 1019 bridging rule; outside the calibrated
// range the boundary segment's slope is extended.  Outlier segments the
// source doc marks as measurement noise (scalar1d i32/f32/bf16 [~170,186]
// peak, scalar2d f16 [80,96]) are dropped — the bridge then interpolates
// across them.
//
// The segment tables are PRE-CV calibrations (pure-vector data, the same
// convention as every other vector table — confirmed 1019 for varange and
// evident here from the tail slopes matching VectorBaseParam.a × the b1
// dtype multiplier exactly, e.g. scalar2d i32 9.5e-6 ≈ 2.4e-6 × 3.96).
// evalPattern therefore multiplies the SLOPE term by VectorCVSlopeFactor
// (0.5) and leaves the intercept alone — the broadcast counterpart of the
// generic chain's uniform CV rule.  The queries below return the CV'd,
// production-ready cost.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_BROADCASTCOSTMODEL_H
#define BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_BROADCASTCOSTMODEL_H

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/CostModelTypes.h"
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/MSCostParams.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::hivm::costmodel::broadcast {

using llvm::ArrayRef;

/// The five broadcast patterns (see the file header for shapes/dims).
enum class Pattern {
  Inner,    // (d0,1) → (d0, pad(d1)), broadcast_dims = [1]
  Outer,    // (1,d1) → (pad(d0), d1), broadcast_dims = [0]
  Middle,   // (d0,1,d2) → (d0, pad(d1), d2), broadcast_dims = [1]
  Scalar1d, // scalar → pad(N)
  Scalar2d, // scalar → pad(d0) × pad(d1)
};

/// Which dtype a segment table was calibrated on.  Where the source doc
/// merges dtypes into one row ("i32/f32", middle's "f16/bf16") both keys
/// dispatch to the SAME table — one calibration covering both dtypes, not
/// two coincidentally-equal tables.
enum class Dtype {
  I8,
  I32,
  F32,
  F16,
  BF16,
};

/// One calibrated slope segment: S = slope over [lo, hi] of the pattern's
/// segmentation variable (d1 for inner/outer, u = d0·d1 for middle, N for
/// the scalar patterns); slope is µs per unit of the pattern's rate
/// variable (d0 for inner/outer, u/N for the others).
struct Segment {
  int64_t lo;
  int64_t hi;
  double slope;
};

// ═════════════════ inner: S(d1)·d0 tables ═════════════════

// inner, i8 — R² 0.937, MAPE 9.5%.
inline constexpr Segment InnerSeg_I8[] = {
    {2, 6, 3.06e-3},   {8, 8, 1.82e-3},    {10, 10, 3.74e-3},
    {12, 48, 1.68e-3}, {64, 512, 1.02e-3},
};

// inner, i32/f32 (one calibration) — R² 0.952, MAPE 8.9%.
inline constexpr Segment InnerSeg_I32F32[] = {
    {2, 6, 1.82e-3},    {8, 8, 9.51e-4},     {10, 12, 1.82e-3},
    {16, 128, 1.09e-3}, {192, 512, 4.73e-3},
};

// inner, f16 — R² 0.976, MAPE 6.3%.
inline constexpr Segment InnerSeg_F16[] = {
    {2, 12, 1.82e-3},   {16, 16, 9.64e-4},   {20, 40, 1.63e-3},
    {48, 256, 1.01e-3}, {384, 512, 4.70e-3},
};

// inner, bf16 — R² 0.957, MAPE 9.3%.
inline constexpr Segment InnerSeg_BF16[] = {
    {2, 12, 1.81e-3},   {16, 16, 9.59e-4},   {20, 40, 1.63e-3},
    {48, 256, 9.59e-4}, {384, 512, 4.70e-3},
};

// ═════════════════ outer: S(d1)·d0 tables ═════════════════

// outer, i8 — R² 1.000, MAPE 0.61%.
inline constexpr Segment OuterSeg_I8[] = {
    {2, 2, 3.64e-3},
    {4, 16, 1.82e-3},
    {32, 128, 6.06e-4},
};

// outer, i32/f32 (one calibration) — R² 1.000, MAPE 0.36%.
inline constexpr Segment OuterSeg_I32F32[] = {
    {2, 4, 1.82e-3},
    {8, 64, 6.06e-4},
    {128, 128, 1.21e-3},
};

// outer, f16 — R² 0.999, MAPE 3.78%.
inline constexpr Segment OuterSeg_F16[] = {
    {2, 2, 1.80e-3},
    {4, 8, 1.82e-3},
    {16, 16, 5.98e-4},
    {32, 128, 6.04e-4},
};

// outer, bf16 — R² 1.000, MAPE 0.48%.
inline constexpr Segment OuterSeg_BF16[] = {
    {2, 8, 1.82e-3},
    {16, 16, 6.06e-4},
    {32, 128, 6.06e-4},
};

// ═════════════════ middle: S(u)·u tables, u = d0·d1 ═════════════════

// middle, i8 — R² 0.939, MAPE 35.8%.
inline constexpr Segment MiddleSeg_I8[] = {
    {4, 16, 2.03e-3},     {32, 64, 2.07e-3},     {128, 256, 2.19e-3},
    {512, 1024, 2.59e-3}, {2048, 4096, 4.54e-3},
};

// middle, i32/f32 (one calibration) — R² 0.956, MAPE 37.6%.  The source
// gives 1.34~1.39e-3 for the [32,64] segment; the midpoint is used.
inline constexpr Segment MiddleSeg_I32F32[] = {
    {4, 16, 1.47e-3},     {32, 64, 1.365e-3},    {128, 256, 1.18e-3},
    {512, 1024, 1.64e-3}, {2048, 4096, 1.83e-3},
};

// middle, f16/bf16 (one calibration) — R² 0.985, MAPE 34.3%.
inline constexpr Segment MiddleSeg_F16BF16[] = {
    {4, 16, 1.64e-3},     {32, 64, 1.63e-3},     {128, 256, 1.47e-3},
    {512, 1024, 2.15e-3}, {2048, 4096, 1.83e-3},
};

// ═════════════════ scalar1d: S(N)·N tables ═════════════════
//
// The source doc's outlier segments (the N≈170~190 isolated-peak noise:
// i32 [178,186]→2.9e-3, bf16 [170,178]→1.1e-2, f32 [170,178]→2.9e-3) are
// dropped; the bridging rule interpolates across the widened gap.

// scalar1d, i8 — R² 0.984, MAPE 0.25%.
inline constexpr Segment Scalar1dSeg_I8[] = {
    {2, 178, 0.},
    {186, 288, 5.8e-6},
    {1458, 7386, 2.2e-6},
};

// scalar1d, i32 — R² 0.783, MAPE 4.2%.
inline constexpr Segment Scalar1dSeg_I32[] = {
    {2, 66, 0.},
    {74, 130, 1.7e-6},
    {138, 170, 0.},
    {288, 7386, 9.6e-6},
};

// scalar1d, f16 — R² 0.998, MAPE 0.19%.
inline constexpr Segment Scalar1dSeg_F16[] = {
    {2, 122, 0.},       {130, 138, 5.4e-5},   {146, 178, 0.},
    {186, 288, 5.8e-6}, {7386, 7386, 4.4e-6},
};

// scalar1d, bf16 — R² 0.269, MAPE 18.5% (the low R² is the dropped noise
// peak, not a structural difference from f16).
inline constexpr Segment Scalar1dSeg_BF16[] = {
    {2, 130, 4.2e-7},
    {138, 162, 0.},
    {186, 288, 0.},
    {1458, 7386, 4.5e-6},
};

// scalar1d, f32 — R² 0.613, MAPE 5.0%.
inline constexpr Segment Scalar1dSeg_F32[] = {
    {2, 34, 4.4e-6},
    {42, 162, 1.1e-5},
    {186, 288, 3.6e-5},
    {1458, 7386, 9.5e-6},
};

// ═════════════════ scalar2d: S(N)·N tables, N = d0·d1 ═════════════════

// scalar2d, i8 — R² 1.000, MAPE 0.01%.
inline constexpr Segment Scalar2dSeg_I8[] = {
    {4, 256, 0.},        {320, 384, 0.},       {512, 640, 4.7e-6},
    {768, 1536, 2.3e-6}, {2048, 8192, 2.4e-6},
};

// scalar2d, i32/f32 (one calibration) — R² 1.000, MAPE 0.12%.
inline constexpr Segment Scalar2dSeg_I32F32[] = {
    {4, 24, 3.0e-5},
    {32, 128, 0.},
    {160, 192, 0.},
    {256, 8192, 9.5e-6},
};

// scalar2d, bf16 — R² 1.000, MAPE 0.02%.
inline constexpr Segment Scalar2dSeg_BF16[] = {
    {4, 128, 0.},       {160, 192, 0.},       {256, 320, 9.4e-6},
    {384, 768, 4.7e-6}, {1024, 8192, 4.7e-6},
};

// scalar2d, f16 — R² 0.352, MAPE 10.4% (the source's [80,96]→-7.1e-3
// outlier segment is dropped; the low R² is that single point).
inline constexpr Segment Scalar2dSeg_F16[] = {
    {4, 64, 0.},
    {128, 192, 9.6e-6},
    {256, 320, 9.4e-6},
    {384, 8192, 4.7e-6},
};

// ═════════════════ per-pattern intercepts (µs) ═════════════════
//
// The source reports per-dtype intercept RANGES, not per-segment values;
// one constant per pattern is used (the spread, ≤ 0.005 µs, is below the
// reported MAPE of every non-scalar pattern).  Scalar1d/2d take the
// doc's recommended constants; inner/outer take the midpoints of the
// quoted ranges (inner 0.018~0.029, outer 0.0197~0.0235); middle takes
// the doc's floor value.
inline constexpr double Intercept_Inner = 0.0235;
inline constexpr double Intercept_Outer = 0.0216;
inline constexpr double Intercept_Middle = 0.021;
inline constexpr double Intercept_Scalar1d = 0.0218;
inline constexpr double Intercept_Scalar2d = 0.0215;

// middle latency floor: the source's [512,1024] segment fits a negative
// intercept (a ≈ -0.11~-0.17) and the doc mandates lat ≥ 0.021 µs when
// budgeting — redundant while Intercept_Middle stays at 0.021, kept as
// the guard the doc asks for.
inline constexpr double MiddleLatencyFloor = 0.021;

// ═════════════════ lookups and evaluation ═════════════════

/// Pattern name for notes.
inline llvm::StringLiteral patternName(Pattern p) {
  switch (p) {
  case Pattern::Inner:
    return "inner";
  case Pattern::Outer:
    return "outer";
  case Pattern::Middle:
    return "middle";
  case Pattern::Scalar1d:
    return "scalar1d";
  case Pattern::Scalar2d:
    return "scalar2d";
  }
  return "?";
}

/// Render `ty` for a note (mirrors MSCostModel.cpp's dtypeName).
inline std::string dtypeNameOf(Type ty) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << ty;
  return s;
}

/// Classify an element type for the broadcast tables; std::nullopt when
/// the dtype has no calibration (i1, i16, i64, f64, f8 ... — vbrc accepts
/// them, the 1019 tables do not cover them).
inline std::optional<Dtype> dtypeOf(Type elemType) {
  if (auto intTy = dyn_cast_if_present<IntegerType>(elemType)) {
    switch (intTy.getWidth()) {
    case 8:
      return Dtype::I8;
    case 32:
      return Dtype::I32;
    default:
      return std::nullopt;
    }
  }
  if (isa<Float32Type>(elemType))
    return Dtype::F32;
  if (isa<Float16Type>(elemType))
    return Dtype::F16;
  if (isa<BFloat16Type>(elemType))
    return Dtype::BF16;
  return std::nullopt;
}

/// Segment table of (pattern, dtype); empty when the combination has no
/// calibration (never happens for a dtype that dtypeOf accepted — the
/// defensive empty return keeps a future table edit from crashing).
inline ArrayRef<Segment> segmentsFor(Pattern p, Dtype d) {
  switch (p) {
  case Pattern::Inner:
    switch (d) {
    case Dtype::I8:
      return InnerSeg_I8;
    case Dtype::I32:
    case Dtype::F32:
      return InnerSeg_I32F32;
    case Dtype::F16:
      return InnerSeg_F16;
    case Dtype::BF16:
      return InnerSeg_BF16;
    }
    return {};
  case Pattern::Outer:
    switch (d) {
    case Dtype::I8:
      return OuterSeg_I8;
    case Dtype::I32:
    case Dtype::F32:
      return OuterSeg_I32F32;
    case Dtype::F16:
      return OuterSeg_F16;
    case Dtype::BF16:
      return OuterSeg_BF16;
    }
    return {};
  case Pattern::Middle:
    switch (d) {
    case Dtype::I8:
      return MiddleSeg_I8;
    case Dtype::I32:
    case Dtype::F32:
      return MiddleSeg_I32F32;
    case Dtype::F16:
    case Dtype::BF16:
      return MiddleSeg_F16BF16;
    }
    return {};
  case Pattern::Scalar1d:
    switch (d) {
    case Dtype::I8:
      return Scalar1dSeg_I8;
    case Dtype::I32:
      return Scalar1dSeg_I32;
    case Dtype::F32:
      return Scalar1dSeg_F32;
    case Dtype::F16:
      return Scalar1dSeg_F16;
    case Dtype::BF16:
      return Scalar1dSeg_BF16;
    }
    return {};
  case Pattern::Scalar2d:
    switch (d) {
    case Dtype::I8:
      return Scalar2dSeg_I8;
    case Dtype::I32:
    case Dtype::F32:
      return Scalar2dSeg_I32F32;
    case Dtype::F16:
      return Scalar2dSeg_F16;
    case Dtype::BF16:
      return Scalar2dSeg_BF16;
    }
    return {};
  }
  return {};
}

/// Per-pattern intercept (µs).
inline double interceptOf(Pattern p) {
  switch (p) {
  case Pattern::Inner:
    return Intercept_Inner;
  case Pattern::Outer:
    return Intercept_Outer;
  case Pattern::Middle:
    return Intercept_Middle;
  case Pattern::Scalar1d:
    return Intercept_Scalar1d;
  case Pattern::Scalar2d:
    return Intercept_Scalar2d;
  }
  return Intercept_Inner; // unreachable
}

/// Piecewise slope S(v) with the 1019 gap-bridging rule:
///   - v inside a calibrated segment [lo, hi] → that segment's slope;
///   - v inside a gap (prev.hi < v < next.lo) → the line connecting the
///     adjacent segments' endpoints, (prev.hi, S_prev) → (next.lo,
///     S_next) — the uncalibrated middle regions are fitted by joining
///     the neighboring pieces, never by a step jump;
///   - v outside the calibrated range → the nearest boundary segment
///     (extend the boundary piece, do not extrapolate a bridge).
inline double slopeAt(ArrayRef<Segment> segs, int64_t v) {
  assert(!segs.empty() && "empty broadcast segment table");
  // First segment whose hi >= v.
  size_t i = 0;
  while (i < segs.size() && v > segs[i].hi)
    ++i;
  if (i == segs.size())
    return segs.back().slope; // above the calibrated range
  if (v >= segs[i].lo)
    return segs[i].slope; // inside segment i
  if (i == 0)
    return segs[0].slope; // below the calibrated range
  // Gap: segs[i-1].hi < v < segs[i].lo — bridge the two endpoints.
  const Segment &prev = segs[i - 1];
  const Segment &next = segs[i];
  double t =
      static_cast<double>(v - prev.hi) / static_cast<double>(next.lo - prev.hi);
  return prev.slope + (next.slope - prev.slope) * t;
}

/// Shared tail of the five queries: dtype dispatch, lat(µs) = intercept +
/// CV·S(v)·rate (the tables are pre-CV calibrations, the slope term alone
/// carries VectorCVSlopeFactor), middle floor, CostResult normalization
/// (same contract as MSCostModel.cpp's normalizeCostTime: scale µs → ns,
/// round, clamp to the minimum cost 1).
inline CostResult evalPattern(Pattern p, Type elemType, int64_t v,
                              int64_t rate) {
  llvm::StringLiteral what = patternName(p);
  std::optional<Dtype> dtype = dtypeOf(elemType);
  if (!dtype)
    return CostResult{-1,
                      "broadcast " + what.str() +
                          " dtype not calibrated: " + dtypeNameOf(elemType)};
  ArrayRef<Segment> segs = segmentsFor(p, *dtype);
  if (segs.empty())
    return CostResult{-1, "broadcast " + what.str() +
                              " has no segment table (internal drift)"};

  double y = interceptOf(p) +
             VectorCVSlopeFactor * slopeAt(segs, v) * static_cast<double>(rate);
  if (p == Pattern::Middle)
    y = std::max(y, MiddleLatencyFloor);
  if (!std::isfinite(y) || y < 0.) {
    std::string ys;
    llvm::raw_string_ostream os(ys);
    os << y;
    return CostResult{-1, "broadcast " + what.str() + " y = " + ys +
                              " after eval, expect >= 0"};
  }
  double rounded = std::round(y * CostTimeUnitScale);
  return CostResult{rounded < 1. ? 1 : static_cast<int64_t>(rounded),
                    std::string()};
}

/// Shared dimension check: every real dimension must be positive (this
/// model never sees kDynamic — the caller resolves real dims first).
inline std::optional<CostResult> badDim(Pattern p, llvm::StringLiteral dim,
                                        int64_t v) {
  if (v <= 0)
    return CostResult{-1, "broadcast " + patternName(p).str() + " " +
                              dim.str() + " = " + std::to_string(v) +
                              ", expect > 0"};
  return std::nullopt;
}

/// Shared dim product with overflow guard (middle's u, scalar2d's N).
inline std::optional<int64_t> dimProduct(int64_t a, int64_t b) {
  if (a > INT64_MAX / b)
    return std::nullopt;
  return a * b;
}

// ═════════════════ the five queries ═════════════════

/// inner (column broadcast): src (d0, 1) → dst (d0, pad(d1)),
/// broadcast_dims = [1].  d0 = row count, d1 = REAL column width.
/// lat(µs) = a + S(d1)·d0.
inline CostResult getInnerCost(Type elemType, int64_t d0, int64_t d1) {
  if (auto err = badDim(Pattern::Inner, "d0", d0))
    return *err;
  if (auto err = badDim(Pattern::Inner, "d1", d1))
    return *err;
  return evalPattern(Pattern::Inner, elemType, /*v=*/d1, /*rate=*/d0);
}

/// outer (row broadcast): src (1, d1) → dst (pad(d0), d1),
/// broadcast_dims = [0].  d0 = REAL row count, d1 = row width.
/// lat(µs) = a + S(d1)·d0.
inline CostResult getOuterCost(Type elemType, int64_t d0, int64_t d1) {
  if (auto err = badDim(Pattern::Outer, "d0", d0))
    return *err;
  if (auto err = badDim(Pattern::Outer, "d1", d1))
    return *err;
  return evalPattern(Pattern::Outer, elemType, /*v=*/d1, /*rate=*/d0);
}

/// middle (mid-axis broadcast): src (d0, 1, d2) → dst (d0, pad(d1), d2),
/// broadcast_dims = [1].  d0 = plane rows, d1 = mid-axis width, d2 =
/// plane columns (d2 does not enter the model — the copy rounds are
/// u = d0·d1).  lat(µs) = a + S(u)·u, u = d0·d1, floored at
/// MiddleLatencyFloor.
inline CostResult getMiddleCost(Type elemType, int64_t d0, int64_t d1) {
  if (auto err = badDim(Pattern::Middle, "d0", d0))
    return *err;
  if (auto err = badDim(Pattern::Middle, "d1", d1))
    return *err;
  std::optional<int64_t> u = dimProduct(d0, d1);
  if (!u)
    return CostResult{-1, "broadcast middle d0*d1 overflows"};
  return evalPattern(Pattern::Middle, elemType, /*v=*/*u, /*rate=*/*u);
}

/// scalar1d: scalar src → pad(N) dst, no broadcast_dims.  N = REAL output
/// length.  lat(µs) = a + S(N)·N.
inline CostResult getScalar1dCost(Type elemType, int64_t n) {
  if (auto err = badDim(Pattern::Scalar1d, "n", n))
    return *err;
  return evalPattern(Pattern::Scalar1d, elemType, /*v=*/n, /*rate=*/n);
}

/// scalar2d: scalar src → pad(d0) × pad(d1) dst, no broadcast_dims.
/// d0 × d1 = REAL output dims, N = d0·d1.  lat(µs) = a + S(N)·N.
inline CostResult getScalar2dCost(Type elemType, int64_t d0, int64_t d1) {
  if (auto err = badDim(Pattern::Scalar2d, "d0", d0))
    return *err;
  if (auto err = badDim(Pattern::Scalar2d, "d1", d1))
    return *err;
  std::optional<int64_t> n = dimProduct(d0, d1);
  if (!n)
    return CostResult{-1, "broadcast scalar2d d0*d1 overflows"};
  return evalPattern(Pattern::Scalar2d, elemType, /*v=*/*n, /*rate=*/*n);
}

} // namespace mlir::hivm::costmodel::broadcast

#endif // BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_BROADCASTCOSTMODEL_H
