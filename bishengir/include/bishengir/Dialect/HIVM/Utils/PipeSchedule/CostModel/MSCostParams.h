//===- MSCostParams.h - MS cost model parameter tables ----------*- C++ -*-===//
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
// All MS cost model parameters live in this file — tuning and list
// maintenance only ever touch this header.  MSCostModel only reads them;
// it must never contain a numeric literal of its own.
//
// 0920 layout: parameters are split into a BASE area (one calibrated
// LinParam per module — cube, vector, and the four transfer pipelines)
// and a SPECIAL area (per-scenario CostMultiplier chains).  Effective
// parameters are always `base × multiplier`; every scenario keeps its
// OWN constant so any entry can be tuned independently (no shared or
// cross-initialized constants).
//
// Layout (top to bottom): LinParam / CostMultiplier structs, dtype byte
// table, base area, special area (cube dtype, transfer direction, vector
// dtype / input-count / shape / op-type, vcast), lookup-function
// declarations, and the vector list area at the END of the file (the area
// for ongoing additions).
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_MSCOSTPARAMS_H
#define BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_MSCOSTPARAMS_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <initializer_list>
#include <optional>
#include <utility>

namespace mlir::hivm::costmodel {

/// Linear parameter y = a·x + b.
struct LinParam {
  double a;
  double b;
};

/// Scenario multiplier: effective parameter = base parameter × multiplier.
///   effective slope     = base.a * slope
///   effective intercept = base.b * intercept
struct CostMultiplier {
  double slope;
  double intercept;

  constexpr bool operator==(const CostMultiplier &rhs) const {
    return slope == rhs.slope && intercept == rhs.intercept;
  }
};

// ===== dtype → byte width table =========================================
//
// Accepted element types and their byte widths (B), one entry per type for
// quick lookup.  Also used to validate an elementType: a type outside the
// table makes the querying cost module return CostResult{-1, note} with the
// actual dtype name in the note.  The lookup helpers bytesOfDtype() and
// isAcceptedDtype() are declared alongside the table; their implementations
// dispatch on the MLIR Type (IntegerType → (width+7)/8 with i1 as 1 B,
// FloatType → width/8, BF16 special-cased to 2 B) and live in
// MSCostModel.cpp.
inline constexpr std::pair<llvm::StringLiteral, int64_t> DtypeBytesList[] = {
    {"i1", 1},  {"i8", 1},  {"i16", 2}, {"f16", 2}, {"bf16", 2},
    {"i32", 4}, {"f32", 4}, {"i64", 8}, {"f64", 8},
};

// ═════════════════ Base area: calibrated LinParams ═════════════════
//
// One calibrated parameter set per module (unit μs); the right-hand
// comment gives the dimension of x and the calibration scenario.
//
// The parameter tables keep their calibration unit (μs) while
// CostResult::costTime is reported in ns: y = a·x + b is multiplied by
// CostTimeUnitScale before the round/clamp normalization.  Scaling BEFORE
// the rounding is the whole point of the ns unit — at μs granularity a
// typical node (cube with x = 512 → 0.39 μs, a 32 KB MTE2 transfer →
// 0.49 μs) rounds below 1 and clamps to the minimum cost 1, which
// collapses the multiplier chains below into one undifferentiated value.
inline constexpr double CostTimeUnitScale = 1000.;

// x = sub-tile count (sm·sn·sk after fractal split); calibrated on
// i8 / f16 mmad.
inline constexpr LinParam CubeBaseParam = {6.06e-4, 0.081};

// x = element count of one input; calibrated on vadd processing one
// 1-D i8 input.
inline constexpr LinParam VectorBaseParam = {2.4e-6, 0.026};

// x = bytes moved; one base set per transfer pipeline.
inline constexpr LinParam TransferBaseParam_MTE1 = {5.54e-6, 0.021};
inline constexpr LinParam TransferBaseParam_MTE2 = {6.39e-6, 0.276};
inline constexpr LinParam TransferBaseParam_MTE3 = {5.65e-6, 0.150};
inline constexpr LinParam TransferBaseParam_FIXPIPE = {5.4e-6, 0.137};

// ═════════════════ Special area: multiplier tables ═════════════════

// ===== Cube: dtype multipliers (one constant per dtype) ==================
//
// Effective params for cross-checking:
//   i8 / f16 / bf16 → slope 6.06e-4, intercept 0.081
//   f32             → slope 4.848e-3, intercept 0.081
inline constexpr CostMultiplier CubeDtypeMultiplier_I8 = {1., 1.};
inline constexpr CostMultiplier CubeDtypeMultiplier_F16 = {1., 1.};
inline constexpr CostMultiplier CubeDtypeMultiplier_BF16 = {1., 1.};
inline constexpr CostMultiplier CubeDtypeMultiplier_F32 = {8., 1.};

// ===== Transfer: direction multipliers (one constant per direction) ======
//
// One multiplier per (pipeline, storageTag) pair produced by DAG2's
// addTransfer(); all nine combinations are covered, an unknown pair means
// DAG2 and the table drifted apart and is reported as CostResult{-1, note}
// rather than silently falling back.
//
// Effective slopes for cross-checking (base × multiplier):
//   MTE1  L1→L0A / L1→L0B      5.54e-6
//   MTE2  GM→L1 / GM→UB        6.39e-6
//   MTE3  UB→L1                5.65e-6 × 0.5 = 2.825e-6
//   MTE3  UB→GM                5.65e-6
//   fixpipe L0C→L1             5.4e-6
//   fixpipe L0C→UB             5.4e-6 × 0.877 = 4.7358e-6
//   fixpipe L0C→GM             5.4e-6
inline constexpr CostMultiplier TransferDirMult_MTE1_L1_to_L0A = {1., 1.};
inline constexpr CostMultiplier TransferDirMult_MTE1_L1_to_L0B = {1., 1.};
inline constexpr CostMultiplier TransferDirMult_MTE2_GM_to_L1 = {1., 1.};
inline constexpr CostMultiplier TransferDirMult_MTE2_GM_to_UB = {1., 1.};
inline constexpr CostMultiplier TransferDirMult_MTE3_UB_to_L1 = {0.5, 1.};
inline constexpr CostMultiplier TransferDirMult_MTE3_UB_to_GM = {1., 1.};
inline constexpr CostMultiplier TransferDirMult_FIXPIPE_L0C_to_L1 = {1., 1.};
inline constexpr CostMultiplier TransferDirMult_FIXPIPE_L0C_to_UB = {0.877, 1.};
inline constexpr CostMultiplier TransferDirMult_FIXPIPE_L0C_to_GM = {1., 1.};

// ===== Vector b1: dtype multipliers (one constant per dtype) =============
//
// Vector supports 8 dtypes (i1, i8, i16, i32, i64, f16, bf16, f32),
// grouped by byte width: 1B slope 1., 2B slope 2., 4B slope 3.96,
// 8B slope 13.96; intercepts are all 1.
inline constexpr CostMultiplier VectorDtypeMult_I1 = {1., 1.};
inline constexpr CostMultiplier VectorDtypeMult_I8 = {1., 1.};
inline constexpr CostMultiplier VectorDtypeMult_I16 = {2., 1.};
inline constexpr CostMultiplier VectorDtypeMult_F16 = {2., 1.};
inline constexpr CostMultiplier VectorDtypeMult_BF16 = {2., 1.};
inline constexpr CostMultiplier VectorDtypeMult_I32 = {3.96, 1.};
inline constexpr CostMultiplier VectorDtypeMult_F32 = {3.96, 1.};
inline constexpr CostMultiplier VectorDtypeMult_I64 = {13.96, 1.};

// ===== Vector b2: input-count coefficients (one constant per dtype) ======
//
// The input-count multiplier is a formula, not a constant:
//   slope = (n_input - 1) * c + 1.,  intercept = 1.
// n_input counts the op's REAL shaped inputs (DPS dst and temp buffers
// excluded); n_input = 0 → multiplier 1.
inline constexpr double VectorInputCountCoeff_I1 = 0.546;
inline constexpr double VectorInputCountCoeff_I8 = 0.546;
inline constexpr double VectorInputCountCoeff_I16 = 0.722;
inline constexpr double VectorInputCountCoeff_F16 = 0.722;
inline constexpr double VectorInputCountCoeff_BF16 = 0.722;
inline constexpr double VectorInputCountCoeff_I32 = 0.821;
inline constexpr double VectorInputCountCoeff_F32 = 0.821;
inline constexpr double VectorInputCountCoeff_I64 = 0.073;

// ===== Vector b3: shape multiplier (TODO: awaiting calibration) ==========
inline constexpr CostMultiplier VectorShapeMultiplier = {1., 1.};

// ===== Vector b4: op-type multipliers =====================================
//
// CV 1:2 property: the slope of EVERY op is additionally multiplied by
// this factor (the final slope multiplier = 0.5 × the table entry); the
// intercept is unaffected.  Applied uniformly in getVectorGroupCost —
// including vcast and varange — so each op path has exactly one
// application point; the broadcast model applies it inside its own
// evalPattern (vbrc never reaches the generic chain).  Confirmed 1019:
// the vector calibrations are pre-CV pure-vector data, this factor
// converts them to the CV execution reality.
inline constexpr double VectorCVSlopeFactor = 0.5;

// Op-type multiplier table for calibrated ops (exact op-name match,
// "hivm.hir." prefix).  Whether an op is "calibrated" is decided by a
// table hit — no separate accept list is maintained.
inline constexpr std::pair<llvm::StringLiteral, CostMultiplier>
    VectorOpTypeMultiplierList[] = {
        {"hivm.hir.vadd", {1., 1.}},    {"hivm.hir.vmul", {1., 1.}},
        {"hivm.hir.vsub", {1., 1.}},    {"hivm.hir.vand", {1., 1.}},
        {"hivm.hir.vxor", {1., 1.}},    {"hivm.hir.vshl", {1., 1.}},
        {"hivm.hir.vshr", {1., 1.}},    {"hivm.hir.vcmp", {1., 1.}},
        {"hivm.hir.vexp", {1.09, 1.}},  {"hivm.hir.vsqrt", {1.09, 1.}},
        {"hivm.hir.vln", {1.09, 1.}},   {"hivm.hir.vdiv", {1.09, 1.}},
        {"hivm.hir.vmod", {22.14, 2.}},
};

// Op-type multiplier for ops missing from the table: same fallback
// philosophy as v1 — the op lands in the note for a caller-side warning,
// but scheduling is not interrupted.
inline constexpr CostMultiplier VectorOpTypeMultiplier_Fallback = {1., 1.};

// ===== vcast: dtype-conversion multipliers ================================
//
// vcast sits OUTSIDE the vector b1/b2 dimensions (requirements §4): one
// real input, dtype effects expressed by the conversion path itself.

// Slope base factor (slope only): multiplied on top of the vector base
// parameter before the per-pair path multiplier.
inline constexpr double VCastSlopeBaseFactor = 2.;

/// One directed single-command conversion edge between two of the 7
/// supported dtypes: {src, dst, slope multiplier, intercept multiplier}.
/// Intercept multipliers are all 1 today.
struct VCastDirectPair {
  llvm::StringLiteral src;
  llvm::StringLiteral dst;
  double slope;
  double intercept;
};

// 23 direct (single-command) edges among the 7 dtypes
// (i8, i16, i32, i64, f16, bf16, f32): 12 integer↔integer, 5 int→float,
// 4 float↔float (bf16 only connects via f32), 2 float→int.  Multipliers
// are NOT symmetric (i16→i8 is 1.31 while i8→i16 is 1.4).  Enumeration
// order is the tie-break for equal-hop equal-sum paths (determinism).
inline constexpr VCastDirectPair VCastDirectPairList[] = {
    {"i16", "f16", 1., 1.},    {"i16", "i8", 1.31, 1.},
    {"i8", "f16", 1.4, 1.},    {"i8", "i16", 1.4, 1.},
    {"f32", "i32", 1.99, 1.},  {"i32", "f32", 1.99, 1.},
    {"i8", "i32", 2.48, 1.},   {"f32", "bf16", 2.65, 1.},
    {"f32", "f16", 2.65, 1.},  {"i32", "i16", 2.65, 1.},
    {"i16", "i32", 2.77, 1.},  {"i16", "f32", 2.77, 1.},
    {"i32", "i8", 2.77, 1.},   {"bf16", "f32", 2.77, 1.},
    {"f16", "f32", 2.77, 1.},  {"i64", "i16", 3.905, 1.},
    {"i64", "i32", 3.905, 1.}, {"i32", "i64", 5.15, 1.},
    {"i64", "i8", 5.95, 1.},   {"i16", "i64", 6.465, 1.},
    {"i8", "i64", 6.465, 1.},  {"i64", "f32", 11.78, 1.},
    {"f32", "i64", 14.99, 1.},
};

// Conversion multiplier for pairs outside the 7 dtypes (e.g. f64) or with
// missing metadata: the 2. base factor still applies, only the pair
// multiplier falls back; the op lands in the note.
inline constexpr CostMultiplier VCastPairMultiplier_Fallback = {1., 1.};

// i1 → i8 widening (1019): no vector time at all — the i1 payload already
// rides the i8 datapath, so the conversion bills neither slope nor
// intercept (both multipliers zero; the 2. base factor is bypassed).
// i1 → any other dtype bills exactly like an i8 source (see
// vcastPairMultiplier); i1 as DESTINATION keeps the fallback semantics.
inline constexpr CostMultiplier VCastPairMultiplier_I1_to_I8 = {0., 0.};

// ===== varange: dedicated output-driven model (1019) =====================
//
// varange (fill a strided index range into its output) sits OUTSIDE the
// vector b1/b2 dimensions like vcast: its only shaped operand is the DPS
// dst (offset/strides are Index scalars that never enter the input
// lists), so the cost driver is the OUTPUT element count N, not the
// inputs.  Absolute calibrated parameters (NOT multipliers on
// VectorBaseParam): raw model in ns, t = 25.60 + (3.04/64)·N, calibrated
// on i32 output; no dtype dimension is given, other output dtypes ride
// the same model.  Stored in µs like every other table entry; the CV
// slope factor applies on top at query time (see getVectorGroupCost) —
// confirmed 1019, the calibration is pre-CV pure-vector data like every
// other vector table.
inline constexpr LinParam VarangeParam = {3.04 / 64. / CostTimeUnitScale,
                                          25.60 / CostTimeUnitScale};

// ===== lookup helpers (implemented in MSCostModel.cpp) ===================

/// Byte width of `elementType`; failure() when the type is not in the
/// table (the calling module converts that to CostResult{-1, note} with
/// the actual dtype name).
FailureOr<int64_t> bytesOfDtype(Type elementType);

/// Whether `elementType` is in the accepted dtype table.
bool isAcceptedDtype(Type elementType);

/// Op-type multiplier for `opName` (exact "hivm.hir.*" match against
/// VectorOpTypeMultiplierList); std::nullopt when the op is not in the
/// table — the caller then uses VectorOpTypeMultiplier_Fallback and
/// records the op in the note.
std::optional<CostMultiplier>
lookupVectorOpTypeMultiplier(llvm::StringRef opName);

/// Vector b1 dtype multiplier (VectorDtypeMult_*); std::nullopt when the
/// dtype is not among the 8 supported ones — the caller then uses
/// {1., 1.} and records the op in the note (non-fatal, unlike cube).
std::optional<CostMultiplier> vectorDtypeMultiplierFor(Type elementType);

/// Vector b2 input-count coefficient (VectorInputCountCoeff_*); the
/// input-count slope multiplier is (n_input - 1) * c + 1.  std::nullopt
/// when the dtype is not among the 8 supported ones — the caller then
/// uses 1. and records the op in the note.
std::optional<double> vectorInputCountCoeffFor(Type elementType);

/// vcast conversion multiplier: given src/dst dtypes, find the
/// minimum-hop conversion path, sum the per-hop multipliers along it, and
/// return {VCastSlopeBaseFactor × slope-sum, intercept-sum}.
///   - src == dst               → {2. × 1., 1.} (identity conversion)
///   - i1 src, i8 dst           → VCastPairMultiplier_I1_to_I8 (no vector
///                               time at all)
///   - i1 src, other dst        → billed as an i8 source (i1 rides the i8
///                               datapath; 1019 calibration note)
///   - dtype outside the 7 (i1 as dst, f64, ...), or unreachable →
///     failure() (the caller then uses VCastPairMultiplier_Fallback and
///     records the op in the note)
/// Path selection among equal-hop paths: smallest slope sum, then
/// earliest VCastDirectPairList enumeration order (deterministic).
FailureOr<CostMultiplier> vcastPairMultiplier(Type srcDtype, Type dstDtype);

// ═════════════════ Vector list area (end of file, ongoing) ═════════════════

// VF fusion discount factor: the vector group's total latency is
// multiplied by this factor.  VF (Vector Fusion) fuses multiple vector ops
// into one execution body; the fused cost is lower than the per-op sum, so
// the factor scales it down.
inline constexpr double VectorVF_Factor = 0.55;

// Vector op ignore list.  Entries are DIALECT PREFIXES matched with
// starts_with (v1 matched full names): a hit is skipped outright — no
// warning, no latency.  "hivm.hir.*" ops never match a prefix here.
inline constexpr std::initializer_list<llvm::StringLiteral> VectorOpIgnoreList =
    {
        "tensor.",        // tensor.empty / tensor.insert_slice ...
        "memref.",        // memref.alloc / memref.copy ...
        "bufferization.", // bufferization.to_memref ...
};

} // namespace mlir::hivm::costmodel

#endif // BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_MSCOSTPARAMS_H
