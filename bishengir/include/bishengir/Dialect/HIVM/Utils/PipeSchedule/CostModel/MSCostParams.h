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
// Parameters are split into a BASE area (one calibrated LinParam per
// module — cube, vector, and the four transfer pipelines) and a SPECIAL
// area (per-scenario CostMultiplier chains).  Effective parameters are
// always `base × multiplier`; every scenario keeps its OWN constant so any
// entry can be tuned independently (no shared or cross-initialized
// constants).
//
// The vector module is deliberately held to three dimensions — base
// parameter, dtype ratio, and the CV 1:2 slope factor.  It carries no
// op-type table and no per-op special models: every billable op of a
// vector group is charged by the same formula and the group cost is their
// plain sum (see MSCostModel::getVectorGroupCost).  Op kind only decides
// whether an op is billable at all, via VectorOpIgnoreList at the end of
// this file.
//
// Layout (top to bottom): LinParam / CostMultiplier structs, dtype byte
// table, base area, special area (cube dtype, transfer direction, vector
// dtype, vector CV factor), lookup-function declarations, and the vector
// list area at the END of the file (the area for ongoing additions).
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

// ===== Vector: dtype multipliers (one constant per dtype) ================
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

// ===== Vector: CV 1:2 slope factor =======================================
//
// CV 1:2 property: the slope of EVERY billable op is additionally
// multiplied by this factor (the final slope multiplier = 0.5 × the dtype
// entry); the intercept is unaffected.  Applied uniformly in
// getVectorGroupCost, so there is exactly one application point.
// Confirmed 1019: the vector calibrations are pre-CV pure-vector data,
// this factor converts them to the CV execution reality.
inline constexpr double VectorCVSlopeFactor = 0.5;

// ===== lookup helpers (implemented in MSCostModel.cpp) ===================

/// Byte width of `elementType`; failure() when the type is not in the
/// table (the calling module converts that to CostResult{-1, note} with
/// the actual dtype name).
FailureOr<int64_t> bytesOfDtype(Type elementType);

/// Whether `elementType` is in the accepted dtype table.
bool isAcceptedDtype(Type elementType);

/// Vector dtype multiplier (VectorDtypeMult_*); std::nullopt when the
/// dtype is not among the 8 supported ones — the caller then uses
/// {1., 1.} and records the op in the note (non-fatal, unlike cube).
std::optional<CostMultiplier> vectorDtypeMultiplierFor(Type elementType);

// ═════════════════ Vector list area (end of file, ongoing) ═════════════════

// Vector op ignore list.  Entries are DIALECT PREFIXES matched with
// starts_with: a hit is skipped outright — no warning, no latency.  This
// is not a cost table but a billability filter: the listed ops are not
// vector instructions at all (buffer bookkeeping and tensor/memref
// plumbing that the group carries along), so charging them even a bare
// intercept would inflate the group.  "hivm.hir.*" ops never match a
// prefix here.
inline constexpr std::initializer_list<llvm::StringLiteral> VectorOpIgnoreList =
    {
        "tensor.",        // tensor.empty / tensor.insert_slice ...
        "memref.",        // memref.alloc / memref.copy ...
        "bufferization.", // bufferization.to_memref ...
};

} // namespace mlir::hivm::costmodel

#endif // BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_MSCOSTPARAMS_H
