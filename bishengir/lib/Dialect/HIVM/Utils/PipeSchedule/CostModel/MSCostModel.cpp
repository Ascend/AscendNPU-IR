//===- MSCostModel.cpp - modulo-schedule linear cost model -------*- C++
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

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/MSCostModel.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::hivm::costmodel;
using llvm::dbgs;

// Unified debug channel for the MS preload scheduler pipeline; the message
// prefix identifies the producing component.
#define DEBUG_TYPE "ms-preload-scheduler"

namespace {

/// Render `ty` for a note ("f16", "bf16", ...).
std::string dtypeName(Type ty) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << ty;
  return s;
}

/// Render a double for a note (shortest sensible form).
std::string formatDouble(double v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << v;
  return s;
}

/// True when any dimension in `shape` is ShapedType::kDynamic.
bool hasDynamicDim(ArrayRef<int64_t> shape) {
  return llvm::any_of(shape,
                      [](int64_t d) { return ShapedType::isDynamic(d); });
}

/// Describe the first unusable dimension of `shape`: dynamic dims first,
/// then non-positive ones.  std::nullopt when every dimension is usable
/// (the failure then came from the overflow guard).
std::optional<std::string> badDimNote(ArrayRef<int64_t> shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (ShapedType::isDynamic(shape[i]))
      return "dim " + std::to_string(i) + " is dynamic";
    if (shape[i] <= 0)
      return "dim " + std::to_string(i) + " = " + std::to_string(shape[i]) +
             ", expect > 0";
  }
  return std::nullopt;
}

/// Build the shape-failure note for a data block, e.g.
/// " has dynamic dim" is appended by callers; this returns the reason
/// suffix ("dim 1 is dynamic" / "dim 2 = 0, expect > 0" / "shape overflows").
std::string shapeFailReason(ArrayRef<int64_t> shape) {
  if (std::optional<std::string> dim = badDimNote(shape))
    return *dim;
  return "shape overflows";
}

/// Normalize a raw y — in the parameter tables' calibration unit (μs) —
/// into the reported costTime (ns): scale by CostTimeUnitScale first, then
/// round and clamp to the minimum cost 1.  The scaling MUST precede the
/// rounding: clamping in μs and scaling afterwards would quantize every
/// sub-μs node to the same 1000 ns and discard the multiplier chain the
/// caller just evaluated.  Shared by all three query paths so cube,
/// vector, and transfer costs always land in the same unit.
int64_t normalizeCostTime(double yMicros) {
  double rounded = std::round(yMicros * CostTimeUnitScale);
  return rounded < 1.0 ? 1 : static_cast<int64_t>(rounded);
}

/// Evaluate y = a·x + b with the shared sanity checks: the result must be
/// finite and non-negative BEFORE the normalization; NaN/Inf or negative
/// values mean a broken parameter table or an upstream overflow and return
/// CostResult{-1, note} carrying the raw y value (in μs).  y == 0 is NOT
/// an error — it clamps to the minimum cost 1 (e.g. an empty vector group:
/// a WorkItem holding only transfer ops legitimately bills no vector cost).
CostResult evalLinear(LinParam param, int64_t x, StringRef what) {
  double y = param.a * static_cast<double>(x) + param.b;
  if (!std::isfinite(y) || y < 0.0)
    return CostResult{-1, what.str() + " y = " + formatDouble(y) +
                              " after eval, expect >= 0"};
  return CostResult{normalizeCostTime(y), std::string()};
}

} // namespace

//===----------------------------------------------------------------------===//
// ShapedOperandDesc — shared shape/dtype evaluation
//===----------------------------------------------------------------------===//

FailureOr<int64_t> ShapedOperandDesc::getTotalElements() const {
  // 0-D scalar tensor: an empty shape counts as one element.
  if (shape.empty())
    return int64_t(1);
  int64_t total = 1;
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim) || dim <= 0)
      return failure();
    if (dim > INT64_MAX / total)
      return failure(); // overflow guard: huge bogus shapes are data errors.
    total *= dim;
  }
  return total;
}

FailureOr<int64_t> ShapedOperandDesc::getTotalBytes() const {
  FailureOr<int64_t> elements = getTotalElements();
  if (failed(elements))
    return failure();
  FailureOr<int64_t> bytes = bytesOfDtype(elementType);
  if (failed(bytes))
    return failure();
  if (*elements > INT64_MAX / *bytes)
    return failure();
  return *elements * *bytes;
}

//===----------------------------------------------------------------------===//
// Dtype table helpers
//===----------------------------------------------------------------------===//

FailureOr<int64_t> mlir::hivm::costmodel::bytesOfDtype(Type elementType) {
  if (auto intTy = dyn_cast_if_present<IntegerType>(elementType)) {
    // i1 weighs 1 B rather than the (width+7)/8 rounding.
    if (intTy.getWidth() == 1)
      return int64_t(1);
    return static_cast<int64_t>((intTy.getWidth() + 7) / 8);
  }
  if (auto floatTy = dyn_cast_if_present<FloatType>(elementType)) {
    if (isa<BFloat16Type>(elementType))
      return int64_t(2);
    return static_cast<int64_t>(floatTy.getWidth() / 8);
  }
  return failure();
}

bool mlir::hivm::costmodel::isAcceptedDtype(Type elementType) {
  return succeeded(bytesOfDtype(elementType));
}

//===----------------------------------------------------------------------===//
// Cube: x = sm·sn·sk from the first two fractal dimensions
//===----------------------------------------------------------------------===//

/// Map an element type to its cube dtype multiplier; std::nullopt when the
/// dtype is not supported (f64 etc. — cube keeps the v1 fatal semantics:
/// a dtype the hardware cannot execute gets no fabricated latency).
static std::optional<CostMultiplier> cubeMultiplierForDtype(Type elemTy) {
  if (isa<Float16Type, BFloat16Type>(elemTy))
    return elemTy.isBF16() ? CubeDtypeMultiplier_BF16 : CubeDtypeMultiplier_F16;
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    if (intTy.getWidth() == 8)
      return CubeDtypeMultiplier_I8;
  if (isa<Float32Type>(elemTy))
    return CubeDtypeMultiplier_F32;
  return std::nullopt;
}

CostResult MSCostModel::getCubeCost(const CubeCostInput &input) const {
  ArrayRef<int64_t> lhs = input.lhs.getShape();
  ArrayRef<int64_t> rhs = input.rhs.getShape();

  // ---- Rank: both shapes must have at least 2 dims (only the first two
  // are read; higher dims are ignored by the formula).
  if (lhs.size() < 2)
    return CostResult{-1, "cube lhs shape rank < 2"};
  if (rhs.size() < 2)
    return CostResult{-1, "cube rhs shape rank < 2"};

  // ---- Dynamic dims among the first two are fatal (higher-dim kDynamic
  // does not affect the formula and is tolerated here).
  auto dimErr = [](StringRef which, int dimIdx) {
    return "cube " + which.str() + " dim " + std::to_string(dimIdx) +
           " is dynamic";
  };
  if (ShapedType::isDynamic(lhs[0]))
    return CostResult{-1, dimErr("lhs", 0)};
  if (ShapedType::isDynamic(lhs[1]))
    return CostResult{-1, dimErr("lhs", 1)};
  if (ShapedType::isDynamic(rhs[0]))
    return CostResult{-1, dimErr("rhs", 0)};
  if (ShapedType::isDynamic(rhs[1]))
    return CostResult{-1, dimErr("rhs", 1)};

  // A fractal: shape[0] = sk, shape[1] = sm.
  // B fractal: shape[0] = sn (or sk under transpose).
  // The sk == skRhs consistency check is intentionally omitted: transpose
  // attributes (a_transpose / b_transpose) reorder the fractal dimensions,
  // so the rhs first two dims do not reliably encode sk.  The cost formula
  // only needs sm, sn, and sk from the lhs side.
  int64_t sk = lhs[0];
  int64_t sm = lhs[1];
  int64_t sn = rhs[0];

  auto positiveErr = [](StringRef sym, int64_t v) {
    return CostResult{-1, "cube " + sym.str() + " = " + std::to_string(v) +
                              ", expect > 0"};
  };
  if (sk <= 0)
    return positiveErr("sk", sk);
  if (sm <= 0)
    return positiveErr("sm", sm);
  if (sn <= 0)
    return positiveErr("sn", sn);

  // ---- dtype dispatch: base parameter × dtype multiplier.
  Type lhsTy = input.lhs.getElementType();
  std::optional<CostMultiplier> mult = cubeMultiplierForDtype(lhsTy);
  if (!mult)
    return CostResult{-1, "cube dtype not supported: " + dtypeName(lhsTy) +
                              ", only f16/bf16/i8/f32 supported"};

  if (sm > INT64_MAX / sn || sm * sn > INT64_MAX / sk)
    return CostResult{-1, "cube sm*sn*sk overflows"};
  int64_t x = sm * sn * sk;

  // Effective params = CubeBaseParam × dtype multiplier.
  LinParam param{CubeBaseParam.a * mult->slope,
                 CubeBaseParam.b * mult->intercept};
  LLVM_DEBUG(dbgs() << "[ms-cost-model] cube dtype=" << dtypeName(lhsTy)
                    << " mult={slope=" << mult->slope
                    << ", intercept=" << mult->intercept << "}"
                    << " a_eff=" << param.a << " b_eff=" << param.b << "\n");

  return evalLinear(param, x, "cube");
}

//===----------------------------------------------------------------------===//
// Vector dtype multiplier lookup
//===----------------------------------------------------------------------===//

std::optional<CostMultiplier>
mlir::hivm::costmodel::vectorDtypeMultiplierFor(Type elementType) {
  if (auto intTy = dyn_cast_if_present<IntegerType>(elementType)) {
    switch (intTy.getWidth()) {
    case 1:
      return VectorDtypeMult_I1;
    case 8:
      return VectorDtypeMult_I8;
    case 16:
      return VectorDtypeMult_I16;
    case 32:
      return VectorDtypeMult_I32;
    case 64:
      return VectorDtypeMult_I64;
    default:
      return std::nullopt;
    }
  }
  if (isa<BFloat16Type>(elementType))
    return VectorDtypeMult_BF16;
  if (isa<Float16Type>(elementType))
    return VectorDtypeMult_F16;
  if (isa<Float32Type>(elementType))
    return VectorDtypeMult_F32;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Vector: per-op dtype × CV billing, summed over the group
//===----------------------------------------------------------------------===//

CostResult
MSCostModel::getVectorGroupCost(const VectorGroupCostInput &input) const {
  double sum = 0.0;
  SmallVector<std::string, 4> fallbackOps;

  // The same op can appear several times in a group; record each name once
  // so the note stays readable.
  auto noteFallback = [&fallbackOps](StringRef opName) {
    if (!llvm::is_contained(fallbackOps, opName))
      fallbackOps.push_back(opName.str());
  };

  for (const VectorOpDesc &opDesc : input.ops) {
    StringRef opName = opDesc.op ? opDesc.op->getName().getStringRef()
                                 : StringRef("<unknown>");

    // ---- ① Billability filter: dialect-prefix match (tensor. / memref. /
    // bufferization.).  A hit is skipped outright — no warning, no
    // latency: those ops are buffer/tensor plumbing rather than vector
    // instructions, so even a bare intercept would inflate the group.
    // This is the ONLY point at which op kind influences the cost.
    if (llvm::any_of(VectorOpIgnoreList, [opName](StringRef prefix) {
          return opName.starts_with(prefix);
        })) {
      LLVM_DEBUG(dbgs() << "[ms-cost-model] vec ignored op=" << opName << "\n");
      continue;
    }

    // ---- ② Input element statistics: x = max element count over the
    // op's inputs (elementwise inputs are same-shaped; a broadcast takes
    // the largest, matching the output — prefer overestimate).
    // Dynamic inputs are skipped (the op contributes zero cost) so that
    // ops like stride_load whose GM source is naturally dynamic do not
    // fail the entire group.  Non-dynamic invalid inputs (non-positive
    // dims, overflow) still fail the group.
    int64_t x = 0;
    bool skippedDynamic = false;
    for (size_t i = 0, e = opDesc.inputs.size(); i < e; ++i) {
      FailureOr<int64_t> elements = opDesc.inputs[i].getTotalElements();
      if (failed(elements)) {
        if (hasDynamicDim(opDesc.inputs[i].getShape())) {
          LLVM_DEBUG(dbgs()
                     << "[ms-cost-model] vec skipped dynamic op=" << opName
                     << " input #" << i << " has "
                     << shapeFailReason(opDesc.inputs[i].getShape()) << "\n");
          skippedDynamic = true;
          break;
        }
        return CostResult{-1, "vector op " + opName.str() + " input #" +
                                  std::to_string(i) + " has " +
                                  shapeFailReason(opDesc.inputs[i].getShape())};
      }
      x = std::max(x, *elements);
    }
    if (skippedDynamic)
      continue;

    // ---- ③ dtype multiplier, read from the FIRST input's dtype.  A miss
    // (f64 etc.) is non-fatal — {1., 1.} plus the op name in the note,
    // unlike cube, which refuses to fabricate a latency.  An op with no
    // shaped input keeps {1., 1.} and x = 0, i.e. it bills one bare
    // intercept.
    CostMultiplier dtypeMult{1., 1.};
    if (!opDesc.inputs.empty()) {
      Type dtype = opDesc.inputs.front().getElementType();
      if (std::optional<CostMultiplier> dm = vectorDtypeMultiplierFor(dtype))
        dtypeMult = *dm;
      else
        noteFallback(opName);
    }

    // ---- ④ Per-op contribution: ONE intercept per op (not per input);
    // the CV slope factor is applied here, the single application point
    // shared by every op on this path.  Op kind does not participate.
    double aEff = VectorBaseParam.a * dtypeMult.slope * VectorCVSlopeFactor;
    double bEff = VectorBaseParam.b * dtypeMult.intercept;
    double yOp = aEff * static_cast<double>(x) + bEff;
    sum += yOp;
    LLVM_DEBUG(dbgs() << "[ms-cost-model] vec op=" << opName << " x=" << x
                      << " dtype_mult={" << dtypeMult.slope << ","
                      << dtypeMult.intercept << "}"
                      << " cv=" << VectorCVSlopeFactor << " y_op=" << yOp
                      << "\n");
  }

  // ---- ⑤ Group level: the plain per-op sum, then normalization.  An
  // empty (or fully ignored) group yields sum = 0, which clamps to the
  // minimum cost 1 rather than failing.
  if (!std::isfinite(sum) || sum < 0.0)
    return CostResult{-1, "vector group y = " + formatDouble(sum) +
                              " after eval, expect >= 0"};
  int64_t costTime = normalizeCostTime(sum);

  LLVM_DEBUG(dbgs() << "[ms-cost-model] vector group y=sum=" << sum
                    << "us costTime=" << costTime << "ns\n");

  std::string note;
  if (!fallbackOps.empty())
    note = "fallback ops: " + llvm::join(fallbackOps, ", ");
  return CostResult{costTime, std::move(note)};
}

//===----------------------------------------------------------------------===//
// Transfer: x = byte size, dispatched on (pipeline, storageTag)
//===----------------------------------------------------------------------===//

CostResult MSCostModel::getTransferCost(const TransferCostInput &input) const {
  // ---- x = total bytes of the moved block.
  FailureOr<int64_t> bytes = input.data.getTotalBytes();
  if (failed(bytes)) {
    if (failed(input.data.getTotalElements())) {
      if (hasDynamicDim(input.data.getShape())) {
        LLVM_DEBUG(dbgs() << "[ms-cost-model] transfer skipped dynamic data ("
                          << input.pipeline << ", " << input.storageTag << "): "
                          << shapeFailReason(input.data.getShape()) << "\n");
        return CostResult{1, "transfer data has dynamic shape, skipped"};
      }
      return CostResult{-1, "transfer data has " +
                                shapeFailReason(input.data.getShape())};
    }
    return CostResult{-1, "transfer dtype not accepted: " +
                              dtypeName(input.data.getElementType())};
  }
  int64_t x = *bytes;

  // ---- Two-level dispatch: pipeline → base param, (pipeline, tag) →
  // direction multiplier.  Effective params = base × multiplier.
  StringRef pipeline = input.pipeline;
  StringRef tag = input.storageTag;
  const LinParam *base = nullptr;
  const CostMultiplier *dirMult = nullptr;
  if (pipeline == "mte1") {
    base = &TransferBaseParam_MTE1;
    if (tag == "L1_to_L0A")
      dirMult = &TransferDirMult_MTE1_L1_to_L0A;
    else if (tag == "L1_to_L0B")
      dirMult = &TransferDirMult_MTE1_L1_to_L0B;
  } else if (pipeline == "mte2") {
    base = &TransferBaseParam_MTE2;
    if (tag == "GM_to_L1")
      dirMult = &TransferDirMult_MTE2_GM_to_L1;
    else if (tag == "GM_to_UB")
      dirMult = &TransferDirMult_MTE2_GM_to_UB;
  } else if (pipeline == "mte3") {
    base = &TransferBaseParam_MTE3;
    if (tag == "UB_to_L1")
      dirMult = &TransferDirMult_MTE3_UB_to_L1;
    else if (tag == "UB_to_GM")
      dirMult = &TransferDirMult_MTE3_UB_to_GM;
  } else if (pipeline == "fixpipe") {
    base = &TransferBaseParam_FIXPIPE;
    if (tag == "L0C_to_L1")
      dirMult = &TransferDirMult_FIXPIPE_L0C_to_L1;
    else if (tag == "L0C_to_UB")
      dirMult = &TransferDirMult_FIXPIPE_L0C_to_UB;
    else if (tag == "L0C_to_GM")
      dirMult = &TransferDirMult_FIXPIPE_L0C_to_GM;
  }

  if (!base || !dirMult) {
    // DAG2 only generates tabled combinations; an unknown pair means DAG2
    // and the parameter table drifted apart — report, don't fall back.
    return CostResult{-1, "transfer (" + pipeline.str() + ", " + tag.str() +
                              ") not in param table"};
  }

  LinParam param{base->a * dirMult->slope, base->b * dirMult->intercept};
  LLVM_DEBUG(dbgs() << "[ms-cost-model] transfer (" << pipeline << ", " << tag
                    << ") x=" << x << "B"
                    << " mult={slope=" << dirMult->slope
                    << ", intercept=" << dirMult->intercept << "}"
                    << " a_eff=" << param.a << " b_eff=" << param.b << "\n");

  return evalLinear(param, x, "transfer");
}
