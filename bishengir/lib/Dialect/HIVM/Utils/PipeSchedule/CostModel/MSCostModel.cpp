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
#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/BroadcastCostModel.h"

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
#include <functional>
#include <iterator>
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

// ===== vcast conversion graph: 7 dtype nodes, 23 directed edges ===========
//
// Node names and their indices; kVCastNodes is the single source of truth
// for the node set (i1 and f64 are deliberately NOT vcast nodes — as a
// DESTINATION (or in f64's case either side) a pair falls back per §9.4
// of the design doc; an i1 SOURCE is remapped onto the i8 node by
// vcastPairMultiplier per the 1019 calibration).
constexpr llvm::StringLiteral kVCastNodes[] = {"i8",  "i16",  "i32", "i64",
                                               "f16", "bf16", "f32"};
constexpr int kVCastNodeCount = static_cast<int>(std::size(kVCastNodes));

/// Node index of a dtype NAME; -1 when outside the 7 vcast dtypes.
int vcastNodeIdxOf(StringRef name) {
  for (int i = 0; i < kVCastNodeCount; ++i)
    if (kVCastNodes[i] == name)
      return i;
  return -1;
}

/// Whether `ty` is the i1 integer type (the vcast i1-source special case).
bool isI1Type(Type ty) {
  auto intTy = dyn_cast_if_present<IntegerType>(ty);
  return intTy && intTy.getWidth() == 1;
}

/// Node index of an MLIR dtype; -1 when outside the 7 vcast dtypes.
int vcastNodeId(Type elementType) {
  llvm::StringRef name;
  if (auto intTy = dyn_cast_if_present<IntegerType>(elementType)) {
    switch (intTy.getWidth()) {
    case 8:
      name = "i8";
      break;
    case 16:
      name = "i16";
      break;
    case 32:
      name = "i32";
      break;
    case 64:
      name = "i64";
      break;
    default:
      return -1;
    }
  } else if (isa<BFloat16Type>(elementType)) {
    name = "bf16";
  } else if (isa<Float16Type>(elementType)) {
    name = "f16";
  } else if (isa<Float32Type>(elementType)) {
    name = "f32";
  } else {
    return -1;
  }
  return vcastNodeIdxOf(name);
}

/// Per-op cost of hivm.hir.vbrc through the dedicated broadcast model.
/// nullopt when the op is not one of the five calibrated patterns (or its
/// dtype/metadata is not covered) — the caller then keeps the generic
/// vector billing.  A returned -1 CostResult means a RECOGNIZED pattern
/// with invalid dims (overflow): fatal for the whole group.
std::optional<CostResult> vbrcBroadcastCost(const VectorOpDesc &opDesc) {
  // Real (unpadded) output shape, resolved by MsDagBuilder; without it
  // (memref form / dynamic shapes) the model cannot be driven — generic
  // fallback.
  if (opDesc.outputs.empty() || opDesc.vbrcRealOutShape.empty())
    return std::nullopt;
  Type elemType = opDesc.outputs.front().getElementType();
  if (!broadcast::dtypeOf(elemType))
    return std::nullopt; // i1/i16/i64/f64/f8 — no broadcast calibration.
  ArrayRef<int64_t> real = opDesc.vbrcRealOutShape;

  // Scalar source: a bare scalar never enters inputs (a 0-D source tensor
  // has an empty shape) — the pattern is the real output rank alone.
  if (opDesc.inputs.empty() || opDesc.inputs.front().getShape().empty()) {
    if (real.size() == 1) {
      LLVM_DEBUG(dbgs() << "[ms-cost-model] vbrc scalar1d n=" << real[0]
                        << "\n");
      return broadcast::getScalar1dCost(elemType, real[0]);
    }
    if (real.size() == 2) {
      LLVM_DEBUG(dbgs() << "[ms-cost-model] vbrc scalar2d " << real[0] << "x"
                        << real[1] << "\n");
      return broadcast::getScalar2dCost(elemType, real[0], real[1]);
    }
    return std::nullopt;
  }

  // Vector source: (d0,1) inner / (1,d1) outer / (d0,1,d2) middle — the
  // broadcast axis is the 1-sized input dim, every other dim passes
  // through unchanged (a mismatch means a form the tables do not cover).
  ArrayRef<int64_t> in = opDesc.inputs.front().getShape();
  if (in.size() == 2 && real.size() == 2) {
    if (in[1] == 1 && in[0] == real[0]) {
      LLVM_DEBUG(dbgs() << "[ms-cost-model] vbrc inner " << real[0] << "x"
                        << real[1] << "\n");
      return broadcast::getInnerCost(elemType, real[0], real[1]);
    }
    if (in[0] == 1 && in[1] == real[1]) {
      LLVM_DEBUG(dbgs() << "[ms-cost-model] vbrc outer " << real[0] << "x"
                        << real[1] << "\n");
      return broadcast::getOuterCost(elemType, real[0], real[1]);
    }
    return std::nullopt;
  }
  if (in.size() == 3 && real.size() == 3 && in[1] == 1 && in[0] == real[0] &&
      in[2] == real[2]) {
    LLVM_DEBUG(dbgs() << "[ms-cost-model] vbrc middle " << real[0] << "x"
                      << real[1] << "x" << real[2] << "\n");
    return broadcast::getMiddleCost(elemType, real[0], real[1]);
  }
  return std::nullopt;
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
// Vector multiplier lookups (b1 dtype / b2 input-count / b4 op type)
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

std::optional<double>
mlir::hivm::costmodel::vectorInputCountCoeffFor(Type elementType) {
  if (auto intTy = dyn_cast_if_present<IntegerType>(elementType)) {
    switch (intTy.getWidth()) {
    case 1:
      return VectorInputCountCoeff_I1;
    case 8:
      return VectorInputCountCoeff_I8;
    case 16:
      return VectorInputCountCoeff_I16;
    case 32:
      return VectorInputCountCoeff_I32;
    case 64:
      return VectorInputCountCoeff_I64;
    default:
      return std::nullopt;
    }
  }
  if (isa<BFloat16Type>(elementType))
    return VectorInputCountCoeff_BF16;
  if (isa<Float16Type>(elementType))
    return VectorInputCountCoeff_F16;
  if (isa<Float32Type>(elementType))
    return VectorInputCountCoeff_F32;
  return std::nullopt;
}

std::optional<CostMultiplier>
mlir::hivm::costmodel::lookupVectorOpTypeMultiplier(StringRef opName) {
  // Linear scan: the table is short and looked up once per op.
  for (const auto &[name, mult] : VectorOpTypeMultiplierList)
    if (opName == name)
      return mult;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// vcast: minimum-hop, minimum-slope-sum conversion path
//===----------------------------------------------------------------------===//

FailureOr<CostMultiplier>
mlir::hivm::costmodel::vcastPairMultiplier(Type srcDtype, Type dstDtype) {
  // ---- ⓪ i1 source (1019): i1→i8 widens with no vector time at all; any
  // other i1→X pair bills exactly like an i8 source (the i1 payload rides
  // the i8 datapath).  i1 as DESTINATION keeps the fallback semantics —
  // only the source side is covered by the calibration.
  bool srcIsI1 = isI1Type(srcDtype);
  if (srcIsI1) {
    auto dstInt = dyn_cast_if_present<IntegerType>(dstDtype);
    if (dstInt && dstInt.getWidth() == 8)
      return VCastPairMultiplier_I1_to_I8;
  }

  // ---- ① dtype → node index (outside the 7 dtypes → failure; the caller
  // applies VCastPairMultiplier_Fallback and records the op in the note).
  int src = srcIsI1 ? vcastNodeIdxOf("i8") : vcastNodeId(srcDtype);
  int dst = vcastNodeId(dstDtype);
  if (src < 0 || dst < 0)
    return failure();

  // Identity conversion (e.g. f16→f16 rounding): {1.×base, 1.}.
  if (src == dst)
    return CostMultiplier{VCastSlopeBaseFactor * 1., 1.};

  // ---- Adjacency lists in VCastDirectPairList enumeration order — that
  // order is the tie-break rule (design §9.2 rule 3), so it must survive
  // into the DFS traversal order.
  const int edgeCount = static_cast<int>(std::size(VCastDirectPairList));
  SmallVector<SmallVector<int, 6>, 8> adj(kVCastNodeCount);
  SmallVector<int, 24> edgeDst(edgeCount, -1);
  for (int e = 0; e < edgeCount; ++e) {
    int from = vcastNodeIdxOf(VCastDirectPairList[e].src);
    int to = vcastNodeIdxOf(VCastDirectPairList[e].dst);
    if (from < 0 || to < 0)
      continue; // malformed table row — skip defensively; never traversed.
    adj[from].push_back(e);
    edgeDst[e] = to;
  }

  // ---- ② BFS: minimum hop count from src to every node.  Hops are the
  // hardware conversion command count — the first-priority path rule.
  SmallVector<int, 8> hops(kVCastNodeCount, -1);
  SmallVector<int, 8> queue;
  queue.push_back(src);
  hops[src] = 0;
  for (size_t qi = 0; qi < queue.size(); ++qi) {
    int u = queue[qi];
    for (int e : adj[u]) {
      int v = edgeDst[e];
      if (hops[v] == -1) {
        hops[v] = hops[u] + 1;
        queue.push_back(v);
      }
    }
  }
  if (hops[dst] == -1)
    return failure(); // unreachable — defensive; the graph is strongly
                      // connected, so this cannot trigger with the current
                      // table.

  // ---- ③ DFS: enumerate all simple paths of exactly hops[dst] edges;
  // smallest slope sum wins, first-enumerated breaks remaining ties (the
  // strict `<` keeps the earliest path among equals).  Every minimum-hop
  // walk is necessarily simple (a repeated node would imply a shorter
  // path, contradicting BFS), so the onPath check is belt-and-braces.
  SmallVector<int, 4> path; // edge indices along the current path
  SmallVector<char, 8> onPath(kVCastNodeCount, 0);
  onPath[src] = 1;
  bool found = false;
  double bestSlopeSum = 0.;
  double bestInterceptSum = 0.;
  SmallVector<int, 4> bestPath;
  std::function<void(int, double, double)> dfs = [&](int node, double slopeSum,
                                                     double interceptSum) {
    if (static_cast<int>(path.size()) == hops[dst]) {
      if (node == dst && (!found || slopeSum < bestSlopeSum)) {
        found = true;
        bestSlopeSum = slopeSum;
        bestInterceptSum = interceptSum;
        bestPath = path;
      }
      return;
    }
    for (int e : adj[node]) {
      int next = edgeDst[e];
      if (onPath[next])
        continue;
      onPath[next] = 1;
      path.push_back(e);
      dfs(next, slopeSum + VCastDirectPairList[e].slope,
          interceptSum + VCastDirectPairList[e].intercept);
      path.pop_back();
      onPath[next] = 0;
    }
  };
  dfs(src, 0., 0.);
  if (!found)
    return failure(); // defensive: BFS promised a path at this hop count.

  LLVM_DEBUG({
    dbgs() << "[ms-cost-model] vcast path " << kVCastNodes[src];
    for (int e : bestPath)
      dbgs() << "->" << kVCastNodes[edgeDst[e]];
    dbgs() << " slope_sum=" << bestSlopeSum
           << " intercept_sum=" << bestInterceptSum << "\n";
  });

  // ---- ④ slope: Σ per-hop slopes × base factor; intercept: Σ per-hop
  // intercepts (= hop count; every edge intercept is 1. today).
  return CostMultiplier{VCastSlopeBaseFactor * bestSlopeSum, bestInterceptSum};
}

//===----------------------------------------------------------------------===//
// Vector: per-op multiplier chain, summed, times VF-factor
//===----------------------------------------------------------------------===//

CostResult
MSCostModel::getVectorGroupCost(const VectorGroupCostInput &input) const {
  double sum = 0.0;
  SmallVector<std::string, 4> fallbackOps;

  // An op can miss several tables at once (op type + dtype); record each
  // op once so the note stays readable.
  auto noteFallback = [&fallbackOps](StringRef opName) {
    if (!llvm::is_contained(fallbackOps, opName))
      fallbackOps.push_back(opName.str());
  };

  for (const VectorOpDesc &opDesc : input.ops) {
    StringRef opName = opDesc.op ? opDesc.op->getName().getStringRef()
                                 : StringRef("<unknown>");

    // ---- ① Ignore list: dialect-prefix match (tensor. / memref. /
    // bufferization.).  A hit is skipped outright — no warning, no latency.
    if (llvm::any_of(VectorOpIgnoreList, [opName](StringRef prefix) {
          return opName.starts_with(prefix);
        })) {
      LLVM_DEBUG(dbgs() << "[ms-cost-model] vec ignored op=" << opName << "\n");
      continue;
    }

    // ---- ② varange: dedicated output-driven model (1019).  varange's
    // only shaped operand is the DPS dst (offset/strides are Index
    // scalars that never enter the inputs list), so the generic
    // input-driven x below would always be 0; the calibration drives the
    // cost by the OUTPUT element count N instead.  A dedicated model is
    // not a table miss — no op-type lookup, no fallback note when the
    // output metadata is present.  Missing outputs (bufferized memref
    // form) → x = 0 + note, mirroring the vcast missing-metadata
    // philosophy (non-fatal fallback).
    if (opName == "hivm.hir.varange") {
      int64_t n = 0;
      bool outputMissing = opDesc.outputs.empty();
      if (!outputMissing) {
        FailureOr<int64_t> elems = opDesc.outputs.front().getTotalElements();
        if (failed(elems)) {
          if (hasDynamicDim(opDesc.outputs.front().getShape())) {
            LLVM_DEBUG(
                dbgs()
                << "[ms-cost-model] vec skipped dynamic op=" << opName
                << " output #0 has "
                << shapeFailReason(opDesc.outputs.front().getShape()) << "\n");
            continue;
          }
          return CostResult{
              -1, "vector op hivm.hir.varange output #0 has " +
                      shapeFailReason(opDesc.outputs.front().getShape())};
        }
        n = *elems;
      } else {
        noteFallback(opName);
      }
      // The CV slope factor applies here as well — the single application
      // point shared with step ⑦ below (vbrc applies the factor inside its
      // own model; the intercept is unaffected, exactly like every other
      // op).
      double aEff = VarangeParam.a * VectorCVSlopeFactor;
      double yOp = aEff * static_cast<double>(n) + VarangeParam.b;
      sum += yOp;
      LLVM_DEBUG(dbgs() << "[ms-cost-model] varange x=" << n << " y_op=" << yOp
                        << (outputMissing ? " [fallback]" : "") << "\n");
      continue;
    }

    // ---- ③ vbrc: dedicated broadcast model (1019).  The five CV
    // patterns are recognized from the input shape plus the REAL output
    // shape the builder resolved (vbrcRealOutShape — the recorded result
    // shape is the padded DPS out).  Forms outside the five patterns,
    // uncalibrated dtypes, and missing metadata fall through to the
    // generic billing below (the op-type miss records the fallback note),
    // preserving the pre-1019 behavior as the fallback path.
    if (opName == "hivm.hir.vbrc") {
      if (std::optional<CostResult> r = vbrcBroadcastCost(opDesc)) {
        if (r->costTime == -1)
          return *r; // recognized pattern, invalid dims — group failure.
        sum += static_cast<double>(r->costTime) / CostTimeUnitScale;
        LLVM_DEBUG(dbgs() << "[ms-cost-model] vbrc y_op=" << r->costTime
                          << "ns\n");
        continue;
      }
    }

    // ---- ④ Op-type multiplier (b4): table lookup; a miss uses the
    // fallback multiplier and lands the op in the note.
    std::optional<CostMultiplier> opTypeMult =
        lookupVectorOpTypeMultiplier(opName);
    bool fellBack = !opTypeMult;
    if (fellBack) {
      noteFallback(opName);
      opTypeMult = VectorOpTypeMultiplier_Fallback;
    }

    // ---- ⑤ Input element statistics: x = max element count over the
    // op's real inputs (elementwise inputs are same-shaped; a broadcast
    // takes the largest, matching the output — prefer overestimate).
    // Dynamic inputs are skipped (the op contributes zero cost) so that
    // ops like stride_load whose GM source is naturally dynamic do not
    // fail the entire group.  Non-dynamic invalid inputs (non-positive
    // dims, overflow) still fail the group.
    int64_t x = 0;
    const size_t nInput = opDesc.inputs.size();
    bool skippedDynamic = false;
    for (size_t i = 0; i < nInput; ++i) {
      FailureOr<int64_t> elements = opDesc.inputs[i].getTotalElements();
      if (failed(elements)) {
        if (hasDynamicDim(opDesc.inputs[i].getShape())) {
          LLVM_DEBUG(
              dbgs() << "[ms-cost-model] vec skipped dynamic op=" << opName
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

    // ---- ⑥ Assemble the multiplier chain.
    double slopeMult = 1.;
    double interceptMult = 1.;
    if (opName == "hivm.hir.vcast") {
      // vcast: dedicated conversion multiplier (design §9) — outside the
      // b1 dtype / b2 input-count dimensions; the pair multiplier already
      // contains the 2. base factor.  CV 0.5 is applied uniformly below.
      // Fallback composition (§5.8): the 2. base factor still applies,
      // only the pair multiplier falls back to 1.
      CostMultiplier pair{VCastSlopeBaseFactor *
                              VCastPairMultiplier_Fallback.slope,
                          VCastPairMultiplier_Fallback.intercept};
      bool pairFellBack = true;
      // src dtype = first real input, dst dtype = first recorded output;
      // missing metadata (no inputs / no outputs) → fallback + note.
      if (!opDesc.inputs.empty() && !opDesc.outputs.empty()) {
        FailureOr<CostMultiplier> m =
            vcastPairMultiplier(opDesc.inputs.front().getElementType(),
                                opDesc.outputs.front().getElementType());
        if (succeeded(m)) {
          pair = *m;
          pairFellBack = false;
        }
      }
      if (pairFellBack)
        noteFallback(opName);
      fellBack = fellBack || pairFellBack;
      slopeMult = pair.slope;
      interceptMult = pair.intercept;
      LLVM_DEBUG(dbgs() << "[ms-cost-model]   vcast x=" << x << " mult={"
                        << pair.slope << "," << pair.intercept << "}"
                        << (pairFellBack ? " [fallback]" : "") << "\n");
    } else {
      // b1 dtype (first real input's dtype): a miss (f64 etc.) is
      // non-fatal — {1., 1.} + note, unlike cube (Q8 vs Q1).
      CostMultiplier dtypeMult{1., 1.};
      // b2 input count: slope = (n_input - 1)·c + 1.; a dtype miss uses
      // c = 1.  n_input = 0 (degenerate op) → multiplier 1., x = 0.
      double inCountSlope = 1.;
      if (nInput > 0) {
        Type dtype = opDesc.inputs.front().getElementType();
        std::optional<CostMultiplier> dm = vectorDtypeMultiplierFor(dtype);
        if (dm) {
          dtypeMult = *dm;
        } else {
          noteFallback(opName);
          fellBack = true;
        }
        std::optional<double> c = vectorInputCountCoeffFor(dtype);
        if (!c)
          c = 1.; // dtype miss already recorded above.
        inCountSlope = (static_cast<double>(nInput) - 1.) * *c + 1.;
      }
      // b3 shape (TODO placeholder) × b4 op type.
      slopeMult = dtypeMult.slope * inCountSlope * VectorShapeMultiplier.slope *
                  opTypeMult->slope;
      interceptMult = dtypeMult.intercept * VectorShapeMultiplier.intercept *
                      opTypeMult->intercept;
      LLVM_DEBUG(dbgs() << "[ms-cost-model]   dtype_mult={" << dtypeMult.slope
                        << "," << dtypeMult.intercept << "}"
                        << " in_count=" << inCountSlope << " shape_mult={"
                        << VectorShapeMultiplier.slope << ","
                        << VectorShapeMultiplier.intercept << "}"
                        << " op_type={" << opTypeMult->slope << ","
                        << opTypeMult->intercept << "}"
                        << " cv=" << VectorCVSlopeFactor
                        << (fellBack ? " [fallback]" : "") << "\n");
    }

    // ---- ⑦ Per-op contribution: ONE intercept per op (v2 semantics —
    // v1 added b per input); the CV slope factor is applied here for
    // every op on this path, vcast included (vbrc applies the factor
    // inside its own model; single application point per path, Q3).
    double aEff = VectorBaseParam.a * slopeMult * VectorCVSlopeFactor;
    double bEff = VectorBaseParam.b * interceptMult;
    sum += aEff * static_cast<double>(x) + bEff;
    LLVM_DEBUG(dbgs() << "[ms-cost-model] vec op=" << opName << " x=" << x
                      << " n_input=" << nInput << " y_op="
                      << aEff * static_cast<double>(x) + bEff << "\n");
  }

  // ---- ⑧ Group level: VF factor + normalization (v1 semantics).
  double y = sum * VectorVF_Factor;
  if (!std::isfinite(y) || y < 0.0)
    return CostResult{-1, "vector group y = " + formatDouble(y) +
                              " after eval, expect >= 0"};
  int64_t costTime = normalizeCostTime(y);

  LLVM_DEBUG(dbgs() << "[ms-cost-model] vector group sum=" << sum
                    << " y=sum*vf=" << y << "us costTime=" << costTime
                    << "ns\n");

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
                          << input.pipeline << ", " << input.storageTag
                          << "): "
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
