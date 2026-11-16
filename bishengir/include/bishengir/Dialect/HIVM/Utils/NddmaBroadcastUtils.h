//===- NddmaBroadcastUtils.h - bounded broadcast views -----------*- C++ -*-===//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_NDDMABROADCASTUTILS_H
#define BISHENGIR_DIALECT_HIVM_UTILS_NDDMABROADCASTUTILS_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::hivm::nddma {
// Flatten may rebuild an unchanged extent as 1*n (and independently on each
// view). Strip only exact arithmetic identities, never arbitrary products.
inline OpFoldResult normalizeBroadcastExtent(OpFoldResult size) {
  if (!size)
    return size;
  while (auto value = dyn_cast<Value>(size)) {
    if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
      if (getConstantIntValue(mul.getLhs()) == 1) {
        size = mul.getRhs();
        continue;
      }
      if (getConstantIntValue(mul.getRhs()) == 1) {
        size = mul.getLhs();
        continue;
      }
    }
    break;
  }
  return size;
}

// A small, explicit allocation-bound proof, not a substitute for memory
// planning. Unsigned min with a nonnegative constant also proves the lower
// bound. Unknown SSA values never become one or a guessed hardware capacity.
inline std::optional<int64_t> getBroadcastExtentBound(OpFoldResult size) {
  size = normalizeBroadcastExtent(size);
  if (!size)
    return std::nullopt;
  if (auto c = getConstantIntValue(size))
    return *c >= 0 ? c : std::nullopt;
  auto value = dyn_cast<Value>(size);
  auto min = value ? value.getDefiningOp<arith::MinUIOp>() : arith::MinUIOp{};
  if (!min)
    return std::nullopt;
  auto lhs = getConstantIntValue(min.getLhs());
  auto rhs = getConstantIntValue(min.getRhs());
  if (lhs && *lhs >= 0)
    return lhs;
  if (rhs && *rhs >= 0)
    return rhs;
  return std::nullopt;
}

inline bool isBoundedBroadcastView(Value value, bool requireAllocation) {
  auto type = cast<MemRefType>(value.getType());
  // Follow static intermediate views too: layout planning can introduce a
  // rank-reducing subview between a bounded dynamic slice and its allocation.
  // PointerCastOp carries MemAlloc semantics and replaces planned UB allocs;
  // its static result shape is the storage contract, not an invented UB cap.
  if (type.hasStaticShape() &&
      (!requireAllocation || value.getDefiningOp<memref::AllocOp>() ||
       value.getDefiningOp<hivm::PointerCastOp>()))
    return true;
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return isBoundedBroadcastView(cast.getSource(), requireAllocation);
  if (auto expand = value.getDefiningOp<memref::ExpandShapeOp>())
    return isBoundedBroadcastView(expand.getSrc(), requireAllocation);
  auto view = value.getDefiningOp<memref::SubViewOp>();
  if (!view || !view.getSourceType().hasStaticShape() ||
      !isBoundedBroadcastView(view.getSource(), requireAllocation))
    return false;
  auto offsets = view.getMixedOffsets();
  auto sizes = view.getMixedSizes();
  auto strides = view.getMixedStrides();
  for (int64_t i = 0; i < view.getSourceType().getRank(); ++i) {
    auto offset = getConstantIntValue(offsets[i]);
    auto stride = getConstantIntValue(strides[i]);
    auto bound = getBroadcastExtentBound(sizes[i]);
    const int64_t capacity = view.getSourceType().getDimSize(i);
    if (!stride || *stride != 1 || !bound)
      return false;
    if (offset) {
      if (*offset < 0 || *offset > capacity || *bound > capacity - *offset)
        return false;
      continue;
    }
    // A suffix [D-n,D) is bounded even though offset is not constant. Require
    // the exact same bounded n; unrelated offsets/sizes are not a proof.
    auto value = dyn_cast<Value>(normalizeBroadcastExtent(offsets[i]));
    // Arith-to-affine canonicalization expresses the same suffix as D-n.
    // Match only this exact one-input affine identity, not arbitrary maps.
    if (auto apply = value ? value.getDefiningOp<affine::AffineApplyOp>()
                           : affine::AffineApplyOp{}) {
      auto map = apply.getAffineMap();
      if (map.getNumInputs() == 1 && map.getNumResults() == 1 &&
          *bound <= capacity &&
          isEqualConstantIntOrValue(
              normalizeBroadcastExtent(apply.getOperand(0)),
              normalizeBroadcastExtent(sizes[i]))) {
        auto input = map.getNumDims() == 1
                         ? getAffineDimExpr(0, value.getContext())
                         : getAffineSymbolExpr(0, value.getContext());
        if (map.getResult(0) ==
            getAffineConstantExpr(capacity, value.getContext()) - input)
          continue;
      }
      return false;
    }
    auto sub = value ? value.getDefiningOp<arith::SubIOp>() : arith::SubIOp{};
    auto end = sub ? getConstantIntValue(sub.getLhs()) : std::nullopt;
    if (!end || *end < 0 || *end > capacity || *bound > *end ||
        !isEqualConstantIntOrValue(normalizeBroadcastExtent(sub.getRhs()),
                                  normalizeBroadcastExtent(sizes[i])))
      return false;
  }
  return true;
}
} // namespace mlir::hivm::nddma
#endif
