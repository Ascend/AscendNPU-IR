//===- ShapedOperandDesc.h - shaped data block value object -----*- C++
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
// The ShapedOperandDesc value object that owns the shared shape/dtype
// evaluation used by the cube, vector, and transfer modules.  The query
// inputs that carry these descriptors live in CostModelTypes.h; the two
// member functions are defined in MSCostModel.cpp, next to the dtype table
// getTotalBytes() consults.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_SHAPEDOPERANDDESC_H
#define BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_SHAPEDOPERANDDESC_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir::hivm::costmodel {

// The structures here and in CostModelTypes.h spell SmallVector/StringRef
// unqualified (matching the surrounding MLIR style); pull the llvm types in
// locally so these headers compile standalone without transitive
// mlir/Support/LLVM.h.
using llvm::SmallVector;

// ═══════════ SmallVector inline capacity constants ═══════════
//
// llvm::SmallVector<T, N>'s N is the inline storage capacity (small-size,
// in elements): with ≤ N elements the vector lives in the inline array with
// no heap allocation; beyond N it spills to the heap.  N only affects
// performance, never semantics — every field using these constants (here
// and in CostModelTypes.h) is an unbounded dynamic container that grows
// automatically (2x growth, amortized O(1)) and never truncates.  The named
// constants only keep typical small objects allocation-free; the builder
// side reserves exact/upper-bound capacities at fill time (see
// buildVectorGroupCostInput).

/// Inline dimension capacity for shapes: covers typical ranks (including
/// the 4-D cube fractal layout).
inline constexpr size_t ShapeInlineCapacity = 4;

/// Description of a single shaped data block (one input or output tensor).
/// An op can own multiple inputs/outputs; each block gets its own instance.
///
/// Implemented as a class rather than a bare struct: element count and byte
/// size are the basic primitives of the cost computation (the x of the
/// cube/vector/transfer formulas are all derived from them), so owning them
/// as methods guarantees all three query paths share one evaluation and
/// validation logic (dynamic dims, empty shape, and unknown dtypes are
/// reported as failure() in exactly one place; the cost modules convert
/// that to a CostResult{-1, note} per §7 of the design doc) instead of each
/// module re-deriving its own multiply loop.
///
/// Ranks are unrestricted: a shape may have any number of dimensions (1-D
/// vector, 2-D matrix, 4-D fractal, 0-D scalar tensor are all legal) and
/// getTotalElements() multiplies over any rank.  The builder side must pass
/// the full, accurate shape (copied wholesale from ShapedType::getShape(),
/// never truncated or filled with placeholders); uncertain dimensions are
/// expressed as ShapedType::kDynamic and rejected by getTotalElements().
class ShapedOperandDesc {
public:
  ShapedOperandDesc() = default;
  ShapedOperandDesc(SmallVector<int64_t, ShapeInlineCapacity> shape,
                    Type elementType)
      : shape(std::move(shape)), elementType(elementType) {}

  /// Total element count = product of all shape dimensions.
  /// Handles any rank (including 0-D: an empty shape with no kDynamic dim
  /// counts as 1 element, i.e. a scalar tensor).  Any dimension being
  /// ShapedType::kDynamic or ≤ 0 → failure().  Ranks above
  /// ShapeInlineCapacity spill to the heap transparently; correctness never
  /// depends on the capacity constant.
  FailureOr<int64_t> getTotalElements() const;

  /// Total byte size = element count × dtype byte width, using the dtype
  /// table in MSCostParams.h.  A dtype outside the table → failure().
  /// A 0-D scalar tensor weighs one element's bytes.
  FailureOr<int64_t> getTotalBytes() const;

  const SmallVector<int64_t, ShapeInlineCapacity> &getShape() const {
    return shape;
  }
  Type getElementType() const { return elementType; }

private:
  /// All dimensions (full rank, never truncated).  Stored by value: the
  /// metadata source (MsNode::OperandInfo::shape) is a SmallVector whose
  /// storage moves while result.nodes grows, so an ArrayRef into it would
  /// dangle.  These structs are small; the copy is negligible.
  SmallVector<int64_t, ShapeInlineCapacity> shape;
  /// Element data type (MLIR Type).  Must be one of the dtypes accepted by
  /// the MSCostParams.h table.
  Type elementType;
};

} // namespace mlir::hivm::costmodel

#endif // BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_SHAPEDOPERANDDESC_H
