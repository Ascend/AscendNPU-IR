//===- CostModelTypes.h - shared data structures for cost models -*- C++
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
// Data structures passed between MsDagBuilder and the cost models, plus the
// ShapedOperandDesc value object that owns the shared shape/dtype evaluation
// used by the cube, vector, and transfer modules.  This header contains no
// cost parameters — see MSCostParams.h for the parameter tables.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_COSTMODELTYPES_H
#define BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_COSTMODELTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <utility>

namespace mlir::hivm::costmodel {

// The structures below spell SmallVector/StringRef unqualified (matching
// the surrounding MLIR style); pull the llvm types in locally so this
// header compiles standalone without transitive mlir/Support/LLVM.h.
using llvm::SmallVector;
using llvm::StringRef;

// ═══════════ SmallVector inline capacity constants ═══════════
//
// llvm::SmallVector<T, N>'s N is the inline storage capacity (small-size,
// in elements): with ≤ N elements the vector lives in the inline array with
// no heap allocation; beyond N it spills to the heap.  N only affects
// performance, never semantics — every field below is an unbounded dynamic
// container that grows automatically (2x growth, amortized O(1)) and never
// truncates.  The named constants below only keep typical small objects
// allocation-free; the builder side reserves exact/upper-bound capacities
// at fill time (see buildVectorGroupCostInput).

/// Inline dimension capacity for shapes: covers typical ranks (including
/// the 4-D cube fractal layout).
inline constexpr size_t ShapeInlineCapacity = 4;
/// Inline capacity for one op's input blocks: common ops take 1-2, bias and
/// other extra inputs fit within 4.
inline constexpr size_t InputsInlineCapacity = 4;
/// Inline capacity for one op's output blocks: HIVM DPS ops are essentially
/// single-output; two outputs is already rare.
inline constexpr size_t OutputsInlineCapacity = 2;
/// Inline capacity for the ops of a vector group.  Group sizes vary widely
/// and have no upper bound (small groups have 1-2 ops, fused groups can
/// reach dozens): this constant only decides whether small groups stay
/// allocation-free.  The builder reserves sn.ops.size() up front.
inline constexpr size_t GroupOpsInlineCapacity = 4;

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

/// Cube query input: the two input matrices (A/B).
struct CubeCostInput {
  ShapedOperandDesc lhs; // operandIndex 0 (A)
  ShapedOperandDesc rhs; // operandIndex 1 (B)
};

/// Description of one op inside a vector group.
///
/// Inputs and outputs are variable-length lists: a single vector op can
/// consume multiple shaped inputs (e.g. add(lhs, rhs, bias) has 3 inputs,
/// each with its own (shape, elementType)); outputs (multiple results)
/// likewise.  Scalar / non-shaped operands (index, i1, ...) never enter
/// the lists.
struct VectorOpDesc {
  /// Owner op.  The model reads its op name twice: to match the ignore
  /// list (non-vector plumbing ops carry no latency) and to assemble the
  /// dtype fallback note.  The cost formula never dereferences it
  /// otherwise — op KIND does not enter the billing.
  Operation *op = nullptr;

  /// All shaped inputs of this op, in operand order.
  SmallVector<ShapedOperandDesc, InputsInlineCapacity> inputs;

  /// All shaped outputs (results) of this op, in result order.  Kept for
  /// diagnostics; the cost formula is driven by the inputs alone.
  SmallVector<ShapedOperandDesc, OutputsInlineCapacity> outputs;
};

/// Vector-group query input: the group's ops in block order.  Op boundaries
/// come from OperandInfo::ownerOp.  Transfer ops (load/nd2nz/...) never
/// enter ops — their cost is modeled separately by the transfer module
/// keyed on (pipeline, storageTag); counting them here would double-bill.
/// The op count is dynamic and unbounded; GroupOpsInlineCapacity is only
/// the inline-optimization threshold.
struct VectorGroupCostInput {
  SmallVector<VectorOpDesc, GroupOpsInlineCapacity> ops;
};

/// Transfer query input: the moved data block plus the two tags identifying
/// the module and direction.  pipeline + storageTag jointly select the
/// parameter row (same execution unit, different directions → different
/// parameters).
struct TransferCostInput {
  ShapedOperandDesc data; // the moved block (shape + dtype)
  StringRef pipeline;     // "mte1", "mte2", "mte3", "fixpipe"
  StringRef storageTag;   // "GM_to_L1", "GM_to_UB", "L0C_to_L1", ...
};

/// Result of one cost query.  Queries never return failure(): apart from
/// vector ops whose dtype is outside the dtype table (warning + fallback
/// computation, still a positive result), every anomaly is expressed as
/// costTime = -1 plus a note carrying the specifics; the caller
/// (MsDagBuilder::assignCosts) prints the note and fails the build.
struct CostResult {
  /// Latency costTime in NANOSECONDS (the linear model y = a·x + b, fitted
  /// in μs, scaled by CostTimeUnitScale and normalized to an integer).
  /// Two possible kinds of values:
  ///   - success: max(1, round(y·1000)), always > 0.  Written directly into
  ///     the DDG latency/issueTime with no further conversion;
  ///   - failure: -1 (error sentinel) — the query could not produce a
  ///     trustworthy cost.  note is then mandatory and carries the
  ///     specifics (which input, which dimension, what dtype/direction,
  ///     why) for the caller to print.
  /// No value other than -1 is ever zero or negative.
  int64_t costTime;

  /// Note: auxiliary information from the computation.  By costTime:
  ///   - == -1 (error): the specifics (mandatory), e.g. "cube lhs dim 1 is
  ///     dynamic", "transfer (mte3, UB_to_UB) not in param table"; the
  ///     caller prints it and fails the build;
  ///   - > 0 (success): diagnostic note.  The only current source is
  ///     vector ops whose dtype is outside the dtype table (the op names
  ///     that hit the fallback multiplier; the caller emits a warning but
  ///     the computation completed and scheduling continues).  Empty
  ///     string means "no note".
  /// The note never participates in any computation.
  std::string note;
};

} // namespace mlir::hivm::costmodel

#endif // BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_COSTMODELTYPES_H
