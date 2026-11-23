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
// Data structures passed between MsDagBuilder and the cost models.  The
// ShapedOperandDesc value object they carry lives in ShapedOperandDesc.h,
// which this header pulls in so including CostModelTypes.h alone is enough
// to build a query input.  This header contains no cost parameters — see
// MSCostParams.h for the parameter tables.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_COSTMODELTYPES_H
#define BISHENGIR_DIALECT_HIVM_UTILS_COSTMODEL_COSTMODELTYPES_H

#include "bishengir/Dialect/HIVM/Utils/PipeSchedule/CostModel/ShapedOperandDesc.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::hivm::costmodel {

// See the note in ShapedOperandDesc.h on spelling the llvm types
// unqualified; SmallVector is already pulled in there.
using llvm::StringRef;

// The inline capacity constants below complete the set opened in
// ShapedOperandDesc.h (alongside ShapeInlineCapacity); that header's
// ═══ SmallVector inline capacity constants ═══ block explains why N is a
// pure performance knob with no bearing on semantics.

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
