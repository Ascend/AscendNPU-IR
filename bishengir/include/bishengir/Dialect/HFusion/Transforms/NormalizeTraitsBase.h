//===- NormalizeTraitsBase.h -----------------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_LIB_DIALECT_HFUSION_TRANSFORMS_NORMALIZETRAITSBASE_H
#define BISHENGIR_LIB_DIALECT_HFUSION_TRANSFORMS_NORMALIZETRAITSBASE_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Transforms/Normalize/Utils/Kinds.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace mlir::hfusion {

class CompareOp;
class ElemwiseBinaryOp;

/// Base traits class for HFusion Normalize operations.
/// Provides common utility methods that can be reused by specific traits.
struct NormalizeTraitsBase {
public:
  static bool matchOp(Operation *op, UnaryKind kind);

  static bool matchOp(Operation *op, BinaryKind kind);

  static Value createCmpOp(PatternRewriter &rewriter, Location loc,
                           Value input, Value dst, CompareKind kind);

  static Value createCmpOp(PatternRewriter &rewriter, Location loc, Value lhs,
                           Value rhs, CompareOp sourceOp);

  static Value createUnaryOp(PatternRewriter &rewriter, Location loc,
                             Value input, Value dst, UnaryKind kind);

  static Value createBinaryOp(PatternRewriter &rewriter, Location loc,
                              Value lhs, Value rhs, Value dst,
                              BinaryKind kind);

  /// Create a dialect cast preserving the source dialect round mode when one is
  /// already available. If no round mode is provided, the dialect default is
  /// selected by the underlying cast builder.
  static Value createCastOp(PatternRewriter &rewriter, Location loc,
                            Value input, Type targetElemType,
                            std::optional<RoundMode> roundMode = std::nullopt);

  /// Create a dialect cast from a normalize-template abstract round kind.
  static Value createCastOp(PatternRewriter &rewriter, Location loc,
                            Value input, Type targetElemType,
                            CastRoundKind kind = CastRoundKind::Default,
                            Value output = Value(),
                            CastSignKind signKind = CastSignKind::Signed);

  static Value createShiftOp(PatternRewriter &rewriter, Location loc,
                             Value lhs, Value rhs, Value dst, ShiftKind kind,
                             Operation *sourceOp = nullptr);

  static Value createFillOp(PatternRewriter &rewriter, Location loc,
                            Value input, Value dst);

  static bool matchFillOp(Operation *op);

  static Value getFillInput(Operation *op);

  static bool matchBroadcastOp(Operation *op);

  static Value getBroadcastInput(Operation *op);

  static SmallVector<int64_t> getBroadcastDims(Operation *op);

  static Value createBroadcastOp(PatternRewriter &rewriter, Location loc,
                                 Value input, Value dst,
                                 ArrayRef<int64_t> dims);

  static Value createBitcastOp(PatternRewriter &rewriter, Location loc,
                               Type resultType, Value source);

  static bool matchCastRoundMode(CastOp op, CastRoundKind kind);

  static bool matchCastUnsignedMode(CastOp op, CastUnsignedModeKind kind);

  static TypeFn mapCastSignKind(CastSignKind kind, TypeFn preserveTypeFn);

  static RoundMode mapCastRoundKind(CastRoundKind kind,
                                        RoundMode defaultRoundMode);

  static UnsignedMode mapCastUnsignedModeKind(CastUnsignedModeKind kind,
                                              UnsignedMode preserveMode);

  static bool archIsRegbased();

  static Value createCastValueFromSourceOp(PatternRewriter &rewriter,
                                           Location loc, CastOp op,
                                           Value input, Type targetElemType,
                                           CastRoundKind executionKind =
                                               CastRoundKind::Default,
                                           CastSignKind signKind =
                                               CastSignKind::Preserve,
                                           bool enableSaturate = false,
                                           CastUnsignedModeKind
                                               unsignedModeKind =
                                                   CastUnsignedModeKind::
                                                       Preserve);

  static Value castScalarThroughTensor(PatternRewriter &rewriter, Location loc,
                                       Value scalar, Type dstType);
  static Value createSelectOp(PatternRewriter &rewriter, Location loc,
                              Value cond, Value a, Value b, Value dst);

  static Value createIsInfOp(PatternRewriter &rewriter, Location loc, Value y);
};

} // namespace mlir::hfusion

#endif // BISHENGIR_LIB_DIALECT_HFUSION_TRANSFORMS_NORMALIZETRAITSBASE_H
