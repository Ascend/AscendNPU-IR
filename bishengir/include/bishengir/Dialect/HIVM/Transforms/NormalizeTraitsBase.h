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

#ifndef BISHENGIR_LIB_DIALECT_HIVM_TRANSFORMS_NORMALIZETRAITSBASE_H
#define BISHENGIR_LIB_DIALECT_HIVM_TRANSFORMS_NORMALIZETRAITSBASE_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Transforms/Normalize/Utils/Kinds.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir::hivm {

/// Base traits class for HIVM Normalize operations.
/// Provides common utility methods that can be reused by specific traits.
struct NormalizeTraitsBase {
public:
  static bool matchOp(Operation *op, UnaryKind kind);

  static bool matchOp(Operation *op, BinaryKind kind);

  static Value createCmpOp(PatternRewriter &rewriter, Location loc,
                           Value input, Value dst, CompareKind kind);

  static Value createUnaryOp(PatternRewriter &rewriter, Location loc,
                             Value input, Value dst, UnaryKind kind);

  static Value createBinaryOp(PatternRewriter &rewriter, Location loc,
                              Value lhs, Value rhs, Value dst,
                              BinaryKind kind);

  static Value createCastOp(PatternRewriter &rewriter, Location loc,
                            Value input, Type targetElemType,
                            CastRoundKind kind);
  static Value createFillOp(PatternRewriter &rewriter, Location loc,
                            Value input, Value dst);

  static Value createBitcastOp(PatternRewriter &rewriter, Location loc,
                               Type resultType, Value source);

  static bool matchCastRoundMode(VCastOp op, CastExecutionKind kind);

  static bool matchCastUnsignedMode(VCastOp op, CastUnsignedModeKind kind);

  static TypeFn mapCastSignKind(CastSignKind kind, TypeFn preserveTypeFn);

  static RoundMode mapCastExecutionKind(CastExecutionKind kind,
                                        RoundMode defaultRoundMode);

  static UnsignedMode mapCastUnsignedModeKind(CastUnsignedModeKind kind,
                                              UnsignedMode preserveMode);

  static bool archIsRegbased();

  static Value castValue(PatternRewriter &rewriter, Location loc, VCastOp op,
                         Value input, Type targetElemType,
                         CastExecutionKind executionKind = CastExecutionKind::Default,
                         CastSignKind signKind = CastSignKind::Preserve,
                         bool enableSaturate = false,
                         CastUnsignedModeKind unsignedModeKind = CastUnsignedModeKind::Preserve);
};

} // namespace mlir::hivm

#endif // BISHENGIR_LIB_DIALECT_HIVM_TRANSFORMS_NORMALIZETRAITSBASE_H
