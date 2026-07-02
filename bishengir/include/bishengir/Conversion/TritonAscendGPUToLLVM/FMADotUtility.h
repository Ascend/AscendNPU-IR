//===-----------------------FMADotUtility.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_FMA_DOT_UTILITY_H
#define TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_FMA_DOT_UTILITY_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::ascend {

/// Abstract interface for scalar multiplication of Value vectors.
///
/// Enable generation of hardware specific code in different backends.
class FMAVectorMultiplier {
public:
  /// \returns scalar product of two arrays, plus c: a·b + c
  virtual Value multiplyVectors(const Value *a, const Value *b, Value c,
                                unsigned K) = 0;
  virtual Value promoteToAccType(Value v, Type accTy) = 0;
  virtual Value emitSingleFMA(Value a, Value b, Value acc) = 0;

  virtual ~FMAVectorMultiplier() = default;
};

/// Implements a framework for FMA dot conversion to llvm.
///
/// This function implements architecture independent part of FMA dot
/// conversion and calls "multiplier" object, which is defined by caller
/// and implements architecture dependant part of conversion.
LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier);

LogicalResult convertFMADot(DotOp op, DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

} // namespace mlir::triton::ascend

#endif // TRITON_CONVERSION_TRITONASCENDGPUTOLLVM_FMA_DOT_UTILITY_H
