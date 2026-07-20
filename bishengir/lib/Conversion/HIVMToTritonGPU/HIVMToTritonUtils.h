//===- HIVMToTritonUtils.h - shared HIVM->Triton pointer helpers ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_LIB_CONVERSION_HIVMTOTRITONGPU_HIVMTOTRITONUTILS_H
#define BISHENGIR_LIB_CONVERSION_HIVMTOTRITONGPU_HIVMTOTRITONUTILS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace hivm {

/// Builds a tensor of Triton pointers for a transfer with a static result shape.
///
/// `originalValue` is inspected for MemRef view semantics, including dynamic
/// subview offsets. `convertedValue` is used only to materialize converted base
/// pointers. The transfer `shape` must be static, but the MemRef layout may be
/// non-contiguous and may have a dynamic offset.
FailureOr<Value> buildMemRefTensorPointers(
    ConversionPatternRewriter &rewriter, Location loc, Value originalValue,
    Value convertedValue, MemRefType memrefTy, ArrayRef<int64_t> shape);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_LIB_CONVERSION_HIVMTOTRITONGPU_HIVMTOTRITONUTILS_H
