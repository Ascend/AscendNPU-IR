//===- HIVMToTritonUtils.h - shared HIVM->Triton pointer helpers ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMTOTRITONGPU_HIVMTOTRITONUTILS_H
#define BISHENGIR_CONVERSION_HIVMTOTRITONGPU_HIVMTOTRITONUTILS_H

#include "bishengir/Conversion/HIVMToTritonGPU/MemRefDescriptor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace hivm {

/// Builds the pointer tile a 1:N MemRef descriptor describes, for the
/// consumer's own transfer `shape`. No view op is traced.
FailureOr<Value> buildMemRefDescriptorPointers(
    ConversionPatternRewriter &rewriter, Location loc,
    const MemRefDescriptor &desc, Type ptrTy, ArrayRef<int64_t> shape);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOTRITONGPU_HIVMTOTRITONUTILS_H
