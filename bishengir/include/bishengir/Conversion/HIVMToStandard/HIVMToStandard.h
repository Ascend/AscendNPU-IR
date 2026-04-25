//===- HIVMToStandard.h - Convert HIVM dialect to Standard dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMTOSTANDARD_HIVMTOSTANDARD_H_
#define BISHENGIR_CONVERSION_HIVMTOSTANDARD_HIVMTOSTANDARD_H_

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

#define GEN_PASS_DECL_CONVERTHIVMTOSTANDARD
#include "bishengir/Conversion/Passes.h.inc"

namespace hivm {
/// Populate the given list with patterns that convert from HIVM to Standard.
void populateHIVMToStandardConversionPatterns(RewritePatternSet &patterns,
                                              bool isOpsAligned = false);
} // namespace hivm

/// Create a pass to convert HIVM operations to the Standard dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertHIVMToStandardPass(
    const ConvertHIVMToStandardOptions &options = {});

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOSTANDARD_HIVMTOSTANDARD_H_
