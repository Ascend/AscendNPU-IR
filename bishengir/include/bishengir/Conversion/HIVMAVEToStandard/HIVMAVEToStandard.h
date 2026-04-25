//===- HIVMAVEToStandard.h - Convert HIVMAVE dialect to Standard dialect --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HIVMAVETOSTANDARD_HIVMAVETOSTANDARD_H_
#define BISHENGIR_CONVERSION_HIVMAVETOSTANDARD_HIVMAVETOSTANDARD_H_

#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
class Pass;

#define GEN_PASS_DECL_CONVERTHIVMAVETOSTANDARD
#include "bishengir/Conversion/Passes.h.inc"

namespace hivmave {
/// Populate the given list with patterns that convert from HIVM to Standard.
void populateHIVMAVEToStandardConversionPatterns(RewritePatternSet &patterns);
} // namespace hivmave

/// Create a pass to convert HIVMAVE operations to the Standard dialect.
std::unique_ptr<Pass> createConvertHIVMAVEToStandardPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMAVETOSTANDARD_HIVMAVETOSTANDARD_H_
