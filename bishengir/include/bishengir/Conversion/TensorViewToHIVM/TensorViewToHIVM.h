//===- TensorViewToHIVM.h ---------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BISHENGIR_CONVERSION_TENSORVIEWTOHIVM_TENSORVIEWTOHIVM_H
#define BISHENGIR_CONVERSION_TENSORVIEWTOHIVM_TENSORVIEWTOHIVM_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
#define GEN_PASS_DECL_CONVERTTENSORVIEWTOHIVM
#include "bishengir/Conversion/Passes.h.inc"
namespace hivm {
void populateTensorViewToHIVMConversionPatterns(RewritePatternSet &patterns);
} // namespace hivm

std::unique_ptr<Pass> createTensorViewToHIVMConversionPass();
} // namespace mlir

#endif
