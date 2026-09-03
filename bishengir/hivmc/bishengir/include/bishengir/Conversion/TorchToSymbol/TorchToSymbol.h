//===--TorchToSymbol.h - Torch to Symbol Dialect Conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_TORCHTOSYMBOL_TORCHTOSYMBOL_H
#define BISHENGIR_CONVERSION_TORCHTOSYMBOL_TORCHTOSYMBOL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace symbol {
/// Populate patterns to convert torch dialect ops to symbol dialect ops.
void populatePatternsAndLegality(TypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ConversionTarget &target);

} // namespace symbol

#define GEN_PASS_DECL_CONVERTTORCHTOSYMBOL
#include "bishengir/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToSymbolPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_TORCHTOSYMBOL_TORCHTOSYMBOL_H
