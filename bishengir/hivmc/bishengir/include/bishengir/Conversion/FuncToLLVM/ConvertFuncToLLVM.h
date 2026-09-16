//===- ConvertFuncToLLVM.h - Convert Func to LLVM ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a set of conversion patterns from the Func dialect to the LLVM IR
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H
#define BISHENGIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace LLVM {
class LLVMFuncOp;
} // namespace LLVM

class FunctionOpInterface;
class ConversionPatternRewriter;
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class SymbolTable;
} // namespace mlir

namespace bishengir {

/// Convert input FunctionOpInterface operation to LLVMFuncOp by using the
/// provided LLVMTypeConverter. Return failure if failed to so.
mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(mlir::FunctionOpInterface funcOp,
                          mlir::ConversionPatternRewriter &rewriter,
                          const mlir::LLVMTypeConverter &converter);

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateFuncToLLVMFuncOpConversionPattern(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

} // namespace bishengir

#endif // BISHENGIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H
