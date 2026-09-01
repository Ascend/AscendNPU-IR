//===- TypeConverter.h - Convert builtin to LLVM dialect types --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a type converter configuration for converting most builtin types to
// LLVM dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_LLVMCOMMON_TYPECONVERTER_H
#define BISHENGIR_CONVERSION_LLVMCOMMON_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace bishengir {

/// Conversion from types to the LLVM IR dialect.
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  /// Create an LLVMTypeConverter using custom LowerToLLVMOptions. Optionally
  /// takes a data layout analysis to use in conversions.
  LLVMTypeConverter(mlir::MLIRContext *ctx,
                    const mlir::LowerToLLVMOptions &options,
                    const mlir::DataLayoutAnalysis *analysis = nullptr)
      : mlir::LLVMTypeConverter(ctx, options, analysis) {}

  /// Convert a function type. The arguments and results are converted one by
  /// one and results are packed into a wrapped LLVM IR structure type. `result`
  /// is populated with argument mapping.
  ///
  /// \param useBarePtrCallConvForArguments To control whether each argument is
  ///   converted with the bare ptr calling convention.
  /// \param useBarePtrCallConvForAllResults To control whether every result is
  ///   converted with the bare ptr calling convention.
  mlir::Type convertFunctionSignature(
      mlir::FunctionType funcTy, bool isVariadic,
      const llvm::SmallVector<bool> &useBarePtrCallConvForArguments,
      bool useBarePtrCallConvForAllResults,
      mlir::TypeConverter::SignatureConversion &result) const override;
};

} // namespace bishengir

#endif // BISHENGIR_CONVERSION_LLVMCOMMON_TYPECONVERTER_H
