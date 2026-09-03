//===- TypeConverter.cpp - Convert builtin to LLVM dialect types ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/LLVMCommon/TypeConverter.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Threading.h"

#include <memory>
#include <mutex>
#include <optional>

using namespace mlir;

Type bishengir::LLVMTypeConverter::convertFunctionSignature(
    FunctionType funcTy, bool isVariadic,
    const SmallVector<bool> &useBarePtrCallConvForArguments,
    bool useBarePtrCallConvForAllResults, SignatureConversion &result) const {
  if (useBarePtrCallConvForArguments.size() != funcTy.getNumInputs())
    llvm_unreachable(
        "invalid arguments passed, number funcTy inputs doesn't match "
        "useBarePtrCallConvForArguments size");

  // Convert argument types one by one and check for errors.
  for (auto [idx, type] : llvm::enumerate(funcTy.getInputs())) {
    SmallVector<Type, 8> converted;
    // Select the argument converter depending on the calling convention.
    auto funcArgConverter = useBarePtrCallConvForArguments[idx]
                                ? barePtrFuncArgTypeConverter
                                : structFuncArgTypeConverter;
    if (failed(funcArgConverter(*this, type, converted)))
      return {};
    result.addInputs(idx, converted);
  }

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types into
  // a struct.

  // TODO: implement on-demand bare-ptr support for function results
  Type resultType = funcTy.getNumResults() == 0
                        ? LLVM::LLVMVoidType::get(&getContext())
                        : packFunctionResults(funcTy.getResults(),
                                              useBarePtrCallConvForAllResults);
  if (!resultType)
    return {};
  return LLVM::LLVMFunctionType::get(resultType, result.getConvertedTypes(),
                                     isVariadic);
}
