//===- ConversionUtils.h - HACC to LLVM Conversion Utility ------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the HACC dialect to the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#ifndef BISHENGIR_CONVERSION_HACCTOLLVM_CONVERSIONUTILS_H
#define BISHENGIR_CONVERSION_HACCTOLLVM_CONVERSIONUTILS_H

namespace mlir {

namespace hacc {

std::string setAlignment(StringRef s, uint8_t alignment);

template <class T>
LLVM::ComdatOp addComdat(ModuleOp module, ConversionPatternRewriter &rewriter,
                         T &op) {
  const std::string comdatName = "__llvm_comdat_" + op.getSymName().str();
  auto comdatOp = module.lookupSymbol<LLVM::ComdatOp>(comdatName);
  if (!comdatOp) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    comdatOp = rewriter.create<LLVM::ComdatOp>(module.getLoc(), comdatName);
    rewriter.setInsertionPointToStart(&comdatOp.getBody().back());
    auto selectorOp = rewriter.create<LLVM::ComdatSelectorOp>(
        comdatOp.getLoc(), op.getSymName(), LLVM::comdat::Comdat::Any);
    op.setComdatAttr(SymbolRefAttr::get(
        rewriter.getContext(), comdatName,
        FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
  }
  return comdatOp;
}

template <uint8_t alignment = 1>
LLVM::GlobalOp
getOrCreateUnnamedGlobalOpByte(ModuleOp module,
                               ConversionPatternRewriter &rewriter,
                               StringRef symName, StringRef content) {
  auto globalDecl = module.lookupSymbol<LLVM::GlobalOp>(symName);
  if (globalDecl)
    return globalDecl;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  std::string alignedContent = setAlignment(content, alignment);
  auto byteType = rewriter.getType<LLVM::LLVMArrayType>(
      rewriter.getIntegerType(8), alignedContent.size());
  globalDecl = rewriter.create<LLVM::GlobalOp>(
      module.getLoc(), byteType, /*isConstant=*/true, LLVM::Linkage::Private,
      symName, rewriter.getStringAttr(alignedContent),
      /* alignment= */ alignment);
  globalDecl.setUnnamedAddr(LLVM::UnnamedAddr::Global);
  return globalDecl;
}

Value loadFromPtr(Location loc, ConversionPatternRewriter &rewriter,
                  Value &valPtr);

Value loadFromPtr(Location loc, ConversionPatternRewriter &rewriter,
                  LLVM::AddressOfOp &valPtr);

Value loadFromPtr(Location loc, ConversionPatternRewriter &rewriter,
                  LLVM::GEPOp &valPtr);

SmallVector<Value> storeArgs(ConversionPatternRewriter &rewriter,
                             LLVM::LLVMFuncOp &func);

std::string getBinaryBuffer(ModuleOp module, StringRef filePath);

} // namespace hacc

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HACCTOLLVM_CONVERSIONUTILS_H
