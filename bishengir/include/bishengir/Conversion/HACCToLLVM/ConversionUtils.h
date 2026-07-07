//===- ConversionUtils.h - HACC to LLVM Conversion Utility ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
