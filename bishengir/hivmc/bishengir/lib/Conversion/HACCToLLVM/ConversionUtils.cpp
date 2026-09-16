//===- ConversionUtils.cpp - HACC to LLVM Conversion Utility ----*- C++ -*-===//
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

#include "bishengir/Conversion/HACCToLLVM/ConversionUtils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "convert-hacc-to-llvm"

namespace mlir {

namespace hacc {

std::string setAlignment(StringRef s, const uint8_t alignment) {
  std::string copied_s = s.str();
  copied_s += '\0';
  assert(alignment != 0);
  const uint8_t r = copied_s.size() % alignment;
  if (r == 0)
    return copied_s;
  copied_s.resize(copied_s.size() + (alignment - r), '\0');
  return copied_s;
}

Value loadFromPtr(Location loc, ConversionPatternRewriter &rewriter,
                  Value &valPtr) {
  auto alloca = valPtr.getDefiningOp<LLVM::AllocaOp>();
  if (!alloca) {
    LLVM_DEBUG(llvm::dbgs() << "Not an allocation Op";);
    return valPtr;
  }
  Type elementType = alloca.getElemType();
  return rewriter.create<LLVM::LoadOp>(loc, elementType, valPtr);
}

Value loadFromPtr(Location loc, ConversionPatternRewriter &rewriter,
                  LLVM::AddressOfOp &valPtr) {
  // TODO: Change the type to the real type of this valPtr
  return rewriter.create<LLVM::LoadOp>(loc, rewriter.getI32Type(), valPtr);
}

Value loadFromPtr(Location loc, ConversionPatternRewriter &rewriter,
                  LLVM::GEPOp &valPtr) {
  // TODO: Change the type to the real type of this valPtr
  return rewriter.create<LLVM::LoadOp>(
      loc, rewriter.getType<LLVM::LLVMPointerType>(), valPtr);
}

SmallVector<Value> storeArgs(ConversionPatternRewriter &rewriter,
                             LLVM::LLVMFuncOp &func) {
  OpBuilder::InsertionGuard guard(rewriter);

  auto loc = func.getLoc();
  rewriter.setInsertionPointToStart(&func.getBody().front());
  SmallVector<Value> allocatedArgs;

  // This defines a pointer in the default address space
  Type llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  auto one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                               rewriter.getI32IntegerAttr(1));
  for (const auto &arg : func.getArguments()) {
    // Allocate
    Value allocated = rewriter.create<LLVM::AllocaOp>(
        loc, /* Return val*/ llvmPtrTy,
        /* Type to be allocated */ arg.getType(), /* Size */ one);
    rewriter.create<LLVM::StoreOp>(loc, arg, allocated);
    allocatedArgs.push_back(allocated);
  }
  return allocatedArgs;
}

using ErrorOrMemBuf = llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>;

std::string getBinaryBuffer(ModuleOp module, StringRef filePath) {
  ErrorOrMemBuf binaryOrError = llvm::MemoryBuffer::getFileOrSTDIN(filePath);
  if (!binaryOrError.getError()) {
    return binaryOrError.get()->getBuffer().str();
  }
  module.emitWarning(filePath.str() + " not found, appending empty string");
  return "";
}

} // namespace hacc

} // namespace mlir
