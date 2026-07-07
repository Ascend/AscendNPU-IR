//===- GetOrCreateUtils.h - Get or Create  Utility --------------*- C++ -*-===//
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

#include "bishengir/Conversion/HACCToLLVM/ConversionUtils.h"
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

#ifndef BISHENGIR_CONVERSION_HACCTOLLVM_GETORCREATEUTILS_H
#define BISHENGIR_CONVERSION_HACCTOLLVM_GETORCREATEUTILS_H

namespace mlir {

namespace hacc {

LLVM::LLVMFuncOp getOrCreateRTLaunch(ModuleOp module,
                                     ConversionPatternRewriter &rewriter);

LLVM::LLVMFuncOp
getOrCreateRTSetupArgument(ModuleOp module,
                           ConversionPatternRewriter &rewriter);

LLVM::LLVMFuncOp
getOrCreateRTConfigureCall(ModuleOp module,
                           ConversionPatternRewriter &rewriter);

LLVM::LLVMFuncOp
getOrCreateRTLinkedDevBinaryRegister(ModuleOp module,
                                     ConversionPatternRewriter &rewriter);
LLVM::LLVMFuncOp
getOrCreateRTFunctionRegister(ModuleOp module,
                              ConversionPatternRewriter &rewriter);
LLVM::GlobalOp
getOrCreateGlobalMainFuncName(ModuleOp module,
                              ConversionPatternRewriter &rewriter,
                              LLVM::LLVMFuncOp mainFunc);
LLVM::GlobalOp
getOrCreateGlobalMainFuncSignature(ModuleOp module,
                                   ConversionPatternRewriter &rewriter,
                                   llvm::StringRef initialName);

LLVM::GlobalOp getOrCreateBlockNumVar(ModuleOp module,
                                      ConversionPatternRewriter &rewriter);

LLVM::LLVMFuncOp
getOrCreateGetOrSetBlockNum(ModuleOp module,
                            ConversionPatternRewriter &rewriter);

LLVM::LLVMFuncOp
getOrCreateRTRegisterGlobals(ModuleOp module,
                             ConversionPatternRewriter &rewriter);

LLVM::LLVMFuncOp getOrCreateCCEModuleCtor(ModuleOp module,
                                          ConversionPatternRewriter &rewriter,
                                          LLVM::GlobalOp cceWrapper,
                                          LLVM::LLVMFuncOp &registerGlobalFunc);

LLVM::LLVMFuncOp
getOrCreatePrepareRTConfigureCall(ModuleOp module,
                                  ConversionPatternRewriter &rewriter);

LLVM::GlobalOp getOrCreateGlobalCtors(ModuleOp module,
                                      ConversionPatternRewriter &rewriter,
                                      LLVM::LLVMFuncOp ref);

LLVM::GlobalOp getOrCreateBinElf(ModuleOp module,
                                 ConversionPatternRewriter &rewriter,
                                 StringRef elf);

LLVM::GlobalOp getOrCreateCCEPTCWrapper(ModuleOp module,
                                        ConversionPatternRewriter &rewriter,
                                        const std::string &filepath);

LLVM::LLVMFuncOp getOrCreateFuncOp(ModuleOp module,
                                   ConversionPatternRewriter &rewriter,
                                   StringRef funcName, Type returnType,
                                   ArrayRef<Type> argumentTypes,
                                   LLVM::Linkage linkage);

LLVM::LLVMFuncOp getOrCreateFuncOp(ModuleOp module,
                                   ConversionPatternRewriter &rewriter,
                                   llvm::StringRef funcName, Type returnType,
                                   ArrayRef<Type> argumentTypes);

} // namespace hacc

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HACCTOLLVM_GETORCREATEUTILS_H
