//===- GetOrCreateUtils.cpp - Get or Create  Utility ------------*- C++ -*-===//
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

#include "bishengir/Conversion/HACCToLLVM/GetOrCreateUtils.h"
#include "bishengir/Conversion/HACCToLLVM/ConversionUtils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "convert-hacc-to-llvm"

namespace mlir {

namespace hacc {

LLVM::LLVMFuncOp getOrCreateRTLaunch(ModuleOp module,
                                     ConversionPatternRewriter &rewriter) {
  auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  auto llvmVoidTy = rewriter.getType<LLVM::LLVMVoidType>();
  return getOrCreateFuncOp(
      module, rewriter, "__cce_rtLaunch", llvmVoidTy,
      {llvmPtrTy, llvmPtrTy, rewriter.getI64Type(), rewriter.getI32Type()});
}

LLVM::LLVMFuncOp
getOrCreateRTSetupArgument(ModuleOp module,
                           ConversionPatternRewriter &rewriter) {
  auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  return getOrCreateFuncOp(
      module, rewriter, "rtSetupArgument", rewriter.getI32Type(),
      {llvmPtrTy, rewriter.getI64Type(), rewriter.getI64Type()});
}

LLVM::LLVMFuncOp
getOrCreateRTConfigureCall(ModuleOp module,
                           ConversionPatternRewriter &rewriter) {
  auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  return getOrCreateFuncOp(module, rewriter, "rtConfigureCall",
                           rewriter.getI32Type(),
                           {rewriter.getI32Type(), llvmPtrTy, llvmPtrTy});
}

LLVM::LLVMFuncOp
getOrCreateRTLinkedDevBinaryRegister(ModuleOp module,
                                     ConversionPatternRewriter &rewriter) {
  auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  return getOrCreateFuncOp(module, rewriter, "rtLinkedDevBinaryRegister",
                           rewriter.getType<LLVM::LLVMVoidType>(),
                           {llvmPtrTy, llvmPtrTy, rewriter.getI32Type()});
}

LLVM::LLVMFuncOp
getOrCreateRTFunctionRegister(ModuleOp module,
                              ConversionPatternRewriter &rewriter) {
  auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  return getOrCreateFuncOp(
      module, rewriter, "rtFunctionRegister", rewriter.getI32Type(),
      {llvmPtrTy, llvmPtrTy, llvmPtrTy, llvmPtrTy, rewriter.getI32Type()});
}

LLVM::GlobalOp
getOrCreateGlobalMainFuncName(ModuleOp module,
                              ConversionPatternRewriter &rewriter,
                              LLVM::LLVMFuncOp mainFunc) {
  return getOrCreateUnnamedGlobalOpByte(
      module, rewriter, "kernelName_" + mainFunc.getSymName().str(),
      mainFunc.getSymName());
}

LLVM::GlobalOp
getOrCreateGlobalMainFuncSignature(ModuleOp module,
                                   ConversionPatternRewriter &rewriter,
                                   llvm::StringRef initialName) {
  return getOrCreateUnnamedGlobalOpByte(
      module, rewriter, "kernelMangle_" + initialName.str(), initialName);
}

LLVM::GlobalOp getOrCreateBlockNumVar(ModuleOp module,
                                      ConversionPatternRewriter &rewriter) {
  llvm::StringRef symName = "blockNumSetterAddress";
  auto globalDecl = module.lookupSymbol<LLVM::GlobalOp>(symName);
  if (!globalDecl) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    globalDecl = rewriter.create<LLVM::GlobalOp>(
        module.getLoc(), rewriter.getI32Type(),
        /*isConstant=*/false, LLVM::Linkage::LinkonceODR, symName,
        rewriter.getI32IntegerAttr(0),
        /*alignment=*/4, 0, false, true); // Alignment for 32 bits are 4 bytes
    addComdat(module, rewriter, globalDecl);
  }
  return globalDecl;
}

LLVM::LLVMFuncOp
getOrCreateGetOrSetBlockNum(ModuleOp module,
                            ConversionPatternRewriter &rewriter) {
  llvm::StringRef blockNumSetter = "__cce_getOrSetBlockNum";
  auto setterFunc =
      getOrCreateFuncOp(module, rewriter, blockNumSetter, rewriter.getI32Type(),
                        {rewriter.getI32Type(), rewriter.getI32Type()},
                        LLVM::Linkage::LinkonceODR);
  if (setterFunc.getBody().empty()) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto *headBlock = setterFunc.addEntryBlock(rewriter);
    auto *endBlock = rewriter.createBlock(&setterFunc.getBody());
    auto loc = setterFunc.getLoc();
    SmallVector<Value> allocatedArgs = storeArgs(rewriter, setterFunc);
    rewriter.setInsertionPointToEnd(headBlock);
    auto arg1 = loadFromPtr(loc, rewriter, allocatedArgs[1]);
    auto cmpRes = rewriter.create<LLVM::ICmpOp>(
        loc, rewriter.getI1Type(), LLVM::ICmpPredicate::eq, arg1,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(0)));
    auto *middleBlock = rewriter.createBlock(
        &setterFunc.getBody(), std::next(Region::iterator(headBlock)));
    rewriter.setInsertionPointToEnd(headBlock);

    /* Create global op block num address */
    auto blockNumAddress = getOrCreateBlockNumVar(module, rewriter);
    /* Global op block num address done */

    rewriter.setInsertionPointToEnd(headBlock);
    LLVM::AddressOfOp addressPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, blockNumAddress);
    rewriter.create<LLVM::CondBrOp>(loc, cmpRes, middleBlock, endBlock);

    rewriter.setInsertionPointToEnd(middleBlock);

    auto arg0 = loadFromPtr(loc, rewriter, allocatedArgs[0]);
    rewriter.create<LLVM::StoreOp>(loc, arg0, addressPtr);

    rewriter.create<LLVM::BrOp>(loc, endBlock);
    rewriter.setInsertionPointToEnd(endBlock);
    auto blockNumPtr = loadFromPtr(loc, rewriter, addressPtr);
    rewriter.create<LLVM::ReturnOp>(loc, blockNumPtr);
    addComdat(module, rewriter, setterFunc);
  }
  return setterFunc;
}

LLVM::LLVMFuncOp
getOrCreateRTRegisterGlobals(ModuleOp module,
                             ConversionPatternRewriter &rewriter) {
  auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  auto llvmVoidTy = rewriter.getType<LLVM::LLVMVoidType>();
  auto registerGlobal =
      getOrCreateFuncOp(module, rewriter, "rtRegisterGlobals", llvmVoidTy,
                        {llvmPtrTy}, LLVM::Linkage::Internal);
  if (SymbolTable::symbolKnownUseEmpty(registerGlobal, module)) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto *headBlock = registerGlobal.addEntryBlock(rewriter);
    rewriter.setInsertionPointToStart(headBlock);
    rewriter.create<LLVM::ReturnOp>(registerGlobal.getLoc(), ValueRange{});
  }
  return registerGlobal;
}

LLVM::LLVMFuncOp
getOrCreateCCEModuleCtor(ModuleOp module, ConversionPatternRewriter &rewriter,
                         LLVM::GlobalOp cceWrapper,
                         LLVM::LLVMFuncOp &registerGlobalFunc) {
  OpBuilder::InsertionGuard guard(rewriter);
  std::string ctorFuncName = "cceModuleCtor";
  auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  auto llvmVoidTy = rewriter.getType<LLVM::LLVMVoidType>();
  auto ctorFunc = getOrCreateFuncOp(module, rewriter, ctorFuncName, llvmVoidTy,
                                    {llvmPtrTy}, LLVM::Linkage::Internal);
  if (ctorFunc.getBody().empty()) {
    auto loc = ctorFunc.getLoc();
    auto *headBlock = ctorFunc.addEntryBlock(rewriter);
    rewriter.setInsertionPointToEnd(headBlock);
    // Call runtime to setup argument
    auto cceWrapperPtr = rewriter.create<LLVM::AddressOfOp>(loc, cceWrapper);
    auto registerGlobalPtr =
        rewriter.create<LLVM::AddressOfOp>(loc, registerGlobalFunc);
    rewriter.create<LLVM::CallOp>(
        loc, getOrCreateRTLinkedDevBinaryRegister(module, rewriter),
        ArrayRef<Value>{
            cceWrapperPtr, registerGlobalPtr,
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                              rewriter.getI64IntegerAttr(0))});
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
  return ctorFunc;
}

LLVM::LLVMFuncOp
getOrCreatePrepareRTConfigureCall(ModuleOp module,
                                  ConversionPatternRewriter &rewriter) {
  llvm::StringRef funcName = "_Z21__cce_rtConfigureCalljPvS_";
  auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
  auto func =
      getOrCreateFuncOp(module, rewriter, funcName, rewriter.getI32Type(),
                        {rewriter.getI32Type(), llvmPtrTy, llvmPtrTy},
                        LLVM::Linkage::LinkonceODR);
  if (func.getBody().empty()) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto loc = func.getLoc();
    // Insert a return void terminator
    auto *headBlock = func.addEntryBlock(rewriter);

    SmallVector<Value> allocatedArgs = storeArgs(rewriter, func);
    rewriter.setInsertionPointToEnd(headBlock);
    auto blockNum = loadFromPtr(loc, rewriter, allocatedArgs[0]);

    auto blockNumSetter = rewriter.create<LLVM::CallOp>(
        loc, getOrCreateGetOrSetBlockNum(module, rewriter),
        ArrayRef<Value>{blockNum, rewriter.create<LLVM::ConstantOp>(
                                      loc, rewriter.getI32Type(),
                                      rewriter.getI32IntegerAttr(0))});
    if (blockNumSetter.getResults().size() != 1) {
      llvm_unreachable("blockNumSetter should only result 1");
    }

    auto rt0 = loadFromPtr(loc, rewriter, allocatedArgs[0]);
    auto rt1 = loadFromPtr(loc, rewriter, allocatedArgs[1]);
    auto rt2 = loadFromPtr(loc, rewriter, allocatedArgs[2]);

    auto configureCall = rewriter.create<LLVM::CallOp>(
        loc, getOrCreateRTConfigureCall(module, rewriter),
        ArrayRef<Value>{rt0, rt1, rt2});
    assert(configureCall.getResults().size() == 1);
    rewriter.create<LLVM::ReturnOp>(loc, configureCall.getResults()[0]);
    addComdat(module, rewriter, func);
  }
  return func;
}

LLVM::GlobalOp getOrCreateGlobalCtors(ModuleOp module,
                                      ConversionPatternRewriter &rewriter,
                                      LLVM::LLVMFuncOp ref) {
  llvm::StringRef ctorsName = "llvm.global_ctors";
  auto ctors = module.lookupSymbol<LLVM::GlobalOp>(ctorsName);
  if (!ctors) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    // Defining Types
    auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
    auto structType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {rewriter.getI32Type(), llvmPtrTy, llvmPtrTy});
    // TODO:: seems can append
    auto type = LLVM::LLVMArrayType::get(structType, 1);

    ctors = rewriter.create<LLVM::GlobalOp>(
        module.getLoc(), type, /*isConstant=*/false, LLVM::Linkage::Appending,
        ctorsName, rewriter.getZeroAttr(structType));

    Location loc = ctors.getLoc();
    Region &region = ctors.getInitializerRegion();
    Block *block = rewriter.createBlock(&region);

    // Initialize the struct and set the execution mode value.
    rewriter.setInsertionPoint(block, block->begin());
    Value structValue = rewriter.create<LLVM::UndefOp>(loc, type);

    // The first field of wrapper is the ASCII number of our devive;
    // TODO: Change it to;
    Value entry = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(65535));
    structValue = rewriter.create<LLVM::InsertValueOp>(
        loc, structValue, entry, ArrayRef<int64_t>({0, 0}));

    // The second field should be cceModuleCtor reference.
    entry = rewriter.create<LLVM::AddressOfOp>(loc, ref);
    structValue = rewriter.create<LLVM::InsertValueOp>(
        loc, structValue, entry, ArrayRef<int64_t>({0, 1}));

    // The third field of wrapper is the elf data that we've generated.
    entry = rewriter.create<LLVM::ZeroOp>(loc, llvmPtrTy);
    structValue = rewriter.create<LLVM::InsertValueOp>(
        loc, structValue, entry, ArrayRef<int64_t>({0, 2})); // insert at idx 2

    rewriter.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({structValue}));
  }
  return ctors;
}

LLVM::GlobalOp getOrCreateBinElf(ModuleOp module,
                                 ConversionPatternRewriter &rewriter,
                                 StringRef elf) {
  const std::string symName = "binElf";
  auto globalDecl = getOrCreateUnnamedGlobalOpByte</* alignment= */ 8>(
      module, rewriter, symName, elf.str());
  // TODO: Replace the hello.so with option pass-in value
  globalDecl.setSectionAttr(rewriter.getStringAttr("__aicore_rel_binary"));
  return globalDecl;
}

LLVM::GlobalOp getOrCreateCCEPTCWrapper(ModuleOp module,
                                        ConversionPatternRewriter &rewriter,
                                        const std::string &filepath) {
  llvm::StringRef globalName = "__cce_ptc_wrapper";
  auto maybeGlobal = module.lookupSymbol<LLVM::GlobalOp>(globalName);
  if (maybeGlobal)
    return maybeGlobal;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  // Defining Types
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto structType = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {rewriter.getI32Type(), rewriter.getI32Type(),
                              ptrType, rewriter.getI64Type()});

  auto global = rewriter.create<LLVM::GlobalOp>(
      module.getLoc(), structType, /*isConstant=*/true, LLVM::Linkage::Internal,
      globalName, Attribute(),
      /*alignment=*/8); // struct type has bigger alignmen, use 8 bytes

  Location loc = global.getLoc();
  Region &region = global.getInitializerRegion();
  Block *block = rewriter.createBlock(&region);

  // Initialize the struct and set the execution mode value.
  rewriter.setInsertionPoint(block, block->begin());
  Value structValue = rewriter.create<LLVM::UndefOp>(loc, structType);

  // The first field of wrapper is the ASCII number of our devive;
  // TODO: Change it to;
  constexpr uint32_t kAICoreMagicNumberVec = 0x41415246;
  Value entry = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(kAICoreMagicNumberVec));
  structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, entry,
                                                     ArrayRef<int64_t>({0}));

  // The second field of wrapper is the fat bin version number, it's 1 now.
  // TODO: consider thin bin and others.
  entry = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                            rewriter.getI32IntegerAttr(1));
  structValue = rewriter.create<LLVM::InsertValueOp>(loc, structValue, entry,
                                                     ArrayRef<int64_t>({1}));

  // The forth field of wrapper is the size of elf data.
  // TODO: This .so file need to be pass in with option.
  std::string elf = getBinaryBuffer(module, filepath);

  // The third field of wrapper is the elf data that we've generated.
  entry = rewriter.create<LLVM::AddressOfOp>(
      loc, getOrCreateBinElf(module, rewriter, elf));
  structValue = rewriter.create<LLVM::InsertValueOp>(
      loc, structValue, entry, ArrayRef<int64_t>({2})); // insert at index 2

  entry = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(elf.size()));
  structValue = rewriter.create<LLVM::InsertValueOp>(
      loc, structValue, entry,
      ArrayRef<int64_t>({3})); // insert at index 3 for this elf

  rewriter.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({structValue}));

  // Set additional attributes
  global.setSectionAttr(rewriter.getStringAttr("__aicore_rel_rec"));
  global.setAlignmentAttr(IntegerAttr::get(rewriter.getI64Type(),
                                           8)); // 8 bytes alignment for 64 type
  return global;
}

LLVM::LLVMFuncOp getOrCreateFuncOp(ModuleOp module,
                                   ConversionPatternRewriter &rewriter,
                                   StringRef funcName, Type returnType,
                                   ArrayRef<Type> argumentTypes,
                                   LLVM::Linkage linkage) {
  auto funcDeclaration = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!funcDeclaration) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto loc = module.getLoc();
    funcDeclaration = rewriter.create<LLVM::LLVMFuncOp>(
        loc, funcName, LLVM::LLVMFunctionType::get(returnType, argumentTypes),
        linkage);
  }
  return funcDeclaration;
}

LLVM::LLVMFuncOp getOrCreateFuncOp(ModuleOp module,
                                   ConversionPatternRewriter &rewriter,
                                   llvm::StringRef funcName, Type returnType,
                                   ArrayRef<Type> argumentTypes) {
  return getOrCreateFuncOp(module, rewriter, funcName, returnType,
                           argumentTypes, LLVM::Linkage::External);
}

} // namespace hacc

} // namespace mlir
