//===- HACCToLLVMIRTranslation.cpp - Translate HACC to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the HACC dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Target/LLVMIR/Dialect/HACC/HACCToLLVMIRTranslation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/IR/IRBuilder.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
/// Implementation of the dialect interface that converts operations belonging
/// to the HACC dialect to LLVM IR.
class HACCDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    return failure();
  }

  /// Attaches module-level metadata for functions marked as kernels.
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {
    LLVM::LLVMFuncOp func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func) {
      return success();
    }
    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    auto haccAttr =
        hacc::symbolizeHACCToLLVMIRTranslateAttr(attribute.getName());
    if (!haccAttr.has_value()) {
      if (allowedHACCAttr(attribute.getName()))
        return success();
      return func->emitOpError()
             << "Unsupported attribute type: " << attribute.getName();
    }
    switch (haccAttr.value()) {
    case hacc::HACCToLLVMIRTranslateAttr::ENTRY: {
      // mark the function as entry using this annotation
      llvm::Module *llvmModule = llvmFunc->getParent();
      // get "hacc.annotations" metadata node
      llvm::NamedMDNode *MD =
          llvmModule->getOrInsertNamedMetadata("hivm.annotations");
      llvm::LLVMContext &Ctx = llvmModule->getContext();
      llvm::Metadata *MDVals[] = {
          llvm::ConstantAsMetadata::get(llvmFunc),
          llvm::MDString::get(Ctx, "kernel"),
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1))};
      // append metadata to hacc.annotations
      MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
    } break;
    case hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE: {
      llvm::LLVMContext &Ctx = llvmFunc->getContext();
      llvmFunc->addFnAttr(
          llvm::Attribute::get(Ctx, llvm::Attribute::AlwaysInline));
    } break;
    case hacc::HACCToLLVMIRTranslateAttr::NOINLINE: {
      llvm::LLVMContext &Ctx = llvmFunc->getContext();
      llvmFunc->addFnAttr(
          llvm::Attribute::get(Ctx, llvm::Attribute::NoInline));
    } break;
    case hacc::HACCToLLVMIRTranslateAttr::MIX_ENTRY:
      llvm_unreachable("unsupported attribute: hacc.mix_entry");
    }
    // mark the function as dso_local calling convention by default
    llvmFunc->setDSOLocal(true);

    return success();
  }

private:
  const StringSet<> kAllowedAttrNames = {
      mlir::hacc::HACCFuncTypeAttr::name, mlir::hacc::HostFuncTypeAttr::name,
      // Host Functions
      mlir::hacc::TilingFunctionAttr::name,
      mlir::hacc::InferOutputShapeFunctionAttr::name,
      mlir::hacc::InferWorkspaceShapeFunctionAttr::name,
      mlir::hacc::GetTilingStructSizeFunctionAttr::name,
      mlir::hacc::ExternalFunctionPathAttr::name,
      mlir::hacc::InferSyncBlockLockNumFunctionAttr::name,
      mlir::hacc::InferSyncBlockLockInitFunctionAttr::name,
      mlir::hacc::InferVFModeFunctionAttr::name,
      mlir::hacc::BlockDimAttr::name};
  bool allowedHACCAttr(StringRef attributeName) const {
    return kAllowedAttrNames.count(attributeName);
  }
};
} // namespace

void mlir::registerHACCDialectTranslation(mlir::DialectRegistry &registry) {
  registry.insert<hacc::HACCDialect>();
  registry.addExtension(+[](MLIRContext *ctx, hacc::HACCDialect *dialect) {
    dialect->addInterfaces<HACCDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerHACCDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerHACCDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
