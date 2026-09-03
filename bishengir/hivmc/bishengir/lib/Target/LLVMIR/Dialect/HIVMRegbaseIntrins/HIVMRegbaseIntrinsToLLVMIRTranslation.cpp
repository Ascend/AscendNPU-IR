//===- HIVMRegbaseToLLVMIRTranslation.cpp - Translate HIVM to LLVM IR -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the HIVMRegbaseIntrins dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Target/LLVMIR/Dialect/HIVMRegbaseIntrins/HIVMRegbaseIntrinsToLLVMIRTranslation.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsHIVM.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

/// Create metadata nodes as {mdName, value} and attach to func
void addAnnotationMD(llvm::Function *func, StringRef mdName, uint32_t value) {
  llvm::LLVMContext &ctx = func->getContext();

  llvm::Metadata *mdVals[] = {
      llvm::MDString::get(ctx, mdName),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), value))};

  llvm::MDNode *mdNode = llvm::MDNode::get(ctx, mdVals);
  assert(mdNode!=nullptr);
  func->addMetadata(llvm::LLVMContext::MD_annotation, *mdNode);
}

/// Implementation of the dialect interface that converts operations belonging
/// to the HIVMRegbaseIntrins dialect to LLVM IR.
class HIVMRegbaseIntrinsDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseConversions.inc"

    // `hivm_regbaseintrins.launch_func` will emit two instructions:
    // - hivm.store.vfsimt.info
    // - call simt_func @foo
    if (auto launchOp = dyn_cast<hivm_regbaseintrins::LaunchFuncOp>(op)) {
      // Firstly setup the configuration for vfsimt.info, verified with
      // test/CodeGen/HiIPU/V310/simt_entry_args_convert/cast_address_space.ll
      // | reserve |    z    |    y    |    x    |
      // 64       48        32        16         0
      llvm::Value *blockX =
          moduleTranslation.lookupValues(launchOp.getBlockSizeX()).front();
      llvm::Value *blockY =
          moduleTranslation.lookupValues(launchOp.getBlockSizeY()).front();
      constexpr uint64_t configValYShift = 16;
      llvm::Value *blockZ =
          moduleTranslation.lookupValues(launchOp.getBlockSizeZ()).front();
      constexpr uint64_t configValZShift = 32;
      llvm::Value *operand =
          builder.CreateOr(builder.CreateShl(blockY, configValYShift),
                           builder.CreateShl(blockZ, configValZShift));
      operand = builder.CreateOr(operand, blockX);
      // Create the intrinsic with the proper block dimensions
      builder.CreateIntrinsic(llvm::Intrinsic::hivm_store_vfsimt_info, {},
                              {operand});

      // Now generate the CallInst
      llvm::Function *func =
          moduleTranslation.lookupFunction(launchOp.getKernel());
      auto funcOperands = moduleTranslation.lookupValues(launchOp.getOpnds());
      auto *callInst = builder.CreateCall(func, funcOperands);
      // ... and set the calling convention
      callInst->setCallingConv(llvm::CallingConv::TPE_SIMTEntry);
      return success();
    }
    return failure();
  }

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {

    LLVM::LLVMFuncOp func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func)
      return op->emitError("Attribute not applicable to non-function ops.");

    llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
    Attribute baseAttr = attribute.getValue();
    // Add target-cpu and target-features to the function attribute.
    if (attribute.getName() ==
        mlir::hivm_regbaseintrins::kDavinciTargetAttrName) {
      if (auto target =
              mlir::dyn_cast<hivm_regbaseintrins::SIMT_TargetAttr>(baseAttr)) {
        llvmFunc->addFnAttr("target-cpu", target.getValue());
        llvmFunc->addFnAttr("target-features",
                            StringRef("+" + target.getValue().str()));
      } else
        op->emitError("Unexpected 'target'");
    } else if (attribute.getName() ==
               mlir::hivm_regbaseintrins::kDavinciCallingConvAttrName) {
      // Set calling convention for SIMT function as well as the max-threads
      // metadata.
      if (auto simtAttr =
              mlir::dyn_cast<hivm_regbaseintrins::SIMT_EntryAttr>(baseAttr)) {
        addAnnotationMD(llvmFunc, "simt_entry", 1);
        addAnnotationMD(llvmFunc, "simt-max-threads", simtAttr.getValue());
        llvmFunc->addFnAttr(llvm::Attribute::NoInline);
        llvmFunc->setCallingConv(llvm::CallingConv::TPE_SIMTEntry);
        llvmFunc->setLinkage(llvm::GlobalValue::InternalLinkage);
      } else if (auto simtAttr =
                     mlir::dyn_cast<hivm_regbaseintrins::SIMT_CallableAttr>(
                         baseAttr)) {
        addAnnotationMD(llvmFunc, "simt_callable", 1);
        llvmFunc->setCallingConv(llvm::CallingConv::TPE_SIMTCallable);
      }
    } else if (attribute.getName() ==
               mlir::hivm_regbaseintrins::kDavinciKernelAttrName) {
      // Set kernel metadata, to signal that this is an entry kernel so that
      // proper ABI can be emitted.
      llvm::Module *llvmModule = llvmFunc->getParent();
      llvm::LLVMContext &ctx = llvmModule->getContext();
      llvm::NamedMDNode *mdNode =
          llvmModule->getOrInsertNamedMetadata("hivm.annotations");
      llvm::Metadata *md[] = {
          llvm::ConstantAsMetadata::get(llvmFunc),
          llvm::MDString::get(ctx, "kernel"),
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1))};
      mdNode->addOperand(llvm::MDNode::get(ctx, md));
    }
    return success();
  }
};
} // namespace

void mlir::registerHIVMRegbaseIntrinsDialectTranslation(
    mlir::DialectRegistry &registry) {
  registry.insert<hivm_regbaseintrins::HIVMRegbaseIntrinsDialect>();
  registry.addExtension(+[](MLIRContext *ctx,
                            hivm_regbaseintrins::HIVMRegbaseIntrinsDialect
                                *dialect) {
    dialect
        ->addInterfaces<HIVMRegbaseIntrinsDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerHIVMRegbaseIntrinsDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerHIVMRegbaseIntrinsDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
