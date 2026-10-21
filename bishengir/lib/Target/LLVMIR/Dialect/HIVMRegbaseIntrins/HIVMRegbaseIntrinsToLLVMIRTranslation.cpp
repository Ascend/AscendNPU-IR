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
#include "llvm/Support/Compiler.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

LLVM_ATTRIBUTE_NOINLINE void
convertHIVMRegbaseIntrinsic(Operation &opInst, llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation,
                            llvm::Intrinsic::ID intrinsic, unsigned numResults,
                            ArrayRef<unsigned> overloadedResults,
                            ArrayRef<unsigned> overloadedOperands) {
  auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, intrinsic, numResults,
      overloadedResults, overloadedOperands, /*immArgPositions=*/{},
      /*immArgAttrNames=*/{});
  if (numResults != 0)
    moduleTranslation.mapValue(opInst.getResult(0), inst);
}

/// Create metadata nodes as {mdName, value} and attach to func
void addAnnotationMD(llvm::Function *func, StringRef mdName, uint32_t value) {
  llvm::LLVMContext &ctx = func->getContext();

  llvm::Metadata *mdVals[] = {
      llvm::MDString::get(ctx, mdName),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), value))};

  llvm::MDNode *mdNode = llvm::MDNode::get(ctx, mdVals);
  assert(mdNode != nullptr);
  func->addMetadata(llvm::LLVMContext::MD_annotation, *mdNode);
}

LogicalResult convertLaunchFunc(hivm_regbaseintrins::LaunchFuncOp launchOp,
                                llvm::IRBuilderBase &builder,
                                LLVM::ModuleTranslation &moduleTranslation) {
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
  builder.CreateIntrinsic(llvm::Intrinsic::hivm_store_vfsimt_info, {},
                          {operand});

  llvm::Function *func = moduleTranslation.lookupFunction(launchOp.getKernel());
  auto funcOperands = moduleTranslation.lookupValues(launchOp.getOpnds());
  auto *callInst = builder.CreateCall(func, funcOperands);
  callInst->setCallingConv(llvm::CallingConv::TPE_SIMTEntry);
  return success();
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

    // hivm.vdintlv: the default toolchain lowers `llvm.hivm.vdintlv.<vec>`
    // (2 results) while a toolchain that only provides the v2 form registers
    // `llvm.hivm.vdintlvv2.<vec>(src0, src1, part) -> <vec>` where
    // part = 0(LOWER)/1(HIGHER). When the module is flagged with the v2
    // intrinsics attribute, split the single op into two v2 calls and
    // reassemble the struct result; otherwise fall through to the generated
    // conversion below to keep the original behavior.
    if (auto vdintlvOp = dyn_cast<hivm_regbaseintrins::VDintlvInstrOp>(op)) {
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      if (moduleOp &&
          moduleOp->hasAttr(
              hivm_regbaseintrins::kDavinciV2IntrinsicsModuleAttrName)) {
        auto operands = moduleTranslation.lookupValues(vdintlvOp.getOperands());
        llvm::Value *src0 = operands[0];
        llvm::Value *src1 = operands[1];
        llvm::Type *vecType = src0->getType();
        llvm::Function *fn = llvm::Intrinsic::getDeclaration(
            builder.GetInsertBlock()->getModule(),
            llvm::Intrinsic::HIVMIntrinsics::hivm_vdintlvv2, {vecType});
        llvm::Value *resLo =
            builder.CreateCall(fn, {src0, src1, builder.getInt32(0)});
        llvm::Value *resHi =
            builder.CreateCall(fn, {src0, src1, builder.getInt32(1)});
        llvm::Type *structType =
            moduleTranslation.convertType(vdintlvOp.getRes().getType());
        llvm::Value *structVal = llvm::PoisonValue::get(structType);
        structVal = builder.CreateInsertValue(structVal, resLo, 0);
        structVal = builder.CreateInsertValue(structVal, resHi, 1);
        moduleTranslation.mapValue(vdintlvOp.getRes()) = structVal;
        return success();
      }
    }

#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseConversions.inc"

    // `hivm_regbaseintrins.launch_func` will emit two instructions:
    // - hivm.store.vfsimt.info
    // - call simt_func @foo
    if (auto launchOp = dyn_cast<hivm_regbaseintrins::LaunchFuncOp>(op))
      return convertLaunchFunc(launchOp, builder, moduleTranslation);
    return failure();
  }

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {

    LLVM::LLVMFuncOp func = dyn_cast<LLVM::LLVMFuncOp>(op);
    if (!func) {
      // Module-level attributes (e.g. hivm_regbaseintrins.v2intrins) do not
      // need per-function handling.
      if (isa<mlir::ModuleOp>(op))
        return success();
      return op->emitError("Attribute not applicable to non-function ops.");
    }

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
