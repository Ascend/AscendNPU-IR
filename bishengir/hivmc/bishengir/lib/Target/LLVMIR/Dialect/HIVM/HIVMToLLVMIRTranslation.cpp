//===- HIVMToLLVMIRTranslation.cpp - Translate HIVM to LLVM IR ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the HIVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Target/LLVMIR/Dialect/HIVM/HIVMToLLVMIRTranslation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsHIVM.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {
static constexpr llvm::StringRef kNoAliasScope = "llvm.noalias_scopes";
/// Create metadata node {mdName, value} and attach it to func's !annotation.
void addAnnotationMD(llvm::Function *func, StringRef mdName, uint32_t value) {
llvm::LLVMContext &ctx = func->getContext();
llvm::Metadata *mdVals[] = {
    llvm::MDString::get(ctx, mdName),
    llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), value))};
func->addMetadata(llvm::LLVMContext::MD_annotation,
                  *llvm::MDNode::get(ctx, mdVals));
}

/// Append {ptr @func, mdName, value} to the module-level !hivm.annotations.
void addHivmAnnotation(llvm::Function *func, StringRef mdName, uint32_t value) {
llvm::Module *mod = func->getParent();
llvm::LLVMContext &ctx = mod->getContext();
llvm::Metadata *md[] = {
    llvm::ConstantAsMetadata::get(func),
    llvm::MDString::get(ctx, mdName),
    llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), value))};
mod->getOrInsertNamedMetadata("hivm.annotations")
    ->addOperand(llvm::MDNode::get(ctx, md));
}

/// Implementation of the dialect interface that converts operations belonging
/// to the HIVM dialect to LLVM IR.
class HIVMDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "bishengir/Dialect/HIVM/IR/HIVMConversions.inc"

    // Non-one-to-one translation:
    if (auto ldDev = dyn_cast<hivm::LoadScalarOp>(op)) {
      assert(isa<LLVM::LLVMPointerType>(ldDev.getAddr().getType()));
      llvm::Value *loadVal = nullptr;
      llvm::Value *arg = moduleTranslation.lookupValue(ldDev.getAddr());
      llvm::ConstantInt *offset = builder.getInt64(0);
      Type mlirTy = ldDev.getResult().getType();
      switch (mlirTy.getIntOrFloatBitWidth()) {
      case 8:
        loadVal = LLVM::detail::createIntrinsicCall(
            builder, llvm::Intrinsic::HIVMIntrinsics::hivm_LD_DEV_u8_GM,
            {arg, offset});
        loadVal = builder.CreateTrunc(loadVal, builder.getInt8Ty());
        break;
      case 16:
        loadVal = LLVM::detail::createIntrinsicCall(
            builder, llvm::Intrinsic::HIVMIntrinsics::hivm_LD_DEV_u16_GM,
            {arg, offset});
        loadVal = builder.CreateTrunc(loadVal, builder.getInt16Ty());
        break;
      case 32:
        loadVal = LLVM::detail::createIntrinsicCall(
            builder, llvm::Intrinsic::HIVMIntrinsics::hivm_LD_DEV_u32_GM,
            {arg, offset});
        loadVal = builder.CreateTrunc(loadVal, builder.getInt32Ty());
        break;
      case 64:
        loadVal = LLVM::detail::createIntrinsicCall(
            builder, llvm::Intrinsic::HIVMIntrinsics::hivm_LD_DEV_u64_GM,
            {arg, offset});
        loadVal = builder.CreateTrunc(loadVal, builder.getInt64Ty());
        break;
      default:
        return emitError(op->getLoc(), "Loading of unexpected scalar type");
      }
      llvm::Type *targetTy =
          moduleTranslation.convertType(ldDev.getResult().getType());
      if (loadVal->getType() != targetTy) {
        loadVal = builder.CreateBitCast(loadVal, targetTy);
      }
      moduleTranslation.mapValue(ldDev, loadVal);
      return success();
    }

    return failure();
  }
  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const final {

    if (attribute.getName() == mlir::hivm::TCoreRatioAttr::name) {
        auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
        if (!func) return success();
        auto ratio = dyn_cast<hivm::TCoreRatioAttr>(attribute.getValue());
        if (!ratio) return success();
        llvm::Function *llvmFunc = moduleTranslation.lookupFunction(func.getName());
        assert(llvmFunc != nullptr && "Expecting function to be found in the module");

        llvmFunc->addFnAttr("mix-kernel-core-ratio");
        addAnnotationMD(llvmFunc, "mix-kernel-core-ratio-M", ratio.getCube());
        addAnnotationMD(llvmFunc, "mix-kernel-core-ratio-N", ratio.getVector());
        addHivmAnnotation(llvmFunc, "mix-kernel-core-ratio-M", ratio.getCube());
        addHivmAnnotation(llvmFunc, "mix-kernel-core-ratio-N", ratio.getVector());
        return success();
    }

    // look for hivm.vector_function attr, add noalias to its ptr arguments
    if (attribute.getName() == mlir::hivm::VectorFunctionAttr::name) {

      if (LLVM::CallOp call = dyn_cast<LLVM::CallOp>(op)) {
        auto llvmCallInst = dyn_cast<llvm::CallInst>(instructions[0]);
        assert(llvmCallInst != nullptr);
        auto calleeName = call.getCallee();
        if (!calleeName)
          return failure();
        auto mod = call->getParentOfType<ModuleOp>();
        if (!mod)
          return failure();
        auto callee = mod.lookupSymbol<LLVM::LLVMFuncOp>(*calleeName);
        if (callee->hasAttr(hivm_regbaseintrins::kDavinciKernelAttrName)) {
          llvmCallInst->setCallingConv(llvm::CallingConv::TPE_SIMTEntry);
        } else {
          llvmCallInst->setCallingConv(llvm::CallingConv::PTC_SimdVf);
        }
        return success();
      }

      LLVM::LLVMFuncOp func = dyn_cast<LLVM::LLVMFuncOp>(op);
      llvm::Function *llvmFunc =
          moduleTranslation.lookupFunction(func.getName());
      // mark each parameter type being !llvm.ptr of llvm.func as NoAlias
      auto llvmFuncArgs = llvmFunc->args();
      for (auto &arg : llvmFuncArgs) {
        if (isa<llvm::PointerType>(arg.getType())) {
          arg.addAttr(llvm::Attribute::AttrKind::NoAlias);
        }
      }
      llvmFunc->setCallingConv(llvm::CallingConv::PTC_SimdVf);

      // When compile as host code, all VF are declarations and cannot be
      // internal.
      if (!llvmFunc->isDeclaration())
        llvmFunc->setLinkage(llvm::GlobalValue::InternalLinkage);
      // disable VF to be inlined by optimization passes
      llvmFunc->addFnAttr(llvm::Attribute::NoInline);
      // disable VF to be duplicated by optimization passes, such as unrolling
      llvmFunc->addFnAttr(llvm::Attribute::NoDuplicate);
      // help VF optimization with CFG related passes in backend compiler with willreturn
      llvmFunc->addFnAttr(llvm::Attribute::WillReturn);
      return success();
    }
    if (attribute.getName() == mlir::hivm::HasAliaScopesAttr::name ||
        attribute.getName() == kNoAliasScope) {
      // should translate to 1 instruction
      assert(instructions.size() == 1);
      llvm::Instruction *inst = instructions[0];
      auto populateScopeMetadata = [&](ArrayAttr aliasScopeAttrs,
                                       unsigned kind) {
        if (!aliasScopeAttrs || aliasScopeAttrs.empty())
          return;
        llvm::MDNode *node = moduleTranslation.getOrCreateAliasScopes(
            llvm::to_vector(aliasScopeAttrs.getAsRange<AliasScopeAttr>()));
        inst->setMetadata(kind, node);
      };

      if (op->hasAttr(LLVM::AliasScopeAttr::name)) {
        auto scopeArrayAttr =
            dyn_cast<ArrayAttr>(op->getAttr(LLVM::AliasScopeAttr::name));
        assert(scopeArrayAttr);
        populateScopeMetadata(scopeArrayAttr,
                              llvm::LLVMContext::MD_alias_scope);
      }
      if (op->hasAttr(kNoAliasScope)) {
        auto scopeArrayAttr = dyn_cast<ArrayAttr>(op->getAttr(kNoAliasScope));
        assert(scopeArrayAttr);
        populateScopeMetadata(scopeArrayAttr, llvm::LLVMContext::MD_noalias);
      }
      return success();
    }
    return success();
  }
};
} // namespace

void mlir::registerHIVMDialectTranslation(mlir::DialectRegistry &registry) {
  registry.insert<hivm::HIVMDialect>();
  registry.addExtension(+[](MLIRContext *ctx, hivm::HIVMDialect *dialect) {
    dialect->addInterfaces<HIVMDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerHIVMDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerHIVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
