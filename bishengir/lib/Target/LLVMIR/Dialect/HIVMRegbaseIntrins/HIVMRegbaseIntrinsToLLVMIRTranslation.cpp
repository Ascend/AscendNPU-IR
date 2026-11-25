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
  auto function = opInst.getParentOfType<LLVM::LLVMFuncOp>();
  bool hasScatter =
      function &&
      function
          .walk([](Operation *op) {
            return isa<hivm_regbaseintrins::VScatterV64S32InstrOp>(op)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted();
  if (hasScatter && inst->onlyReadsMemory()) {
    // CANN 9.2 can reuse a pre-scatter vector load despite VST_VLD.
    // Forget pointer invariance for reads in this VF, including loop reads
    // and aliases. This is a compiler compatibility measure, not a fence;
    // the AVE program must still provide the required hardware ordering.
    llvm::IRBuilderBase::InsertPointGuard guard(builder);
    builder.SetInsertPoint(inst);
    for (unsigned i = 0; i < inst->arg_size(); ++i)
      if (auto *type =
              dyn_cast<llvm::PointerType>(inst->getArgOperand(i)->getType());
          type && type->getAddressSpace() == 6)
        inst->setArgOperand(
            i, builder.CreateLaunderInvariantGroup(inst->getArgOperand(i)));
  }
  if (isa<hivm_regbaseintrins::VScatterV64S32InstrOp>(opInst)) {
    // Older LLVM definitions mark the scatter destination readonly.
    auto *callee = inst->getCalledFunction();
    callee->removeParamAttr(1, llvm::Attribute::ReadOnly);
    callee->addParamAttr(1, llvm::Attribute::WriteOnly);
    inst->removeParamAttr(1, llvm::Attribute::ReadOnly);
    inst->addParamAttr(1, llvm::Attribute::WriteOnly);
  }
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

/// Rewrite a dual-output predicate interleave/de-interleave op into two v2
/// part-select calls under the dav-920r1 (V121) toolchain, mirroring how
/// VDintlvInstrOp is split. `isDeintlv` selects pdintlvv2 vs pintlvv2; the
/// element bit width is taken from the op mnemonic suffix (b8/b16/b32).
template <bool isDeintlv, typename OpTy>
LogicalResult
rewritePredicateInterleaveV2(OpTy op, llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) {
  auto moduleOp = op->template getParentOfType<mlir::ModuleOp>();
  if (!moduleOp || !moduleOp->hasAttr(
                       hivm_regbaseintrins::kDavinciV2IntrinsicsModuleAttrName))
    return failure();

  auto operands = moduleTranslation.lookupValues(op.getOperands());
  llvm::Value *src0 = operands[0];
  llvm::Value *src1 = operands[1];

  llvm::Intrinsic::ID id;
  if (StringRef(op.getOperationName()).contains(".b8"))
    id = isDeintlv ? llvm::Intrinsic::HIVMIntrinsics::hivm_pdintlvv2_b8
                   : llvm::Intrinsic::HIVMIntrinsics::hivm_pintlvv2_b8;
  else if (StringRef(op.getOperationName()).contains(".b16"))
    id = isDeintlv ? llvm::Intrinsic::HIVMIntrinsics::hivm_pdintlvv2_b16
                   : llvm::Intrinsic::HIVMIntrinsics::hivm_pintlvv2_b16;
  else
    id = isDeintlv ? llvm::Intrinsic::HIVMIntrinsics::hivm_pdintlvv2_b32
                   : llvm::Intrinsic::HIVMIntrinsics::hivm_pintlvv2_b32;

  // The predicate interleave intrinsics use the fixed llvm.v256i1 predicate
  // type (no overloads), so no type list may be passed to getDeclaration.
  llvm::Function *fn = llvm::Intrinsic::getDeclaration(
      builder.GetInsertBlock()->getModule(), id, {});
  llvm::Value *resLo =
      builder.CreateCall(fn, {src0, src1, builder.getInt32(0)});
  llvm::Value *resHi =
      builder.CreateCall(fn, {src0, src1, builder.getInt32(1)});
  llvm::Type *structType = moduleTranslation.convertType(op.getRes().getType());
  llvm::Value *structVal = llvm::PoisonValue::get(structType);
  structVal = builder.CreateInsertValue(structVal, resLo, 0);
  structVal = builder.CreateInsertValue(structVal, resHi, 1);
  moduleTranslation.mapValue(op.getRes()) = structVal;
  return success();
}

/// Rewrite a legacy shift intrinsic into the fully typed dav-920r1 (Ascend960)
/// form.
///
/// The legacy forms (`llvm.hivm.vshrs.s.x` / `.u.x` and the vector-amount
/// variants) take a signed shift amount. The Ascend960 toolchain only registers
/// the typed, zeroing-mode forms whose shift amount is unsigned:
///
///   llvm.hivm.<vshl|vshls|vshr|vshrs>.v<vl><elem><amount>.z
///
/// e.g. `llvm.hivm.vshrs.v128s16u16.z`. Those names are not part of the LLVM
/// dialect's intrinsic table yet, so emit a call to the declared function by
/// name; the external backend recognizes it as an intrinsic.
static LogicalResult
rewriteShiftV2(Operation *op, llvm::IRBuilderBase &builder,
               LLVM::ModuleTranslation &moduleTranslation) {
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp || !moduleOp->hasAttr(
                       hivm_regbaseintrins::kDavinciV2IntrinsicsModuleAttrName))
    return failure();

  bool isScalar;
  bool isLeft;
  bool isSigned;
  if (isa<hivm_regbaseintrins::VshlsSXInstrOp>(op)) {
    isScalar = true;
    isLeft = true;
    isSigned = true;
  } else if (isa<hivm_regbaseintrins::VshlsUXInstrOp>(op)) {
    isScalar = true;
    isLeft = true;
    isSigned = false;
  } else if (isa<hivm_regbaseintrins::VshrsSXInstrOp>(op)) {
    isScalar = true;
    isLeft = false;
    isSigned = true;
  } else if (isa<hivm_regbaseintrins::VshrsUXInstrOp>(op)) {
    isScalar = true;
    isLeft = false;
    isSigned = false;
  } else if (isa<hivm_regbaseintrins::VshlSXInstrOp>(op)) {
    isScalar = false;
    isLeft = true;
    isSigned = true;
  } else if (isa<hivm_regbaseintrins::VshlUXInstrOp>(op)) {
    isScalar = false;
    isLeft = true;
    isSigned = false;
  } else if (isa<hivm_regbaseintrins::VshrSXInstrOp>(op)) {
    isScalar = false;
    isLeft = false;
    isSigned = true;
  } else if (isa<hivm_regbaseintrins::VshrUXInstrOp>(op)) {
    isScalar = false;
    isLeft = false;
    isSigned = false;
  } else {
    return failure();
  }

  auto vecType = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!vecType || !vecType.getElementType().isInteger())
    return failure();

  unsigned vl;
  const char *elemSuffix;
  const char *amountSuffix;
  unsigned amountBits;
  switch (vecType.getElementTypeBitWidth()) {
  case 8:
    vl = 256;
    elemSuffix = isSigned ? "s8" : "u8";
    amountSuffix = "u8";
    amountBits = 16;
    break;
  case 16:
    vl = 128;
    elemSuffix = isSigned ? "s16" : "u16";
    amountSuffix = "u16";
    amountBits = 16;
    break;
  case 32:
    vl = 64;
    elemSuffix = isSigned ? "s32" : "u32";
    amountSuffix = "u32";
    amountBits = 32;
    break;
  case 64:
    vl = 32;
    elemSuffix = isSigned ? "s64" : "u64";
    amountSuffix = "u32";
    amountBits = 32;
    break;
  default:
    return failure();
  }

  std::string name = "llvm.hivm.";
  name += isLeft ? "vshl" : "vshr";
  if (isScalar)
    name += "s";
  name += ".v" + std::to_string(vl) + elemSuffix + amountSuffix + ".z";

  SmallVector<llvm::Value *> args =
      moduleTranslation.lookupValues(op->getOperands());
  llvm::Type *resTy = moduleTranslation.convertType(op->getResult(0).getType());
  llvm::Type *maskTy =
      moduleTranslation.convertType(op->getOperand(2).getType());

  SmallVector<llvm::Type *> params{resTy};
  if (isScalar) {
    llvm::Type *amountTy = builder.getIntNTy(amountBits);
    if (args[1]->getType() != amountTy)
      args[1] = builder.CreateZExtOrTrunc(args[1], amountTy);
    params.push_back(amountTy);
  } else {
    params.push_back(resTy);
  }
  params.push_back(maskTy);

  auto *fnTy = llvm::FunctionType::get(resTy, params, /*isVarArg=*/false);
  llvm::FunctionCallee callee =
      builder.GetInsertBlock()->getModule()->getOrInsertFunction(name, fnTy);
  llvm::CallInst *call = builder.CreateCall(callee, args);
  moduleTranslation.mapValue(op->getResult(0), call);
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

    // hivm.pintlv.b<w>/hivm.pdintlv.b<w>: like vdintlv, the v2 toolchain only
    // provides the part-select single-output forms pintlvv2/pdintlvv2
    // (src0, src1, part) -> pred with part = 0(LOWER)/1(HIGHER). When the
    // module is flagged with the v2 intrinsics attribute, split the dual-output
    // predicate interleave/de-interleave into two v2 calls.
    if (auto pintlvOp = dyn_cast<hivm_regbaseintrins::PintlvB8InstrOp>(op))
      if (succeeded(rewritePredicateInterleaveV2<false>(pintlvOp, builder,
                                                        moduleTranslation)))
        return success();
    if (auto pintlvOp = dyn_cast<hivm_regbaseintrins::PintlvB16InstrOp>(op))
      if (succeeded(rewritePredicateInterleaveV2<false>(pintlvOp, builder,
                                                        moduleTranslation)))
        return success();
    if (auto pintlvOp = dyn_cast<hivm_regbaseintrins::PintlvB32InstrOp>(op))
      if (succeeded(rewritePredicateInterleaveV2<false>(pintlvOp, builder,
                                                        moduleTranslation)))
        return success();
    if (auto pdintlvOp = dyn_cast<hivm_regbaseintrins::PDintlvB8InstrOp>(op))
      if (succeeded(rewritePredicateInterleaveV2<true>(pdintlvOp, builder,
                                                       moduleTranslation)))
        return success();
    if (auto pdintlvOp = dyn_cast<hivm_regbaseintrins::PDintlvB16InstrOp>(op))
      if (succeeded(rewritePredicateInterleaveV2<true>(pdintlvOp, builder,
                                                       moduleTranslation)))
        return success();
    if (auto pdintlvOp = dyn_cast<hivm_regbaseintrins::PDintlvB32InstrOp>(op))
      if (succeeded(rewritePredicateInterleaveV2<true>(pdintlvOp, builder,
                                                       moduleTranslation)))
        return success();

    // dav-920r1 (Ascend960) only registers the typed, unsigned-amount shift
    // intrinsics (`...v<vl><elem><amount>.z`). Rewrite the legacy shift forms
    // before falling through to the generated conversions.
    if (succeeded(rewriteShiftV2(op, builder, moduleTranslation)))
      return success();

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

    if (isa<LLVM::CallOp>(op) &&
        attribute.getName() ==
            hivm_regbaseintrins::kDavinciCallingConvAttrName &&
        isa<hivm_regbaseintrins::SIMT_CallableAttr>(attribute.getValue())) {
      if (instructions.size() != 1 ||
          !llvm::isa<llvm::CallInst>(instructions.front()))
        return op->emitError("expected a single LLVM call instruction");
      llvm::cast<llvm::CallInst>(instructions.front())
          ->setCallingConv(llvm::CallingConv::TPE_SIMTCallable);
      return success();
    }

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
