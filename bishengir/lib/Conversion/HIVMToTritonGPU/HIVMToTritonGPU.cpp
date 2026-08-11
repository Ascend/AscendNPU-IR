//===- HIVMToTritonGPU.cpp - conversion from HIVM to Triton dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToArith.h"
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToMath.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTHIVMTOTRITONGPU
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct HIVMToTritonGPUConversionPass
    : public impl::ConvertHIVMToTritonGPUBase<HIVMToTritonGPUConversionPass> {
  void runOnOperation() override;
};
} // namespace

/// Converts a Triton function signature, expanding MemRef arguments into
/// bare pointers or descriptor fields based on the calling convention.
/// Also collects shared-memory argument indices and fractal layout attribute
/// mappings.
FunctionType TritonTypeConverter::convertTTFunctionSignature(
    func::FuncOp funcOp, bool useBarePtrCallConv, SignatureConversion &result,
    SmallVector<std::optional<int>> &sharedIds,
    SmallVector<std::pair<int, Attribute>> &fractalAttrArgIds) const {
  auto ttFuncArgConverter = useBarePtrCallConv ? bareTTPtrFuncArgTypeConverter
                                               : structTTFuncArgTypeConverter;
  FunctionType funcTy = funcOp.getFunctionType();
  // Convert argument types one by one and check for errors.
  for (auto [idx, type] : llvm::enumerate(funcTy.getInputs())) {
    SmallVector<Type, 8> converted;
    if (failed(ttFuncArgConverter(*this, type, converted)))
      return {};

    if (funcOp.getArgAttr(idx, SharedMemoryAttr::name)) {
      sharedIds.push_back(idx);
    }

    if (auto fractalAttr = funcOp.getArgAttr(idx, "hivm.fractal_layout")) {
      fractalAttrArgIds.push_back({idx, fractalAttr});
    }

    result.addInputs(idx, converted);
  }

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types
  // into a struct.
  FunctionType resultType = FunctionType::get(
      funcTy.getContext(), result.getConvertedTypes(), funcTy.getResults());

  return resultType;
}

/// Returns the Triton address space corresponding to a MemRef's memory
/// space attribute. Maps HIVM memory spaces (e.g. UB, shared) to Triton
/// address space values.
uint32_t TritonTypeConverter::getMemRefAddressSpace(BaseMemRefType type) const {
  if (auto attr = type.getMemorySpace()) {
    if (auto ASAttr = dyn_cast<AddressSpaceAttr>(attr)) {
      auto AS = static_cast<uint32_t>(ASAttr.getAddressSpace());
      return AS;
    }
  }
  // Note: If the address space of memref type is not specified, treat it as
  //       in global memory.
  return 1;
}

/// Converts a MemRef type to a bare pointer (tt.ptr) in the appropriate
/// Triton address space based on the MemRef's memory space.
Type TritonTypeConverter::convertMemRefToTTBarePtr(BaseMemRefType type) const {
  if (!canConvertToBarePtr(type))
    return {};
  Type elementType = type.getElementType();
  uint32_t addressSpace = getMemRefAddressSpace(type);
  return triton::getPointerType(elementType, addressSpace);
}

/// Converts a type to the appropriate calling convention type.
/// If useBarePtrCallConv is true, MemRef types become bare pointers;
/// otherwise they become descriptor-based types.
Type TritonTypeConverter::convertTTCallingConventionType(
    Type type, bool useBarePtrCallConv) const {
  if (useBarePtrCallConv)
    if (auto memrefTy = dyn_cast<BaseMemRefType>(type))
      return convertMemRefToTTBarePtr(memrefTy);

  return convertType(type);
}

/// Returns the descriptor field types for a ranked MemRef. A descriptor
/// typically contains: { basePtr, alignedPtr, offset, sizes[rank],
/// strides[rank] }.
SmallVector<Type, 5> TritonTypeConverter::getMemRefDescriptorFields(
    MemRefType type) const {
  if (!isStrided(type)) {
    emitError(UnknownLoc::get(type.getContext()),
        "conversion to strided form failed either due to non-strided layout "
        "maps (which should have been normalized away) or other reasons");
    return {};
  }

  Type elementType = convertType(type.getElementType());
  if (!elementType)
    return {};

  uint32_t addressSpace = getMemRefAddressSpace(type);

  auto ptrTy = triton::getPointerType(elementType, addressSpace);

  auto indexTy = IntegerType::get(type.getContext(), 64);

  SmallVector<Type, 5> results = {ptrTy, ptrTy, indexTy};
  auto rank = type.getRank();
  if (rank == 0)
    return results;

  results.insert(results.end(), 2 * rank, indexTy);
  return results;
}

/// Returns the descriptor field types for an unranked MemRef, which
/// contains the rank and a pointer to a descriptor struct.
SmallVector<Type, 2> TritonTypeConverter::getUnrankedMemRefDescriptorFields(
    Type elemTy) const {
  return {IntegerType::get(elemTy.getContext(), 64),
      triton::getPointerType(elemTy)};
}

/// Callback to convert function argument types. It converts a MemRef function
/// argument to a list of non-aggregate types containing descriptor
/// information, and an UnrankedmemRef function argument to a list containing
/// the rank and a pointer to a descriptor struct.
LogicalResult hivm::structTTFuncArgTypeConverter(
    const TritonTypeConverter &converter, Type type,
    SmallVectorImpl<Type> &result) {
  if (auto memref = dyn_cast<MemRefType>(type)) {
    // In signatures, Memref descriptors are expanded into lists of
    // non-aggregate values.
    auto converted = converter.getMemRefDescriptorFields(memref);
    if (converted.empty())
      return failure();
    result.append(converted.begin(), converted.end());
    return success();
  }
  if (isa<UnrankedMemRefType>(type)) {
    auto converted = converter.getUnrankedMemRefDescriptorFields(
        dyn_cast<UnrankedMemRefType>(type).getElementType());
    if (converted.empty())
      return failure();
    result.append(converted.begin(), converted.end());
    return success();
  }
  auto converted = converter.convertType(type);
  if (!converted)
    return failure();
  result.push_back(converted);
  return success();
}

/// Callback to convert function argument types. It converts MemRef function
/// arguments to bare pointers to the MemRef element type.
LogicalResult hivm::bareTTPtrFuncArgTypeConverter(
    const TritonTypeConverter &converter, Type type,
    SmallVectorImpl<Type> &result) {
  auto ttTy = converter.convertTTCallingConventionType(
      type, /*useBarePointerCallConv=*/true);
  if (!ttTy)
    return failure();

  result.push_back(ttTy);
  return success();
}

Type mlir::hivm::HIVMToTritonTypeConvert(Type ty) {
  if (auto memrefTy = dyn_cast<MemRefType>(ty)) {
    auto elemTy = memrefTy.getElementType();
    // Note: If the address space of memref type is not specified, treat it as
    //       in global memory.
    if (auto attr = memrefTy.getMemorySpace()) {
      if (auto ASAttr = dyn_cast<AddressSpaceAttr>(attr)) {
        auto AS = static_cast<uint32_t>(ASAttr.getAddressSpace());
        return triton::getPointerType(elemTy, AS);
      }
      llvm::report_fatal_error("Invalid memory space");
    } else {
      return triton::getPointerType(elemTy);
    }
  }
  return ty;
}

void HIVMToTritonGPUConversionPass::runOnOperation() {
  auto module = getOperation();
  auto &ctx = getContext();

  // FIXME: Compose the parent reinterpret_cast offset into the subview offset
  // instead of rejecting this nested view.
  bool hasUnsupportedNestedView = false;
  module->walk([&](memref::SubViewOp subviewOp) {
    auto parentCast =
        subviewOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!parentCast || parentCast.getStaticOffsets().front() == 0)
      return;
    subviewOp.emitOpError(
        "parent memref.reinterpret_cast before memref.subview is not "
        "supported: parent view offset is not yet composed");
    hasUnsupportedNestedView = true;
  });
  if (hasUnsupportedNestedView) {
    return signalPassFailure();
  }

  // Stage1: Convert operators in function body
  ConversionTarget stage1Target(ctx);
  stage1Target
      .addLegalDialect<arith::ArithDialect, math::MathDialect,
                       triton::TritonDialect, triton::gpu::TritonGPUDialect,
                       func::FuncDialect>();
  stage1Target.addIllegalDialect<hivm::HIVMDialect>();
  stage1Target.addIllegalOp<memref::ReinterpretCastOp>();
  stage1Target.addIllegalOp<memref::SubViewOp>();
  stage1Target.addIllegalOp<memref::LoadOp>();
  stage1Target.addIllegalOp<memref::ExtractAlignedPointerAsIndexOp>();
  // The TTIR path only accepts Triton ops plus tensor types
  stage1Target.addLegalOp<tensor::EmptyOp>();
  stage1Target.addIllegalOp<tensor::ExpandShapeOp, tensor::CollapseShapeOp>();
  stage1Target.addIllegalOp<bufferization::ToTensorOp>();
  stage1Target.addIllegalOp<affine::AffineApplyOp>();
  stage1Target.addLegalOp<mlir::UnrealizedConversionCastOp>();

  RewritePatternSet stage1Patterns(&ctx);

  populateHIVMToArithConversionPatterns(stage1Patterns);
  populateHIVMToMathConversionPatterns(stage1Patterns);
  populateTensorToTritonPatterns(stage1Patterns);
  populateReinterpretCastToUnrealizedCastPatterns(stage1Patterns);
  populateMemRefLoadToTritonPatterns(stage1Patterns);
  populateExtractAlignedPointerToTritonPatterns(stage1Patterns);
  populateHIVMToTritonPatterns(stage1Patterns);
  populateBufferizationToTritonPatterns(stage1Patterns);
  populateAffineToTritonPatterns(stage1Patterns);

  if (failed(applyPartialConversion(module, stage1Target,
                                    std::move(stage1Patterns)))) {
    module->emitError("Stage1 failed: HIVM/Bufferization Op conversion failed");
    return signalPassFailure();
  }

  // Remove redundant memref.alloc
  module->walk([&](memref::AllocOp alloc) {
    if (alloc.use_empty()) {
      alloc.erase();
    } else {
      module->emitError("memref.alloc should have no use!");
      signalPassFailure();
    }
  });

  // Stage2: Convert FuncOp alone
  // Note: Return values are allowed only in test scenarios.
  if (!allowReturnValue) {
    RewritePatternSet stage2Patterns(&ctx);
    ConversionTarget stage2Target(ctx);
    stage2Target.addLegalDialect<
        triton::TritonDialect, triton::gpu::TritonGPUDialect,
        arith::ArithDialect, math::MathDialect, mlir::BuiltinDialect>();
    stage2Target.addLegalOp<tensor::EmptyOp>();
    stage2Target.addLegalOp<tensor::ExtractOp>();
    stage2Target.addLegalOp<tensor::ExtractSliceOp>();
    stage2Target.addIllegalOp<func::FuncOp>();
    stage2Target.addLegalOp<UnrealizedConversionCastOp>();
    stage2Target.addLegalDialect<scf::SCFDialect>();

    // Convert function signatures with dynamic and static calling conventions
    // separately, as hivm to llvm precedure
    LowerToTritonGPUOptions dynOptions;
    dynOptions.useBarePtrCallConv = false;
    TritonTypeConverter dynConverter(&ctx, dynOptions);
    populateFuncToTritonPatterns(dynConverter, stage2Patterns);

    LowerToTritonGPUOptions staticOptions;
    staticOptions.useBarePtrCallConv = true;
    TritonTypeConverter staticConverter(&ctx, staticOptions);
    populateFuncToTritonPatterns(staticConverter, stage2Patterns);

    if (failed(applyPartialConversion(module, stage2Target,
                                      std::move(stage2Patterns)))) {
      module->emitError("Stage2 failed: FuncOp to Triton conversion failed");
      signalPassFailure();
    }
  }

  // Stage3: Clean up redundant UnrealizedCastsOp
  OpPassManager dynPM;
  dynPM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(runPipeline(dynPM, module))) {
    module->emitError("Stage3 failed: Reconcile unrealized casts failed");
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createHIVMToTritonGPUConversionPass() {
  return std::make_unique<HIVMToTritonGPUConversionPass>();
}
