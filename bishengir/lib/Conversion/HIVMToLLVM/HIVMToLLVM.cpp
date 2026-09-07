//===- HIVMToLLVM.cpp - HIVM to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert HIVM dialect into the
// LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HIVMToLLVM/HIVMToLLVM.h"
#include "bishengir/Conversion/ArithToHIVMLLVM/ArithToHIVMLLVM.h"
#include "bishengir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "bishengir/Conversion/LLVMCommon/TypeConverter.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Transforms.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <regex>
#include <type_traits>

namespace mlir {
#define GEN_PASS_DEF_CONVERTHIVMTOLLVM
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define PASS_NAME "convert-hivm-to-llvm"

// Discardable marker placed on the problematic memref.cast by One-Shot Bufferize.
static constexpr llvm::StringLiteral kFoldOffsetMarker = "fold_offset_into_ptr";

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

template <
    typename OpTy,
    std::enable_if_t<std::disjunction_v<std::is_same<OpTy, LLVM::LoadOp>,
                                        std::is_same<OpTy, LLVM::StoreOp>>,
                     int> = 0>
bool isLLVMLoadStoreOnSsbuf(OpTy loadStoreOp) {
  LLVM::LLVMPointerType ptrType = loadStoreOp.getAddr().getType();
  return ptrType && ptrType.getAddressSpace() ==
                        static_cast<std::underlying_type_t<AddressSpace>>(
                            AddressSpace::SSBUF);
}

/// A pass converting MLIR operations into the LLVM IR dialect.
struct ConvertHIVMToLLVM
    : public impl::ConvertHIVMToLLVMBase<ConvertHIVMToLLVM> {
  using Base::Base;

  /// Run the dialect converter on the module.
  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    auto moduleOp = isa<ModuleOp>(getOperation())
                        ? cast<ModuleOp>(getOperation())
                        : getOperation()->getParentOfType<ModuleOp>();
    bool isRegBased = hacc::utils::isRegBasedArch(moduleOp);

    // clang-format off
    target.addIllegalDialect<
        arith::ArithDialect,
        cf::ControlFlowDialect,
        hivm::HIVMDialect,
        func::FuncDialect,
        math::MathDialect,
        memref::MemRefDialect,
        scf::SCFDialect,
        vector::VectorDialect
        >();
    // clang-format on
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // Patterns for intrinsic lowering.
    configureHIVMLegalizeForExportTarget(target);

    LowerToLLVMOptions dynOptions(&getContext());
    dynOptions.useBarePtrCallConv = false;
    dynOptions.onDemandBarePtrCallConv = onDemandBarePtrCallConv;
    bishengir::LLVMTypeConverter dynConverter(&getContext(), dynOptions);
    if (useBarePtrCallConv) {
      // convert func with dynamic shape memref (which cannot converted to bare
      // ptr) to llvm struct
      mlir::hivm::populateHIVMToLLVMConversionPatterns(dynConverter, patterns,
                                                       isRegBased);
    }

    // convert func with static shape memref to bare ptr first, note the pattern
    // that is pushed back later will be applied first
    LowerToLLVMOptions staticOptions(&getContext());
    staticOptions.useBarePtrCallConv = useBarePtrCallConv;
    // If `useBarePtrCallConv` is true, then `onDemandBarePtrCallConv` is
    // useless and should be set to false.
    staticOptions.onDemandBarePtrCallConv =
        useBarePtrCallConv ? false : onDemandBarePtrCallConv;
    bishengir::LLVMTypeConverter staticConverter(&getContext(), staticOptions);
    mlir::hivm::populateHIVMToLLVMConversionPatterns(staticConverter, patterns,
                                                     isRegBased);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();

    // debug-related external functions linkage settings
    auto funcOpWalkFn = [](LLVM::LLVMFuncOp llvmFuncOp) {
      auto funcNameStr = llvmFuncOp.getSymName();
      if (funcNameStr.starts_with("_mlir_ciface_init_debug") ||
          funcNameStr.starts_with("_mlir_ciface_finish_debug") ||
          funcNameStr.starts_with("_mlir_ciface_print_") ||
          funcNameStr.starts_with("_mlir_ciface_assert_")) {
        llvmFuncOp.setLinkage(LLVM::Linkage::ExternWeak);
      }
      return WalkResult::advance();
    };

    static constexpr llvm::StringRef kMemrefExtVolatile = "memref_ext.volatile";

    // mark ssbuf-related load/store ops as volatile to prevent ccec from
    // hoisting
    auto loadOpWalkFn = [](LLVM::LoadOp loadOp) {
      if (!isLLVMLoadStoreOnSsbuf(loadOp)) {
        return WalkResult::advance();
      };
      auto markOpOpt = utils::getAnnotateOpWithAttr(loadOp, kMemrefExtVolatile);
      if (!markOpOpt.has_value()) {
        return WalkResult::advance();
      }
      loadOp.setVolatile_(true);
      auto markOp = markOpOpt.value();
      markOp->removeAttr(kMemrefExtVolatile);
      if (llvm::all_of(markOp->getAttrs(), [](NamedAttribute attr) {
            return attr.getName() == "effect";
          })) {
        markOp->erase();
      }
      return WalkResult::advance();
    };

    auto storeOpWalkFn = [](LLVM::StoreOp storeOp) {
      if (isLLVMLoadStoreOnSsbuf(storeOp)) {
        // TODO: think of a better way of handling ssbuf stores.
        storeOp.setVolatile_(true);
      };
      return WalkResult::advance();
    };

    getOperation()->walk([&](Operation *op) {
      return llvm::TypeSwitch<Operation *, WalkResult>(op)
          .Case<LLVM::LLVMFuncOp>(funcOpWalkFn)
          .Case<LLVM::LoadOp>(loadOpWalkFn)
          .Case<LLVM::StoreOp>(storeOpWalkFn)
          .Default([](auto) { return WalkResult::advance(); });
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PointerCastOp Lowering
//===----------------------------------------------------------------------===//
struct PointerCastOpLowering
    : public ConvertOpToLLVMPattern<hivm::PointerCastOp> {
  using ConvertOpToLLVMPattern<hivm::PointerCastOp>::ConvertOpToLLVMPattern;

  explicit PointerCastOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<hivm::PointerCastOp>(converter) {}

  LogicalResult
  matchAndRewrite(hivm::PointerCastOp op,
                  typename hivm::PointerCastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType memRefType = cast<MemRefType>(op->getResult(0).getType());
    auto memorySpace = memRefType.getMemorySpace();
    unsigned addrSpace;
    // Handle hivm AddressSpace.
    if (memorySpace) {
      auto hivmAddressSpace = dyn_cast<AddressSpaceAttr>(memorySpace);
      if (!hivmAddressSpace)
        return failure();
      addrSpace = static_cast<unsigned>(hivmAddressSpace.getAddressSpace());
    } else
      addrSpace = memRefType.getMemorySpaceAsInt();
    if (op->getNumOperands() == 0) // Addr argument not specified.
      return failure();

    Type pType = LLVM::LLVMPointerType::get(rewriter.getContext(), addrSpace);
    Value allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(
        op.getLoc(), pType, op->getOperands()[0]);
    // Create the MemRef descriptor.
    Value size;
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    getMemRefDescriptorSizes(op.getLoc(), memRefType, adaptor.getDynamicSizes(),
                             rewriter, sizes, strides, size);
    MemRefDescriptor descriptor =
        createMemRefDescriptor(op.getLoc(), memRefType, allocatedPtr,
                               allocatedPtr, sizes, strides, rewriter);
    rewriter.replaceOp(op, {descriptor});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BitcastOp Lowering
//===----------------------------------------------------------------------===//
struct BitcastOpLowering : public ConvertOpToLLVMPattern<hivm::BitcastOp> {
  using ConvertOpToLLVMPattern<hivm::BitcastOp>::ConvertOpToLLVMPattern;

  explicit BitcastOpLowering(LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<hivm::BitcastOp>(converter) {}

  LogicalResult
  matchAndRewrite(hivm::BitcastOp bitcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = bitcastOp.getLoc();
    // memref<f32> -> !llvm.struct<(allocatedPtr, alignedPtr, offset)

    // Get result memref descriptor type
    auto resultMemRefType = cast<MemRefType>(bitcastOp.getResult().getType());
    auto resultDescType = typeConverter->convertType(resultMemRefType);
    if (!resultDescType)
      return failure();

    // Create descriptors for source and result
    MemRefDescriptor srcDesc(adaptor.getSrc());
    auto resultDesc = MemRefDescriptor::undef(rewriter, loc, resultDescType);

    // Copy the allocated and aligned pointers
    Value allocatedPtr = srcDesc.allocatedPtr(rewriter, loc);
    Value alignedPtr = srcDesc.alignedPtr(rewriter, loc);
    resultDesc.setAllocatedPtr(rewriter, loc, allocatedPtr);
    resultDesc.setAlignedPtr(rewriter, loc, alignedPtr);

    // Copy the offset
    Value offset = srcDesc.offset(rewriter, loc);
    resultDesc.setOffset(rewriter, loc, offset);

    // Copy sizes and strides for each dimension
    for (unsigned i = 0; i < resultMemRefType.getRank(); ++i) {
      Value size = srcDesc.size(rewriter, loc, i);
      Value stride = srcDesc.stride(rewriter, loc, i);

      resultDesc.setSize(rewriter, loc, i, size);
      resultDesc.setStride(rewriter, loc, i, stride);
    }

    rewriter.replaceOp(bitcastOp, {resultDesc});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Marked memref.cast Lowering (fold a dynamic offset into the base pointer)
//===----------------------------------------------------------------------===//
struct FoldOffsetIntoPtrCastOpLowering
    : public ConvertOpToLLVMPattern<memref::CastOp> {
  explicit FoldOffsetIntoPtrCastOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<memref::CastOp>(converter, /*benefit=*/2) {}

  LogicalResult
  matchAndRewrite(memref::CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only act on casts the producer explicitly marked.
    if (!castOp->hasAttr(kFoldOffsetMarker))
      return failure();

    Location loc = castOp.getLoc();

    auto resultMemRefType = dyn_cast<MemRefType>(castOp.getType());
    if (!resultMemRefType)
      return failure();

    Type resultDescType = typeConverter->convertType(resultMemRefType);
    if (!resultDescType)
      return failure();

    Value srcStruct = adaptor.getSource();
    if (!isa<LLVM::LLVMStructType>(srcStruct.getType()))
      return failure();

    MemRefDescriptor srcDesc(srcStruct);
    Value allocatedPtr = srcDesc.allocatedPtr(rewriter, loc);
    Value alignedPtr = srcDesc.alignedPtr(rewriter, loc);
    Value offset = srcDesc.offset(rewriter, loc);

    // The single pointer add: aligned_ptr + offset, scaled by sizeof(element).
    Type elementType =
        typeConverter->convertType(resultMemRefType.getElementType());

    auto alloPtrType = cast<LLVM::LLVMPointerType>(allocatedPtr.getType());
    Value newAllocatedPtr = rewriter.create<LLVM::GEPOp>(
        loc, alloPtrType, elementType, allocatedPtr, ValueRange{offset});

    auto alignPtrType = cast<LLVM::LLVMPointerType>(alignedPtr.getType());
    Value newAlignedPtr = rewriter.create<LLVM::GEPOp>(
        loc, alignPtrType, elementType, alignedPtr, ValueRange{offset});

    // Bare-pointer convention
    if (isa<LLVM::LLVMPointerType>(resultDescType)) {
      rewriter.replaceOp(castOp, {newAlignedPtr});
      return success();
    }

    // Descriptor convention: rebuild with the advanced pointer and offset = 0,
    // copying sizes/strides unchanged.
    auto resultDesc = MemRefDescriptor::undef(rewriter, loc, resultDescType);
    resultDesc.setAllocatedPtr(rewriter, loc, newAllocatedPtr);
    resultDesc.setAlignedPtr(rewriter, loc, newAlignedPtr);
    resultDesc.setOffset(
        rewriter, loc,
        createIndexAttrConstant(rewriter, loc, getIndexType(), 0));
    for (unsigned i = 0; i < resultMemRefType.getRank(); ++i) {
      resultDesc.setSize(rewriter, loc, i, srcDesc.size(rewriter, loc, i));
      resultDesc.setStride(rewriter, loc, i, srcDesc.stride(rewriter, loc, i));
    }

    rewriter.replaceOp(castOp, {resultDesc});
    return success();
  }
};

void mlir::hivm::populateHIVMToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool isRegBased) {
  populateHIVMLegalizeForLLVMExportPatterns(converter, patterns, isRegBased);
  mlir::hivm::populateHIVMAddressSpaceAttributeConversions(converter);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  mlir::arith::populateArithToHIVMLLVMConversionPatterns(converter, patterns);
  mlir::populateMathToLLVMConversionPatterns(converter, patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  patterns.add<PointerCastOpLowering>(converter);
  patterns.add<BitcastOpLowering>(converter);
  patterns.add<FoldOffsetIntoPtrCastOpLowering>(converter);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

  // Func dialect related conversion patterns.
  if (converter.getOptions().onDemandBarePtrCallConv) {
    bishengir::populateFuncToLLVMFuncOpConversionPattern(converter, patterns);
  }
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
}

std::unique_ptr<Pass> mlir::createConvertHIVMToLLVMPass() {
  return std::make_unique<ConvertHIVMToLLVM>();
}

std::unique_ptr<Pass>
mlir::createConvertHIVMToLLVMPass(const ConvertHIVMToLLVMOptions &options) {
  return std::make_unique<ConvertHIVMToLLVM>(options);
}
