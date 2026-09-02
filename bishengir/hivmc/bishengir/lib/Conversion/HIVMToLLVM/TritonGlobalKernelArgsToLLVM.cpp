//===- TritonGlobalKernelArgsToLLVM.cpp - Replace and eliminate args ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to replace and eliminate args in LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

/// Since the memref is transformed into descriptor in HIVMToLLVM.The llvm ir is
/// like.
/// llvm.func device_kernel(%tensor_aligned, %tensor_allocated, %offset, %size,
/// %stride)
///   do something
///   return
/// }
/// However, the kernel launch is like:
/// extern "C" __global__ __aicore__ void device_launch_kernel
///         (__gm__ void* __restrict__ tensor, int_32 tensor_length,
///         int_32 tiling_data) {
///    call device_kernel(tensor,tensor,0,0,1,tensor_length,tiling_data);
/// }
/// To match the difference between device_kernel_launch and device_kernel, this
/// pass eliminate [%offset, %size, %stride] by replacing with constant value.
/// Meanwhile, relpace %tensor_allocated by %tensor_aligned.
/// The result is like:
/// llvm.func device_launch_kernel(%tensor_aligned, %tensor_length,
/// %tiling_data) {
///   read(%tensor_aligned, %tensor_aligned,, 0, 0, 1)
///   do something
///   return
/// }

#include "bishengir/Conversion/HIVMToLLVM/TritonGlobalKernelArgsToLLVM.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONGLOBALKERNELARGSTOLLVM
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "triton-global-kernel-args-to-llvm"

namespace {
struct TritonGlobalKernelArgsToLLVM
    : public impl::TritonGlobalKernelArgsToLLVMBase<
          TritonGlobalKernelArgsToLLVM> {
  using Base::Base;
  void runOnOperation() override;
};

inline llvm::SmallVector<bool> getMemrefToDesciptorFlag(Attribute attr) {
  auto arr = cast<DenseIntElementsAttr>(attr);
  llvm::SmallVector<bool> memrefToDesciptorFlag;
  for (auto d : arr) {
    memrefToDesciptorFlag.push_back(d.getBoolValue());
  }
  return memrefToDesciptorFlag;
}

inline size_t getStartPos(const llvm::SmallVector<bool> memrefToDesciptorFlag,
                          size_t idx) {
  size_t startPos = 0;
  for (size_t i = 0; i < idx; i++) {
    if (memrefToDesciptorFlag[i]) {
      startPos += kMemRefDescriptorArgsNum;
    } else {
      startPos++;
    }
  }
  return startPos;
}

inline std::pair<llvm::SmallVector<size_t>, llvm::SmallVector<size_t>>
getEliminateIdxVector(Attribute attr) {
  llvm::SmallVector<bool> memrefToDesciptorFlag =
      getMemrefToDesciptorFlag(attr);
  llvm::SmallVector<size_t> eliminateIdxVec;
  llvm::SmallVector<size_t> eliminateIdx2ArgIdxVec;
  for (size_t idx = 0; idx < memrefToDesciptorFlag.size(); idx++) {
    if (memrefToDesciptorFlag[idx]) {
      size_t startPos = getStartPos(memrefToDesciptorFlag, idx);
      eliminateIdxVec.push_back(startPos + kOffsetPosInMemRefDescriptor);
      eliminateIdxVec.push_back(startPos + kSizePosInMemRefDescriptor);
      eliminateIdxVec.push_back(startPos + kStridePosInMemRefDescriptor);
      eliminateIdx2ArgIdxVec.push_back(idx);
      eliminateIdx2ArgIdxVec.push_back(idx);
      eliminateIdx2ArgIdxVec.push_back(idx);
    }
  }
  return {eliminateIdxVec, eliminateIdx2ArgIdxVec};
}

inline llvm::SmallVector<size_t> getReplacedIdxVector(Attribute attr) {
  llvm::SmallVector<bool> memrefToDesciptorFlag =
      getMemrefToDesciptorFlag(attr);
  llvm::SmallVector<size_t> replacedIdxVec;

  for (size_t idx = 0; idx < memrefToDesciptorFlag.size(); idx++) {
    if (memrefToDesciptorFlag[idx])
      replacedIdxVec.push_back(getStartPos(memrefToDesciptorFlag, idx) +
                               kAllocatedPtrPosInMemRefDescriptor);
  }
  return replacedIdxVec;
}

static inline void eraseLLVMFuncArgs(LLVM::LLVMFuncOp funcOp,
                                     const BitVector &indicesToErase) {
  const auto &ctx = funcOp->getContext();

  // Update LLVM function type
  LLVM::LLVMFunctionType oldLLVMFuncType = funcOp.getFunctionType();
  llvm::ArrayRef<Type> argTypes = funcOp.getArgumentTypes();
  llvm::SmallVector<Type, 4> newArgTypes;
  for (size_t idx = 0; idx < indicesToErase.size(); ++idx) {
    if (indicesToErase[idx])
      continue;
    newArgTypes.push_back(argTypes[idx]);
  }

  auto newLLVMFuncType = LLVM::LLVMFunctionType::get(
      oldLLVMFuncType.getReturnType(), llvm::ArrayRef(newArgTypes),
      oldLLVMFuncType.getVarArg());
  funcOp.setType(newLLVMFuncType);

  // Update arguments attributes
  SmallVector<Attribute> newArgAttrs(newLLVMFuncType.getNumParams(),
                                     DictionaryAttr::get(ctx, {}));
  funcOp.setAllArgAttrs(ArrayAttr::get(ctx, newArgAttrs));

  // Update function body
  if (!funcOp.isExternal()) {
    Block &entry = funcOp.getRegion().front();
    entry.eraseArguments(indicesToErase);
  }
}

static void replaceHACCGMAddrArgs(PatternRewriter &rewriter,
                                  LLVM::LLVMFuncOp funcOp,
                                  BitVector &indicesToErase) {
  SmallVector<size_t> gmArgsIndices{};
  for (size_t i : llvm::seq(funcOp.getNumArguments())) {
    if (const auto argTypeAttr =
            funcOp.getArgAttrOfType<hacc::KernelArgTypeAttr>(
                i, hacc::KernelArgTypeAttr::name)) {
      if (argTypeAttr.getArgType() == hacc::KernelArgType::kGMAddr)
        gmArgsIndices.push_back(i);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "gmArgsIndices: [";
             llvm::interleaveComma(gmArgsIndices, llvm::dbgs());
             llvm::dbgs() << "]\n");

  if (gmArgsIndices.empty())
    return;

  auto loc = funcOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);

  for (size_t i : gmArgsIndices) {
    const size_t allocIdx = i + kAllocatedPtrPosInMemRefDescriptor;
    const size_t alignedIdx = i + kAlignedPtrPosInMemRefDescriptor;
    const size_t offsetIdx = i + kOffsetPosInMemRefDescriptor;
    const size_t sizeIdx = i + kSizePosInMemRefDescriptor;
    const size_t strideIdx = i + kStridePosInMemRefDescriptor;

    const auto alloc = funcOp.getArgument(allocIdx);
    const auto aligned = funcOp.getArgument(alignedIdx);
    const auto offset = funcOp.getArgument(offsetIdx);
    const auto size = funcOp.getArgument(sizeIdx);
    const auto stride = funcOp.getArgument(strideIdx);

    rewriter.replaceAllUsesWith(alloc, aligned);
    rewriter.replaceAllUsesWith(offset, zero);
    rewriter.replaceAllUsesWith(size, zero);
    rewriter.replaceAllUsesWith(stride, zero);

    indicesToErase.set(allocIdx);
    indicesToErase.set(offsetIdx);
    indicesToErase.set(sizeIdx);
    indicesToErase.set(strideIdx);
  }
}

struct ReplaceGlobalKernelArgsToLLVM
    : public mlir::OpRewritePattern<LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp llvmFuncOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!hacc::utils::isDeviceEntry(llvmFuncOp))
      return failure();

    BitVector indicesToErase(llvmFuncOp.getNumArguments());
    replaceHACCGMAddrArgs(rewriter, llvmFuncOp, indicesToErase);

    auto attr =
        llvmFuncOp->getAttr(hivm::HIVMFuncDynMemrefArgsAttr::getMnemonic());
    if (!attr)
      return failure();

    // eliminate [%offset, %size, %stride] from func arguments by replacing
    // users with constant value
    auto [eliminateIdxVec, eliminateIdx2ArgIdxVec] = getEliminateIdxVector(attr);
    LLVM_DEBUG({
      llvm::dbgs() << "eliminateIdxVec: [";
      llvm::interleaveComma(eliminateIdxVec, llvm::dbgs());
      llvm::dbgs() << "]\n";
      llvm::dbgs() << "eliminateIdx2ArgIdxVec: [";
      llvm::interleaveComma(eliminateIdx2ArgIdxVec, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&llvmFuncOp.getBody().front());
    auto constZero = rewriter.create<LLVM::ConstantOp>(
        llvmFuncOp->getLoc(), rewriter.getI64Type(), 0);

    constexpr StringRef directlyUsedGMArgListName = "DirectlyUsedGMArgIdxList";
    SmallVector<int64_t> argIndices;
    if (auto existingAttr =
            llvmFuncOp->getAttrOfType<ArrayAttr>(directlyUsedGMArgListName)) {
      for (auto attr : existingAttr) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          argIndices.push_back(intAttr.getInt());
        }
      }
    }

    for (size_t i = 0; i < eliminateIdxVec.size(); ++i) {
      size_t idx = eliminateIdxVec[i];
      BlockArgument operand = llvmFuncOp.getBody().getArgument(idx);
      if (operand.getUsers().empty()) {
        indicesToErase.set(idx);
        continue;
      }

      size_t argIdx = eliminateIdx2ArgIdxVec[i];
      if (llvm::is_contained(argIndices, argIdx)) {
        indicesToErase.set(idx);
        rewriter.replaceAllUsesWith(operand, constZero.getResult());
      }
      // dynamic shape's offset/size/stride must be passed in as args
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After checking offset/size/stride, indicesToErase: ";
      for (unsigned i = 0; i < indicesToErase.size(); ++i) {
        if (indicesToErase.test(i)) {
          llvm::dbgs() << i << " ";
        }
      }
      llvm::dbgs() << "\n";
    });

    // Replace %tensor_allocated by %tensor_aligned.
    llvm::SmallVector<size_t> replacedIdxVec = getReplacedIdxVector(attr);
    for (size_t allocatedIdx : replacedIdxVec) {
      BlockArgument allocated = llvmFuncOp.getBody().getArgument(allocatedIdx);
      if (allocated.getUsers().empty()) {
        indicesToErase.set(allocatedIdx);
        continue;
      }

      static_assert(kAllocatedPtrPosInMemRefDescriptor ==
                    kAlignedPtrPosInMemRefDescriptor + 1);
      BlockArgument aligned =
          llvmFuncOp.getBody().getArgument(allocatedIdx - 1);
      if (allocated.getType() != aligned.getType())
        continue;
      rewriter.replaceAllUsesWith(allocated, aligned);
      indicesToErase.set(allocatedIdx);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After all, indicesToErase: ";
      for (unsigned i = 0; i < indicesToErase.size(); ++i) {
        if (indicesToErase.test(i)) {
          llvm::dbgs() << i << " ";
        }
      }
      llvm::dbgs() << "\n";
    });

    eraseLLVMFuncArgs(llvmFuncOp, indicesToErase);

    llvmFuncOp->removeAttr(hivm::HIVMFuncDynMemrefArgsAttr::getMnemonic());
    return success();
  }
};

} // namespace
void TritonGlobalKernelArgsToLLVM::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());

  patterns.insert<ReplaceGlobalKernelArgsToLLVM>(patterns.getContext());
  if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createTritonGlobalKernelArgsToLLVMPass() {
  return std::make_unique<TritonGlobalKernelArgsToLLVM>();
}
