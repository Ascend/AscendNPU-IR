//===- ConvertMemRefToBarePtr.cpp - Replace memref args to base ptr -------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to replace and eliminate args in LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

/// Since the memref is transformed into descriptor in HIVMToLLVM.The llvm ir is
/// like.
/// llvm.func device_kernel(%tensor_aligned, %tensor_allocated, %offset, %size,
///                         %stride, %tensor_length, %tiling_data) {
///   read(%tensor_aligned, %tensor_allocated, %offset, %size, %stride)
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

#include "bishengir/Conversion/HIVMToLLVM/ConvertMemRefToBarePtr.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOBAREPTR
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "convert-memref-to-bare-ptr"

namespace {
struct ConvertMemRefToBarePtr
    : public impl::ConvertMemRefToBarePtrBase<ConvertMemRefToBarePtr> {
  using Base::Base;
  void runOnOperation() override;
};

inline llvm::SmallVector<bool> getMemrefToDescriptorFlag(Attribute attr) {
  auto arr = cast<DenseIntElementsAttr>(attr);
  llvm::SmallVector<bool> memrefToDescriptorFlag;
  for (auto d : arr) {
    memrefToDescriptorFlag.push_back(d.getBoolValue());
  }
  return memrefToDescriptorFlag;
}

inline size_t getStartPos(const llvm::SmallVector<bool> memrefToDescriptorFlag,
                          size_t idx) {
  size_t startPos = 0;
  for (size_t i = 0; i < idx; i++) {
    if (memrefToDescriptorFlag[i]) {
      startPos += kMemRefDescriptorArgsNum;
    } else {
      startPos++;
    }
  }
  return startPos;
}
inline llvm::SmallVector<size_t> getEliminateIdxVector(Attribute attr) {
  llvm::SmallVector<bool> memrefToDescriptorFlag =
      getMemrefToDescriptorFlag(attr);
  llvm::SmallVector<size_t> eliminateIdxVec;
  for (size_t idx = 0; idx < memrefToDescriptorFlag.size(); idx++) {
    if (memrefToDescriptorFlag[idx]) {
      size_t startPos = getStartPos(memrefToDescriptorFlag, idx);
      eliminateIdxVec.push_back(startPos + kOffsetPosInMemRefDescriptor);
      eliminateIdxVec.push_back(startPos + kSizePosInMemRefDescriptor);
      eliminateIdxVec.push_back(startPos + kStridePosInMemRefDescriptor);
    }
  }
  return eliminateIdxVec;
}

inline llvm::SmallVector<size_t> getReplacedIdxVector(Attribute attr) {
  llvm::SmallVector<bool> memrefToDescriptorFlag =
      getMemrefToDescriptorFlag(attr);
  llvm::SmallVector<size_t> replacedIdxVec;

  for (size_t idx = 0; idx < memrefToDescriptorFlag.size(); idx++) {
    if (memrefToDescriptorFlag[idx]) {
      replacedIdxVec.push_back(getStartPos(memrefToDescriptorFlag, idx) +
                               kMemRefAlignedPtrPosInMemRefDescriptor);
    }
  }
  return replacedIdxVec;
}

inline void replaceEliminatedVal(Attribute attr, LLVM::LLVMFuncOp op,
                                 mlir::PatternRewriter &rewriter) {
  auto ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&op.getBody().front());
  auto constOne =
      rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), 1);
  auto constZero =
      rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(), 0);
  rewriter.restoreInsertionPoint(ip);
  llvm::SmallVector<bool> memrefToDescriptorFlag =
      getMemrefToDescriptorFlag(attr);
  for (size_t idx = 0; idx < memrefToDescriptorFlag.size(); idx++) {
    if (memrefToDescriptorFlag[idx]) {
      size_t startPos = getStartPos(memrefToDescriptorFlag, idx);
      assert((startPos + kStridePosInMemRefDescriptor) < op.getBody().getNumArguments() && "func signature has broken.");
      BlockArgument operand =
          op.getBody().getArgument(startPos + kOffsetPosInMemRefDescriptor);
      rewriter.replaceAllUsesWith(operand, constZero.getResult());
      operand = op.getBody().getArgument(startPos + kSizePosInMemRefDescriptor);
      rewriter.replaceAllUsesWith(operand, constOne.getResult());
      operand =
          op.getBody().getArgument(startPos + kStridePosInMemRefDescriptor);
      rewriter.replaceAllUsesWith(operand, constOne.getResult());
    }
  }
}

struct ReplaceGlobalKernelArgsToLLVM
    : public mlir::OpRewritePattern<LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp llvmFuncOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (!hacc::utils::isDeviceEntry(llvmFuncOp)) {
      return failure();
    }

    // --------------------------------------------------------

      for (unsigned int idx = 0; idx < llvmFuncOp.getNumArguments(); ++idx) {
        auto dictAttr = llvmFuncOp.getArgAttrDict(idx);
        if (!dictAttr) {
          break;
        }

        bool hasDivisibility = dictAttr.contains("tt.divisibility");
        bool hasTensorkind = dictAttr.contains("tt.tensor_kind");

        if (hasTensorkind && hasDivisibility) {
          llvmFuncOp.removeArgAttr(idx, "tt.divisibility");
          llvmFuncOp.removeArgAttr(idx, "tt.tensor_kind");
        }
        if (auto argAttr = llvmFuncOp.getArgAttr(idx, "hacc.arg_type")) {
            auto castArg = dyn_cast<hacc::KernelArgTypeAttr>(argAttr);
            if (!castArg) {
              break;
            }
            auto argName = castArg.getArgType();
            if (argName == hacc::KernelArgType::kSyncBlockLock || argName == hacc::KernelArgType::kWorkspace) {
              llvmFuncOp.removeArgAttr(idx, "hacc.arg_type");
            }
        }
      }

    // -------------------------------------------------------

    auto attr =
        llvmFuncOp->getAttr(hivm::HIVMFuncDynMemrefArgsAttr::getMnemonic());
    if (!attr) {
      return failure();
    }

    // eliminate [%offset, %size, %stride] from func arguments by replacing
    // users with constant value
    llvm::SmallVector<size_t> eliminateIdxVec = getEliminateIdxVector(attr);

    replaceEliminatedVal(attr, llvmFuncOp, rewriter);
// -------------------------------------------------------------------------------
    // replace %tensor_allocated by %tensor_aligned.
    llvm::SmallVector<size_t> replacedIdxVec = getReplacedIdxVector(attr);
    for (size_t idx : replacedIdxVec) {
      BlockArgument replacedOperand = llvmFuncOp.getBody().getArgument(idx);
      if (replacedOperand.getUsers().empty()) {
        continue;
      }
      BlockArgument replaceOperand = llvmFuncOp.getBody().getArgument(idx - 1);
      rewriter.replaceAllUsesWith(replacedOperand, replaceOperand);
    }
// -------------------------------------------------------------------------------
    // erase argument
    BitVector indicesToErase(llvmFuncOp.getArguments().size());
    for (auto idx : replacedIdxVec) {
      indicesToErase.set(idx);
    }
    for (auto idx : eliminateIdxVec) {
      indicesToErase.set(idx);
    }

    Block &entry = llvmFuncOp.getRegion().front();
    entry.eraseArguments(indicesToErase);

    // update llvmtype.
    mlir::LLVM::LLVMFunctionType llvmType = llvmFuncOp.getFunctionType();
    llvm::ArrayRef<Type> argumentTypes = llvmFuncOp.getArgumentTypes();
    llvm::SmallVector<Type> newArgumentTypes;
    for (size_t idx = 0; idx < indicesToErase.size(); idx++) {
      if (indicesToErase[idx]) {
        continue;
      }
      newArgumentTypes.push_back(argumentTypes[idx]);
    }
    Type newllvmType = mlir::LLVM::LLVMFunctionType::get(
        llvmType.getReturnType(), llvm::ArrayRef(newArgumentTypes),
        llvmType.getVarArg());
    llvmFuncOp.setType(newllvmType);

    // update ArgAttrs.
    SmallVector<Attribute> newArgAttrs(
        cast<LLVM::LLVMFunctionType>(newllvmType).getNumParams());
    for (size_t j = 0;
         j < cast<LLVM::LLVMFunctionType>(newllvmType).getNumParams(); ++j)
      newArgAttrs[j] = DictionaryAttr::get(rewriter.getContext(), {});
    llvmFuncOp.setAllArgAttrs(rewriter.getArrayAttr(newArgAttrs));

    llvmFuncOp->removeAttr(hivm::HIVMFuncDynMemrefArgsAttr::getMnemonic());
    return success();
  }
};

} // namespace
void ConvertMemRefToBarePtr::runOnOperation() {
  ModuleOp op = cast<ModuleOp>(getOperation());
  if (!op->hasAttr(utils::kMemrefAsPtr)) {
    return;
  }

  RewritePatternSet patterns(&getContext());
  patterns.insert<ReplaceGlobalKernelArgsToLLVM>(patterns.getContext());
  if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createConvertMemRefToBarePtrPass() {
  return std::make_unique<ConvertMemRefToBarePtr>();
}
