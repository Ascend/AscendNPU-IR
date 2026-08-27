//===- TritonGlobalKernelArgsToHIVMOp.cpp ---------------------------------===//
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
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_TRITONGLOBALKERNELARGSTOHIVMOP
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "triton-global-kernel-args-to-hivm-op"

using namespace mlir;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// GlobalKernelArgsToHIVMOpPass
//===----------------------------------------------------------------------===//
namespace {
static inline constexpr int kProgramNumArgsNum = 3;
static inline constexpr int kProgramIdArgsNum = 3;

/// This pass convert global kernel function arguments to hivm op
struct TritonGlobalKernelArgsToHIVMOpPass
    : public impl::TritonGlobalKernelArgsToHIVMOpBase<
          TritonGlobalKernelArgsToHIVMOpPass> {
  using TritonGlobalKernelArgsToHIVMOpBase<
      TritonGlobalKernelArgsToHIVMOpPass>::TritonGlobalKernelArgsToHIVMOpBase;

public:
  void runOnOperation() override;
};
} // end anonymous namespace

// The launch grid of triton is always 3D while hivm::get_block_idx is just 1D.
// So the following wanna transform 1D index to 3D.
//
// Currently, shape of triton launch grid, like [x, y, z], will be really passed
// as final three i32 args of global kernel.
// And before this pass, final six i32 args of global kernel represent orderly
// three PROGRAM_NUM_ARGS and three PROGRAM_ID_ARGS. Therefore PROGRAM_NUM_ARGS
// is equivalent to the 3 actual args, [x, y, z], and PROGRAM_ID_ARGS will
// later be erased from func args.
//
// The program_id decode follows Triton's x-fastest launch order:
// idx = hivm::get_block_idx
//     = program_id_0
//     + program_id_1 * program_num_0(x)
//     + program_id_2 * program_num_0(x) * program_num_1(y)
// so,
// program_id_0 = idx // (1)     mod x
// program_id_1 = idx // (x)     mod y
// program_id_2 = idx // (x * y) mod z
//
// FixMe: How to take advantage of hivm::get_block_num?
LogicalResult replaceProgramID(func::FuncOp funOp, IRRewriter &rewriter) {
  constexpr int kGridArgsNum = kProgramNumArgsNum + kProgramIdArgsNum;
  const int argNum = static_cast<int>(funOp.getNumArguments());
  // Verify whether there exist final 6 args to express BLOCK info
  if (argNum < kGridArgsNum) {
    funOp.emitError("arguments program id or program num are missing");
    return failure();
  }

  // Verify type of final 6 args.
  for (int i = argNum - kGridArgsNum; i < argNum; ++i) {
    if (funOp.getArgument(i).getType() != rewriter.getI32Type()) {
      funOp.emitError(
          "incompatible types of arguments program id or program num");
      return failure();
    }
  }

  Block &block = funOp.getBody().front();
  rewriter.setInsertionPointToStart(&block);
  mlir::Location loc = block.front().getLoc();
  const int progNumBase = argNum - kGridArgsNum;
  const int progIdBase = argNum - kProgramIdArgsNum;
  Value progNumX = funOp.getArgument(progNumBase);
  Value progNumY = funOp.getArgument(progNumBase + 1);
  Value progNumZ = funOp.getArgument(progNumBase + 2);
  auto tempMul = rewriter.create<arith::MulIOp>(loc, progNumX, progNumY);
  auto logicBlockNum = rewriter.create<arith::MulIOp>(loc, tempMul, progNumZ);
  auto mark = rewriter.create<annotation::MarkOp>(loc, logicBlockNum);
  mark->setAttr(kLogicalBlockNumAttr, rewriter.getUnitAttr());
  // Replace used program_id args
  auto hivmOp =
      rewriter.create<hivm::GetBlockIdxOp>(loc, rewriter.getI64Type());
  Value castedBlockID = rewriter.create<arith::TruncIOp>(
      loc, rewriter.getI32Type(), hivmOp.getResult());
  Value accumulateShape = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
  // Decode axes fastest-first, following the x-fastest launch order.
  for (int i = 0; i < kProgramIdArgsNum; ++i) {
    Value progNum = funOp.getArgument(progNumBase + i);
    auto indexAlongCurAxis =
        rewriter.create<arith::DivSIOp>(loc, castedBlockID, accumulateShape);
    auto realIndexAlongCurAxis =
        rewriter.create<arith::RemSIOp>(loc, indexAlongCurAxis, progNum);
    rewriter.replaceAllUsesWith(funOp.getArgument(progIdBase + i),
                                realIndexAlongCurAxis);
    if (i != kProgramIdArgsNum - 1) {
      accumulateShape =
          rewriter.create<arith::MulIOp>(loc, accumulateShape, progNum);
    }
  }

  return success();
}

void eraseReplacedFuncArgs(func::FuncOp funOp) {
  const unsigned argNum = funOp.getNumArguments();
  BitVector indicesToErase(argNum);
  indicesToErase.set(argNum - kProgramIdArgsNum, argNum);
  funOp.eraseArguments(indicesToErase);
}

void addFuncDynMemrefArgAttr(func::FuncOp funOp, IRRewriter &rewriter) {
  llvm::SmallVector<bool> memrefToDescriptorFlag;
  for (Type type : funOp.getFunctionType().getInputs()) {
    auto memref = dyn_cast<MemRefType>(type);
    memrefToDescriptorFlag.push_back(memref && !memref.hasStaticShape());
  }
  funOp->setAttr(hivm::HIVMFuncDynMemrefArgsAttr::getMnemonic(),
                 rewriter.getBoolVectorAttr(memrefToDescriptorFlag));
}

void TritonGlobalKernelArgsToHIVMOpPass::runOnOperation() {
  func::FuncOp funOp = getOperation();
  if (!funOp) {
    return;
  }
  if (!hacc::utils::isDeviceEntry(funOp)) {
    return;
  }
  MLIRContext *ctx = funOp->getContext();
  IRRewriter rewriter(ctx);
  if (failed(replaceProgramID(funOp, rewriter))) {
    return signalPassFailure();
  }
  eraseReplacedFuncArgs(funOp);
  addFuncDynMemrefArgAttr(funOp, rewriter);
}

std::unique_ptr<Pass> mlir::hivm::createTritonGlobalKernelArgsToHIVMOpPass() {
  return std::make_unique<TritonGlobalKernelArgsToHIVMOpPass>();
}
