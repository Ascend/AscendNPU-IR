//===- TritonGlobalKernelArgsToHIVMOp.cpp - Convert args of global kernel -===//
//===- function to HIVM Ops -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
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
// New program_id expression:
// idx = hivm::get_block_idx
// idx = program_id_0 * program_num_1(y) * program_num_2(z)
//     + program_id_1 * program_num_2(z)
//     + program_id_2
// so,
// program_id_2 = idx // (1)     mod z
// program_id_1 = idx // (z)     mod y
// program_id_0 = idx // (y * z) mod x
//
// FixMe: How to take advantage of hivm::get_block_num?
LogicalResult replaceProgramID(func::FuncOp funOp, IRRewriter &rewriter) {
  auto args = funOp.getArguments();
  auto argNum = funOp.getNumArguments();
  // Verify whether there exist final 6 args to express BLOCK info
  if (argNum < kProgramIdArgsNum + kProgramNumArgsNum) {
    funOp.emitError("arguments program id or program num are missing");
    return failure();
  }

  // Verify type of final 6 args.
  for (auto itr = (args.end() - (kProgramIdArgsNum + kProgramNumArgsNum));
       itr != args.end(); itr++) {
    if ((*itr).getType() != rewriter.getI32Type()) {
      funOp.emitError(
          "incompatible types of arguments program id or program num");
      return failure();
    }
  }

  Block &block = funOp.getBody().front();
  rewriter.setInsertionPointToStart(&block);
  mlir::Location loc = block.front().getLoc();
  auto *argEnd = args.end();
  auto progID0 = argEnd[-(kProgramNumArgsNum + kProgramIdArgsNum)];
  auto progID1 = argEnd[-(kProgramNumArgsNum + kProgramIdArgsNum) + 1];
  auto progID2 = argEnd[-(kProgramNumArgsNum + kProgramIdArgsNum) + 2];
  auto tempMul = rewriter.create<arith::MulIOp>(loc, progID0, progID1);
  auto logicBlockNum = rewriter.create<arith::MulIOp>(loc, tempMul, progID2);
  auto mark = rewriter.create<annotation::MarkOp>(loc, logicBlockNum);
  mark->setAttr(kLogicalBlockNumAttr, rewriter.getUnitAttr());
  // Replace used program_id args
  auto hivmOp =
      rewriter.create<hivm::GetBlockIdxOp>(loc, rewriter.getI64Type());
  Value castedBlockID = rewriter.create<arith::TruncIOp>(
      loc, rewriter.getI32Type(), hivmOp.getResult());
  Value accumulateShape = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
  auto argProgNumAxis0 =
      (args.end() - (kProgramNumArgsNum + kProgramIdArgsNum));
  for (int i = kProgramIdArgsNum - 1; i >= 0; --i) {
    auto curProgID = args.end() - (kProgramIdArgsNum) + i;

    auto indexAlongCurAxis =
        rewriter.create<arith::DivSIOp>(loc, castedBlockID, accumulateShape);
    auto realIndexAlongCurAxis = rewriter.create<arith::RemSIOp>(
        loc, indexAlongCurAxis, *(argProgNumAxis0 + i));
    rewriter.replaceAllUsesWith(*curProgID, realIndexAlongCurAxis);
    if (i != 0) {
      accumulateShape = rewriter.create<arith::MulIOp>(loc, accumulateShape,
                                                       *(argProgNumAxis0 + i));
    }
  }

  return success();
}

void eraseReplacedFuncArgs(func::FuncOp funOp) {
  auto argNum = funOp.getNumArguments();
  BitVector indicesToErase(argNum);
  for (auto argIndex : llvm::seq<int>(0, (kProgramIdArgsNum))) {
    indicesToErase.set(argNum - 1 - argIndex);
  }
  funOp.eraseArguments(indicesToErase);
}

void addFuncDynMemrefArgAttr(func::FuncOp funOp, IRRewriter &rewriter) {
  auto argumentArgs = funOp.getArguments();
  llvm::SmallVector<bool> memrefToDescriptorFlag(argumentArgs.size(), 0);
  FunctionType funcTy = funOp.getFunctionType();
  for (auto [idx, type] : llvm::enumerate(funcTy.getInputs())) {
    auto memref = dyn_cast<MemRefType>(type);
    if (memref != nullptr && !memref.hasStaticShape()) {
      memrefToDescriptorFlag[idx] = true;
    }
  }
  funOp->setAttr(
      hivm::HIVMFuncDynMemrefArgsAttr::getMnemonic(),
      rewriter.getBoolVectorAttr(llvm::ArrayRef(memrefToDescriptorFlag)));
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
