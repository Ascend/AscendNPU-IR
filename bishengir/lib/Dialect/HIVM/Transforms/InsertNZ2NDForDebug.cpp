//===--------------------- InsertNZ2NDForDebug.cpp ------------------------===//
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
// This pass inserts the nz2nd op for debug.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTNZ2NDFORDEBUG
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-insert-nz2nd-for-debug"
static constexpr llvm::StringLiteral alreadyInsertNZ2ND = "alreadyInsertNZ2ND";

namespace {
struct InsertNZ2NDForDebug
    : public impl::InsertNZ2NDForDebugBase<InsertNZ2NDForDebug> {
  using Base::Base;
  void runOnOperation() override;
};

LogicalResult insertNZ2NDForOperand(PatternRewriter &rewriter,
                                    hivm::DebugOp debugOp) {
  Value operand = debugOp.getArg();
  Operation *definingOp = operand.getDefiningOp();
  auto resultTensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());

  rewriter.setInsertionPointAfter(definingOp);
  auto shape = resultTensorType.getShape();
  Location loc = definingOp->getLoc();
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return emitError(operand.getLoc())
             << "insertNZ2NDForOperand requires static shape, but got dynamic "
                "dimension in "
             << resultTensorType;
    }
  }

  // step1: Allocate the GM space
  Value workSpaceTensor = getLocalWorkSpaceTensor(
      rewriter, definingOp->getLoc(), shape,
      hivm::getTensorDynamicValues(rewriter, loc, operand),
      getElementTypeOrSelf(resultTensorType));

  // step2: Call nz2nd to move the data to the GM so that it can be printed
  // by device_print
  auto nz2nd = rewriter.create<hivm::NZ2NDOp>(
      definingOp->getLoc(), workSpaceTensor.getType(),
      /*src=*/operand, /*dst=*/workSpaceTensor);

  // step3: Replace printed value of L1 device_print
  rewriter.modifyOpInPlace(debugOp, [&]() {
    debugOp.getArgMutable().assign(nz2nd.getResultTensor());
    debugOp->setAttr(alreadyInsertNZ2ND, rewriter.getBoolAttr(true));
    debugOp.setMemscopeAttr(hivm::AddressSpaceAttr::get(
        debugOp.getContext(), hivm::AddressSpace::GM));
  });
  return success();
}

/// Insert nz2nd for the inputs of hivm::DebugOp.
struct InsertNZ2NDForDebugOpPattern : public OpRewritePattern<hivm::DebugOp> {
public:
  using OpRewritePattern<hivm::DebugOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::DebugOp op,
                                PatternRewriter &rewriter) const override {
    auto memscope = op.getMemscope();
    bool isL1Memspace =
        memscope && memscope->getAddressSpace() == hivm::AddressSpace::L1;
    if (!isL1Memspace || op->getAttr(alreadyInsertNZ2ND))
      return failure();

    auto printValue = op.getArg();
    auto resultTensorType =
        mlir::dyn_cast<RankedTensorType>(printValue.getType());
    if (!resultTensorType) {
      return failure();
    }
    if (printValue.getDefiningOp() == nullptr) {
      return failure();
    }

    ArrayRef<int64_t> shape = resultTensorType.getShape();
    assert(shape.size() == 2 || shape.size() == 3);
    return insertNZ2NDForOperand(rewriter, op);
/// Insert nz2nd for the inputs of hivm::MmadL1Op.
struct InsertNZ2NDForDebugPattern : public OpRewritePattern<hivm::MmadL1Op> {
public:
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> l1values = {op.getA(), op.getB()};
    bool allInserted = true;
    for (Value val : l1values) {
      if (!isa<TensorType>(val.getType())) {
        // currently only support tensors
        continue;
      }
      TensorType tensorType = cast<TensorType>(val.getType());
      if (val.getDefiningOp() == nullptr) {
        // currently only support MmadL1Op inputs with defining op
        continue;
      }
      Operation *definingOp = val.getDefiningOp();
      bool inserted = false;
      for (Operation *user : val.getUsers()) {
        if (isa<hivm::NZ2NDOp>(user)) {
          inserted = true;
          break;
        }
      }
      if (inserted) {
        continue;
      }
      allInserted = false;
      for (Operation *user : val.getUsers()) {
        if (isa<hivm::DebugOp>(user)) {
          hivm::DebugOp debugOp = cast<hivm::DebugOp>(user);
          rewriter.setInsertionPointAfter(definingOp);
          Value workSpaceTensor = getLocalWorkSpaceTensor(
              rewriter, definingOp->getLoc(), tensorType.getShape(),
              getElementTypeOrSelf(tensorType));
          auto res = rewriter.create<hivm::NZ2NDOp>(
              definingOp->getLoc(), workSpaceTensor.getType(),
              /*src=*/val, /*dst=*/workSpaceTensor);
          rewriter.modifyOpInPlace(debugOp, [&]() {
            OpOperand &arg = debugOp.getArgMutable();
            arg.assign(res.getResultTensor());
          });
        }
      }
    }
    return allInserted ? failure() : success();
  }
};

void InsertNZ2NDForDebug::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<InsertNZ2NDForDebugOpPattern>(patterns.getContext());
  patterns.add<InsertNZ2NDForDebugPattern>(patterns.getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertNZ2NDForDebugPass() {
  return std::make_unique<InsertNZ2NDForDebug>();
}
}
