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

namespace {
struct InsertNZ2NDForDebug
    : public impl::InsertNZ2NDForDebugBase<InsertNZ2NDForDebug> {
  using Base::Base;
  void runOnOperation() override;
};

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
  patterns.add<InsertNZ2NDForDebugPattern>(patterns.getContext());
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertNZ2NDForDebugPass() {
  return std::make_unique<InsertNZ2NDForDebug>();
}
