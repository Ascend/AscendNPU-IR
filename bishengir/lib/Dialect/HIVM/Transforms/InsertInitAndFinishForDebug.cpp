//===------------------ InsertInitAndFinishForDebug.cpp -------------------===//
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
// This pass inserts hivm.hir.init_debug and hivm.hir.finish_debug
// for hivm.hir.debug.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTINITANDFINISHFORDEBUG
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-insert-init-and-finish-for-debug"

namespace {
struct InsertInitAndFinishForDebug
    : public impl::InsertInitAndFinishForDebugBase<
          InsertInitAndFinishForDebug> {
  using Base::Base;
  void runOnOperation() override;
};

bool hasDebugOp(func::FuncOp funcOp) {
  bool hasDebugOp = false;
  funcOp.walk([&](Operation *opInner) {
    if (isa<hivm::DebugOp>(opInner)) {
      hasDebugOp = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasDebugOp;
}

bool hasInitDebugOp(func::FuncOp funcOp) {
  bool hasInitDebugOp = false;
  funcOp.walk([&](Operation *opInner) {
    if (isa<hivm::InitDebugOp>(opInner)) {
      hasInitDebugOp = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasInitDebugOp;
}

struct InsertFinish : public OpRewritePattern<hivm::DebugOp> {
  constexpr static llvm::StringRef finishInsertionMarker = "finishInserted";
  using OpRewritePattern<hivm::DebugOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::DebugOp debugOp,
                                PatternRewriter &rewriter) const override {
    // first check if the FinishDebugOp is already inserted
    if (debugOp.getOperation()->hasAttr(finishInsertionMarker)) {
      return failure();
    }
    // if it's not inserted, then insert it
    rewriter.setInsertionPointAfter(debugOp);
    rewriter.create<hivm::FinishDebugOp>(debugOp.getLoc());
    debugOp.getOperation()->setAttr(finishInsertionMarker,
                                    rewriter.getI32IntegerAttr(0));
    return success();
  }
};

void InsertInitAndFinishForDebug::runOnOperation() {
  func::FuncOp funcOp = cast<func::FuncOp>(getOperation());
  // check if debug is present
  if (!hasDebugOp(funcOp)) {
    return;
  }
  // check if init and finish have been inserted
  if (hasInitDebugOp(funcOp)) {
    return;
  }

  // insert init at the entry
  MLIRContext *context = &getContext();
  OpBuilder builder(context);
  Region &body = funcOp.getBody();
  Block &block = body.front();
  builder.setInsertionPointToStart(&block);
  builder.create<hivm::InitDebugOp>(funcOp.getLoc());

  // insert finish for every debug
  RewritePatternSet patterns(context);
  patterns.add<InsertFinish>(patterns.getContext());
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createInsertInitAndFinishForDebugPass() {
  return std::make_unique<InsertInitAndFinishForDebug>();
}
