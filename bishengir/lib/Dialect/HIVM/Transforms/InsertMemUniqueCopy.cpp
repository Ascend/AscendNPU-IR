//===- InsertMemUniqueCopy.h -----------------------------------------*- C++
//-*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "bishengir/Dialect/HIVM/Transforms/InsertMemUniqueCopy.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "insert-mem-unique-copy"

namespace mlir {
#define GEN_PASS_DEF_INSERTMEMUNIQUECOPY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

template <typename OpTy>
class InsertMemUniqueCopy : public OpRewritePattern<OpTy> {
public:
  InsertMemUniqueCopy(MLIRContext *context,
                      const DenseSet<Value> &uniqueBuffers)
      : OpRewritePattern<OpTy>(context), uniqueBuffers(uniqueBuffers) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    bool modified = false;
    for (auto &opOperand : op->getOpOperands()) {
      Value value = opOperand.get();
      auto maybeRootValue = utils::tracebackMemRefToAlloc(value);
      if (!maybeRootValue.has_value()) {
        continue;
      }
      if (uniqueBuffers.contains(maybeRootValue.value())) {
        Value dstValue = utils::createEmptyOp(rewriter, op->getLoc(), value);
        rewriter.create<hivm::CopyOp>(op->getLoc(), TypeRange{},
                                      /*src*/ value, /*dst*/ dstValue);
        rewriter.modifyOpInPlace(op, [&]() {
          op.setOperand(opOperand.getOperandNumber(), dstValue);
        });
        modified = true;
      }
    }
    return success(modified);
  }

private:
  const DenseSet<Value> &uniqueBuffers;
};

} // anonymous namespace

void mlir::hivm::populateInsertMemUniqueCopyPattern(
    RewritePatternSet &patterns, const DenseSet<Value> &uniqueBuffers) {
  patterns.add<
      InsertMemUniqueCopy<scf::ConditionOp>, InsertMemUniqueCopy<scf::YieldOp>,
      InsertMemUniqueCopy<scf::ForOp>, InsertMemUniqueCopy<scf::WhileOp>>(
      patterns.getContext(), uniqueBuffers);
}