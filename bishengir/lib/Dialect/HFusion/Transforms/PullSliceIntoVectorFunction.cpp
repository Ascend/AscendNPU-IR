//===- PullSliceIntoVectorFunction.cpp - Pull extract/insert_slice into VF -===//
//
// Part of the BiShengIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass pulls tensor.extract_slice (and optionally tensor.insert_slice)
// into Vector Function (VF) callees to help one-shot-bufferize reduce copies.
// Applied when a CallOp operand is produced by extract_slice with non-standard
// stride (stride != 1 or shape change).
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_PULLSLICEINTOVECTORFUNCTION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// PullExtractInsertSliceIntoVectorFunction: slice operand/result handling
//===----------------------------------------------------------------------===//
//
// Pull tensor.extract_slice (and optionally tensor.insert_slice) into VF
// callee to help one-shot-bufferize reduce copies. Applied when a CallOp
// operand is produced by extract_slice with non-standard stride
// (stride != 1 or shape change).
//
// Supported scenarios:
//
// [Scenario 1: Passthrough / identity-like return]
//   Caller: %slice = extract_slice(%full); %x = call @vf(%slice)
//   VF returns a value that flows from the slice arg (e.g. via block args or
//   scf.for). Caller then does extract_slice on the full result to obtain the
//   sliced view.
//   After: call passes %full + slice params; VF does extract at entry, returns
//   full tensor; caller does extract_slice on call result.
//
// [Scenario 2: Read-modify-write]
//   Caller: %slice = extract_slice(%full); %b = call @vf(%slice);
//           %c = insert_slice(%b, %full)
//   VF returns computed slice; caller writes it back into %full via
//   insert_slice. After: call passes %full + slice params; VF does extract at
//   entry, compute, insert_slice at return; call result is the full tensor
//   (insert_slice removed).
//

/// Match result for slice pull: describes which operand/result to rewrite and
/// how.
struct SlicePullMatch {
  /// The extract_slice op producing the slice passed as call operand.
  tensor::ExtractSliceOp extractSlice;

  /// Index of the call operand that is the slice (will be replaced by
  /// extractSlice.getSource() + offsets/sizes/strides).
  size_t argIdx;

  /// Index of the call result that should become full-tensor type.
  /// -1 if none. In scenario 1: the return flows from the arg; in scenario 2:
  /// the result feeds insert_slice. Rewrite changes this result to full tensor.
  int64_t fullTensorResultIdx;

  /// The insert_slice op that consumes the call result and writes back to the
  /// source. Non-null only in scenario 2; in scenario 1, caller uses
  /// extract_slice on the full result instead.
  tensor::InsertSliceOp insertSlice;
};

struct PullExtractInsertSliceIntoVectorFunction
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr(mlir::hivm::VectorFunctionAttr::name))
      return failure();
    auto funcOp =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(op);
    if (!funcOp)
      return failure();

    LogicalResult res = failure();
    func::CallOp currentCall = op;
    rewriter.setInsertionPoint(currentCall);

    while (auto match = tryMatchSliceOperand(currentCall, funcOp)) {
      modifyCalleeForSlicePull(funcOp, *match, rewriter);
      currentCall =
          replaceCallWithPulledSlice(currentCall, funcOp, *match, rewriter);
      rewriter.setInsertionPoint(currentCall);
      res = success();
    }

    return res;
  }

private:
  // True if slice is "passthrough": stride 1, no shape change (skip this
  // pattern).
  bool isStandardStride(tensor::ExtractSliceOp op) const {
    auto src = op.getSource();
    auto srcShape = cast<RankedTensorType>(src.getType()).getShape();
    auto resShape = cast<RankedTensorType>(op.getType()).getShape();
    if (auto prevOp = src.getDefiningOp<tensor::ExtractSliceOp>();
        prevOp && !isStandardStride(prevOp))
      return false;
    if (!llvm::all_of(op.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }))
      return false;
    for (auto [srcDim, resDim] :
         llvm::drop_begin(llvm::zip_equal(srcShape, resShape))) {
      if (srcDim != resDim)
        return false;
    }
    return true;
  }

  bool isNonStandardStride(tensor::ExtractSliceOp op) const {
    return !isStandardStride(op);
  }

  bool sliceParamsMatch(tensor::ExtractSliceOp extractOp,
                        tensor::InsertSliceOp insertOp) const {
    return mlir::detail::sameOffsetsSizesAndStrides(
        extractOp, insertOp, [](OpFoldResult a, OpFoldResult b) {
          return isEqualConstantIntOrValue(a, b);
        });
  }

  void appendSliceOperands(SmallVectorImpl<Value> &operands, size_t pos,
                           tensor::ExtractSliceOp extractSlice) const {
    auto offsets = extractSlice.getOffsets();
    auto sizes = extractSlice.getSizes();
    auto strides = extractSlice.getStrides();
    operands.insert(operands.begin() + pos, strides.begin(), strides.end());
    operands.insert(operands.begin() + pos, sizes.begin(), sizes.end());
    operands.insert(operands.begin() + pos, offsets.begin(), offsets.end());
  }

  void appendSliceTypes(SmallVectorImpl<Type> &types, size_t pos,
                        tensor::ExtractSliceOp extractSlice) const {
    auto offsets = TypeRange(extractSlice.getOffsets());
    auto sizes = TypeRange(extractSlice.getSizes());
    auto strides = TypeRange(extractSlice.getStrides());
    types.insert(types.begin() + pos, strides.begin(), strides.end());
    types.insert(types.begin() + pos, sizes.begin(), sizes.end());
    types.insert(types.begin() + pos, offsets.begin(), offsets.end());
  }

  SmallVector<Value> appendSliceBlockArgs(ValueRange args, size_t pos,
                                          Block &block) const {
    auto res = llvm::map_to_vector(args, [&](Value v) -> Value {
      return block.insertArgument(pos, v.getType(), v.getLoc());
    });
    std::reverse(res.begin(), res.end());
    return res;
  }

  // Traces value to originating func arg index; returns -1 if not from an arg.
  int64_t traceValueToFuncArg(Value v) const {
    if (auto blockArgument = dyn_cast<BlockArgument>(v))
      return blockArgument.getArgNumber();
    if (auto forOp = v.getDefiningOp<scf::ForOp>()) {
      auto opRes = cast<OpResult>(v);
      return traceValueToFuncArg(forOp.getInitArgs()[opRes.getResultNumber()]);
    }
    return -1;
  }

  // Returns insert_slice if callResult has exactly one user that is
  // insert_slice(dest=source) with same slice params as extractSlice; else
  // null.
  tensor::InsertSliceOp
  tryMatchInsertSliceUser(Value callResult, Value source,
                          tensor::ExtractSliceOp extractSlice) const {
    if (!callResult.hasOneUse())
      return nullptr;
    auto insOp =
        dyn_cast<tensor::InsertSliceOp>(*callResult.getUsers().begin());
    if (!insOp || insOp.getDest() != source)
      return nullptr;
    if (!sliceParamsMatch(extractSlice, insOp))
      return nullptr;
    return insOp;
  }

  std::optional<SlicePullMatch>
  tryMatchSliceOperand(func::CallOp call, func::FuncOp callee) const {
    auto operands = call.getOperands();
    for (auto [idx, operand] : llvm::enumerate(operands)) {
      auto defOp = operand.getDefiningOp<tensor::ExtractSliceOp>();
      if (!defOp || !isNonStandardStride(defOp))
        continue;

      tensor::ExtractSliceOp extractSlice = defOp;
      Value src = extractSlice.getSource();

      int64_t fullTensorResultIdx = -1;
      tensor::InsertSliceOp insertSlice = nullptr;

      // [Scenario 1] Return value flows from this arg (e.g. block arg,
      // scf.for).
      auto returnOp =
          cast<func::ReturnOp>(callee.getBody().front().getTerminator());
      for (auto [resIdx, retVal] : llvm::enumerate(returnOp.getOperands())) {
        if (traceValueToFuncArg(retVal) == static_cast<int64_t>(idx)) {
          fullTensorResultIdx = static_cast<int64_t>(resIdx);
          break;
        }
      }

      // [Scenario 2] Call result feeds insert_slice(dest=src) with same slice
      // params.
      if (fullTensorResultIdx == -1) {
        for (auto [resIdx, callResult] : llvm::enumerate(call.getResults())) {
          if (auto insOp =
                  tryMatchInsertSliceUser(callResult, src, extractSlice)) {
            fullTensorResultIdx = static_cast<int64_t>(resIdx);
            insertSlice = insOp;
            break;
          }
        }
      }

      return SlicePullMatch{extractSlice, idx, fullTensorResultIdx,
                            insertSlice};
    }
    return std::nullopt;
  }

  void modifyCalleeForSlicePull(func::FuncOp callee,
                                const SlicePullMatch &match,
                                PatternRewriter &rewriter) const {
    auto extractSlice = match.extractSlice;
    size_t idx = match.argIdx;
    int64_t fullTensorResultIdx = match.fullTensorResultIdx;

    Value src = extractSlice.getSource();
    auto offsets = extractSlice.getOffsets();
    auto sizes = extractSlice.getSizes();
    auto strides = extractSlice.getStrides();

    auto &block = callee.getBody().front();
    auto oldFuncType = callee.getFunctionType();
    SmallVector<Type> newInputTypes(oldFuncType.getInputs().begin(),
                                    oldFuncType.getInputs().end());
    SmallVector<Type> newResultTypes(oldFuncType.getResults().begin(),
                                     oldFuncType.getResults().end());

    newInputTypes[idx] = src.getType();
    appendSliceTypes(newInputTypes, idx + 1, extractSlice);
    if (fullTensorResultIdx != -1)
      newResultTypes[fullTensorResultIdx] = src.getType();

    SmallVector<Value> newOffsets, newSizes, newStrides;
    BlockArgument newArg;

    rewriter.modifyOpInPlace(callee, [&]() {
      callee.setFunctionType(
          rewriter.getFunctionType(newInputTypes, newResultTypes));
      auto oldArg = block.getArgument(idx);

      newStrides = appendSliceBlockArgs(strides, idx, block);
      newSizes = appendSliceBlockArgs(sizes, idx, block);
      newOffsets = appendSliceBlockArgs(offsets, idx, block);
      newArg = block.insertArgument(idx, src.getType(), oldArg.getLoc());

      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      auto newExtract = rewriter.create<tensor::ExtractSliceOp>(
          newArg.getLoc(), cast<RankedTensorType>(oldArg.getType()), newArg,
          newOffsets, newSizes, newStrides, extractSlice.getStaticOffsets(),
          extractSlice.getStaticSizes(), extractSlice.getStaticStrides());
      rewriter.replaceAllUsesWith(oldArg, newExtract);
      block.eraseArgument(idx + offsets.size() + sizes.size() + strides.size() +
                          1);
    });

    // Wrap return value in insert_slice so VF returns full tensor (both
    // scenarios).
    if (fullTensorResultIdx != -1) {
      auto returnOp = cast<func::ReturnOp>(block.getTerminator());
      rewriter.modifyOpInPlace(returnOp, [&]() {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(returnOp);
        Value opr = returnOp.getOperands()[fullTensorResultIdx];
        auto newInsert = rewriter.create<tensor::InsertSliceOp>(
            opr.getLoc(), cast<RankedTensorType>(newArg.getType()), opr, newArg,
            newOffsets, newSizes, newStrides, extractSlice.getStaticOffsets(),
            extractSlice.getStaticSizes(), extractSlice.getStaticStrides());
        returnOp.getOperandsMutable()[fullTensorResultIdx].set(newInsert);
      });
    }
  }

  func::CallOp replaceCallWithPulledSlice(func::CallOp call,
                                          func::FuncOp funcOp,
                                          const SlicePullMatch &match,
                                          PatternRewriter &rewriter) const {
    auto extractSlice = match.extractSlice;
    SmallVector<Value> newOperands(call.getOperands().begin(),
                                   call.getOperands().end());
    newOperands[match.argIdx] = extractSlice.getSource();
    appendSliceOperands(newOperands, match.argIdx + 1, extractSlice);

    auto newCall =
        rewriter.create<func::CallOp>(call.getLoc(), funcOp, newOperands);
    newCall->setAttrs(call->getAttrs());

    SmallVector<Value> newResults(newCall->result_begin(),
                                  newCall->result_end());
    // [Scenario 2] Call returns full tensor; replace insert_slice uses with it.
    if (match.insertSlice) {
      tensor::InsertSliceOp insertOp = match.insertSlice;
      rewriter.replaceAllUsesWith(insertOp.getResult(),
                                  newCall.getResult(match.fullTensorResultIdx));
      rewriter.eraseOp(insertOp);
    }
    // [Scenario 1] Call returns full tensor; caller needs extract_slice on
    // result.
    else if (match.fullTensorResultIdx != -1) {
      newResults[match.fullTensorResultIdx] =
          rewriter.create<tensor::ExtractSliceOp>(
              call.getLoc(), newResults[match.fullTensorResultIdx],
              extractSlice.getMixedOffsets(), extractSlice.getMixedSizes(),
              extractSlice.getMixedStrides());
    }
    rewriter.replaceOp(call, newResults);
    return newCall;
  }
};

struct PullSliceIntoVectorFunction
    : public impl::PullSliceIntoVectorFunctionBase<PullSliceIntoVectorFunction> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PullExtractInsertSliceIntoVectorFunction>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createPullSliceIntoVectorFunctionPass() {
  return std::make_unique<PullSliceIntoVectorFunction>();
}
