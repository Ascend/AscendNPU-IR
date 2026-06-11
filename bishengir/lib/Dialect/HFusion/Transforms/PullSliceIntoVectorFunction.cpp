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
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
    for (int64_t off : op.getStaticOffsets()) {
      if (!ShapedType::isDynamic(off) && off != 0)
        return false;
    }
    if (resShape.size() != srcShape.size())
      return false;
    if (srcShape.size() == 1)
      return srcShape[0] == resShape[0];
    for (auto [srcDim, resDim] :
         llvm::drop_begin(llvm::zip_equal(srcShape, resShape))) {
      if (srcDim != resDim)
        return false;
    }
    return true;
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
      if (!defOp || isStandardStride(defOp))
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
    // result. Use original slice type for rank-reduce (e.g. 2D->1D).
    else if (match.fullTensorResultIdx != -1) {
      newResults[match.fullTensorResultIdx] =
          rewriter.create<tensor::ExtractSliceOp>(
              call.getLoc(), extractSlice.getType(),
              newResults[match.fullTensorResultIdx],
              extractSlice.getMixedOffsets(), extractSlice.getMixedSizes(),
              extractSlice.getMixedStrides());
    }
    rewriter.replaceOp(call, newResults);
    return newCall;
  }
};

//===----------------------------------------------------------------------===//
// PullInsertedSliceOperandIntoVectorFunction
//===----------------------------------------------------------------------===//
//
// Pull a call operand built as:
//
//   %patch = tensor.extract_slice %patch_src[...]
//   %base  = tensor.extract_slice %base_src[...]
//   %arg   = tensor.insert_slice %patch into %base[...]
//   call @vf(%arg, ...)
//
// into the VF callee.  The caller passes %base_src, %patch_src and all dynamic
// slice operands; the callee reconstructs %arg at function entry.
struct PullInsertedSliceOperandIntoVectorFunction
    : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (!call->hasAttr(mlir::hivm::VectorFunctionAttr::name))
      return failure();

    auto callee =
        mlir::utils::getCalledFunction<func::FuncOp, func::CallOp>(call);
    if (!callee)
      return failure();

    for (auto [argIdx, operand] : llvm::enumerate(call.getOperands())) {
      auto insertSlice = operand.getDefiningOp<tensor::InsertSliceOp>();
      if (!insertSlice)
        continue;

      auto patchSlice =
          insertSlice.getSource().getDefiningOp<tensor::ExtractSliceOp>();
      auto baseSlice =
          insertSlice.getDest().getDefiningOp<tensor::ExtractSliceOp>();
      if (!patchSlice || !baseSlice)
        continue;

      if (!isa<RankedTensorType>(patchSlice.getSource().getType()) ||
          !isa<RankedTensorType>(baseSlice.getSource().getType()) ||
          !isa<RankedTensorType>(patchSlice.getType()) ||
          !isa<RankedTensorType>(baseSlice.getType()))
        continue;

      modifyCallee(callee, argIdx, baseSlice, patchSlice, insertSlice,
                   rewriter);
      replaceCall(call, callee, argIdx, baseSlice, patchSlice, insertSlice,
                  rewriter);
      return success();
    }

    return failure();
  }

private:
  static void appendDynamicTypes(SmallVectorImpl<Type> &types, unsigned &pos,
                                 ValueRange values) {
    for (Value value : values)
      types.insert(types.begin() + pos++, value.getType());
  }

  static void appendDynamicSliceTypes(SmallVectorImpl<Type> &types,
                                      unsigned &pos,
                                      OffsetSizeAndStrideOpInterface op) {
    appendDynamicTypes(types, pos, op.getOffsets());
    appendDynamicTypes(types, pos, op.getSizes());
    appendDynamicTypes(types, pos, op.getStrides());
  }

  static SmallVector<Value> appendDynamicBlockArgs(Block &block, unsigned &pos,
                                                   ValueRange values) {
    SmallVector<Value> args;
    for (Value value : values)
      args.push_back(block.insertArgument(pos++, value.getType(),
                                          value.getLoc()));
    return args;
  }

  static void appendDynamicOperands(SmallVectorImpl<Value> &operands,
                                    unsigned &pos, ValueRange values) {
    operands.insert(operands.begin() + pos, values.begin(), values.end());
    pos += values.size();
  }

  static void appendDynamicSliceOperands(SmallVectorImpl<Value> &operands,
                                         unsigned &pos,
                                         OffsetSizeAndStrideOpInterface op) {
    appendDynamicOperands(operands, pos, op.getOffsets());
    appendDynamicOperands(operands, pos, op.getSizes());
    appendDynamicOperands(operands, pos, op.getStrides());
  }

  static void eraseIfUnused(Operation *op, PatternRewriter &rewriter) {
    if (op->use_empty())
      rewriter.eraseOp(op);
  }

  void modifyCallee(func::FuncOp callee, unsigned argIdx,
                    tensor::ExtractSliceOp baseSlice,
                    tensor::ExtractSliceOp patchSlice,
                    tensor::InsertSliceOp insertSlice,
                    PatternRewriter &rewriter) const {
    auto &block = callee.getBody().front();
    auto oldArg = block.getArgument(argIdx);
    auto oldFuncType = callee.getFunctionType();

    SmallVector<Type> newInputTypes(oldFuncType.getInputs().begin(),
                                    oldFuncType.getInputs().end());
    newInputTypes[argIdx] = baseSlice.getSource().getType();

    unsigned typePos = argIdx + 1;
    newInputTypes.insert(newInputTypes.begin() + typePos++,
                         patchSlice.getSource().getType());
    appendDynamicSliceTypes(newInputTypes, typePos, baseSlice);
    appendDynamicSliceTypes(newInputTypes, typePos, patchSlice);
    appendDynamicSliceTypes(newInputTypes, typePos, insertSlice);

    SmallVector<Type> newResultTypes(oldFuncType.getResults().begin(),
                                     oldFuncType.getResults().end());

    rewriter.modifyOpInPlace(callee, [&]() {
      callee.setFunctionType(
          rewriter.getFunctionType(newInputTypes, newResultTypes));

      unsigned insertPos = argIdx;
      auto baseSourceArg = block.insertArgument(
          insertPos++, baseSlice.getSource().getType(), oldArg.getLoc());
      auto patchSourceArg = block.insertArgument(
          insertPos++, patchSlice.getSource().getType(), oldArg.getLoc());

      SmallVector<Value> baseOffsets =
          appendDynamicBlockArgs(block, insertPos, baseSlice.getOffsets());
      SmallVector<Value> baseSizes =
          appendDynamicBlockArgs(block, insertPos, baseSlice.getSizes());
      SmallVector<Value> baseStrides =
          appendDynamicBlockArgs(block, insertPos, baseSlice.getStrides());
      SmallVector<Value> patchOffsets =
          appendDynamicBlockArgs(block, insertPos, patchSlice.getOffsets());
      SmallVector<Value> patchSizes =
          appendDynamicBlockArgs(block, insertPos, patchSlice.getSizes());
      SmallVector<Value> patchStrides =
          appendDynamicBlockArgs(block, insertPos, patchSlice.getStrides());
      SmallVector<Value> insertOffsets =
          appendDynamicBlockArgs(block, insertPos, insertSlice.getOffsets());
      SmallVector<Value> insertSizes =
          appendDynamicBlockArgs(block, insertPos, insertSlice.getSizes());
      SmallVector<Value> insertStrides =
          appendDynamicBlockArgs(block, insertPos, insertSlice.getStrides());

      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);

      auto newPatchSlice = rewriter.create<tensor::ExtractSliceOp>(
          oldArg.getLoc(), patchSlice.getType(), patchSourceArg, patchOffsets,
          patchSizes, patchStrides, patchSlice.getStaticOffsets(),
          patchSlice.getStaticSizes(), patchSlice.getStaticStrides());
      auto newBaseSlice = rewriter.create<tensor::ExtractSliceOp>(
          oldArg.getLoc(), baseSlice.getType(), baseSourceArg, baseOffsets,
          baseSizes, baseStrides, baseSlice.getStaticOffsets(),
          baseSlice.getStaticSizes(), baseSlice.getStaticStrides());
      auto merged = rewriter.create<tensor::InsertSliceOp>(
          oldArg.getLoc(), cast<RankedTensorType>(oldArg.getType()),
          newPatchSlice, newBaseSlice, insertOffsets, insertSizes,
          insertStrides, insertSlice.getStaticOffsets(),
          insertSlice.getStaticSizes(), insertSlice.getStaticStrides());

      rewriter.replaceAllUsesWith(oldArg, merged);
      block.eraseArgument(oldArg.getArgNumber());
    });
  }

  void replaceCall(func::CallOp call, func::FuncOp callee, unsigned argIdx,
                   tensor::ExtractSliceOp baseSlice,
                   tensor::ExtractSliceOp patchSlice,
                   tensor::InsertSliceOp insertSlice,
                   PatternRewriter &rewriter) const {
    SmallVector<Value> newOperands(call.getOperands().begin(),
                                   call.getOperands().end());
    newOperands[argIdx] = baseSlice.getSource();

    unsigned operandPos = argIdx + 1;
    newOperands.insert(newOperands.begin() + operandPos++,
                       patchSlice.getSource());
    appendDynamicSliceOperands(newOperands, operandPos, baseSlice);
    appendDynamicSliceOperands(newOperands, operandPos, patchSlice);
    appendDynamicSliceOperands(newOperands, operandPos, insertSlice);

    rewriter.setInsertionPoint(call);
    auto newCall =
        rewriter.create<func::CallOp>(call.getLoc(), callee, newOperands);
    newCall->setAttrs(call->getAttrs());
    rewriter.replaceOp(call, newCall.getResults());

    eraseIfUnused(insertSlice, rewriter);
    eraseIfUnused(baseSlice, rewriter);
    eraseIfUnused(patchSlice, rewriter);
  }
};

/// Swap a rank-preserving extract_slice followed by a unit-dim expand_shape
/// into expand_shape + extract_slice.  Only handles expand_shape that inserts
/// unit dims (size 1).
///
///   %s = extract_slice %src[2, 0] [4, 64] [1, 1]
///       : tensor<12x64xf16> to tensor<4x64xf16>
///   %e = expand_shape %s [[0], [1, 2]] output_shape [4, 1, 64]
///       : tensor<4x64xf16> into tensor<4x1x64xf16>
///
/// becomes:
///
///   %x = expand_shape %src [[0], [1, 2]] output_shape [12, 1, 64]
///       : tensor<12x64xf16> into tensor<12x1x64xf16>
///   %r = extract_slice %x[2, 0, 0] [4, 1, 64] [1, 1, 1]
///       : tensor<12x1x64xf16> to tensor<4x1x64xf16>
///
/// This exposes the extract_slice as the direct operand of a VF call,
/// allowing PullExtractInsertSliceIntoVectorFunction to pull it in.
struct SwapUnitDimExpandShapeOfExtractSlice
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto extractOp =
        expandOp.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();

    auto srcType =
        cast<RankedTensorType>(extractOp.getSource().getType());
    auto sliceType = extractOp.getType();
    auto expandedType = expandOp.getResultType();

    if (!srcType.hasStaticShape() || !expandedType.hasStaticShape())
      return failure();

    if (srcType.getRank() != sliceType.getRank())
      return failure();

    auto reassoc = expandOp.getReassociationIndices();

    // Build expanded source shape.  Verify expand only inserts unit dims.
    // Within each group, exactly one dim equals sliceDim (the "major" dim)
    // and the rest must be 1.  The major dim is mapped to srcDim.
    SmallVector<int64_t> expandedSrcShape;
    for (auto [groupIdx, group] : llvm::enumerate(reassoc)) {
      int64_t srcDim = srcType.getShape()[groupIdx];
      int64_t sliceDim = sliceType.getShape()[groupIdx];
      for (int64_t resIdx : group) {
        int64_t resDim = expandedType.getShape()[resIdx];
        if (resDim == 1) {
          expandedSrcShape.push_back(1);
        } else if (resDim == sliceDim) {
          expandedSrcShape.push_back(srcDim);
        } else {
          return failure();
        }
      }
    }

    auto expandedSrcType =
        RankedTensorType::get(expandedSrcShape, srcType.getElementType());

    SmallVector<OpFoldResult> expandedSrcOutputShape;
    for (int64_t dim : expandedSrcShape)
      expandedSrcOutputShape.push_back(rewriter.getIndexAttr(dim));

    auto newExpand = rewriter.create<tensor::ExpandShapeOp>(
        expandOp.getLoc(), expandedSrcType, extractOp.getSource(),
        reassoc, expandedSrcOutputShape);

    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    auto oldOffsets = extractOp.getMixedOffsets();
    auto oldStrides = extractOp.getMixedStrides();

    // Within each group, assign the original offset/stride to the non-unit
    // dim; unit dims get offset=0, stride=1.
    for (auto [groupIdx, group] : llvm::enumerate(reassoc)) {
      bool assignedOriginal = false;
      for (int64_t resIdx : group) {
        int64_t resDim = expandedType.getShape()[resIdx];
        newSizes.push_back(rewriter.getIndexAttr(resDim));
        if (resDim != 1 && !assignedOriginal) {
          newOffsets.push_back(oldOffsets[groupIdx]);
          newStrides.push_back(oldStrides[groupIdx]);
          assignedOriginal = true;
        } else {
          newOffsets.push_back(rewriter.getIndexAttr(0));
          newStrides.push_back(rewriter.getIndexAttr(1));
        }
      }
      // If all dims in the group are unit (size 1), assign to the last one.
      if (!assignedOriginal) {
        newOffsets.back() = oldOffsets[groupIdx];
        newStrides.back() = oldStrides[groupIdx];
      }
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        expandOp, expandedType, newExpand, newOffsets, newSizes, newStrides);
    return success();
  }
};

/// Fold a rank-reducing extract_slice followed by an expand_shape back to the
/// source rank into a single non-rank-reducing extract_slice.
///
///   %s = extract_slice %src[1, 0] [1, 64] [1, 1]
///       : tensor<3x64xf16> to tensor<64xf16>
///   %e = expand_shape %s [[0, 1]] : tensor<64xf16> into tensor<1x64xf16>
///
/// becomes:
///
///   %r = extract_slice %src[1, 0] [1, 64] [1, 1]
///       : tensor<3x64xf16> to tensor<1x64xf16>
struct FoldRankReducingSliceExpandShape
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto extractOp =
        expandOp.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();

    auto srcType =
        cast<RankedTensorType>(extractOp.getSource().getType());
    auto sliceType = extractOp.getType();
    auto resultType = expandOp.getResultType();

    // Only handle rank-reducing extract_slice (e.g. 3x64 → 64).
    if (srcType.getRank() <= sliceType.getRank())
      return failure();

    // The folded result must restore back to the source rank.
    if (srcType.getRank() != resultType.getRank())
      return failure();

    // The extract_slice sizes (in source rank) must match the expand result
    // shape exactly, so we can directly produce a non-rank-reducing
    // extract_slice with the expand result type.
    if (ArrayRef<int64_t>(extractOp.getStaticSizes()) !=
        resultType.getShape())
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        expandOp, resultType, extractOp.getSource(),
        extractOp.getMixedOffsets(), extractOp.getMixedSizes(),
        extractOp.getMixedStrides());
    return success();
  }
};

/// Swap a rank-preserving extract_slice followed by a collapse_shape into
/// collapse_shape + extract_slice.  The collapse may reduce rank by grouping
/// several source dims.  The extract_slice on the original tensor must satisfy:
/// within every collapse group, the slice must be expressible as a single
/// strided extract_slice on the collapsed view.
///
///   %s = extract_slice %src[0, o1, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1]
///       : tensor<1x16x4x16xf16> to tensor<1x1x4x16xf16>
///   %c = collapse_shape %s [[0, 1], [2, 3]]
///       : tensor<1x1x4x16xf16> into tensor<1x64xf16>
///
/// becomes:
///
///   %x = collapse_shape %src [[0, 1], [2, 3]]
///       : tensor<1x16x4x16xf16> into tensor<16x64xf16>
///   %r = extract_slice %x[o1, 0] [1, 64] [1, 1]
///       : tensor<16x64xf16> to tensor<1x64xf16>
///
/// This exposes the extract_slice as the direct operand of a VF call,
/// allowing PullExtractInsertSliceIntoVectorFunction to pull it in.
struct SwapCollapseShapeOfExtractSlice
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto extractOp =
        collapseOp.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp) {
      return failure();
    }

    auto collapsedType = collapseOp.getResultType();
    auto reassoc = collapseOp.getReassociationIndices();

    FailureOr<CollapsedSliceParams> params = computeCollapsedSliceParams(
        rewriter, collapseOp.getLoc(), extractOp, reassoc);
    if (failed(params)) {
      return failure();
    }

    auto newCollapse = rewriter.create<tensor::CollapseShapeOp>(
        collapseOp.getLoc(), params->collapsedSourceType,
        extractOp.getSource(), reassoc);

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        collapseOp, collapsedType, newCollapse, params->offsets, params->sizes,
        params->strides);
    return success();
  }

private:
  struct CollapsedSliceParams {
    RankedTensorType collapsedSourceType;
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
  };

  static int64_t product(ArrayRef<int64_t> shape,
                         ArrayRef<int64_t> dims) {
    int64_t result = 1;
    for (int64_t dim : dims) {
      result *= shape[dim];
    }
    return result;
  }

  // A group is representable after collapse when its active slice range maps to
  // one strided interval in the collapsed dimension.
  static FailureOr<int64_t>
  getCollapsedSliceStride(ArrayRef<int64_t> sourceShape,
                          ArrayRef<int64_t> sliceShape,
                          ArrayRef<int64_t> group) {
    std::optional<unsigned> firstActivePos;
    std::optional<unsigned> lastActivePos;
    for (auto [pos, dim] : llvm::enumerate(group)) {
      if (sliceShape[dim] <= 1) {
        continue;
      }
      firstActivePos = firstActivePos.value_or(pos);
      lastActivePos = pos;
    }

    if (!firstActivePos) {
      return int64_t{1};
    }

    for (unsigned pos = *firstActivePos + 1; pos <= *lastActivePos; ++pos) {
      int64_t dim = group[pos];
      if (sliceShape[dim] != sourceShape[dim]) {
        return failure();
      }
    }

    return product(sourceShape,
                   group.slice(*lastActivePos + 1,
                               group.size() - *lastActivePos - 1));
  }

  static OpFoldResult linearizeOffsets(OpBuilder &builder, Location loc,
                                       ArrayRef<int64_t> sourceShape,
                                       ArrayRef<int64_t> group,
                                       ArrayRef<OpFoldResult> offsets) {
    SmallVector<OpFoldResult> indices;
    SmallVector<OpFoldResult> basis;
    for (int64_t dim : group) {
      indices.push_back(offsets[dim]);
      basis.push_back(builder.getIndexAttr(sourceShape[dim]));
    }

    // Reuse upstream affine linearization for row-major offset computation.
    ImplicitLocOpBuilder locBuilder(loc, builder);
    return affine::linearizeIndex(indices, basis, locBuilder);
  }

  static FailureOr<CollapsedSliceParams>
  computeCollapsedSliceParams(OpBuilder &builder, Location loc,
                              tensor::ExtractSliceOp extractOp,
                              ArrayRef<ReassociationIndices> reassociation) {
    auto sourceType =
        dyn_cast<RankedTensorType>(extractOp.getSource().getType());
    auto sliceType = dyn_cast<RankedTensorType>(extractOp.getType());
    if (!sourceType || !sliceType) {
      return failure();
    }

    if (!sourceType.hasStaticShape() || !sliceType.hasStaticShape()) {
      return failure();
    }

    if (sourceType.getRank() != sliceType.getRank()) {
      return failure();
    }

    if (!llvm::all_of(extractOp.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; })) {
      return failure();
    }

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    ArrayRef<int64_t> sliceShape = sliceType.getShape();
    SmallVector<OpFoldResult> originalOffsets = extractOp.getMixedOffsets();

    CollapsedSliceParams params;
    SmallVector<int64_t> collapsedSourceShape;
    for (ArrayRef<int64_t> group : reassociation) {
      FailureOr<int64_t> collapsedStride =
          getCollapsedSliceStride(sourceShape, sliceShape, group);
      if (failed(collapsedStride)) {
        return failure();
      }

      collapsedSourceShape.push_back(product(sourceShape, group));
      params.offsets.push_back(
          linearizeOffsets(builder, loc, sourceShape, group, originalOffsets));
      params.sizes.push_back(builder.getIndexAttr(product(sliceShape, group)));
      params.strides.push_back(builder.getIndexAttr(*collapsedStride));
    }

    params.collapsedSourceType =
        RankedTensorType::get(collapsedSourceShape, sourceType.getElementType());
    return params;
  }
};

/// Pre-process extract_slice + expand_shape patterns generated by the flatten
/// unit-dims pass.  These patterns block PullSlice from matching the
/// extract_slice because expand_shape sits between it and the VF call operand.
/// FoldRankReducingSliceExpandShape handles rank-restore cases (srcRank ==
/// resRank); SwapUnitDimExpandShapeOfExtractSlice handles rank-increase cases
/// (srcRank < resRank).
// TODO: Fix the root cause in the flatten pass so that these pre-process
// patterns are not needed.
static void preProcessShapeChangeAfterExtractSlice(MLIRContext *ctx,
                                                   Operation *op) {
  RewritePatternSet patterns(ctx);
  patterns.add<FoldRankReducingSliceExpandShape,
               SwapUnitDimExpandShapeOfExtractSlice,
               SwapCollapseShapeOfExtractSlice>(ctx);
  (void)applyPatternsGreedily(op, std::move(patterns));
}

struct PullSliceIntoVectorFunction
    : public impl::PullSliceIntoVectorFunctionBase<
          PullSliceIntoVectorFunction> {
  void runOnOperation() override {
    preProcessShapeChangeAfterExtractSlice(&getContext(), getOperation());

    RewritePatternSet patterns(&getContext());
    patterns.add<PullExtractInsertSliceIntoVectorFunction,
                 PullInsertedSliceOperandIntoVectorFunction>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hfusion::createPullSliceIntoVectorFunctionPass() {
  return std::make_unique<PullSliceIntoVectorFunction>();
}
