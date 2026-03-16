//===- FoldUnitDims.cpp - Fold unit dims ops on memrefs & tensors ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-fold-unit-dims"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_HFUSIONFOLDUNITDIMS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct HFusionFoldUnitDimsPass
    : public impl::HFusionFoldUnitDimsBase<HFusionFoldUnitDimsPass> {
public:
  void runOnOperation() override;
};
} // anonymous namespace

template <typename PatternT, typename OpTy>
class FilteredPattern : public OpRewritePattern<OpTy> {
public:
  FilteredPattern(MLIRContext *context,
                  std::function<bool(Operation *)> shouldSkip)
      : OpRewritePattern<OpTy>(context), innerPattern(context),
        shouldSkip(shouldSkip) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (shouldSkip(op)) {
      return failure();
    }

    return innerPattern.matchAndRewrite(op, rewriter);
  }

private:
  PatternT innerPattern;
  std::function<bool(Operation *)> shouldSkip;
};

/**
 * @brief Recursively check if value is defined by a chain of arith::MulIOp
 * using only input values
 * @details Verify if `out` is either in `ins` (base case) or defined by
 * arith::MulIOp whose operands are all in/derived from `ins`; removes matched
 * input values from `ins` (modifies in-place)
 *
 * Matching Logic Examples:
 * - Case 1 (Direct Input Match):
 *   ins = [%a, %b], out = %a → returns true; ins becomes [%b] (erase %a)
 *
 * - Case 2 (Mul Chain Match):
 *   %mul1 = arith.muli %a, %b : index
 *   %mul2 = arith.muli %mul1, %c : index
 *   ins = [%a, %b, %c], out = %mul2 → returns true; ins becomes [] (all erased)
 *
 * - Case 3 (Non-Mul Chain):
 *   %add = arith.addi %a, %b : index
 *   ins = [%a, %b], out = %add → returns false; ins remains [%a, %b]
 *
 * - Case 4 (Partial Mul Chain):
 *   %mul = arith.muli %a, %d : index
 *   ins = [%a, %b], out = %mul → returns false (%d not in ins); ins remains
 * [%a, %b]
 *
 * @param[in] out Target value to check (must be index type for arith::MulIOp)
 * @param[in,out] ins List of input values to match against (modified in-place:
 * matched values are erased)
 * @return bool True if `out` is input value or mul chain of input values; false
 * otherwise
 * @note 1. Recursive check for nested arith::MulIOp (AND logic for lhs/rhs
 * operands)
 *       2. Modifies input `ins` vector (erases values used in the mul chain)
 *       3. Returns false for non-arith::MulIOp defining ops (e.g., add/sub)
 */
bool isDefinedByMulChainUsingInputs(Value out, SmallVector<Value> &ins) {
  auto it = std::find(ins.begin(), ins.end(), out);
  if (it != ins.end()) {
    ins.erase(it);
    return true;
  }

  if (auto mulIOp = out.getDefiningOp<arith::MulIOp>()) {
    return isDefinedByMulChainUsingInputs(mulIOp.getLhs(), ins) &&
           isDefinedByMulChainUsingInputs(mulIOp.getRhs(), ins);
  }

  return false;
}

/**
 * @brief Check if output OFRs have product relation with input OFRs via
 * reassociation indices
 * @details Verify if any output Value (from OFR) is defined by a mul chain of
 * input Values (mapped via reassoc indices); returns true if a full match (mul
 * chain uses all mapped inputs) is found
 *
 * Matching Logic Flow:
 * 1. Iterate over reassociation index groups (reassocIdxVec)
 * 2. Map output OFR (outs[i]) to Value (skip non-Value OFRs like constants)
 * 3. Map input OFRs (ins[j] for j in reassoc[i]) to Values (skip constants)
 * 4. Check if output Value is mul chain of mapped input Values (via
 * isDefinedByMulChainUsingInputs)
 * 5. Return true if ANY group has full match (mul chain uses all mapped inputs
 * → insVal empty)
 *
 * Example (Full Match):
 * - reassocIdxVec = [[0,1], [2]]
 * - outs = [%mul_ab, 42] (outs[0] = %mul_ab (Value), outs[1] = 42 (constant))
 * - ins = [%a, %b, %c] (all Values)
 * - Step 2: outVal = %mul_ab (outs[0] is Value)
 * - Step 3: insVal = [%a, %b] (reassoc[0] = [0,1] → ins[0], ins[1])
 * - Step 4: isDefinedByMulChainUsingInputs(%mul_ab, [%a,%b]) → true (insVal
 * empty)
 * - Step 5: Return true
 *
 * Example (No Match):
 * - Same setup as above, but %mul_ab = arith.muli %a, %d : index (%d not in
 * ins)
 * - Step 4: returns false (insVal = [%a,%b] not empty) → final return false
 *
 * @param[in] outs List of output OpFoldResults (constants skipped, only Values
 * checked)
 * @param[in] ins List of input OpFoldResults (constants skipped, only Values
 * mapped)
 * @param[in] reassocIdxVec Nested reassociation indices mapping ins → outs
 * @return bool True if any output Value is full mul chain of mapped input
 * Values; false otherwise
 * @note 1. Skips non-Value OpFoldResults (constants/attrs) for both outs/ins
 *       2. Relies on isDefinedByMulChainUsingInputs (recursive mul chain check
 * + in-place insVal modification)
 *       3. Returns true on FIRST full match (no further iteration)
 *       4. Returns false if no full matches (or all outs/ins are constants)
 */
bool checkProductRelation(
    SmallVectorImpl<OpFoldResult> &outs, SmallVectorImpl<OpFoldResult> &ins,
    SmallVectorImpl<ReassociationIndices> &reassocIdxVec) {
  for (auto [i, reassoc] : llvm::enumerate(reassocIdxVec)) {
    auto out = outs[i];
    auto outVal = llvm::dyn_cast_if_present<Value>(out);
    if (!outVal) {
      // static size does not matter
      continue;
    }

    SmallVector<Value> insVal;
    for (auto j : reassoc) {
      auto inVal = llvm::dyn_cast_if_present<Value>(ins[j]);
      if (!inVal) {
        continue;
      }
      insVal.push_back(inVal);
    }

    bool isMatched = isDefinedByMulChainUsingInputs(outVal, insVal);
    if (isMatched && insVal.empty()) {
      return true;
    }
  }

  return false;
}

/**
 * @brief Check if value matches ExpandType→ExtractType pattern with product
 * size relation
 * @details Verify if value is defined by ExpandType op whose source is
 * ExtractType op, and source sizes have product relation with expand sizes (via
 * reassociation indices); returns source of ExtractType op if full match, else
 * nullopt
 *
 * Template Specialization Support:
 * - ExpandType: memref::ExpandShapeOp / tensor::ExpandShapeOp
 * - ExtractType: memref::SubViewOp / tensor::ExtractSliceOp
 *
 * Matching Logic Flow:
 * 1. Check if val is defined by ExpandType op (return nullopt if not)
 * 2. Get expand op sizes (valSizes: mixed static/dynamic via getMixedValues)
 * 3. Get expand op source (srcOfVal) and check if it's defined by ExtractType
 * op (return nullopt if not)
 * 4. Get extract op sizes (srcOfValSizes: mixed static/dynamic)
 * 5. Get expand op reassociation indices (reassocIdxVec)
 * 6. Check product relation (srcOfValSizes → valSizes via reassoc) via
 * checkProductRelation
 * 7. Return srcOfVal if matched, else nullopt
 *
 * Example (MemRef Match):
 * - val = %expand (memref::ExpandShapeOp, output shape [2x4],
 * reassocIdxVec=[[0], [1]])
 * - srcOfVal = %subview (memref::SubViewOp, sizes [%a, %b])
 * - valSizes = [%mul_ab, 4] (outs), srcOfValSizes = [%a, %b] (ins)
 * - checkProductRelation → true (%mul_ab = arith.muli %a, %b)
 * - Return: %subview (srcOfVal)
 *
 * Example (No Match):
 * - Same setup but %mul_ab = arith.muli %a, %c (%c not in srcOfValSizes)
 * - checkProductRelation → false → Return: std::nullopt
 *
 * @tparam ExpandType Target expand op type (memref::ExpandShapeOp /
 * tensor::ExpandShapeOp)
 * @tparam ExtractType Target extract op type (memref::SubViewOp /
 * tensor::ExtractSliceOp)
 * @param[in] val Value to check against Expand→Extract pattern
 * @param[in,out] rewriter PatternRewriter for mixed size value generation
 * (getMixedValues)
 * @return std::optional<Value> Source of ExtractType op if pattern matches;
 * nullopt otherwise
 * @note 1. Relies on getMixedValues for static/dynamic size resolution
 *       2. Relies on checkProductRelation for size product relation validation
 *       3. Returns nullopt if any step fails (non-Expand/Extract op, no product
 * relation)
 *       4. Supports both MemRef and Tensor dialect ops via template
 * specialization
 */
template <typename ExpandType, typename ExtractType>
std::optional<Value> checkValueMatchPattern(Value val,
                                            mlir::PatternRewriter &rewriter) {
  auto valDefOp = val.getDefiningOp<ExpandType>();
  if (!valDefOp)
    return std::nullopt;
  auto valSizes = getMixedValues(valDefOp.getStaticOutputShape(),
                                 valDefOp.getOutputShape(), rewriter);
  auto srcOfVal = valDefOp.getSrc();
  auto srcOfValDefOp = srcOfVal.template getDefiningOp<ExtractType>();
  if (!srcOfValDefOp)
    return std::nullopt;
  auto srcOfValSizes = srcOfValDefOp.getMixedSizes();
  auto reassocIdxVec = valDefOp.getReassociationIndices();
  bool isMatched = checkProductRelation(srcOfValSizes, valSizes, reassocIdxVec);
  if (isMatched) {
    return srcOfVal;
  } else {
    return std::nullopt;
  }
}

// clang-format off
/**
 * @brief Collapse unit dimensions from MemRef/Tensor value via CollapseShapeOp
 * @details Removes unit-sized dimensions from shaped value
 * (MemRef/RankedTensor) by creating CollapseShapeOp; returns failure if no unit
 * dims exist
 *
 * IR Transformation (MemRef Example):
 * - Before: %0 : memref<2x1x3xf32> (unit dim at index 1)
 * - After:  %1 = memref.collapse_shape %0 [[0], [1, 2]] : memref<2x1x3xf32>
 * to memref<2x3xf32>
 *
 * IR Transformation (Tensor Example):
 * - Before: %0 : tensor<4x1x5xf32>
 * - After:  %1 = tensor.collapse_shape %0 [[0], [1, 2]] : tensor<4x1x5xf32>
 * to tensor<4x5xf32>
 *
 * @note Only applies to:
 *       1. ShapedType with rank > 1
 *       2. MemRefType or RankedTensorType (aborts for other types via
 * llvm_unreachable)
 *       3. Shape containing at least one unit dimension
 *
 * @param[in] oldVal Input MemRef/Tensor value to collapse
 * @param[in,out] builder OpBuilder for creating CollapseShapeOp
 * @param[in] loc Location for the new operation
 * @return FailureOr<Value> Collapsed value on success, failure otherwise
 * @retval success(Value) Collapsed MemRef/Tensor value
 * @retval failure() No unit dimensions to collapse (or rank <= 1)
 * @warning Aborts if input type is not MemRefType/RankedTensorType
 */
// clang-format on
FailureOr<Value> tryCollapseValue(Value oldVal, OpBuilder &builder,
                                  Location loc) {
  auto oldTy = dyn_cast<ShapedType>(oldVal.getType());
  if (!oldTy) {
    return failure();
  }
  if (oldTy.getRank() <= 1) {
    return failure();
  }
  Value newVal;
  bool isModified = false;
  auto oldShape = utils::getShape(oldTy);
  auto squeezedShape = reshape_utils::getSqueezedShape(oldShape);
  if (squeezedShape != oldShape) {
    auto maybeRankReducedUnusedDims = computeRankReductionMask(
        oldShape, squeezedShape, /*matchDynamic=*/true);
    if (!maybeRankReducedUnusedDims) {
      LDBG("oldVal = " << oldVal << " fails to compute rank-reduced mask");
      return failure();
    }
    // It is possible that both dynamic shape and static shape have unit dim.
    // (dynamic shape can be implicitly unit dim if it is defined by
    // arith.minsi against 1.)
    // Currently we let tryCollapseValue work for the situation when all unit
    // dims are static. Let the pattern matcher in the CopyOp rewriter work
    // for the situation where dynamic unit dims and static unit dims co-exist.
    // Thus here we check if dynamic unit dim exist. Since reaching here means
    // static unit dim exists, we would return failure if dynamic unit dim is
    // found.
    bool dynShapeHasOne = false;
    auto sizeInterface = oldVal.getDefiningOp<OffsetSizeAndStrideOpInterface>();
    if (sizeInterface) {
      for (size_t i = 0; i < oldShape.size(); ++i) {
        if (!sizeInterface.isDynamicSize(i)) {
          continue;
        }
        auto dynShape = sizeInterface.getDynamicSize(i);
        bool isMinAgainstOne = false;
        if (auto defOp = dynShape.getDefiningOp<arith::MinSIOp>()) {
          isMinAgainstOne = reshape_utils::isConstIntOne(defOp.getLhs()) ||
                            reshape_utils::isConstIntOne(defOp.getRhs());
        }
        if (isMinAgainstOne) {
          dynShapeHasOne = true;
          break;
        }
      }
    }
    if (dynShapeHasOne) {
      LDBG("[tryCollapseValue] both dynamic shape and static shape "
           "have unit dims. This must be rewriting CopyOp. "
           "Do not collapse the source. Let the pattern matcher work.");
      // reaching here means it has already found static shape 1.
      return failure();
    }
    auto unusedDims = maybeRankReducedUnusedDims.value();
    SmallVector<int64_t> expandDims;
    expandDims.reserve(unusedDims.size());
    for (unsigned v : unusedDims) {
      expandDims.push_back(static_cast<int64_t>(v));
    }
    // clang-format off
    LLVM_DEBUG(
      llvm::dbgs() << "[tryCollapseValue] RankReducedUnusedDims are [";
      for (size_t i = 0; i < expandDims.size() - 1; ++i) {
        llvm::dbgs() << expandDims[i] << ", ";
      }
      llvm::dbgs() << expandDims[expandDims.size() - 1] << "]\n";
    );
    // clang-format on
    auto reassocIdxVec =
        reshape_utils::getReAssociation(expandDims, oldShape.size());
    LLVM_DEBUG(llvm::dbgs() << "[tryCollapseValue] reassocIdxVec is\n";
               utils::dumpReassociationIndicesVector(reassocIdxVec););
    if (auto oldMemrefTy = dyn_cast<MemRefType>(oldTy)) {
      // Why not using memref::CollapseShapeOp::isGuaranteedCollapsible?
      // Because it is strict when dealing with the dynamic shape.
      // We do not use such strict rule to collapse.
      // In fact, memref<1x?xf32, strided<[5, 1], offset: ?>> could be collapsed
      // to memref<?xf32, strided<[1], offset: ?>>.
      newVal =
          builder.create<memref::CollapseShapeOp>(loc, oldVal, reassocIdxVec);
    } else if (auto oldTensorTy = dyn_cast<RankedTensorType>(oldTy)) {
      auto newTy = tensor::CollapseShapeOp::inferCollapsedType(oldTensorTy,
                                                               reassocIdxVec);
      newVal = builder.create<tensor::CollapseShapeOp>(loc, newTy, oldVal,
                                                       reassocIdxVec);
    } else {
      llvm_unreachable("[tryCollapseValue] "
                       "type must be MemRefType or RankedTensorType");
    }
    isModified = true;
  }
  if (!isModified) {
    LDBG("Give up collapsing with IR not modified");
    return failure();
  }
  return newVal;
}

// clang-format off
/**
 * @brief Expand shaped value with unit dimensions via ExpandShapeOp
 * @details Reintroduces unit dimensions to MemRef/Tensor value by creating
 * ExpandShapeOp to match the target ShapedType (with unit dims)
 *
 * IR Transformation (MemRef Example):
 * - Before: %0 : memref<2x3xf32>
 * - After:  %1 = memref.expand_shape %0 [[0], [1, 2]] : memref<2x3xf32> to
 * memref<2x1x3xf32>
 *
 * IR Transformation (Tensor Example):
 * - Before: %0 : tensor<4x5xf32>
 * - After:  %1 = tensor.expand_shape %0 [[0], [1, 2]] : tensor<4x5xf32> to
 * tensor<4x1x5xf32>
 *
 * @note Only supports MemRefType/RankedTensorType (aborts for other types via
 * llvm_unreachable)
 * @param[in] oldVal Input MemRef/Tensor value to expand
 * @param[in] newTy Target ShapedType (with unit dimensions to expand to)
 * @param[in,out] builder OpBuilder for creating ExpandShapeOp
 * @param[in] loc Location for the new operation
 * @return FailureOr<Value> Expanded value on success (never returns failure)
 * @retval success(Value) Expanded MemRef/Tensor value matching newTy
 * @warning Aborts if newTy is not MemRefType/RankedTensorType
 */
// clang-format on
std::optional<Value>
tryExpandValueWithUnitDim(Value oldVal, Value refVal,
                          SmallVector<ReassociationIndices> &reassocIdxVec,
                          OpBuilder &builder, Location loc) {
  auto refTy = dyn_cast<ShapedType>(refVal.getType());
  if (!refTy) {
    return std::nullopt;
  }
  bool reassocIdxVecIsGenerated = false;
  auto newShape = refTy.getShape();
  if (reassocIdxVec.empty()) {
    reassocIdxVecIsGenerated = true;
    auto oldTy = cast<ShapedType>(oldVal.getType());
    auto oldShape = oldTy.getShape();
    auto maybeRankReducedUnusedDims =
        computeRankReductionMask(newShape, oldShape, /*matchDynamic=*/true);
    if (!maybeRankReducedUnusedDims) {
      return std::nullopt;
    }
    auto unusedDims = maybeRankReducedUnusedDims.value();
    SmallVector<int64_t> expandDims;
    expandDims.reserve(unusedDims.size());
    for (unsigned v : unusedDims) {
      expandDims.push_back(static_cast<int64_t>(v));
    }
    // clang-format off
    LLVM_DEBUG(
      llvm::dbgs() << "[tryExpandValueWithUnitDim] RankReducedUnusedDims are [";
      for (size_t i = 0; i < expandDims.size() - 1; ++i) {
        llvm::dbgs() << expandDims[i] << ", ";
      }
      llvm::dbgs() << expandDims[expandDims.size() - 1] << "]\n";
    );
    // clang-format on
    reassocIdxVec =
        reshape_utils::getReAssociation(expandDims, newShape.size());
    LLVM_DEBUG(llvm::dbgs() << "[tryExpandValueWithUnitDim] reassocIdxVec is\n";
               utils::dumpReassociationIndicesVector(reassocIdxVec););
  }

  Operation *expandOp;
  if (auto newMemrefTy = dyn_cast<MemRefType>(refTy)) {
    if (reassocIdxVecIsGenerated) {
      expandOp = builder.create<memref::ExpandShapeOp>(loc, newShape, oldVal,
                                                       reassocIdxVec);
    } else {
      auto outputShape = memref::getMixedSizes(builder, loc, refVal);
      expandOp = builder.create<memref::ExpandShapeOp>(
          loc, newShape, oldVal, reassocIdxVec, outputShape);
    }
  } else if (auto newTensorTy = dyn_cast<RankedTensorType>(refTy)) {
    if (reassocIdxVecIsGenerated) {
      expandOp = builder.create<tensor::ExpandShapeOp>(loc, refTy, oldVal,
                                                       reassocIdxVec);
    } else {
      auto outputShape = tensor::getMixedSizes(builder, loc, refVal);
      expandOp = builder.create<tensor::ExpandShapeOp>(
          loc, refTy, oldVal, reassocIdxVec, outputShape);
    }
  } else {
    llvm_unreachable("[tryExpandValueWithUnitDim] "
                     "type must be MemRefType or RankedTensorType");
  }
  Value newVal = expandOp->getResult(0);
  return newVal;
}

/**
 * @brief Get squeezed layout (offset/size/stride) and reassociation indices for
 * shaped ops
 * @details Extract non-unit-dim layout info (offsets/sizes/strides) from
 * OffsetSizeAndStrideOpInterface ops, and generate reassociation indices for
 * shape expansion/collapse
 *
 * Layout + Reassociation Transformation Example:
 * - Input (origShape): [2, 1, 3] (unit dim at index 1)
 * - SqueezedShape: [2, 3] (unit dim removed)
 * - UnusedDims: {1} (index of unit dim)
 * - Layout Transformation:
 *   OldOffsets:  [o0, o1, o2] → NewOffsets: [o0, o2] (drop index 1)
 *   OldSizes:    [s0, s1, s2] → NewSizes:   [s0, s2] (drop index 1)
 *   OldStrides:  [st0, st1, st2] → NewStrides: [st0, st2] (drop index 1)
 * - Reassociation Indices: getReAssociation([1], 3) → [[0], [1,2]]
 *
 * @param[in] op Target operation (must implement
 * OffsetSizeAndStrideOpInterface)
 * @return std::optional<tuple<newOffsets, newSizes, newStrides, reassocIdxVec>>
 *         - Non-nullopt: Valid squeezed layout + reassociation indices (if op
 * matches requirements)
 *         - nullopt: Op does not implement interface / rank reduction mask
 * computation fails
 *
 * @note 1. UnusedDims = indices of unit dimensions in original shape (from
 * computeRankReductionMask)
 *       2. New layout vectors exclude entries at unusedDims indices
 *       3. Reassociation indices follow getReAssociation() rules (expandDims =
 * unusedDims)
 *       4. Supports dynamic shapes (matchDynamic=true in
 * computeRankReductionMask)
 */
std::optional<
    std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
               SmallVector<OpFoldResult>, SmallVector<ReassociationIndices>>>
getSqueezedLayoutReassoc(Operation *op, SmallVectorImpl<int64_t> &fullShape,
                         OpBuilder &builder, Location loc) {
  auto sizeInterface = dyn_cast_or_null<OffsetSizeAndStrideOpInterface>(op);
  if (!sizeInterface) {
    return std::nullopt;
  }

  auto squeezedShape = reshape_utils::getSqueezedShape(fullShape);
  auto maybeRankReducedUnusedDims =
      computeRankReductionMask(fullShape, squeezedShape, /*matchDynamic=*/true);
  if (!maybeRankReducedUnusedDims) {
    return std::nullopt;
  }
  auto unusedDims = maybeRankReducedUnusedDims.value();

  SmallVector<int64_t> expandDims;
  expandDims.reserve(unusedDims.size());
  for (unsigned v : unusedDims) {
    expandDims.push_back(static_cast<int64_t>(v));
  }
  auto reassocIdxVec =
      reshape_utils::getReAssociation(expandDims, fullShape.size());

  auto oldOffsets = sizeInterface.getMixedOffsets();
  auto oldSizes = sizeInterface.getMixedSizes();
  auto oldStrides = sizeInterface.getMixedStrides();
  SmallVector<OpFoldResult> newOffsets;
  SmallVector<OpFoldResult> newSizes;
  SmallVector<OpFoldResult> newStrides;

  for (auto &reassoc : reassocIdxVec) {
    OpFoldResult prod = oldSizes[reassoc[0]];
    for (size_t j = 1; j < reassoc.size(); ++j) {
      prod = reshape_utils::mulOFRs(prod, oldSizes[reassoc[j]], builder, loc);
    }
    newSizes.push_back(prod);
  }
  // In triton, no dynamic stride would exist. Thus do not need to take that
  // into account.
  if (isa<memref::ReinterpretCastOp>(op)) {
    // memref::ReinterpretCastOp has single offset
    newOffsets.push_back(oldOffsets[0]);
    for (auto [idx, stride] : llvm::enumerate(oldStrides)) {
      if (!unusedDims.contains(idx)) {
        newStrides.push_back(stride);
      }
    }
  } else {
    for (auto [idx, stride] : llvm::enumerate(oldStrides)) {
      if (!unusedDims.contains(idx)) {
        newOffsets.push_back(oldOffsets[idx]);
        newStrides.push_back(stride);
      }
    }
  }
  // clang-format off
  LLVM_DEBUG(
    llvm::dbgs() << "[getSqueezedLayoutReassoc] RankReducedUnusedDims are [";
    for (size_t i = 0; i < expandDims.size() - 1; ++i) {
      llvm::dbgs() << expandDims[i] << ", ";
    }
    llvm::dbgs() << expandDims[expandDims.size() - 1] << "]\n";
    llvm::dbgs() << "[getSqueezedLayoutReassoc] reassocIdxVec is\n";
    utils::dumpReassociationIndicesVector(reassocIdxVec);
    auto dumpLayout =
      [&](SmallVectorImpl<OpFoldResult> &vec, const std::string vecName) {
        llvm::dbgs() << "[getSqueezedLayoutReassoc] " << vecName << " is\n";
        for (size_t i = 0; i < vec.size() - 1; ++i) {
          llvm::dbgs() << "[" << i << "]: " << vec[i] << "\n";
        }
        llvm::dbgs() << "[" << vec.size() - 1 << "]: " << vec.back() << "\n";
      };
    dumpLayout(newOffsets, "newOffsets");
    dumpLayout(newSizes, "newSizes");
    dumpLayout(newStrides, "newStrides");
  );
  // clang-format on
  return std::make_tuple(newOffsets, newSizes, newStrides, reassocIdxVec);
}

/**
 * @brief Rewrite pattern to drop unit dimensions from memref::AllocOp's shape
 * @details Removes unit-sized dimensions from the alloc'ed memref shape and
 * re-expand to original type
 *
 * IR Transformation:
 * - Before: alloc() : memref<2x1x3xf32> (shape with unit dim at index 1)
 * - After:  alloc() : memref<2x3xf32> → expand to memref<2x1x3xf32> (via unit
 * dim expansion)
 *
 * @note Only applies to:
 *       1. AllocOp with memref rank > 1
 *       2. MemRefType with identity layout
 *       3. Shape containing at least one unit dimension
 *
 * @warning Aborts if tryExpandValueWithUnitDim fails (llvm_unreachable)
 */
struct DropUnitDimsAllocPattern : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto loc = allocOp.getLoc();
    auto dst = allocOp.getMemref();
    auto dstTy = cast<MemRefType>(dst.getType());
    if (dstTy.getRank() <= 1) {
      return rewriter.notifyMatchFailure(
          allocOp, "[DropUnitDimsAllocPattern] AllocOp rank <= 1");
    }
    if (!dstTy.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(
          allocOp, "[DropUnitDimsAllocPattern] supports only identity layout");
    }

    auto dstShape = utils::getShape(dstTy);
    auto squeezedDstShape = reshape_utils::getSqueezedShape(dstShape);
    if (squeezedDstShape == dstShape) {
      return rewriter.notifyMatchFailure(
          allocOp,
          "[DropUnitDimsAllocPattern] shape does not contain unit dim");
    }

    auto squeezedDstTy =
        MemRefType::get(squeezedDstShape, dstTy.getElementType());
    auto newAllocOp = rewriter.create<memref::AllocOp>(loc, squeezedDstTy);
    // pass empty reassocIdxVec to create reassociations inside the func
    SmallVector<ReassociationIndices> reassocIdxVec;
    auto maybeNewDst = tryExpandValueWithUnitDim(newAllocOp.getMemref(), dst,
                                                 reassocIdxVec, rewriter, loc);
    if (maybeNewDst.has_value()) {
      rewriter.replaceAllUsesWith(dst, maybeNewDst.value());
    } else {
      llvm_unreachable(
          "[DropUnitDimsAllocPattern] fails in tryExpandValueWithUnitDim");
    }

    return success();
  }
};

struct CanonicalizeBufMIDPattern
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
public:
  using OpRewritePattern<
      bufferization::MaterializeInDestinationOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(bufferization::MaterializeInDestinationOp copyOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto loc = copyOp.getLoc();
    auto dst = copyOp.getDest();
    auto dstTy = cast<ShapedType>(dst.getType());
    if (dstTy.getRank() <= 1) {
      return rewriter.notifyMatchFailure(
          copyOp, "[CanonicalizeBufMIDPattern] MaterializeInDestinationOp's "
                  "dst rank <= 1");
    }

    auto *dstDefOp = dst.getDefiningOp();
    if (!dstDefOp || !isa<memref::CastOp>(dstDefOp)) {
      return rewriter.notifyMatchFailure(
          copyOp, "[CanonicalizeBufMIDPattern] MaterializeInDestinationOp's "
                  "dst is not defined by memref.cast");
    }
    auto castOp = cast<memref::CastOp>(dstDefOp);
    auto castSrc = castOp.getSource();
    auto castSrcTy = cast<ShapedType>(castSrc.getType());
    if (!castSrcTy.hasStaticShape()){
      return rewriter.notifyMatchFailure(
          copyOp, "[CanonicalizeBufMIDPattern] MaterializeInDestinationOp's "
                  "dst is not defined by memref.cast with static shape");
    }
    auto numElem = castSrcTy.getNumElements();
    if (numElem != 1) {
      return rewriter.notifyMatchFailure(
          copyOp, "[CanonicalizeBufMIDPattern] MaterializeInDestinationOp's "
                  "dst has more than one elements");
    }

    auto src = copyOp.getSource();
    auto srcTy = cast<TensorType>(src.getType());
    auto srcShape = utils::getShape(srcTy);
    auto castSrcShape = utils::getShape(castSrcTy);
    if (castSrcShape != srcShape) {
      return rewriter.notifyMatchFailure(
          copyOp, "[CanonicalizeBufMIDPattern] MaterializeInDestinationOp's "
                  "src is not the same shape as that of memref.cast's src");
    }

    auto newCopyOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, src, castSrc);
    newCopyOp.setRestrictAttr(copyOp.getRestrictAttr());
    newCopyOp.setWritableAttr(copyOp.getWritableAttr());
    rewriter.replaceOp(copyOp, newCopyOp);

    return success();
  }
};

// clang-format off
/**
 * @brief Rewrite pattern to drop unit dimensions from bufferization::ToTensorOp
 * @details Collapse unit dims of input memref → create new ToTensorOp → expand
 * unit dims to original tensor shape
 *
 * IR Transformation Flow:
 * - Before:
 *   %memref = memref.alloc() : memref<2x1x3xf32>
 *   %tensor = bufferization.to_tensor %memref : memref<2x1x3xf32> to
 * tensor<2x1x3xf32>
 *
 * - Step 1 (Collapse):
 *   %collapsed_memref = memref.collapse_shape %memref [[0], [1, 2]] :
 * memref<2x1x3xf32> to memref<2x3xf32>
 *
 * - Step 2 (New ToTensorOp):
 *   %collapsed_tensor = bufferization.to_tensor %collapsed_memref :
 * memref<2x3xf32> to tensor<2x3xf32>
 *
 * - Step 3 (Expand):
 *   %expanded_tensor = tensor.expand_shape %collapsed_tensor [[0], [1, 2]] :
 * tensor<2x3xf32> to tensor<2x1x3xf32>
 *
 * - After:
 *   Replace all uses of %tensor with %expanded_tensor
 *
 * @note Match/Rewrite Rules:
 *       1. Fails if tryCollapseValue on input memref returns failure (no unit
 * dims)
 *       2. Aborts if tryExpandValueWithUnitDim fails (llvm_unreachable)
 *       3. Only applies to RankedTensorType output of ToTensorOp
 *       4. Preserves ToTensorOp attrs (restrict/writable) in new op
 */
// clang-format on
struct DropUnitDimsToTensorPattern
    : public OpRewritePattern<bufferization::ToTensorOp> {
public:
  using OpRewritePattern<bufferization::ToTensorOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(bufferization::ToTensorOp toTensorOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto loc = toTensorOp.getLoc();
    auto src = toTensorOp.getMemref();
    auto dst = toTensorOp.getResult();

    auto maybeCollapsedSrc = tryCollapseValue(src, rewriter, loc);

    if (succeeded(maybeCollapsedSrc)) {
      auto newToTensorOp = rewriter.create<bufferization::ToTensorOp>(
          loc, maybeCollapsedSrc.value(), toTensorOp.getRestrictAttr(),
          toTensorOp.getWritableAttr());
      // pass empty reassocIdxVec to create reassociations inside the func
      SmallVector<ReassociationIndices> reassocIdxVec;
      auto maybeNewDst = tryExpandValueWithUnitDim(
          newToTensorOp.getResult(), dst, reassocIdxVec, rewriter, loc);
      if (maybeNewDst.has_value()) {
        rewriter.replaceAllUsesWith(dst, maybeNewDst.value());
      } else {
        llvm_unreachable(
            "[DropUnitDimsToTensorPattern] fails in tryExpandValueWithUnitDim");
      }
    } else {
      return failure();
    }

    return success();
  }
};

// clang-format off
/**
 * @brief Generic rewrite pattern to drop unit dimensions from CopyOp variants
 * @details Collapse unit dims of CopyOp src/dst (MemRef/Tensor) via pattern
 * matching -> create new CopyOp with collapsed operands
 *
 * Template Specialization Support:
 * - CopyOp = memref::CopyOp (MemRef copy)
 * - CopyOp = bufferization::MaterializeInDestinationOp (bufferization copy)
 *
 * IR Transformation Flow (memref::CopyOp Example):
 * - Before:
 * memref.copy %reinterpret_cast, %alloc
 * : memref<5x50x2x1xi16, strided<[100, 2, 1, 1]>> to memref<5x50x2x1xi16>
 * - After:
 * %collapse_shape = memref.collapse_shape %reinterpret_cast [[0], [1], [2, 3]]
 * : memref<5x50x2x1xi16, strided<[100, 2, 1, 1]>> into memref<5x50x2xi16, strided<[100, 2, 1]>>
 * %collapse_shape_0 = memref.collapse_shape %alloc [[0], [1], [2, 3]]
 * : memref<5x50x2x1xi16> into memref<5x50x2xi16>
 * memref.copy %collapse_shape, %collapse_shape_0
 * : memref<5x50x2xi16, strided<[100, 2, 1]>> to memref<5x50x2xi16>
 *
 * @tparam CopyOp Target copy operation type (memref::CopyOp /
 * bufferization::MaterializeInDestinationOp)
 * @note Match/Rewrite Rules:
 *       1. Supports MemRefType/RankedTensorType src/dst operands
 *       2. Priority: pattern matching (ExpandShape->ExtractSlice/SubView) >
 * direct collapse (tryCollapseValue)
 *       3. Only proceeds if both src AND dst are collapsible (needUpdate=true)
 *       4. Preserves restrict/writable attrs for MaterializeInDestinationOp
 *       5. Fails if src/dst cannot be collapsed (no unit dims / pattern
 * mismatch)
 */
// clang-format on
template <typename OpTy>
struct DropUnitDimsLoadStorePattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(OpTy op, mlir::PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value src, dst;
    // We statically check the type of OpTy and throw error when OpTy type
    // is not memref::CopyOp or bufferization::MaterializeInDestinationOp in
    // the end of this func.
    src = op.getOperand(0);
    dst = op.getOperand(1);

    Value collapsedSrc, collapsedDst;
    bool needUpdate = false;
    std::optional<Value> maybeCollapsedSrc, maybeCollapsedDst;
    if (isa<RankedTensorType>(src.getType())) {
      maybeCollapsedSrc =
          checkValueMatchPattern<tensor::ExpandShapeOp, tensor::ExtractSliceOp>(
              src, rewriter);
    } else if (isa<MemRefType>(src.getType())) {
      maybeCollapsedSrc =
          checkValueMatchPattern<memref::ExpandShapeOp, memref::SubViewOp>(
              src, rewriter);
    }
    if (isa<RankedTensorType>(dst.getType())) {
      maybeCollapsedDst =
          checkValueMatchPattern<tensor::ExpandShapeOp, tensor::ExtractSliceOp>(
              dst, rewriter);
    } else if (isa<MemRefType>(dst.getType())) {
      maybeCollapsedDst =
          checkValueMatchPattern<memref::ExpandShapeOp, memref::SubViewOp>(
              dst, rewriter);
    }
    if (maybeCollapsedSrc.has_value() && maybeCollapsedDst.has_value()) {
      collapsedSrc = maybeCollapsedSrc.value();
      collapsedDst = maybeCollapsedDst.value();
      needUpdate = true;
    } else {
      auto maybeCollapsedSrc = tryCollapseValue(src, rewriter, loc);
      auto maybeCollapsedDst = tryCollapseValue(dst, rewriter, loc);
      if (succeeded(maybeCollapsedSrc) && succeeded(maybeCollapsedDst)) {
        collapsedSrc = maybeCollapsedSrc.value();
        collapsedDst = maybeCollapsedDst.value();
        needUpdate = true;
      } else {
        // clang-format off
        LLVM_DEBUG(
          llvm::dbgs() << "[DropUnitDimsLoadStorePattern]"
                          " src|dst is non-collapsible\n"
                       << "src: " << src << "\n"
                       << "dst: " << dst << "\n";
        );
        // clang-format on
        if (succeeded(maybeCollapsedSrc)) {
          collapsedSrc = maybeCollapsedSrc.value();
          auto *defOp = collapsedSrc.getDefiningOp();
          rewriter.eraseOp(defOp);
        }
        if (succeeded(maybeCollapsedDst)) {
          collapsedDst = maybeCollapsedDst.value();
          auto *defOp = collapsedDst.getDefiningOp();
          rewriter.eraseOp(defOp);
        }
      }
    }

    Operation *newOp;
    if (needUpdate) {
      if constexpr (std::is_same_v<OpTy, memref::CopyOp>) {
        newOp =
            rewriter.create<memref::CopyOp>(loc, collapsedSrc, collapsedDst);
      } else if constexpr (std::is_same_v<
                               OpTy,
                               bufferization::MaterializeInDestinationOp>) {
        newOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
            loc, collapsedSrc, collapsedDst);
        auto bufMIDOp = cast<bufferization::MaterializeInDestinationOp>(newOp);
        bufMIDOp.setRestrictAttr(op.getRestrictAttr());
        bufMIDOp.setWritableAttr(op.getWritableAttr());
      } else if constexpr (std::is_same_v<OpTy, hivm::StoreOp>) {
        // Only handle side-effecting hivm::StoreOp with no SSA results.
        if (!op->getResults().empty()) {
          return rewriter.notifyMatchFailure(
              op, "[DropUnitDimsLoadStorePattern] hivm::StoreOp with results "
                  "is not supported");
        }
        TypeRange resTypes;
        newOp = rewriter.create<hivm::StoreOp>(loc, resTypes, collapsedSrc,
                                               collapsedDst);
        auto oldStore = cast<hivm::StoreOp>(op);
        if (oldStore.getAtomicKindAttr())
          cast<hivm::StoreOp>(newOp).setAtomicKindAttr(
              oldStore.getAtomicKindAttr());
      } else {
        static_assert(std::is_same_v<OpTy, memref::AllocOp>,
                      "Unsupported OpTy type");
      }
      rewriter.replaceOp(op, newOp);
    } else {
      return failure();
    }

    return success();
  }
};

// clang-format off
/**
 * @brief Rewrite pattern to drop unit dimensions from memref::ReinterpretCastOp
 * @details Collapse unit dims of ReinterpretCastOp result shape → create
 * squeezed ReinterpretCastOp → expand back to original shape
 *
 * IR Transformation Flow:
 * - Before:
 *   %src = memref<2x1x3xf32, strided<[3,3,1], offset:0>>
 *   %cast = memref.reinterpret_cast %src to memref<2x1x3xf32, strided<[3,3,1],
 * offset:0>> : memref<2x1x3xf32> to memref<2x1x3xf32>
 *
 * - Step 1 (Squeeze Layout/Reassoc):
 *   newOffsets = [0], newSizes = [2,3], newStrides = [3,1] (unit dim 1 removed)
 *   reassocIdxVec = [[0], [1,2]]
 *   squeezedResType = memref<2x3xf32, strided<[3,1], offset:0>>
 *
 * - Step 2 (New ReinterpretCastOp):
 *   %squeezed_cast = memref.reinterpret_cast %src to %squeezedResType :
 * memref<2x1x3xf32> to memref<2x3xf32>
 *
 * - Step 3 (Expand Shape):
 *   %expanded = memref.expand_shape %squeezed_cast [[0], [1,2]] :
 * memref<2x3xf32> to memref<2x1x3xf32>
 *
 * - After:
 *   Replace original %cast with %expanded; ExpandShapeOp+CollapseShapeOp will
 * be folded later
 *
 * @note Match/Rewrite Rules:
 *       1. Fails if ReinterpretCastOp result rank ≤ 1 or shape has no unit dims
 *       2. Triton-specific: assumes ReinterpretCastOp has static shape (no
 * dynamic dims)
 *       3. Relies on getSqueezedLayoutReassoc for squeezed
 * offsets/sizes/strides + reassociation indices
 *       4. Ignore debug-mode type mismatch warnings
 * (ExpandShapeOp/CollapseShapeOp will be folded)
 *       5. Uses CollapseShapeOp::computeCollapsedType to derive squeezed memref
 * type
 */
// clang-format on
struct DropUnitDimsReinterpretCastPattern
    : public OpRewritePattern<memref::ReinterpretCastOp> {
  using OpRewritePattern<memref::ReinterpretCastOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resTy = op.getType();
    if (resTy.getRank() <= 1) {
      return rewriter.notifyMatchFailure(
          op,
          "[DropUnitDimsReinterpretCastPattern] ReinterpretCastOp's rank <= 1");
    }
    auto resShape = utils::getShape(resTy);
    auto squeezedResShape = reshape_utils::getSqueezedShape(resShape);
    if (squeezedResShape == resShape) {
      return rewriter.notifyMatchFailure(
          op, "[DropUnitDimsReinterpretCastPattern] ReinterpretCastOp's shape "
              "has none of unit dim");
    }
    // In triton, memref.reinterpret_cast will not be dynamic shaped.
    auto maybeSqueezedLayoutReassoc =
        getSqueezedLayoutReassoc(op, resShape, rewriter, loc);
    if (!maybeSqueezedLayoutReassoc) {
      // clang-format off
      LLVM_DEBUG(llvm::dbgs()
        << "[DropUnitDimsReinterpretCastPattern] ReinterpretCastOp fails "
           "to getSqueezedLayoutReassoc\n";
      );
      // clang-format off
      return failure();
    }
    auto [newOffsets, newSizes, newStrides, reassocIdxVec] =
        maybeSqueezedLayoutReassoc.value();
    auto squeezedResType =
        memref::CollapseShapeOp::computeCollapsedType(resTy, reassocIdxVec);
    auto newOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, squeezedResType, op.getSource(), newOffsets, newSizes, newStrides);
    auto expandOp = rewriter.create<memref::ExpandShapeOp>(
        loc, resShape, newOp.getResult(), reassocIdxVec);
    // memref::ExpandShapeOp's build func calls
    // ExpandShapeOp::computeExpandedType which gives memref<5xf32> instead of
    // memref<5xf32, strided<[1]>>. Then CollapseShapeOp::verify reports type
    // mismatch between the above two. Meanwhile, if you use the build func with
    // resTy, ExpandShapeOp::verify reports type mismatch between resTy you pass
    // in and the type given by ExpandShapeOp::computeExpandedType. However,
    // only -debug mode reports such "error" messages. The pair of ExpandShapeOp
    // & CollapseShapeOp would be folded. So ignore the above messages when
    // running with -debug.
    rewriter.replaceOp(op, expandOp);

    return success();
  }
};

// clang-format off
/**
 * @brief Generic rewrite pattern to drop unit dimensions from SubView/ExtractSliceOp variants
 * @details Collapse unit dims of source (MemRef/Tensor) → generate squeezed layout → create new SubView/ExtractSliceOp → expand to original shape;
 *          supports both memref::SubViewOp and tensor::ExtractSliceOp via template specialization
 *
 * Template Specialization Support:
 * - SubViewOp = memref::SubViewOp (MemRef dialect rank-reduced subview)
 * - SubViewOp = tensor::ExtractSliceOp (Tensor dialect rank-reduced extract_slice)
 *
 * IR Transformation Flow (memref::SubViewOp Example):
 * - Before:
 *   %src = memref<2x1x3xf32, strided<[3,3,1], offset:0>>
 *   %subview = memref.subview %src[0,0,0] [2,1,3] [1,1,1] : memref<2x1x3xf32> to memref<2x1x3xf32>
 *
 * - Step 1 (Collapse Source):
 *   %collapsed_src = memref.collapse_shape %src [[0], [1], [2]] : memref<2x1x3xf32> to memref<2x3xf32>
 *
 * - Step 2 (Squeezed Layout):
 *   newOffsets = [0,0], newSizes = [2,3], newStrides = [1,1] (unit dim 1 removed)
 *   reassocIdxVec = [[0], [1,2]]
 *
 * - Step 3 (New SubViewOp):
 *   %new_subview = memref.subview %collapsed_src[0,0] [2,3] [1,1] : memref<2x3xf32> to memref<2x3xf32>
 *
 * - Step 4 (Rank Reduction (if droppedDims)):
 *   If droppedDims=[1]:
 *   %rank_reduced = memref.collapse_shape %new_subview [[0], [1]] : memref<2x3xf32> to memref<2xf32>
 *   reassocIdxVec shrunk to [[0], [1]] (via shrinkReassocIdxByDroppedDims)
 *
 * - Step 5 (Expand to Original Shape):
 *   %expanded = memref.expand_shape %rank_reduced/%new_subview reassocIdxVec : memref<2xf32>/memref<2x3xf32> to memref<2x1x3xf32>
 *
 * - After:
 *   Replace all uses of %subview with %expanded
 *
 * @tparam SubViewOp Target op type (memref::SubViewOp / tensor::ExtractSliceOp)
 * @note Match/Rewrite Rules:
 *       1. Fails if tryCollapseValue on source / new op result returns failure
 *       2. Fails if getSqueezedLayoutReassoc cannot generate squeezed layout/reassociation
 *       3. Handles rank-reduced ops (droppedDims) via shrinkReassocIdxByDroppedDims (most efficient path)
 *       4. Aborts (llvm_unreachable) if tryExpandValueWithUnitDim fails (invalid unit dim expansion)
 *       5. Preserves core semantics (offsets/sizes/strides) for non-unit dims across MemRef/Tensor dialects
 */
// clang-format on
template <typename SubViewOp>
struct DropUnitDimsSubViewPattern : public OpRewritePattern<SubViewOp> {
  using OpRewritePattern<SubViewOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(SubViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSource();

    auto maybeCollapsedSrc = tryCollapseValue(src, rewriter, loc);
    if (!succeeded(maybeCollapsedSrc)) {
      LLVM_DEBUG(llvm::dbgs() << "[DropUnitDimsSubViewPattern] "
                                 "SubViewOp fails to tryCollapseValue\n";);
      return failure();
    }

    auto srcShape = utils::getShape(op.getSourceType());
    auto maybeSqueezedLayoutReassoc =
        getSqueezedLayoutReassoc(op, srcShape, rewriter, loc);
    if (!maybeSqueezedLayoutReassoc) {
      rewriter.eraseOp(maybeCollapsedSrc.value().getDefiningOp());
      LLVM_DEBUG(llvm::dbgs() << "[DropUnitDimsSubViewPattern] SubViewOp fails "
                                 "to getSqueezedLayoutReassoc\n";);
      return failure();
    }
    auto [newOffsets, newSizes, newStrides, reassocIdxVec] =
        maybeSqueezedLayoutReassoc.value();

    auto collapsedSrc = maybeCollapsedSrc.value();
    auto newOp = rewriter.create<SubViewOp>(loc, collapsedSrc, newOffsets,
                                            newSizes, newStrides);

    Value expandSrc = newOp.getResult();
    // memref::SubViewOp/tensor::ExtractSliceOp can be rank-reduced
    auto droppedDims = op.getDroppedDims();
    if (droppedDims.any()) {
      auto maybeRankReducedSrc =
          tryCollapseValue(newOp.getResult(), rewriter, loc);
      if (!succeeded(maybeRankReducedSrc)) {
        rewriter.eraseOp(newOp);
        rewriter.eraseOp(maybeCollapsedSrc.value().getDefiningOp());
        LLVM_DEBUG(llvm::dbgs()
                       << "[DropUnitDimsSubViewPattern] "
                          "New SubViewOp fails to tryCollapseValue\n";);
        return failure();
      }
      expandSrc = maybeRankReducedSrc.value();
      // To modify reassociation indices is the most effecient.
      reshape_utils::shrinkReassocIdxByDroppedDims(reassocIdxVec, droppedDims);
    }

    auto res = op.getResult();
    auto maybeNewDst =
        tryExpandValueWithUnitDim(expandSrc, res, reassocIdxVec, rewriter, loc);
    if (maybeNewDst.has_value()) {
      rewriter.replaceAllUsesWith(op.getResult(), maybeNewDst.value());
    } else {
      llvm_unreachable(
          "[DropUnitDimsSubViewPattern] fails in tryExpandValueWithUnitDim");
    }

    return success();
  }
};

struct DropUnitDimsInsertSlicePattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.getSource();
    auto dst = op.getDest();

    bool srcCollapsed = false;
    bool srcExpanded = false;
    Value collapsedSrc = src;
    if (src.getDefiningOp()) {
      auto expandOp = dyn_cast<tensor::ExpandShapeOp>(src.getDefiningOp());
      if (!expandOp) {
        return rewriter.notifyMatchFailure(
            op, "[DropUnitDimsInsertSlicePattern] src's definingOp is not "
                "tensor::ExpandShapeOp");
      }
      auto expandedShape = expandOp.getMixedOutputShape();
      auto oldSizes = op.getMixedSizes();
      LLVM_DEBUG(llvm::dbgs() << "expandOp: " << *expandOp << "\n";
                 llvm::dbgs() << "InsertSliceOp: " << *op << "\n";);
      if (isEqualConstantIntOrValueArray(expandedShape, oldSizes)) {
        srcExpanded = true;
        collapsedSrc = expandOp.getSrc();
      }
    }

    if(!srcExpanded && !src.getType().hasStaticShape()){
      LLVM_DEBUG(llvm::dbgs()
                     << "[DropUnitDimsInsertSlicePattern] "
                        "Can't deal with dynamic src shape\n";);
      return failure();
    }

    auto maybeCollapsedDst = tryCollapseValue(dst, rewriter, loc);
    if (!succeeded(maybeCollapsedDst)) {
      LLVM_DEBUG(llvm::dbgs()
                     << "[DropUnitDimsInsertSlicePattern] "
                        "InsertSliceOp fails to tryCollapseValue on dst\n";);
      return failure();
    }

    if (!srcExpanded) {
      auto maybeCollapsedSrc = tryCollapseValue(src, rewriter, loc);
      if (succeeded(maybeCollapsedSrc)) {
        collapsedSrc = maybeCollapsedSrc.value();
        srcCollapsed = true;
      }
    }

    auto dstShape = utils::getShape(op.getDestType());
    auto maybeSqueezedLayoutReassoc =
        getSqueezedLayoutReassoc(op, dstShape, rewriter, loc);
    if (!maybeSqueezedLayoutReassoc) {
      if (srcCollapsed && collapsedSrc.getDefiningOp()) {
        rewriter.eraseOp(collapsedSrc.getDefiningOp());
      }

      rewriter.eraseOp(maybeCollapsedDst.value().getDefiningOp());
      LLVM_DEBUG(llvm::dbgs()
                     << "[DropUnitDimsInsertSlicePattern] InsertSliceOp fails "
                        "to getSqueezedLayoutReassoc\n";);
      return failure();
    }
    auto [newOffsets, newSizes, newStrides, reassocIdxVec] =
        maybeSqueezedLayoutReassoc.value();

    auto collapsedDst = maybeCollapsedDst.value();
    auto newOp = rewriter.create<tensor::InsertSliceOp>(
        loc, collapsedSrc, collapsedDst, newOffsets, newSizes, newStrides);

    // TODO: check insert_slice is valid
    Value expandSrc = newOp.getResult();

    auto res = op.getResult();
    auto maybeNewDst =
        tryExpandValueWithUnitDim(expandSrc, res, reassocIdxVec, rewriter, loc);
    if (maybeNewDst.has_value()) {
      rewriter.replaceAllUsesWith(op.getResult(), maybeNewDst.value());
    } else {
      llvm_unreachable("[DropUnitDimsInsertSlicePattern] fails in "
                       "tryExpandValueWithUnitDim");
    }

    return success();
  }
};

struct DropUnitDimsSCFForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mutableInits = op.getInitArgsMutable();
    auto mutableYields = op.getYieldedValuesMutable().value();
    auto results = op.getResults();
    auto yieldOp = dyn_cast<scf::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp) {
      // In fact, even if scf.for has no ret val, there will be an implicitly
      // inserted scf.yield.
      return rewriter.notifyMatchFailure(op, "scf.for has no yield|inits");
    }
    bool isModified = false;
    for (auto [idx, init] : llvm::enumerate(mutableInits)) {
      auto initVal = init.get();
      rewriter.setInsertionPoint(op);
      auto maybeCollapsedInit = tryCollapseValue(initVal, rewriter, loc);
      if (succeeded(maybeCollapsedInit)) {
        LDBG("Before altering: " << *(op->getParentOp()));
        // inits
        auto collapsedInit = maybeCollapsedInit.value();
        auto collapsedInitTy = collapsedInit.getType();
        mutableInits[idx].assign(collapsedInit);
        // iter args
        auto iterArg = op.getRegionIterArg(idx);
        iterArg.setType(collapsedInitTy);
        SmallVector<ReassociationIndices> reassocIdxVec;
        rewriter.setInsertionPointToStart(op.getBody());
        auto maybeExpandedIterArg = tryExpandValueWithUnitDim(
            iterArg, initVal, reassocIdxVec, rewriter, loc);
        if (maybeExpandedIterArg.has_value()) {
          auto expandedIterArg = maybeExpandedIterArg.value();
          rewriter.replaceAllUsesExcept(iterArg, expandedIterArg,
                                        expandedIterArg.getDefiningOp());
        } else {
          llvm_unreachable(
              "[DropUnitDimsSCFForPattern] fails in tryExpandValueWithUnitDim");
        }
        // yield values
        auto origYield = mutableYields[idx].get();
        rewriter.setInsertionPoint(yieldOp);
        auto collapsedYieldVal = rewriter.create<tensor::CollapseShapeOp>(
            loc, collapsedInitTy, origYield, reassocIdxVec);
        mutableYields[idx].assign(collapsedYieldVal);
        // results
        auto res = results[idx];
        auto resOrigTy = res.getType();
        res.setType(collapsedInitTy);
        rewriter.setInsertionPointAfter(op);
        auto expandResOp = rewriter.create<tensor::ExpandShapeOp>(
            loc, resOrigTy, res, reassocIdxVec);
        auto expandedRes = expandResOp.getResult();
        rewriter.replaceAllUsesExcept(res, expandedRes, expandResOp);
        LDBG("After altering: " << *(op->getParentOp()));
        isModified = true;
      }
    }
    if (!isModified) {
      return rewriter.notifyMatchFailure(op, "scf.for has nothing to modify");
    }
    return success();
  }
};

struct DropUnitDimsLinalgOp
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  DropUnitDimsLinalgOp(MLIRContext *context,
                       std::function<bool(Operation *)> shouldSkip,
                       linalg::ControlDropUnitDims options = {},
                       PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        options(std::move(options)), shouldSkip(shouldSkip) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (isa<linalg::MatmulOp, linalg::QuantizedMatmulOp,
            linalg::MatmulTransposeAOp, linalg::MatmulTransposeBOp,
            linalg::Mmt4DOp, linalg::BatchMmt4DOp, linalg::BatchMatmulOp,
            linalg::BatchMatmulTransposeAOp, linalg::BatchMatmulTransposeBOp,
            linalg::QuantizedBatchMatmulOp, linalg::BatchReduceMatmulOp,
            linalg::Conv1DOp, linalg::Conv2DOp, linalg::Conv3DOp,
            linalg::Conv1DNwcWcfOp, linalg::Conv1DNcwFcwOp,
            linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcFhwcOp,
            linalg::Conv2DNhwcHwcfQOp, linalg::Conv2DNhwcFhwcQOp,
            linalg::Conv2DNchwFchwOp, linalg::Conv2DNgchwFgchwOp,
            linalg::Conv2DNgchwGfchwOp, linalg::Conv2DNgchwGfchwQOp,
            linalg::Conv3DNdhwcDhwcfOp, linalg::Conv3DNdhwcDhwcfQOp,
            linalg::Conv3DNcdhwFcdhwOp, linalg::DepthwiseConv1DNwcWcOp,
            linalg::DepthwiseConv1DNcwCwOp, linalg::DepthwiseConv1DNwcWcmOp,
            linalg::DepthwiseConv2DNhwcHwcOp, linalg::DepthwiseConv2DNchwChwOp,
            linalg::DepthwiseConv2DNhwcHwcQOp,
            linalg::DepthwiseConv2DNhwcHwcmOp,
            linalg::DepthwiseConv2DNhwcHwcmQOp,
            linalg::DepthwiseConv3DNdhwcDhwcOp,
            linalg::DepthwiseConv3DNcdhwCdhwOp,
            linalg::DepthwiseConv3DNdhwcDhwcmOp, linalg::PoolingNdhwcSumOp,
            linalg::PoolingNdhwcMaxOp, linalg::PoolingNdhwcMinOp,
            linalg::FillRng2DOp>(linalgOp.getOperation())) {
      return rewriter.notifyMatchFailure(
          linalgOp,
          "Ops like matmul cannot drop unit-dim (e.g. matmul 5x1 X 1x6)");
    }

    if (shouldSkip(linalgOp.getOperation())) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Do not process ops inside scope");
    }

    FailureOr<linalg::DropUnitDimsResult> result =
        linalg::dropUnitDims(rewriter, linalgOp, options);
    if (failed(result)) {
      return failure();
    }
    rewriter.replaceOp(linalgOp, result->replacements);
    return success();
  }

private:
  linalg::ControlDropUnitDims options;
  std::function<bool(Operation *)> shouldSkip;
};

namespace {
/// Convert `extract_slice` operations to rank-reduced versions.
struct RankReducedExtractSliceOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType resultType = sliceOp.getType();
    SmallVector<OpFoldResult> targetShape;
    for (auto size : resultType.getShape())
      targetShape.push_back(rewriter.getIndexAttr(size));
    auto reassociation =
        linalg::getReassociationMapForFoldingUnitDims(targetShape);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(resultType.getRank()))
      return failure();

    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    auto rankReducedType = cast<RankedTensorType>(
        tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            reassociation->size(), sliceOp.getSourceType(), offsets, sizes,
            strides));

    Location loc = sliceOp.getLoc();
    Value newSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, rankReducedType, sliceOp.getSource(), offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        sliceOp, resultType, newSlice, *reassociation);
    return success();
  }
};

/// Convert `insert_slice` operations to rank-reduced versions.
/// This patterns works with both InsertSliceOp and ParallelInsertSliceOp.
template <typename InsertOpTy>
struct RankReducedInsertSliceOp : public OpRewritePattern<InsertOpTy> {
  using OpRewritePattern<InsertOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOpTy insertSliceOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType sourceType = insertSliceOp.getSourceType();
    SmallVector<OpFoldResult> targetShape;
    for (auto size : sourceType.getShape())
      targetShape.push_back(rewriter.getIndexAttr(size));
    auto reassociation =
        linalg::getReassociationMapForFoldingUnitDims(targetShape);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(sourceType.getRank()))
      return failure();

    Location loc = insertSliceOp.getLoc();
    tensor::CollapseShapeOp reshapedSource;
    {
      OpBuilder::InsertionGuard g(rewriter);
      // The only difference between InsertSliceOp and ParallelInsertSliceOp
      // is the insertion point is just before the ParallelCombiningOp in the
      // parallel case.
      if (std::is_same<InsertOpTy, tensor::ParallelInsertSliceOp>::value)
        rewriter.setInsertionPoint(insertSliceOp->getParentOp());
      reshapedSource = rewriter.create<tensor::CollapseShapeOp>(
          loc, insertSliceOp.getSource(), *reassociation);
    }
    rewriter.replaceOpWithNewOp<InsertOpTy>(
        insertSliceOp, reshapedSource, insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());
    return success();
  }
};
} // namespace

void HFusionFoldUnitDimsPass::runOnOperation() {
  Operation *funcOp = getOperation();
  bool foundMatmul = false;
  funcOp->walk([&](Operation *op) {
    if (llvm::isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
      foundMatmul = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (foundMatmul) {
    return;
  }

  MLIRContext *context = &getContext();

  auto shouldSkip = [](Operation *op) -> bool {
    auto scopeOp = op->getParentOfType<scope::ScopeOp>();
    if (!scopeOp) {
      return false;
    }

    if (scopeOp->hasAttr("vector_mode") && scopeOp->hasAttr("outline")) {
      return true;
    }

    return false;
  };

  // First drop unit dims on memref related ops
  {
    RewritePatternSet patterns(context);
    patterns.insert<FilteredPattern<DropUnitDimsLoadStorePattern<memref::CopyOp>,
                                    memref::CopyOp>>(context, shouldSkip);
    patterns.insert<FilteredPattern<
        DropUnitDimsLoadStorePattern<bufferization::MaterializeInDestinationOp>,
        bufferization::MaterializeInDestinationOp>>(context, shouldSkip);
    patterns.insert<FilteredPattern<DropUnitDimsLoadStorePattern<hivm::StoreOp>,
        hivm::StoreOp>>(context, shouldSkip);
    patterns.insert<FilteredPattern<DropUnitDimsToTensorPattern,
                                    bufferization::ToTensorOp>>(context,
                                                                shouldSkip);
    patterns.insert<FilteredPattern<DropUnitDimsAllocPattern, memref::AllocOp>>(
        context, shouldSkip);
    patterns.insert<FilteredPattern<CanonicalizeBufMIDPattern,
                                    bufferization::MaterializeInDestinationOp>>(
        context, shouldSkip);
    patterns.insert<FilteredPattern<DropUnitDimsReinterpretCastPattern,
                                    memref::ReinterpretCastOp>>(context,
                                                                shouldSkip);
    patterns.insert<FilteredPattern<
        DropUnitDimsSubViewPattern<memref::SubViewOp>, memref::SubViewOp>>(
        context, shouldSkip);
    patterns.insert<
        FilteredPattern<DropUnitDimsSubViewPattern<tensor::ExtractSliceOp>,
                        tensor::ExtractSliceOp>>(context, shouldSkip);
    patterns.insert<
        FilteredPattern<DropUnitDimsInsertSlicePattern, tensor::InsertSliceOp>>(
        context, shouldSkip);
    memref::CollapseShapeOp::getCanonicalizationPatterns(patterns,
                                                         patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }

  // Then drop unit dims on linalg related ops
  {
    RewritePatternSet patterns(context);
    linalg::ControlDropUnitDims options;
    options.foldRankReducingSlices = true;
    options.enableExperimentalDropping = true;
    options.avoidZeroRanks = true;
    patterns.add<DropUnitDimsLinalgOp>(context, shouldSkip, options);
    patterns.add<FilteredPattern<DropUnitDimsSCFForPattern, scf::ForOp>>(
        context, shouldSkip);
    patterns.add<RankReducedExtractSliceOp,
                 RankReducedInsertSliceOp<tensor::InsertSliceOp>,
                 RankReducedInsertSliceOp<tensor::ParallelInsertSliceOp>>(
        context);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    memref::populateResolveShapedTypeResultDimsPatterns(patterns);
    linalg::populateMoveInitOperandsToInputPattern(patterns);
    SmallVector<std::string> disabledPatterns;
    disabledPatterns.push_back("FoldFillWithTensorReshape");
    FrozenRewritePatternSet frozenPatterns(std::move(patterns),
                                           disabledPatterns);
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns)))
      return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hfusion::createHFusionFoldUnitDimsPass() {
  return std::make_unique<HFusionFoldUnitDimsPass>();
}
