//===- HIVMCanonicalizations.cpp - HIVM Canonicalization implementation ---===//
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

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {
void collectNotOneDims(ShapedType st, const ArrayRef<int64_t> &checkDims,
                       llvm::SmallVectorImpl<int64_t> &notOneDims) {
  const auto &shape = st.getShape();
  for (const auto &dim : checkDims) {
    if (shape[dim] != 1) {
      notOneDims.push_back(dim);
    }
  }
}

struct RedudantVBrcOp : public OpRewritePattern<VBrcOp> {
  using OpRewritePattern<VBrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VBrcOp brcOp,
                                PatternRewriter &rewriter) const final {
    auto brcDims = brcOp.getBroadcastDims();
    auto dstType = brcOp.getDst().getType();
    if (!isa<ShapedType>(dstType) ||
        (!brcOp.hasPureBufferSemantics() && !brcOp.hasPureTensorSemantics())) {
      return failure();
    }

    llvm::SmallVector<int64_t> keepBrcDims;
    collectNotOneDims(cast<ShapedType>(dstType), brcDims, keepBrcDims);
    if (keepBrcDims.size() == brcDims.size()) {
      return failure();
    }

    if (keepBrcDims.empty()) {
      if (brcOp.hasPureTensorSemantics()) {
        rewriter.replaceAllUsesWith(brcOp->getResults(),
                                    ArrayRef{brcOp.getSrc()});
        rewriter.eraseOp(brcOp);
      } else {
        if (brcOp.getSrc().getType().isIntOrFloat()) {
          // TODO : support to replace by scalar operation.
          return failure();
        }
        rewriter.create<hivm::CopyOp>(brcOp.getLoc(), brcOp->getResultTypes(),
                                      brcOp.getSrc(), brcOp.getDst());
        rewriter.eraseOp(brcOp);
      }
    } else {
      assert(keepBrcDims.size() < brcDims.size());
      rewriter.modifyOpInPlace(brcOp.getOperation(), [&]() {
        brcOp.setBroadcastDims(ArrayRef<int64_t>(keepBrcDims));
      });
    }

    return success();
  }
};

struct RedudantVReduceOp : public OpRewritePattern<VReduceOp> {
  using OpRewritePattern<VReduceOp>::OpRewritePattern;

  /// Decompose Redundant Reduce With Index
  ///
  /// This function converts reduceOp with index for 1-sized dimension into
  /// copyOp and brcOp
  /// e.g.
  ///   hivm.hir.vreduce <max_with_index> ins(%arg0 : memref<3x1xf32>)
  ///     outs(%arg1, %arg2 : memref<3x1xf32>, memref<3x1xi32>)
  ///     reduce_dims = [1]
  /// converts to
  ///   %const0 = arith.constant 0
  ///   hivm.hir.copy ins(%arg0 : memref<3x1xf32>) outs(%arg1 : memref<3x1xf32>)
  ///   hivm.hir.vbrc ins(%const0 : i32) outs(%arg2 : memref<3x1xi32>)
  LogicalResult decomposeRedundantReduceWithIndex(VReduceOp &reduceOp,
                                                  PatternRewriter &rewriter,
                                                  bool isTensor) const {
    auto loc = reduceOp.getLoc();
    // Step 1: Convert Reduced Value into CopyOp
    auto copyOp = rewriter.create<hivm::CopyOp>(
        loc, isTensor ? reduceOp->getResult(0).getType() : TypeRange(),
        reduceOp.getSrc(), reduceOp.getDstValue());
    // Step 2: Convert Reduced Index into BroadcastOp with ConstantOp as input
    Value constZero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto vbrcOp = rewriter.create<hivm::VBrcOp>(
        loc, isTensor ? reduceOp->getResult(1).getType() : TypeRange(),
        constZero, reduceOp.getDstIndex(), rewriter.getDenseI64ArrayAttr({}));
    if (isTensor) {
      rewriter.replaceAllUsesWith(reduceOp->getResult(0), copyOp->getResult(0));
      rewriter.replaceAllUsesWith(reduceOp->getResult(1), vbrcOp->getResult(0));
    }
    rewriter.eraseOp(reduceOp);
    return success();
  }

  LogicalResult matchAndRewrite(VReduceOp reduceOp,
                                PatternRewriter &rewriter) const final {
    auto reduceDims = reduceOp.getReduceDims();
    auto srcType = reduceOp.getSrc().getType();
    if ((!isa<TensorType>(srcType) && !isa<MemRefType>(srcType)) ||
        (!reduceOp.hasPureTensorSemantics() &&
         !reduceOp.hasPureBufferSemantics())) {
      return failure();
    }

    llvm::SmallVector<int64_t> keepReduceDims;
    collectNotOneDims(cast<ShapedType>(srcType), reduceDims, keepReduceDims);
    if (keepReduceDims.size() == reduceDims.size()) {
      return failure();
    }

    if (keepReduceDims.empty()) {
      if (reduceOp.hasPureTensorSemantics()) {
        assert(isa<TensorType>(srcType));
        auto arith = reduceOp.getArithAttr().getReduceOp();
        if (VReduceOp::isArgminOrArgmax(arith)) {
          if (!reduceOp->getResult(1).getUsers().empty()) {
            return decomposeRedundantReduceWithIndex(reduceOp, rewriter,
                                                     isa<TensorType>(srcType));
          }
        }
        rewriter.replaceAllUsesWith(reduceOp->getResults()[0],
                                    ArrayRef{reduceOp.getSrc()});
        rewriter.eraseOp(reduceOp);
      } else {
        if (reduceOp.getDstValue().getType().isIntOrFloat()) {
          // TODO : support to replace by scalar operation.
          return failure();
        }
        auto arith = reduceOp.getArithAttr().getReduceOp();
        if (VReduceOp::isArgminOrArgmax(arith)) {
          return decomposeRedundantReduceWithIndex(reduceOp, rewriter,
                                                   isa<TensorType>(srcType));
        }
        rewriter.create<hivm::CopyOp>(reduceOp.getLoc(), TypeRange(),
                                      reduceOp.getSrc(),
                                      reduceOp.getDstValue());
        rewriter.eraseOp(reduceOp);
      }
    } else {
      assert(keepReduceDims.size() < reduceDims.size());
      rewriter.modifyOpInPlace(reduceOp.getOperation(), [&]() {
        reduceOp.setReduceDims(ArrayRef<int64_t>(keepReduceDims));
      });
    }

    return success();
  }
};

struct RedudantVReduceInitOp : public OpRewritePattern<VReduceOp> {
  using OpRewritePattern<VReduceOp>::OpRewritePattern;

  bool isFillByConst(Value v, Attribute cstAttr) const {
    if (isa<BlockArgument>(v)) {
      auto blockArg = cast<BlockArgument>(v);
      auto parentOp = blockArg.getOwner()->getParentOp();
      auto blockIndx = blockArg.getArgNumber();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        auto forInitVal = forOp.getInitArgs()[blockIndx];
        return isFillByConst(forInitVal, cstAttr);
      }
      return false;
    }

    if (auto brcOp = v.getDefiningOp<hivm::VBrcOp>()) {
      auto brcSrc = brcOp.getSrc();
      if (auto cstOp = brcSrc.getDefiningOp<arith::ConstantOp>()) {
        auto valAttr = cstOp.getValueAttr();
        return valAttr == cstAttr;
      }
    }

    if (auto expandOp = v.getDefiningOp<tensor::ExpandShapeOp>()) {
      return isFillByConst(expandOp.getSrc(), cstAttr);
    } else if (auto collapseOp = v.getDefiningOp<tensor::CollapseShapeOp>()) {
      return isFillByConst(collapseOp.getSrc(), cstAttr);
    }

    return false;
  }

  LogicalResult matchAndRewrite(VReduceOp reduceOp,
                                PatternRewriter &rewriter) const final {
    auto reduceInitOperand = reduceOp.getDpsInitOperand(0);
    auto initAttr = reduceOp.getInit();
    if (!isFillByConst(reduceInitOperand->get(), initAttr)) {
      return failure();
    }
    auto emptyValue = mlir::tensor::createTensorEmptyOp(
        rewriter, reduceOp->getLoc(), reduceInitOperand->get());
    rewriter.modifyOpInPlace(reduceOp, [&]() {
      reduceOp.getDpsInitsMutable()[0].assign(emptyValue);
    });

    return success();
  }
};

template <typename CumOp>
struct RedundantVCumOp : public OpRewritePattern<CumOp> {
  using OpRewritePattern<CumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CumOp cumOp,
                                PatternRewriter &rewriter) const final {
    auto cumDims = cumOp.getCumDims();
    auto srcType = cumOp.getSrc().getType();
    if ((!isa<ShapedType>(srcType)) ||
        (!cumOp.hasPureTensorSemantics() && !cumOp.hasPureBufferSemantics())) {
      return failure();
    }

    llvm::SmallVector<int64_t> keepCumDims;
    collectNotOneDims(cast<ShapedType>(srcType), cumDims, keepCumDims);
    if (keepCumDims.size() == cumDims.size()) {
      // all cum_dims need to keep, so nothing to do with canonicalization
      return failure();
    }

    if (keepCumDims.empty()) {
      if (cumOp.hasPureTensorSemantics()) {
        rewriter.replaceAllUsesWith(cumOp->getResults()[0],
                                    ArrayRef{cumOp.getSrc()});
        rewriter.eraseOp(cumOp);
      } else {
        if (cumOp.getDst().getType().isIntOrFloat()) {
          // TODO : support to replace by scalar operation.
          return failure();
        }
        rewriter.create<hivm::CopyOp>(cumOp.getLoc(), TypeRange(),
                                      cumOp.getSrc(), cumOp.getDst());
        rewriter.eraseOp(cumOp);
      }
    } else {
      assert(keepCumDims.size() < cumDims.size());
      rewriter.modifyOpInPlace(cumOp.getOperation(),
                               [&]() { cumOp.setCumDims(keepCumDims); });
    }

    return success();
  }
};

struct RedudantVTransposeOpOp : public OpRewritePattern<VTransposeOp> {
  using OpRewritePattern<VTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VTransposeOp transOp,
                                PatternRewriter &rewriter) const final {
    if (isOpTriviallyDead(transOp)) {
      rewriter.eraseOp(transOp);
      return success();
    }

    // Initialize baseline as golden for simulation result comparison
    SmallVector<int64_t> base(transOp.getPermutation().size());
    for (size_t i = 0; i < base.size(); ++i)
      base[i] = static_cast<int>(i);

    // Helper function simulating permutations on input data
    auto simulate = [](const ArrayRef<int64_t> input,
                       const ArrayRef<int64_t> permutation) {
      assert(input.size() == permutation.size());
      SmallVector<int64_t> res(permutation.size());
      for (const auto [i, v] : llvm::enumerate(permutation))
        res[v] = input[i];
      return res;
    };

    SmallVector<int64_t> sim(base);
    // Bottom-up searching the chain of transpose ops
    VTransposeOp nextTransOp = transOp;
    while (nextTransOp) {
      sim = simulate(sim, nextTransOp.getPermutation());
      if (sim == base) {
        rewriter.replaceOp(transOp, {nextTransOp.getSrc()});
        return success();
      }
      nextTransOp = nextTransOp.getSrc().getDefiningOp<VTransposeOp>();
    }

    return failure();
  }
};

// This will check Low or High if padding exists and whether it is
// Static or Dynamic and return it as a mlir Value for the selected dimension
Value createPaddingValue(OpBuilder &rewriter, VPadOp padOp, bool isLow,
                         unsigned int dim) {
  Value padValue = nullptr;
  ArrayRef<int64_t> padStaticArray =
      isLow ? padOp.getStaticLow() : padOp.getStaticHigh();
  if (!padStaticArray.empty()) {
    assert(dim < padStaticArray.size() && "Dimension index out of bounds");
    int64_t padStatic = padStaticArray[dim];
    if (ShapedType::isDynamic(padStatic)) {
      padValue = isLow ? cast<Value>(padOp.getLow()[dim])
                       : cast<Value>(padOp.getHigh()[dim]);
    } else {
      auto indexAttr = rewriter.getIndexAttr(static_cast<size_t>(padStatic));
      padValue = rewriter.create<arith::ConstantOp>(padOp->getLoc(), indexAttr);
    }
  }
  return padValue;
}
/// if hivm.vpad op is 1 dimensional and has source of hivm.load type, create a
/// new hivm.load op with adjusted left/right padding, then replace all users.
struct FoldLoadAndVPadPattern : public OpRewritePattern<VPadOp> {
  using OpRewritePattern<VPadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VPadOp padOp,
                                PatternRewriter &rewriter) const override {
    // convert hivm.hir.vpad to hivm.hir.load with padding
    auto padSrc = padOp.getSrc();
    hivm::LoadOp loadOp = padSrc.getDefiningOp<hivm::LoadOp>();
    if (!loadOp)
      return failure();
    auto hasPad = loadOp.getPadMode().has_value();
    if (hasPad)
      // TODO: support load with pad and sum pad numbers
      return rewriter.notifyMatchFailure(padOp, "pad source has pad");
    auto src = loadOp.getSrc();
    Value dst = padOp.getDst();
    auto padValue = padOp.getPadValue();
    auto resType = dyn_cast<RankedTensorType>(padOp->getResultTypes()[0]);
    // this pattern does not support multiple dimension padding because
    // hivm.load/hardware only supports one dimension padding.
    if (resType.getRank() > 1) {
      return rewriter.notifyMatchFailure(padOp, "unsupported dimensions");
    }
    Value leftPad =
        createPaddingValue(rewriter, padOp, /*isLow=*/true, /*dim=*/0);
    Value rightPad =
        createPaddingValue(rewriter, padOp, /*isLow=*/false, /*dim=*/0);
    auto padModeAttr = rewriter.getAttr<PadModeAttr>(PadMode::PadValue);
    rewriter.replaceOpWithNewOp<hivm::LoadOp>(
        padOp, resType, src, dst, padModeAttr, padValue, leftPad, rightPad);
    return success();
  }
};

// Eliminate inline broadcast that has equivalent src and dst sizes on the
// broadcasted dim
struct EliminateTrivialInlineBrc
    : public OpInterfaceRewritePattern<HIVMStructuredOp> {
  using ParentClass = OpInterfaceRewritePattern<HIVMStructuredOp>;
  using ParentClass::ParentClass;

  LogicalResult matchAndRewrite(HIVMStructuredOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<::mlir::OpTrait::BroadcastableOTF>())
      return failure();
    if (op.getBroadcastArray().empty())
      return failure();
    llvm::SmallVector<int64_t> newBrcDims;
    // filter prev broadcast dims
    for (const int64_t dim : op.getBroadcastArray()) {
      std::optional<int64_t> dimSize = std::nullopt;
      auto isSameAtDim = [&dimSize, dim](Type vType) -> bool {
        auto tp = dyn_cast<ShapedType>(vType);
        // ignore non-ShapedType
        if (!tp || tp.getRank() <= dim)
          return true;
        // Treat each dynamic size as different size
        if (tp.isDynamicDim(dim))
          return false;
        int64_t x = tp.getDimSize(dim);
        // Store the first size and compare latter sizes with the stored one.
        if (!dimSize.has_value()) {
          dimSize = x;
          return true;
        }
        return dimSize.value() == x;
      };
      if (!llvm::all_of(op.getHIVMOperandTypes(/*includeExtraBuffer=*/false),
                        isSameAtDim))
        newBrcDims.emplace_back(dim);
    }
    if (newBrcDims.size() == op.getBroadcastArray().size())
      return failure();
    rewriter.modifyOpInPlace(op, [&]() {
      op->setAttr(op.getBroadcastAttrString(),
                  rewriter.getDenseI64ArrayAttr(newBrcDims));
    });
    return llvm::success();
  }
};

} // namespace

void VBrcOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  results.add<RedudantVBrcOp>(context);
}

void VReduceOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                            ::mlir::MLIRContext *context) {
  results.add<RedudantVReduceOp
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  , RedudantVReduceInitOp
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY
  >(context);
}

void VCumsumOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                            ::mlir::MLIRContext *context) {
  results.add<RedundantVCumOp<VCumsumOp>>(context);
}

void VCumprodOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                             ::mlir::MLIRContext *context) {
  results.add<RedundantVCumOp<VCumprodOp>>(context);
}

void VTransposeOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add<RedudantVTransposeOpOp>(context);
}

void VPadOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  results.add<FoldLoadAndVPadPattern>(context);
}

//===----------------------------------------------------------------------===//
// HIVM DMA Ops
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::fold(hivm::CopyOp::FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

LogicalResult LoadOp::fold(hivm::LoadOp::FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

LogicalResult StoreOp::fold(hivm::StoreOp::FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// HIVM Dialect
//===----------------------------------------------------------------------===//
void mlir::hivm::HIVMDialect::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results) const {
  results.add<EliminateTrivialInlineBrc>(getContext());
}
