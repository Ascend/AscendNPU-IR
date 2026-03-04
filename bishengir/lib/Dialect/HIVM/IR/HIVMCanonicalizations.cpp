//===- HIVMCanonicalizations.cpp - HIVM Canonicalization implementation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
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

struct RedudantVPowOp : public OpRewritePattern<VPowOp> {
  using OpRewritePattern<VPowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VPowOp powOp,
                                PatternRewriter &rewriter) const final {
    if (!powOp.hasPureTensorSemantics()) {
      return failure();
    }

    auto src = powOp.getSrc();
    auto src0Type = src[0].getType();
    auto inits = powOp.getDpsInits();

    auto src1ConstOp =
        dyn_cast_or_null<arith::ConstantOp>(src[1].getDefiningOp());
    if (src1ConstOp == nullptr) {
      return failure();
    }

    auto src1ConstValue = src1ConstOp.getValue();
    auto elementType = getElementTypeOrSelf(src1ConstValue.getType());

    if (utils::isConst<FloatAttr, float_t>(src1ConstValue, 0.0)) {
      // vpow(x, 0.0) converts to 1
      Value one = rewriter.create<arith::ConstantOp>(
          powOp.getLoc(), elementType, rewriter.getOneAttr(elementType));
      auto hivmVBrcOp = rewriter.create<hivm::VBrcOp>(
          powOp.getLoc(), src0Type, one, inits[0],
          rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
      rewriter.replaceOp(powOp, hivmVBrcOp);
      return success();
    }

    if (utils::isConst<FloatAttr, float_t>(src1ConstValue, 0.5)) {
      // vpow(x, 0.5) converts to vsqrt(x)
      auto hivmVSqrtOp = rewriter.create<hivm::VSqrtOp>(
          powOp.getLoc(), src0Type, src[0], inits[0]);
      rewriter.replaceOp(powOp, hivmVSqrtOp);
      return success();
    }

    if (utils::isConst<FloatAttr, float_t>(src1ConstValue, 1.0)) {
      // vpow(x, 1.0) converts to x
      rewriter.replaceOp(powOp, {src[0]});
      return success();
    }

    if (utils::isConst<FloatAttr, float_t>(src1ConstValue, 2.0)) {
      // vpow(x, 2.0) converts to x * x
      auto hivmVMulOp = rewriter.create<hivm::VMulOp>(
          powOp.getLoc(), src0Type, ValueRange{src[0], src[0]}, inits[0]);
      rewriter.replaceOp(powOp, hivmVMulOp);
      return success();
    }

    if (utils::isConst<FloatAttr, float_t>(src1ConstValue, 3.0)) {
      // vpow(x, 3.0) converts to x * x * x
      auto emptyOp =
          mlir::utils::createEmptyOp(rewriter, powOp.getLoc(), inits[0]);

      /// step 1: y = x * x
      auto oneHivmVMulOp = rewriter.create<hivm::VMulOp>(
          powOp.getLoc(), src0Type, ValueRange{src[0], src[0]},
          ValueRange(emptyOp));
      auto oneVmulDst = oneHivmVMulOp.getResult();

      /// step 2: z = x * y
      auto twoHivmVMulOp = rewriter.create<hivm::VMulOp>(
          powOp.getLoc(), src0Type, ValueRange{src[0], oneVmulDst[0]},
          inits[0]);
      rewriter.replaceOp(powOp, twoHivmVMulOp);
      return success();
    }

    return failure();
  }
};

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
        if (utils::isReduceWithIndex(arith)) {
          if (!reduceOp->getResult(1).use_empty()) {
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
        if (utils::isReduceWithIndex(arith)) {
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
    auto moduleOp = reduceOp->getParentOfType<mlir::ModuleOp>();
    if (mlir::hacc::utils::isAscend950(moduleOp)) {
      return failure();
    }
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

} // namespace

void VPowOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  results.add<RedudantVPowOp>(context);
}

void VBrcOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  results.add<RedudantVBrcOp>(context);
}

void VReduceOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                            ::mlir::MLIRContext *context) {
  results.add<RedudantVReduceOp, RedudantVReduceInitOp>(context);
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

template <typename CustomOpT>
struct CustomOpCanonicalizer : public OpRewritePattern<CustomOpT> {
  using OpRewritePattern<CustomOpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomOpT customOp,
                                PatternRewriter &rewriter) const final {
    if (!customOp.isBuiltin())
      return failure();

    const auto &builtinInfo = CustomOpT::kBuiltins.at(customOp.getName());
    const auto &coreType = customOp.getCoreType();
    if (!coreType || *coreType != builtinInfo.coreType) {
      customOp.setCoreType(builtinInfo.coreType);
      return success();
    }

    if constexpr (std::is_same_v<CustomOpT, CustomOp>) {
      if (customOp.getPipe() != builtinInfo.pipe) {
        customOp.setPipe(builtinInfo.pipe);
        return success();
      }
    } else if constexpr (std::is_same_v<CustomOpT, CustomMacroOp>) {
      if (customOp.getInPipe() != builtinInfo.inPipe) {
        customOp.setInPipe(builtinInfo.inPipe);
        return success();
      }
      if (customOp.getOutPipe() != builtinInfo.outPipe) {
        customOp.setOutPipe(builtinInfo.outPipe);
        return success();
      }
    }

    const auto &vfMode = customOp.getVFMode();
    if (!vfMode || *vfMode != builtinInfo.vfMode) {
      customOp.setVFMode(builtinInfo.vfMode);
      return success();
    }

    const auto &gmAddrArgsIndices = customOp.getGMAddrArgsIndices();
    if (!gmAddrArgsIndices ||
        *gmAddrArgsIndices != builtinInfo.gmAddrArgsIndices) {
      customOp.setGMAddrArgsIndices(builtinInfo.gmAddrArgsIndices);
      return success();
    }

    return failure();
  }
};

void CustomOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                           ::mlir::MLIRContext *context) {
  results.add<CustomOpCanonicalizer<CustomOp>>(context);
}

void CustomMacroOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add<CustomOpCanonicalizer<CustomMacroOp>>(context);
}
