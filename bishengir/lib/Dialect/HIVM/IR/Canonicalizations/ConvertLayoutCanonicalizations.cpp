//===- ConvertLayoutCanonicalization.cpp - ConvertLayout Canonicalization -===//
//
// Canonicalization patterns for hivm.hir.convert_layout operations
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "convert-layout-canonicalizations"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")
namespace mlir::hivm {

//===----------------------------------------------------------------------===//
// Eliminate Consecutive Inverse Conversions
//===----------------------------------------------------------------------===//

/// Checks if two layout conversions are inverses of each other
static bool areInverseLayouts(DataLayoutAttr srcLayout,
                              DataLayoutAttr dstLayout,
                              DataLayoutAttr nextSrcLayout,
                              DataLayoutAttr nextDstLayout) {
  LDBG(srcLayout);
  LDBG(dstLayout);
  LDBG(nextSrcLayout);
  LDBG(nextDstLayout);
  return dstLayout == nextSrcLayout && srcLayout == nextDstLayout;
}

/// Pattern to eliminate consecutive inverse layout conversions
/// Example: convert(A, ND->NZ) followed by convert(B, NZ->ND) = identity
struct EliminateRedundantConversionPattern : public OpRewritePattern<
      ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceOp = op.getSource().getDefiningOp<ConvertLayoutOp>();
    if (!sourceOp)
      return rewriter.notifyMatchFailure(op, "source is not a ConvertLayoutOp");

    if (!areInverseLayouts(sourceOp.getSrcLayout(), sourceOp.getDstLayout(),
                           op.getSrcLayout(), op.getDstLayout()))
      return rewriter.notifyMatchFailure(
          op, "layouts are not inverse conversions");

    if (!sourceOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "source conversion has multiple uses");

    rewriter.replaceOp(op, sourceOp.getSource());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fold Empty + ConvertLayout into Empty with Target Layout
//===----------------------------------------------------------------------===//

struct FoldEmptyConvertLayoutPattern : public OpRewritePattern<ConvertLayoutOp> {
  FoldEmptyConvertLayoutPattern(MLIRContext *context)
    : OpRewritePattern(context) {
  }

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto emptyOp = op.getSource().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp)
      return rewriter.notifyMatchFailure(op, "source is not a tensor.empty");
    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(),
        op.getResult().getType(), ValueRange());
    rewriter.replaceOp(op, newEmptyOp.getResult());
    if (emptyOp.getResult().use_empty())
      rewriter.eraseOp(emptyOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fold tensor.cast + ConvertLayout into ConvertLayout
//===----------------------------------------------------------------------===//

/// Pattern to fold tensor.cast followed by convert_layout.
/// This eliminates unnecessary cast operations when convert_layout can work
/// directly with the original tensor.
/// Example:
///   %cast = tensor.cast %src : tensor<?x4x16x16xf16> to tensor<?x?x?x?xf16>
///   %result = hivm.hir.convert_layout %cast ... : (tensor<?x?x?x?xf16>) -> tensor<?x64xf16>
/// Becomes:
///   %result = hivm.hir.convert_layout %src ...  : (tensor<?x4x16x16xf16>) -> tensor<?x64xf16>
struct FoldCastConvertLayoutPattern : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto castOp = op.getSource().getDefiningOp<tensor::CastOp>();
    if (!castOp)
      return rewriter.notifyMatchFailure(op, "source is not a tensor.cast");

    // Check if the cast's source can be used directly with convert_layout
    auto originalSource = castOp.getSource();
    // Replace the convert_layout's operand with the original tensor from the cast
    rewriter.modifyOpInPlace(op, [&] {
      op.getSourceMutable().assign(originalSource);
    });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertLayoutOp Canonicalization
//===----------------------------------------------------------------------===//

void ConvertLayoutOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<EliminateRedundantConversionPattern>(context);
  results.add<FoldEmptyConvertLayoutPattern>(context);
  results.add<FoldCastConvertLayoutPattern>(context);
}

} // namespace mlir::hivm