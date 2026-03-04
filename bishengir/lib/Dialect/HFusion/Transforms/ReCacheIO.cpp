//===-------------------- ReCacheIO.h - recache io ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-recache-io"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_RECACHEIO
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::tensor;
using namespace mlir::hfusion;

namespace {

void recacheLoadOpToSliceUsers(PatternRewriter &rewriter,
                               hfusion::LoadOp loadOp,
                               const SmallVector<ExtractSliceOp> &sliceUsers) {
  for (ExtractSliceOp user : sliceUsers) {
    // set slice user source to hfusion.load source
    rewriter.modifyOpInPlace(
        user, [&]() { user->setOperands({loadOp.getInputs()[0]}); });
    // replace the uses of slice user with new hfusion.load
    rewriter.setInsertionPointAfter(user);
    Location loc = user->getLoc();
    auto emptyTensor = utils::createEmptyOp(rewriter, loc, user.getResult());
    auto newLoadOp = rewriter.create<hfusion::LoadOp>(
        loc, ValueRange{user->getResults()}, ValueRange{emptyTensor});
    rewriter.replaceAllUsesExcept(user.getResult(), newLoadOp.getResult(0),
                                  newLoadOp);
  }
}

/// For unalign ub access, adjust to access gm unalignly and access ub
/// e.g.
///   %tmp = hfusion.load %gmSrc
///   %res = tensor.extract_slice %tmp [offsets][][]
///   ... // other op use %tmp
/// => if offsets is not aligned
///   %tmp = hfusion.load %gmSrc
///   %tmp2 = tensor.extract_slice %gmSrc[offsets][][]
///   %newRes = hfusion.load %tmp2 // replace %res by %newRes
///   ... // other op use %tmp
struct ReCacheUnalignedAccessPattern
    : public OpRewritePattern<hfusion::LoadOp> {
  using OpRewritePattern<hfusion::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (loadOp->hasOneUse()) {
      return failure();
    }
    SmallVector<ExtractSliceOp> unalignedUsers = getUnalignedSliceUsers(loadOp);
    if (unalignedUsers.empty()) {
      return rewriter.notifyMatchFailure(
          loadOp, "expect at least one unaligned slice user for load op");
    }
    recacheLoadOpToSliceUsers(rewriter, loadOp, unalignedUsers);
    return success();
  }

private:
  SmallVector<ExtractSliceOp> getUnalignedSliceUsers(hfusion::LoadOp op) const {
    SmallVector<ExtractSliceOp> result;
    for (Operation *user : op->getUsers()) {
      auto sliceUser = dyn_cast<ExtractSliceOp>(user);
      if (!sliceUser) {
        continue;
      }
      if (!mlir::tensor::isOffsetBytesAligned(sliceUser, 32)) {
        result.push_back(sliceUser);
      }
    }
    return result;
  }
};

/// For exclusive slices from load, adjust to access gm after each slice,
/// to avoid ub out of bound access may caused by union extract_slice users.
///
/// input:
///   %load = hfusion.load %gmSrc : tensor<32xf32>
///   %slice0 = tensor.extract_slice %tmp [0][16][]
///   %slice1 = tensor.extract_slice %tmp [100][16][]
/// output:
///   %slice0 = tensor.extract_slice %gmSrc[0][16][]
///   %slice1 = tensor.extract_slice %gmSrc[100][16][]
///   %newLoad0 = hfusion.load %slice0
///   %newLoad1 = hfusion.load %slice1
struct ReCacheExclusiveSlicePattern : public OpRewritePattern<hfusion::LoadOp> {
  using OpRewritePattern<hfusion::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (loadOp->hasOneUse()) {
      return failure();
    }

    if (hasDynamicSliceUser(loadOp)) {
      return rewriter.notifyMatchFailure(
          loadOp,
          "do not recache for slice users with dynamic offsets or sizes");
    }

    auto sliceUsers = getSliceUsers(loadOp);
    if (sliceUsers.empty()) {
      return rewriter.notifyMatchFailure(loadOp,
                                         "no extract_slice users found");
    }

    if (!isExclusiveSlices(sliceUsers)) {
      return rewriter.notifyMatchFailure(
          loadOp, "only recache exclusive extract_slice users");
    }
    recacheLoadOpToSliceUsers(rewriter, loadOp, sliceUsers);
    return success();
  }

private:
  struct SliceRange {
    int64_t lowerBound{-1};
    int64_t upperBound{-1};
    bool operator<(const SliceRange &other) const {
      return lowerBound == other.lowerBound ? upperBound < other.upperBound
                                            : lowerBound < other.lowerBound;
    }
  };

  // slices are exclusive if they have exclusive ranges on at least one dim and
  // have the same range on other dims
  bool isExclusiveSlices(SmallVector<ExtractSliceOp> &slices) const {
    auto dimAndRanges = getSliceDimAndRanges(slices);
    if (0 == llvm::count_if(dimAndRanges, [](auto &&pair) {
          const auto &[dim, range] = pair;
          return range.size() != 1;
        })) {
      // no exclusive slices if each dim have the same slice ranges
      return false;
    }

    bool exclusive = true;
    for (auto &[dim, range] : dimAndRanges) {
      if (range.size() == 1) {
        continue;
      }
      exclusive = exclusive && isExclusiveSlices(range);
    }
    return exclusive;
  }

  // slices are exclusive if their slice ranges don't overlap on any dim
  bool isExclusiveSlices(const std::set<SliceRange> &sliceSet) const {
    SmallVector<SliceRange> slices{sliceSet.begin(), sliceSet.end()};
    llvm::sort(slices, [&](SliceRange lhs, SliceRange rhs) {
      return lhs.lowerBound < rhs.lowerBound;
    });

    int64_t rank = static_cast<int64_t>(slices.size());
    for (int64_t i = 1; i < rank; ++i) {
      if (slices[i - 1].upperBound > slices[i].lowerBound) {
        return false;
      }
    }
    return true;
  }

  std::set<SliceRange> getSliceRangesForDim(SmallVector<ExtractSliceOp> &slices,
                                            int64_t dim) const {
    std::set<SliceRange> sliceRanges;
    for (ExtractSliceOp slice : slices) {
      int64_t lowerBound = slice.getStaticOffset(dim);
      int64_t upperBound =
          slice.getStaticOffset(dim) + slice.getStaticSize(dim);
      sliceRanges.insert(SliceRange{lowerBound, upperBound});
    }
    return sliceRanges;
  }

  std::map<int64_t, std::set<SliceRange>>
  getSliceDimAndRanges(SmallVector<ExtractSliceOp> &slices) const {
    int64_t rank = slices.front().getSourceType().getRank();
    std::map<int64_t, std::set<SliceRange>> sliceRanges;
    for (int64_t i = 0; i < rank; ++i) {
      sliceRanges[i] = getSliceRangesForDim(slices, i);
    }
    return sliceRanges;
  }

  bool hasDynamicSliceUser(hfusion::LoadOp loadOp) const {
    return llvm::any_of(loadOp->getUsers(), [](Operation *op) {
      auto sliceUser = dyn_cast<tensor::ExtractSliceOp>(op);
      if (!sliceUser) {
        return false;
      }
      SmallVector<Value> dynamicOffsets;
      SmallVector<int64_t> staticOffsets;
      dispatchIndexOpFoldResults(sliceUser.getMixedOffsets(), dynamicOffsets,
                                 staticOffsets);
      SmallVector<Value> dynamicSizes;
      SmallVector<int64_t> staticSizes;
      dispatchIndexOpFoldResults(sliceUser.getMixedSizes(), dynamicSizes,
                                 staticSizes);
      return !dynamicOffsets.empty() || !dynamicSizes.empty();
    });
  }

  SmallVector<ExtractSliceOp> getSliceUsers(hfusion::LoadOp op) const {
    SmallVector<ExtractSliceOp> result;
    for (Operation *user : op->getUsers()) {
      auto sliceUser = dyn_cast<ExtractSliceOp>(user);
      if (!sliceUser) {
        continue;
      }
      result.push_back(sliceUser);
    }
    return result;
  }
};

struct ReCacheIOPass : public impl::ReCacheIOBase<ReCacheIOPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReCacheUnalignedAccessPattern>(context);
    patterns.add<ReCacheExclusiveSlicePattern>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::hfusion::createReCacheIO() {
  return std::make_unique<ReCacheIOPass>();
}
