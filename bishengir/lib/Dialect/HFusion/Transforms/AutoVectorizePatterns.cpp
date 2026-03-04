//===- AutoVectorizePatterns.cpp - Auto-vectorization cleanup patterns -----===//
//
// Part of the BiShengIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements cleanup rewrite patterns used by the auto-vectorization
// passes. In particular, it provides patterns to simplify vector-related IR
// produced during vectorization, such as eliminating redundant vector index
// casts when it is safe to do so.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/AutoVectorizePatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
using namespace mlir;

namespace {

// This pattern is driven by the following test cases:
//
// 1) vector<i32> -> vector<index> index_cast feeding ONLY vector.gather
//    should be eliminated:
//
//    %idx_i = arith.index_cast %idx : vector<64xi32> to vector<64xindex>
//    %v = vector.gather %tbl[%c0] [%idx_i] ...
//
//    After rewrite, vector.gather directly consumes vector<64xi32> indices
//    and the index_cast must disappear.
template <typename CastOpT>
struct ElimVectorIndexCastPattern : OpRewritePattern<CastOpT> {
  using OpRewritePattern<CastOpT>::OpRewritePattern;

  static bool isVecIntToVecIndex(Type srcTy, Type dstTy) {
    auto srcV = dyn_cast<VectorType>(srcTy);
    auto dstV = dyn_cast<VectorType>(dstTy);
    if (!srcV || !dstV)
      return false;
    if (srcV.getShape() != dstV.getShape())
      return false;
    auto srcElem = dyn_cast<IntegerType>(srcV.getElementType());
    if (!srcElem)
      return false;
    if (!isa<IndexType>(dstV.getElementType()))
      return false;
    return true;
  }

  LogicalResult matchAndRewrite(CastOpT op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getIn();
    Value dst = op.getOut();
    if (!isVecIntToVecIndex(src.getType(), dst.getType()))
      return failure();

    const bool allUsersAreGather =
        llvm::all_of(op->getUsers(), [](const auto &user) {
          return isa<vector::GatherOp>(user);
        });

    if (!allUsersAreGather)
      return failure();

    rewriter.replaceOp(op, src);
    return success();
  }
};

} // namespace

namespace mlir {
namespace hfusion {

void populateAutoVectorizeCleanUpPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ElimVectorIndexCastPattern<arith::IndexCastOp>>(ctx);
  patterns.add<ElimVectorIndexCastPattern<arith::IndexCastUIOp>>(ctx);
}

} // namespace hfusion
} // namespace mlir
