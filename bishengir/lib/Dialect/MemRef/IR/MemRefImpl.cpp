//===- MemRefImpl.cpp -----------------------------------------------------===//
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

#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#include "mlir/Dialect/Utils/ExpandShapeUtils.h"
#endif
#include "mlir/IR/TypeUtilities.h"

namespace {

using namespace mlir;

/// Return the canonical type of the result of a ReinterpretCastOp.
struct ReinterpretCastReturnTypeCanonicalizer {
  MemRefType operator()(memref::ReinterpretCastOp op,
                        ArrayRef<OpFoldResult> mixedOffsets,
                        ArrayRef<OpFoldResult> mixedSizes,
                        ArrayRef<OpFoldResult> mixedStrides) {
    SmallVector<int64_t> staticOffsets;
    SmallVector<int64_t> staticSizes;
    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicOffsets;
    SmallVector<Value> dynamicSizes;
    SmallVector<Value> dynamicStrides;
    dispatchIndexOpFoldResults(mixedOffsets, dynamicOffsets, staticOffsets);
    dispatchIndexOpFoldResults(mixedSizes, dynamicSizes, staticSizes);
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    auto sourceMemRefType = cast<BaseMemRefType>(op.getSource().getType());
    auto resType =
        MemRefType::get(staticSizes, sourceMemRefType.getElementType(),
                        StridedLayoutAttr::get(sourceMemRefType.getContext(),
                                               staticOffsets[0], staticStrides),
                        sourceMemRefType.getMemorySpace());
    return resType;
  }
};

/// A canonicalizer wrapper to replace ReinterpretCastOps.
struct ReinterpretCastCanonicalizer {
  void operator()(PatternRewriter &rewriter, memref::ReinterpretCastOp op,
                  memref::ReinterpretCastOp newOp) {
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), newOp);
  }
};

/// 1. fold `copy A B; copy B C` to `copy A C`
/// 2. fold `copy A B; B0 = reshape(B); copy B0 C`
///    to   `A0 = reshape(A); copy A0 C;`
///    if `B` only has one user `B0 = reshape(B)`
struct FoldRedundantCopy : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSource();
    Value dst = copyOp.getTarget();
    if (src == dst)
      return failure();

    SmallVector<Operation *> reshapeTrace;
    auto copyMaybe = traceSingleReshapeUtilCopy(dst, copyOp, reshapeTrace);
    if (!copyMaybe.has_value()) {
      // avoid fold `copy A B; not_reshape_use(B); copy B C` to `copy A C`
      return failure();
    }

    memref::CopyOp copyFromDst = copyMaybe.value();
    if (!copyOp->isBeforeInBlock(copyFromDst)) {
      // avoid fold `copy B C; copy A B` to `copy A C`, wrong order
      return failure();
    }

    DenseSet<Operation *> skipCopyOps{copyOp, copyFromDst};
    if (!isUsersMemoryEffectFree(src, skipCopyOps) ||
        !isUsersMemoryEffectFree(copyFromDst.getTarget(), skipCopyOps)) {
      // 1. avoid fold `copy A B; memory_effect(A); copy B C` to `copy A C`
      // 2. avoid fold `copy A B; memory_effect(C); copy B C` to `copy A C`
      return failure();
    }

    // create new copy source following the reshape trace on old copy source
    Value reshapeFromSrc = src;
    Location loc = copyOp.getLoc();
    for (Operation *op : reshapeTrace) {
      if (auto collapse = dyn_cast<memref::CollapseShapeOp>(op)) {
        reshapeFromSrc = rewriter.create<memref::CollapseShapeOp>(
            loc, reshapeFromSrc, collapse.getReassociationIndices());
        continue;
      }
      if (auto expand = dyn_cast<memref::ExpandShapeOp>(op)) {
        reshapeFromSrc = rewriter.create<memref::ExpandShapeOp>(
            loc, expand.getResultType(), reshapeFromSrc,
            expand.getReassociationIndices());
        continue;
      }
      llvm_unreachable("invalid reshape op");
    }

    rewriter.setInsertionPointAfter(copyFromDst);
    rewriter.create<memref::CopyOp>(copyFromDst.getLoc(), reshapeFromSrc,
                                    copyFromDst.getTarget());

    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(copyFromDst);
    return success();
  }

  bool isUsersMemoryEffectFree(Value src,
                               const DenseSet<Operation *> &skipCopyOps) const {
    return llvm::all_of(src.getUsers(), [&](Operation *user) {
      if (skipCopyOps.contains(user)) {
        return true;
      }
      if (isMemoryEffectFree(user)) {
        return true;
      }
      // make sure all op with memory effect exist before or after skipCopyOps
      bool userBeforeAll = llvm::all_of(skipCopyOps, [&](Operation *skipOp) {
        return user->isBeforeInBlock(skipOp);
      });
      bool userAfterAll = llvm::all_of(skipCopyOps, [&](Operation *skipOp) {
        return skipOp->isBeforeInBlock(user);
      });
      return userBeforeAll || userAfterAll;
    });
  }

  // Returns the single user of src except for `skipOp`
  // If there are more than one users except for `skipOp`, return nullptr
  Operation *getSingleUser(Value src, Operation *skipOp) const {
    Operation *user = nullptr;
    for (OpOperand &use : src.getUses()) {
      Operation *curUser = use.getOwner();
      if (curUser == skipOp) {
        // skip the origin user to avoid confusion
        continue;
      }
      if (user != nullptr) {
        // there should be only one user other than the origin user
        return nullptr;
      }
      user = curUser;
    }
    return user;
  }

  // Trace reshape op starting from src, util find a interested memref::CopyOp.
  // If there are multiple users along the way, or no target copy, or other op
  // type other than reshape/copy, return nullptr
  std::optional<memref::CopyOp>
  traceSingleReshapeUtilCopy(Value src, Operation *skipOp,
                             SmallVector<Operation *> &useChain) const {
    Operation *user = getSingleUser(src, skipOp);
    if (user == nullptr) {
      // no fold is needed if there is no other users
      return std::nullopt;
    }

    if (isa<memref::CollapseShapeOp>(user) ||
        isa<memref::ExpandShapeOp>(user)) {
      useChain.push_back(user);
      return traceSingleReshapeUtilCopy(user->getResult(0), user, useChain);
    }
    if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
      if (copyOp.getSource() == src && copyOp.getTarget() != src) {
        // only allow copy from src to other dst
        return copyOp;
      }
    }
    return std::nullopt;
  }
};

struct CanonicalizeAllocToTensor : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return failure();
    auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(*op->user_begin());
    if (!toTensorOp)
      return failure();
    auto tensorType = toTensorOp.getType();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(toTensorOp, tensorType.getShape(), tensorType.getElementType());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir {
namespace memref {

Value createMemRefAllocOpWithTargetElemType(
    OpBuilder &builder, Location loc, Value source, Type targetElemType,
    std::optional<MemRefLayoutAttrInterface> layout) {
  auto shapedType = cast<ShapedType>(source.getType());
  ArrayRef<int64_t> staticShapes = shapedType.getShape();
  llvm::SmallVector<Value, 2> dynamicSizes;
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      Operation *dynDimOp = builder.create<memref::DimOp>(loc, source, i);
      dynamicSizes.push_back(dynDimOp->getResults()[0]);
    }
  }
  MemRefType origType = cast<MemRefType>(shapedType);
  MemRefType newMemTy = MemRefType::get(
      staticShapes, targetElemType,
      layout.has_value() ? layout.value() : origType.getLayout(),
      origType.getMemorySpace());
  return builder.create<memref::AllocOp>(loc, newMemTy, dynamicSizes);
}

Value createMemRefAllocOp(OpBuilder &builder, Location loc, Value source) {
  auto elementType = mlir::getElementTypeOrSelf(source);
  auto emptyOp =
      createMemRefAllocOpWithTargetElemType(builder, loc, source, elementType);
  return emptyOp;
}

void getExtendedCanonicalizationPatterns(mlir::RewritePatternSet &results) {
  auto *context = results.getContext();
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  results.add<OpWithOffsetSizesAndStridesConstantArgumentFolder<
      ReinterpretCastOp, ReinterpretCastReturnTypeCanonicalizer,
      ReinterpretCastCanonicalizer>>(context,
                                     "ReinterpretCastConstantArgumentFolder");
  results.add<FoldRedundantCopy, CanonicalizeAllocToTensor>(context);
  results.add<FoldConstantDimOfOutputShape<ExpandShapeOp, CastOp>>(context);
#endif
}

} // namespace memref
} // namespace mlir
