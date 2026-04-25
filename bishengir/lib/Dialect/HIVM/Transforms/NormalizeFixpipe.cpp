//===------------- NormalizeFixpipe.cpp - normalize fixpipe op ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bishengir-c/Dialect/HIVM.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZEFIXPIPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"


static FailureOr<hivm::AddressSpaceAttr> getHIVMAddressSpaceAttr(Type type) {
  if (!type) {
    return failure();
  }
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  if (!memRefType) {
    return failure();
  }
  if (!memRefType.getMemorySpace()) {
    return failure();
  }
  auto scopeAttr = dyn_cast<hivm::AddressSpaceAttr>(memRefType.getMemorySpace());
  if (!scopeAttr) {
    return failure();
  }
  return success(scopeAttr);
}

static FailureOr<hivm::AddressSpace> getHIVMAddressSpace(Type type) {
  auto scopeAttr = getHIVMAddressSpaceAttr(type);
  if (failed(scopeAttr)) {
    return failure();
  }
  return scopeAttr->getAddressSpace();
}

struct Normalize32To16UBFixpipe : public OpRewritePattern<hivm::FixpipeOp> {
  using OpRewritePattern<hivm::FixpipeOp>::OpRewritePattern;
  
  static LogicalResult maySplitFixpipe(hivm::FixpipeOp op) {
    if (op.getDmaMode() == hivm::FixpipeDMAMode::NZ2DN) {
      return failure();
    }

    auto dstTy = dyn_cast<MemRefType>(op.getDst().getType());
    if (!dstTy)
      return failure();

    auto space = getHIVMAddressSpace(dstTy);
    if (space != hivm::AddressSpace::UB) {
      return failure();
    }

    auto maybeAlloc = hivm::traceDefOp<memref::AllocOp>(op.getDst());
    if (!maybeAlloc) {
      return failure();
    }

    auto ty = cast<memref::AllocOp>(*maybeAlloc).getType();
    if (!ty.hasStaticShape()) {
      return failure();
    }

    auto shape = ty.getShape();
    int64_t base = op.getDst().getDefiningOp<memref::SubViewOp>() ? 1 : 0;

    auto legal = [&](int64_t dim, int64_t align) {
      return dim < static_cast<int64_t>(shape.size()) && shape[dim] % align == 0;
    };

    return LogicalResult::success(legal(base, 2) || legal(base + 1, 32));
  }

  LogicalResult matchAndRewrite(hivm::FixpipeOp op, PatternRewriter &rewriter) const override {
    if (isa<TensorType>(op.getDst().getType())) {
      return failure();
    }
    if (op.getPreQuant() != hivm::FixpipePreQuantMode::F322BF16 && 
        op.getPreQuant() != hivm::FixpipePreQuantMode::F322F16) {
      // failures detected only in these cases
      return failure();
    }
    auto src = op.getSrc();
    auto dst = op.getDst();
    if (!src.getType().hasRank() || !dst.getType().hasRank()) {
      return failure();
    }

    // conservatively estimate will it be splitted in TileAndBindSubBlock or not
    if (failed(maySplitFixpipe(op))) {
      return failure();
    }

    op.setPreQuant(hivm::FixpipePreQuantMode::NO_QUANT);
    rewriter.setInsertionPoint(op);
    // can't reuse fixpipe from cube side since there's no way in hivm to perform 
    // data movement from l1 to l0c bypassing cube, so doing cast in ub
    auto tmpBufferType = MemRefType::get(
      src.getType().getShape(),
      src.getType().getElementType(),
      mlir::MemRefLayoutAttrInterface{},
      hivm::AddressSpaceAttr::get(rewriter.getContext(), 
          mlir::hivm::AddressSpace::UB)
    );
    auto tmpBuf = rewriter.create<memref::AllocOp>(op.getLoc(), tmpBufferType);
    op->setOperand(1, tmpBuf);

    rewriter.setInsertionPointAfter(op);
    rewriter.create<hivm::VCastOp>(op.getLoc(), 
      TypeRange{}, static_cast<Value>(tmpBuf), dst,
      hivm::RoundMode::RINT);
    return success();
  }
};

struct NormalizeFixpipePass : public impl::NormalizeFixpipeBase<NormalizeFixpipePass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<Normalize32To16UBFixpipe>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir


std::unique_ptr<mlir::Pass> mlir::hivm::createNormalizeFixpipePass() {
  return std::make_unique<NormalizeFixpipePass>();
}