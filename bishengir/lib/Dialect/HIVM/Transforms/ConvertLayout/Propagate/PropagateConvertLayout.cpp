//===-------------------- PropagateConvertLayout.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/ConvertLayoutUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hivm-propagate-convert-layout"

namespace mlir {
#define GEN_PASS_DEF_PROPAGATECONVERTLAYOUT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {

//===----------------------------------------------------------------------===//
// Move ConvertLayout from UB->CBuf copy destination to its source
//===----------------------------------------------------------------------===//

/// Patter to move ConvertLayout operation from UB->CBuf copy destination to
/// its source to be implemented on Vector device.
///
/// Before:
///   %alloc = memref.alloc {alloc_shape<cbuf>}
///   %spacecast = memref.memory_space_cast %alloc
///   hivm.hir.copy %source %spacecast
///   %toTensor = bufferization.to_tensor %spacecast
///   %convertLayout = hivm.hir.convert_layout %toTensor {converted_shape}
///
/// After:
///   %newAlloc = memref.alloc {converted_shape<cbuf>}
///   %newSpacecast = memref.memory_space_cast %newAlloc
///   %convertLayout = hivm.hir.convert_layout %source {converted_shape}
///   hivm.hir.copy %convertLayout %newSpacecast
///   %newToTensor = bufferization.to_tensor %newSpacecast
///
/// Uses of %convertLayout replaced with %newToTensor
///
struct MoveConvertLayoutToSourceOfUBToCBufCopy
    : public OpRewritePattern<ConvertLayoutOp> {
  explicit MoveConvertLayoutToSourceOfUBToCBufCopy(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp convertLayout,
                                PatternRewriter &rewriter) const override {
    Operation *mmad = nullptr;
    for (auto *user : convertLayout->getUsers()) {
      mmad = dyn_cast<hivm::MmadL1Op>(user);
      if (mmad) {
        break;
      }
    }
    if (!mmad) {
      return failure();
    }

    auto toTensorDef = convertLayout.getSource().getDefiningOp();
    if (!toTensorDef) {
      return failure();
    }
    auto toTensor = dyn_cast<bufferization::ToTensorOp>(toTensorDef);
    if (!toTensor) {
      return failure();
    }

    auto spaceCastDef = toTensor.getMemref().getDefiningOp();
    if (!spaceCastDef) {
      return failure();
    }
    auto spaceCast = dyn_cast<memref::MemorySpaceCastOp>(spaceCastDef);
    if (!spaceCast) {
      return failure();
    }

    auto allocDef = spaceCast.getSource().getDefiningOp();
    if (!allocDef) {
      return failure();
    }
    auto alloc = dyn_cast<memref::AllocOp>(allocDef);
    if (!alloc) {
      return failure();
    }

    auto maybeAllocSpace = hivm::getOptionalHIVMAddressSpace(alloc.getType());
    if (!maybeAllocSpace.has_value() ||
        (maybeAllocSpace.value() != hivm::AddressSpace::L1)) {
      return failure();
    }

    Operation *copyOpt = nullptr;
    for (auto *user : spaceCast->getUsers()) {
      copyOpt = dyn_cast<hivm::CopyOp>(user);
      if (copyOpt) {
        break;
      }
    }
    if (!copyOpt) {
      return failure();
    }
    auto copy = dyn_cast<hivm::CopyOp>(copyOpt);

    // Pattern matched, now code will be rewritten

    auto loc = alloc.getLoc();
    rewriter.setInsertionPointAfter(alloc);

    auto shape = convertLayout.getStaticOutputShape();
    auto type = alloc.getMemref().getType().getElementType();
    auto layout = MemRefLayoutAttrInterface();
    auto space = alloc.getMemref().getType().getMemorySpace();
    auto memRefType = MemRefType::get(shape, type, layout, space);

    auto newAlloc = rewriter.create<memref::AllocOp>(loc, memRefType);
    auto newSpaceCast =
        rewriter.create<memref::MemorySpaceCastOp>(loc, memRefType, newAlloc);
    auto newToTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, newSpaceCast, toTensor.getRestrict(), toTensor.getWritable());

    rewriter.modifyOpInPlace(convertLayout, [&] {
      convertLayout.getSourceMutable().assign(copy.getSource());
    });

    rewriter.moveOpBefore(convertLayout, copy);
    rewriter.modifyOpInPlace(copy, [&] {
      copy.getSrcMutable().assign(convertLayout->getResult(0));
      copy.getDstMutable().assign(newSpaceCast->getResult(0));
    });

    SmallPtrSet<Operation *, 4> except;
    except.insert(copy);
    rewriter.replaceAllUsesExcept(convertLayout, newToTensor, except);

    markAsNotPropagatingUp(rewriter, convertLayout);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct PropagateConvertLayoutPass
    : public impl::PropagateConvertLayoutBase<PropagateConvertLayoutPass> {
  using Base::Base;
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet firstPatterns(context);
    firstPatterns.add<MoveConvertLayoutToSourceOfUBToCBufCopy>(context);
    if (failed(applyPatternsGreedily(module, std::move(firstPatterns))))
      signalPassFailure();

    RewritePatternSet patterns(context);
    // PropagateConvertLayoutInternalOptions class is defined in
    // `ConvertLayoutUtils.h` and is different with
    // PropagateConvertLayoutOptions from the `HIVM/Transforms/Passes.td`
    PropagateConvertLayoutInternalOptions options;
    options.allowAgnosticOps = allowAgnosticOps;
    if (enableElementwisePropagate)
      populateConvertLayoutElementwise(patterns, context, options);
    populateConvertLayoutExtractSlice(patterns, context);
    // Enable this only for vcast
    populateConvertLayoutScfIf(patterns, context);
    populateConvertLayoutScfFor(patterns, context);
    populateConvertLayoutScfWhile(patterns, context);
    populateHoistConvertLayout(patterns, context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createPropagateConvertLayoutPass(
    const PropagateConvertLayoutOptions &options) {
  return std::make_unique<PropagateConvertLayoutPass>(options);
}