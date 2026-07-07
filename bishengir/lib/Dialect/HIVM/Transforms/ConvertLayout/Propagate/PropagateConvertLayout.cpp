//===-------------------- PropagateConvertLayout.cpp ----------------------===//
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
// Move ConvertLayout before tensor copy
//===----------------------------------------------------------------------===//

/// Move a layout conversion from the result of a tensor copy to the copy
/// source. This handles the general UB-to-MMAD operand path after
/// InsertLoadStoreForMixCV has already analyzed memory scope. At this point
/// every `mmadL1` operand must be separated from its producer by one DMA op:
///   - tensor-based load, when data comes from Vector through GM;
///   - pure-memref load, when data comes directly from GM;
///   - tensor-based copy, when data comes from Vector through UB.
///
/// For the tensor-copy case we must swap `convert_layout` with `hivm.copy`.
/// `mmadL1` consumes Fractal/NZ input, while the ND->Fractal conversion should
/// still be decomposed to vector-side transpose immediately before the copy.
/// The moved conversion is therefore marked as not propagating up, keeping the
/// vtranspose close to the copy/consumer boundary instead of drifting farther
/// toward the original producer.
///
/// Before:
///   %dst = tensor.empty() : tensor<16x16xf32>
///   %copy = hivm.hir.copy ins(%arg : tensor<16x16xf32>)
///                         outs(%dst : tensor<16x16xf32>)
///                         -> tensor<16x16xf32>
///   %converted = hivm.hir.convert_layout %copy
///       : tensor<16x16xf32> -> tensor<2x1x16x8xf32>
///
/// After:
///   %converted_src = hivm.hir.convert_layout %arg
///       : tensor<16x16xf32> -> tensor<2x1x16x8xf32>
///   %new_dst = tensor.empty() : tensor<2x1x16x8xf32>
///   %new_copy = hivm.hir.copy ins(%converted_src)
///                            outs(%new_dst)
///                            -> tensor<2x1x16x8xf32>
struct MoveConvertLayoutBeforeTensorCopy
    : public OpRewritePattern<ConvertLayoutOp> {
  explicit MoveConvertLayoutBeforeTensorCopy(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ConvertLayoutOp convertLayout,
                                PatternRewriter &rewriter) const override {
    if (!isPropagatingUp(convertLayout))
      return rewriter.notifyMatchFailure(convertLayout, "not propagating-up");

    auto copyOp = convertLayout.getSource().getDefiningOp<hivm::CopyOp>();
    if (!copyOp)
      return rewriter.notifyMatchFailure(convertLayout,
                                         "source is not a CopyOp");
    if (!copyOp.hasPureTensorSemantics() ||
        !convertLayout.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(convertLayout,
                                         "copy is not tensor-based");

    if (copyOp.getPadMode() || copyOp.getPadValue())
      return rewriter.notifyMatchFailure(convertLayout,
                                         "padded copy is not supported");

    PatternRewriter::InsertionGuard guard(rewriter);
    Location loc = convertLayout.getLoc();
    auto resultTensorType = cast<RankedTensorType>(convertLayout.getType());

    rewriter.setInsertionPoint(copyOp);
    auto newConvert = rewriter.create<ConvertLayoutOp>(
        loc, resultTensorType, copyOp.getSrc(),
        convertLayout.getSrcLayoutAttr(), convertLayout.getDstLayoutAttr(),
        convertLayout.getMixedOutputShape());
    markAsNotPropagatingUp(rewriter, newConvert);

    auto newDst = rewriter.create<tensor::EmptyOp>(
        loc, convertLayout.getMixedOutputShape(),
        resultTensorType.getElementType());
    // Warning: if the original copy result has other users, this clones the
    // converted copy path for the convert_layout user and leaves the old copy.
    auto newCopy = rewriter.create<hivm::CopyOp>(
        loc, resultTensorType, newConvert.getResult(), newDst);
    newCopy->setAttrs(copyOp->getAttrs());

    rewriter.replaceOp(convertLayout, newCopy.getResultTensor());
    if (copyOp->use_empty())
      rewriter.eraseOp(copyOp);
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
    // Run these prioritized copy-boundary rewrites before the generic
    // propagation patterns, so later patterns see the intended converted value.
    RewritePatternSet firstPatterns(context);
    firstPatterns.add<MoveConvertLayoutToSourceOfUBToCBufCopy>(context);
    firstPatterns.add<MoveConvertLayoutBeforeTensorCopy>(context);
    if (failed(applyPatternsGreedily(module, std::move(firstPatterns))))
      signalPassFailure();

    RewritePatternSet patterns(context);
    // PropagateConvertLayoutInternalOptions class is defined in
    // `ConvertLayoutUtils.h` and is different with
    // PropagateConvertLayoutOptions from the `HIVM/Transforms/Passes.td`
    PropagateConvertLayoutInternalOptions options;
    options.allowAgnosticOps = allowAgnosticOps;
    populateConvertLayoutExtractSlice(patterns, context);
    // Enable this only for vcast
    populateConvertLayoutElementwise(patterns, context, options);
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