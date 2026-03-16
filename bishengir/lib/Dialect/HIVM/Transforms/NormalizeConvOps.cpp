//===- NormalizeConvOps.cpp - normalize hivm conv ops ---------------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"

#include <memory>

namespace mlir {
#define GEN_PASS_DEF_NORMALIZECONVOPS
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-normalize-convops"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
static constexpr llvm::StringLiteral outputAlreadyNormalized =
    "outputAlreadyNormalized";
} // namespace

namespace {
inline RoundModeAttr getRoundAttr(mlir::OpBuilder &b, Type srcType,
                                  Type dstType) {
  return hivm::RoundModeAttr::get(
      b.getContext(),
      mlir::utils::selectRoundMode<hivm::RoundMode>(srcType, dstType));
}

struct NormalizeConvOpsPass
    : public impl::NormalizeConvOpsBase<NormalizeConvOpsPass> {
  using Base::Base;
  void runOnOperation() override;
};

/// This pattern transforms bf16/fp16 output of Conv1dL1 to fp32
/// and then cast back
template <typename TargetType>
struct NormalizeConvResultTypePattern
    : public OpRewritePattern<hivm::Conv1DL1Op> {
public:
  using OpRewritePattern<hivm::Conv1DL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::Conv1DL1Op op,
                                PatternRewriter &rewriter) const override {
    
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto resultType =
        dyn_cast<RankedTensorType>(op.getResultTensors()[0].getType());
    if (!resultType)
      return failure();

    auto elemType = resultType.getElementType();
    if (!isa<TargetType>(elemType)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto fp32Ty = rewriter.getF32Type();

    Value bias = op.getBias();
    Value newBias = bias;

    if (bias && elemType.isBF16()) {
      auto biasType = mlir::cast<RankedTensorType>(bias.getType());
      auto biasShape = biasType.getShape();
      auto biasElemType = biasType.getElementType();

      auto biasCastType = RankedTensorType::get(biasShape, fp32Ty);
      auto biasCastInit = 
          rewriter.create<tensor::EmptyOp>(loc, biasShape, fp32Ty);
      auto biasRoundAttr = getRoundAttr(rewriter, biasElemType, fp32Ty);

      newBias = rewriter
                    .create<hivm::VCastOp>(
                        loc, TypeRange{biasCastType}, ValueRange{bias},
                        ValueRange{biasCastInit.getResult()}, Value(),
                        biasRoundAttr, hivm::TypeFnAttr{})
                    ->getResult(0);
    }

    SmallVector<int64_t> shape(resultType.getShape().begin(),
                               resultType.getShape().end());

    auto fp32ResultType = RankedTensorType::get(shape, fp32Ty);
    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, fp32Ty);

    auto input = op.getInput();
    auto weight = op.getWeight();
    auto initCondition = op.getInitCondition();
    auto padding = op.getPaddingAttr();
    auto groups = op.getGroupsAttr();

    auto newConv =
        rewriter.create<hivm::Conv1DL1Op>(loc, TypeRange{fp32ResultType},
                                          input,         // input
                                          weight,        // weight
                                          newBias,       // bias
                                          init,          // init
                                          initCondition, // init_condition
                                          ValueRange{},  // sync_related_args
                                          padding,       // padding
                                          groups         // groups
        );

    auto castInit = rewriter.create<tensor::EmptyOp>(loc, shape, elemType);
    auto roundAttr = getRoundAttr(rewriter, fp32Ty, elemType);

    auto castResult =
        rewriter
            .create<hivm::VCastOp>(loc, TypeRange{castInit.getType()},
                                   ValueRange{newConv.getResultTensors()[0]},
                                   ValueRange{castInit.getResult()},
                                   /*temp_buffer=*/Value(), roundAttr,
                                   hivm::TypeFnAttr{})
            ->getResult(0);

    rewriter.replaceOp(op, castResult);

    return success();
  }
};

/// This pattern decomposes Conv1dL1 with bias into Conv1dL1(no bias) + vadd.
/// Vadd is inserted to different position (after Conv1dL1 + vcast or directly
/// after Conv1dL1) according to different dtype
template <typename TargetType> 
struct DecomposeConv1dWithBiasPattern
    : public OpRewritePattern<hivm::Conv1DL1Op> {
public:
  using OpRewritePattern<hivm::Conv1DL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::Conv1DL1Op op,
                                PatternRewriter &rewriter) const override {

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    Value bias = op.getBias();
    if (!bias)
      return failure();

    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType) {
      return failure();
    }

    auto biasElemType = biasType.getElementType();
    if (!isa<TargetType>(biasElemType)) {
      return failure();
    }

    auto resultType =
        dyn_cast<RankedTensorType>(op.getResultTensors()[0].getType());
    if (!resultType)
      return failure();

    int64_t rank = resultType.getRank();
    if (rank != 2 && rank != 3)
      return failure();

    Location loc = op.getLoc();

    auto input = op.getInput();
    auto weight = op.getWeight();
    auto initCondition = op.getInitCondition();
    auto padding = op.getPaddingAttr();
    auto groups = op.getGroupsAttr();

    auto elemType = resultType.getElementType();
    SmallVector<int64_t> shape(resultType.getShape().begin(),
                               resultType.getShape().end());

    auto convNoBiasInit =
        rewriter.create<tensor::EmptyOp>(loc, shape, elemType);

    auto newConv =
        rewriter.create<hivm::Conv1DL1Op>(loc, TypeRange{resultType},
                                          input,              // input
                                          weight,             // weight
                                          /* bias */ Value(), // remove bias
                                          convNoBiasInit,     // init
                                          initCondition,      // init_condition
                                          ValueRange{}, // sync_related_args
                                          padding,      // padding
                                          groups        // groups
        );

    Value convResult = newConv.getResultTensors()[0];

    int64_t oC =
        (rank == 2) ? resultType.getDimSize(0) : resultType.getDimSize(1);

    SmallVector<int64_t> expandedShape;
    SmallVector<SmallVector<int64_t, 2>> reassoc;

    if (rank == 2) {
      expandedShape = {oC, 1};
      reassoc = {{0, 1}};
    } else {
      expandedShape = {1, oC, 1};
      reassoc = {{0, 1, 2}};
    }

    auto expandedType = RankedTensorType::get(expandedShape, biasElemType);
    auto expandedBias = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedType, bias, reassoc);

    auto convResultElemType = 
        dyn_cast<RankedTensorType>(convResult.getType()).getElementType();
    
    if (convResultElemType != biasElemType) {
      return rewriter.notifyMatchFailure(
          op, "bias type mismatch with convResult type");
    }

    SmallVector<int64_t> broadcastDims;
    if (rank == 2)
      broadcastDims = {1};
    else
      broadcastDims = {0, 2};

    auto broadcastAttr = rewriter.getDenseI64ArrayAttr(broadcastDims);

    auto vaddInit = 
        rewriter.create<tensor::EmptyOp>(loc, shape, convResultElemType);
    
    auto vadd = rewriter.create<hivm::VAddOp>(
        loc, TypeRange{vaddInit.getType()},
        ValueRange{convResult, expandedBias}, ValueRange{vaddInit}, Value(),
        nullptr, broadcastAttr);

    rewriter.replaceOp(op, vadd->getResult(0));

    return success();
  }
};

/// This pattern normalizes the output layout of hivm::Conv1DL1Op by fusing
/// batch and group dimensions and aligning width and per-group channels,
/// then restoring the result back to the user-visible layout.
///
/// Motivation:
///   The original Conv1DL1 produces output in logical layout:
///     - without batch: [oC, oW]
///     - with batch:    [B, oC, oW]
///   For hardware execution, batch and group are fused together, and the
///   output layout is required to be aligned:
///     - oW aligned to FRACTAL_BLOCK_NUM
///     - oCPerGroup aligned to FRACTAL_BLOCK_NUM
///
/// Rewrite overview:
///   1. Rewrite Conv1DL1 to produce a fused & aligned layout:
///        [oWCeil, fusedOCCeil]
///      where:
///        - fusedOC     = B * oC
///        - fusedOCCeil = B * ceil(oC / groups, FRACTAL_BLOCK_NUM) * groups
///        - oWCeil      = ceil(oW, FRACTAL_BLOCK_NUM)
///
///   2. Post-process the aligned layout:
///      - Slice away padding on width and channels.
///      - Transpose the result to get [fusedOC, oW].
///      - For grouped and unaligned channels, remove per-group padding by
///        per-group slice + insert.
///
///   3. Restore user-visible layout if batch exists:
///        [fusedOC, oW] -> [B, oC, oW]
///
/// Semantics:
///   out[(n), c, w] = conv1d(input, weight, bias)[(n), c, w]
///   This pattern only changes the physical layout for alignment and
///   hardware execution, and then restores the logical layout expected by
///   users. The numerical semantics are unchanged.
///
/// In short:
///   user layout
///     -> fused & aligned layout (for hardware)
///     -> sliced / transposed / reshaped back to user layout
struct NormalizeConvOutputPattern : public OpRewritePattern<hivm::Conv1DL1Op> {
public:
  using OpRewritePattern<hivm::Conv1DL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::Conv1DL1Op op,
                                PatternRewriter &rewriter) const override {
    
    if (op->getAttr(outputAlreadyNormalized)) {
      return failure();
    }

    auto input = op.getInput();
    auto weight = op.getWeight();
    auto bias = op.getBias();
    auto init = op.getInit();
    auto initCondition = op.getInitCondition();
    auto padding = op.getPaddingAttr();
    auto groups = op.getGroupsAttr();
    int64_t groupsVal = groups.getInt();

    auto convResult = op.getResultTensors()[0];
    auto resultType = dyn_cast<RankedTensorType>(convResult.getType());
    if (!resultType || (resultType.getRank() != 2 && resultType.getRank() != 3)) {
      return failure();
    }

    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!initType || (initType.getRank() != 2 && initType.getRank() != 3)) {
      return failure();
    }

    bool hasBatch = resultType.getRank() == 3;

    int64_t batch = 1;
    int64_t oC, oW;

    if (!hasBatch) {
      // [oC, oW]
      oC = resultType.getDimSize(0);
      oW = resultType.getDimSize(1);
    } else {
      // [B, oC, oW]
      batch = resultType.getDimSize(0);
      oC = resultType.getDimSize(1);
      oW = resultType.getDimSize(2);
    }

    int64_t oCPerGroup = oC / groupsVal;
    int64_t oCPerGroupCeil = CEIL_FACTOR(oCPerGroup, utils::FRACTAL_BLOCK_NUM);
    int64_t oCCeil = oCPerGroupCeil * groupsVal;

    bool isOCPerGroupAligned = (oCPerGroup % utils::FRACTAL_BLOCK_NUM == 0);
    int64_t oWCeil = CEIL_FACTOR(oW, utils::FRACTAL_BLOCK_NUM);
    int64_t fusedOC = batch * oC;
    int64_t fusedOCCeil = batch * oCCeil;
    int64_t fusedGroupsVal = batch * groupsVal;

    auto elementType = resultType.getElementType();
    Location loc = op.getLoc();

    SmallVector<int64_t> newShape{oWCeil, fusedOCCeil};
    auto newResultType = RankedTensorType::get(newShape, elementType);
    Value newEmpty =
 	      rewriter.create<tensor::EmptyOp>(loc, newShape, elementType);

    // === create new ConvOp with result of new shape ===
    auto newConvOp = rewriter.create<hivm::Conv1DL1Op>(
        loc,           // location
        newResultType, // result type: [oWCeil, fusedOCCeil]
        input,         // input
        weight,        // weight
        bias,          // bias
        newEmpty,      // init: [oWCeil, fusedOCCeil]
        initCondition, // init condition
        ValueRange{},  // sync_related_args
        padding,       // padding attribute
        groups         // groups attribute
    );

    Value newResult = newConvOp.getResultTensors()[0];

    // === rewrite vcast and update target if exist ===
    Value target = convResult;
    Value newTarget = newResult;
    hivm::VCastOp castOp = nullptr;

    if (convResult.hasOneUse()) {
      castOp = dyn_cast<hivm::VCastOp>(*convResult.user_begin());
    }

    if (castOp) {
      auto castResultType = 
          mlir::cast<RankedTensorType>(castOp->getResult(0).getType());
      auto newCastResultType =
          RankedTensorType::get(newShape, castResultType.getElementType());
      auto newCastInit = rewriter.create<tensor::EmptyOp>(
          loc, newShape, castResultType.getElementType());
      
      auto newCastOp = rewriter.create<hivm::VCastOp>(
          loc, TypeRange{newCastResultType}, ValueRange{newResult},
          ValueRange{newCastInit.getResult()}, /*temp_buffer=*/Value(),
          castOp.getRoundModeAttr(), hivm::TypeFnAttr{});
      
      target = castOp->getResult(0);
      newTarget = newCastOp->getResult(0);
    }

    auto newTargetType =
        mlir::cast<RankedTensorType>(newTarget.getType()).getElementType();
    
    // === post process to [fusedOC, oW] ===
    if (isOCPerGroupAligned || fusedGroupsVal == 1) {
 	    // case: aligned or fusedGroupsVal == 1
      // step 1: [oWCeil, fusedOCCeil] -> [oW, fusedOC]
      SmallVector<OpFoldResult, 2> offsets{
          rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(0),
      };

      SmallVector<OpFoldResult, 2> sizes{
          rewriter.getIndexAttr(oW),
          rewriter.getIndexAttr(fusedOC),
      };

      SmallVector<OpFoldResult, 2> strides{
          rewriter.getIndexAttr(1),
          rewriter.getIndexAttr(1),
      };
 	 
 	    auto sliceType = RankedTensorType::get({oW, fusedOC}, newTargetType);
      auto slice = rewriter.create<tensor::ExtractSliceOp>(
          loc, sliceType, newTarget, offsets, sizes, strides);
 	 
 	    // step 2: transpose [oW, fusedOC] -> [fusedOC, oW]
      SmallVector<int64_t> tShape{fusedOC, oW};
      Value tInit =
          rewriter.create<tensor::EmptyOp>(loc, tShape, newTargetType);
      SmallVector<int64_t, 2> perm{1, 0};

      newResult = rewriter
                      .create<hivm::VTransposeOp>(
                          loc, TypeRange{tInit.getType()}, slice, tInit,
                          rewriter.getDenseI64ArrayAttr(perm))
                      ->getResult(0);
    } else {
      // case: not aligned && fusedGroupsVal != 1
      // step 1: transpose [oWCeil, fusedOCCeil] -> [fusedOCCeil, oWCeil]
      SmallVector<int64_t> tShape{fusedOCCeil, oWCeil};
      Value tInit =
          rewriter.create<tensor::EmptyOp>(loc, tShape, newTargetType);
      SmallVector<int64_t, 2> perm{1, 0};

      Value transposedOutput =
          rewriter
              .create<hivm::VTransposeOp>(loc, TypeRange{tInit.getType()},
                                          newTarget, tInit,
                                          rewriter.getDenseI64ArrayAttr(perm))
              ->getResult(0);
    
      // step 2: alloc [fusedOC, oWCeil]
      SmallVector<int64_t> sShape{fusedOC, oWCeil};
      Value slicedOutput =
          rewriter.create<tensor::EmptyOp>(loc, sShape, newTargetType);

      // step 3: for i in [0, fusedGroupsVal), extract_slice + insert_slice
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upper = rewriter.create<arith::ConstantIndexOp>(loc, fusedGroupsVal);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      auto forOp = rewriter.create<scf::ForOp>(loc, zero, upper, step,
                                               ValueRange{slicedOutput});

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());

        Value iv = forOp.getInductionVar();
        Value slicedOutput = forOp.getRegionIterArg(0);

        auto oCPerGroupCeilVal =
            rewriter.create<arith::ConstantIndexOp>(loc, oCPerGroupCeil);
        auto oCPerGroupVal =
            rewriter.create<arith::ConstantIndexOp>(loc, oCPerGroup);

        Value srcOffset0 =
            rewriter.create<arith::MulIOp>(loc, iv, oCPerGroupCeilVal);
        Value dstOffset0 =
            rewriter.create<arith::MulIOp>(loc, iv, oCPerGroupVal);

        // subTransposedOutput = extract_slice transposedOutput 
        // [oCPerGroup, oWCeil]
        SmallVector<OpFoldResult, 2> srcOffsets{srcOffset0,
                                                rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult, 2> srcSizes{rewriter.getIndexAttr(oCPerGroup),
                                              rewriter.getIndexAttr(oWCeil)};
        SmallVector<OpFoldResult, 2> srcStrides{rewriter.getIndexAttr(1),
                                                rewriter.getIndexAttr(1)};

        auto subTransposedOutput = rewriter.create<tensor::ExtractSliceOp>(
            loc, transposedOutput, srcOffsets, srcSizes, srcStrides);

        // newSlicedOutput = insert_slice subTransposedOutput into slicedOutput
        SmallVector<OpFoldResult, 2> dstOffsets{dstOffset0,
                                                rewriter.getIndexAttr(0)};
        SmallVector<OpFoldResult, 2> dstSizes{rewriter.getIndexAttr(oCPerGroup),
                                              rewriter.getIndexAttr(oWCeil)};
        SmallVector<OpFoldResult, 2> dstStrides{rewriter.getIndexAttr(1),
                                                rewriter.getIndexAttr(1)};

        auto newSlicedOutput = rewriter.create<tensor::InsertSliceOp>(
            loc, subTransposedOutput.getResult(), slicedOutput, dstOffsets,
            dstSizes, dstStrides);

        rewriter.create<scf::YieldOp>(loc, newSlicedOutput.getResult());
      }

      Value slicedResult = forOp.getResult(0);

      // step 4: extract_slice slicedResult: [fusedOC, oWCeil] -> [fusedOC, oW]
      SmallVector<OpFoldResult, 2> offsets{
          rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(0),
      };
      SmallVector<OpFoldResult, 2> sizes{
          rewriter.getIndexAttr(fusedOC),
          rewriter.getIndexAttr(oW),
      };
      SmallVector<OpFoldResult, 2> strides{
          rewriter.getIndexAttr(1),
          rewriter.getIndexAttr(1),
      };

      auto finalType = RankedTensorType::get({fusedOC, oW}, newTargetType);
      auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
          loc, finalType, slicedResult, offsets, sizes, strides);

      newResult = extractSliceOp.getResult();
    }

    // === batch reshape ===
    if (hasBatch) {
      auto finalType = RankedTensorType::get({batch, oC, oW}, newTargetType);

      SmallVector<ReassociationIndices> reassoc = {
          {0, 1}, // B * oC
          {2}     // oW
      };

      newResult = rewriter.create<tensor::ExpandShapeOp>(loc, finalType,
                                                         newResult, reassoc);
    }
 	 
    rewriter.replaceOp(target.getDefiningOp(), newResult);
    newConvOp->setAttr(outputAlreadyNormalized, rewriter.getUnitAttr());
    return success();
  }
};

void populateNormalizeConvOpsPattern1(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvResultTypePattern<BFloat16Type>>(
      patterns.getContext());
  patterns.add<DecomposeConv1dWithBiasPattern<Float16Type>>(
      patterns.getContext());
}

void populateNormalizeConvOpsPattern2(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvResultTypePattern<Float16Type>>(
      patterns.getContext());
  patterns.add<DecomposeConv1dWithBiasPattern<Float32Type>>(
      patterns.getContext());
}

void populateNormalizeConvOpsPattern3(RewritePatternSet &patterns) {
  patterns.add<NormalizeConvOutputPattern>(patterns.getContext());
}

void NormalizeConvOpsPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto *context = &getContext();
  auto *funcOp = getOperation();

  // First Round
  RewritePatternSet patterns1(context);
  populateNormalizeConvOpsPattern1(patterns1);
  GreedyRewriteConfig config1 = GreedyRewriteConfig();
  (void)applyPatternsGreedily(funcOp, std::move(patterns1), config1);

  // Second Round
  RewritePatternSet patterns2(context);
  populateNormalizeConvOpsPattern2(patterns2);
  GreedyRewriteConfig config2 = GreedyRewriteConfig();
  (void)applyPatternsGreedily(funcOp, std::move(patterns2), config2);

  // Third Round
  RewritePatternSet patterns3(context);
  populateNormalizeConvOpsPattern3(patterns3);
  GreedyRewriteConfig config3 = GreedyRewriteConfig();
  (void)applyPatternsGreedily(funcOp, std::move(patterns3), config3);
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createNormalizeConvOpsPass() {
  return std::make_unique<NormalizeConvOpsPass>();
}