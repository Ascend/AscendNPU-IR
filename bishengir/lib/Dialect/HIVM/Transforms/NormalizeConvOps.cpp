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
  ~NormalizeConvResultTypePattern() override = default;

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
  ~DecomposeConv1dWithBiasPattern() override = default;

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
}

} // namespace

std::unique_ptr<Pass> mlir::hivm::createNormalizeConvOpsPass() {
  return std::make_unique<NormalizeConvOpsPass>();
}