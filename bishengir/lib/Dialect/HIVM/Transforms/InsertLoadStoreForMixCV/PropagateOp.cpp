//===- PropagateOp.cpp - Propagate pattern of InsertLoadStoreForMixCV -----===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/PropagateOp.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "insert-load-store-propagate-op"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {

static bool checkPropagate(PropagationStep step,
                           UnrealizedConversionCastOp propagateOp) {
  auto addressSpaces = PropagatorUtil::getAddressSpace(propagateOp);
  switch (step) {
  case PropagationStep::LOCAL:
    return PropagatorUtil::getCoreType(propagateOp) ==
           TCoreType::CUBE_AND_VECTOR;
  case PropagationStep::GM:
    return llvm::find(addressSpaces, hivm::AddressSpace::GM) !=
           addressSpaces.end();
  case PropagationStep::UB:
    return PropagatorUtil::getCoreType(propagateOp) == TCoreType::VECTOR;
  case PropagationStep::L1:
    return PropagatorUtil::getCoreType(propagateOp) == TCoreType::CUBE;
  default:
    return true;
  }
}

// %19 = scf.if %7 -> (tensor<32xf32>) {
//   %30 = builtin.unrealized_conversion_cast %29 : tensor<32xf32> to
//   tensor<32xf32> {hivm.address_space = [#hivm.address_space<ub>],
//   hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_up} scf.yield %30 :
//   tensor<32xf32>
// } else {
//   %31 = builtin.unrealized_conversion_cast %30 : tensor<32xf32> to
//   tensor<32xf32> {hivm.address_space = [#hivm.address_space<ub>],
//   hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_up} scf.yield %31 :
//   tensor<32xf32>
// }
// %20 = builtin.unrealized_conversion_cast %19 : tensor<32xf32> to
// tensor<32xf32> {hivm.address_space = [#hivm.address_space<ub>],
// hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_down}
static LogicalResult propagateIfOp(scf::IfOp op, OpResult res,
                                   UnrealizedConversionCastOp propagateOp,
                                   PatternRewriter &rewriter) {
  auto resIdx = res.getResultNumber();
  PropagatorUtil::createPropagatorDown(res, propagateOp, rewriter);
  PropagatorUtil::createPropagatorUp(
      &op.thenYield().getResultsMutable()[resIdx], propagateOp, rewriter);
  PropagatorUtil::createPropagatorUp(
      &op.elseYield().getResultsMutable()[resIdx], propagateOp, rewriter);
  return success();
}

static LogicalResult
propagateIndexSwitchOp(scf::IndexSwitchOp op, OpResult res,
                       UnrealizedConversionCastOp propagateOp,
                       PatternRewriter &rewriter) {
  auto resIdx = res.getResultNumber();
  PropagatorUtil::createPropagatorDown(res, propagateOp, rewriter);
  for (auto &region : op->getRegions()) {
    auto yieldOp = cast<scf::YieldOp>(region.front().getTerminator());
    PropagatorUtil::createPropagatorUp(&yieldOp.getResultsMutable()[resIdx],
                                       propagateOp, rewriter);
  }
  return success();
}

// %229 = builtin.unrealized_conversion_cast %228 : tensor<32x64xbf16> to
// tensor<32x64xbf16> {hivm.address_space = [#hivm.address_space<ub>],
// hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_up} %230 = scf.for ...
// iter_args(%arg31 = %229) -> (tensor<32x64xbf16>) {
//   %280 = builtin.unrealized_conversion_cast %arg31 : tensor<32x64xbf16> to
//   tensor<32x64xbf16> {hivm.address_space = [#hivm.address_space<ub>],
//   hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_down}
//   ...
//   %284 = builtin.unrealized_conversion_cast %283 : tensor<32x64xbf16> to
//   tensor<32x64xbf16> {hivm.address_space = [#hivm.address_space<ub>],
//   hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_up} scf.yield %284 :
//   tensor<32x64xbf16>
// }
// %231 = builtin.unrealized_conversion_cast %230 : tensor<32x64xbf16> to
// tensor<32x64xbf16> {hivm.address_space = [#hivm.address_space<ub>],
// hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_down
static LogicalResult propagateForOp(scf::ForOp op, OpOperand &operand,
                                    UnrealizedConversionCastOp propagateOp,
                                    PatternRewriter &rewriter) {
  PropagatorUtil::createPropagatorUp(&operand, propagateOp, rewriter);

  auto regionIterArg = op.getTiedLoopRegionIterArg(&operand);
  if (regionIterArg) {
    PropagatorUtil::createPropagatorDown(regionIterArg, propagateOp, rewriter);

    auto *yield = op.getTiedLoopYieldedValue(regionIterArg);
    PropagatorUtil::createPropagatorUp(yield, propagateOp, rewriter);

    auto res = op.getTiedLoopResult(&operand);
    PropagatorUtil::createPropagatorDown(res, propagateOp, rewriter);
  }
  return success();
}

// %16 = builtin.unrealized_conversion_cast %15 : tensor<128xf32> to
// tensor<128xf32> {hivm.address_space = [#hivm.address_space<cbuf>],
// hivm.tcore_type = #hivm.tcore_type<CUBE>, propagate_up} %17:2 = scf.while
// (%arg8 = %16, %arg9 = %c0_i32) : (tensor<128xf32>, i32) -> (tensor<128xf32>,
// i32) {
//   %18 = builtin.unrealized_conversion_cast %arg8 : tensor<128xf32> to
//   tensor<128xf32> {hivm.address_space = [#hivm.address_space<cbuf>],
//   hivm.tcore_type = #hivm.tcore_type<CUBE>, propagate_down}
//   ...
//   %20 = builtin.unrealized_conversion_cast %18 : tensor<128xf32> to
//   tensor<128xf32> {hivm.address_space = [#hivm.address_space<ub>],
//   hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_up}
//   scf.condition(%19) %20, %arg9 : tensor<128xf32>, i32
// } do {
// ^bb0(%arg8: tensor<128xf32>, %arg9: i32):
//   %18 = builtin.unrealized_conversion_cast %arg8 : tensor<128xf32> to
//   tensor<128xf32> {hivm.address_space = [#hivm.address_space<ub>],
//   hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_down}
//   ...
//   %40 = builtin.unrealized_conversion_cast %39 : tensor<128xf32> to
//   tensor<128xf32> {hivm.address_space = [#hivm.address_space<cbuf>],
//   hivm.tcore_type = #hivm.tcore_type<CUBE>, propagate_up} scf.yield %40, %38
//   : tensor<128xf32>, i32
// }
// %18 = builtin.unrealized_conversion_cast %17#0 : tensor<128xf32> to
// tensor<128xf32> {hivm.address_space = [#hivm.address_space<ub>],
// hivm.tcore_type = #hivm.tcore_type<VECTOR>, propagate_down}
static LogicalResult
propagateWhileOpOperand(scf::WhileOp op, size_t operandNumber,
                        UnrealizedConversionCastOp propagateOp,
                        PatternRewriter &rewriter) {
  auto &initArg = op.getInitsMutable()[operandNumber];
  PropagatorUtil::createPropagatorUp(&initArg, propagateOp, rewriter);
  auto beforeArg = op.getBeforeArguments()[operandNumber];
  PropagatorUtil::createPropagatorDown(beforeArg, propagateOp, rewriter);
  auto &yield = op.getYieldOp().getResultsMutable()[operandNumber];
  PropagatorUtil::createPropagatorUp(&yield, propagateOp, rewriter);
  return success();
}

static LogicalResult
propagateWhileOpResult(scf::WhileOp op, size_t resultNumber,
                       UnrealizedConversionCastOp propagateOp,
                       PatternRewriter &rewriter) {
  auto res = op->getResult(resultNumber);
  PropagatorUtil::createPropagatorDown(res, propagateOp, rewriter);
  auto beforeArg = op.getAfterArguments()[resultNumber];
  PropagatorUtil::createPropagatorDown(beforeArg, propagateOp, rewriter);
  auto &condArg = op.getConditionOp().getArgsMutable()[resultNumber];
  PropagatorUtil::createPropagatorUp(&condArg, propagateOp, rewriter);
  return success();
}

static LogicalResult propagateScopeOp(scope::ScopeOp op, size_t resultNumber,
                                      UnrealizedConversionCastOp propagateOp,
                                      PatternRewriter &rewriter) {
  auto res = op->getResult(resultNumber);
  PropagatorUtil::createPropagatorDown(res, propagateOp, rewriter);
  auto &scopeBody = op.getRegion().front();
  auto returnOp = cast<scope::ReturnOp>(scopeBody.getTerminator());
  auto &operand = returnOp.getResultsMutable()[resultNumber];
  PropagatorUtil::createPropagatorUp(&operand, propagateOp, rewriter);
  return success();
}

LogicalResult
PropagateDownPattern::propagateDownForOp(scf::ForOp op, OpOperand &operand,
                                         UnrealizedConversionCastOp propagateOp,
                                         PatternRewriter &rewriter) const {
  return propagateForOp(op, operand, propagateOp, rewriter);
}

LogicalResult PropagateDownPattern::propagateDownWhileOp(
    scf::WhileOp op, OpOperand &operand, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {
  return propagateWhileOpOperand(op, operand.getOperandNumber(), propagateOp,
                                 rewriter);
}

LogicalResult PropagateDownPattern::propagateDownYieldOp(
    scf::YieldOp op, OpOperand &operand, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {
  return TypeSwitch<Operation *, LogicalResult>(op->getParentOp())
      .Case([&](scf::ForOp forOp) {
        auto *init = &forOp.getInitArgsMutable()[operand.getOperandNumber()];
        return propagateForOp(forOp, *init, propagateOp, rewriter);
      })
      .Case([&](scf::WhileOp whileOp) {
        return propagateWhileOpOperand(whileOp, operand.getOperandNumber(),
                                       propagateOp, rewriter);
      })
      .Case([&](scf::IfOp ifOp) {
        return propagateIfOp(ifOp, ifOp->getResult(operand.getOperandNumber()),
                             propagateOp, rewriter);
      })
      .Case([&](scf::IndexSwitchOp indexSwitchOp) {
        return propagateIndexSwitchOp(
            indexSwitchOp, indexSwitchOp->getResult(operand.getOperandNumber()),
            propagateOp, rewriter);
      })
      .Default([&](auto *op) {
        llvm_unreachable("Unhandled YieldOp");
        return failure();
      });
}

LogicalResult PropagateDownPattern::propagateDownConditionOp(
    scf::ConditionOp op, OpOperand &operand,
    UnrealizedConversionCastOp propagateOp, PatternRewriter &rewriter) const {
  auto whileOp = cast<scf::WhileOp>(op->getParentOp());
  auto oprNum = operand.getOperandNumber();
  if (oprNum == 0) {
    return failure();
  }
  return propagateWhileOpResult(whileOp, oprNum - 1, propagateOp, rewriter);
}

LogicalResult PropagateDownPattern::propagateDownReturnOp(
    scope::ReturnOp op, OpOperand &operand,
    UnrealizedConversionCastOp propagateOp, PatternRewriter &rewriter) const {
  auto scopeOp = cast<scope::ScopeOp>(op->getParentOp());
  return propagateScopeOp(scopeOp, operand.getOperandNumber(), propagateOp,
                          rewriter);
}

LogicalResult PropagateDownPattern::propagateDownDmaOp(
    hivm::HIVMStructuredOp op, OpOperand &operand,
    UnrealizedConversionCastOp propagateOp, PatternRewriter &rewriter) const {
  // Same boundary rule as propagateDownForCustomLikeOp; DMA updates all inits
  // when any init operand is reached.
  LDBG("Propagating dma down: " << op << "\n" << propagateOp);
  for (auto *input : op.getDpsInputOperands()) {
    if (input->get() == operand.get()) {
      LDBG("Operand: " << input->get());
      PropagatorUtil::createPropagatorUp(input, propagateOp, rewriter);
      return success();
    }
  }

  bool isInit = false;
  for (auto init : op.getDpsInits()) {
    if (init == operand.get()) {
      isInit = true;
      break;
    }
  }
  if (isInit) {
    for (auto &init : op.getDpsInitsMutable())
      PropagatorUtil::createPropagatorUp(&init, propagateOp, rewriter);
    PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
    return success();
  }
  return failure();
}

LogicalResult PropagateUpPattern::propagateUpBlockArgument(
    BlockArgument blockArgument, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {
  auto *parentBlock = blockArgument.getOwner();
  return TypeSwitch<Operation *, LogicalResult>(parentBlock->getParentOp())
      .Case([&](scf::ForOp forOp) {
        if (auto res = forOp.getTiedLoopResult(blockArgument))
          return propagateUpForOp(forOp, res, propagateOp, rewriter);
        PropagatorUtil::createPropagatorDown(blockArgument, propagateOp,
                                             rewriter);
        return success();
      })
      .Case([&](scf::WhileOp whileOp) {
        if (whileOp.getBeforeBody() == parentBlock) {
          return propagateWhileOpOperand(whileOp, blockArgument.getArgNumber(),
                                         propagateOp, rewriter);
        }
        if (whileOp.getAfterBody() == parentBlock) {
          return propagateWhileOpResult(whileOp, blockArgument.getArgNumber(),
                                        propagateOp, rewriter);
        }
        llvm_unreachable(
            "WhileOp should have either before body or after body");
        return failure();
      })
      .Default([&](auto *parentOp) {
        PropagatorUtil::createPropagatorDown(blockArgument, propagateOp,
                                             rewriter);
        return success();
      });
}

LogicalResult
PropagateUpPattern::propagateUpIfOp(scf::IfOp op, OpResult res,
                                    UnrealizedConversionCastOp propagateOp,
                                    PatternRewriter &rewriter) const {
  return propagateIfOp(op, res, propagateOp, rewriter);
}

LogicalResult PropagateUpPattern::propagateUpIndexSwitchOp(
    scf::IndexSwitchOp op, OpResult res, UnrealizedConversionCastOp propagateOp,
    PatternRewriter &rewriter) const {
  return propagateIndexSwitchOp(op, res, propagateOp, rewriter);
}

LogicalResult
PropagateUpPattern::propagateUpForOp(scf::ForOp op, OpResult res,
                                     UnrealizedConversionCastOp propagateOp,
                                     PatternRewriter &rewriter) const {
  auto *init = op.getTiedLoopInit(res);
  if (op->hasAttr("ExtractedLoadOrStore") &&
      !op->getParentOp()->hasAttr("ExtractedLoadOrStore")) {
    // Unstructured load case should propagated from the inside
    return failure();
  }
  return propagateForOp(op, *init, propagateOp, rewriter);
}

LogicalResult
PropagateUpPattern::propagateUpWhileOp(scf::WhileOp op, OpResult res,
                                       UnrealizedConversionCastOp propagateOp,
                                       PatternRewriter &rewriter) const {
  return propagateWhileOpResult(op, res.getResultNumber(), propagateOp,
                                rewriter);
}

LogicalResult
PropagateUpPattern::propagateUpScopeOp(scope::ScopeOp op, OpResult res,
                                       UnrealizedConversionCastOp propagateOp,
                                       PatternRewriter &rewriter) const {
  return propagateScopeOp(op, res.getResultNumber(), propagateOp, rewriter);
}

LogicalResult
PropagateUpPattern::propagateUpDmaOp(hivm::HIVMStructuredOp op, OpResult res,
                                     UnrealizedConversionCastOp propagateOp,
                                     PatternRewriter &rewriter) const {
  // Same boundary rule as propagateUpForCustomLikeOp.
  LDBG("Propagating dma up: " << op << "\n" << propagateOp);
  PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
  for (auto &init : op.getDpsInitsMutable()) {
    PropagatorUtil::createPropagatorUp(&init, propagateOp, rewriter);
  }
  return success();
}

LogicalResult
PropagateDownPattern::matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                      PatternRewriter &rewriter) const {
  if (!propagateOp->hasAttr(kPropagateDownAttr))
    return failure();
  if (propagateOp->getResult(0).getType().isIntOrIndexOrFloat())
    return failure();
  if (!checkPropagate(step, propagateOp))
    return failure();
  SmallVector<OpOperand *> uses;
  for (auto &use : propagateOp->getUses())
    uses.push_back(&use);
  LogicalResult result = failure();
  for (auto *use : uses) {
    auto *user = use->getOwner();
    if (user->hasAttr(kPropagateUpAttr))
      continue;
    auto newRes =
        TypeSwitch<Operation *, LogicalResult>(user)
            .Case([&](scf::ForOp op) {
              if (step == PropagationStep::LOCAL)
                return failure();
              return propagateDownForOp(op, *use, propagateOp, rewriter);
            })
            .Case([&](scf::WhileOp op) {
              if (step == PropagationStep::LOCAL || step == PropagationStep::GM)
                return failure();
              return propagateDownWhileOp(op, *use, propagateOp, rewriter);
            })
            .Case([&](scf::YieldOp op) {
              if (step == PropagationStep::LOCAL ||
                  (step == PropagationStep::GM &&
                   isa<scf::WhileOp>(op->getParentOp())))
                return failure();
              return propagateDownYieldOp(op, *use, propagateOp, rewriter);
            })
            .Case([&](scf::ConditionOp op) {
              if (step == PropagationStep::LOCAL)
                return failure();
              return propagateDownConditionOp(op, *use, propagateOp, rewriter);
            })
            .Case([&](scope::ReturnOp op) {
              if (step == PropagationStep::LOCAL)
                return failure();
              return propagateDownReturnOp(op, *use, propagateOp, rewriter);
            })
            .Case<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"
                >([&](auto op) {
              if (step == PropagationStep::LOCAL)
                return failure();
              auto dmaOp = dyn_cast<hivm::HIVMStructuredOp>(op.getOperation());
              if (!dmaOp)
                return failure();
              return propagateDownDmaOp(dmaOp, *use, propagateOp, rewriter);
            })
            .Case<hivm::CustomMacroOp, hivm::CustomOp>([&](Operation *op) {
              return PropagatorUtil::propagateDownForCustomLikeOp(
                  op, use, propagateOp, rewriter);
            })
            .Default([&](Operation *op) {
              PropagatorUtil::createPropagatorsUp(op, propagateOp, rewriter);
              PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
              return success();
            });
    if (succeeded(newRes))
      result = newRes;
  }
  return result;
}

LogicalResult
PropagateUpPattern::matchAndRewrite(UnrealizedConversionCastOp propagateOp,
                                    PatternRewriter &rewriter) const {
  if (!propagateOp->hasAttr(kPropagateUpAttr))
    return failure();
  auto input = propagateOp.getInputs()[0];
  auto res = dyn_cast<OpResult>(input);
  if (input.getType().isIntOrIndexOrFloat())
    return failure();
  if (!checkPropagate(step, propagateOp))
    return failure();
  if (!res) {
    if (step == PropagationStep::LOCAL)
      return failure();
    auto blockArgument = cast<BlockArgument>(input);
    return propagateUpBlockArgument(blockArgument, propagateOp, rewriter);
  }
  auto *defOp = res.getDefiningOp();
  if (!defOp || defOp->hasAttr(kPropagateDownAttr))
    return failure();
  return TypeSwitch<Operation *, LogicalResult>(defOp)
      .Case([&](scf::IfOp op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        return propagateUpIfOp(op, res, propagateOp, rewriter);
      })
      .Case([&](scf::IndexSwitchOp op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        return propagateUpIndexSwitchOp(op, res, propagateOp, rewriter);
      })
      .Case([&](scf::ForOp op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        return propagateUpForOp(op, res, propagateOp, rewriter);
      })
      .Case([&](scf::WhileOp op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        return propagateUpWhileOp(op, res, propagateOp, rewriter);
      })
      .Case([&](scope::ScopeOp op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        return propagateUpScopeOp(op, res, propagateOp, rewriter);
      })
      .Case<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.cpp.inc"
          >([&](Operation *op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        auto dmaOp = cast<hivm::HIVMStructuredOp>(op);
        return propagateUpDmaOp(dmaOp, res, propagateOp, rewriter);
      })
      .Case([&](tensor::InsertSliceOp op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        if (PropagatorUtil::getCoreType(propagateOp) == TCoreType::CUBE) {
          PropagatorUtil::createPropagatorsUp(op, TCoreType::VECTOR,
                                              hivm::AddressSpace::UB, rewriter);
          PropagatorUtil::createPropagatorsDown(
              op, TCoreType::VECTOR, hivm::AddressSpace::UB, rewriter);
        } else {
          PropagatorUtil::createPropagatorsUp(op, propagateOp, rewriter);
          PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
        }
        return success();
      })
      .Case<hivm::CustomMacroOp, hivm::CustomOp>([&](Operation *op) {
        if (step == PropagationStep::LOCAL)
          return failure();
        return PropagatorUtil::propagateUpForCustomLikeOp(op, propagateOp,
                                                          rewriter);
      })
      .Default([&](Operation *op) {
        if (step == PropagationStep::LOCAL &&
            !llvm::all_of(op->getResultTypes(),
                          [](auto type) { return isa<ShapedType>(type); }))
          return failure();
        PropagatorUtil::createPropagatorsUp(op, propagateOp, rewriter);
        PropagatorUtil::createPropagatorsDown(op, propagateOp, rewriter);
        return success();
      });
}

} // namespace mlir::hivm
