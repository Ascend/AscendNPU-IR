//===---------------------- InlineFixpipe.cpp -----------------------------===//
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
//
// This pass converts ops to hivm.fixpipe.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTFIXPIPE
#define GEN_PASS_DEF_INLINEFIXPIPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "hivm-inline-fixpipe"

namespace {
static constexpr llvm::StringLiteral printType = "print";
static constexpr llvm::StringLiteral mmadFixpipeForResultAlreadyInserted =
    "fixpipe_for_result_already_inserted";

static constexpr llvm::StringLiteral fixpipeDoNotMoveOutOfScfFor =
    "do_not_move_out_of_scffor";

static constexpr llvm::StringLiteral scfforFixpipeForMMADResultAlreadyInserted =
    "fixpipe_for_mmad_result_already_inserted";

static constexpr llvm::StringLiteral usedForDebugOp = "used_for_debug_op";
} // namespace

namespace {
struct InsertFixpipe : public impl::InsertFixpipeBase<InsertFixpipe> {
  using Base::Base;
  void runOnOperation() override;
};

struct InlineFixpipe : public impl::InlineFixpipeBase<InlineFixpipe> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

std::optional<bool> isStoreOp(Operation *dstOp) {
  if (isa<hivm::StoreOp>(dstOp)) {
    return true;
  }
  bool isPreOp = isa<hivm::VCastOp>(dstOp) || isa<hivm::VReluOp>(dstOp);

  if (dstOp->getDialect()->getNamespace() ==
          HIVMDialect::getDialectNamespace() &&
      !isPreOp) {
    return false;
  }
  return std::nullopt;
}

// ---- membase version of getInsertPoint ----
Operation *getInsertPoint_membase(Operation *op, int &resultIndx) {
  auto users = op->getResult(resultIndx).getUsers();
  std::set<scf::YieldOp> yieldOperands;
  for (auto *user : users) {
    auto forOp = user->getParentOfType<scf::ForOp>();
    if (!isa<scf::YieldOp>(user) || !forOp) {
      continue;
    } else {
      yieldOperands.emplace(user);
    }
  }

  if (yieldOperands.empty()) {
    return op;
  }

  if (yieldOperands.size() > 1) {
    op->emitError("unsupport cases");
    return op;
  }
  auto yieldOperand = *yieldOperands.begin();
  auto yieldParentOp = yieldOperand->getParentOp();
  auto yieldValueIndx = findIdx(llvm::to_vector(yieldOperand->getOperands()),
                                op->getResult(resultIndx));
  if (!yieldValueIndx.has_value())
    llvm::report_fatal_error("yield value must have user");
  resultIndx = yieldValueIndx.value();
  return getInsertPoint_membase(yieldParentOp, resultIndx);
}

bool isAccumulationImpl(Operation *op, Value accumulator) {
  if (!accumulator)
    return false;

  auto forOp = op->getParentOfType<scf::ForOp>();
  if (!forOp)
    return false;

  auto accArg = dyn_cast<BlockArgument>(accumulator);
  if (!accArg || accArg.getOwner() != forOp.getBody() ||
      accArg.getArgNumber() < forOp.getNumInductionVars()) {
    return false;
  }

  unsigned iterIdx = accArg.getArgNumber() - forOp.getNumInductionVars();

  Value val = op->getResult(0);
  while (val) {
    if (val == forOp.getBody()->getTerminator()->getOperand(iterIdx)) {
      return true;
    }

    if (!val.hasOneUse()) {
      return false;
    }

    auto yieldOp = dyn_cast<scf::YieldOp>(*val.user_begin());
    if (!yieldOp) {
      return false;
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
      unsigned operandIdx = val.use_begin()->getOperandNumber();
      val = ifOp.getResult(operandIdx);
    } else {
      return false;
    }
  }

  return false;
}

bool isAccumulation(Operation *op) {
  if (auto mmadOp = dyn_cast<hivm::MmadL1Op>(op))
    return isAccumulationImpl(op, mmadOp.getC());

  if (auto conv1d = dyn_cast<hivm::Conv1DL1Op>(op))
    return isAccumulationImpl(op, conv1d.getInit());

  if (auto conv2d = dyn_cast<hivm::Conv2DL1Op>(op))
    return isAccumulationImpl(op, conv2d.getInit());

  if (auto conv3d = dyn_cast<hivm::Conv3DL1Op>(op))
    return isAccumulationImpl(op, conv3d.getInit());

  return false;
}

// ---- regbase-only functions ----
bool hasSameSource(Value valA, Value valB) {
  auto maybeMmadA = traceDefOp<hivm::MmadL1Op>(valA);
  auto maybeMmadB = traceDefOp<hivm::MmadL1Op>(valB);

  auto isMatchedEmptyOrAllocOp = [](Value valA, Value valB) {
    LDBG("Begin to find matched dst for scf.if \n  -" << valA << "\n  -"
                                                      << valB);
    auto maybeTensorA = traceDefOp<tensor::EmptyOp>(valA);
    auto maybeTensorB = traceDefOp<tensor::EmptyOp>(valB);
    if (maybeTensorA.has_value() && maybeTensorB.has_value() &&
        (maybeTensorA.value() == maybeTensorB.value()))
      return true;

    auto maybeAllocA = traceDefOp<memref::AllocOp>(valA);
    auto maybeAllocB = traceDefOp<memref::AllocOp>(valB);
    if (maybeAllocA.has_value() && maybeAllocB.has_value() &&
        (maybeAllocA.value() == maybeAllocB.value()))
      return true;

    if (maybeAllocA.has_value()) {
      auto alloc = cast<memref::AllocOp>(maybeAllocA.value());
      hivm::AddressSpace addrSpace{hivm::AddressSpace::Zero};
      if (auto memSpaceAttr = alloc.getType().getMemorySpace()) {
        addrSpace = dyn_cast<AddressSpaceAttr>(memSpaceAttr).getAddressSpace();
      }
      return (addrSpace == hivm::AddressSpace::L0C && maybeTensorB.has_value());
    }

    if (maybeAllocB.has_value()) {
      auto alloc = cast<memref::AllocOp>(maybeAllocB.value());
      hivm::AddressSpace addrSpace{hivm::AddressSpace::Zero};
      if (auto memSpaceAttr = alloc.getType().getMemorySpace()) {
        addrSpace = dyn_cast<AddressSpaceAttr>(memSpaceAttr).getAddressSpace();
      }
      return (addrSpace == hivm::AddressSpace::L0C && maybeTensorA.has_value());
    }
    LDBG("There is no matched dst for scf.if \n");
    return false;
  };
  if (maybeMmadA.has_value() && maybeMmadB.has_value()) {
    auto mmadA = cast<hivm::MmadL1Op>(maybeMmadA.value());
    auto mmadB = cast<hivm::MmadL1Op>(maybeMmadB.value());
    return isMatchedEmptyOrAllocOp(mmadA.getC(), mmadB.getC());
  }

  if (maybeMmadA.has_value()) {
    auto mmadA = cast<hivm::MmadL1Op>(maybeMmadA.value());
    return isMatchedEmptyOrAllocOp(mmadA.getC(), valB);
  }

  if (maybeMmadB.has_value()) {
    auto mmadB = cast<hivm::MmadL1Op>(maybeMmadB.value());
    return isMatchedEmptyOrAllocOp(valA, mmadB.getC());
  }
  return false;
}

bool isMixKernel(scf::IfOp ifOp, Value val) {
  if (ifOp.getElseRegion().empty())
    return true;

  if (auto idx =
          findIdx(llvm::to_vector(ifOp.thenYield().getOperands()), val)) {
    if (idx.has_value()) {
      auto elseValue = ifOp.elseYield().getOperands()[idx.value()];
      return !hasSameSource(val, elseValue);
    }
  }

  if (auto idx =
          findIdx(llvm::to_vector(ifOp.elseYield().getOperands()), val)) {
    if (idx.has_value()) {
      auto thenValue = ifOp.thenYield().getOperands()[idx.value()];
      return !hasSameSource(val, thenValue);
    }
  }

  return true;
}

bool needYieldOut(Operation *user, Value val) {
  if (isa<scf::ForOp>(user->getParentOp()))
    return true;
  if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp()))
    return !isMixKernel(ifOp, val);

  return false;
}

FixpipeDMAMode getInsertedFixpipeDmaMode(Value src, Value dst,
                                         bool inferFixpipeDmaMode) {
  if (!inferFixpipeDmaMode)
    return FixpipeDMAMode::NZ2ND;

  auto srcType = dyn_cast<ShapedType>(src.getType());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!srcType || !dstType)
    return FixpipeDMAMode::NZ2ND;

  if (srcType.hasRank() && dstType.hasRank() &&
      succeeded(
          verifyCompatibleShape(srcType.getShape(), dstType.getShape()))) {
    if (dstType.getRank() == 2)
      return FixpipeDMAMode::NZ2ND;
    return FixpipeDMAMode::NZ2NZ;
  }

  return FixpipeDMAMode::NZ2ND;
}

// ---- membase version of insertFixpipe ----
static FixpipeOp insertFixpipe_membase(PatternRewriter &rewriter,
                                       Operation *point, Value src) {

  rewriter.setInsertionPointAfter(point);

  auto dst = utils::createEmptyOp(rewriter, point->getLoc(), src);
  MLIRContext *ctx = rewriter.getContext();
  FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);

  auto fixpipe = rewriter.create<FixpipeOp>(
      point->getLoc(), /*result_tensor=*/dst.getType(), src, dst, dmaModeAttr,
      /*dual_dst_mode=*/nullptr,
      /*pre_quant=*/nullptr, /*pre_relu=*/nullptr, /*channel_split=*/nullptr);

  SmallPtrSet<Operation *, 4> exceptedOps;
  exceptedOps.insert(fixpipe);
  for (Operation *use : src.getUsers()) {
    if (isa<DebugOp>(use) || isa<FixpipeOp>(use) ||
        isa<annotation::MarkOp>(use)) {
      exceptedOps.insert(use);
    }
  }
  rewriter.replaceAllUsesExcept(src, fixpipe.getResultTensor(), exceptedOps);
  return fixpipe;
}

// ---- regbase version of insertFixpipe ----
static FixpipeOp insertFixpipe_regbase(PatternRewriter &rewriter,
                                       InsertFixpipePatternOptions options,
                                       Operation *point, Value src) {

  rewriter.setInsertionPointAfter(point);

  auto dst = utils::createEmptyOp(rewriter, point->getLoc(), src);
  auto dmaMode =
      getInsertedFixpipeDmaMode(src, dst, options.inferFixpipeDmaMode);
  auto dmaModeAttr = FixpipeDMAModeAttr::get(rewriter.getContext(), dmaMode);

  auto fixpipe = rewriter.create<FixpipeOp>(
      point->getLoc(), /*result_tensor=*/dst.getType(), src, dst, dmaModeAttr,
      /*dual_dst_mode=*/nullptr,
      /*pre_quant=*/nullptr, /*pre_relu=*/nullptr, /*channel_split=*/nullptr);

  SmallPtrSet<Operation *, 4> exceptedOps;
  exceptedOps.insert(fixpipe);
  for (Operation *use : src.getUsers()) {
    if (isa<DebugOp>(use) || isa<FixpipeOp>(use) ||
        isa<annotation::MarkOp>(use)) {
      exceptedOps.insert(use);
    }
  }
  rewriter.replaceAllUsesExcept(src, fixpipe.getResultTensor(), exceptedOps);
  return fixpipe;
}

struct InsertFixpipeForIterArgMMAD_membase : public OpRewritePattern<scf::ForOp> {
public:
  explicit InsertFixpipeForIterArgMMAD_membase(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp scffor,
                                PatternRewriter &rewriter) const override {
    if (scffor->getAttr(scfforFixpipeForMMADResultAlreadyInserted)) {
      return failure();
    }

    auto mmads =
        utils::collectScfForBodyOperations<hivm::MmadL1Op>(scffor, false);
    if (mmads.empty()) {
      return failure();
    }

    auto *forBlock = &(scffor.getRegion().getBlocks().front());

    bool changed = false;

    for (auto mmad : mmads) {
      auto args =
          utils::tracebackOperandsToBlockArguments(mmad.getA(), forBlock);
      args.append(
          utils::tracebackOperandsToBlockArguments(mmad.getB(), forBlock));

      for (auto arg : args) {
        auto idx = arg.getArgNumber();
        if (idx == 0) {
          continue;
        }

        auto yielded = scffor.getYieldedValues()[idx - 1];

        auto stopOp =
            utils::valueCalculatedUsingOperationInsideBlock<hivm::MmadL1Op>(
                yielded, mmad, forBlock);
        if (stopOp && *stopOp == mmad) {
          LDBG("Inserting fix pipe for " << scffor);

          auto fixpipe =
              insertFixpipe_membase(rewriter, mmad, mmad->getResults()[0]);
          fixpipe->setAttr(fixpipeDoNotMoveOutOfScfFor,
                           rewriter.getBoolAttr(true));
          mmad->setAttr(mmadFixpipeForResultAlreadyInserted,
                        rewriter.getBoolAttr(true));
          changed = true;
          break;
        }
      }
    }

    scffor->setAttr(scfforFixpipeForMMADResultAlreadyInserted,
                    rewriter.getBoolAttr(true));
    return changed ? success() : failure();
  }
};

struct InsertFixpipeForIterArgMMAD_regbase : public OpRewritePattern<scf::ForOp> {
public:
  InsertFixpipeForIterArgMMAD_regbase(MLIRContext *context,
                              InsertFixpipePatternOptions options)
      : OpRewritePattern<scf::ForOp>(context), options(options) {}

  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp scffor,
                                PatternRewriter &rewriter) const override {
    if (scffor->getAttr(scfforFixpipeForMMADResultAlreadyInserted)) {
      return failure();
    }

    auto mmads =
        utils::collectScfForBodyOperations<hivm::MmadL1Op>(scffor, false);
    if (mmads.empty()) {
      return failure();
    }

    auto *forBlock = &(scffor.getRegion().getBlocks().front());

    bool changed = false;

    for (auto mmad : mmads) {
      auto args =
          utils::tracebackOperandsToBlockArguments(mmad.getA(), forBlock);
      args.append(
          utils::tracebackOperandsToBlockArguments(mmad.getB(), forBlock));

      for (auto arg : args) {
        auto idx = arg.getArgNumber();
        if (idx == 0) {
          continue;
        }

        auto yielded = scffor.getYieldedValues()[idx - 1];

        auto stopOp =
            utils::valueCalculatedUsingOperationInsideBlock<hivm::MmadL1Op>(
                yielded, mmad, forBlock);
        if (stopOp && *stopOp == mmad) {
          LDBG("Inserting fix pipe for " << scffor);

          auto fixpipe =
              insertFixpipe_regbase(rewriter, options, mmad, mmad->getResults()[0]);
          fixpipe->setAttr(fixpipeDoNotMoveOutOfScfFor,
                           rewriter.getBoolAttr(true));
          mmad->setAttr(mmadFixpipeForResultAlreadyInserted,
                        rewriter.getBoolAttr(true));
          changed = true;
          break;
        }
      }
    }

    scffor->setAttr(scfforFixpipeForMMADResultAlreadyInserted,
                    rewriter.getBoolAttr(true));
    return changed ? success() : failure();
  }

private:
  InsertFixpipePatternOptions options;
};

FixpipeDMAMode inferSwapFixpipeDmaMode(Value src, Value dst,
                                       FixpipeDMAMode fallbackMode) {
  auto srcType = dyn_cast<ShapedType>(src.getType());
  auto dstType = dyn_cast<ShapedType>(dst.getType());
  if (!srcType || !dstType)
    return fallbackMode;

  if (srcType.hasRank() && dstType.hasRank() &&
      succeeded(
          verifyCompatibleShape(srcType.getShape(), dstType.getShape()))) {
    if (dstType.getRank() == 2)
      return FixpipeDMAMode::NZ2ND;
    return FixpipeDMAMode::NZ2NZ;
  }
  return fallbackMode;
}

bool hasCompatibleShape(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  if (!lhsType || !rhsType || !lhsType.hasRank() || !rhsType.hasRank())
    return false;
  if (lhsType.getRank() != rhsType.getRank())
    return false;
  return succeeded(
      verifyCompatibleShape(lhsType.getShape(), rhsType.getShape()));
}

Operation *getInsertPoint_regbase(Operation *op, int &resultIndx) {
  int32_t count = 0;
  auto users = op->getResult(resultIndx).getUsers();
  std::set<scf::YieldOp> yieldOperands;
  for (auto *user : users) {
    if (!isa<hivm::DebugOp>(user))
      count++;
    if (!isa<scf::YieldOp>(user) ||
        !needYieldOut(user, op->getResult(resultIndx))) {
      continue;
    } else {
      yieldOperands.emplace(user);
    }
  }
  if (count > 1)
    return op;

  if (yieldOperands.empty()) {
    return op;
  }

  if (yieldOperands.size() > 1) {
    op->emitError("unsupport cases");
    return op;
  }
  auto yieldOperand = *yieldOperands.begin();
  auto yieldParentOp = yieldOperand->getParentOp();
  auto yieldValueIndx = findIdx(llvm::to_vector(yieldOperand->getOperands()),
                                op->getResult(resultIndx));
  if (!yieldValueIndx.has_value())
    llvm_unreachable("yield value must have user");
  resultIndx = yieldValueIndx.value();
  return getInsertPoint_regbase(yieldParentOp, resultIndx);
}

template <typename OpType>
struct InsertFixpipeOpPattern_membase : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto mmadLikeOpRes = op.getResultTensors()[0];

    if (op.shouldDecomposeBiasByElementAdd()) {
      return failure();
    }

    if (op->getAttr(mmadFixpipeForResultAlreadyInserted))
      return failure();

    auto isMatchedOp = [](Operation *op, Value v) {
      LDBG("Matching this current op " << *op);
      if (isLocalMatmulInit(op, v)) {
        return true;
      }
      return false;
    };
    if (traceSingleChainUser(mmadLikeOpRes, isMatchedOp))
      return failure();

    int resultIndx = 0;
    Operation *insertAfterOp = nullptr;
    if (isAccumulation(op)) {
      insertAfterOp = getInsertPoint_membase(op, resultIndx);
    } else {
      insertAfterOp = op;
    }
    rewriter.setInsertionPointAfter(insertAfterOp);

    LDBG("Replacing fix pipe for " << op);
    insertFixpipe_membase(rewriter, insertAfterOp,
                  insertAfterOp->getResult(resultIndx));
    op->setAttr(mmadFixpipeForResultAlreadyInserted,
                rewriter.getBoolAttr(true));

    if (isAccumulation(op)) {
      scf::ForOp forOp = op->template getParentOfType<scf::ForOp>();
      SmallVector<OpOperand *, 4> inLoopVecOperands;
      for (OpOperand &use : mmadLikeOpRes.getUses()) {
        Operation *user = use.getOwner();
        if (isa<scf::YieldOp>(user))
          continue;
        if (user->getParentOfType<scf::ForOp>() != forOp)
          continue;
        FailureOr<TCoreType> coreType = getCoreType(user);
        if (failed(coreType) || *coreType != TCoreType::VECTOR)
          continue;
        inLoopVecOperands.push_back(&use);
      }
      if (!inLoopVecOperands.empty()) {
        rewriter.setInsertionPointAfter(op);
        MLIRContext *ctx = rewriter.getContext();
        FixpipeDMAModeAttr dmaModeAttr =
            FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
        Value innerInit =
            utils::createEmptyOp(rewriter, op.getLoc(), mmadLikeOpRes);
        auto innerFixpipe = rewriter.create<FixpipeOp>(
            op.getLoc(), /*result_tensor=*/innerInit.getType(),
            /*src=*/mmadLikeOpRes,
            /*dst=*/innerInit, dmaModeAttr, FixpipeDualDstModeAttr{},
            /*pre_quant=*/nullptr, /*pre_relu=*/nullptr,
            /*channel_split=*/nullptr);
        for (OpOperand *operand : inLoopVecOperands) {
          rewriter.modifyOpInPlace(operand->getOwner(),
              [&]() { operand->set(innerFixpipe.getResultTensor()); });
        }
        LDBG("Insert in-loop fixpipe for Vector consumer of accumulation mmad");
      }
    }
    return success();
  }
};

template <typename ConvOp>
struct InsertFixpipeForConvOpPattern : public OpRewritePattern<ConvOp> {
public:
  using OpRewritePattern<ConvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter &rewriter) const override {

    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op->getAttr(mmadFixpipeForResultAlreadyInserted))
      return failure();

    auto result = op.getResultTensors()[0];
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType) {
      return failure();
    }

    auto init = op.getInit();
    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!initType) {
      return failure();
    }

    auto elementType = resultType.getElementType();

    int resultIndx = 0;
    Operation *insertAfterOp = nullptr;
    if (isAccumulation(op)) {
      insertAfterOp = getInsertPoint_membase(op, resultIndx);
    } else {
      insertAfterOp = op;
    }
    rewriter.setInsertionPointAfter(insertAfterOp);
    Location loc = insertAfterOp->getLoc();

    Value fixpipeInit = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);

    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);

    auto fixpipeOp = rewriter.create<FixpipeOp>(
        loc,
        fixpipeInit.getType(),
        insertAfterOp->getResult(resultIndx),
        fixpipeInit,
        dmaModeAttr,
        FixpipeDualDstModeAttr{},
        /*pre_quant=*/nullptr,
        /*pre_relu=*/nullptr,
        /*channel_split=*/nullptr
    );

    rewriter.replaceAllUsesExcept(insertAfterOp->getResult(resultIndx),
                                  fixpipeOp.getResultTensor(), fixpipeOp);

    op->setAttr(mmadFixpipeForResultAlreadyInserted, rewriter.getBoolAttr(true));
    return success();
  }
};

template <typename OpType>
struct InsertFixpipeOpPattern_regbase : public OpRewritePattern<OpType> {
public:
  InsertFixpipeOpPattern_regbase(MLIRContext *context,
                         InsertFixpipePatternOptions options)
      : OpRewritePattern<OpType>(context), options(options) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto mmadLikeOpRes = op.getResultTensors()[0];
    if (op.shouldDecomposeBiasByElementAdd() && !op.isInitConstant(true)) {
      return failure();
    }

    if (op->getAttr(mmadFixpipeForResultAlreadyInserted))
      return failure();

    auto isMatchedOp = [](Operation *op, Value v) {
      LDBG("Matching this current op " << *op);
      if (isa<hivm::FixpipeOp>(op)) {
        return true;
      }
      if (isLocalMatmulInit(op, v)) {
        return true;
      }
      return false;
    };
    if (traceSingleChainUser(mmadLikeOpRes, isMatchedOp))
      return failure();

    int resultIndx = 0;
    auto insertAfterOp = getInsertPoint_regbase(op, resultIndx);
    rewriter.setInsertionPointAfter(insertAfterOp);

    LDBG("Replacing fix pipe for " << op);
    insertFixpipe_regbase(rewriter, options, insertAfterOp,
                  insertAfterOp->getResult(resultIndx));
    op->setAttr(mmadFixpipeForResultAlreadyInserted,
                rewriter.getBoolAttr(true));
    return success();
  }

private:
  InsertFixpipePatternOptions options;
};

template <>
struct InsertFixpipeOpPattern_regbase<hivm::MmadMxL1Op>
    : public OpRewritePattern<hivm::MmadMxL1Op> {
public:
  InsertFixpipeOpPattern_regbase(MLIRContext *context,
                         InsertFixpipePatternOptions options)
      : OpRewritePattern<hivm::MmadMxL1Op>(context), options(options) {}

  LogicalResult matchAndRewrite(hivm::MmadMxL1Op op,
                                PatternRewriter &rewriter) const override {
    auto mmadLikeOpRes = op->getResults()[0];

    auto isMatchedOp = [](Operation *op, Value v) {
      LDBG("Matching this current op " << *op);
      if (isa<hivm::FixpipeOp>(op)) {
        return true;
      }
      if (isLocalMatmulInit(op, v)) {
        return true;
      }
      return false;
    };
    if (traceSingleChainUser(mmadLikeOpRes, isMatchedOp))
      return failure();

    int resultIndx = 0;
    auto insertAfterOp = getInsertPoint_regbase(op, resultIndx);
    rewriter.setInsertionPointAfter(insertAfterOp);

    Value fixpipeInit =
        utils::createEmptyOp(rewriter, insertAfterOp->getLoc(), mmadLikeOpRes);
    LDBG("Replacing fix pipe for " << op);
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr = FixpipeDMAModeAttr::get(
        ctx,
        getInsertedFixpipeDmaMode(insertAfterOp->getResult(resultIndx),
                                  fixpipeInit, options.inferFixpipeDmaMode));
    auto res = rewriter.create<FixpipeOp>(
        op.getLoc(), /*result_tensor=*/fixpipeInit.getType(),
        /*src=*/insertAfterOp->getResult(resultIndx),
        /*dst=*/fixpipeInit, dmaModeAttr, /*dual_dst_mode=*/nullptr,
        /*pre_quant=*/nullptr, /*pre_relu=*/nullptr, /*channel_split=*/nullptr);
    rewriter.replaceAllUsesExcept(insertAfterOp->getResult(resultIndx),
                                  res.getResultTensor(), res);
    return success();
  }

private:
  InsertFixpipePatternOptions options;
};

std::optional<FixpipePreQuantMode> getQuantMode(hivm::VCastOp castOp) {
  auto inputType = getElementTypeOrSelf(castOp.getSrc()[0].getType());
  auto outputType = getElementTypeOrSelf(castOp.getDst()[0].getType());
  if (inputType.isF32() && outputType.isF16()) {
    return symbolizeFixpipePreQuantMode("F322F16");
  }
  if (inputType.isF32() && outputType.isBF16()) {
    return symbolizeFixpipePreQuantMode("F322BF16");
  }
  if (inputType.isInteger(32) && outputType.isInteger(8)) {
    return symbolizeFixpipePreQuantMode("S322I8");
  }
  return std::nullopt;
}

bool isActivationOp(Operation *op) { return isa<hivm::VReluOp>(op); }

template <typename OpType>
std::optional<FixpipePreReluMode> getReluMode_membase(OpType op) {
  if constexpr (std::is_same_v<OpType, hivm::VReluOp>) {
    return hivm::symbolizeFixpipePreReluMode("NORMAL_RELU");
  }
  llvm::report_fatal_error("unsupported ReluValue");
}

template <typename OpType>
std::optional<FixpipePreReluMode> getReluMode_regbase(OpType op) {
  if constexpr (std::is_same_v<OpType, hivm::VReluOp>) {
    return hivm::symbolizeFixpipePreReluMode("NORMAL_RELU");
  }
  llvm_unreachable("unsupported ReluValue");
}

Type getInitType_membase(Value v, hivm::FixpipePreQuantMode quant,
                 PatternRewriter &rewriter) {
  if (quant == FixpipePreQuantMode ::NO_QUANT)
    return getElementTypeOrSelf(v);
  if (quant == FixpipePreQuantMode ::F322F16)
    return rewriter.getF16Type();
  if (quant == FixpipePreQuantMode ::F322BF16)
    return rewriter.getBF16Type();
  if (quant == FixpipePreQuantMode::S322I8)
    return rewriter.getI8Type();
  llvm::report_fatal_error("unsupported QuantMode");
}

Type getInitType_regbase(Value v, hivm::FixpipePreQuantMode quant,
                 PatternRewriter &rewriter) {
  if (quant == FixpipePreQuantMode::NO_QUANT)
    return getElementTypeOrSelf(v);
  if (quant == FixpipePreQuantMode::F322F16)
    return rewriter.getF16Type();
  if (quant == FixpipePreQuantMode::F322BF16)
    return rewriter.getBF16Type();
  if (quant == FixpipePreQuantMode::S322I8)
    return rewriter.getI8Type();
  if (quant == FixpipePreQuantMode::QF322F32_PRE)
    return rewriter.getF32Type();
  llvm_unreachable("unsupported QuantMode");
}

int64_t getSiftedUsersNum_membase(Value v) {
  const DenseSet<Operation *> container(v.getUsers().begin(),
                                        v.getUsers().end());
  auto filteredRange = llvm::make_filter_range(container, [](Operation *op) {
    return !isa<tensor::DimOp>(op) && !isa<hivm::DebugOp>(op);
  });
  return DenseSet<Operation *>(filteredRange.begin(), filteredRange.end())
      .size();
}

int64_t getSiftedUsersNum_regbase(Value v) {
  const DenseSet<Operation *> container(v.getUsers().begin(),
                                        v.getUsers().end());
  auto filteredRange = llvm::make_filter_range(container, [](Operation *op) {
    return !isa<annotation::MarkOp, hivm::DebugOp, tensor::DimOp>(op);
  });
  return DenseSet<Operation *>(filteredRange.begin(), filteredRange.end())
      .size();
}

//===----------------------------------------------------------------------===//
// InlineFixpipeOpPattern_membase
//===----------------------------------------------------------------------===//
struct InlineFixpipeOpPattern_membase : public OpRewritePattern<FixpipeOp> {
public:
  using OpRewritePattern<FixpipeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResultTensor())
      return failure();

    auto fixpipeResTensor = op.getResultTensor();
    if (fixpipeResTensor.getUsers().empty())
      return failure();

    if (getSiftedUsersNum_membase(fixpipeResTensor) != 1)
      return failure();

    return inlineFixpipeOp(rewriter, op);
  }

private:
  LogicalResult inlineFixpipeOp(PatternRewriter &rewriter, FixpipeOp op) const {
    bool matched = false;
    auto loc = op.getLoc();
    Operation *curOp = nullptr;
    for (Operation *maybeDebugOp : op.getResultTensor().getUsers()) {
      if (isa<hivm::DebugOp>(maybeDebugOp) && !op->getAttr(usedForDebugOp)) {
        rewriter.setInsertionPoint(maybeDebugOp);
        FixpipeOp clonedFixpipeOp = cast<FixpipeOp>(rewriter.clone(*op));
        clonedFixpipeOp->setAttr(usedForDebugOp, rewriter.getBoolAttr(true));
        Value clonedResult = clonedFixpipeOp->getResult(0);
        hivm::DebugOp debugOp = cast<hivm::DebugOp>(maybeDebugOp);
        rewriter.modifyOpInPlace(
            debugOp, [&]() { debugOp.getArgMutable().assign(clonedResult); });
      } else {
        curOp = maybeDebugOp;
      }
    }
    if (curOp == nullptr)
      return success();

    auto castOp = dyn_cast_if_present<hivm::VCastOp>(curOp);
    if (op.getFixpipeState() <= op.needFixpipePreFuse() && castOp &&
        getQuantMode(castOp).has_value()) {
      matched = true;
      inlineFixPipeWithRreQuant(rewriter, loc, op, castOp,
                                op.getDpsInputOperand(0)->get());
    } else if (op.getFixpipeState() <= op.needFixpipePreFuse() &&
               isActivationOp(curOp)) {
      matched = true;
      auto reluOp = llvm::dyn_cast_if_present<hivm::VReluOp>(curOp);
      inlineFixPipeWithRreRelu(rewriter, loc, op, reluOp);
    } else if (auto storeOp = llvm::dyn_cast_if_present<hivm::StoreOp>(curOp)) {
      auto storeAttr = storeOp.getAtomicKindAttr();
      hivm::AtomicKind atomicKind = hivm::AtomicKind::NONE;
      if (storeAttr)
        atomicKind = storeAttr.getValue();
      if (atomicKind == AtomicKind::NONE || atomicKind == AtomicKind::ADD ||
          atomicKind == AtomicKind::MAX || atomicKind == AtomicKind::MIN) {
        matched = true;
        inlineFixPipeWithStoreOp(rewriter, loc, op, storeOp,
                                 op.getDpsInputOperand(0)->get());
      }
    } else if (auto extractSliceOp =
                   dyn_cast_if_present<tensor::ExtractSliceOp>(curOp)) {
      if (op->getBlock() == extractSliceOp->getBlock()) {
        matched = true;
        swapFixpipeAndExtractSliceOp(rewriter, loc, op, extractSliceOp);
      }
    } else if (auto insertSliceOp =
                   dyn_cast_if_present<tensor::InsertSliceOp>(curOp)) {
      if (traceDownStoreOpWithSingleChain(insertSliceOp.getResult())) {
        matched = true;
        swapFixpipeAndInsertSliceOp(rewriter, loc, op, insertSliceOp);
      }
    } else if (isa<scf::YieldOp>(curOp) &&
               isa<scf::ForOp>(curOp->getParentOp()) &&
               !op->getAttr(fixpipeDoNotMoveOutOfScfFor)) {
      matched = true;
      auto scfForOp = dyn_cast_if_present<scf::ForOp>(curOp->getParentOp());
      moveFixpipeOutOfScfFor(rewriter, loc, op, scfForOp, op.getResultTensor());
    }
    return matched ? success() : failure();
  }

  void inlineFixPipeWithRreQuant(PatternRewriter &rewriter, Location loc,
                                 hivm::FixpipeOp op, hivm::VCastOp castOp,
                                 Value newFixpipeSrcTensor) const {
    std::optional<FixpipePreQuantMode> quantMode = getQuantMode(castOp);
    auto quantModeAttr =
        FixpipePreQuantModeAttr::get(op.getContext(), quantMode.value());
    auto reluModeAttr = op.getPreReluAttr();

    rewriter.setInsertionPointAfter(castOp);
    Value fixpipeInit =
        utils::createEmptyOp(rewriter, loc, castOp.getResult()[0]);
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        loc, fixpipeInit.getType(), /*src=*/newFixpipeSrcTensor,
        /*dst=*/fixpipeInit, dmaModeAttr, FixpipeDualDstModeAttr{},
        quantModeAttr, reluModeAttr);
    rewriter.replaceAllUsesWith(castOp.getResult()[0],
                                newFixpipeOp.getResultTensor());
    rewriter.eraseOp(castOp);
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithPreQuant");
  }

  void inlineFixPipeWithRreRelu(PatternRewriter &rewriter, Location loc,
                                hivm::FixpipeOp op,
                                hivm::VReluOp reluOp) const {
    std::optional<FixpipePreReluMode> reluMode = getReluMode_membase(reluOp);
    rewriter.modifyOpInPlace(op, [&]() { op.setPreRelu(reluMode); });
    rewriter.replaceAllUsesWith(reluOp.getResult()[0], op.getResult(0));
    rewriter.eraseOp(reluOp);
    LDBG("InlineFixpipeWithPreRelu");
  }

  void inlineFixPipeWithStoreOp(PatternRewriter &rewriter, Location loc,
                                hivm::FixpipeOp op, hivm::StoreOp storeOp,
                                Value fixpipeSrcTensor) const {
    assert(storeOp->getNumResults() == 0 && "StoreOp must have 0 results");
    rewriter.setInsertionPointAfter(storeOp);
    auto dst = storeOp.getDst();
    auto storeAttr = storeOp.getAtomicKindAttr();
    auto noneAtomicAttr =
        AtomicKindAttr::get(op->getContext(), ::mlir::hivm::AtomicKind::NONE);
    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
        loc, TypeRange{}, fixpipeSrcTensor, dst, op.getDmaModeAttr(),
        FixpipeDualDstModeAttr{}, op.getPreQuantAttr(), op.getPreReluAttr());
    if (storeAttr) {
      auto typeAttr =
          TypeAttr::get(mlir::cast<ShapedType>(dst.getType()).getElementType());
      rewriter.setInsertionPoint(newFixpipeOp);
      rewriter.create<SetAtomicOp>(loc, storeAttr, typeAttr);
      rewriter.setInsertionPointAfter(newFixpipeOp);
      rewriter.create<SetAtomicOp>(loc, noneAtomicAttr, typeAttr);
    }
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeEnd");
  }

  void
  swapFixpipeAndExtractSliceOp(PatternRewriter &rewriter, Location loc,
                               hivm::FixpipeOp op,
                               tensor::ExtractSliceOp extractSliceOp) const {
    rewriter.setInsertionPointAfter(extractSliceOp);
    auto fixpipeSrc = op.getDpsInputOperand(0)->get();

    auto newExtractSliceResType =
        extractSliceOp.getResultType().clone(getElementTypeOrSelf(fixpipeSrc));
    auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        extractSliceOp.getLoc(), newExtractSliceResType, fixpipeSrc,
        extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());

    auto newExtractSliceResult = newExtractSliceOp->getResult(0);
    auto quantModeAttr = op.getPreQuantAttr();
    auto reluModeAttr = op.getPreReluAttr();
    Value fixpipeInit = nullptr;
    fixpipeInit = utils::createEmptyOpWithTargetElemType(
        rewriter, extractSliceOp.getLoc(), newExtractSliceResult,
        getInitType_membase(newExtractSliceResult, op.getPreQuant(), rewriter));

    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        extractSliceOp.getLoc(), fixpipeInit.getType(),
        /*src=*/newExtractSliceResult, /*dst=*/fixpipeInit, dmaModeAttr,
        FixpipeDualDstModeAttr{}, quantModeAttr, reluModeAttr);
    rewriter.replaceOp(extractSliceOp, newFixpipeOp.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithExtractSliceReshape");
  }

  void swapFixpipeAndInsertSliceOp(PatternRewriter &rewriter, Location loc,
                                   hivm::FixpipeOp op,
                                   tensor::InsertSliceOp insertSliceOp) const {
    rewriter.setInsertionPointAfter(insertSliceOp);
    auto fixpipeSrc = op.getDpsInputOperand(0)->get();

    auto newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        insertSliceOp.getLoc(), fixpipeSrc, insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());

    auto newInsertSliceResult = newInsertSliceOp->getResult(0);
    auto quantModeAttr = op.getPreQuantAttr();
    auto reluModeAttr = op.getPreReluAttr();
    Value fixpipeInit = utils::createEmptyOpWithTargetElemType(
        rewriter, insertSliceOp.getLoc(), newInsertSliceResult,
        getInitType_membase(newInsertSliceResult, op.getPreQuant(), rewriter));
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        insertSliceOp.getLoc(), TypeRange{fixpipeInit}, newInsertSliceResult,
        fixpipeInit, dmaModeAttr, FixpipeDualDstModeAttr{}, quantModeAttr,
        reluModeAttr);
    rewriter.replaceOp(insertSliceOp, newFixpipeOp.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithInsertSliceOpReshape");
  }

  bool traceDownStoreOpWithSingleChain(Value v) const {
    auto isMachedOp = [](Operation *op, Value v) {
      return isa<hivm::StoreOp>(op);
    };
    return traceSingleChainUser(v, isMachedOp);
  }

  void moveFixpipeOutOfScfFor(PatternRewriter &rewriter, Location loc,
                              hivm::FixpipeOp fixPipeOp, scf::ForOp scfForOp,
                              Value fixpipeResTensor) const {
    SmallVector<Value> yieldValues =
        llvm::to_vector(scfForOp.getYieldedValues());
    auto idx = findIdx(yieldValues, fixpipeResTensor);
    if (idx.has_value()) {
      LDBG("InlineFixpipeWithYield");
      rewriter.replaceAllUsesWith(fixpipeResTensor,
                                  fixPipeOp.getDpsInputOperand(0)->get());

      rewriter.setInsertionPointAfter(scfForOp);
      auto fixpipeInit =
          utils::createEmptyOp(rewriter, scfForOp->getLoc(), fixpipeResTensor);
      auto quantModeAttr = fixPipeOp.getPreQuantAttr();
      auto reluModeAttr = fixPipeOp.getPreReluAttr();
      MLIRContext *ctx = rewriter.getContext();
      FixpipeDMAModeAttr dmaModeAttr =
          FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
      auto newFixpipeOp = rewriter.create<FixpipeOp>(
          fixPipeOp.getLoc(), TypeRange{fixpipeInit},
          scfForOp->getResult(idx.value()), fixpipeInit, dmaModeAttr,
          FixpipeDualDstModeAttr{}, quantModeAttr, reluModeAttr);
      rewriter.replaceAllUsesExcept(scfForOp->getResult(idx.value()),
                                    newFixpipeOp.getResultTensor(),
                                    newFixpipeOp);
    }
    LDBG("moveFixpipeOutOfScfFor");
  }
};

//===----------------------------------------------------------------------===//
// InlineFixpipeOpPattern_regbase
//===----------------------------------------------------------------------===//
struct InlineFixpipeOpPattern_regbase : public OpRewritePattern<FixpipeOp> {
public:
  InlineFixpipeOpPattern_regbase(MLIRContext *ctx, InlineFixpipePatternOptions options)
      : OpRewritePattern<FixpipeOp>(ctx), options(options) {}

  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResultTensor())
      return failure();

    auto fixpipeResTensor = op.getResultTensor();
    if (fixpipeResTensor.getUsers().empty())
      return failure();

    if (getSiftedUsersNum_regbase(fixpipeResTensor) != 1)
      return failure();

    return inlineFixpipeOp(rewriter, op);
  }

private:
  LogicalResult inlineFixpipeOp(PatternRewriter &rewriter, FixpipeOp op) const {
    bool matched = false;
    auto loc = op.getLoc();
    Operation *curOp = nullptr;
    for (Operation *maybeDebugOp : op.getResultTensor().getUsers()) {
      if (isa<hivm::DebugOp>(maybeDebugOp)) {
        rewriter.setInsertionPoint(maybeDebugOp);
        FixpipeOp clonedFixpipeOp = cast<FixpipeOp>(rewriter.clone(*op));
        Value clonedResult = clonedFixpipeOp.getResult(0);
        hivm::DebugOp debugOp = cast<hivm::DebugOp>(maybeDebugOp);
        rewriter.modifyOpInPlace(
            debugOp, [&]() { debugOp.getArgMutable().assign(clonedResult); });
      } else {
        curOp = maybeDebugOp;
      }
    }
    if (curOp == nullptr)
      return success();

    auto castOp = dyn_cast_if_present<hivm::VCastOp>(curOp);
    if (op.getFixpipeState() <= op.needFixpipePreFuse() && castOp &&
        getQuantMode(castOp).has_value()) {
      matched = true;
      inlineFixPipeWithRreQuant(rewriter, loc, op, castOp,
                                op.getDpsInputOperand(0)->get());
    } else if (op.getFixpipeState() <= op.needFixpipePreFuse() &&
               isActivationOp(curOp)) {
      matched = true;
      auto reluOp = llvm::dyn_cast_if_present<hivm::VReluOp>(curOp);
      inlineFixPipeWithRreRelu(rewriter, loc, op, reluOp);
    } else if (auto storeOp = llvm::dyn_cast_if_present<hivm::StoreOp>(curOp)) {
      auto storeAttr = storeOp.getAtomicKindAttr();
      hivm::AtomicKind atomicKind = hivm::AtomicKind::NONE;
      if (storeAttr)
        atomicKind = storeAttr.getValue();
      if (atomicKind == AtomicKind::NONE || atomicKind == AtomicKind::ADD ||
          atomicKind == AtomicKind::MAX || atomicKind == AtomicKind::MIN) {
        matched = true;
        inlineFixPipeWithStoreOp(rewriter, loc, op, storeOp,
                                 op.getDpsInputOperand(0)->get());
      }
    } else if (auto vMulOp = dyn_cast<hivm::VMulOp>(curOp);
               vMulOp &&
               (options.inlineQuantScale || hasQuantScaleCompileHint(vMulOp)) &&
               isUserQuantScaleInlinable(op, vMulOp)) {
      matched = true;
      inlineFixPipeWithQuantScale(rewriter, op, vMulOp);
    } else if (isUserTransposeInlineable(op, curOp)) {
      auto vTransposeOp = cast<hivm::VTransposeOp>(curOp);
      matched = true;
      inlineFixPipeWithTranspose(rewriter, op, vTransposeOp);
    } else if (isa<tensor::ExtractSliceOp>(curOp) &&
               hasCompatibleShape(
                   op.getSource(),
                   cast<tensor::ExtractSliceOp>(curOp).getSource())) {
      auto extractSliceOp = cast<tensor::ExtractSliceOp>(curOp);
      if (op->getBlock() == extractSliceOp->getBlock()) {
        matched = true;
        swapFixpipeAndExtractSliceOp(rewriter, loc, op, extractSliceOp);
      }
    } else if (isa<tensor::InsertSliceOp>(curOp) &&
               hasCompatibleShape(
                   op.getSource(),
                   cast<tensor::InsertSliceOp>(curOp).getSource())) {
      auto insertSliceOp = cast<tensor::InsertSliceOp>(curOp);
      if (traceDownStoreOpWithSingleChain(insertSliceOp.getResult())) {
        matched = true;
        swapFixpipeAndInsertSliceOp(rewriter, loc, op, insertSliceOp);
      }
    } else if (isa<scf::YieldOp>(curOp) &&
               isa<scf::ForOp>(curOp->getParentOp()) &&
               !op->getAttr(fixpipeDoNotMoveOutOfScfFor)) {
      matched = true;
      auto scfForOp = dyn_cast_if_present<scf::ForOp>(curOp->getParentOp());
      moveFixpipeOutOfScfFor(rewriter, loc, op, scfForOp, op.getResultTensor());
    }
    return matched ? success() : failure();
  }

  void inlineFixPipeWithRreQuant(PatternRewriter &rewriter, Location loc,
                                 hivm::FixpipeOp op, hivm::VCastOp castOp,
                                 Value newFixpipeSrcTensor) const {
    std::optional<FixpipePreQuantMode> quantMode = getQuantMode(castOp);
    if (!quantMode) {
      LDBG("cast op quant mode is null");
      return;
    }
    auto quantModeAttr =
        FixpipePreQuantModeAttr::get(op.getContext(), quantMode.value());

    rewriter.setInsertionPointAfter(castOp);
    Value fixpipeInit =
        utils::createEmptyOp(rewriter, loc, castOp.getResult()[0]);
    FixpipeDMAModeAttr dmaModeAttr = op.getDmaModeAttr();
    SmallVector<Value> oprs({newFixpipeSrcTensor, fixpipeInit});
    if (auto quantScale = op.getQuantScale())
      oprs.push_back(quantScale);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        op.getLoc(), fixpipeInit.getType(), oprs, op->getAttrs());
    newFixpipeOp.setDmaModeAttr(dmaModeAttr);
    newFixpipeOp.setPreQuantAttr(quantModeAttr);
    rewriter.replaceAllUsesWith(castOp.getResult()[0],
                                newFixpipeOp.getResultTensor());
    rewriter.eraseOp(castOp);
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithPreQuant");
  }

  void inlineFixPipeWithRreRelu(PatternRewriter &rewriter, Location loc,
                                hivm::FixpipeOp op,
                                hivm::VReluOp reluOp) const {
    std::optional<FixpipePreReluMode> reluMode = getReluMode_regbase(reluOp);
    rewriter.modifyOpInPlace(op, [&]() { op.setPreRelu(reluMode); });
    rewriter.replaceAllUsesWith(reluOp.getResult()[0], op.getResult(0));
    rewriter.eraseOp(reluOp);
    LDBG("InlineFixpipeWithPreRelu");
  }

  void inlineFixPipeWithStoreOp(PatternRewriter &rewriter, Location loc,
                                hivm::FixpipeOp op, hivm::StoreOp storeOp,
                                Value fixpipeSrcTensor) const {
    assert(storeOp->getNumResults() == 0 && "StoreOp must have 0 results");
    rewriter.setInsertionPointAfter(storeOp);
    auto dst = storeOp.getDst();
    auto storeAttr = storeOp.getAtomicKindAttr();
    auto noneAtomicAttr =
        AtomicKindAttr::get(op->getContext(), ::mlir::hivm::AtomicKind::NONE);
    SmallVector<Value> oprs({fixpipeSrcTensor, dst});
    if (auto quantScale = op.getQuantScale())
      oprs.push_back(quantScale);
    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(loc, TypeRange{}, oprs,
                                                         op->getAttrs());
    if (storeAttr) {
      auto typeAttr =
          TypeAttr::get(mlir::cast<ShapedType>(dst.getType()).getElementType());
      rewriter.setInsertionPoint(newFixpipeOp);
      rewriter.create<SetAtomicOp>(loc, storeAttr, typeAttr);
      rewriter.setInsertionPointAfter(newFixpipeOp);
      rewriter.create<SetAtomicOp>(loc, noneAtomicAttr, typeAttr);
    }
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeEnd");
  }

  FixpipePreQuantMode
  inferPreQuantMode(hivm::FixpipeOp op,
                    std::optional<hivm::VMulOp> vMulOp) const {
    LDBG("inferring pre quant mode");
    auto curPreQuant = op.getPreQuant();
    if (curPreQuant != FixpipePreQuantMode::NO_QUANT)
      return curPreQuant;
    Type srcElemType = getElementTypeOrSelf(op.getSrcOperandType());
    Type dstElemType = getElementTypeOrSelf(op.getDstOperandType());
    if (srcElemType.isF32() && dstElemType.isF32() &&
        op.getDualDstMode().getDualDstMode() == FixpipeDualDstMode::NO_DUAL)
      return FixpipePreQuantMode::QF322F32_PRE;
    return curPreQuant;
  }

  bool hasQuantScaleCompileHint(hivm::VMulOp op) const {
    return any_of(op->getUsers(), [](Operation *userOp) {
      auto markOp = dyn_cast<annotation::MarkOp>(userOp);
      if (!markOp)
        return false;
      return markOp->hasAttr(utils::kInlinableQuantScaleAttr);
    });
  }

  bool isUserQuantScaleInlinable(hivm::FixpipeOp op, Operation *userOp) const {
    auto vMulOp = dyn_cast<hivm::VMulOp>(userOp);
    if (!vMulOp)
      return false;
    if (op.getDualDstMode().getDualDstMode() != FixpipeDualDstMode::NO_DUAL)
      return false;
    if (op.getQuantScale())
      return false;
    if (llvm::count_if(userOp->getUsers(), [](Operation *afterVMulOp) {
          return !isa<annotation::MarkOp>(afterVMulOp);
        }) != 1)
      return false;
    if (!traceDownStoreOpWithSingleChain(userOp->getResult(0)))
      return false;
    Value quantScaleValue =
        vMulOp
            .getDpsInputOperand(static_cast<unsigned>(
                vMulOp.getDpsInputOperand(0)->get().getDefiningOp() == op))
            ->get();
    if (!utils::isScalarLike(quantScaleValue))
      return false;
    return true;
  }

  bool isUserTransposeInlineable(hivm::FixpipeOp op, Operation *userOp) const {
    auto vTransposeOp = dyn_cast<hivm::VTransposeOp>(userOp);
    if (!vTransposeOp)
      return false;
    auto dmaModeAttr = op.getDmaModeAttr();
    bool isNZ2ND = (dmaModeAttr && dmaModeAttr.getValue() == FixpipeDMAMode::NZ2ND);
    if (!isNZ2ND)
      return false;
    ArrayRef<int64_t> permutation = vTransposeOp.getPermutation();
    if (permutation[0] != 1 || permutation[1] != 0)
      return false;
    return true;
  }

  void inlineFixPipeWithQuantScale(PatternRewriter &rewriter,
                                   hivm::FixpipeOp op,
                                   hivm::VMulOp vMulOp) const {
    assert(op.getDualDstMode().getDualDstMode() ==
               FixpipeDualDstMode::NO_DUAL &&
           "illegal fixpipe config check ISA");
    LDBG("inling fixpipe with quant scale");

    rewriter.setInsertionPointAfter(vMulOp);
    assert(vMulOp.getNumDpsInits() == 1);
    Value vMulDst = *vMulOp.getDst().begin();
    assert(vMulOp.getNumDpsInputs() == 2);
    Value quantScaleValue =
        vMulOp
            .getDpsInputOperand(static_cast<unsigned>(
                vMulOp.getDpsInputOperand(0)->get().getDefiningOp() == op))
            ->get();
    assert(utils::isScalarLike(quantScaleValue) &&
           "currently, only handle scalar quant scale");
    Value fixPipeSrc = op.getSource();
    auto quantPreMode = inferPreQuantMode(op, vMulOp);
    LDBG("inferred pre quant mode id:" << quantPreMode);
    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
        op.getLoc(), vMulDst.getType(), fixPipeSrc, vMulDst,
        op.getUnitFlagCond(), op.getDmaModeAttr(), op.getDualDstModeAttr(),
        FixpipePreQuantModeAttr::get(rewriter.getContext(), quantPreMode),
        op.getPreReluAttr(), op.getChannelSplitAttr(), op.getUnitFlagModeAttr(),
        quantScaleValue);
    for (auto *user : llvm::make_early_inc_range(vMulOp->getUsers()))
      if (isa<annotation::MarkOp>(user)) {
        newFixpipeOp->setAttr(utils::kInlinedQuantScaleAttr,
                              rewriter.getUnitAttr());
        rewriter.eraseOp(user);
      }
    rewriter.replaceOp(vMulOp, newFixpipeOp);
    rewriter.eraseOp(op);
    LDBG("inlineFixPipeWithPreQuantEnd");
  }


  void inlineFixPipeWithTranspose(PatternRewriter &rewriter,
                                   hivm::FixpipeOp op,
                                   hivm::VTransposeOp vTransOp) const {
    LDBG("inling fixpipe with transpose");

    rewriter.setInsertionPointAfter(vTransOp);
    Value vTransDst = vTransOp.getDst();
    Value fixPipeSrc = op.getSource();
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr NZ2DNAttr = FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2DN);
    SmallVector<Value> oprs({fixPipeSrc, vTransDst});
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        op.getLoc(), vTransDst.getType(), oprs, op->getAttrs());
    newFixpipeOp.setDmaModeAttr(NZ2DNAttr);
    rewriter.replaceOp(vTransOp, newFixpipeOp);
    rewriter.eraseOp(op);
    LDBG("inlineFixPipeWithTranspose End");
  }

  void
  swapFixpipeAndExtractSliceOp(PatternRewriter &rewriter, Location loc,
                               hivm::FixpipeOp op,
                               tensor::ExtractSliceOp extractSliceOp) const {
    rewriter.setInsertionPointAfter(extractSliceOp);
    auto fixpipeSrc = op.getDpsInputOperand(0)->get();

    auto newExtractSliceResType =
        extractSliceOp.getResultType().clone(getElementTypeOrSelf(fixpipeSrc));
    auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        extractSliceOp.getLoc(), newExtractSliceResType, fixpipeSrc,
        extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());

    auto newExtractSliceResult = newExtractSliceOp->getResult(0);
    Value fixpipeInit = utils::createEmptyOpWithTargetElemType(
        rewriter, extractSliceOp.getLoc(), newExtractSliceResult,
        getInitType_regbase(newExtractSliceResult, op.getPreQuant(), rewriter));
    MLIRContext *ctx = rewriter.getContext();
    auto dmaMode = inferSwapFixpipeDmaMode(newExtractSliceResult, fixpipeInit,
                                           op.getDmaMode());
    FixpipeDMAModeAttr dmaModeAttr = FixpipeDMAModeAttr::get(ctx, dmaMode);
    SmallVector<Value> oprs({newExtractSliceResult, fixpipeInit});
    if (auto quantScale = op.getQuantScale())
      oprs.push_back(quantScale);
    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
        extractSliceOp.getLoc(), fixpipeInit.getType(), oprs, op->getAttrs());
    newFixpipeOp.setDmaModeAttr(dmaModeAttr);
    rewriter.replaceOp(extractSliceOp, newFixpipeOp.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithExtractSliceReshape");
  }

  void swapFixpipeAndInsertSliceOp(PatternRewriter &rewriter, Location loc,
                                   hivm::FixpipeOp op,
                                   tensor::InsertSliceOp insertSliceOp) const {
    rewriter.setInsertionPointAfter(insertSliceOp);
    auto fixpipeSrc = op.getDpsInputOperand(0)->get();

    auto newInsertSliceOp = rewriter.create<tensor::InsertSliceOp>(
        insertSliceOp.getLoc(), fixpipeSrc, insertSliceOp.getDest(),
        insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
        insertSliceOp.getMixedStrides());

    auto newInsertSliceResult = newInsertSliceOp->getResult(0);
    Value fixpipeInit = utils::createEmptyOpWithTargetElemType(
        rewriter, insertSliceOp.getLoc(), newInsertSliceResult,
        getInitType_regbase(newInsertSliceResult, op.getPreQuant(), rewriter));
    MLIRContext *ctx = rewriter.getContext();
    auto dmaMode = inferSwapFixpipeDmaMode(newInsertSliceResult, fixpipeInit,
                                           op.getDmaMode());
    FixpipeDMAModeAttr dmaModeAttr = FixpipeDMAModeAttr::get(ctx, dmaMode);
    SmallVector<Value> oprs({newInsertSliceResult, fixpipeInit});
    if (auto quantScale = op.getQuantScale())
      oprs.push_back(quantScale);
    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
        insertSliceOp.getLoc(), TypeRange{fixpipeInit}, oprs, op->getAttrs());
    newFixpipeOp.setDmaModeAttr(dmaModeAttr);
    rewriter.replaceOp(insertSliceOp, newFixpipeOp.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithInsertSliceOpReshape");
  }

  bool traceDownStoreOpWithSingleChain(Value v) const {
    auto isMachedOp = [](Operation *op, Value v) {
      return isa<hivm::StoreOp>(op);
    };
    return traceSingleChainUser(v, isMachedOp);
  }

  void moveFixpipeOutOfScfFor(PatternRewriter &rewriter, Location loc,
                              hivm::FixpipeOp fixPipeOp, scf::ForOp scfForOp,
                              Value fixpipeResTensor) const {
    SmallVector<Value> yieldValues =
        llvm::to_vector(scfForOp.getYieldedValues());
    auto idx = findIdx(yieldValues, fixpipeResTensor);
    if (idx.has_value()) {
      LDBG("InlineFixpipeWithYield");
      rewriter.replaceAllUsesWith(fixpipeResTensor,
                                  fixPipeOp.getDpsInputOperand(0)->get());

      rewriter.setInsertionPointAfter(scfForOp);
      auto fixpipeInit =
          utils::createEmptyOp(rewriter, scfForOp->getLoc(), fixpipeResTensor);
      MLIRContext *ctx = rewriter.getContext();
      FixpipeDMAModeAttr dmaModeAttr =
          FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
      SmallVector<Value> oprs({scfForOp->getResult(idx.value()), fixpipeInit});
      if (auto quantScale = fixPipeOp.getQuantScale())
        oprs.push_back(quantScale);
      auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
          fixPipeOp.getLoc(), TypeRange{fixpipeInit}, oprs,
          fixPipeOp->getAttrs());
      newFixpipeOp.setDmaModeAttr(dmaModeAttr);
      rewriter.replaceAllUsesExcept(scfForOp->getResult(idx.value()),
                                    newFixpipeOp.getResultTensor(),
                                    newFixpipeOp);
    }
    LDBG("moveFixpipeOutOfScfFor");
  }

  InlineFixpipePatternOptions options;
};

void mlir::hivm::populateInsertFixpipeForIterArgMMADPattern(
    RewritePatternSet &patterns, InsertFixpipePatternOptions options) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InsertFixpipeForIterArgMMAD_regbase>(ctx, options);
}

void mlir::hivm::populateInsertFixpipePatterns(
    RewritePatternSet &patterns, InsertFixpipePatternOptions options) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InsertFixpipeOpPattern_regbase<hivm::MmadL1Op>>(ctx, options);
  patterns.add<InsertFixpipeOpPattern_regbase<hivm::BatchMmadL1Op>>(ctx, options);
  patterns.add<InsertFixpipeOpPattern_regbase<hivm::MmadMxL1Op>>(ctx, options);
}

void mlir::hivm::populateInlineFixpipePatterns(
    RewritePatternSet &patterns, InlineFixpipePatternOptions options) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InlineFixpipeOpPattern_regbase>(ctx, options);
}

//===----------------------------------------------------------------------===//
// InsertFixpipeForDevicePrint_membase
//===----------------------------------------------------------------------===//
struct InsertFixpipeForDevicePrint_membase : public OpRewritePattern<DebugOp> {
public:
  using OpRewritePattern<DebugOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DebugOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDebugtype() != printType)
      return failure();

    auto maybeMmadRes = op.getArg();
    if (!traceDefOp<MmadL1Op>(maybeMmadRes).has_value() &&
        !traceDefOp<BatchMmadL1Op>(maybeMmadRes).has_value())
      return failure();

    Operation *definingOp = maybeMmadRes.getDefiningOp();
    if (!definingOp)
      return failure();
    rewriter.setInsertionPointAfter(definingOp);
    Location loc = definingOp->getLoc();

    auto resultTensorType =
        mlir::dyn_cast<RankedTensorType>(maybeMmadRes.getType());
    if (!resultTensorType)
      return failure();

    Value workSpaceTensor = getLocalWorkSpaceTensor(
        rewriter, loc, resultTensorType.getShape(),
        hivm::getTensorDynamicValues(rewriter, loc, maybeMmadRes),
        resultTensorType.getElementType());
    auto toTensorOp =
        cast<bufferization::ToTensorOp>(workSpaceTensor.getDefiningOp());
    Value workSpaceMemref = toTensorOp.getMemref();

    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto fixpipeOp = rewriter.create<FixpipeOp>(
        loc, TypeRange{}, maybeMmadRes, workSpaceMemref, dmaModeAttr,
        FixpipeDualDstModeAttr{}, nullptr, nullptr);
    fixpipeOp->setAttr(usedForDebugOp, rewriter.getBoolAttr(true));

    rewriter.modifyOpInPlace(op, [&]() {
      op.getArgMutable().assign(workSpaceTensor);
      op.setMemscopeAttr(hivm::AddressSpaceAttr::get(
        ctx, hivm::AddressSpace::GM));
      op.setTcoretypeAttr(hivm::TCoreTypeAttr::get(
        ctx, hivm::TCoreType::CUBE));
    });
    LDBG("InsertFixpipeForDevicePrint");
    return success();
  }

  bool isUsedByDebugOp(Value v) const {
    for (Operation *user : v.getUsers()) {
      if (isa<DebugOp>(user))
        return true;
    }
    return false;
  }
};

// InsertFixpipeForDevicePrint_regbase
struct InsertFixpipeForDevicePrint_regbase : public OpRewritePattern<DebugOp> {
public:
  using OpRewritePattern<DebugOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DebugOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeMmadRes = op.getArg();
    if (!traceDefOp<MmadL1Op>(maybeMmadRes).has_value() &&
        !traceDefOp<BatchMmadL1Op>(maybeMmadRes).has_value())
      return failure();

    Operation *definingOp = maybeMmadRes.getDefiningOp();
    rewriter.setInsertionPointAfter(definingOp);

    TensorType tensorType = cast<TensorType>(maybeMmadRes.getType());
    Value localWorkSpace = createAllocLocalWorkSpace(
        rewriter, definingOp->getLoc(), tensorType.getShape(),
        getElementTypeOrSelf(tensorType));

    auto toTensor = rewriter.create<bufferization::ToTensorOp>(
        definingOp->getLoc(), localWorkSpace, /*restrict=*/true,
        /*writable=*/true);

    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    rewriter.create<FixpipeOp>(
        op.getLoc(), /*result_tensor=*/TypeRange{},
        /*src=*/maybeMmadRes,
        /*dst=*/localWorkSpace, dmaModeAttr, /*dual_dst_mode=*/nullptr,
        /*pre_quant=*/nullptr, /*pre_relu=*/nullptr, /*channel_split=*/nullptr);
    rewriter.modifyOpInPlace(op, [&]() {
      OpOperand &arg = op.getArgMutable();
      arg.assign(toTensor.getResult());
    });

    return success();
  }
};

void populateDevicePrintPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InsertFixpipeForDevicePrint_regbase>(ctx);
}

void eraseInlinableQuantScaleMarkOps(Operation *op) {
  SmallVector<annotation::MarkOp> inlinableQuantScaleMarkOps;
  op->walk([&](annotation::MarkOp markOp) {
    if (markOp->hasAttrOfType<UnitAttr>(utils::kInlinableQuantScaleAttr))
      inlinableQuantScaleMarkOps.push_back(markOp);
  });
  for (annotation::MarkOp markOp : inlinableQuantScaleMarkOps)
    markOp.erase();
}

void InsertFixpipe::runOnOperation() {
  InsertFixpipePatternOptions options;
  options.inferFixpipeDmaMode = inferFixpipeDmaMode;

  RewritePatternSet iterArgMMAPattern(&getContext());
  mlir::hivm::populateInsertFixpipeForIterArgMMADPattern(iterArgMMAPattern,
                                                         options);

  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(iterArgMMAPattern)))) {
    signalPassFailure();
  }

  RewritePatternSet patterns(&getContext());
  mlir::hivm::populateInsertFixpipePatterns(patterns, options);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }

  // special treatment for device print fixpipe
  RewritePatternSet devicePrintPatterns(&getContext());
  populateDevicePrintPatterns(devicePrintPatterns);

  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(devicePrintPatterns)))) {
    signalPassFailure();
  }
}

void InlineFixpipe::runOnOperation() {
  // membase patterns
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<InsertFixpipeOpPattern_membase<hivm::MmadL1Op>>(
        patterns.getContext());
    patterns.add<InsertFixpipeOpPattern_membase<hivm::BatchMmadL1Op>>(
        patterns.getContext());
    patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv1DL1Op>>(
        patterns.getContext());
    patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv2DL1Op>>(
        patterns.getContext());
    patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv3DL1Op>>(
        patterns.getContext());
    patterns.add<InlineFixpipeOpPattern_membase>(patterns.getContext());
    patterns.add<InsertFixpipeForIterArgMMAD_membase>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  // regbase patterns
  {
    RewritePatternSet patterns(&getContext());
    InlineFixpipePatternOptions options;
    options.inlineQuantScale = inlineQuantScale;
    mlir::hivm::populateInlineFixpipePatterns(patterns, options);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  // device print patterns (membase version)
  {
    RewritePatternSet insertFixpipeForDevicePrintPattern(&getContext());
    MLIRContext *ctx = insertFixpipeForDevicePrintPattern.getContext();
    insertFixpipeForDevicePrintPattern.add<InsertFixpipeForDevicePrint_membase>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(insertFixpipeForDevicePrintPattern)))) {
      signalPassFailure();
    }
  }
}

std::unique_ptr<Pass>
mlir::hivm::createInsertFixpipePass(const InsertFixpipeOptions &options) {
  return std::make_unique<InsertFixpipe>(options);
}

std::unique_ptr<Pass>
mlir::hivm::createInlineFixpipePass(const InlineFixpipeOptions &options) {
  return std::make_unique<InlineFixpipe>(options);
}