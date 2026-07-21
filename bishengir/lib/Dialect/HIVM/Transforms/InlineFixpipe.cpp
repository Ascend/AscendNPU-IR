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
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
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
} // namespace

namespace {
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

Operation *getInsertPoint(Operation *op, int &resultIndx) {
  auto users = op->getResult(resultIndx).getUsers();
  std::set<scf::YieldOp> yieldOperands;
  for (auto *user : users) {
    // TODO: add auto tracedDownUser = traceDown(user) and use tracedDownUser to
    // judge
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
  return getInsertPoint(yieldParentOp, resultIndx);
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

static FixpipeOp insertFixpipe(PatternRewriter &rewriter, Operation *point,
                               Value src) {

  rewriter.setInsertionPointAfter(point);

  auto dst = utils::createEmptyOp(rewriter, point->getLoc(), src);
  MLIRContext *ctx = rewriter.getContext();
  FixpipeDMAModeAttr dmaModeAttr =
      FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);

  auto fixpipe = rewriter.create<FixpipeOp>(
      point->getLoc(), /*result_tensor=*/dst.getType(), src, dst, dmaModeAttr,
      /*dual_dst_mode=*/nullptr,
      /*sub_block_idx=*/nullptr,
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

/// Insert fixpipe after hivm::MmadL1Op inside scf.for when a loop-carried
/// iter_arg used by the mmad is updated from a yield that depends on that mmad.
/// Created fixpipes are tagged do_not_move_out_of_scffor so later patterns do
/// not hoist them out of the loop.
///
/// Example (accumulator iter_arg is mmad outs; yield feeds the next iteration):
///
/// Before:
///   %res = scf.for %i = %c0 to %N step %c1 iter_args(%acc = %init)
///       -> (tensor<32x32xf32>) {
///     %mmad = hivm.hir.mmadL1 ins(%a, %b, %true, %c32, %c32, %c32
///         : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index)
///         outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
///     scf.yield %mmad : tensor<32x32xf32>
///   }
///
/// After:
///   %res = scf.for %i = %c0 to %N step %c1 iter_args(%acc = %init)
///       -> (tensor<32x32xf32>) {
///     %mmad = hivm.hir.mmadL1 {fixpipe_for_result_already_inserted = true}
///         ins(%a, %b, %true, %c32, %c32, %c32 : ...)
///         outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
///     %dst = tensor.empty() : tensor<32x32xf32>
///     %fix = hivm.hir.fixpipe {do_not_move_out_of_scffor = true, dma_mode =
///     #hivm.dma_mode<nz2nd>}
///         ins(%mmad : tensor<32x32xf32>) outs(%dst : tensor<32x32xf32>)
///         -> tensor<32x32xf32>
///     scf.yield %fix : tensor<32x32xf32>
///   } {fixpipe_for_mmad_result_already_inserted = true}
struct InsertFixpipeForIterArgMMAD : public OpRewritePattern<scf::ForOp> {
public:
  explicit InsertFixpipeForIterArgMMAD(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp scffor,
                                PatternRewriter &rewriter) const override {
    if (scffor->getAttr(scfforFixpipeForMMADResultAlreadyInserted)) {
      return failure();
    }

    // Collect hivm::MmadL1Op in the scffor's regions, but do not descend into
    // nested scf::ForOp to keep this rewrite scoped to the current loop level.
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
          // skip loop counterFixpipe
          continue;
        }

        auto yielded = scffor.getYieldedValues()[idx - 1];

        auto stopOp =
            utils::valueCalculatedUsingOperationInsideBlock<hivm::MmadL1Op>(
                yielded, mmad, forBlock);
        if (stopOp && *stopOp == mmad) {
          LDBG("Inserting fix pipe for " << scffor);

          auto fixpipe = insertFixpipe(rewriter, mmad, mmad->getResults()[0]);
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

static bool isRegBasedArch(Operation *op);

/// Check whether every meaningful user reaches a fixpipe, tracing through
/// extract_slice chains. This covers branch fan-out where a single-user-chain
/// query cannot detect that fixpipes have already been inserted.
static bool allUsersReachFixpipe(Value value) {
  SmallVector<Operation *> users;
  for (Operation *user : value.getUsers()) {
    if (isa<tensor::DimOp, annotation::MarkOp>(user))
      continue;
    users.push_back(user);
  }
  if (users.empty())
    return false;
  return llvm::all_of(users, [](Operation *user) {
    if (isa<hivm::FixpipeOp>(user))
      return true;
    if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(user))
      return allUsersReachFixpipe(extractSlice.getResult());
    return false;
  });
}

/// Insert fixpipe when there is hivm::MmadL1Op or hivm::BatchMmadL1Op.
template <typename OpType>
struct InsertFixpipeOpPattern : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto mmadLikeOpRes = op.getResultTensors()[0];

    // shouldDecomposeBiasByElementAdd is true for ElementwiseAdd regardless of
    // init; NormalizeMatmul only decomposes when init is const false, or
    // non-const on reg-based. Skip fixpipe only in those cases (not init=true).
    if (op.shouldDecomposeBiasByElementAdd() &&
        (op.isInitConstant(false) ||
         (!op.isInitConstant() && isRegBasedArch(op)))) {
      // the op will decompose to mmadL1 + vadd, so fixpipe cannot be inserted
      // now, and fixpipe should be inserted after the decomposition
      return failure();
    }

    if (op->getAttr(mmadFixpipeForResultAlreadyInserted))
      return failure();

    if (isRegBasedArch(op) && allUsersReachFixpipe(mmadLikeOpRes))
      return failure();

    auto isMatchedOp = [](Operation *op, Value v) {
      LDBG("Matching this current op " << *op);
      if (isa<hivm::FixpipeOp>(op))
        return true;
      if (isLocalMatmulInit(op, v)) {
        // no need to insert fixpipe because the single user can directly use
        // result stay in local buffer.
        return true;
      }
      return false;
    };
    if (traceSingleChainUser(mmadLikeOpRes, isMatchedOp))
      return failure();

    int resultIndx = 0;
    Operation *insertAfterOp = nullptr;
    if (isAccumulation(op)) {
      // only insert fixpipe outside of the for loop when it is an accumulation
      // loop
      insertAfterOp = getInsertPoint(op, resultIndx);
    } else {
      insertAfterOp = op;
    }
    rewriter.setInsertionPointAfter(insertAfterOp);

    LDBG("Replacing fix pipe for " << op);
    insertFixpipe(rewriter, insertAfterOp,
                  insertAfterOp->getResult(resultIndx));
    op->setAttr(mmadFixpipeForResultAlreadyInserted,
                rewriter.getBoolAttr(true));

    // When the mmad-like op is an accumulation, the fixpipe above only serves
    // external consumers of the loop's final accumulated value. If the mmad
    // result is also consumed by a Vector op *inside* the same loop (a Cube->
    // Vector cross-core junction), that consumer reads the raw L1 result and
    // no downstream pass can trace it back to a FixpipeOp/StoreOp. Insert a
    // fixpipe right after the mmad-like op inside the loop and redirect only
    // those in-loop Vector consumers to it; the accumulation yield is left on
    // the raw result so the iter_arg chain stays in L1.
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
            FixpipeSubBlockAttr{},
            /*pre_quant=*/nullptr, /*pre_relu=*/nullptr,
            /*channel_split=*/nullptr);
        for (OpOperand *operand : inLoopVecOperands) {
          rewriter.modifyOpInPlace(operand->getOwner(), [&]() {
            operand->set(innerFixpipe.getResultTensor());
          });
        }
        LDBG("Insert in-loop fixpipe for Vector consumer of accumulation mmad");
      }
    }
    return success();
  }
};

/// Insert fixpipe for hivm::ConvOp.
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

    // === insert fixpipe ===
    int resultIndx = 0;
    Operation *insertAfterOp = nullptr;
    if (isAccumulation(op)) {
      // only insert fixpipe outside of the for loop when it is an accumulation
      // loop
      insertAfterOp = getInsertPoint(op, resultIndx);
    } else {
      insertAfterOp = op;
    }
    rewriter.setInsertionPointAfter(insertAfterOp);
    Location loc = insertAfterOp->getLoc();

    Value fixpipeInit = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elementType);

    // Create FixpipeDMAModeAttr with NZ2ND mode
    MLIRContext *ctx = rewriter.getContext();
    FixpipeDMAModeAttr dmaModeAttr =
        FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);

    auto fixpipeOp = rewriter.create<FixpipeOp>(
        loc,
        fixpipeInit.getType(),                // result_tensor
        insertAfterOp->getResult(resultIndx), // src
        fixpipeInit,                          // dst
        dmaModeAttr,                          // dma_mode (NZ2ND)
        FixpipeDualDstModeAttr{},             // dual_dst_mode (default)
        FixpipeSubBlockAttr{},                // sub_block_idx (default)
        /*pre_quant=*/nullptr,                // pre_quant
        /*pre_relu=*/nullptr,                 // pre_relu
        /*channel_split=*/nullptr             // channel_split
    );

    rewriter.replaceAllUsesExcept(insertAfterOp->getResult(resultIndx),
                                  fixpipeOp.getResultTensor(), fixpipeOp);

    op->setAttr(mmadFixpipeForResultAlreadyInserted,
                rewriter.getBoolAttr(true));
    return success();
  }
};

static bool isVcastInlinableIntoFixpipe(hivm::VCastOp castOp);

std::optional<FixpipePreQuantMode> getQuantMode(hivm::VCastOp castOp) {
  if (!isVcastInlinableIntoFixpipe(castOp))
    return std::nullopt;

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

/// when all the activationOps are ready, there should be relu, leaky-relu and
/// p-relu
bool isActivationOp(Operation *op) { return isa<hivm::VReluOp>(op); }

static bool isRegBasedArch(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  return module && hacc::utils::isRegBasedArch(module);
}

static bool hasCompatibleShape(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  if (!lhsType || !rhsType || !lhsType.hasRank() || !rhsType.hasRank())
    return false;
  if (lhsType.getRank() != rhsType.getRank())
    return false;
  return succeeded(
      verifyCompatibleShape(lhsType.getShape(), rhsType.getShape()));
}

/// Fixpipe pre-quant cannot implement integer narrowing casts that disable
/// saturation (for example, trunc-with-overflow semantics).
static bool isVcastInlinableIntoFixpipe(hivm::VCastOp castOp) {
  if (!isRegBasedArch(castOp))
    return true;

  auto inputType = getElementTypeOrSelf(castOp.getSrc()[0].getType());
  auto outputType = getElementTypeOrSelf(castOp.getDst()[0].getType());
  if (!inputType.isIntOrIndex() || !outputType.isIntOrIndex())
    return true;

  int64_t srcBitWidth = inputType.getIntOrFloatBitWidth();
  int64_t dstBitWidth = outputType.getIntOrFloatBitWidth();
  if (srcBitWidth <= dstBitWidth || outputType.isInteger(1))
    return true;

  if (auto enableSaturate = castOp->getAttrOfType<BoolAttr>("enable_saturate"))
    return enableSaturate.getValue();
  return true;
}

template <typename OpType>
std::optional<FixpipePreReluMode> getReluMode(OpType op) {
  if constexpr (std::is_same_v<OpType, hivm::VReluOp>) {
    return hivm::symbolizeFixpipePreReluMode("NORMAL_RELU");
  }
  llvm::report_fatal_error("unsupported ReluValue");
}

Type getInitType(Value v, hivm::FixpipePreQuantMode quant,
                 PatternRewriter &rewriter) {
  if (quant == FixpipePreQuantMode ::NO_QUANT)
    return getElementTypeOrSelf(v);
  if (quant == FixpipePreQuantMode ::F322F16)
    return rewriter.getF16Type();
  if (quant == FixpipePreQuantMode ::F322BF16)
    return rewriter.getBF16Type();
  if (quant == FixpipePreQuantMode::S322I8)
    return rewriter.getI8Type();
  if (quant == FixpipePreQuantMode::QF322F32_PRE)
    return rewriter.getF32Type();
  llvm::report_fatal_error("unsupported QuantMode");
}

int64_t getSiftedUsersNum(Value v) {
  const DenseSet<Operation *> container(v.getUsers().begin(),
                                        v.getUsers().end());
  auto filteredRange = llvm::make_filter_range(container, [](Operation *op) {
    return !isa<annotation::MarkOp, hivm::DebugOp, tensor::DimOp>(op);
  });
  return DenseSet<Operation *>(filteredRange.begin(), filteredRange.end())
      .size();
}

//===----------------------------------------------------------------------===//
// InlineFixpipeOpPattern
//===----------------------------------------------------------------------===//
// Fixpipe can complete 3 inner action with origin matrixC operand following
// conditions
//   1. cast or quantization
//   2. relu and other activation function
//   3. store or layout
// Potential optimization is to fuse condition 1&2&3 into fixpipe.
struct InlineFixpipeOpPattern : public OpRewritePattern<FixpipeOp> {
public:
  InlineFixpipeOpPattern(MLIRContext *context, bool inlineQuantScale)
      : OpRewritePattern<FixpipeOp>(context),
        inlineQuantScale(inlineQuantScale) {}

  LogicalResult matchAndRewrite(FixpipeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResultTensor())
      return failure();

    auto fixpipeResTensor = op.getResultTensor();
    if (fixpipeResTensor.getUsers().empty())
      return failure();

    if (getSiftedUsersNum(fixpipeResTensor) != 1)
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
    // FixPipe followed by debugOp only, no need to inline
    if (curOp == nullptr)
      return success();
    // Operation curOp = *maybeInlinedOp;
    // 1. cast or quantization
    auto castOp = dyn_cast_if_present<hivm::VCastOp>(curOp);
    if (op.getFixpipeState() <= op.needFixpipePreFuse() && castOp &&
        getQuantMode(castOp).has_value()) {
      matched = true;
      inlineFixPipeWithRreQuant(rewriter, loc, op, castOp,
                                op.getDpsInputOperand(0)->get());
    } else if (op.getFixpipeState() <= op.needFixpipePreFuse() &&
               isActivationOp(curOp)) {
      // 2. relu and other activation function
      matched = true;
      auto reluOp = llvm::dyn_cast_if_present<hivm::VReluOp>(curOp);
      inlineFixPipeWithRreRelu(rewriter, loc, op, reluOp);
    } else if (auto storeOp = llvm::dyn_cast_if_present<hivm::StoreOp>(curOp)) {
      //   3. store or layout
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
               vMulOp && isRegBasedArch(op) && inlineQuantScale &&
               isUserQuantScaleInlinable(op, vMulOp)) {
      matched = true;
      inlineFixPipeWithQuantScale(rewriter, op, vMulOp);
    } else if (isUserTransposeInlinable(op, curOp)) {
      matched = true;
      inlineFixPipeWithTranspose(rewriter, op, cast<hivm::VTransposeOp>(curOp));
    } else if (auto extractSliceOp =
                   dyn_cast_if_present<tensor::ExtractSliceOp>(curOp);
               extractSliceOp &&
               (!isRegBasedArch(op) ||
                hasCompatibleShape(op.getSource(),
                                   extractSliceOp.getSource()))) {
      // change to fixpipe op + extract_slice to extract_slice + fixpipe op
      if (op->getBlock() == extractSliceOp->getBlock()) {
        // only swap when fixpipe op and extract slice op are in same block,
        // otherwise, extract slice op may be in sub block loop and fixpipe
        // cannot be fused into.
        matched = true;
        swapFixpipeAndExtractSliceOp(rewriter, loc, op, extractSliceOp);
      }
    } else if (auto insertSliceOp =
                   dyn_cast_if_present<tensor::InsertSliceOp>(curOp);
               insertSliceOp &&
               (!isRegBasedArch(op) ||
                hasCompatibleShape(op.getSource(),
                                   insertSliceOp.getSource()))) {
      // change to fixpipe op + insert_slice + store op to insert_slice +
      // fixpipe op + store op, and besides store op, there is no anther user
      // for insert_slice
      if (traceDownStoreOpWithSingleChain(insertSliceOp.getResult())) {
        matched = true;
        swapFixpipeAndInsertSliceOp(rewriter, loc, op, insertSliceOp);
      }
    } else if (isa<scf::YieldOp>(curOp) &&
               isa<scf::ForOp>(curOp->getParentOp()) &&
               !op->getAttr(fixpipeDoNotMoveOutOfScfFor)) {
      // move fixpipe out of scf.for
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
    bool regBased = isRegBasedArch(op);
    FixpipeDMAModeAttr dmaModeAttr =
        regBased ? op.getDmaModeAttr()
                 : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        loc, fixpipeInit.getType(), /*src=*/newFixpipeSrcTensor,
        /*dst=*/fixpipeInit, dmaModeAttr, op.getDualDstModeAttr(),
        op.getSubBlockIdxAttr(), quantModeAttr, reluModeAttr,
        op.getChannelSplitAttr(), op.getQuantScale());
    rewriter.replaceAllUsesWith(castOp.getResult()[0],
                                newFixpipeOp.getResultTensor());
    rewriter.eraseOp(castOp);
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithPreQuant");
  }

  void inlineFixPipeWithRreRelu(PatternRewriter &rewriter, Location loc,
                                hivm::FixpipeOp op,
                                hivm::VReluOp reluOp) const {
    std::optional<FixpipePreReluMode> reluMode = getReluMode(reluOp);
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
        op.getDualDstModeAttr(), op.getSubBlockIdxAttr(), op.getPreQuantAttr(),
        op.getPreReluAttr(), op.getChannelSplitAttr(), op.getQuantScale());
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

  bool isUserQuantScaleInlinable(hivm::FixpipeOp op,
                                 hivm::VMulOp vMulOp) const {
    auto dualDstMode = op.getDualDstModeAttr();
    if ((dualDstMode &&
         dualDstMode.getDualDstMode() != FixpipeDualDstMode::NO_DUAL) ||
        op.getQuantScale())
      return false;
    if (!vMulOp->hasOneUse())
      return false;
    if (!traceDownStoreOpWithSingleChain(vMulOp.getResult()[0]))
      return false;

    unsigned fixpipeOperand =
        vMulOp.getDpsInputOperand(0)->get().getDefiningOp() == op ? 0 : 1;
    Value quantScale = vMulOp.getDpsInputOperand(1 - fixpipeOperand)->get();
    return utils::isScalarLike(quantScale);
  }

  bool isUserTransposeInlinable(hivm::FixpipeOp op, Operation *user) const {
    if (!isRegBasedArch(op))
      return false;
    auto transpose = dyn_cast<hivm::VTransposeOp>(user);
    if (!transpose || op.getDmaMode() != FixpipeDMAMode::NZ2ND)
      return false;
    ArrayRef<int64_t> permutation = transpose.getPermutation();
    return permutation.size() == 2 && permutation[0] == 1 &&
           permutation[1] == 0;
  }

  void inlineFixPipeWithQuantScale(PatternRewriter &rewriter,
                                   hivm::FixpipeOp op,
                                   hivm::VMulOp vMulOp) const {
    unsigned fixpipeOperand =
        vMulOp.getDpsInputOperand(0)->get().getDefiningOp() == op ? 0 : 1;
    Value quantScale = vMulOp.getDpsInputOperand(1 - fixpipeOperand)->get();
    Value dst = vMulOp.getDpsInitOperand(0)->get();

    auto preQuant = op.getPreQuant();
    if (preQuant == FixpipePreQuantMode::NO_QUANT &&
        getElementTypeOrSelf(op.getSrcOperandType()).isF32() &&
        getElementTypeOrSelf(dst.getType()).isF32()) {
      preQuant = FixpipePreQuantMode::QF322F32_PRE;
    }

    rewriter.setInsertionPointAfter(vMulOp);
    auto newFixpipe = rewriter.create<FixpipeOp>(
        op.getLoc(), dst.getType(), op.getSource(), dst, op.getDmaModeAttr(),
        op.getDualDstModeAttr(), op.getSubBlockIdxAttr(),
        FixpipePreQuantModeAttr::get(rewriter.getContext(), preQuant),
        op.getPreReluAttr(), op.getChannelSplitAttr(), quantScale);
    rewriter.replaceOp(vMulOp, newFixpipe.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithQuantScale");
  }

  void inlineFixPipeWithTranspose(PatternRewriter &rewriter, hivm::FixpipeOp op,
                                  hivm::VTransposeOp transpose) const {
    rewriter.setInsertionPointAfter(transpose);
    auto dmaMode =
        FixpipeDMAModeAttr::get(rewriter.getContext(), FixpipeDMAMode::NZ2DN);
    auto newFixpipe = rewriter.create<FixpipeOp>(
        op.getLoc(), transpose.getResult()[0].getType(), op.getSource(),
        transpose.getDst(), dmaMode, op.getDualDstModeAttr(),
        op.getSubBlockIdxAttr(), op.getPreQuantAttr(), op.getPreReluAttr(),
        op.getChannelSplitAttr(), op.getQuantScale());
    rewriter.replaceOp(transpose, newFixpipe.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithTranspose");
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
        getInitType(newExtractSliceResult, op.getPreQuant(), rewriter));

    MLIRContext *ctx = rewriter.getContext();
    bool regBased = isRegBasedArch(op);
    FixpipeDMAModeAttr dmaModeAttr =
        regBased ? op.getDmaModeAttr()
                 : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        extractSliceOp.getLoc(), fixpipeInit.getType(),
        /*src=*/newExtractSliceResult, /*dst=*/fixpipeInit, dmaModeAttr,
        op.getDualDstModeAttr(), op.getSubBlockIdxAttr(), quantModeAttr,
        reluModeAttr, op.getChannelSplitAttr(), op.getQuantScale());
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
        getInitType(newInsertSliceResult, op.getPreQuant(), rewriter));
    MLIRContext *ctx = rewriter.getContext();
    bool regBased = isRegBasedArch(op);
    FixpipeDMAModeAttr dmaModeAttr =
        regBased ? op.getDmaModeAttr()
                 : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        insertSliceOp.getLoc(), TypeRange{fixpipeInit}, newInsertSliceResult,
        fixpipeInit, dmaModeAttr, op.getDualDstModeAttr(),
        op.getSubBlockIdxAttr(), quantModeAttr, reluModeAttr,
        op.getChannelSplitAttr(), op.getQuantScale());
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
          FixpipeDualDstModeAttr{}, fixPipeOp.getSubBlockIdxAttr(),
          quantModeAttr, reluModeAttr);
      rewriter.replaceAllUsesExcept(scfForOp->getResult(idx.value()),
                                    newFixpipeOp.getResultTensor(),
                                    newFixpipeOp);
    }
    LDBG("moveFixpipeOutOfScfFor");
  }

  const bool inlineQuantScale;
};

//===----------------------------------------------------------------------===//
// InsertFixpipeForDevicePrint
//===----------------------------------------------------------------------===//
// Insert fixpipe for the hivm.print that prints the mm result and mm result is
// yield in scf.for
// eg.
// %init = tensor.empty()
// %res = scf.for iter_arg(%arg = %init) {
//   %t = hivm.mmadL1 ins() outs(%arg)
//   hivm.print %t
//   scf.yield %t
// }
// is converted to
// %init = tensor.empty()
// %res = scf.for iter_arg(%arg = %init) {

//   %t = hivm.mmadL1 ins() outs(%arg)
//   %fixpipe = hivm.fixpipe int(%t)
//   hivm.print %fixpipe
//   scf.yield %t
// }
struct InsertFixpipeForDevicePrint : public OpRewritePattern<DebugOp> {
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
        FixpipeDualDstModeAttr{}, FixpipeSubBlockAttr{}, nullptr, nullptr);
    fixpipeOp->setAttr(usedForDebugOp, rewriter.getBoolAttr(true));

    rewriter.modifyOpInPlace(op, [&]() {
      op.getArgMutable().assign(workSpaceTensor);
      op.setMemscopeAttr(
          hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::GM));
      op.setTcoretypeAttr(hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE));
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

void populateInlineFixpipePatterns(RewritePatternSet &patterns,
                                   bool inlineQuantScale) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InsertFixpipeOpPattern<hivm::MmadL1Op>>(ctx);
  patterns.add<InsertFixpipeOpPattern<hivm::BatchMmadL1Op>>(ctx);
  patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv1DL1Op>>(ctx);
  patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv2DL1Op>>(ctx);
  patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv3DL1Op>>(ctx);
  patterns.add<InlineFixpipeOpPattern>(ctx, inlineQuantScale);
  patterns.add<InsertFixpipeForIterArgMMAD>(ctx);
}

void InlineFixpipe::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateInlineFixpipePatterns(patterns, inlineQuantScale);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
  RewritePatternSet insertFixpipeForDevicePrintPattern(&getContext());
  MLIRContext *ctx = insertFixpipeForDevicePrintPattern.getContext();
  insertFixpipeForDevicePrintPattern.add<InsertFixpipeForDevicePrint>(ctx);
  if (failed(applyPatternsGreedily(
          getOperation(), std::move(insertFixpipeForDevicePrintPattern)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInlineFixpipePass() {
  return std::make_unique<InlineFixpipe>();
}
