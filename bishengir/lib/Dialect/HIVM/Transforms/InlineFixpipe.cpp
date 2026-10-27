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

// This pass converts ops to hivm.fixpipe.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/Contracts/FixpipePreQuantContract.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/FixpipeUtils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_INLINEFIXPIPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-inline-fixpipe"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {

/// Attribute stamped by Normalize Matmul on scf.if / scf.for when the matmul
/// may not execute in some iterations. Used by Inline Fixpipe to sink store
/// (and related consumers) into the if and split Cube vs Vector data paths.
constexpr llvm::StringLiteral kFallbackNotExec = "fallback_not_exec";

/// Return true when \p op is nested in a scope marked for the vector core.
/// Fixpipe must not be fused with a store in such scopes: fusion would place
/// the cube fixpipe inside the vector scope and break mix AIC/AIV splitting.
static bool isInsideVectorScope(Operation *op) {
  auto scopeOp = op->getParentOfType<scope::ScopeOp>();
  if (!scopeOp)
    return false;
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  return coreTypeAttr && coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

struct InlineFixpipe : public impl::InlineFixpipeBase<InlineFixpipe> {
  using Base::Base;
  void runOnOperation() override;
};

static Operation *getSingleSiftedUser(Value v) {
  Operation *singleUser = nullptr;
  for (Operation *user : v.getUsers()) {
    if (isa<annotation::MarkOp, hivm::DebugOp, tensor::DimOp>(user))
      continue;
    if (singleUser)
      return nullptr;
    singleUser = user;
  }
  return singleUser;
}

/// Fixpipe pre-quant cannot implement integer narrowing casts that disable
/// saturation (e.g. trunc-with-overflow semantics).
static bool isIntegerNarrowingCastInlinable(Type inputType, Type outputType,
                                            Operation *castOp) {
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

static bool isVcastInlinableIntoFixpipe(hivm::VCastOp castOp) {
  if (!isOnRegBasedArch(castOp))
    return true;
  auto inputType = getElementTypeOrSelf(castOp.getSrc()[0].getType());
  auto outputType = getElementTypeOrSelf(castOp.getDst()[0].getType());
  return isIntegerNarrowingCastInlinable(inputType, outputType, castOp);
}

/// Collect a single-use chain of VCastOps starting at \p firstCast.
/// The chain always includes at least \p firstCast.
static SmallVector<hivm::VCastOp> collectVCastChain(hivm::VCastOp firstCast) {
  SmallVector<hivm::VCastOp> chain;
  hivm::VCastOp cur = firstCast;
  while (true) {
    chain.push_back(cur);
    Value res = cur.getResult()[0];
    Operation *nextUser = getSingleSiftedUser(res);
    auto nextCast = dyn_cast_if_present<hivm::VCastOp>(nextUser);
    if (!nextCast || nextCast.getSrc()[0] != res)
      break;
    cur = nextCast;
  }
  return chain;
}

static std::optional<FixpipePreQuantMode>
getQuantModeForTypes(Type inputType, Type outputType) {
  // Candidates come from the generated pre-quant type contract. InlineFixpipe
  // only fuses type-changing quantizing modes: a signature with several
  // candidates (f32 -> f32 matches both NO_QUANT and QF322F32_PRE) or with
  // only identity candidates is deliberately not fused.
  SmallVector<FixpipePreQuantMode> candidates =
      getFixpipePreQuantCandidates(inputType, outputType);
  // NO_QUANT describes identity signatures and is not an explicit quantizing
  // fusion choice.
  llvm::erase(candidates, FixpipePreQuantMode::NO_QUANT);
  if (candidates.size() != 1)
    return std::nullopt;
  FixpipePreQuantMode mode = candidates.front();
  if (mode == FixpipePreQuantMode::QF322F32_PRE)
    return std::nullopt;
  return mode;
}

/// True when \p inputType -> \p outputType is an integer bit-width narrowing.
static bool isIntegerNarrowing(Type inputType, Type outputType) {
  if (!inputType.isIntOrIndex() || !outputType.isIntOrIndex())
    return false;
  if (outputType.isInteger(1))
    return false;
  return inputType.getIntOrFloatBitWidth() > outputType.getIntOrFloatBitWidth();
}

/// Decide pre-quant mode from the overall cast chain: element type of the
/// first cast's input to element type of the last cast's output. Every cast
/// in the chain (and the overall conversion) must be inlinable.
static std::optional<FixpipePreQuantMode>
getQuantModeForCastChain(ArrayRef<hivm::VCastOp> castChain) {
  assert(!castChain.empty() && "cast chain must be non-empty");
  for (hivm::VCastOp castOp : castChain) {
    if (!isVcastInlinableIntoFixpipe(castOp))
      return std::nullopt;
  }

  // Copy out of ArrayRef: getSrc/getDst are non-const accessors.
  hivm::VCastOp firstCast = castChain.front();
  hivm::VCastOp lastCast = castChain.back();
  Type inputType = getElementTypeOrSelf(firstCast.getSrc()[0].getType());
  Type outputType = getElementTypeOrSelf(lastCast.getDst()[0].getType());

  // For overall integer narrowing (e.g. i32->i8):
  // - Integer-narrowing steps must not disable saturation.
  // - Float intermediates (i32->f32->f16->i8) often set enable_saturate=false;
  //   ignore those and instead require the last cast to allow saturation.
  if (isIntegerNarrowing(inputType, outputType)) {
    bool sawIntegerNarrowingCast = false;
    for (hivm::VCastOp castOp : castChain) {
      Type castIn = getElementTypeOrSelf(castOp.getSrc()[0].getType());
      Type castOut = getElementTypeOrSelf(castOp.getDst()[0].getType());
      if (!isIntegerNarrowing(castIn, castOut))
        continue;
      sawIntegerNarrowingCast = true;
      if (auto enableSaturate =
              castOp->getAttrOfType<BoolAttr>("enable_saturate")) {
        if (!enableSaturate.getValue())
          return std::nullopt;
      }
    }
    if (!sawIntegerNarrowingCast) {
      if (auto enableSaturate =
              lastCast->getAttrOfType<BoolAttr>("enable_saturate")) {
        if (!enableSaturate.getValue())
          return std::nullopt;
      }
    }
  }

  return getQuantModeForTypes(inputType, outputType);
}

std::optional<FixpipePreQuantMode> getQuantMode(hivm::VCastOp castOp) {
  return getQuantModeForCastChain(collectVCastChain(castOp));
}

/// when all the activationOps are ready, there should be relu, leaky-relu and
/// p-relu
bool isActivationOp(Operation *op) { return isa<hivm::VReluOp>(op); }

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

static bool isInsertedToStatic(tensor::ExtractSliceOp extractSlice) {
  auto sliceType =
      dyn_cast<RankedTensorType>(extractSlice.getResult().getType());
  if (!sliceType || sliceType.hasStaticShape() ||
      !extractSlice.getResult().hasOneUse())
    return false;

  auto insertSlice =
      dyn_cast<tensor::InsertSliceOp>(*extractSlice->user_begin());
  if (!insertSlice)
    return false;

  auto destType = dyn_cast<RankedTensorType>(insertSlice.getDest().getType());
  return destType && destType.hasStaticShape();
}

static bool isUserSliceSwappable(hivm::FixpipeOp fixpipe,
                                 tensor::ExtractSliceOp extractSlice) {
  if (!extractSlice)
    return false;
  if (!isOnRegBasedArch(fixpipe))
    return true;
  return hasCompatibleShape(fixpipe.getSource(), extractSlice.getSource()) &&
         !isInsertedToStatic(extractSlice);
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

static int64_t getSiftedUsersNum(Value v) {
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
    // Avoid fusing fixpipe into a VECTOR-scope store (mix AIC/AIV).
    if (isInsideVectorScope(curOp))
      return failure();

    if (isOnRegBasedArch(op) && op.getDmaMode() != FixpipeDMAMode::NZ2NZ) {
      if (all_of(op->getUsers(), [](auto *user) {
            if (isa<annotation::MarkOp>(user))
              return true;
            return isa<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.cpp.inc"
                >(user);
          })) {

        convertNz2NdFixpipeToNz2Nz(rewriter, op);
        return success();
      }
    }

    // 1. cast or quantization (single VCast or a single-use VCast chain).
    //    Quant mode is decided from first-cast input type to last-cast output
    //    type so decomposed casts (e.g. i32->i16->i8) still fuse as S322I8.
    auto castOp = dyn_cast_if_present<hivm::VCastOp>(curOp);
    if (op.getFixpipeState() <= op.needFixpipePreFuse() && castOp) {
      SmallVector<hivm::VCastOp> castChain = collectVCastChain(castOp);
      if (getQuantModeForCastChain(castChain).has_value()) {
        matched = true;
        inlineFixPipeWithRreQuant(rewriter, loc, op, castChain,
                                  op.getDpsInputOperand(0)->get());
      }
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
               vMulOp && isOnRegBasedArch(op) &&
               (inlineQuantScale || hasQuantScaleCompileHint(vMulOp)) &&
               isUserQuantScaleInlinable(op, vMulOp)) {
      matched = true;
      inlineFixPipeWithQuantScale(rewriter, op, vMulOp);
    } else if (isUserTransposeInlinable(op, curOp)) {
      matched = true;
      inlineFixPipeWithTranspose(rewriter, op, cast<hivm::VTransposeOp>(curOp));
    } else if (auto extractSliceOp =
                   dyn_cast_if_present<tensor::ExtractSliceOp>(curOp);
               isUserSliceSwappable(op, extractSliceOp)) {
      // change to fixpipe op + extract_slice to extract_slice + fixpipe op
      if (op->getBlock() == extractSliceOp->getBlock()) {
        // only swap when fixpipe op and extract slice op are in same block,
        // otherwise, extract slice op may be in sub block loop and fixpipe
        // cannot be fused into.
        matched = true;
        swapFixpipeAndExtractSliceOp(rewriter, loc, op, extractSliceOp);
      }
    } else if (isa<scf::YieldOp>(curOp) &&
               isa<scf::ForOp>(curOp->getParentOp()) &&
               !op->getAttr(fixpipeDoNotMoveOutOfScfFor) &&
               !op.getSource().getDefiningOp<hivm::BatchMmadL1Op>()) {
      // Move fixpipe out of scf.for.
      // The BatchMmadL1 case is excluded because TileBatchMMIntoLoop requires
      // `BatchMmadL1 -> Fixpipe` use chain to stay in the same block.
      // Do not hoist a type-converting fixpipe out of a loop-carried value:
      // hoisting rewires the yield to the fixpipe source, which would change
      // the loop yield type.
      if (op.getSource().getType() != op.getResultTensor().getType())
        return rewriter.notifyMatchFailure(
            op, "fixpipe converts the loop-carried value type");
      matched = true;
      auto scfForOp = dyn_cast_if_present<scf::ForOp>(curOp->getParentOp());
      moveFixpipeOutOfScfFor(rewriter, loc, op, scfForOp, op.getResultTensor());
    } else if (isa<scf::YieldOp>(curOp) &&
               isa<scf::IfOp>(curOp->getParentOp()) &&
               curOp->getParentOp()->hasAttr(kFallbackNotExec)) {
      // Sink the outer consumer chain into both fallback_not_exec branches.
      // Cast / transpose / extract_slice / store fusion is left to the
      // existing greedy InlineFixpipe patterns on the Cube-side fixpipe.
      if (succeeded(inlineFixpipeThroughMayNotExecIf(
              rewriter, loc, op, cast<scf::IfOp>(curOp->getParentOp()))))
        matched = true;
    }
    return matched ? success() : failure();
  }

  void inlineFixPipeWithRreQuant(PatternRewriter &rewriter, Location loc,
                                 hivm::FixpipeOp op,
                                 ArrayRef<hivm::VCastOp> castChain,
                                 Value newFixpipeSrcTensor) const {
    std::optional<FixpipePreQuantMode> quantMode =
        getQuantModeForCastChain(castChain);
    if (!quantMode) {
      LDBG("cast op quant mode is null");
      return;
    }
    auto quantModeAttr =
        FixpipePreQuantModeAttr::get(op.getContext(), quantMode.value());
    auto reluModeAttr = op.getPreReluAttr();

    hivm::VCastOp lastCast = castChain.back();
    rewriter.setInsertionPointAfter(lastCast);
    Value fixpipeInit =
        utils::createEmptyOp(rewriter, loc, lastCast.getResult()[0]);
    MLIRContext *ctx = rewriter.getContext();
    bool regBased = isOnRegBasedArch(op);
    FixpipeDMAModeAttr dmaModeAttr =
        regBased ? op.getDmaModeAttr()
                 : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        loc, fixpipeInit.getType(), /*src=*/newFixpipeSrcTensor,
        /*dst=*/fixpipeInit, dmaModeAttr, op.getDualDstModeAttr(),
        op.getSubBlockIdxAttr(), quantModeAttr, reluModeAttr,
        op.getChannelSplitAttr(), op.getC0PadEnAttr(), op.getQuantScale());
    rewriter.replaceAllUsesWith(lastCast.getResult()[0],
                                newFixpipeOp.getResultTensor());
    for (hivm::VCastOp castOp : llvm::reverse(castChain))
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
    OperationState state(loc, op->getName());
    state.addAttributes(op->getAttrs());
    SmallVector<Value> operands(op->getOperands());
    operands[0] = fixpipeSrcTensor;
    operands[1] = dst;
    state.addOperands(operands);
    auto newFixpipeOp = cast<hivm::FixpipeOp>(rewriter.create(state));
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

  bool hasQuantScaleCompileHint(hivm::VMulOp op) const {
    return llvm::any_of(op->getUsers(), [](Operation *userOp) {
      auto markOp = dyn_cast<annotation::MarkOp>(userOp);
      return markOp && markOp->hasAttr(utils::kInlinableQuantScaleAttr);
    });
  }

  bool isUserQuantScaleInlinable(hivm::FixpipeOp op,
                                 hivm::VMulOp vMulOp) const {
    auto dualDstMode = op.getDualDstModeAttr();
    if ((dualDstMode &&
         dualDstMode.getDualDstMode() != FixpipeDualDstMode::NO_DUAL) ||
        op.getQuantScale())
      return false;
    if (llvm::count_if(vMulOp->getUsers(), [](Operation *userOp) {
          return !isa<annotation::MarkOp>(userOp);
        }) != 1)
      return false;
    if (!traceDownStoreOpWithSingleChain(vMulOp.getResult()[0]))
      return false;

    unsigned fixpipeOperand =
        vMulOp.getDpsInputOperand(0)->get().getDefiningOp() == op ? 0 : 1;
    Value quantScale = vMulOp.getDpsInputOperand(1 - fixpipeOperand)->get();
    return utils::isScalarLike(quantScale);
  }

  bool isUserTransposeInlinable(hivm::FixpipeOp op, Operation *user) const {
    if (!isOnRegBasedArch(op))
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
        op.getLoc(), dst.getType(), op.getSource(), dst, op.getUnitFlagCond(),
        op.getDmaModeAttr(), op.getDualDstModeAttr(), op.getSubBlockIdxAttr(),
        FixpipePreQuantModeAttr::get(rewriter.getContext(), preQuant),
        op.getPreReluAttr(), op.getChannelSplitAttr(), op.getC0PadEnAttr(),
        quantScale, op.getUnitFlagModeAttr(), op.getUnitFlagGroupIdAttr());
    for (Operation *user : llvm::make_early_inc_range(vMulOp->getUsers())) {
      if (isa<annotation::MarkOp>(user)) {
        newFixpipe->setAttr(utils::kInlinedQuantScaleAttr,
                            rewriter.getUnitAttr());
        rewriter.eraseOp(user);
      }
    }
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
        op.getChannelSplitAttr(), op.getC0PadEnAttr(), op.getQuantScale());
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
    bool regBased = isOnRegBasedArch(op);
    FixpipeDMAModeAttr dmaModeAttr =
        regBased ? op.getDmaModeAttr()
                 : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        extractSliceOp.getLoc(), fixpipeInit.getType(),
        /*src=*/newExtractSliceResult, /*dst=*/fixpipeInit, dmaModeAttr,
        op.getDualDstModeAttr(), op.getSubBlockIdxAttr(), quantModeAttr,
        reluModeAttr, op.getChannelSplitAttr(), op.getC0PadEnAttr(),
        op.getQuantScale());
    rewriter.replaceOp(extractSliceOp, newFixpipeOp.getResultTensor());
    rewriter.eraseOp(op);
    LDBG("InlineFixpipeWithExtractSliceReshape");
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

  struct MayNotExecConsumerChain {
    SmallVector<Operation *> ops; // from if-result toward store (exclusive)
    hivm::StoreOp store;
  };

  /// Collect a single-use chain from if-result to StoreOp through ops that the
  /// existing InlineFixpipe patterns already know how to fold (Cube path) or
  /// that are valid to replay as vector ops (no_exec path). This pattern only
  /// sinks the chain; fusion is left to greedy rewrite.
  FailureOr<MayNotExecConsumerChain>
  collectMayNotExecStoreChain(PatternRewriter &rewriter, Operation *notifyOp,
                              hivm::FixpipeOp fixpipe, Value ifResult) const {
    MayNotExecConsumerChain chain;
    Value cur = ifResult;
    while (true) {
      Operation *user = getSingleSiftedUser(cur);
      if (!user)
        return rewriter.notifyMatchFailure(
            notifyOp, "fallback_not_exec consumer chain is not single-use");

      // The whole chain must sit in the same block as the fallback if. If it
      // is nested in another region (e.g. a guarding scf.if / scf.for), its
      // external operands (store dst, dynamic slice indices) are defined in
      // that region and cannot dominate the sunk if, and sinking the store
      // would also drop the guarding condition.
      if (user->getBlock() != notifyOp->getBlock())
        return rewriter.notifyMatchFailure(
            user,
            "fallback_not_exec consumer chain is nested in another region");

      if (auto storeOp = dyn_cast<hivm::StoreOp>(user)) {
        // TODO: share this atomic-kind whitelist with the greedy store-fusion
        // path in inlineFixpipeOp (NONE/ADD/MAX/MIN). No helper exists yet.
        auto storeAttr = storeOp.getAtomicKindAttr();
        hivm::AtomicKind atomicKind = hivm::AtomicKind::NONE;
        if (storeAttr)
          atomicKind = storeAttr.getValue();
        if (atomicKind != AtomicKind::NONE && atomicKind != AtomicKind::ADD &&
            atomicKind != AtomicKind::MAX && atomicKind != AtomicKind::MIN)
          return rewriter.notifyMatchFailure(
              storeOp, "fallback_not_exec store has unsupported atomic kind");
        if (isInsideVectorScope(storeOp))
          return rewriter.notifyMatchFailure(
              storeOp, "fallback_not_exec store is inside a vector scope");
        chain.store = storeOp;
        return chain;
      }

      if (auto castOp = dyn_cast<hivm::VCastOp>(user)) {
        // getQuantMode includes isVcastInlinableIntoFixpipe check
        if (!getQuantMode(castOp).has_value())
          return rewriter.notifyMatchFailure(
              castOp, "fallback_not_exec vcast is not inlinable as pre_quant");
        if (castOp.getSrc()[0] != cur)
          return rewriter.notifyMatchFailure(
              castOp, "fallback_not_exec vcast src is not the chain value");
        chain.ops.push_back(castOp);
        cur = castOp.getResult()[0];
        continue;
      }

      if (auto reluOp = dyn_cast<hivm::VReluOp>(user)) {
        if (!getReluMode(reluOp).has_value())
          return rewriter.notifyMatchFailure(
              reluOp, "fallback_not_exec vrelu is not inlinable as pre_relu");
        if (reluOp.getSrc()[0] != cur)
          return rewriter.notifyMatchFailure(
              reluOp, "fallback_not_exec vrelu src is not the chain value");
        chain.ops.push_back(reluOp);
        cur = reluOp.getResult()[0];
        continue;
      }

      if (isa<hivm::VTransposeOp>(user)) {
        if (!isUserTransposeInlinable(fixpipe, user))
          return rewriter.notifyMatchFailure(
              user,
              "fallback_not_exec vtranspose is not inlinable into fixpipe");
        auto transposeOp = cast<hivm::VTransposeOp>(user);
        if (transposeOp.getSrc() != cur)
          return rewriter.notifyMatchFailure(
              transposeOp,
              "fallback_not_exec vtranspose src is not the chain value");
        chain.ops.push_back(transposeOp);
        cur = transposeOp.getResult()[0];
        continue;
      }

      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
        if (extractSliceOp.getSource() != cur)
          return rewriter.notifyMatchFailure(
              extractSliceOp,
              "fallback_not_exec extract_slice source is not the chain value");
        chain.ops.push_back(extractSliceOp);
        cur = extractSliceOp.getResult();
        continue;
      }

      return rewriter.notifyMatchFailure(
          user, "unsupported op in fallback_not_exec consumer chain");
    }
  }

  /// Replay \p chain after \p src (branch-local value) and emit the terminal
  /// store. Used identically for both fallback_not_exec branches.
  void
  emitSunkConsumerChainAndStore(PatternRewriter &rewriter, Location loc,
                                Value src,
                                const MayNotExecConsumerChain &chain) const {
    Value cur = src;
    for (Operation *op : chain.ops) {
      if (auto castOp = dyn_cast<hivm::VCastOp>(op)) {
        Value init = utils::createEmptyOp(rewriter, loc, castOp.getResult()[0]);
        IRMapping mapping;
        mapping.map(castOp.getSrc()[0], cur);
        mapping.map(castOp.getDst()[0], init);
        auto *cloned = rewriter.clone(*castOp, mapping);
        cur = cloned->getResult(0);
        continue;
      }
      if (auto reluOp = dyn_cast<hivm::VReluOp>(op)) {
        Value init = utils::createEmptyOp(rewriter, loc, reluOp.getResult()[0]);
        IRMapping mapping;
        mapping.map(reluOp.getSrc()[0], cur);
        mapping.map(reluOp.getDst()[0], init);
        auto *cloned = rewriter.clone(*reluOp, mapping);
        cur = cloned->getResult(0);
        continue;
      }
      if (auto transposeOp = dyn_cast<hivm::VTransposeOp>(op)) {
        Value init =
            utils::createEmptyOp(rewriter, loc, transposeOp.getResult()[0]);
        auto newTranspose = rewriter.create<hivm::VTransposeOp>(
            transposeOp.getLoc(), TypeRange{init.getType()}, cur, init,
            transposeOp.getPermutationAttr());
        cur = newTranspose.getResult()[0];
        continue;
      }
      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
        auto newExtract = rewriter.create<tensor::ExtractSliceOp>(
            extractSliceOp.getLoc(), extractSliceOp.getResultType(), cur,
            extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
            extractSliceOp.getMixedStrides());
        cur = newExtract.getResult();
        continue;
      }
      llvm_unreachable("unexpected fallback_not_exec consumer op");
    }

    auto storeOp = chain.store;
    auto newStore = rewriter.create<hivm::StoreOp>(
        storeOp.getLoc(), TypeRange{}, cur, storeOp.getDst());
    if (auto atomicAttr = storeOp.getAtomicKindAttr())
      newStore.setAtomicKindAttr(atomicAttr);
  }

  /// Clone ops from \p srcBlock into the builder's current block, excluding
  /// the terminator. Returns the mapped value of \p yieldedValue.
  Value cloneBlockBodyExceptTerminator(PatternRewriter &rewriter,
                                       Block &srcBlock,
                                       Value yieldedValue) const {
    IRMapping mapping;
    for (Operation &op : srcBlock.without_terminator())
      rewriter.clone(op, mapping);
    return mapping.lookupOrDefault(yieldedValue);
  }

  /// Values from the sunk chain that are external to the if-result pipeline
  /// (store dst, dynamic extract_slice indices). These must dominate the new
  /// if after the sink.
  void collectMayNotExecExternalValues(const MayNotExecConsumerChain &chain,
                                       SmallVectorImpl<Value> &values) const {
    // StoreOp::getDst() is non-const; copy the lightweight handle first.
    hivm::StoreOp storeOp = chain.store;
    values.push_back(storeOp.getDst());
    for (Operation *op : chain.ops) {
      auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op);
      if (!extractSliceOp)
        continue;
      // Source is the chain value; only offsets/sizes/strides are external.
      values.append(extractSliceOp.getOffsets().begin(),
                    extractSliceOp.getOffsets().end());
      values.append(extractSliceOp.getSizes().begin(),
                    extractSliceOp.getSizes().end());
      values.append(extractSliceOp.getStrides().begin(),
                    extractSliceOp.getStrides().end());
    }
  }

  /// Hoist effect-free defs of \p values (and their deps) that sit after
  /// \p ifOp in the same block so they dominate uses sunk into the if.
  /// Returns failure if a required def cannot be safely hoisted.
  LogicalResult hoistValuesBeforeIf(PatternRewriter &rewriter, scf::IfOp ifOp,
                                    ArrayRef<Value> values) const {
    Block *block = ifOp->getBlock();
    SmallVector<Operation *> toHoist;
    SmallPtrSet<Operation *, 8> seen;

    std::function<LogicalResult(Value)> collect =
        [&](Value v) -> LogicalResult {
      Operation *def = v.getDefiningOp();
      if (!def)
        return success(); // block/arg — already dominates
      if (def->getBlock() != block)
        return success(); // defined in a dominating block
      if (def->isBeforeInBlock(ifOp))
        return success();
      if (!seen.insert(def).second)
        return success();
      if (!isMemoryEffectFree(def))
        return rewriter.notifyMatchFailure(
            ifOp,
            "cannot hoist side-effecting def before fallback_not_exec if");
      for (Value operand : def->getOperands())
        if (failed(collect(operand)))
          return failure();
      toHoist.push_back(def);
      return success();
    };

    for (Value v : values)
      if (failed(collect(v)))
        return failure();

    for (Operation *op : toHoist)
      rewriter.moveOpBefore(op, ifOp);
    return success();
  }

  /// Sink the outer consumer chain into a resultless fallback_not_exec if.
  ///
  /// After this rewrite the Cube branch still has the original UB-dest
  /// fixpipe followed by the sunk cast/transpose/extract_slice/store ops.
  /// Subsequent greedy applications of the existing InlineFixpipe patterns
  /// fold those consumers into fixpipe (including extract_slice swap, which
  /// only requires same-block dominance — satisfied after the sink).
  ///
  /// TODO: consider the wrap-around-store alternative: keep the original
  /// scf.if, move it to just before the outer store, and rebuild it in place
  /// as a resultless if containing the sunk chain in both branches. That
  /// removes the need for hoistValuesBeforeIf
  LogicalResult inlineFixpipeThroughMayNotExecIf(PatternRewriter &rewriter,
                                                 Location loc,
                                                 hivm::FixpipeOp fixpipe,
                                                 scf::IfOp ifOp) const {
    if (ifOp.getElseRegion().empty())
      return rewriter.notifyMatchFailure(
          ifOp, "fallback_not_exec if has no else region");
    // v1: only handle single-result ifs so we can rewrite to a void if.
    if (ifOp.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          ifOp, "fallback_not_exec if does not have exactly one result");

    Value fixpipeRes = fixpipe.getResultTensor();
    if (!fixpipeRes)
      return rewriter.notifyMatchFailure(
          fixpipe, "fallback_not_exec fixpipe has no result tensor");

    Value thenYielded = ifOp.thenYield().getOperand(0);
    Value elseYielded = ifOp.elseYield().getOperand(0);

    bool fixpipeInThen = thenYielded == fixpipeRes;
    bool fixpipeInElse = elseYielded == fixpipeRes;
    if (fixpipeInThen == fixpipeInElse)
      return rewriter.notifyMatchFailure(
          ifOp, "fixpipe is yielded by both branches or by neither");

    Value vectorYielded = fixpipeInThen ? elseYielded : thenYielded;
    if (vectorYielded.getDefiningOp<hivm::FixpipeOp>())
      return rewriter.notifyMatchFailure(ifOp,
                                         "fixpipe found in both branches");

    Value ifResult = ifOp.getResult(0);
    auto maybeChain =
        collectMayNotExecStoreChain(rewriter, ifOp, fixpipe, ifResult);
    if (failed(maybeChain))
      return failure();
    MayNotExecConsumerChain chain = *maybeChain;

    SmallVector<Value> externalValues;
    collectMayNotExecExternalValues(chain, externalValues);
    // Store dst (e.g. memref.subview) and dynamic slice indices often sit
    // between the if and the outer store; hoist them before the if so sunk
    // uses dominate.
    if (failed(hoistValuesBeforeIf(rewriter, ifOp, externalValues)))
      return failure();

    LDBG("InlineFixpipeThroughMayNotExecIf");

    SmallVector<Operation *> toErase;
    toErase.push_back(chain.store);
    for (Operation *op : llvm::reverse(chain.ops))
      toErase.push_back(op);

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(ifOp);
    auto newIf =
        rewriter.create<scf::IfOp>(loc, TypeRange{}, ifOp.getCondition(),
                                   /*withElseRegion=*/true);
    newIf->setAttr(kFallbackNotExec, rewriter.getUnitAttr());

    // Resultless scf.if already has empty scf.yield terminators from the
    // builder (ensureTerminator). Insert branch bodies before those yields;
    // do not create a second terminator.
    auto populateBranch = [this, &rewriter, &loc, &chain](Block *destBlock,
                                                          Block &srcBlock,
                                                          Value yielded) {
      rewriter.setInsertionPointToStart(destBlock);
      Value mapped =
          cloneBlockBodyExceptTerminator(rewriter, srcBlock, yielded);
      emitSunkConsumerChainAndStore(rewriter, loc, mapped, chain);
    };

    populateBranch(newIf.thenBlock(), ifOp.getThenRegion().front(),
                   thenYielded);
    populateBranch(newIf.elseBlock(), ifOp.getElseRegion().front(),
                   elseYielded);

    // Erase outer consumer chain (store first), ignored users of the if
    // result, then the old if (which owns the original in-branch fixpipe).
    // Marks / debug / dim on the if-result cannot be preserved: the rewrite
    // deletes that SSA value. Warn so the drop is visible, but do not fail.
    for (Operation *op : toErase)
      rewriter.eraseOp(op);
    SmallVector<Operation *> ignoredUsers(ifResult.getUsers().begin(),
                                          ifResult.getUsers().end());
    for (Operation *user : ignoredUsers) {
      if (isa<annotation::MarkOp, hivm::DebugOp, tensor::DimOp>(user)) {
        user->emitWarning()
            << "dropping " << user->getName()
            << " on fallback_not_exec if result; the tensor no longer exists "
               "after InlineFixpipe sink";
        rewriter.eraseOp(user);
      }
    }
    rewriter.eraseOp(ifOp);
    return success();
  }
};

void populateInlineFixpipePatterns(RewritePatternSet &patterns,
                                   bool inlineQuantScale) {
  patterns.add<InlineFixpipeOpPattern>(patterns.getContext(), inlineQuantScale);
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

void InlineFixpipe::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateInlineFixpipePatterns(patterns, inlineQuantScale);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  eraseInlinableQuantScaleMarkOps(getOperation());
}

std::unique_ptr<Pass>
createInlineFixpipePass(const InlineFixpipeOptions &options) {
  return std::make_unique<InlineFixpipe>(options);
}

} // namespace mlir::hivm
