//===---------------------- InlineFixpipe.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts ops to hivm.fixpipe .
//
//===----------------------------------------------------------------------===//

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
constexpr llvm::StringLiteral mmadFixpipeForResultAlreadyInserted =
    "fixpipe_for_result_already_inserted";

constexpr llvm::StringLiteral fixpipeDoNotMoveOutOfScfFor =
    "do_not_move_out_of_scffor";

constexpr llvm::StringLiteral scfforFixpipeForMMADResultAlreadyInserted =
    "fixpipe_for_mmad_result_already_inserted";
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

// ifOp except the case
// %empty_tensor = tensor.empty
// %if = if %condition {
//  %mmad1 = hivm.hir.mmad -> %empty_tensor
//  yield %mmad1
// } else {
//  %mmad2 = hivm.hir.mmad -> %empty_tensor
//  yield %mmad2
// }
// %fixpipe = hivm.hir.fixpipe(%if)
// other cases are expected that fixpipe is inserted inside if body. Hence, the
// function will return true
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

/// Infer a conservative fixpipe dma mode from src/dst shaped values.
///
/// If src and dst are shape-compatible, fixpipe acts as a layout-preserving
/// copy (e.g. NZ2NZ or ND2ND-like path), so use the normal dma mode. Otherwise,
/// fallback to NZ2ND for layout-conversion cases.
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
    // For same-shape cases, use rank to distinguish ND-like tensors.
    // A 2D destination is treated as ND; otherwise keep normal mode.
    if (dstType.getRank() == 2)
      return FixpipeDMAMode::NZ2ND;
    return FixpipeDMAMode::NZ2NZ;
  }

  return FixpipeDMAMode::NZ2ND;
}

static FixpipeOp insertFixpipe(PatternRewriter &rewriter,
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
  InsertFixpipeForIterArgMMAD(MLIRContext *context,
                              InsertFixpipePatternOptions options)
      : OpRewritePattern<scf::ForOp>(context), options(options) {}

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
          // skip loop counter
          continue;
        }

        auto yielded = scffor.getYieldedValues()[idx - 1];

        auto stopOp =
            utils::valueCalculatedUsingOperationInsideBlock<hivm::MmadL1Op>(
                yielded, mmad, forBlock);
        if (stopOp && *stopOp == mmad) {
          LDBG("Inserting fix pipe for " << scffor);

          auto fixpipe =
              insertFixpipe(rewriter, options, mmad, mmad->getResults()[0]);
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

/// Pushed down insert point only when a unique yieldop is trace
Operation *getInsertPoint(Operation *op, int &resultIndx) {
  // if op has multiple users, don't push the insert point down
  int32_t count = 0;
  auto users = op->getResult(resultIndx).getUsers();
  std::set<scf::YieldOp> yieldOperands;
  for (auto *user : users) {
    // TODO: add auto tracedDownUser = traceDown(user) and use tracedDownUser to
    // judge
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
  return getInsertPoint(yieldParentOp, resultIndx);
}

/// Check if all non-ignored users of a value are fixpipe ops, optionally
/// traced through tensor.extract_slice chains.
/// This handles cases where the mmad result flows into multiple fixpipes
/// (e.g., inside scf.if branches), where traceSingleChainUser fails
/// because there isn't a single user chain.
static bool allUsersReachFixpipe(Value v) {
  SmallVector<Operation *> users;
  for (auto *user : v.getUsers()) {
    if (isa<tensor::DimOp, annotation::MarkOp>(user))
      continue;
    users.push_back(user);
  }
  if (users.empty())
    return false;
  return llvm::all_of(users, [](auto user) {
    if (isa<hivm::FixpipeOp>(user))
      return true;
    // Trace through extract_slice
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user))
      return allUsersReachFixpipe(extractSliceOp.getResult());
    return false;
  });
}

/// Insert fixpipe when there is hivm::MmadL1Op or hivm::BatchMmadL1Op.
template <typename OpType>
struct InsertFixpipeOpPattern : public OpRewritePattern<OpType> {
public:
  InsertFixpipeOpPattern(MLIRContext *context,
                         InsertFixpipePatternOptions options)
      : OpRewritePattern<OpType>(context), options(options) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto mmadLikeOpRes = op.getResultTensors()[0];
    if (op.shouldDecomposeBiasByElementAdd() && !op.isInitConstant(true)) {
      // the op will decompose to mmadL1 + vadd, so fixpipe cannot be inserted
      // now, and fixpipe should be inserted after the decomposition
      return failure();
    }

    // the following check is needed because now there could be two fixpipe ops
    // from the same mmad; you cannot rely on traceSingleChainUser to avoid
    // duplicating fixpipes anymore
    if (op->getAttr(mmadFixpipeForResultAlreadyInserted))
      return failure();

    // Check if all non-ignored users of the mmad result are fixpipes
    // (possibly through extract_slice/insert_slice chains such as in
    // scf.if branches). traceSingleChainUser cannot detect this case
    // because there is not a single user chain.
    if (allUsersReachFixpipe(mmadLikeOpRes))
      return failure();

    auto isMatchedOp = [](Operation *op, Value v) {
      LDBG("Matching this current op " << *op);
      if (isa<hivm::FixpipeOp>(op)) {
        // already insert fixpipe, no need to insert fixpipe again
        return true;
      }
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
    auto insertAfterOp = getInsertPoint(op, resultIndx);
    rewriter.setInsertionPointAfter(insertAfterOp);

    LDBG("Replacing fix pipe for " << op);
    insertFixpipe(rewriter, options, insertAfterOp,
                  insertAfterOp->getResult(resultIndx));
    op->setAttr(mmadFixpipeForResultAlreadyInserted,
                rewriter.getBoolAttr(true));
    return success();
  }

private:
  InsertFixpipePatternOptions options;
};

template <>
struct InsertFixpipeOpPattern<hivm::MmadMxL1Op>
    : public OpRewritePattern<hivm::MmadMxL1Op> {
public:
  InsertFixpipeOpPattern(MLIRContext *context,
                         InsertFixpipePatternOptions options)
      : OpRewritePattern<hivm::MmadMxL1Op>(context), options(options) {}

  LogicalResult matchAndRewrite(hivm::MmadMxL1Op op,
                                PatternRewriter &rewriter) const override {
    auto mmadLikeOpRes = op->getResults()[0];

    // Check if all non-ignored users of the mmad result are fixpipes
    // (possibly through extract_slice/insert_slice chains such as in
    // scf.if branches). traceSingleChainUser cannot detect this case
    // because there is not a single user chain.
    if (allUsersReachFixpipe(mmadLikeOpRes))
      return failure();

    auto isMatchedOp = [](Operation *op, Value v) {
      LDBG("Matching this current op " << *op);
      if (isa<hivm::FixpipeOp>(op)) {
        // already insert fixpipe, no need to insert fixpipe again
        return true;
      }
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
    auto insertAfterOp = getInsertPoint(op, resultIndx);
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

/// Fixpipe pre-quant cannot implement integer narrowing casts that disable
/// saturation (e.g. trunc-with-overflow semantics).
static bool isVcastInlinableIntoFixpipe(hivm::VCastOp castOp) {
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

template <typename OpType>
std::optional<FixpipePreReluMode> getReluMode(OpType op) {
  if constexpr (std::is_same_v<OpType, hivm::VReluOp>) {
    return hivm::symbolizeFixpipePreReluMode("NORMAL_RELU");
  }
  llvm_unreachable("unsupported ReluValue");
}

Type getInitType(Value v, hivm::FixpipePreQuantMode quant,
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
  InlineFixpipeOpPattern(MLIRContext *ctx, InlineFixpipePatternOptions options)
      : OpRewritePattern<FixpipeOp>(ctx), options(options) {}

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
    // FixPipe followed by debugOp only, no need to inline
    if (curOp == nullptr)
      return success();

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
      // change to fixpipe op + extract_slice to extract_slice + fixpipe op
      if (op->getBlock() == extractSliceOp->getBlock()) {
        // only swap when fixpipe op and extract slice op are in same block,
        // otherwise, extract slice op may be in sub block loop and fixpipe
        // cannot be fused into.
        matched = true;
        swapFixpipeAndExtractSliceOp(rewriter, loc, op, extractSliceOp);
      }
    } else if (isa<tensor::InsertSliceOp>(curOp) &&
               hasCompatibleShape(
                   op.getSource(),
                   cast<tensor::InsertSliceOp>(curOp).getSource())) {
      auto insertSliceOp = cast<tensor::InsertSliceOp>(curOp);
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
    // it will infer the quant pre mode and set the quant scale
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
        getInitType(newExtractSliceResult, op.getPreQuant(), rewriter));
    SmallVector<Value> oprs({newExtractSliceResult, fixpipeInit});
    if (auto quantScale = op.getQuantScale())
      oprs.push_back(quantScale);
    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
        extractSliceOp.getLoc(), fixpipeInit.getType(), oprs, op->getAttrs());
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
        getInitType(newInsertSliceResult, op.getPreQuant(), rewriter));
    SmallVector<Value> oprs({newInsertSliceResult, fixpipeInit});
    if (auto quantScale = op.getQuantScale())
      oprs.push_back(quantScale);
    auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
        insertSliceOp.getLoc(), TypeRange{fixpipeInit}, oprs, op->getAttrs());
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
  patterns.add<InsertFixpipeForIterArgMMAD>(ctx, options);
}

void mlir::hivm::populateInsertFixpipePatterns(
    RewritePatternSet &patterns, InsertFixpipePatternOptions options) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InsertFixpipeOpPattern<hivm::MmadL1Op>>(ctx, options);
  patterns.add<InsertFixpipeOpPattern<hivm::BatchMmadL1Op>>(ctx, options);
  patterns.add<InsertFixpipeOpPattern<hivm::MmadMxL1Op>>(ctx, options);
}

void mlir::hivm::populateInlineFixpipePatterns(
    RewritePatternSet &patterns, InlineFixpipePatternOptions options) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InlineFixpipeOpPattern>(ctx, options);
}

struct InsertFixpipeForDevicePrint : public OpRewritePattern<DebugOp> {
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

    // 2. Use bufferization::ToTensorOp to convert current workspace to tensor
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
  patterns.add<InsertFixpipeForDevicePrint>(ctx);
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
  RewritePatternSet patterns(&getContext());
  InlineFixpipePatternOptions options;
  options.inlineQuantScale = inlineQuantScale;
  mlir::hivm::populateInlineFixpipePatterns(patterns, options);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }

  eraseInlinableQuantScaleMarkOps(getOperation());
}

std::unique_ptr<Pass>
mlir::hivm::createInsertFixpipePass(const InsertFixpipeOptions &options) {
  return std::make_unique<InsertFixpipe>(options);
}

std::unique_ptr<Pass>
mlir::hivm::createInlineFixpipePass(const InlineFixpipeOptions &options) {
  return std::make_unique<InlineFixpipe>(options);
}
