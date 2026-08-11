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
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTFIXPIPE
#define GEN_PASS_DEF_INLINEFIXPIPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-inline-fixpipe"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {
static constexpr llvm::StringLiteral printType = "print";
static constexpr llvm::StringLiteral mmadFixpipeForResultAlreadyInserted =
    "fixpipe_for_result_already_inserted";

static constexpr llvm::StringLiteral fixpipeDoNotMoveOutOfScfFor =
    "do_not_move_out_of_scffor";

static constexpr llvm::StringLiteral scfforFixpipeForMMADResultAlreadyInserted =
    "fixpipe_for_mmad_result_already_inserted";

/// Return true when \p op is nested in a scope marked for the vector core.
/// Fixpipe must not be fused with a store in such scopes: fusion would place
/// the cube fixpipe inside the vector scope and break mix AIC/AIV splitting.
static bool isInsideVectorScope(Operation *op) {
  auto scopeOp = op->getParentOfType<scope::ScopeOp>();
  if (!scopeOp)
    return false;
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  return coreTypeAttr &&
         coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

struct InsertFixpipe : public impl::InsertFixpipeBase<InsertFixpipe> {
  using Base::Base;
  void runOnOperation() override;
};

struct InlineFixpipe : public impl::InlineFixpipeBase<InlineFixpipe> {
  using Base::Base;
  void runOnOperation() override;
};

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

static bool isRegBasedArch(Operation *op);

static bool hasSameSource(Value valA, Value valB) {
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

static bool isMixKernel(scf::IfOp ifOp, Value val) {
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

static bool needYieldOut(Operation *user, Value val) {
  if (isa<scf::ForOp>(user->getParentOp()))
    return true;
  if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp()))
    return !isMixKernel(ifOp, val);
  return false;
}

/// Push the insert point out of every scf.if that merges the result with a
/// sibling branch writing the same destination, so that a single fixpipe after
/// the scf.if serves the merged value. Inserting inside the branches instead
/// would emit one fixpipe per branch and hand consumers of the merged result
/// (such as annotation.mark bind_buffer) the fixpipe output rather than the
/// value still living in L0C.
Operation *getInsertPointOutOfIf(Operation *op, int &resultIndx) {
  Value result = op->getResult(resultIndx);
  // if op has multiple users, don't push the insert point down
  int32_t count = 0;
  scf::YieldOp yieldOperand = nullptr;
  for (Operation *user : result.getUsers()) {
    if (!isa<hivm::DebugOp>(user))
      count++;
    auto yieldOp = dyn_cast<scf::YieldOp>(user);
    if (!yieldOp)
      continue;
    auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp());
    if (!ifOp || isMixKernel(ifOp, result))
      continue;
    yieldOperand = yieldOp;
  }

  if (count > 1 || !yieldOperand)
    return op;

  auto yieldValueIndx =
      findIdx(llvm::to_vector(yieldOperand->getOperands()), result);
  if (!yieldValueIndx.has_value())
    return op;

  resultIndx = yieldValueIndx.value();
  return getInsertPointOutOfIf(yieldOperand->getParentOp(), resultIndx);
}

Operation *getInsertPoint(Operation *op, int &resultIndx) {
  auto users = op->getResult(resultIndx).getUsers();
  std::set<scf::YieldOp> yieldOperands;

  if (isRegBasedArch(op)) {
    int32_t count = 0;

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
  } else {
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

// Return the single convert_layout{ND→Fractal} user, or null.
static hivm::ConvertLayoutOp getOutputFractalConvert(Value mmadResult) {
  if (!mmadResult.hasOneUse())
    return nullptr;
  auto convert = dyn_cast<hivm::ConvertLayoutOp>(*mmadResult.user_begin());
  if (!convert)
    return nullptr;
  auto srcLayout = convert.getSrcLayoutAttr();
  auto dstLayout = convert.getDstLayoutAttr();
  if (!srcLayout.isNDLayout() ||
      dstLayout.getDataLayout() != hivm::DataLayout::Fractal)
    return nullptr;
  return convert;
}

// Emit an NZ2NZ fixpipe when result feeds a single ND→Fractal convert_layout.
// Returns false so the caller falls back to NZ2ND.
static bool tryInsertFractalOutputFixpipe(PatternRewriter &rewriter,
                                          Operation *insertAfterOp,
                                          Value result) {
  auto convert = getOutputFractalConvert(result);
  if (!convert)
    return false;
  rewriter.setInsertionPointAfter(insertAfterOp);
  Value dst = utils::createEmptyOp(rewriter, insertAfterOp->getLoc(),
                                   convert.getResult());
  auto dmaModeAttr =
      FixpipeDMAModeAttr::get(rewriter.getContext(), FixpipeDMAMode::NZ2NZ);
  auto fixpipe = rewriter.create<FixpipeOp>(
      insertAfterOp->getLoc(), /*result_tensor=*/dst.getType(), result, dst,
      dmaModeAttr, /*dual_dst_mode=*/nullptr, /*sub_block_idx=*/nullptr,
      /*pre_quant=*/nullptr, /*pre_relu=*/nullptr, /*channel_split=*/nullptr);
  rewriter.replaceAllUsesWith(convert.getResult(), fixpipe.getResultTensor());
  rewriter.eraseOp(convert);
  return true;
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

static FixpipeOp insertFixpipeToL1(PatternRewriter &rewriter, Operation *point,
                                   Value src) {
  rewriter.setInsertionPointAfter(point);

  MLIRContext *ctx = rewriter.getContext();
  auto tensorType = cast<RankedTensorType>(src.getType());
  int64_t M = tensorType.getDimSize(0);
  int64_t N = tensorType.getDimSize(1);
  static constexpr int32_t alignM = 16;
  auto numElemPerBlock = mlir::utils::getNumPerBlock(tensorType);
  int64_t M1 = M / alignM;
  int64_t N1 = N / numElemPerBlock;
  auto dstTy = RankedTensorType::get({N1, M1, alignM, numElemPerBlock},
                                     tensorType.getElementType());

  // FixpipeOp with channel_split enabled may split each channel (C0) in two
  // parts in destination.
  bool channelSplit = numElemPerBlock == alignM / 2;
  Value fixpipeInit =
      rewriter.create<tensor::EmptyOp>(point->getLoc(), dstTy,
                                       /*dynamicSizes=*/ValueRange{});
  FixpipeDMAModeAttr dmaModeAttr =
      FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2NZ);

  return rewriter.create<FixpipeOp>(
      point->getLoc(), /*result_tensor=*/fixpipeInit.getType(), src,
      fixpipeInit, dmaModeAttr,
      /*dual_dst_mode=*/nullptr, /*sub_block_idx=*/nullptr,
      /*pre_quant=*/nullptr, /*pre_relu=*/nullptr,
      /*channel_split=*/rewriter.getBoolAttr(channelSplit));
}

static FixpipeOp insertFixpipeToLocal(PatternRewriter &rewriter,
                                      Operation *point, Value src) {
  rewriter.setInsertionPointAfter(point);

  auto dst = utils::createEmptyOp(rewriter, point->getLoc(), src);
  MLIRContext *ctx = rewriter.getContext();
  FixpipeDMAModeAttr dmaModeAttr =
      FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);

  return rewriter.create<FixpipeOp>(
      point->getLoc(), /*result_tensor=*/dst.getType(), src, dst, dmaModeAttr,
      /*dual_dst_mode=*/nullptr,
      /*sub_block_idx=*/nullptr,
      /*pre_quant=*/nullptr, /*pre_relu=*/nullptr, /*channel_split=*/nullptr);
}

static bool isInsertingFixpipeToL1(Value src) {
  if (src.use_empty())
    return false;
  // used by L1 operations, such as hivm::MmadL1Op, hivm::MmadMxL1Op, etc.
  // TODO: replace to any_of to prioritize fixpipe to L1 (if enhance perf)
  return llvm::all_of(src.getUsers(), [](auto *user) {
    return isa<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.cpp.inc"
        >(user);
  });
}

static FixpipeOp insertFixpipe(PatternRewriter &rewriter, Operation *point,
                               Value src) {
  rewriter.setInsertionPointAfter(point);

  bool isMovingToL1 =
      hacc::utils::isRegBasedArch(point->getParentOfType<ModuleOp>()) &&
      isInsertingFixpipeToL1(src);

  auto fixpipe = (isMovingToL1 ? insertFixpipeToL1
                               : insertFixpipeToLocal)(rewriter, point, src);

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

/// Insert fixpipe when there is hivm::MmadL1Op, hivm::BatchMmadL1Op, or
/// hivm::MmadMxL1Op.
template <typename OpType>
struct InsertFixpipeOpPattern : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    Operation *opInst = op.getOperation();
    auto mmadLikeOpRes = op.getResultTensors()[0];

    // shouldDecomposeBiasByElementAdd is true for ElementwiseAdd regardless of
    // init; NormalizeMatmul only decomposes when init is const false, or
    // non-const on reg-based. MmadMxL1 skips fixpipe whenever init is not
    // const true (inline-bias decompose path).
    bool skipFixpipeForBiasDecompose = false;
    if (op.shouldDecomposeBiasByElementAdd()) {
      if constexpr (std::is_same_v<OpType, hivm::MmadMxL1Op>) {
        skipFixpipeForBiasDecompose = !op.isInitConstant(true);
      } else {
        skipFixpipeForBiasDecompose =
            op.isInitConstant(false) ||
            (!op.isInitConstant() && isRegBasedArch(opInst));
      }
    }
    if (skipFixpipeForBiasDecompose) {
      // the op will decompose to mmadL1 + vadd, so fixpipe cannot be inserted
      // now, and fixpipe should be inserted after the decomposition
      return failure();
    }

    if (opInst->getAttr(mmadFixpipeForResultAlreadyInserted))
      return failure();

    if (isRegBasedArch(opInst) && allUsersReachFixpipe(mmadLikeOpRes))
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
    if (isAccumulation(opInst)) {
      // only insert fixpipe outside of the for loop when it is an accumulation
      // loop
      insertAfterOp = getInsertPoint(opInst, resultIndx);
    } else {
      insertAfterOp = getInsertPointOutOfIf(opInst, resultIndx);
    }
    rewriter.setInsertionPointAfter(insertAfterOp);

    LDBG("Replacing fix pipe for " << op);
    Value result = insertAfterOp->getResult(resultIndx);
    if (!tryInsertFractalOutputFixpipe(rewriter, insertAfterOp, result))
      insertFixpipe(rewriter, insertAfterOp, result);
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
    if (isAccumulation(opInst)) {
      scf::ForOp forOp = opInst->getParentOfType<scf::ForOp>();
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
        rewriter.setInsertionPointAfter(opInst);
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

/// Return the single non-ignored user of \p v, or nullptr if there is not
/// exactly one.
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
  if (!isRegBasedArch(castOp))
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
  if (inputType.isF32() && outputType.isF16())
    return symbolizeFixpipePreQuantMode("F322F16");
  if (inputType.isF32() && outputType.isBF16())
    return symbolizeFixpipePreQuantMode("F322BF16");
  if (inputType.isInteger(32) && outputType.isInteger(8))
    return symbolizeFixpipePreQuantMode("S322I8");
  return std::nullopt;
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
    // Avoid fusing fixpipe into a VECTOR-scope store (mix AIC/AIV).
    if (isInsideVectorScope(curOp))
      return failure();

    if (isRegBasedArch(op) && op.getDmaMode() != FixpipeDMAMode::NZ2NZ) {
      if (all_of(op->getUsers(), [](auto *user) {
            return isa<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.cpp.inc"
                >(user);
          })) {

        MLIRContext *ctx = rewriter.getContext();
        auto tensorType = cast<RankedTensorType>(op.getDst().getType());
        int64_t M = tensorType.getDimSize(0);
        int64_t N = tensorType.getDimSize(1);
        static constexpr int32_t alignM = 16;
        auto numElemPerBlock = mlir::utils::getNumPerBlock(tensorType);
        int64_t M1 = M / alignM;
        int64_t N1 = N / numElemPerBlock;
        auto dstTy = RankedTensorType::get({N1, M1, alignM, numElemPerBlock},
                                           tensorType.getElementType());

        // FixpipeOp with channel_split enabled may split each channel (C0) in
        // two parts in destination.
        bool channelSplit = numElemPerBlock == alignM / 2;
        Value fixpipeInit = rewriter.create<mlir::tensor::EmptyOp>(
            op->getLoc(), dstTy, mlir::ValueRange{});
        FixpipeDMAModeAttr dmaModeAttr =
            FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2NZ);

        auto fixpipe = rewriter.create<FixpipeOp>(
            op->getLoc(), /*result_tensor=*/fixpipeInit.getType(), op.getSrc(),
            fixpipeInit, dmaModeAttr,
            /*dual_dst_mode=*/op.getDualDstModeAttr(),
            /*sub_block_idx=*/op.getSubBlockIdxAttr(),
            /*pre_quant=*/op.getPreQuantAttr(),
            /*pre_relu=*/op.getPreReluAttr(),
            /*channel_split=*/rewriter.getBoolAttr(channelSplit));

        rewriter.replaceOp(op, fixpipe.getResult(0));
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
               vMulOp && isRegBasedArch(op) &&
               (inlineQuantScale || hasQuantScaleCompileHint(vMulOp)) &&
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
    bool regBased = isRegBasedArch(op);
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
      return markOp &&
             markOp->hasAttr(utils::kInlinableQuantScaleAttr);
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
        op.getPreReluAttr(), op.getChannelSplitAttr(), op.getC0PadEnAttr(), quantScale);
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
    bool regBased = isRegBasedArch(op);
    FixpipeDMAModeAttr dmaModeAttr =
        regBased ? op.getDmaModeAttr()
                 : FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto newFixpipeOp = rewriter.create<FixpipeOp>(
        extractSliceOp.getLoc(), fixpipeInit.getType(),
        /*src=*/newExtractSliceResult, /*dst=*/fixpipeInit, dmaModeAttr,
        op.getDualDstModeAttr(), op.getSubBlockIdxAttr(), quantModeAttr,
        reluModeAttr, op.getChannelSplitAttr(), op.getC0PadEnAttr(), op.getQuantScale());
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
        FixpipeDualDstModeAttr{}, FixpipeSubBlockAttr{}, nullptr, nullptr, nullptr, nullptr);
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

void populateInsertFixpipeForIterArgPatterns(RewritePatternSet &patterns) {
  patterns.add<InsertFixpipeForIterArgMMAD>(patterns.getContext());
}

void populateInsertFixpipePatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<InsertFixpipeOpPattern<hivm::MmadL1Op>>(ctx);
  patterns.add<InsertFixpipeOpPattern<hivm::BatchMmadL1Op>>(ctx);
  patterns.add<InsertFixpipeOpPattern<hivm::MmadMxL1Op>>(ctx);
  patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv1DL1Op>>(ctx);
  patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv2DL1Op>>(ctx);
  patterns.add<InsertFixpipeForConvOpPattern<hivm::Conv3DL1Op>>(ctx);
}

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

void InsertFixpipe::runOnOperation() {
  RewritePatternSet iterArgPatterns(&getContext());
  populateInsertFixpipeForIterArgPatterns(iterArgPatterns);
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(iterArgPatterns)))) {
    signalPassFailure();
    return;
  }

  RewritePatternSet patterns(&getContext());
  populateInsertFixpipePatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  RewritePatternSet insertFixpipeForDevicePrintPattern(&getContext());
  MLIRContext *ctx = insertFixpipeForDevicePrintPattern.getContext();
  insertFixpipeForDevicePrintPattern.add<InsertFixpipeForDevicePrint>(ctx);
  if (failed(applyPatternsGreedily(
          getOperation(), std::move(insertFixpipeForDevicePrintPattern)))) {
    signalPassFailure();
  }
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

std::unique_ptr<Pass> createInsertFixpipePass() {
  return std::make_unique<InsertFixpipe>();
}

std::unique_ptr<Pass>
createInlineFixpipePass(const InlineFixpipeOptions &options) {
  return std::make_unique<InlineFixpipe>(options);
}

} // namespace mlir::hivm
