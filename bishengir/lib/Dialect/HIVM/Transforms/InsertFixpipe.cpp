//===- InsertFixpipe.cpp - Insert fixpipe ops -----------------------------===//
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
#define GEN_PASS_DEF_INSERTFIXPIPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "hivm-insert-fixpipe"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::hivm {

static constexpr llvm::StringLiteral printType = "print";
static constexpr llvm::StringLiteral mmadFixpipeForResultAlreadyInserted =
    "fixpipe_for_result_already_inserted";

static constexpr llvm::StringLiteral scfforFixpipeForMMADResultAlreadyInserted =
    "fixpipe_for_mmad_result_already_inserted";

struct InsertFixpipe : public impl::InsertFixpipeBase<InsertFixpipe> {
  using Base::Base;
  void runOnOperation() override;
};

/// Walk a region result through ancestor yields. True when this result, or a
/// result it is forwarded to, is required to stay in L0C.
static bool isForwardedResultRequiredInL0C(OpResult result) {
  if (isResultInL0C(result))
    return true;
  for (OpOperand &use : result.getUses()) {
    auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner());
    if (!yieldOp)
      continue;
    auto parent = dyn_cast<RegionBranchOpInterface>(yieldOp->getParentOp());
    if (!parent)
      continue;
    if (isForwardedResultRequiredInL0C(
            parent->getOpResult(use.getOperandNumber())))
      return true;
  }
  return false;
}

/// Keep a yield on the raw L0C value when rewriting it to a fixpipe result
/// would evict a value that must stay in L0C:
///  1. The producer is marked remain_in_l0c (the enclosing scf.for may not
///     carry the loop-level attribute, e.g. a nested accumulation), or
///  2. This yield, or an ancestor yield it is forwarded to, produces a
///     region result required in L0C.
static bool shouldKeepYieldInL0C(OpOperand &use) {
  if (!isa<scf::YieldOp>(use.getOwner()))
    return false;
  auto parentOp =
      dyn_cast<RegionBranchOpInterface>(use.getOwner()->getParentOp());
  if (!parentOp)
    return false;
  return isForwardedResultRequiredInL0C(
      parentOp->getOpResult(use.getOperandNumber()));
}

static hivm::MmadL1Op traceAlongL0C(Value val) {
  hivm::MmadL1Op source = llvm::dyn_cast_if_present<hivm::MmadL1Op>(
      traceDefOp<hivm::MmadL1Op>(val).value_or(nullptr));
  if (!source)
    return nullptr;
  hivm::MmadL1Op upstream = traceAlongL0C(source.getC());
  if (upstream)
    return upstream;
  return source;
}

static bool hasSameSource(Value valA, Value valB) {
  auto mmadA = traceAlongL0C(valA);
  auto mmadB = traceAlongL0C(valB);

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
  if (mmadA && mmadB) {
    return isMatchedEmptyOrAllocOp(mmadA.getC(), mmadB.getC());
  }

  if (mmadA) {
    return isMatchedEmptyOrAllocOp(mmadA.getC(), valB);
  }

  if (mmadB) {
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
  OpResult result = op->getResult(resultIndx);
  if (!op->hasAttr(RemainInL0CAttr::name))
    return op;
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
    if (!ifOp || (!isResultInL0C(result) && isMixKernel(ifOp, result)))
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

  if (auto mmadMxOp = dyn_cast<hivm::MmadMxL1Op>(op))
    return isAccumulationImpl(op, mmadMxOp.getC());

  if (auto conv1d = dyn_cast<hivm::Conv1DL1Op>(op))
    return isAccumulationImpl(op, conv1d.getInit());

  if (auto conv2d = dyn_cast<hivm::Conv2DL1Op>(op))
    return isAccumulationImpl(op, conv2d.getInit());

  if (auto conv3d = dyn_cast<hivm::Conv3DL1Op>(op))
    return isAccumulationImpl(op, conv3d.getInit());

  return false;
}

/// NZ fractal dest type for L1 fixpipe: [N1, M1, 16, C0].
/// Use ceilDiv so M/N below the fractal tile (e.g. M=1) pad instead of
/// producing a zero-sized dimension (M1 = M/16 == 0).

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

static FixpipeOp insertFixpipeToL1(PatternRewriter &rewriter, Operation *point,
                                   Value src) {
  auto fixpipe = insertFixpipeToLocal(rewriter, point, src);
  return convertNz2NdFixpipeToNz2Nz(rewriter, fixpipe);
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

/// Return true when \p v is consumed only as the C init of a single local
/// matmul. If \p v also feeds that matmul's A/B inputs, fixpipe is still
/// required to move the L0C result to L1 for the input operand.
static bool isExclusiveLocalMatmulInit(Operation *op, Value v) {
  if (!isLocalMatmulInit(op, v))
    return false;
  unsigned usesOnOp = 0;
  for (OpOperand &use : v.getUses()) {
    if (use.getOwner() == op)
      ++usesOnOp;
  }
  return usesOnOp == 1;
}

static FixpipeOp insertFixpipe(PatternRewriter &rewriter, Operation *point,
                               Value src) {
  rewriter.setInsertionPointAfter(point);
  const bool isRegBased = isOnRegBasedArch(point);
  bool isMovingToL1 = isRegBased && isInsertingFixpipeToL1(src);

  auto fixpipe = (isMovingToL1 ? insertFixpipeToL1
                               : insertFixpipeToLocal)(rewriter, point, src);

  rewriter.replaceUsesWithIf(
      src, fixpipe.getResultTensor(), [](OpOperand &use) {
        auto *op = use.getOwner();
        if (isa<DebugOp>(op) || isa<FixpipeOp>(op) ||
            isa<annotation::MarkOp>(op)) {
          return false;
        }
        // mmadL1 and mmadmxL1 both implement LocalMatmulLikeOpInterface, and
        // their C operand is the DPS init. Keep that init on a raw result of
        // the same op so the L0C accumulator is not replaced by a fixpipe.
        // batchMmadL1 is the other implementer; a direct same-op C chain is
        // treated the same way. A/B uses still take the fixpipe.
        if (isa<hivm::LocalMatmulLikeOpInterface>(op)) {
          auto dpsOp = cast<DestinationStyleOpInterface>(op);
          return !llvm::any_of(
              dpsOp.getDpsInitsMutable(), [&](OpOperand &init) {
                Operation *def = init.get().getDefiningOp();
                return &use == &init && def && def->getName() == op->getName();
              });
        }
        // handle normalized_in_l0c and remain_in_l0c, including nested
        // scf.for yields whose enclosing loop (or the mmad itself) keeps
        // the value in L0C.
        if (isRemainInL0c(use) && shouldKeepYieldInL0C(use))
          return false;
        return true;
      });

  auto dst = fixpipe.getDst();
  // in MemBased Architecture, there is fixpipe that goes to
  // GM with load user.
  if (isRegBased && !fixpipe->getUsers().empty()) {
    auto emptyOp = dst.getDefiningOp<tensor::EmptyOp>();
    if (emptyOp) {
      emptyOp->setAttr(hivm::kInsertedTensorAttr::name, rewriter.getUnitAttr());
      emptyOp->setAttr(
          hivm::AddressSpaceAttr::name,
          rewriter.getAttr<hivm::AddressSpaceAttr>(
              isMovingToL1 ? hivm::AddressSpace::L1 : hivm::AddressSpace::UB));
    }
  }

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

/// Skip inserting outer fixpipe after an accumulation loop when that *loop
/// result* stays in L0C for a later matmul outs. The in-loop mmad result
/// always cycles back to iter_arg outs, so consulting it would skip every
/// remain_in_l0c accumulation, including results consumed by Vector ops
/// such as vsel. remain_in_l0c is loop-wide; normalized_in_L0C names the
/// result indices that actually stay in L0C.
static bool shouldSkipOuterFixpipeForAccumulation(Operation *opInst,
                                                  Value mmadLikeOpRes) {
  if (!isAccumulation(opInst))
    return false;

  int resultIndx = 0;
  Operation *insertAfterOp = getInsertPoint(opInst, resultIndx);
  OpResult loopResult = insertAfterOp->getResult(resultIndx);
  if (Value(loopResult) == mmadLikeOpRes)
    return false;

  auto forOp = dyn_cast<scf::ForOp>(insertAfterOp);
  if (!forOp)
    return false;
  if (!isResultInL0C(loopResult))
    return false;

  return anyUserReachesMatmulOuts(loopResult);
}

/// Check whether every meaningful user reaches a fixpipe, tracing through
/// extract_slice chains. This covers branch fan-out where a single-user-chain
/// query cannot detect that fixpipes have already been inserted.
static bool allUsersReachFixpipe(Value value) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : value.getUses()) {
    if (isa<tensor::DimOp, annotation::MarkOp>(use.getOwner()))
      continue;
    uses.push_back(&use);
  }
  if (uses.empty())
    return false;
  return llvm::all_of(uses, [](OpOperand *use) {
    if (auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(use->getOwner());
        fixpipeOp && fixpipeOp.getSrc() == use->get())
      return true;
    if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(use->getOwner()))
      return allUsersReachFixpipe(extractSlice.getResult());
    if (auto forOp = dyn_cast<scf::ForOp>(use->getOwner())) {
      if (auto iterArg = forOp.getTiedLoopRegionIterArg(use))
        return allUsersReachFixpipe(iterArg);
      if (auto result = forOp.getTiedLoopResult(use))
        return allUsersReachFixpipe(result);
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(use->getOwner()))
      return allUsersReachFixpipe(
          yieldOp->getParentOp()->getResult(use->getOperandNumber()));
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
            (!op.isInitConstant() && isOnRegBasedArch(opInst));
      }
      skipFixpipeForBiasDecompose =
          skipFixpipeForBiasDecompose && !isResultInL0C(mmadLikeOpRes);
    }
    if (skipFixpipeForBiasDecompose) {
      // the op will decompose to mmadL1 + vadd, so fixpipe cannot be inserted
      // now, and fixpipe should be inserted after the decomposition
      return failure();
    }

    if (opInst->getAttr(mmadFixpipeForResultAlreadyInserted))
      return failure();

    bool changed = false;
    if (isOnRegBasedArch(opInst) && allUsersReachFixpipe(mmadLikeOpRes))
      return failure();
    if (!shouldSkipOuterFixpipeForAccumulation(opInst, mmadLikeOpRes)) {

      auto isMatchedOp = [](Operation *op, Value v) {
        LDBG("Matching this current op " << *op);
        if (isa<hivm::FixpipeOp>(op))
          return true;
        if (isExclusiveLocalMatmulInit(op, v)) {
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
        // only insert fixpipe outside of the for loop when it is an
        // accumulation loop
        insertAfterOp = getInsertPoint(opInst, resultIndx);
      } else {
        insertAfterOp = getInsertPointOutOfIf(opInst, resultIndx);
      }
      rewriter.setInsertionPointAfter(insertAfterOp);

      LDBG("Replacing fix pipe for " << op);
      Value result = insertAfterOp->getResult(resultIndx);
      if (!tryInsertFractalOutputFixpipe(rewriter, insertAfterOp, result)) {
        LDBG("Inserting fractal output fixpipe after " << *insertAfterOp);
        insertFixpipe(rewriter, insertAfterOp, result);
        op->setAttr(mmadFixpipeForResultAlreadyInserted,
                    rewriter.getBoolAttr(true));
        changed = true;
      }
    }

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
        changed = true;
      }
    }
    return changed ? success() : failure();
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
        FixpipeDualDstModeAttr{}, FixpipeSubBlockAttr{}, nullptr, nullptr,
        nullptr, nullptr);
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

/// Move fixpipe from after scf.for into the scf.if branch that yields the loop
/// result.
///
/// Before:
///   %res = scf.for ... { scf.yield %mmad }
///   %fix = hivm.hir.fixpipe ins(%res) ...
///   %if = scf.if ... {
///     ...
///   } else {
///     scf.yield %res
///   }
///
/// After:
///   %res = scf.for ... { scf.yield %mmad }
///   %if = scf.if ... {
///     ...
///   } else {
///     %fix = hivm.hir.fixpipe ins(%res) ...
///     scf.yield %fix
///   }
struct MoveFixpipeIntoScfIfBranchPattern : public OpRewritePattern<FixpipeOp> {
  using OpRewritePattern<FixpipeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FixpipeOp fixpipe,
                                PatternRewriter &rewriter) const override {
    auto fixpipeResTensor = fixpipe.getResultTensor();
    if (!fixpipeResTensor)
      return rewriter.notifyMatchFailure(fixpipe,
                                         "fixpipe has no result tensor");

    auto forResult = dyn_cast<OpResult>(fixpipe.getSource());
    if (!forResult || !isa<scf::ForOp>(forResult.getOwner()))
      return rewriter.notifyMatchFailure(
          fixpipe, "fixpipe source is not an scf.for result");

    if (!fixpipeResTensor.hasOneUse())
      return rewriter.notifyMatchFailure(fixpipe,
                                         "fixpipe result has multiple uses");

    scf::YieldOp yieldOp =
        dyn_cast<scf::YieldOp>(*fixpipeResTensor.user_begin());
    if (!yieldOp)
      return rewriter.notifyMatchFailure(
          fixpipe, "fixpipe result is not yielded by an scf.yield");

    if (fixpipe->getParentRegion() == yieldOp->getParentRegion())
      return rewriter.notifyMatchFailure(
          fixpipe, "fixpipe is already in the same region as the yield");

    auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp());
    if (!ifOp)
      return rewriter.notifyMatchFailure(yieldOp,
                                         "fixpipe yield parent is not scf.if");

    auto yieldIdx =
        findIdx(llvm::to_vector(yieldOp.getOperands()), fixpipeResTensor);
    if (!yieldIdx.has_value())
      return rewriter.notifyMatchFailure(
          yieldOp, "fixpipe result is not a yield operand");

    // Only move when the loop result is consumed by this fixpipe.
    if (!forResult.hasOneUse() ||
        *forResult.user_begin() != fixpipe.getOperation())
      return rewriter.notifyMatchFailure(
          fixpipe, "scf.for result has uses other than this fixpipe");

    LDBG("MoveFixpipeIntoScfIfBranch for " << fixpipe);
    rewriter.moveOpBefore(fixpipe, yieldOp);
    return success();
  }
};

void populateMoveFixpipePatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<MoveFixpipeIntoScfIfBranchPattern>(ctx);
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

  RewritePatternSet moveFixpipePatterns(&getContext());
  mlir::hivm::populateMoveFixpipePatterns(moveFixpipePatterns);
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(moveFixpipePatterns)))) {
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

std::unique_ptr<Pass> createInsertFixpipePass() {
  return std::make_unique<InsertFixpipe>();
}

} // namespace mlir::hivm
