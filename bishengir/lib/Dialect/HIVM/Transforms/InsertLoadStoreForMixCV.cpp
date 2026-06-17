//===-------------------- InsertLoadStoreForMixCV.cpp----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts load/store op for mix cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/InsertPropagation.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/PropagateOp.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/ResolvePropagation.h"
#include "bishengir/Dialect/HIVM/Transforms/InsertLoadStoreForMixCV/Utils.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_INSERTLOADSTOREFORMIXCV
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "insert-load-store"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
static const llvm::StringLiteral insertedConvertLayout = "inserted-convert-layout";

llvm::SmallVector<size_t> getIntegersOfArrayAttr(Operation *op, llvm::StringLiteral attrName) {
  ArrayAttr arr = op->getAttrOfType<ArrayAttr>(attrName);
  llvm::SmallVector<size_t> result;
  if (!arr)
    return result;
  result.reserve(arr.size());
  for (Attribute attr : arr) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    // In production you may want to assert or handle errors.
    assert(intAttr && "ArrayAttr element is not an IntegerAttr");
    // Use getValue().getZExtValue() to interpret as unsigned.
    result.push_back(intAttr.getValue().getZExtValue());
  }
  return result;
}

void insertToIntegerAttrOfArrayAttr(Operation *op, llvm::StringLiteral attrName, size_t value, PatternRewriter &rewriter) {
  auto arr = getIntegersOfArrayAttr(op, attrName);
  if (std::find(arr.begin(), arr.end(), value) != arr.end())
    return;
  arr.push_back(value);
  std::sort(arr.begin(), arr.end());
  
  SmallVector<Attribute> newAttrs;
  for(auto idx : arr) {
    newAttrs.push_back(rewriter.getIntegerAttr(rewriter.getIntegerType(64), idx));
  }
  op->setAttr(attrName, rewriter.getArrayAttr(newAttrs));
}

struct InsertLoadStoreForMixCVPass
    : public impl::InsertLoadStoreForMixCVBase<InsertLoadStoreForMixCVPass> {
public:
  using Base::Base;
  void runOnOperation() override;
private:
  void runLegacyInsertLoadStoreForMixCV();
  bool isA5Target();
  bool isEnabledTightCoupledBuffer();
  LogicalResult insertPropagationOp(func::FuncOp funcOp);
  LogicalResult propagateAndResolve(func::FuncOp funcOp);
  LogicalResult runPropagateOpPatterns(func::FuncOp funcOp,
    PropagationStep step);
  LogicalResult ensureAllocInUbAddressSpace(func::FuncOp funcOp);
  LogicalResult addConvertLayoutUBToL1(func::FuncOp funcOp);
};

bool InsertLoadStoreForMixCVPass::isA5Target() {
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
  auto moduleTarget = hacc::utils::getTargetDevice(moduleOp);
  if (moduleTarget.has_value()) {
    return hacc::utils::isAscend950(moduleTarget.value());
  }
  return hacc::utils::isAscend950(this->target);
}

bool InsertLoadStoreForMixCVPass::isEnabledTightCoupledBuffer()  {
  if (disableTightCoupledBuffer) return false;
  return isA5Target();
}


enum class InsertMode { LoadOnly = 0, StoreOnly, LoadAndStore };

LogicalResult
InsertOpHelper(PatternRewriter &rewriter, Location loc,
               const llvm::SmallVector<OpOperand *> &consumerOperands,
               InsertMode insertMode,
               std::optional<Value> insertInit = std::nullopt) {
  if (consumerOperands.empty()) {
    return failure();
  }

  for (OpOperand *consumerOperand : consumerOperands) {
    Operation *lastInsertOp = nullptr;
    rewriter.setInsertionPointAfterValue(consumerOperand->get());
    Type type = consumerOperand->get().getType();
    Type elemType = getElementTypeOrSelf(type);
    if (insertMode == InsertMode::LoadOnly) {
      Value loadInit =
          insertInit.has_value()
              ? insertInit.value()
              : mlir::utils::createEmptyOpWithTargetElemType(
                    rewriter, loc, consumerOperand->get(), elemType);
      lastInsertOp = rewriter.create<hivm::LoadOp>(
          loc, TypeRange(type), consumerOperand->get(), loadInit);
    } else if (insertMode == InsertMode::StoreOnly) {
      Value storeInit =
          insertInit.has_value()
              ? insertInit.value()
              : utils::createEmptyOp(rewriter, loc, consumerOperand->get());
      lastInsertOp = rewriter.create<hivm::StoreOp>(
          loc, TypeRange(type), consumerOperand->get(), storeInit);
    } else if (insertMode == InsertMode::LoadAndStore) {
      Value storeInit =
          utils::createEmptyOp(rewriter, loc, consumerOperand->get());
      auto storeOp = rewriter.create<hivm::StoreOp>(
          loc, TypeRange(type), consumerOperand->get(), storeInit);
      Value loadInit = mlir::utils::createEmptyOpWithTargetElemType(
          rewriter, loc, consumerOperand->get(), elemType);
      lastInsertOp = rewriter.create<hivm::LoadOp>(
          loc, TypeRange(type), storeOp->getResults()[0], loadInit);
    }
    if (!lastInsertOp) {
      llvm_unreachable("lastInsertOp not defined");
      return failure();
    }
    rewriter.modifyOpInPlace(consumerOperand->getOwner(), [&]() {
      consumerOperand->set(lastInsertOp->getResult(0));
    });
  }

  return success();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// InsertLoadStoreOp
//===----------------------------------------------------------------------===//
/// pattern1 : fixpipe op / store op + vector/mmadl1
/// convert into fixpipe op / store op  + hivm.load + vector/mmadl1
template <typename OpType>
struct InsertLoadOpBetweenStoreLikeAndVectorOrCube
    : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!isa<hivm::HIVMStructuredOp, hivm::BitcastOp>(op.getOperation())) {
      return failure();
    }

    auto hivmOp = op.getOperation();
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : hivmOp->getOpOperands()) {
      if (traceDefOp<hivm::FixpipeOp>(operand.get()).has_value() ||
          traceDefOp<hivm::StoreOp>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return InsertOpHelper(rewriter, hivmOp->getLoc(), consumerOperands,
                          InsertMode::LoadOnly);
  }
};

/// pattern2 : vector(src, dst) + hivm.load(dst)
/// convert into  vector(src, dst) + hivm.store(dst) + hivm.load
template <typename OpType>
struct InsertStoreOpBetweenVectorAndLoad
    : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;
  ~InsertStoreOpBetweenVectorAndLoad() override = default;
  LogicalResult matchAndRewrite(hivm::LoadOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (traceDefOp<OpType>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::StoreOnly);
  }
};

/// pattern3 : vector(dst) + cube(dst)
/// convert into vector + hivm.store + hivm.load  +cube
template <typename OpType>
struct InsertLoadStoreOpBetweenVectorAndCube
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  ~InsertLoadStoreOpBetweenVectorAndCube() override = default;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (traceDefOp<OpType>(operand.get()).has_value()) {
        consumerOperands.push_back(&operand);
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern3 : scf::for{discrete load src using scalar} with attr {ExtractedLoadOrStore} + cube(src)
/// ExtractedLoadOrStore describes the process of discretely loading scalars on ub.
/// convert into scf::for{discrete load src using scalar} + hivm.store + hivm.load  + cube
/// for example
/// dst = tensor.empty : 16
/// for i in 16
///   dst[i]=src[offset[i]]
/// mmadl1(dst)
/// convert into
/// dst = tensor.empty : 16
/// for i in 16
///    dst[i]=src[offset[i]]
///  gm_dst = store dst to gm
///  l1_dst = load dst from gm
///  mmadl1(l1_dst)
template <>
struct InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      auto scfForDef = traceDefOp<scf::ForOp>(operand.get());
      if (scfForDef.has_value()) {
        auto forOp = llvm::cast<scf::ForOp>(scfForDef.value());
        if (forOp->getAttr(ExtractLoadStoreAttr) != nullptr) {
          consumerOperands.push_back(&operand);
        }
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern3 : bufferization::ToTensorOp with ToBeTransposed mark + cube
/// describe there is an implicit transpose for the input of cube. store & load
/// op will be added here in order to make transpose operation happen in vector
template <>
struct InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<OpOperand *> consumerOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      auto toTensorOpDef = traceDefOp<bufferization::ToTensorOp>(operand.get());
      if (!toTensorOpDef.has_value())
        continue;
      auto toTensorOp =
          llvm::cast<bufferization::ToTensorOp>(toTensorOpDef.value());
      auto maybeAnnotateOp = utils::getAnnotateOpWithAttr(
          toTensorOp->getResult(0), "ToBeTransposed");
      if (maybeAnnotateOp.has_value()) {
        consumerOperands.push_back(&operand);
        rewriter.eraseOp(maybeAnnotateOp.value());
      }
    }
    return InsertOpHelper(rewriter, op.getLoc(), consumerOperands,
                          InsertMode::LoadAndStore);
  }
};

/// pattern4
/// %1 = fixpipe
/// %4 = scf.for iter_args(%arg0 = %1) {
///    %2 = load(%arg0)
///    %3 = vadd(%2, ...)
///    scf.yield %3
/// }
///
/// is converted into
///
/// %1 = fixpipe
/// %5 = scf.for iter_args(%arg0 = %1) {
///    %2 = load(%arg0)
///    %3 = vadd(%2, ...)
///    %4 = %store (%3)
///    scf.yield %4
/// }

struct InsertStoreForSCFYield : public OpRewritePattern<hivm::LoadOp> {
  using OpRewritePattern<hivm::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp.hasPureTensorSemantics()) {
      return failure();
    }
    auto blockArg = dyn_cast_if_present<BlockArgument>(loadOp.getSrc());
    if (!blockArg) {
      return failure();
    }
    auto scfForOp =
        dyn_cast_if_present<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!scfForOp) {
      return failure();
    }
    OpOperand *yieldOperand = scfForOp.getTiedLoopYieldedValue(blockArg);
    if (!yieldOperand) {
      return failure();
    }
    auto correspondYieldVal = yieldOperand->get();
    if (traceDefOp<hivm::FixpipeOp>(correspondYieldVal).has_value() ||
        traceDefOp<hivm::StoreOp>(correspondYieldVal).has_value()) {
      return failure();
    }
    auto yieldOp = cast<scf::YieldOp>(scfForOp.getBody()->getTerminator());
    return InsertOpHelper(rewriter, yieldOp.getLoc(),
                          llvm::SmallVector<OpOperand *>{yieldOperand},
                          InsertMode::StoreOnly, blockArg);
  }
};

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<OpType>,
               InsertStoreOpBetweenVectorAndLoad<OpType>,
               InsertLoadOpBetweenStoreLikeAndVectorOrCube<OpType>>(
      patterns.getContext());
}

template <typename... OpTypes>
static void registerAll(RewritePatternSet &patterns) {
  (registerOne<OpTypes>(patterns), ...);
}

void populateInsertLoadStorePattern(RewritePatternSet &patterns) {
  registerAll<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
      >(patterns);
  registerOne<func::CallOp>(patterns);
  registerOne<hivm::IndirectLoadOp>(patterns);
  registerOne<hivm::StrideLoadOp>(patterns);
  registerOne<hivm::BitcastOp>(patterns);
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::MmadL1Op>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::StoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::IndirectStoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadOpBetweenStoreLikeAndVectorOrCube<hivm::StrideStoreOp>>(
      patterns.getContext());
  patterns.add<InsertLoadStoreOpBetweenVectorAndCube<scf::ForOp>>(
      patterns.getContext());
  patterns
      .add<InsertLoadStoreOpBetweenVectorAndCube<bufferization::ToTensorOp>>(
          patterns.getContext());
}

void InsertLoadStoreForMixCVPass::runLegacyInsertLoadStoreForMixCV() {
  OpBuilder builder(&getContext());
  auto context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  populateInsertLoadStorePattern(patterns);
  patterns.insert<InsertStoreForSCFYield>(patterns.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

LogicalResult InsertLoadStoreForMixCVPass::runPropagateOpPatterns(func::FuncOp funcOp,
                                            PropagationStep step) {
  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  patterns.add<PropagateUpPattern, PropagateDownPattern>(patterns.getContext(),
                                                         step);
  patterns.add<ResolvePropagationPattern, RemoveRedundantPropagationPattern>(
      patterns.getContext());
  rewriteConfig.fold = false;

  if (isEnabledTightCoupledBuffer()) {
    patterns.add<TightCoupledBufferResolvePropagationPattern>(patterns.getContext());
  }

  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }
  LDBG("After propagation step " << static_cast<int>(step));
  LDBG(funcOp);
  return success();
}

LogicalResult InsertLoadStoreForMixCVPass::insertPropagationOp(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  for (auto arg : funcOp.getArguments()) {
    if (isa<ShapedType>(arg.getType())) {
      Value newArg = PropagatorUtil::createPropagator(
          arg, kPropagateDownAttr, hivm::AddressSpace::GM, rewriter);
      rewriter.replaceAllUsesExcept(arg, newArg, newArg.getDefiningOp());
    }
  }

  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  patterns.add<InsertPropagationPattern>(patterns.getContext());
  patterns.add<A5InsertionPattern>(patterns.getContext());
  if (isEnabledTightCoupledBuffer()) {
    patterns.add<TightCoupledBufferInsertionPattern>(patterns.getContext());
  }
  // TODO: add case for InsertUBAfterFixpipePattern
  // case of InsertL1BeforeMmadPattern tested by compile-triton-hstu-attn-fwd.mlir
  rewriteConfig.fold = false;
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }
  return success();
}

LogicalResult InsertLoadStoreForMixCVPass::propagateAndResolve(func::FuncOp funcOp) {
  SmallVector<PropagationStep> propagations = {
      PropagationStep::LOCAL, PropagationStep::UB, PropagationStep::L1,
      PropagationStep::ALL};
  for (auto step : propagations) {
    if (failed(runPropagateOpPatterns(funcOp, step))) {
      return failure();
    }
  }
  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  patterns.add<CloneMultipleAddressSpaceOperation>(patterns.getContext());
  rewriteConfig.fold = false;
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }
  LDBG("After clone multiple address space: " << funcOp);
  if (failed(runPropagateOpPatterns(funcOp, PropagationStep::ALL))) {
    return failure();
  }
  return success();
}

// TODO: use this pattern to ensure alloc in UB for tight coupled buffer
struct EnsureAllocInUbAddressSpace
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  ~EnsureAllocInUbAddressSpace() override = default;
  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    auto oldType = dyn_cast<MemRefType>(allocOp.getMemref().getType());
    if (!oldType)
      return failure();
    auto maybeSpace = mlir::hivm::getOptionalHIVMAddressSpace(oldType);
    if (maybeSpace.has_value())
      return failure();
    if (!any_of(allocOp->getUsers(), [](Operation *op) -> bool {
      return isa<bufferization::ToTensorOp>(op);
    }))
      return failure();
    auto ubSpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
    auto newType = mlir::MemRefType::get(oldType.getShape(),
                                        oldType.getElementType(),
                                        oldType.getLayout(),
                                        ubSpaceAttr);
    if (newType == oldType)
      return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.modifyOpInPlace(allocOp,
                            [&]() { allocOp.getMemref().setType(newType); });

    rewriter.setInsertionPointAfter(allocOp);
    auto memSpaceCastOp = rewriter.create<memref::MemorySpaceCastOp>(
        allocOp.getLoc(), oldType, allocOp);
    rewriter.replaceOpUsesWithIf(allocOp, memSpaceCastOp.getResult(), [](OpOperand &opr) {
      return !isa<memref::MemorySpaceCastOp>(opr.getOwner());
    });
    return success();
  }
};

LogicalResult InsertLoadStoreForMixCVPass::ensureAllocInUbAddressSpace(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  patterns.add<EnsureAllocInUbAddressSpace>(patterns.getContext());
  rewriteConfig.fold = false;
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }
  return success();
}


namespace mlir::hivm {

constexpr static llvm::StringLiteral maybeUnCollapsibleReshape =
       "maybeUnCollapsibleReshape";

static uint64_t getElemBytesForAlign(Type t) {
  if (auto ft = dyn_cast<FloatType>(t))
    return (uint64_t)((ft.getWidth() + 7) / 8);
  if (auto it = dyn_cast<IntegerType>(t))
    return (uint64_t)((it.getWidth() + 7) / 8);
  if (isa<IndexType>(t))
    return 8ULL;
  if (auto ct = dyn_cast<ComplexType>(t))
    return 2ULL * getElemBytesForAlign(ct.getElementType());
  return 0ULL;
}

static FailureOr<uint64_t> getBlockElemsFor32BAlign(Type elemType) {
  constexpr uint64_t kAlignBytes = 32;
  uint64_t elemBytes = getElemBytesForAlign(elemType);
  if (elemBytes <= 0)
    return failure();
  if (elemBytes >= kAlignBytes)
    return 1;
  if (kAlignBytes % elemBytes != 0)
    return failure();
  return kAlignBytes / elemBytes;
}

static void ensureAllocInUbAddressSpaceIfNeeded(PatternRewriter &rewriter,
                                                memref::AllocOp allocOp) {
  auto oldType = dyn_cast<MemRefType>(allocOp.getMemref().getType());
  if (!oldType)
    return;
  auto maybeSpace = mlir::hivm::getOptionalHIVMAddressSpace(oldType);
  if (maybeSpace.has_value())
    return;

  MLIRContext *ctx = rewriter.getContext();
  auto ubSpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
  auto newType = mlir::MemRefType::get(oldType.getShape(),
                                       oldType.getElementType(),
                                       oldType.getLayout(),
                                       ubSpaceAttr);

  SmallVector<OpOperand *> originalUses;
  for (OpOperand &use : allocOp.getMemref().getUses())
    originalUses.push_back(&use);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.modifyOpInPlace(allocOp,
                           [&]() { allocOp.getMemref().setType(newType); });

  rewriter.setInsertionPointAfter(allocOp);
  Value replacement = allocOp.getMemref();
  if (replacement.getType() != oldType) {
    replacement = rewriter.create<memref::MemorySpaceCastOp>(
        allocOp.getLoc(), oldType, replacement);
  }
  for (OpOperand *use : originalUses)
    use->set(replacement);
}

/// pattern2
/// %53 = hivm.hir.vcast ins(%42 : tensor<16x16xf32>) outs(%19 : ...) -> ...
/// %54 = tensor.empty() : tensor<16x16xf32>
/// %55 = hivm.hir.mmadL1 ins(%53, %52, %true, %c16, %c16, %c16 : ...
///
/// is converted into
///
/// %53 = hivm.hir.vcast ins(%42 : tensor<16x16xf32>) outs(%19 : ...) -> ...
/// %alloc_0 = memref.alloc() : memref<..., #hivm.address_space<cbuf>>
/// %memspacecast_2 = memref.memory_space_cast %alloc_0
/// : memref<..., #hivm.address_space<cbuf>> to ...
/// %empty = bufferization.to_tensor %memspacecast_2 restrict writable : ...
/// %copy = hivm.hir.copy ins(%53 : ...) outs(%empty : ...) -> ...
/// %54 = tensor.empty() : tensor<16x16xf32>
/// %55 = hivm.hir.mmadL1 ins(%copy, %52, %true, %c16, %c16, %c16 : ...


LogicalResult insertConvertLayout(PatternRewriter &rewriter,
      hivm::MmadL1Op mmadOp,
      const llvm::SmallVector<OpOperand *> &consumerOperands) {
  if (consumerOperands.empty()) {
    return failure();
  }

  bool changed = false;
  for (OpOperand *consumerOperand : consumerOperands) {
    Value origTensor = consumerOperand->get();
    // TODO: enhance support for dynamic shape
    auto tensorType = mlir::dyn_cast<RankedTensorType>(origTensor.getType());
    if (!tensorType)
      continue;

    Operation *consumerOp = consumerOperand->getOwner();
    Location loc = consumerOp->getLoc();
    rewriter.setInsertionPoint(consumerOp);

    // TODO: Consider encapsulating it as an nd2dz function
    int64_t M = tensorType.getDimSize(0);
    int64_t N = tensorType.getDimSize(1);
    bool isA = (origTensor == mmadOp.getA());
    bool isTranspose = isA ? mmadOp.getATranspose().has_value()
                           : mmadOp.getBTranspose().has_value();
    auto blockSizes = mmadOp.getBlockSizesTile(origTensor, isTranspose, isA);
    int32_t alignM = static_cast<int32_t>(blockSizes[0]);
    int32_t alignN = static_cast<int32_t>(blockSizes[1]);
    auto elemType = tensorType.getElementType();
    int64_t newN = static_cast<int64_t>(AlignUp(static_cast<uint64_t>(N),
                            static_cast<uint64_t>(alignN)));
    if ((M != ShapedType::kDynamic && (M % alignM)) || (newN != N)) {
      int64_t newM = static_cast<int64_t>(
        AlignUp(static_cast<uint64_t>(M), static_cast<uint64_t>(alignM)));
      auto paddedType = RankedTensorType::get({newM, newN}, elemType);
      Value zeroConst = rewriter.create<arith::ConstantOp>(loc, elemType, rewriter.getZeroAttr(elemType));

      Value emptyForVbrc = rewriter.create<tensor::EmptyOp>(
              loc, paddedType.getShape(), elemType);
      auto vbrcOp = rewriter.create<hivm::VBrcOp>(
              loc, paddedType, zeroConst, emptyForVbrc);
      Value initializedMatrix = vbrcOp->getResult(0);
      SmallVector<OpFoldResult> offsets = {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(M), rewriter.getIndexAttr(N)};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
      origTensor = rewriter.create<tensor::InsertSliceOp>(
              loc, origTensor, initializedMatrix, offsets, sizes, strides);
      tensorType = mlir::cast<RankedTensorType>(origTensor.getType());
      M = newM;
      N = newN;
    }

    SmallVector<ReassociationIndices> reassociation = {{0}, {1, 2}};
    auto blkOr = getBlockElemsFor32BAlign(elemType);
    if (failed(blkOr)) {
      return consumerOp->emitOpError()
              << "unsupported element type for 32B-aligned expand_shape: "
              << elemType;
    }
    int64_t blk = (int64_t)*blkOr;

    int64_t M1 = M / alignM;
    // TODO: enhance UB alignment
    int64_t N1 = N / blk;
    auto dstTy = RankedTensorType::get({M, N1, blk}, elemType);
    auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, dstTy, origTensor, reassociation);
    auto emptyTensorType = RankedTensorType::get({N1, M, blk}, elemType);
    auto emptyTransposed = rewriter.create<tensor::EmptyOp>(
        loc, emptyTensorType.getShape(), emptyTensorType.getElementType());
    SmallVector<int64_t> premVec = {1, 0, 2};
    auto transposed = rewriter.create<hivm::VTransposeOp>(
        loc, emptyTransposed->getResultTypes(), expandOp.getResult(),
        emptyTransposed.getResult(), rewriter.getDenseI64ArrayAttr(premVec));
    auto nzTy = RankedTensorType::get({N1, M1, alignM, blk}, elemType);
    SmallVector<ReassociationIndices> nzReassoc = {{0}, {1, 2}, {3}};
    auto nzOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, nzTy, transposed->getResult(0), nzReassoc);
    rewriter.modifyOpInPlace(consumerOp,
                              [&]() { consumerOperand->set(nzOp); });
    changed = true;
  }
  return changed ? success() : failure();
}

static bool isVFCall(Operation *op) {
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    if (callOp->hasAttr(hivm::VectorFunctionAttr::name))
      return true;
  }
  return false;
}

template<typename OpType>
struct AddConvertLayoutUBToL1
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  ~AddConvertLayoutUBToL1() override = default;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
  bool changed = false;
  for (OpOperand *operand : op.getDpsInputOperands()) {
    Value beforeValue = operand->get();
    auto operandIdx = operand->getOperandNumber();
    auto inserted = getIntegersOfArrayAttr(op, insertedConvertLayout);
    if (std::find(inserted.begin(), inserted.end(), operandIdx) != inserted.end()) continue;
    auto producerOps = traceDefOps<OpType>(beforeValue);
    if (producerOps.empty())
      continue;

    bool matched = false;
    for (Operation *producer : producerOps) {
      if constexpr (std::is_same_v<OpType, mlir::scf::ForOp>) {
        auto scfForOp = llvm::cast<mlir::scf::ForOp>(producer);
        if (!scfForOp->hasAttr(ExtractLoadStoreAttr)) {
          continue;
        }
      }
      if constexpr (std::is_same_v<OpType, func::CallOp>) {
        if (!isVFCall(producer))
          continue;
      }
      if constexpr (std::is_same_v<OpType, memref::AllocOp>) {
        auto allocOp = llvm::cast<memref::AllocOp>(producer);
        auto maybeSpace = mlir::hivm::getOptionalHIVMAddressSpace(allocOp.getMemref().getType());
        if (!maybeSpace.has_value() || maybeSpace.value() != hivm::AddressSpace::UB)
          continue;
      }
      if constexpr (std::is_same_v<OpType, bufferization::ToTensorOp>) {
        auto toTensorOp = llvm::cast<bufferization::ToTensorOp>(producer);
        auto maybeAnnotateOp = utils::getAnnotateOpWithAttr(
            toTensorOp.getResult(), kMayImplicitTransposeWithLastAxis);
        if (!maybeAnnotateOp.has_value()) {
          maybeAnnotateOp = utils::getAnnotateOpWithAttr(
              toTensorOp.getMemref(), kMayImplicitTransposeWithLastAxis);
        }
        if (!maybeAnnotateOp.has_value())
          continue;
      }
      if constexpr (std::is_same_v<OpType, tensor::CollapseShapeOp>) {
        auto collapseShapeOp = llvm::cast<tensor::CollapseShapeOp>(producer);
        auto maybeAnnotateOp = utils::getAnnotateOpWithAttr(
            collapseShapeOp.getResult(), maybeUnCollapsibleReshape);
        if (!maybeAnnotateOp.has_value())
          continue;
      }
      matched = true;
      break;
    }
    if (!matched)
      continue;

    auto allocOps = traceDefOps<memref::AllocOp>(beforeValue);
    llvm::SmallVector<OpOperand *> consumerOperands{operand};
    bool isBiasOperand = (beforeValue == op.getPerChannelBias());
    auto tensorType = mlir::dyn_cast<RankedTensorType>(beforeValue.getType());
    // Only rank-2 tensors (original ND format) need to be transposed.
    // After ND2NZ conversion the tensor becomes rank-4 (NZ format), so
    // transposition is not required.
    bool needTransposed = (tensorType.getRank() == 2);
    if (isBiasOperand || !needTransposed)
      continue;
    LogicalResult result = insertConvertLayout(rewriter, op, consumerOperands);
    if (failed(result))
      continue;
    insertToIntegerAttrOfArrayAttr(op, insertedConvertLayout, operandIdx, rewriter);

    // Handle case where multiple operands of `op` are same SSA value.
    // e.g., `hivm.hir.mmadL1 ins(%vec, %vec, ...)` where both operands are the same vector.
    // In this case, we only want to rewrite the operand once, and make sure that other operand
    // is updated with the new value as well.
    Value converted = operand->get();
    rewriter.modifyOpInPlace(op, [&beforeValue, &converted, &op, &rewriter]() {
      // Use for loop to insert operand index for `insertedConvertLayout`
      for (OpOperand *operand : op.getDpsInputOperands()) {
        if(operand->get() != beforeValue)
          continue;
        op->setOperand(operand->getOperandNumber(), converted);
        insertToIntegerAttrOfArrayAttr(op, insertedConvertLayout, operand->getOperandNumber(), rewriter);
      }
    });

    for (Operation *allocOp : allocOps)
      ensureAllocInUbAddressSpaceIfNeeded(rewriter, llvm::cast<memref::AllocOp>(allocOp));
    changed = true;
  }
  return changed ? success() : failure();
}
};

template<>
struct AddConvertLayoutUBToL1<hivm::FixpipeOp>
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  ~AddConvertLayoutUBToL1() override = default;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    for (OpOperand &operand : op->getOpOperands()) {
      auto operandIdx = operand.getOperandNumber();
      auto inserted = getIntegersOfArrayAttr(op, insertedConvertLayout);
      if (std::find(inserted.begin(), inserted.end(), operandIdx) != inserted.end()) continue;
      auto maybeFixpipe = traceDefOp<hivm::FixpipeOp>(operand.get());
      if (!maybeFixpipe)
        continue;
      llvm::SmallVector<OpOperand *> consumerOperands{&operand};

      Value valueAfterUb = operand.get();

      bool isBiasOperand = (operand.get() == op.getPerChannelBias());
      if (isBiasOperand)
        continue;
      LogicalResult l1Result = insertConvertLayout(rewriter, op, consumerOperands);
      if (failed(l1Result))
        continue;
      insertToIntegerAttrOfArrayAttr(op.getOperation(), insertedConvertLayout, operand.getOperandNumber(), rewriter);

      Value convertedToL1 = operand.get();
      if (valueAfterUb != convertedToL1) {
        rewriter.modifyOpInPlace(
            op, [&]() { op->replaceUsesOfWith(valueAfterUb, convertedToL1); });
      }
      changed = true;
    }
    return changed ? success() : failure();
  }
};

std::optional<hivm::MmadL1Op> getMmadL1OpFromUpPropOp(UnrealizedConversionCastOp downPropOp,
                                   UnrealizedConversionCastOp upPropOp,
                                   PatternRewriter &rewriter) {
  // trace uppropOp, check if it's mmadop
  Operation *cur = upPropOp.getOperation();
  while(isa<UnrealizedConversionCastOp>(cur)) {
    if (!cur->hasOneUse()) // unhandled case if not one user.
      return std::nullopt;
    cur = *cur->user_begin();
  }
  return dyn_cast<hivm::MmadL1Op>(cur);
}

} // namespace mlir::hivm

template <typename ...OpTypes>
void populateAddConvertLayoutUBToL1(RewritePatternSet &patterns) {
  (patterns.add<AddConvertLayoutUBToL1<OpTypes>>(patterns.getContext()), ...);
}

LogicalResult InsertLoadStoreForMixCVPass::addConvertLayoutUBToL1(func::FuncOp funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  GreedyRewriteConfig rewriteConfig;
  populateAddConvertLayoutUBToL1<
    hivm::FixpipeOp,
    func::CallOp,
    mlir::scf::ForOp,
    hivm::IndirectLoadOp,
    hivm::StrideLoadOp,
    mlir::hivm::GatherLoadOp,
    mlir::scope::ScopeOp,
    tensor::CollapseShapeOp,
    bufferization::ToTensorOp,
    memref::AllocOp,
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
  >(patterns);
  rewriteConfig.fold = false;
  if (failed(
          applyPatternsGreedily(funcOp, std::move(patterns), rewriteConfig))) {
    return failure();
  }

  funcOp->walk([](hivm::MmadL1Op op) -> void {
    op->removeAttr(insertedConvertLayout);
  });
  return success();
}

void InsertLoadStoreForMixCVPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto *ctx = &getContext();
  auto funcOp = getOperation();

  if (enableLegacy || !isA5Target())
    return runLegacyInsertLoadStoreForMixCV();
  if (!hacc::utils::isDeviceEntry(funcOp))
    return;
  
  PassManager pm(ctx);

  if (isEnabledTightCoupledBuffer() && failed(addConvertLayoutUBToL1(funcOp))) {
    return signalPassFailure();
  }

  if (failed(pm.run(funcOp))) {
    return signalPassFailure();
  }
  if (failed(insertPropagationOp(funcOp))) {
    return signalPassFailure();
  }
  LDBG("After inserting Propagation Op: " << funcOp);
  if (failed(propagateAndResolve(funcOp))) {
    return signalPassFailure();
  }
  LDBG("After propagation & resolve: " << funcOp);
  LLVM_DEBUG({
    if (failed(verifyPropagation(funcOp))) {
      return signalPassFailure();
    }
  });

  /// Postprocess of adding core type attribute
  funcOp->walk([](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](hivm::DebugOp op) {
          auto upProp = PropagatorUtil::getUpPropagator(&op.getArgMutable());
          if (upProp) {
            auto coreType = PropagatorUtil::getCoreType(upProp);
            auto addressSpaces = PropagatorUtil::getAddressSpace(upProp);
            if (!addressSpaces.empty()) {
              auto memScopeAttr = hivm::AddressSpaceAttr::get(
                  op.getContext(), addressSpaces[0]);
              op.setMemscopeAttr(memScopeAttr);
            }
            if (coreType != TCoreType::CUBE_AND_VECTOR) {
              op.setTcoretypeAttr(
                  TCoreTypeAttr::get(op.getContext(), coreType));
            } else {
              auto addressSpace = addressSpaces.empty() ? hivm::AddressSpace::UB
                                                        : addressSpaces[0];
              op.setTcoretypeAttr(TCoreTypeAttr::get(
                  op.getContext(),
                  PropagatorUtil::kAddressSpace2CoreType.at(addressSpace)));
            }
          }
        })
        .Case([&](hivm::LoadOp op) {
          auto upProp = PropagatorUtil::getUpPropagator(&op.getDstMutable());
          if (upProp) {
            auto coreType = PropagatorUtil::getCoreType(upProp);
            if (coreType != TCoreType::CUBE_AND_VECTOR) {
              op.setTcoretypeAttr(
                  TCoreTypeAttr::get(op.getContext(), coreType));
            } else {
              auto addressSpaces = PropagatorUtil::getAddressSpace(upProp);
              auto addressSpace = addressSpaces.empty() ? hivm::AddressSpace::UB
                                                        : addressSpaces[0];
              op.setTcoretypeAttr(TCoreTypeAttr::get(
                  op.getContext(),
                  PropagatorUtil::kAddressSpace2CoreType.at(addressSpace)));
            }
          }
        });
  });

  PassManager pm2(ctx);
  pm2.addPass(createCanonicalizerPass());
  if (failed(pm2.run(funcOp))) {
    return signalPassFailure();
  }

  PassManager pm3(ctx);
  pm3.addPass(createInsertLoadStoreForScalarPass());
  if (failed(pm3.run(funcOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInsertLoadStoreForMixCVPass(
    const InsertLoadStoreForMixCVOptions &options) {
  return std::make_unique<InsertLoadStoreForMixCVPass>(options);
}
