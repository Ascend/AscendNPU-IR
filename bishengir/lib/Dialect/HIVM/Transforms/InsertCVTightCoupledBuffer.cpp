//===-------------------- InsertCVTightCoupledBuffer.cpp-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts cv tight coupled buffer for mix cv function.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/Passes.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "insert-cv-tight-coupled-buffer"

namespace mlir {
#define GEN_PASS_DEF_INSERTCVTIGHTCOUPLEDBUFFER
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

template <template <typename> class PatternT, typename... OpTypes>
static void registerAll(mlir::RewritePatternSet &patterns) {
  (patterns.add<PatternT<OpTypes>>(patterns.getContext()), ...);
}

template <template <typename> class PatternT>
static void registerAllVectorOp(mlir::RewritePatternSet &patterns) {
  registerAll<PatternT,
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"
              >(patterns);
}

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct InsertCVTightCoupledBufferPass
    : public impl::InsertCVTightCoupledBufferBase<
          InsertCVTightCoupledBufferPass> {
  using Base::Base;
  void runOnOperation() override;
};

enum class InsertMode { MoveToUb = 0, MoveToL1 };

template <InsertMode Mode>
LogicalResult InsertOpHelper(PatternRewriter &,
                             const llvm::SmallVector<OpOperand *> &);

/// pattern1
/// fixpipe has vector or vf users
/// %21 = hivm.hir.mmadL1 {b_transpose} ins ...
/// %23 = hivm.hir.fixpipe {enable_nz2nd} ins(...) outs(...) -> ...
/// %24 = tensor.empty() : tensor<16x16xi32>
/// %25 = hivm.hir.bitcast %23 : tensor<16x16xf32> -> tensor<16x16xi32>
///
/// is converted into
///
/// %21 = hivm.hir.mmadL1 {b_transpose} ins(...) outs(...) -> tensor<...>
/// %alloc = memref.alloc : memref<..., #hivm.address_space<ub>>
/// %no_ub = memref.memory_space_cast %alloc:
///  memref<..., #hivm.address_space<ub>> to memref<...>
/// hivm.hir.fixpipe {enable_nz2nd} ins(%21 : ...) outs(%alloc : memref<...)
/// %to_tensor = bufferization.to_tensor %no_ub restrict writable :
/// memref<16x16xf32> %24 = tensor.empty() : tensor<16x16xi32> %25 =
/// hivm.hir.bitcast %to_tensor : tensor<16x16xf32> -> tensor<16x16xi32>
template <>
LogicalResult InsertOpHelper<InsertMode::MoveToUb>(
    PatternRewriter &rewriter,
    const llvm::SmallVector<OpOperand *> &consumerOperands) {
  if (consumerOperands.empty()) {
    return failure();
  }
  llvm::DenseSet<Operation *> processed;
  bool changed = false;

  for (OpOperand *consumerOperand : consumerOperands) {
    Value usedVal = consumerOperand->get();
    auto fixpipes = traceDefOps<hivm::FixpipeOp>(usedVal);
    if (fixpipes.empty())
      continue;
    for (Operation *fixpipe : fixpipes) {
      auto fixpipeOp = llvm::cast<hivm::FixpipeOp>(fixpipe);
      // The same FixpipeOp can feed multiple vector/vf users. Use a set
      // to make sure we rewrite each FixpipeOp at most once.
      if (!processed.insert(fixpipe).second)
        continue;

      auto resultTensorType =
          mlir::dyn_cast<RankedTensorType>(fixpipeOp.getResult(0).getType());
      if (!resultTensorType)
        continue;

      auto elemType = resultTensorType.getElementType();
      auto shape = resultTensorType.getShape();
      MLIRContext *ctx = rewriter.getContext();
      auto ubSpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::UB);
      auto ubMemrefType =
          mlir::MemRefType::get(shape, elemType, /*layout=*/nullptr, ubSpaceAttr);
      auto noUbMemrefType = mlir::MemRefType::get(shape, elemType);
      rewriter.setInsertionPoint(fixpipe);
      Location fixLoc = fixpipeOp.getLoc();
  
      Value alloc;
      bool hasDynamicShape = llvm::any_of(
        shape, [](int64_t dim) { return dim == ShapedType::kDynamic; });
      // dynamic shape
      if (hasDynamicShape) {
        Value sourceVal = fixpipeOp.getSrc();
        auto maybeMmad = traceDefOp<hivm::MmadL1Op>(sourceVal);

        if (maybeMmad.has_value()) {
          auto mmadOp = cast<hivm::MmadL1Op>(maybeMmad.value());
          auto mmadResult = mmadOp.getResult(0);
          auto mmadType = cast<ShapedType>(mmadResult.getType());
          auto mmadShape = mmadType.getShape();
          auto mmadElemType = mmadType.getElementType();

          Value emptyTensor = fixpipeOp.getOperand(1);
          auto emptyOp = emptyTensor.getDefiningOp<tensor::EmptyOp>();
          SmallVector<Value> dynamicDims;

          if (emptyOp) {
            dynamicDims.append(emptyOp.getDynamicSizes().begin(),
                              emptyOp.getDynamicSizes().end());
          }
          alloc = createAllocWithMark(rewriter, fixLoc, ubMemrefType,
                                      dynamicDims, mmadShape, mmadElemType);
        } else {
          llvm_unreachable(
            "Unable to trace to MmadL1 op for dynamic shape fixpipe\n");
        }
      } else {
        alloc = rewriter.create<memref::AllocOp>(fixLoc, ubMemrefType);
      }
      Value noUb = rewriter.create<memref::MemorySpaceCastOp>(
          fixLoc, noUbMemrefType, alloc);
      SmallVector<Value> oprs({fixpipeOp.getSrc(), alloc});
      if (auto quantScale = fixpipeOp.getQuantScale())
        oprs.push_back(quantScale);
      auto newFixpipeOp = rewriter.create<hivm::FixpipeOp>(
          fixLoc, TypeRange{}, oprs, fixpipeOp->getAttrs());
      rewriter.setInsertionPointAfter(newFixpipeOp);
      auto toTensor = rewriter.create<bufferization::ToTensorOp>(
          fixLoc, resultTensorType, noUb,
          /*restrict=*/true,
          /*writable=*/true);
      rewriter.replaceOp(fixpipeOp, toTensor.getResult());
      changed = true;
    }
  }
  return changed ? success() : failure();
}

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

template <>
LogicalResult InsertOpHelper<InsertMode::MoveToL1>(
    PatternRewriter &rewriter,
    const llvm::SmallVector<OpOperand *> &consumerOperands) {
  if (consumerOperands.empty()) {
    return failure();
  }

  MLIRContext *ctx = rewriter.getContext();
  bool changed = false;

  for (OpOperand *consumerOperand : consumerOperands) {
    Value origTensor = consumerOperand->get();
    // TODO: enhance support for dynamic shape
    auto tensorType = origTensor.getType().dyn_cast<RankedTensorType>();
    if (!tensorType)
      continue;

    Operation *consumerOp = consumerOperand->getOwner();
    Location loc = consumerOp->getLoc();
    rewriter.setInsertionPoint(consumerOp);
    // TODO: Consider encapsulating it as an nd2dz function
    int64_t M = tensorType.getDimSize(0);
    int64_t N = tensorType.getDimSize(1);
    int32_t alignM = 16;
    int32_t alignN = 32;
    auto elemType = tensorType.getElementType();

    int64_t elemTypeSize = (int64_t)getElemBytesForAlign(elemType);
    int64_t newN = (((elemTypeSize * N + (alignN - 1)) / alignN) * alignN) / elemTypeSize;
    if ((M != ShapedType::kDynamic && (M % alignM)) || (newN != N)) {
      int64_t newM = ((M + alignM - 1) / alignM) * alignM;
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
      tensorType = origTensor.getType().cast<RankedTensorType>();
      M = newM;
      N = newN;
    }

    SmallVector<ReassociationIndices> reassociation = {{0}, {1,2}};
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
    if (N1 > 1) {
      auto markOp = rewriter.create<annotation::MarkOp>(loc, expandOp);
      auto tilingDimAttr = rewriter.getDictionaryAttr(SmallVector<NamedAttribute>{
          NamedAttribute(rewriter.getStringAttr("1"), rewriter.getIndexAttr(1))});
      markOp->setAttr(kTilingDimMappingAttrName, tilingDimAttr);
    }
    auto emptyTensorType = RankedTensorType::get({N1, M, blk}, elemType);
    auto emptyTransposed = rewriter.create<tensor::EmptyOp>(
        loc, emptyTensorType.getShape(), emptyTensorType.getElementType());
    SmallVector<int64_t> premVec = {1, 0, 2};
    auto transposed = rewriter.create<hivm::VTransposeOp>(
        loc, emptyTransposed->getResultTypes(), expandOp.getResult(),
        emptyTransposed.getResult(), rewriter.getDenseI64ArrayAttr(premVec));
    auto nzTy = RankedTensorType::get({N1, M1, 16, blk}, elemType);
    SmallVector<ReassociationIndices> nzReassoc = {{0}, {1, 2}, {3}};
    auto nzOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, nzTy, transposed->getResult(0), nzReassoc);
    if (M1 > 1) {
      auto markOp = rewriter.create<annotation::MarkOp>(loc, nzOp);
      auto tilingDimAttr = rewriter.getDictionaryAttr(SmallVector<NamedAttribute>{
          NamedAttribute(rewriter.getStringAttr("1"), rewriter.getIndexAttr(1))});
      markOp->setAttr(kTilingDimMappingAttrName, tilingDimAttr);
    }
    auto l1SpaceAttr = hivm::AddressSpaceAttr::get(ctx, hivm::AddressSpace::L1);
    auto l1MemrefType = mlir::MemRefType::get(nzTy.getShape(),
                                              nzTy.getElementType(),
                                              nullptr, l1SpaceAttr);
    auto plainMemrefType = mlir::MemRefType::get(
        nzTy.getShape(), nzTy.getElementType());
    Value alloc = rewriter.create<memref::AllocOp>(loc, l1MemrefType);
    Value memspacecast =
        rewriter.create<memref::MemorySpaceCastOp>(loc, plainMemrefType, alloc);
    auto emptyTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, nzTy, memspacecast,
        /*restrict=*/true,
        /*writable=*/true);
    Value src = nzOp.getResult();
    Value dst = emptyTensor.getResult();
    rewriter.create<hivm::CopyOp>(loc,
                                  /*resultType=*/TypeRange(),
                                  /*src=*/src,
                                  /*dst=*/memspacecast);
    rewriter.modifyOpInPlace(consumerOp, [&]() { consumerOperand->set(dst); });
    changed = true;
  }
  return changed ? success() : failure();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// InsertMoveUb
//===----------------------------------------------------------------------===//
/// pattern1 : fixpipe op + vector/vf
/// convert into memref.alloc + memref.memory_space_cast + fixpipe op +
/// bufferization.to_tensor + vector/vf
///
/// pattern2 : fixpipe op + tensor.extract
/// convert into memref.alloc + memref.memory_space_cast + fixpipe op +
/// bufferization.to_tensor + tensor.extract
///
/// Extra:
/// If a vector input which match the pattern can be traced to a allocOp,
/// and do InsertOpHelper success, the alloc will add hivm.address_space<ub>
/// to ensure the buffer will be allocated only in UB.
template <typename OpType>
struct InsertMoveUbBetweenFixpipeAndVector : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    Operation *rawOp = op.getOperation();
    bool isStructured = isa<hivm::HIVMStructuredOp>(rawOp);
    bool isVF = isVFCall(rawOp);
    bool isExtract = false;
    if (isa<tensor::ExtractOp>(rawOp)) {
      if (rawOp->hasAttr(
              "DuplicateTensorExtractForCube::newExtractLabel")) {
        return failure();
      }
      isExtract = true;
    }
    if (!isStructured && !isVF && !isExtract) {
      return failure();
    }

    bool changed = false;
    for (OpOperand &operand : op->getOpOperands()) {
      if (traceDefOps<hivm::FixpipeOp>(operand.get()).empty())
        continue;

      auto allocOps = traceDefOps<memref::AllocOp>(operand.get());
      llvm::SmallVector<OpOperand *> consumerOperands{&operand};
      LogicalResult result = InsertOpHelper<InsertMode::MoveToUb>(rewriter, consumerOperands);
      if (failed(result))
        continue;

      for (Operation *allocOp : allocOps)
        ensureAllocInUbAddressSpaceIfNeeded(rewriter, llvm::cast<memref::AllocOp>(allocOp));
      changed = true;
    }
    return changed ? success() : failure();
  }
};

//===----------------------------------------------------------------------===//
// InsertMoveL1
//===----------------------------------------------------------------------===//
/// pattern2 : vector/vf(dst) + cube(dst)
/// convert into vector +  memref.alloc + memory_space_cast +
/// bufferization.to_tensor + hivm.hir.copy +cube
///
/// Extra:
/// Like pattern1, if a mmadL1 input which match the pattern can be traced to
/// a allocOp, and do InsertOpHelper success, the alloc will add
/// hivm.address_space<ub> to ensure the buffer will be allocated only in UB.
template <typename OpType>
struct InsertMoveL1BetweenVectorAndCube
    : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  virtual ~InsertMoveL1BetweenVectorAndCube() override = default;
  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    for (OpOperand &operand : op->getOpOperands()) {
      auto producerOps = traceDefOps<OpType>(operand.get());
      if (producerOps.empty())
        continue;

      bool matched = false;
      for (Operation *producer : producerOps) {
        if constexpr (std::is_same_v<OpType, mlir::scf::ForOp>) {
          auto scfForOp = llvm::cast<mlir::scf::ForOp>(producer);
          if (!scfForOp->hasAttr("ExtractedLoadOrStore")) {
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
        matched = true;
        break;
      }
      if (!matched)
        continue;

      Value beforeValue = operand.get();
      auto allocOps = traceDefOps<memref::AllocOp>(beforeValue);
      llvm::SmallVector<OpOperand *> consumerOperands{&operand};
      LogicalResult result = InsertOpHelper<InsertMode::MoveToL1>(rewriter, consumerOperands);
      if (failed(result))
        continue;

      // Handle the case where multiple operands of `op` are the same SSA value.
      // e.g., `hivm.hir.mmadL1 ins(%vec, %vec, ...)` where both operands are the same vector.
      // In this case, we only want to rewrite the operand once, and make sure the other operand
      // is updated with the new value as well.
      Value converted = operand.get();
      rewriter.modifyOpInPlace(op, [&]() {
        op->replaceUsesOfWith(beforeValue, converted);
      });

      for (Operation *allocOp : allocOps)
        ensureAllocInUbAddressSpaceIfNeeded(rewriter, llvm::cast<memref::AllocOp>(allocOp));
      changed = true;
    }
    return changed ? success() : failure();
  }
};

struct InsertDataMovementFixpipeToL1 : public OpRewritePattern<hivm::MmadL1Op> {
  using OpRewritePattern<hivm::MmadL1Op>::OpRewritePattern;
  virtual ~InsertDataMovementFixpipeToL1() override = default;

  LogicalResult matchAndRewrite(hivm::MmadL1Op op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    for (OpOperand &operand : op->getOpOperands()) {
      auto maybeFixpipe = traceDefOp<hivm::FixpipeOp>(operand.get());
      if (!maybeFixpipe)
        continue;
      auto fixpipeOp = llvm::cast<hivm::FixpipeOp>(maybeFixpipe.value());

      llvm::SmallVector<OpOperand *> consumerOperands{&operand};

      LogicalResult ubResult =
          InsertOpHelper<InsertMode::MoveToUb>(rewriter, consumerOperands);
      if (failed(ubResult))
        continue;

      Value valueAfterUb = operand.get(); 

      LogicalResult l1Result =
          InsertOpHelper<InsertMode::MoveToL1>(rewriter, consumerOperands);
      if (failed(l1Result))
        continue;

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

template <typename OpType>
static void registerOne(RewritePatternSet &patterns) {
  patterns.add<InsertMoveUbBetweenFixpipeAndVector<OpType>,
               InsertMoveL1BetweenVectorAndCube<OpType>>(patterns.getContext());
}

void populateInsertCVTightCoupledBufferPattern(RewritePatternSet &patterns) {
  registerAllVectorOp<InsertMoveUbBetweenFixpipeAndVector>(patterns);
  registerAllVectorOp<InsertMoveL1BetweenVectorAndCube>(patterns);
  registerOne<func::CallOp>(patterns);
  registerOne<mlir::scf::ForOp>(patterns);

  // Treat UB alloc as CV connection point for MoveToL1
  patterns.add<InsertMoveL1BetweenVectorAndCube<memref::AllocOp>>(patterns.getContext());
  patterns.add<InsertDataMovementFixpipeToL1>(patterns.getContext());
  patterns.add<InsertMoveUbBetweenFixpipeAndVector<hivm::StoreOp>>(patterns.getContext());
  patterns.add<InsertMoveUbBetweenFixpipeAndVector<tensor::ExtractOp>>(patterns.getContext());
}

void InsertCVTightCoupledBufferPass::runOnOperation() {
  OpBuilder builder(&getContext());
  auto context = &getContext();
  auto funcOp = getOperation();
  RewritePatternSet patterns(context);
  populateInsertCVTightCoupledBufferPattern(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::hivm::createInsertCVTightCoupledBufferPass() {
  return std::make_unique<InsertCVTightCoupledBufferPass>();
}
