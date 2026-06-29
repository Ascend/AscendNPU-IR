//===- NormalizeAtomicTemplate.h -------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEATOMICTEMPLATE_H
#define BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEATOMICTEMPLATE_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include <optional>

namespace mlir {

/// Keeps the decomposed atomic read-modify-write serialized:
///   lock(); old = GM; new = f(old, UB); GM = new; unlock().
struct SyncBlockLockGuard {
  hivm::CreateSyncBlockLockOp createdLock;
  OpBuilder &builder;
  OpBuilder::InsertionGuard guard;

  SyncBlockLockGuard(OpBuilder &builder, Location loc)
      : builder(builder), guard(builder) {
    Type memrefi64 = MemRefType::get({1}, builder.getI64Type());
    createdLock =
        builder.create<hivm::CreateSyncBlockLockOp>(loc, memrefi64, Value());
    builder.create<hivm::SyncBlockLockOp>(loc, createdLock.getResult());
  }

  ~SyncBlockLockGuard() {
    builder.create<hivm::SyncBlockUnlockOp>(createdLock.getLoc(), createdLock);
  }
};

/// Static backing allocation for a GM destination that may be a dynamic subview.
///
/// `sBuffer` is used for tensor conversion, while `dBuffer` has the same
/// logical shape as the original GM memref:
///   static GM:        dBuffer == sBuffer
///   dynamic subview:  dBuffer = subview(sBuffer, sizes/strides from GM)
struct StaticSizedBuffer {
  TypedValue<MemRefType> dBuffer;
  TypedValue<MemRefType> sBuffer;

  StaticSizedBuffer(OpBuilder &builder, Location loc,
                    TypedValue<MemRefType> gmMemrefSource)
      : builder(builder), loc(loc), gmMemrefSource(gmMemrefSource) {
    const Type elemType = gmMemrefSource.getType().getElementType();
    const MemRefType rawMemref =
        MemRefType::get(gmMemrefSource.getType().getShape(), elemType);
    if (gmMemrefSource.getType().hasStaticShape()) {
      dBuffer = sBuffer = builder.create<memref::AllocOp>(loc, rawMemref);
      return;
    }

    definingOp = gmMemrefSource.getDefiningOp<memref::SubViewOp>();
    if (!definingOp || !definingOp.getSourceType().hasStaticShape()) {
      valid = false;
      return;
    }
    sBuffer = builder.create<memref::AllocOp>(
        loc, rawMemref.clone(definingOp.getSourceType().getShape()));
    SmallVector<OpFoldResult> zeroOffsets(gmMemrefSource.getType().getRank(),
                                          builder.getI64IntegerAttr(0));
    dBuffer = builder.create<memref::SubViewOp>(loc, sBuffer, zeroOffsets,
                                                definingOp.getMixedSizes(),
                                                definingOp.getMixedStrides());
  }

  bool isValid() const { return valid; }

  void copyMemrefIntoBuffer(TypedValue<MemRefType> memref) {
    builder.create<memref::CopyOp>(loc, memref, dBuffer);
  }

  /// Stages a memref or tensor atomic input into the static-sized buffer.
  ///
  /// Atomic ops are expected to have identical `ins` and `outs` shapes. The
  /// staged tile is materialized into `dBuffer`, while compute uses the full
  /// static `sBuffer` view via `toStaticTensor()`.
  FailureOr<TypedValue<TensorType>> stageInput(Value input) {
    if (auto memref = dyn_cast<TypedValue<MemRefType>>(input)) {
      copyMemrefIntoBuffer(memref);
      return toStaticTensor();
    }
    auto tensor = dyn_cast<TypedValue<TensorType>>(input);
    if (!tensor)
      return failure();

    builder.create<bufferization::MaterializeInDestinationOp>(
        loc, Type(), tensor, dBuffer, /*restrict=*/false, /*writable=*/true);
    return toStaticTensor();
  }

  /// Tensor view of the full static staging allocation.
  TypedValue<TensorType> toStaticTensor() {
    return builder
        .create<bufferization::ToTensorOp>(
            loc, sBuffer, /*restrict=*/true, /*writable=*/true)
        .getResult();
  }

  /// Tensor view matching the GM destination shape (`ins` == `outs`).
  TypedValue<TensorType> toLogicalTensor() {
    return builder
        .create<bufferization::ToTensorOp>(
            loc, dBuffer, /*restrict=*/true, /*writable=*/true)
        .getResult();
  }

  void storeBack(TypedValue<TensorType> tensorSrc) {
    if (dBuffer.getType().hasStaticShape()) {
      builder.create<bufferization::MaterializeInDestinationOp>(
          loc, Type(), tensorSrc, gmMemrefSource, /*restrict=*/false,
          /*writable=*/true);
      return;
    }
    SmallVector<OpFoldResult> zeroOffsets(gmMemrefSource.getType().getRank(),
                                          builder.getI64IntegerAttr(0));
    Value dtensor = builder.create<tensor::ExtractSliceOp>(
        loc, tensorSrc, zeroOffsets, definingOp.getMixedSizes(),
        definingOp.getMixedStrides());
    builder.create<bufferization::MaterializeInDestinationOp>(
        loc, Type(), dtensor, gmMemrefSource, /*restrict=*/false,
        /*writable=*/true);
  }

private:
  OpBuilder &builder;
  const Location loc;

  const TypedValue<MemRefType> gmMemrefSource;
  memref::SubViewOp definingOp;
  bool valid = true;
};

/// If `input` is a `bufferization.to_memref`, return its source tensor so
/// decomposition can stage UB via `materialize_in_destination` instead of
/// `memref.copy`. The `to_memref` is erased after the atomic op is removed.
inline Value resolveAtomicInput(Value input) {
  if (auto toMemref = input.getDefiningOp<bufferization::ToMemrefOp>())
    return toMemref.getTensor();
  return input;
}

inline void eraseToMemrefIfUnused(Value input, PatternRewriter &rewriter) {
  if (auto toMemref = input.getDefiningOp<bufferization::ToMemrefOp>())
    if (toMemref->use_empty())
      rewriter.eraseOp(toMemref);
}

/// FP8 atomic arithmetic is evaluated in FP32, then cast back before storing.
inline bool isAtomicElementFp8(ShapedType st) {
  auto t = dyn_cast<FloatType>(st.getElementType());
  return t && t.getWidth() == 8;
}

template <typename Traits>
inline Value castAtomicOpToF32IfFp8(PatternRewriter &rewriter, Location loc,
                                    Value sourceTensor, ShapedType originalType,
                                    bool isForward) {
  if (!isAtomicElementFp8(originalType))
    return sourceTensor;

  Type srcType = originalType.getElementType();
  Type targetType = rewriter.getF32Type();
  return Traits::createCastOp(rewriter, loc, sourceTensor,
                              isForward ? targetType : srcType,
                              CastRoundKind::RInt);
}

template <typename AtomicKindTy>
inline std::optional<BinaryKind>
lookupAtomicKindToBinary(const llvm::DenseMap<AtomicKindTy, BinaryKind> &map,
                         AtomicKindTy kind) {
  auto it = map.find(kind);
  if (it == map.end())
    return std::nullopt;
  return it->second;
}

template <typename AtomicKindTy>
inline std::optional<BinaryKind>
mapAlwaysSoftwareAtomicKindToBinary(AtomicKindTy kind) {
  static const llvm::DenseMap<AtomicKindTy, BinaryKind> kindToBinary = {
      {AtomicKindTy::AND, BinaryKind::And},
      {AtomicKindTy::OR, BinaryKind::Or},
      {AtomicKindTy::XOR, BinaryKind::Xor},
      {AtomicKindTy::UMAX, BinaryKind::MaxUnsigned},
      {AtomicKindTy::UMIN, BinaryKind::MinUnsigned},
  };

  return lookupAtomicKindToBinary(kindToBinary, kind);
}

template <typename AtomicKindTy>
inline std::optional<BinaryKind>
mapNativeUnsupportedAtomicKindToBinary(AtomicKindTy kind) {
  static const llvm::DenseMap<AtomicKindTy, BinaryKind> kindToBinary = {
      {AtomicKindTy::ADD, BinaryKind::Add},
      {AtomicKindTy::MAX, BinaryKind::MaxSigned},
      {AtomicKindTy::MIN, BinaryKind::MinSigned},
  };

  return lookupAtomicKindToBinary(kindToBinary, kind);
}

/// Rewrites an atomic store with an elementwise update:
///   GM = atomic_kind(GM, UB)
///
/// For FP8, the binary is evaluated as:
///   result_fp8 = cast_fp8(binary(cast_f32(GM), cast_f32(UB))).
template <typename StoreOpTy, typename Traits>
struct NormalizeAtomicStoreElemwise : public OpRewritePattern<StoreOpTy> {
  using OpRewritePattern<StoreOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOpTy op,
                                PatternRewriter &rewriter) const override {
    if (!Traits::shouldDecomposeStore(op))
      return failure();

    Location loc = op->getLoc();
    rewriter.setInsertionPointAfter(op);

    if (op.getDpsInputs().empty() || op.getDpsInits().empty())
      return rewriter.notifyMatchFailure(op, "expected store input and output");

    Value ubInput = op.getDpsInputs().front();
    Value resolvedUbInput = resolveAtomicInput(ubInput);
    TypedValue<MemRefType> gmMemref =
        dyn_cast<TypedValue<MemRefType>>(op.getDpsInits().front());
    if (!gmMemref)
      return rewriter.notifyMatchFailure(op, "expected memref store output");

    StaticSizedBuffer rhsBuffer(rewriter, loc, gmMemref);
    if (!rhsBuffer.isValid())
      return rewriter.notifyMatchFailure(
          op, "expected dynamic GM memref to be a static subview");
    FailureOr<TypedValue<TensorType>> maybeRhsTensor =
        rhsBuffer.stageInput(resolvedUbInput);
    if (failed(maybeRhsTensor))
      return rewriter.notifyMatchFailure(op, "expected memref or tensor input");

    Value rhsTensor = castAtomicOpToF32IfFp8<Traits>(
        rewriter, loc, *maybeRhsTensor, gmMemref.getType(), /*isForward=*/true);
    StaticSizedBuffer lhsBuffer(rewriter, loc, gmMemref);
    if (!lhsBuffer.isValid())
      return rewriter.notifyMatchFailure(
          op, "expected dynamic GM memref to be a static subview");

    // The staged UB value is immutable; only the GM read-modify-write needs the
    // critical section.
    SyncBlockLockGuard lock(rewriter, loc);
    rewriter.create<memref::CopyOp>(loc, gmMemref, lhsBuffer.dBuffer);
    Value lhsTensor =
        castAtomicOpToF32IfFp8<Traits>(rewriter, loc, lhsBuffer.toStaticTensor(),
                             gmMemref.getType(), /*isForward=*/true);
    Value binOpResultBuffer = rewriter.create<tensor::EmptyOp>(
        loc, lhsTensor.getType(), ValueRange({}));
    FailureOr<Value> maybeResult = Traits::createStoreBinary(
        rewriter, loc, op, lhsTensor, rhsTensor, binOpResultBuffer);
    if (failed(maybeResult))
      return failure();

    Value result = castAtomicOpToF32IfFp8<Traits>(rewriter, loc, *maybeResult,
                                        gmMemref.getType(),
                                        /*isForward=*/false);
    auto resultTensor = dyn_cast<TypedValue<TensorType>>(result);
    if (!resultTensor)
      return rewriter.notifyMatchFailure(op, "expected tensor result");
    lhsBuffer.storeBack(resultTensor);
    rewriter.eraseOp(op);
    eraseToMemrefIfUnused(ubInput, rewriter);
    return success();
  }
};

/// Rewrites compare-and-swap:
///   GM = (GM == compare) ? store : GM
///
/// FP8 equality is evaluated after widening both sides to FP32.
template <typename AtomicCasOpTy, typename Traits>
struct NormalizeAtomicCASTemplate : public OpRewritePattern<AtomicCasOpTy> {
  using OpRewritePattern<AtomicCasOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicCasOpTy op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    Location loc = op->getLoc();
    auto srcOperands = op.getODSOperands(0);
    auto dstOperands = op.getODSOperands(1);
    if (srcOperands.size() < 2 || dstOperands.empty())
      return rewriter.notifyMatchFailure(op, "expected CAS source and output");

    TypedValue<MemRefType> ubComparing = dyn_cast<TypedValue<MemRefType>>(srcOperands[0]);
    TypedValue<MemRefType> ubStoring = dyn_cast<TypedValue<MemRefType>>(srcOperands[1]);
    TypedValue<MemRefType> gmMemref =
        dyn_cast<TypedValue<MemRefType>>(dstOperands.front());
    if (!gmMemref)
      return rewriter.notifyMatchFailure(op, "expected memref CAS output");

    StaticSizedBuffer comparingBuffer(rewriter, loc, gmMemref);
    StaticSizedBuffer storingBuffer(rewriter, loc, gmMemref);
    StaticSizedBuffer gmValBuffer(rewriter, loc, gmMemref);
    if (!comparingBuffer.isValid() || !storingBuffer.isValid() ||
        !gmValBuffer.isValid())
      return rewriter.notifyMatchFailure(
          op, "expected dynamic GM memref to be a static subview");

    FailureOr<TypedValue<TensorType>> maybeComparingTensor =
        comparingBuffer.stageInput(resolveAtomicInput(srcOperands[0]));
    FailureOr<TypedValue<TensorType>> maybeStoringTensor =
        storingBuffer.stageInput(resolveAtomicInput(srcOperands[1]));
    if (failed(maybeComparingTensor) || failed(maybeStoringTensor))
      return rewriter.notifyMatchFailure(op,
                                         "expected memref or tensor CAS inputs");

    SyncBlockLockGuard lock(rewriter, loc);
    rewriter.create<memref::CopyOp>(loc, gmMemref, gmValBuffer.dBuffer);
    const TypedValue<TensorType> gmValTensor = gmValBuffer.toStaticTensor();

    Value cmpResult = Traits::createCmpOp(
        rewriter, loc,
        castAtomicOpToF32IfFp8<Traits>(rewriter, loc, gmValTensor, gmValTensor.getType(),
                             /*isForward=*/true),
        castAtomicOpToF32IfFp8<Traits>(rewriter, loc, *maybeComparingTensor,
                             gmValTensor.getType(), /*isForward=*/true),
        CompareKind::EQ);
    Value selectedResult =
        Traits::createSelectOp(rewriter, loc, cmpResult, *maybeStoringTensor,
                               gmValTensor, gmValTensor);
    auto selectedResultTensor =
        dyn_cast<TypedValue<TensorType>>(selectedResult);
    if (!selectedResultTensor)
      return rewriter.notifyMatchFailure(op, "expected tensor select result");
    gmValBuffer.storeBack(selectedResultTensor);
    rewriter.eraseOp(op);
    eraseToMemrefIfUnused(srcOperands[0], rewriter);
    eraseToMemrefIfUnused(srcOperands[1], rewriter);
    return success();
  }
};

/// Rewrites atomic exchange:
///   old = GM; GM = UB; UB = old
///
/// The lock covers all three copies so no intermediate exchange state is
/// observable.
template <typename AtomicXchgOpTy, typename Traits>
struct NormalizeAtomicXCHGTemplate : public OpRewritePattern<AtomicXchgOpTy> {
  using OpRewritePattern<AtomicXchgOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(AtomicXchgOpTy op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    Location loc = op->getLoc();
    auto srcOperands = op.getODSOperands(0);
    auto dstOperands = op.getODSOperands(1);
    if (srcOperands.empty() || dstOperands.empty())
      return rewriter.notifyMatchFailure(op, "expected XCHG input and output");

    Value ubInput = resolveAtomicInput(srcOperands.front());
    TypedValue<MemRefType> ubMemref = dyn_cast<TypedValue<MemRefType>>(ubInput);
    TypedValue<TensorType> ubTensor = dyn_cast<TypedValue<TensorType>>(ubInput);
    TypedValue<MemRefType> gmMemref =
        dyn_cast<TypedValue<MemRefType>>(dstOperands.front());
    if ((!ubMemref && !ubTensor) || !gmMemref)
      return rewriter.notifyMatchFailure(op,
                                         "expected memref or tensor XCHG input "
                                         "and memref output");

    const bool hasReturn = op->getNumResults() > 0;
    scope::ScopeOp scopeOp;
    if (hasReturn)
      scopeOp = rewriter.create<scope::ScopeOp>(loc, op->getResultTypes());
    else
      scopeOp = rewriter.create<scope::ScopeOp>(loc, TypeRange{});
    scopeOp->setAttr(
        hivm::TCoreTypeAttr::name,
        hivm::TCoreTypeAttr::get(rewriter.getContext(), hivm::TCoreType::VECTOR));
    rewriter.createBlock(&scopeOp.getRegion());
    OpBuilder::InsertionGuard scopeGuard(rewriter);
    rewriter.setInsertionPointToStart(scopeOp.getBody());

    StaticSizedBuffer gmBuffer(rewriter, loc, gmMemref);
    StaticSizedBuffer tmpBuffer(rewriter, loc, gmMemref);
    if (!gmBuffer.isValid() || !tmpBuffer.isValid())
      return rewriter.notifyMatchFailure(
          op, "expected dynamic GM memref to be a static subview");

    Type memrefI64 = MemRefType::get({1}, rewriter.getI64Type());
    auto createdLock =
        rewriter.create<hivm::CreateSyncBlockLockOp>(loc, memrefI64, Value());
    rewriter.create<hivm::SyncBlockLockOp>(loc, createdLock.getResult());
    rewriter.create<memref::CopyOp>(loc, gmMemref, tmpBuffer.dBuffer);
    if (ubMemref) {
      rewriter.create<memref::CopyOp>(loc, ubMemref, gmMemref);
      rewriter.create<memref::CopyOp>(loc, tmpBuffer.dBuffer, ubMemref);
    } else {
      gmBuffer.storeBack(ubTensor);
    }
    rewriter.create<hivm::SyncBlockUnlockOp>(loc, createdLock);
    if (hasReturn)
      rewriter.create<scope::ReturnOp>(loc, tmpBuffer.toLogicalTensor());
    else
      rewriter.create<scope::ReturnOp>(loc);

    if (hasReturn)
      rewriter.replaceOp(op, scopeOp.getResults());
    else
      rewriter.eraseOp(op);
    eraseToMemrefIfUnused(srcOperands.front(), rewriter);
    return success();
  }
};

} // namespace mlir

#endif // BISHENGIR_TRANSFORMS_NORMALIZE_NORMALIZEATOMICTEMPLATE_H
