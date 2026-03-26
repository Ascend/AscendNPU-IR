//===- NormalizeAtomic.cpp ----------------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HFusion/Transforms/NormalizeUtils.h"

namespace mlir::hfusion {
namespace NormalizeAtomicOps {

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

class StaticSizedBuffer {
  OpBuilder &builder;
  const Location loc;

  const TypedValue<MemRefType> gmMemrefSource;
  memref::SubViewOp definingOp;

public:
  // maybe dynnamic sized Memref
  TypedValue<MemRefType> dBuffer;
  // static sized Memref
  TypedValue<MemRefType> sBuffer;

  StaticSizedBuffer(OpBuilder &_builder, Location _loc,
                    TypedValue<MemRefType> _gmMemrefSource)
      : builder(_builder), loc(_loc), gmMemrefSource(_gmMemrefSource) {
    const Type elemType = gmMemrefSource.getType().getElementType();
    const MemRefType rawMemref =
        MemRefType::get(gmMemrefSource.getType().getShape(), elemType);
    if (gmMemrefSource.getType().hasStaticShape()) {
      dBuffer = sBuffer = builder.create<memref::AllocOp>(loc, rawMemref);
      return;
    }
    definingOp = gmMemrefSource.getDefiningOp<memref::SubViewOp>();
    // dynamically shaped GM should come from a subview of statically shaped GM
    assert(definingOp && definingOp.getSourceType().hasStaticShape() &&
           "dynamically shaped GM should come from a subview of statically "
           "shaped GM");
    sBuffer = builder.create<memref::AllocOp>(
        loc, rawMemref.clone(definingOp.getSourceType().getShape()));
    SmallVector<OpFoldResult> zeroOffsets(gmMemrefSource.getType().getRank(),
                                          builder.getI64IntegerAttr(0));
    dBuffer = builder.create<memref::SubViewOp>(loc, sBuffer, zeroOffsets,
                                                definingOp.getMixedSizes(),
                                                definingOp.getMixedStrides());
  }
  TypedValue<TensorType> toStaticTensor() {
    return builder
        .create<bufferization::ToTensorOp>(
            loc, /*memref:*/ sBuffer, /*restrict:*/ true, /*writable:*/ true)
        .getResult();
  }

  void storeBack(TypedValue<TensorType> tensorSrc) {
    if (dBuffer.getType().hasStaticShape()) {
      builder.create<bufferization::MaterializeInDestinationOp>(
          loc, /*result:*/ Type(), /*source:*/ tensorSrc,
          /*dest:*/ gmMemrefSource, /*restrict:*/ false, /*writable:*/ true);
      return;
    }
    SmallVector<OpFoldResult> zeroOffsets(gmMemrefSource.getType().getRank(),
                                          builder.getI64IntegerAttr(0));
    Value dtensor = builder.create<tensor::ExtractSliceOp>(
        loc, /*source*/ tensorSrc, /*offsets*/ zeroOffsets,
        /*sizes*/ definingOp.getMixedSizes(),
        /*strides*/ definingOp.getMixedStrides());
    builder.create<bufferization::MaterializeInDestinationOp>(
        loc, /*result:*/ Type(), /*source:*/ dtensor,
        /*dest:*/ gmMemrefSource, /*restrict:*/ false, /*writable:*/ true);
  }
};

inline bool isElementFP8(ShapedType st) {
  auto t = dyn_cast<FloatType>(st.getElementType());
  return t && t.getWidth() == 8;
}

struct Elemwise : public OpRewritePattern<hfusion::StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  inline bool isHardwareSupported(Type dtype) const {
    return TypeSwitch<Type, bool>(getElementTypeOrSelf(dtype))
        .Case<Float16Type, Float32Type, BFloat16Type>([](Type) { return true; })
        .Case<IntegerType>([](IntegerType t) {
          return t.getWidth() == 8 || t.getWidth() == 16 || t.getWidth() == 32;
        })
        .Default([](Type) { return false; });
  }

  inline std::optional<std::variant<hfusion::BinaryFn, linalg::BinaryFn>>
  getDecomposedBinFn(hfusion::StoreOp op) const {
    Type elemType = getElementTypeOrSelf(op.getOutputs()[0].getType());
    switch (op.getAtomicKind()) {
    case AtomicKind::AND:
      return hfusion::BinaryFn::vand;
    case AtomicKind::OR:
      return hfusion::BinaryFn::vor;
    case AtomicKind::XOR:
      return hfusion::BinaryFn::vxor;
    case AtomicKind::ADD:
      if (isHardwareSupported(elemType))
        return std::nullopt;
      return linalg::BinaryFn::add;
    case AtomicKind::MAX:
      if (isHardwareSupported(elemType))
        return std::nullopt;
      return linalg::BinaryFn::max_signed;
    case AtomicKind::UMAX:
      return linalg::BinaryFn::max_unsigned;
    case AtomicKind::MIN:
      if (isHardwareSupported(elemType))
        return std::nullopt;
      return linalg::BinaryFn::min_signed;
    case AtomicKind::UMIN:
      return linalg::BinaryFn::min_unsigned;
    default:
      return std::nullopt;
    }
  }

  LogicalResult matchAndRewrite(hfusion::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto maybeFn = getDecomposedBinFn(op);
    if (!maybeFn.has_value())
      return failure();
    auto getElemwiseBinOpResult =
        [fn /*: variant<BinaryFn, BinaryFn>*/ = maybeFn.value(), &rewriter,
         loc](Value lhsTensor, Value rhsTensor, Value resultTensor) {
          if (std::holds_alternative<hfusion::BinaryFn>(fn)) {
            auto binaryAttr = rewriter.getAttr<hfusion::BinaryFnAttr>(
                std::get<hfusion::BinaryFn>(fn));
            auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
            return rewriter
                .create<hfusion::ElemwiseBinaryOp>(
                    loc, ValueRange{lhsTensor, rhsTensor},
                    ValueRange{resultTensor}, ArrayRef{fnAttr})
                .getResult(0);
          } else {
            auto binaryAttr = rewriter.getAttr<linalg::BinaryFnAttr>(
                std::get<linalg::BinaryFn>(fn));
            auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
            return rewriter
                .create<linalg::ElemwiseBinaryOp>(
                    loc, ValueRange{lhsTensor, rhsTensor},
                    ValueRange{resultTensor}, ArrayRef{fnAttr})
                .getResult(0);
          }
        };

    rewriter.setInsertionPointAfter(op);

    TypedValue<MemRefType> ubMemref = dyn_cast<TypedValue<MemRefType>>(
                               op.getInputs()[0]),
                           gmMemref = dyn_cast<TypedValue<MemRefType>>(
                               op.getOutputs()[0]);
    assert(ubMemref);
    assert(gmMemref);

    // prepare buffer and RHS
    StaticSizedBuffer rhsBuffer(rewriter, loc, gmMemref);
    rewriter.create<memref::CopyOp>(loc, ubMemref, rhsBuffer.dBuffer);

    // Lambda to cast to/from F32 if it's FP8
    // isForward: true => FP8 to FP32; false => FP32 to FP8
    auto castF32IfFP8 = [&rewriter, isFP8 = isElementFP8(gmMemref.getType()),
                         srcType = gmMemref.getType().getElementType(),
                         tarType = rewriter.getF32Type()](
                            Value sourceTensor, bool isForward) -> Value {
      return isFP8
                 ? castTo(rewriter, sourceTensor, isForward ? tarType : srcType)
                 : sourceTensor;
    };

    // compute as f32 if elem type is fp8
    Value rhsTensor = castF32IfFP8(rhsBuffer.toStaticTensor(), true);
    // get LHS in tensor
    StaticSizedBuffer lhsBuffer(rewriter, loc, gmMemref);

    // wrap in lock
    SyncBlockLockGuard _(rewriter, loc);
    // Binary Op
    rewriter.create<memref::CopyOp>(loc, gmMemref, lhsBuffer.dBuffer);
    Value lhsTensor = castF32IfFP8(lhsBuffer.toStaticTensor(), true);
    Value result = castF32IfFP8(
        getElemwiseBinOpResult(lhsTensor, rhsTensor, lhsTensor), false);
    assert(isa<TypedValue<TensorType>>(result));
    // store the result
    lhsBuffer.storeBack(cast<TypedValue<TensorType>>(result));
    rewriter.eraseOp(op);
    return success();
  }
};

struct CAS : public OpRewritePattern<hfusion::AtomicCasOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::AtomicCasOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    Location loc = op->getLoc();
    TypedValue<MemRefType> ubComparing = dyn_cast<TypedValue<MemRefType>>(
                               op.getInput()[0]),
                           ubStoring = dyn_cast<TypedValue<MemRefType>>(
                               op.getInput()[1]),
                           gmMemref =
                               dyn_cast<TypedValue<MemRefType>>(op.getDst());
    assert(ubComparing);
    assert(ubStoring);
    assert(gmMemref);
    StaticSizedBuffer comparingBuffer(rewriter, loc, gmMemref);
    StaticSizedBuffer storingBuffer(rewriter, loc, gmMemref);
    StaticSizedBuffer gmValBuffer(rewriter, loc, gmMemref);
    rewriter.create<memref::CopyOp>(loc, ubComparing, comparingBuffer.dBuffer);
    rewriter.create<memref::CopyOp>(loc, ubStoring, storingBuffer.dBuffer);

    SyncBlockLockGuard _(rewriter, loc);
    rewriter.create<memref::CopyOp>(loc, gmMemref, gmValBuffer.dBuffer);
    const TypedValue<TensorType> gmValTensor = gmValBuffer.toStaticTensor();
    // Lambda to cast to/from F32 if it's FP8
    // isForward: true => FP8 to FP32; false => FP32 to FP8
    auto castF32IfFP8 = [&rewriter, isFP8 = isElementFP8(gmValTensor.getType()),
                         srcType = gmValTensor.getType().getElementType(),
                         tarType = rewriter.getF32Type()](
                            Value sourceTensor, bool isForward) -> Value {
      return isFP8
                 ? castTo(rewriter, sourceTensor, isForward ? tarType : srcType)
                 : sourceTensor;
    };

    // compare
    // use f32 instead of fp8 to cmp
    Value cmpResult =
        createCmpOp(rewriter, loc, castF32IfFP8(gmValTensor, true),
                    castF32IfFP8(comparingBuffer.toStaticTensor(), true),
                    CompareFn::veq)
            ->getResult(0);
    // select
    // since intrinsics support selecting fp8, no need for casting
    Value selectedResult =
        rewriter
            .create<hfusion::SelectOp>(
                loc, /*resultTensorTypes*/ gmValTensor.getType(),
                /*inputs*/
                ValueRange{cmpResult, storingBuffer.toStaticTensor(),
                           gmValTensor},
                /*outputs*/ gmValTensor)
            .getResult(0);
    // store the result
    assert(isa<TypedValue<TensorType>>(selectedResult));
    gmValBuffer.storeBack(cast<TypedValue<TensorType>>(selectedResult));
    rewriter.eraseOp(op);
    return success();
  }
};

struct XCHG : public OpRewritePattern<hfusion::AtomicXchgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::AtomicXchgOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    Location loc = op->getLoc();
    TypedValue<MemRefType> ubMemref = dyn_cast<TypedValue<MemRefType>>(
                               op.getInput()[0]),
                           gmMemref =
                               dyn_cast<TypedValue<MemRefType>>(op.getDst());
    assert(ubMemref);
    assert(gmMemref);
    StaticSizedBuffer tmpBuffer(rewriter, loc, gmMemref);
    SyncBlockLockGuard _(rewriter, loc);
    rewriter.create<memref::CopyOp>(loc, gmMemref, tmpBuffer.dBuffer);
    rewriter.create<memref::CopyOp>(loc, ubMemref, gmMemref);
    rewriter.create<memref::CopyOp>(loc, tmpBuffer.dBuffer, ubMemref);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace NormalizeAtomicOps

// For inputs with non-F16/F32 types, cast to F32 for sorting first, then cast
// back to the original type;
struct NormalizeSortOp : public OpRewritePattern<hfusion::SortOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hfusion::SortOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Value input = op.getSrc();
    auto inputType = llvm::dyn_cast<mlir::TensorType>(input.getType());
    mlir::Type elemType = inputType.getElementType();
    auto floatType = llvm::dyn_cast<mlir::FloatType>(elemType);

    // only inputs with non-F16/F32 types will trigger the subsequent rewriting
    if (floatType && (floatType.isF16() || floatType.isF32())) {
      return mlir::failure();
    }

    auto intType = llvm::dyn_cast<mlir::IntegerType>(elemType);

    if (intType && (intType.isInteger(32) || intType.isInteger(64))) {
      return mlir::failure();
    }

    mlir::Type f32Type = mlir::Float32Type::get(rewriter.getContext());

    // convert type of input to f32
    auto castToF32 =
        hfusion::castTo(rewriter, input, f32Type, hfusion::RoundMode::ROUND);

    auto newSortOp = rewriter.create<hfusion::SortOp>(
        op.getLoc(), castToF32.getType(), castToF32, op.getDescending(),
        op.getSortAxis());

    // convert type from f32 to origin type
    auto castBack = hfusion::castTo(rewriter, newSortOp.getResult()[0],
                                    elemType, hfusion::RoundMode::ROUND);

    rewriter.replaceOp(op, castBack);

    return mlir::success();
  }
};

void populateNormalizeAtomicAndSortPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  if (archisAscend950) {
    patterns.add<NormalizeAtomicOps::Elemwise>(ctx);
    patterns.add<NormalizeAtomicOps::CAS>(ctx);
    patterns.add<NormalizeAtomicOps::XCHG>(ctx);
  }
  patterns.add<NormalizeSortOp>(ctx);
}
} // namespace mlir::hfusion
