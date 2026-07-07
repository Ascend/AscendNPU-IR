//===-------- NormalizeTraitsBase.cpp ----------------------------*- C++ -*-===//
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

#include "bishengir/Dialect/HIVM/Transforms/NormalizeTraitsBase.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/NormalizePatterns.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Normalize/Utils/CastingTemplateHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
namespace mlir::hivm {

template <typename MapT, typename KeyT>
static typename MapT::mapped_type lookupOrUnreachable(const MapT &map, KeyT key,
                                                      const char *errorMessage) {
  auto it = map.find(key);
  if (it == map.end())
    llvm_unreachable(errorMessage);
  return it->second;
}

template <typename MapT, typename KeyT>
static std::optional<typename MapT::mapped_type> lookupOrNullopt(
    const MapT &map, KeyT key) {
  auto it = map.find(key);
  if (it == map.end())
    return std::nullopt;
  return it->second;
}

static Value getPrimaryCastValue(VCastOp op) {
  return op->getResults().empty() ? op.getSingleDst() : op->getResults()[0];
}

static hivm::RoundMode getIntegerCastRoundMode(Type sourceElemType,
                                               Type targetElemType) {
  auto sourceIntType = dyn_cast<IntegerType>(sourceElemType);
  auto targetIntType = dyn_cast<IntegerType>(targetElemType);
  if (!sourceIntType || !targetIntType)
    return hivm::RoundMode::RINT;
  if (targetIntType.getWidth() >= sourceIntType.getWidth())
    return hivm::RoundMode::RINT;
  return hivm::RoundMode::TRUNCWITHOVERFLOW;
}

static void annotateCast(PatternRewriter &rewriter, Location loc, VCastOp op,
                         ArrayRef<NamedAttribute> attrs) {
  auto markOp =
      rewriter.create<annotation::MarkOp>(loc, getPrimaryCastValue(op));
  markOp->setAttrs(DictionaryAttr::get(rewriter.getContext(), attrs));
}

bool mlir::hivm::NormalizeTraitsBase::archIsRegbased() {
  return hivm::archIsRegbased;
}

bool mlir::hivm::NormalizeTraitsBase::archIsAscend950() {
  return hivm::archisAscend950;
}

Value mlir::hivm::NormalizeTraitsBase::createIsNanOp(
    PatternRewriter &rewriter, Location loc, Value src) {
  Value emptyDst = utils::createEmptyOpWithTargetElemType(rewriter, loc, src,
                                                         rewriter.getI1Type());
  return rewriter
      .create<hivm::VIsNanOp>(loc, mlir::TypeRange{emptyDst.getType()},
                               mlir::ValueRange{src}, mlir::ValueRange{emptyDst})
      .getResults()[0];
}

Value mlir::hivm::NormalizeTraitsBase::createCastValueFromSourceOp(
    PatternRewriter &rewriter, Location loc, VCastOp op, Value input,
    Type targetElemType, CastRoundKind executionKind, CastSignKind signKind,
    bool enableSaturate, CastUnsignedModeKind unsignedModeKind) {
  hivm::RoundMode defaultRoundMode =
      utils::selectRoundMode<hivm::RoundMode>(
          getElementTypeOrSelf(input.getType()), targetElemType);
  hivm::RoundMode roundMode =
      mapCastRoundKind(executionKind, defaultRoundMode);
  hivm::TypeFn typeFn = mapCastSignKind(signKind, op.getCast());
  hivm::UnsignedMode unsignedMode =
      mapCastUnsignedModeKind(unsignedModeKind, hivm::UnsignedMode::SI2SI);
  auto castOp = hivm::castTo(rewriter, loc, input,
                             rewriter.getAttr<hivm::RoundModeAttr>(roundMode),
                             targetElemType);
  castOp.setCastAttr(rewriter.getAttr<hivm::TypeFnAttr>(typeFn));
  if (executionKind == CastRoundKind::Default)
    return getPrimaryCastValue(castOp);

  if (executionKind == CastRoundKind::TruncEnableOverflow)
    annotateCast(rewriter, loc, castOp,
                 {rewriter.getNamedAttr(kOverflowModeAttr,
                                        rewriter.getStringAttr("trunc"))});
  if (enableSaturate) {
    SmallVector<NamedAttribute> attrs{
        rewriter.getNamedAttr(kOverflowModeAttr,
                              rewriter.getStringAttr("saturate"))};
    if (unsignedMode == hivm::UnsignedMode::UI2SI ||
        unsignedMode == hivm::UnsignedMode::UI2UI)
      attrs.push_back(rewriter.getNamedAttr(kSaturateSrcUnsignedAttr,
                                            rewriter.getUnitAttr()));
    if (unsignedMode == hivm::UnsignedMode::SI2UI ||
        unsignedMode == hivm::UnsignedMode::UI2UI)
      attrs.push_back(rewriter.getNamedAttr(kSaturateDstUnsignedAttr,
                                            rewriter.getUnitAttr()));
    annotateCast(rewriter, loc, castOp, attrs);
  }
  return getPrimaryCastValue(castOp);
}

Value mlir::hivm::NormalizeTraitsBase::castScalarThroughTensor(
    PatternRewriter &rewriter, Location loc, Value scalar, Type dstType) {
  auto tensorType = RankedTensorType::get({1}, scalar.getType());
  Value fromElementsOp =
      rewriter.create<tensor::FromElementsOp>(loc, tensorType, scalar);
  hivm::RoundMode roundMode =
      utils::selectRoundMode<hivm::RoundMode>(scalar.getType(), dstType);
  auto castOp = hivm::castTo(
      rewriter, loc, fromElementsOp,
      rewriter.getAttr<hivm::RoundModeAttr>(roundMode), dstType);
  Value castValue = getPrimaryCastValue(castOp);
  auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  return rewriter.create<tensor::ExtractOp>(loc, castValue, ValueRange{c0});
}

template <typename UnaryOp>
static mlir::Value createHIVMUnaryOp(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value input,
                                     mlir::Value dst) {
  return rewriter
      .create<UnaryOp>(loc, mlir::TypeRange{dst.getType()},
                       mlir::ValueRange{input}, mlir::ValueRange{dst})
      .getResults()[0];
}

template <typename BinaryOp>
static mlir::Value createHIVMBinaryOp(mlir::PatternRewriter &rewriter,
                                      mlir::Location loc, mlir::Value lhs,
                                      mlir::Value rhs, mlir::Value dst) {
  return rewriter
      .create<BinaryOp>(loc, mlir::TypeRange{dst.getType()},
                        mlir::ValueRange{lhs, rhs}, mlir::ValueRange{dst})
      .getResults()[0];
}

template <typename TernaryOp>
mlir::Value createHIVMTernaryOp(mlir::PatternRewriter *rewriter,
                                mlir::Location loc, mlir::Value lhs,
                                mlir::Value mid, mlir::Value rhs,
                                mlir::Value dst) {
  return rewriter
      ->template create<TernaryOp>(loc, mlir::TypeRange{dst.getType()},
                         mlir::ValueRange{lhs, mid, rhs},
                         mlir::ValueRange{dst}, mlir::Value{},
                         llvm::ArrayRef<int64_t>{},
                         llvm::ArrayRef<int64_t>{})
      .getResults()[0];
}

using UnaryOpFn = Value (*)(PatternRewriter &, Location, Value, Value);
using BinaryOpFn = Value (*)(PatternRewriter &, Location, Value, Value, Value);
using TernaryOpFn =
    Value (*)(PatternRewriter *, Location, Value, Value, Value, Value);
using UnaryOpMatcherFn = bool (*)(Operation *);
using BinaryOpMatcherFn = bool (*)(Operation *);

template <typename OpType>
static bool matchHIVMOp(Operation *op) {
  auto typedOp = dyn_cast_or_null<OpType>(op);
  return typedOp && typedOp.hasPureTensorSemantics();
}

template <typename Kind, typename Fn>
static std::optional<Fn>
lookupMappedFn(const llvm::DenseMap<Kind, Fn> &kindToFn, Kind kind) {
  auto it = kindToFn.find(kind);
  if (it == kindToFn.end())
    return std::nullopt;
  return it->second;
}

static std::optional<TernaryOpFn> mapTernaryKindToCreator(TernaryKind kind) {
  static const llvm::DenseMap<TernaryKind, TernaryOpFn> kindToFn = {
      {TernaryKind::Select, createHIVMTernaryOp<hivm::VSelOp>},
  };
  return lookupMappedFn(kindToFn, kind);
}

static const llvm::DenseMap<UnaryKind, UnaryOpFn> unaryOpMap = {
    {UnaryKind::Rec, createHIVMUnaryOp<hivm::VRecOp>},
    {UnaryKind::Sqrt, createHIVMUnaryOp<hivm::VSqrtOp>},
    {UnaryKind::Abs, createHIVMUnaryOp<hivm::VAbsOp>},
    {UnaryKind::Not, createHIVMUnaryOp<hivm::VNotOp>},
    {UnaryKind::Exp, createHIVMUnaryOp<hivm::VExpOp>},
    {UnaryKind::Ln, createHIVMUnaryOp<hivm::VLnOp>},
    {UnaryKind::Log2, createHIVMUnaryOp<hivm::VLog2Op>},
};

static const llvm::DenseMap<BinaryKind, BinaryOpFn> binaryOpMap = {
    {BinaryKind::Add, createHIVMBinaryOp<hivm::VAddOp>},
    {BinaryKind::Sub, createHIVMBinaryOp<hivm::VSubOp>},
    {BinaryKind::Mul, createHIVMBinaryOp<hivm::VMulOp>},
    {BinaryKind::Div, createHIVMBinaryOp<hivm::VDivOp>},
    {BinaryKind::Mod, createHIVMBinaryOp<hivm::VModOp>},
    {BinaryKind::ModUnsigned, createHIVMBinaryOp<hivm::VModUIOp>},
    {BinaryKind::Min, createHIVMBinaryOp<hivm::VMinOp>},
    {BinaryKind::Max, createHIVMBinaryOp<hivm::VMaxOp>},
    {BinaryKind::And, createHIVMBinaryOp<hivm::VAndOp>},
    {BinaryKind::Xor, createHIVMBinaryOp<hivm::VXorOp>},
    {BinaryKind::Or, createHIVMBinaryOp<hivm::VOrOp>},
    {BinaryKind::Powf, createHIVMBinaryOp<hivm::VPowOp>},
    {BinaryKind::MinSigned, createHIVMBinaryOp<hivm::VMinOp>},
    {BinaryKind::MaxSigned, createHIVMBinaryOp<hivm::VMaxOp>},
    {BinaryKind::MinUnsigned, createHIVMBinaryOp<hivm::VMinOp>},
    {BinaryKind::MaxUnsigned, createHIVMBinaryOp<hivm::VMaxOp>},
};

static const llvm::DenseMap<UnaryKind, UnaryOpMatcherFn> unaryOpMatcherMap = {
    {UnaryKind::Rec, matchHIVMOp<hivm::VRecOp>},
    {UnaryKind::Sqrt, matchHIVMOp<hivm::VSqrtOp>},
    {UnaryKind::Abs, matchHIVMOp<hivm::VAbsOp>},
    {UnaryKind::Not, matchHIVMOp<hivm::VNotOp>},
    {UnaryKind::Exp, matchHIVMOp<hivm::VExpOp>},
};

static const llvm::DenseMap<BinaryKind, BinaryOpMatcherFn> binaryOpMatcherMap = {
    {BinaryKind::Add, matchHIVMOp<hivm::VAddOp>},
    {BinaryKind::Sub, matchHIVMOp<hivm::VSubOp>},
    {BinaryKind::Mul, matchHIVMOp<hivm::VMulOp>},
    {BinaryKind::Div, matchHIVMOp<hivm::VDivOp>},
    {BinaryKind::Mod, matchHIVMOp<hivm::VModOp>},
    {BinaryKind::ModUnsigned, matchHIVMOp<hivm::VModUIOp>},
    {BinaryKind::Min, matchHIVMOp<hivm::VMinOp>},
    {BinaryKind::Max, matchHIVMOp<hivm::VMaxOp>},
    {BinaryKind::And, matchHIVMOp<hivm::VAndOp>},
    {BinaryKind::Powf, matchHIVMOp<hivm::VPowOp>},
    {BinaryKind::Or, matchHIVMOp<hivm::VOrOp>},
    {BinaryKind::Xor, matchHIVMOp<hivm::VXorOp>},
    {BinaryKind::MinSigned, matchHIVMOp<hivm::VMinOp>},
    {BinaryKind::MaxSigned, matchHIVMOp<hivm::VMaxOp>},
    {BinaryKind::MinUnsigned, matchHIVMOp<hivm::VMinOp>},
    {BinaryKind::MaxUnsigned, matchHIVMOp<hivm::VMaxOp>},
};

bool mlir::hivm::NormalizeTraitsBase::matchOp(Operation *op, UnaryKind kind) {
  return lookupOrUnreachable(unaryOpMatcherMap, kind, "unsupported unary kind")(
      op);
}

bool mlir::hivm::NormalizeTraitsBase::matchOp(Operation *op, BinaryKind kind) {
  return lookupOrUnreachable(binaryOpMatcherMap, kind,
                             "unsupported binary kind")(op);
}

static CompareMode mapCompareKindToCompareMode(CompareKind kind) {
  static const llvm::DenseMap<CompareKind, CompareMode> compareKindMap = {
      {CompareKind::EQ, CompareMode::EQ},
      {CompareKind::NE, CompareMode::NE},
      {CompareKind::LT, CompareMode::LT},
      {CompareKind::GT, CompareMode::GT},
      {CompareKind::GE, CompareMode::GE},
      {CompareKind::LE, CompareMode::LE}};
  return lookupOrUnreachable(compareKindMap, kind, "Unknown CompareKind");
}

static std::optional<hivm::RoundMode>
mapCastRoundKindToRoundMode(CastRoundKind kind) {
  static const llvm::DenseMap<CastRoundKind, hivm::RoundMode> kindToMode = {
      {CastRoundKind::Round, hivm::RoundMode::ROUND},
      {CastRoundKind::Floor, hivm::RoundMode::FLOOR},
      {CastRoundKind::Ceil, hivm::RoundMode::CEIL},
      {CastRoundKind::Trunc, hivm::RoundMode::TRUNC},
      {CastRoundKind::TruncWithOverflow, hivm::RoundMode::TRUNCWITHOVERFLOW},
      {CastRoundKind::RInt, hivm::RoundMode::RINT},
      {CastRoundKind::TruncEnableOverflow, hivm::RoundMode::TRUNC},
  };
  return lookupOrNullopt(kindToMode, kind);
}

static std::optional<hivm::UnsignedMode>
mapCastUnsignedModeKindToUnsignedMode(CastUnsignedModeKind kind) {
  static const llvm::DenseMap<CastUnsignedModeKind, hivm::UnsignedMode>
      kindToMode = {
          {CastUnsignedModeKind::SignedToSigned, hivm::UnsignedMode::SI2SI},
          {CastUnsignedModeKind::SignedToUnsigned, hivm::UnsignedMode::SI2UI},
          {CastUnsignedModeKind::UnsignedToSigned, hivm::UnsignedMode::UI2SI},
          {CastUnsignedModeKind::UnsignedToUnsigned,
           hivm::UnsignedMode::UI2UI},
      };

  return lookupOrNullopt(kindToMode, kind);
}

static std::optional<hivm::TypeFn> mapCastSignKindToTypeFn(CastSignKind kind) {
  static const llvm::DenseMap<CastSignKind, hivm::TypeFn> kindToTypeFn = {
      {CastSignKind::Signed, hivm::TypeFn::cast_signed},
      {CastSignKind::Unsigned, hivm::TypeFn::cast_unsigned},
  };

  return lookupOrNullopt(kindToTypeFn, kind);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createCmpOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst,
    CompareKind kind) {
  CompareMode cmpMode = mapCompareKindToCompareMode(kind);
  Type boolType = rewriter.getIntegerType(1);
  auto emptyOp = utils::createEmptyOpWithTargetElemType(rewriter, loc, input,
                                                        boolType);
  auto cmpOp = rewriter.create<VCmpOp>(
      loc, TypeRange(emptyOp), ValueRange({input, dst}), ValueRange(emptyOp),
      /*is_signed=*/true, cmpMode);
  return cmpOp.getResult()[0];
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createCmpOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs,
    VCmpOp sourceOp) {
  Type boolType = rewriter.getIntegerType(1);
  auto emptyOp =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, lhs, boolType);
  auto cmpOp = rewriter.create<VCmpOp>(
      loc, TypeRange(emptyOp), ValueRange{lhs, rhs}, ValueRange(emptyOp),
      sourceOp.getIsSigned(), sourceOp.getCompareMode());
  return cmpOp.getResult()[0];
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createUnaryOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst,
    UnaryKind kind) {
  return lookupOrUnreachable(unaryOpMap, kind, "unsupported unary kind")(
      rewriter, loc, input, dst);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createFloorOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst) {
  return createCastOp(rewriter, loc, input, getElementTypeOrSelf(dst.getType()),
                      CastRoundKind::Floor);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createBinaryOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value dst,
    BinaryKind kind) {
  return lookupOrUnreachable(binaryOpMap, kind, "unsupported binary kind")(
      rewriter, loc, lhs, rhs, dst);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createShiftOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value dst,
    ShiftKind kind, Operation *sourceOp) {
  switch (kind) {
  case ShiftKind::Left:
    if (sourceOp && !isa<hivm::VShLOp>(sourceOp))
      llvm_unreachable("source op shift kind mismatch");
    return rewriter
        .create<hivm::VShLOp>(
            loc, TypeRange(dst.getType()), ValueRange({lhs, rhs}),
            ValueRange({dst}),
            rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}),
            rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}))
        .getResult()[0];
  case ShiftKind::RightSigned:
  case ShiftKind::RightUnsigned: {
    BoolAttr round = rewriter.getBoolAttr(true);
    if (sourceOp) {
      auto shiftOp = cast<hivm::VShROp>(sourceOp);
      round = shiftOp.getRoundAttr();
    }
    return rewriter
        .create<hivm::VShROp>(loc, TypeRange(dst.getType()),
                              ValueRange({lhs, rhs}), ValueRange({dst}), round,
                              nullptr, nullptr)
        .getResult()[0];
  }
  }
  llvm_unreachable("unsupported shift kind");
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createTernaryOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value mid, Value rhs,
    Value dst, TernaryKind kind) {
  auto fn = mapTernaryKindToCreator(kind);
  if (!fn)
    llvm_unreachable("unsupported ternary kind");
  return (*fn)(&rewriter, loc, lhs, mid, rhs, dst);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createCastOp(
    PatternRewriter &rewriter, Location loc, Value input, Type targetElemType,
    std::optional<RoundMode> roundMode) {
  if (!isa<ShapedType>(input.getType()))
    return castScalarThroughTensor(rewriter, loc, input, targetElemType);

  if (!roundMode) {
    hivm::RoundMode defaultRoundMode =
        utils::selectRoundMode<hivm::RoundMode>(
            getElementTypeOrSelf(input.getType()), targetElemType);
    auto castOp = hivm::castTo(
        rewriter, loc, input,
        rewriter.getAttr<hivm::RoundModeAttr>(defaultRoundMode),
        targetElemType);
    return getPrimaryCastValue(castOp);
  }

  Type srcElemType = getElementTypeOrSelf(input.getType());
  hivm::RoundMode actualRoundMode = *roundMode;
  if ((srcElemType.isF16() || srcElemType.isBF16()) && targetElemType.isF32()) {
    // HIVM VCastOp only supports f16/bf16 -> f32 in rint mode.
    actualRoundMode = hivm::RoundMode::RINT;
  }
  auto castOp = hivm::castTo(rewriter, loc, input,
                             rewriter.getAttr<hivm::RoundModeAttr>(actualRoundMode),
                             targetElemType);
  castOp.setCastAttr(rewriter.getAttr<hivm::TypeFnAttr>(
      hivm::TypeFn::cast_signed));
  return getPrimaryCastValue(castOp);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createCastOp(
    PatternRewriter &rewriter, Location loc, Value input, Type targetElemType,
    CastRoundKind kind, Value output, CastSignKind signKind) {
  if (signKind == CastSignKind::Preserve)
    llvm_unreachable("createCastOp does not support CastSignKind::Preserve");
  (void)output;
  if (!isa<ShapedType>(input.getType()))
    return castScalarThroughTensor(rewriter, loc, input, targetElemType);
  Type srcElemType = getElementTypeOrSelf(input.getType());
  auto roundMode =
      kind == CastRoundKind::Default
          ? mlir::utils::selectRoundMode<hivm::RoundMode>(srcElemType,
                                                          targetElemType)
          : mapCastRoundKindToRoundMode(kind).value();
  if ((srcElemType.isF16() || srcElemType.isBF16()) &&
      targetElemType.isF32()) {
    // HIVM VCastOp only supports f16/bf16 -> f32 in rint mode.
    roundMode = hivm::RoundMode::RINT;
  }
  auto typeFn = mapCastSignKind(signKind, hivm::TypeFn::cast_signed);
  auto castOp = hivm::castTo(rewriter, loc, input,
                             rewriter.getAttr<hivm::RoundModeAttr>(roundMode),
                             targetElemType);
  castOp.setCastAttr(rewriter.getAttr<hivm::TypeFnAttr>(typeFn));
  return getPrimaryCastValue(castOp);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createFillOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst) {
  if (isa<ShapedType>(input.getType()) || !isa<TensorType>(dst.getType()))
    llvm_unreachable(
        "NormalizeTraitsBase::createFillOp only supports scalar-to-tensor fills");
  return rewriter
      .create<VBrcOp>(loc, TypeRange(dst), input, dst,
                      rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}))
      .getResult()[0];
}

bool mlir::hivm::NormalizeTraitsBase::matchFillOp(Operation *op) {
  return isa<hivm::VBrcOp>(op) &&
         !isa<ShapedType>(cast<hivm::VBrcOp>(op).getSrc().getType());
}

Value mlir::hivm::NormalizeTraitsBase::getFillInput(Operation *op) {
  return cast<hivm::VBrcOp>(op).getSrc();
}

bool mlir::hivm::NormalizeTraitsBase::matchBroadcastOp(Operation *op) {
  return isa<hivm::VBrcOp>(op) &&
         isa<ShapedType>(cast<hivm::VBrcOp>(op).getSrc().getType());
}

Value mlir::hivm::NormalizeTraitsBase::getBroadcastInput(Operation *op) {
  return cast<hivm::VBrcOp>(op).getSrc();
}

SmallVector<int64_t>
mlir::hivm::NormalizeTraitsBase::getBroadcastDims(Operation *op) {
  return llvm::to_vector(cast<hivm::VBrcOp>(op).getBroadcastDims());
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createBroadcastOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst,
    ArrayRef<int64_t> dims) {
  // HIVM vbrc requires empty broadcast_dims for scalar sources. Treat a
  // rank-0 tensor source as scalar by extracting it before materializing vbrc.
  if (auto inputType = dyn_cast<RankedTensorType>(input.getType());
      inputType && inputType.getRank() == 0) {
    Value scalar = rewriter.create<tensor::ExtractOp>(loc, input, ValueRange{});
    return createFillOp(rewriter, loc, scalar, dst);
  }

  auto brcOp = rewriter.create<hivm::VBrcOp>(
      loc, TypeRange(dst.getType()), input, dst,
      /*tempBuffer=*/Value{}, rewriter.getDenseI64ArrayAttr(dims));
  return brcOp->getResult(0);
}
mlir::Value mlir::hivm::NormalizeTraitsBase::createBitcastOp(
    PatternRewriter &rewriter, Location loc, Type resultType, Value source) {
  return rewriter.create<BitcastOp>(loc, resultType, source).getResult();
}

static SmallVector<Type> getReductionResultTypes(VReduceOp op, ValueRange dsts) {
  SmallVector<Type> resultTypes;
  if (!op.hasPureTensorSemantics())
    return resultTypes;

  resultTypes.reserve(dsts.size());
  for (Value dst : dsts)
    resultTypes.push_back(dst.getType());
  return resultTypes;
}

Operation *mlir::hivm::NormalizeTraitsBase::createReduceOp(
    PatternRewriter &rewriter, Location loc, VReduceOp op, Value src,
    ValueRange dsts, DenseI64ArrayAttr reduceDimsAttr) {
  SmallVector<Type> resultTypes = getReductionResultTypes(op, dsts);
  if (auto tieBreakLeft = op.getTieBreakLeftAttr()) {
    return rewriter.create<hivm::VReduceOp>(
        loc, resultTypes, src, dsts, op.getArithAttr(), op.getUnsignedSrcAttr(),
        tieBreakLeft, reduceDimsAttr);
  }
  return rewriter.create<hivm::VReduceOp>(
      loc, resultTypes, src, dsts, op.getArithAttr(), op.getUnsignedSrcAttr(),
      reduceDimsAttr);
}

Operation *mlir::hivm::NormalizeTraitsBase::createReduceWithIndexOp(
    PatternRewriter &rewriter, Location loc, VReduceOp op,
    ArrayRef<Value> newInputs, ArrayRef<Value> newInits) {
  if (newInputs.empty())
    llvm::report_fatal_error("VReduceOp expects a source input");
  return createReduceOp(rewriter, loc, op, newInputs.front(), newInits,
                        op.getReduceDimsAttr());
}

Value mlir::hivm::NormalizeTraitsBase::castReduceIndexTensor(
    PatternRewriter &rewriter, Location loc, Value value,
    IntegerType targetElemType, Value) {
  Type sourceElemType = getElementTypeOrSelf(value.getType());
  auto castOp = hivm::castTo(
      rewriter, loc, value,
      rewriter.getAttr<hivm::RoundModeAttr>(
          getIntegerCastRoundMode(sourceElemType, targetElemType)),
      targetElemType);
  return getPrimaryCastValue(castOp);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createGather1DOp(
    PatternRewriter &rewriter, Location loc, Value source, Value indices) {
  Type sourceElemType = getElementTypeOrSelf(source.getType());
  Value init = utils::createEmptyOpWithTargetElemType(rewriter, loc, indices,
                                                      sourceElemType);
  return rewriter
      .create<VGatherOp>(loc, TypeRange(init.getType()), source, indices, init)
      .getResult()[0];
}

bool mlir::hivm::NormalizeTraitsBase::matchCastRoundMode(
    hivm::VCastOp op, CastRoundKind kind) {
  auto roundMode = mapCastRoundKindToRoundMode(kind);
  return roundMode && op.getRoundMode() == *roundMode;
}

bool mlir::hivm::NormalizeTraitsBase::matchCastUnsignedMode(
    hivm::VCastOp op, CastUnsignedModeKind kind) {
  return getSaturateUnsignedMode(op) == kind;
}

hivm::TypeFn mlir::hivm::NormalizeTraitsBase::mapCastSignKind(
    CastSignKind kind, hivm::TypeFn preserveTypeFn) {
  auto typeFn = mapCastSignKindToTypeFn(kind);
  return typeFn.value_or(preserveTypeFn);
}

hivm::RoundMode mlir::hivm::NormalizeTraitsBase::mapCastRoundKind(
    CastRoundKind kind, hivm::RoundMode defaultRoundMode) {
  auto roundMode = mapCastRoundKindToRoundMode(kind);
  return roundMode.value_or(defaultRoundMode);
}

hivm::UnsignedMode mlir::hivm::NormalizeTraitsBase::mapCastUnsignedModeKind(
    CastUnsignedModeKind kind, hivm::UnsignedMode preserveMode) {
  auto unsignedMode = mapCastUnsignedModeKindToUnsignedMode(kind);
  return unsignedMode.value_or(preserveMode);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createSelectOp(
    PatternRewriter &rewriter, Location loc, Value cond, Value a, Value b,
    Value dst) {
  return rewriter
      .create<hivm::VSelOp>(loc, mlir::TypeRange{dst.getType()},
                            mlir::ValueRange{cond, a, b},
                            mlir::ValueRange{dst}, mlir::Value{},
                            rewriter.getDenseI64ArrayAttr({}),
                            rewriter.getDenseI64ArrayAttr({}))
      .getResults()[0];
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createIsInfOp(
    PatternRewriter &rewriter, Location loc, Value y) {
  auto shapedType = cast<ShapedType>(y.getType());
  Type boolShapedType = shapedType.clone(rewriter.getI1Type());
  auto tensorType = cast<TensorType>(boolShapedType);
  auto emptyDst = rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                   tensorType.getElementType());
  return rewriter
      .create<hivm::VIsInfOp>(loc, mlir::TypeRange{emptyDst.getType()},
                              mlir::ValueRange{y}, mlir::ValueRange{emptyDst})
      .getResults()[0];
}

} // namespace mlir::hivm
