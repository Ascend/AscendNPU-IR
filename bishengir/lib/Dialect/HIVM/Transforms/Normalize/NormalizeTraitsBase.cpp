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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
namespace mlir::hivm {

template <typename UnaryOp>
mlir::Value createHIVMUnaryOp(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value input,
                              mlir::Value dst) {
  return rewriter
      .create<UnaryOp>(loc, mlir::TypeRange{dst.getType()},
                       mlir::ValueRange{input}, mlir::ValueRange{dst})
      .getResults()[0];
}

template <typename BinaryOp>
mlir::Value createHIVMBinaryOp(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value lhs,
                               mlir::Value rhs, mlir::Value dst) {
  return rewriter
      .create<BinaryOp>(loc, mlir::TypeRange{dst.getType()},
                        mlir::ValueRange{lhs, rhs}, mlir::ValueRange{dst})
      .getResults()[0];
}

using UnaryOpFn = Value (*)(PatternRewriter &, Location, Value, Value);
using BinaryOpFn = Value (*)(PatternRewriter &, Location, Value, Value, Value);
using UnaryOpMatcherFn = bool (*)(Operation *);
using BinaryOpMatcherFn = bool (*)(Operation *);

template <typename OpType>
static bool matchHIVMOp(Operation *op) {
  auto typedOp = dyn_cast_or_null<OpType>(op);
  return typedOp && typedOp.hasPureTensorSemantics();
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
    {BinaryKind::Min, createHIVMBinaryOp<hivm::VMinOp>},
    {BinaryKind::Max, createHIVMBinaryOp<hivm::VMaxOp>},
    {BinaryKind::And, createHIVMBinaryOp<hivm::VAndOp>},
    {BinaryKind::MinSigned, createHIVMBinaryOp<hivm::VMinOp>},
    {BinaryKind::MaxSigned, createHIVMBinaryOp<hivm::VMaxOp>},
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
    {BinaryKind::Min, matchHIVMOp<hivm::VMinOp>},
    {BinaryKind::Max, matchHIVMOp<hivm::VMaxOp>},
    {BinaryKind::And, matchHIVMOp<hivm::VAndOp>},
    {BinaryKind::MinSigned, matchHIVMOp<hivm::VMinOp>},
    {BinaryKind::MaxSigned, matchHIVMOp<hivm::VMaxOp>},
};

bool mlir::hivm::NormalizeTraitsBase::matchOp(Operation *op, UnaryKind kind) {
  auto it = unaryOpMatcherMap.find(kind);
  if (it == unaryOpMatcherMap.end())
    llvm_unreachable("unsupported unary kind");
  return it->second(op);
}

bool mlir::hivm::NormalizeTraitsBase::matchOp(Operation *op, BinaryKind kind) {
  auto it = binaryOpMatcherMap.find(kind);
  if (it == binaryOpMatcherMap.end())
    llvm_unreachable("unsupported binary kind");
  return it->second(op);
}

CompareMode mapCompareKindToCompareMode(CompareKind kind) {
  static const llvm::DenseMap<CompareKind, CompareMode> compareKindMap = {
      {CompareKind::EQ, CompareMode::EQ},
      {CompareKind::NE, CompareMode::NE},
      {CompareKind::LT, CompareMode::LT},
      {CompareKind::GT, CompareMode::GT},
      {CompareKind::GE, CompareMode::GE},
      {CompareKind::LE, CompareMode::LE}};
  auto it = compareKindMap.find(kind);
  if (it == compareKindMap.end())
    llvm_unreachable("Unknown CompareKind");
  return it->second;
}

static hivm::RoundMode mapCastRoundKindToRoundMode(CastRoundKind kind) {
  static const llvm::DenseMap<CastRoundKind, hivm::RoundMode> castRoundKindMap = {
      {CastRoundKind::Round, hivm::RoundMode::ROUND},
      {CastRoundKind::Floor, hivm::RoundMode::FLOOR},
  };

  auto it = castRoundKindMap.find(kind);
  if (it == castRoundKindMap.end())
    llvm_unreachable("Unknown CastRoundKind");
  return it->second;
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createCmpOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst,
    CompareKind kind) {
  CompareMode cmpMode = mapCompareKindToCompareMode(kind);
  Type boolType = rewriter.getIntegerType(1);
  auto emptyOp = utils::createEmptyOpWithTargetElemType(rewriter, loc, input,
                                                        boolType);
  auto cmpOp = rewriter.create<VCmpOp>(loc, TypeRange(emptyOp),
                                       ValueRange({input, dst}),
                                       ValueRange(emptyOp), cmpMode);
  return cmpOp.getResult()[0];
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createUnaryOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst,
    UnaryKind kind) {
  auto it = unaryOpMap.find(kind);
  if (it == unaryOpMap.end())
    llvm_unreachable("unsupported unary kind");
  return it->second(rewriter, loc, input, dst);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createBinaryOp(
    PatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value dst,
    BinaryKind kind) {
  auto it = binaryOpMap.find(kind);
  if (it == binaryOpMap.end())
    llvm_unreachable("unsupported binary kind");
  return it->second(rewriter, loc, lhs, rhs, dst);
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createCastOp(
    PatternRewriter &rewriter, Location loc, Value input, Type targetElemType,
    CastRoundKind kind) {
  Type srcElemType = getElementTypeOrSelf(input.getType());
  hivm::RoundMode roundMode = mapCastRoundKindToRoundMode(kind);
  if ((srcElemType.isF16() || srcElemType.isBF16()) && targetElemType.isF32()) {
    // HIVM VCastOp only supports f16/bf16 -> f32 in rint mode.
    roundMode = hivm::RoundMode::RINT;
  }
  auto castOp = hivm::castTo(
      rewriter, loc, input, rewriter.getAttr<hivm::RoundModeAttr>(roundMode),
      targetElemType);
  return castOp->getResults().empty() ? castOp.getSingleDst()
                                      : castOp->getResults()[0];
}

mlir::Value mlir::hivm::NormalizeTraitsBase::createFillOp(
    PatternRewriter &rewriter, Location loc, Value input, Value dst) {
  if (isa<ShapedType>(input.getType()) || !isa<TensorType>(dst.getType()))
    llvm_unreachable(
        "NormalizeTraitsBase::createFillOp only supports scalar-to-tensor fills");
  return rewriter
      .create<hivm::VBrcOp>(loc, TypeRange(dst.getType()), input, dst)
      .getResult()[0];
}
mlir::Value mlir::hivm::NormalizeTraitsBase::createBitcastOp(
    PatternRewriter &rewriter, Location loc, Type resultType, Value source) {
  return rewriter.create<BitcastOp>(loc, resultType, source).getResult();
}
} // namespace mlir::hivm
