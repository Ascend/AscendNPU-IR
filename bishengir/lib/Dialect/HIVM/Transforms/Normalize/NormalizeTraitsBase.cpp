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
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/TypeUtilities.h"

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

using UnaryOpFn = Value (*)(PatternRewriter &, Location, Value, Value);

#define UNARY_OP_ENTRY(KIND, OP)                                                \
  { UnaryKind::KIND, createHIVMUnaryOp<OP> }

static const llvm::DenseMap<UnaryKind, UnaryOpFn> unaryOpMap = {
    UNARY_OP_ENTRY(Rec, hivm::VRecOp),
    UNARY_OP_ENTRY(Sqrt, hivm::VSqrtOp),
    UNARY_OP_ENTRY(Not, hivm::VNotOp),
};

CompareMode mapCompareKindToCompareMode(CompareKind kind) {
  static const llvm::DenseMap<CompareKind, CompareMode> compareKindMap = {
    {CompareKind::EQ, CompareMode::EQ},
    {CompareKind::NE, CompareMode::NE},
    {CompareKind::LT, CompareMode::LT},
    {CompareKind::GT, CompareMode::GT},
    {CompareKind::GE, CompareMode::GE},
    {CompareKind::LE, CompareMode::LE}
  };
  auto it = compareKindMap.find(kind);
  if (it == compareKindMap.end())
    llvm_unreachable("Unknown CompareKind");
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
  if (it == unaryOpMap.end()) {
    llvm_unreachable("unsupported unary kind");
  }
  return it->second(rewriter, loc, input, dst);
}
} // namespace mlir::hivm