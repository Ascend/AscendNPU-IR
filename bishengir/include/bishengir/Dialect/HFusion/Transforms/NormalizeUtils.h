//===- NormalizeUtils.h -----------------------------------------*- C++ -*-===//
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

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <variant>
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Mutex.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
namespace mlir::hfusion {

extern thread_local bool archIsRegbased;
extern thread_local bool archisAscend950;
extern thread_local bool archisAscend310B;
extern thread_local bool archisMembased;

enum class CalcMode { SIN, COS };

// AnnotationHelpers.cpp
template <typename OpType> std::optional<StringRef> getAnnotateOverflowMode(OpType op);

// CastHelpers.cpp
Value castInToF32ToOut(hfusion::CastOp &op, PatternRewriter &rewriter);
Value castU32ToI64ToF32(hfusion::CastOp &op, PatternRewriter &rewriter);
Value castU32ToI64ToF32ToOut(hfusion::CastOp &op, Type targetType,
                             PatternRewriter &rewriter);
Value castSrcToFp16ToTargetType(hfusion::CastOp &op, Type targetType,
                                PatternRewriter &rewriter);
Value castSrcTypeToI1ByVCmp(hfusion::CastOp &op, Type srcType,
                            PatternRewriter &rewriter);
Value castI8ToI64(hfusion::CastOp &op, PatternRewriter &rewriter);
hfusion::CastMode getCastMode(hfusion::CastOp op);
template <typename OpType> std::optional<bool> getAnnotateAttrBool(OpType op, StringRef attr);
LogicalResult handleSaturateOverFlowMode(hfusion::CastOp op,
                                         PatternRewriter &rewriter);
LogicalResult handleTruncOverFlowMode(hfusion::CastOp op,
                                      PatternRewriter &rewriter);
LogicalResult handleOverflowModeForSaturate(hfusion::CastOp op,
                                            PatternRewriter &rewriter,
                                            bool enableSaturate);
LogicalResult handleOverflowModeForTrunc(hfusion::CastOp op,
                                         PatternRewriter &rewriter);

// MathHelpers.cpp
Value norm(PatternRewriter &rewriter, Location loc, Value x, Value xRound,
           const llvm::SmallVector<double> &piApproParams,
           std::optional<float> offset = std::nullopt);
template <hfusion::TaylerMode taylerMode>
Value sign(PatternRewriter &rewriter, Location loc, Value x);
template <hfusion::TaylerMode taylerMode>
Value tayler(OpBuilder &b, Location loc, Value x, int taylerExpansionNum);
SmallVector<double> getTaylerParams(hfusion::TaylerMode taylerMode,
                                    int taylerExpansionNum);
Value getSinSign(PatternRewriter &rewriter, Location loc, Value x);
double getFPMAX(FloatType fType);
double getFPMIN(FloatType fType);
Value getAtanSign(PatternRewriter &rewriter, Location loc, Value x);
Value constructTaylerSeries(OpBuilder &b, Location loc, Value lastTaylerTerm,
                            Value emptyOp, Value xPow, int taylerExpansionNum,
                            const SmallVector<double> &taylerParams);
template <size_t N>
FailureOr<Value> emitGlobalTableFromUI32ArrayAndLoadAsTensorI32(
    PatternRewriter &rewriter, Operation *anchorOp, ModuleOp module,
    Location loc, llvm::StringRef globalName,
    const std::array<uint32_t, N> &arr);
Value buildSinOrCosCalc(OpBuilder &b, Location loc,
                               Value in,    // tensor<...xf32> any rank
                               Value tbl_t, // tensor<320xi32>
                               CalcMode mode);
FailureOr<Value> buildSinOrCos(PatternRewriter &rewriter,
                               hfusion::ElemwiseUnaryOp op, Value input,
                               CalcMode mode);
Value genPolyExpr(PatternRewriter &rewriter, Location loc,
                  const Value squareSrc, Value input,
                  const llvm::SmallVector<double> &numerCoeff,
                  bool enableLastMulTerm = true);

// ScalerHelpers.cpp
std::optional<Value> getScalarFromConstantOp(PatternRewriter &rewriter,
                                             Location loc,
                                             arith::ConstantOp constant);
std::optional<Value> singleElemDenseTensorToScalar(Value operand,
                                                   PatternRewriter &rewriter);

// TypeHelpers.cpp
bool isI1ElemType(Type type);
bool isI8ElemType(Type type);
bool isI16ElemType(Type type);
bool isF16ElemType(Type type);
template <typename srcType> bool isElemType(Type valueType);
bool hasI1ElemType(const SmallVector<Value> &values);
bool hasI8ElemType(const SmallVector<Value> &values);
bool hasI16ElemType(const SmallVector<Value> &values);
bool hasF16ElemType(const SmallVector<Value> &values);
bool allI1ElemType(const SmallVector<Value> &values);
bool allI8ElemType(const SmallVector<Value> &values);
bool allI16ElemType(const SmallVector<Value> &values);
template <typename srcType, typename targetType,
          typename = std::enable_if<(std::is_same_v<targetType, Float16Type> ||
                                     std::is_same_v<targetType, Float32Type>)>>
SmallVector<Value> normalizeSrcToTargetType(PatternRewriter &rewriter,
                                            const SmallVector<Value> &values,
                                            bool unsignedSrc = false);
Operation *createNewReduceOp(linalg::ReduceOp op, PatternRewriter &rewriter,
                             Type srcType, Type targetType,
                             SmallVector<Value> &newInputs,
                             SmallVector<Value> &newInits);
void replaceI8ResultsWithTargetType(const SmallVector<Value> &oldResults,
                                    const SmallVector<Value> &newResults,
                                    PatternRewriter &rewriter,
                                    bool enableOverflow = true,
                                    bool isUnsigned = false);
void replaceI16ResultsWithTargetType(const SmallVector<Value> &oldResults,
                                     const SmallVector<Value> &newResults,
                                     PatternRewriter &rewriter);

} // namespace mlir::hfusion