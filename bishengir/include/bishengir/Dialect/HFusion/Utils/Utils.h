//===- Utils.h ------------------------------------------------------------===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_UTILS_UTILS_H
#define BISHENGIR_DIALECT_HFUSION_UTILS_UTILS_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir {
namespace hfusion {

/// Create arith index cast op to cast value `v` to index type.
/// If `isUnsigned` is true, create `arith.index_castui`, otherwise create
/// `arith.index_cast`.
Value castToIndex(Value v, OpBuilder &opBuilder, bool isUnsigned = true);

/// Create `arith.index_cast` op to cast value index-typed value `v` to type
/// `t`.
/// If `isUnsigned` is true, create `arith.index_castui`, otherwise create
/// `arith.index_cast`.
Value castIndexTo(Value v, Type t, OpBuilder &opBuilder,
                  bool isUnsigned = true);

Operation *createCmpOp(PatternRewriter &rewriter, Location loc, Value lhs,
                       Value rhs, CompareFn cmpFn);

Operation *createVandOp(PatternRewriter &rewriter, Location loc, Value lhs,
                        Value rhs);
/// Tiling related utilities
namespace tiling {

/// Caller information.
struct CallerInfo {
  func::FuncOp caller;
  /// Callers original argument number.
  size_t callerOriginalArgNumber;
  /// Function called by the caller.
  func::FuncOp callee;
  /// Call sites within the caller calling callee.
  SmallVector<func::CallOp> callSites;
};

using CallSiteArgsBuilderFn = std::function<SmallVector<Value>(
    /*callSite=*/func::CallOp, OpBuilder &)>;

struct CallSiteBuilderInfo;
using CallSiteBuilderFn = std::function<LogicalResult(
    /*callSite=*/func::CallOp, OpBuilder &,
    /*newArgs=*/const SmallVector<Value> &,
    /*irMap=*/DenseMap<Operation *, Operation *> &)>;

LogicalResult callSiteBuilderFnForTilingModification(
    func::CallOp callSite, OpBuilder &opBuilder,
    const SmallVector<Value> &newArguments,
    DenseMap<Operation *, Operation *> &irMap);

/// Information needed to construct new callee.
struct CallSiteBuilderInfo {
  /// Function to create arguments for new call site.
  CallSiteArgsBuilderFn argBuilderFn;
  /// Function to create new call site.
  CallSiteBuilderFn siteBuilderFn;
};

/// Get callee's caller's information.
void getCallerInfo(func::FuncOp callee, ModuleOp enclosingModule,
                   DenseMap<func::FuncOp, CallerInfo> &info);

/// Get call site arguments that corresponds to tiling data arguments in callee.
SmallVector<Value> getCalleeTilingArguments(func::FuncOp callee,
                                            func::CallOp callSite);
/// Fix the call sites by replacing arguments.
LogicalResult doFixCallSite(CallerInfo &callerInfo,
                            CallSiteBuilderInfo &builderInfo,
                            DenseMap<Operation *, Operation *> &irMap,
                            OpBuilder &opBuilder);

/// Crosscheck tiling function call with tiling operands.
LogicalResult
checkCallCalcTilingWithTilingOperands(Operation *calcTilingOp,
                                      ArrayRef<Value> tilingOperands);

/// Tiling functions should only be returning i64.
LogicalResult verifyTilingFunc(func::FuncOp &tilingFunc);

/// Crosscheck device functions with the tiling function.
LogicalResult deviceFuncsMatchTilingFunc(SmallVector<func::FuncOp> &deviceFuncs,
                                         func::FuncOp &tilingFunc);
} // namespace tiling

namespace auto_schedule {
/// Generate payload tag from kernel name.
inline std::string getPayloadRootTag(const std::string &kernelName) {
  return kernelName + "_payload";
}

/// Generate transform tag from kernel name.
inline std::string getTransformRootTag(const std::string &kernelName) {
  return kernelName + "_transform";
}
} // namespace auto_schedule

/// Whether the operation is a `tensor.expand_shape`, `tensor.collapse_shape`.
bool isReshapeOp(Operation *op);

/// Whether the operation is a rehape op or slice op
bool isReshapeOrSliceOp(Operation *op);

/// Whether the operation is a tensor manipulation (pad, concat, slice).
bool isTensorManipulationOp(Operation *op);

bool isMatmulOps(Operation *op);

Value getReshapeSource(Operation *op);
Value getReshapeResult(Operation *op);

Value getReshapeOrSliceSource(Operation *op);
Value getReshapeOrSliceResult(Operation *op);

/// Trace back use-def chain to get the original value before reshape or slice.
FailureOr<Value> traceReshapeOrSliceSingleProducer(Value input);

/// Trace back use-def chain to get the original value before reshape or slice
/// if possible. Otherwise, return the input itself.
Value traceReshapeOrSliceSingleProducerOrSelf(Value input);

/// Trace back use-def chain to get the reshape or slice operations from current
/// input value to original value.
SmallVector<Operation *> getReshapeOrSliceOpProduceTrace(Value input);

/// Trace back use-def chain to get the original value before reshape.
FailureOr<Value> traceReshapeSingleProducer(Value input);

/// Trace back use-def chain to get the original value before reshape
/// if possible. Otherwise, return the input itself.
Value traceReshapeSingleProducerOrSelf(Value input);

/// Trace back use-def chain to get the reshape operations from current
/// input value to original value.
SmallVector<Operation *> getReshapeOpProduceTrace(Value input);

/// Trace the use-def chain to get the value after reshape or slice. The input
/// value should have only one RESHAPE consumer. (can have non-reshape user)
FailureOr<Value> traceReshapeOrSliceSingleConsumer(Value input);

/// Trace the use-def chain to get the value after reshape or slice if possible.
/// Otherwise, return the input itself. (can have non-reshape user)
Value traceReshapeOrSliceSingleConsumerOrSelf(Value input);

/// Trace the use-def chain to get the value after reshape or slice. The input
/// value should have only one consumer (can't have non-reshape user either).
FailureOr<Value> traceReshapeOrSliceOnlyOneUser(Value input);

/// Trace the use-def chain to get the value after reshape or slice if possible.
/// Otherwise, return the input itself. (can't have non-reshape user either).
Value traceReshapeOrSliceOnlyOneUserOrSelf(Value input);

/// Whether is scalar-vector binary op.
template <typename SrcOp>
bool isSVOp(SrcOp op) {
  llvm::SmallVector<Value> inputs = op.getDpsInputs();
  if (inputs.size() != 2) {
    return false;
  }
  return (inputs[0].getType().isIntOrFloat() &&
          llvm::isa<ShapedType>(inputs[1].getType()));
}

void trySetFusionKind(func::FuncOp func, const FusionKind &fusionKind);
std::optional<FusionKind> tryGetFusionKind(func::FuncOp func);

namespace reshape_utils {

bool isInitOp(Operation *op);

bool isReshapingOp(Operation *op);

bool isSlicingOp(Operation *op);

bool isArgOp(Operation *op);

bool isStopPropagatable(Operation *op);

bool isOutOp(Operation *op);

bool isUnsupportedOp(Operation *op);

bool isSkippableOp(Operation *op);

bool isExplicitlyAllowedCollapseOp(Operation *op);

bool isLegalOp(Operation *op);

bool isReturnOp(Operation *op);

bool isContainerAllocator(Operation *op);

bool isMarkedAsElementwiseOp(Operation *op);

bool isZeroDimensionOp(Operation *op);

bool isMarkedAsElementwiseUnaryOp(Operation *op);

bool isAllParallelOp(Operation *op);

} // namespace reshape_utils

void setInsertionPointBeforeOrAfter(OpBuilder &builder, Value &value,
                                    bool isAfter);

void setInsertionPointAfterValue(OpBuilder &builder, Value &value);

void setInsertionPointBeforeValue(OpBuilder &builder, Value &value);

std::optional<int64_t>
getFuncArgTiedResultReturnIdx(BlockArgument &ba, bool &funcArgIsReshaped,
                              bool &funcResultIsReshaped);

tensor::EmptyOp createEmptyOpWithSameShape(OpBuilder &rewriter, Value operand,
                                           SmallPtrSet<Operation *, 4> &newOps,
                                           Location loc);

hfusion::LoadOp createCacheRead(OpBuilder &rewriter, Value operand,
                                Location loc);

struct CacheWriteOptions {
  bool outputOnly;
  bool cacheWriteToOutputInit;
  /// For output only mode. Stores the reshape produce trace of return operands
  std::optional<SmallVector<Operation *>> reshapeTrace = std::nullopt;
};

FailureOr<hfusion::StoreOp> createCacheWrite(OpBuilder &rewriter,
                                             OpResult result,
                                             CacheWriteOptions options);

Value OverflowProcess(OpBuilder &builder, Value src, Type targetElemType);

/// clip input to range [lowerBound, upperBound]
Value ClipInput(PatternRewriter &rewriter, Location loc, Value input,
                double upperBound, double lowerBound);

// erase unused func args by attrs
BitVector eraseFuncArgsWithAttr(func::FuncOp &funcOp,
                                SmallVector<NamedAttribute> &attrs);

// erase unused func args except attrs
BitVector eraseFuncArgsExceptAttr(func::FuncOp &funcOp, NamedAttribute &attr);

SmallVector<Value> computeExtractCollapsedIndices(
    const SmallVector<ReassociationIndices> &reassociation,
    OperandRange &inputIndices, function_ref<Value(int idx)> getDimSize,
    OpBuilder &builder, Location loc);

std::optional<ArrayAttr> getSymbolicTensor(Type tensorType);

/// Perform specialized offset modification for ArangeOp when tiling.
void offsetArangeOp(OpBuilder &builder, Operation *tiledOp,
                    ArrayRef<OpFoldResult> offsets);

/// divide and cast the results to certain type with certain rounding mode
Value divWithRoundMode(OpBuilder &builder, Location loc, Type resType,
                       Value src0, Value src1, Value resTensor,
                       hfusion::RoundMode roundingMode,
                       std::optional<Operation **> divOp = std::nullopt);

namespace util {
constexpr static unsigned int VL = 256;
constexpr static unsigned int BL = VL / 8;
const static int vectorBlockSizeBit = 256;
const static int srcNumPerRepeatOfVBRCBIntrin = 8;

constexpr static unsigned int INTR_BYTES_PER_BLOCK = 32;
constexpr static unsigned int INTR_BYTES_PER_REPEAT = 256;
constexpr static unsigned int VNCHWCONV_INTR_BYTES_PER_REPEAT = 512;

/// Deduce Alignment information for DPS Op's init operand.
///
/// If operand has memref semantic, we try to deduce the information from the
/// memref type. Otherwise, we look for annotations on the tied result value. If
/// there is conflicting annotations, a warning is produced.
hivm::AlignKind deduceAlignmentForDPSInitOperand(OpOperand &operand);

hivm::AlignKind deduceAlignmentForMemRefType(MemRefType vecType);

bool hasDynamicShapeOperand(Operation *op);

} // namespace util

namespace trig {
const float PI_FOR_X_DIV = 3.1830987334251404e-01f;
const float ONE_OVER_2048 = 4.8828125000000000e-04f;
const float CONST_2048 = 2.0480000000000000e+03f;

const float HALF = 5.0e-01f;
const float FOUR = 4.0e+00f;
const float NEG_TWO = -2.0e+00f;
const float ONE = 1.0e+00f;

const float PI_0 = 3.1416015625000000e+00f;
const float PI_1 = -8.9071691036224365e-06f;
const float PI_2 = -1.7412276065442711e-09f;
const float PI_3 = 1.2446743939339977e-13f;

const float RES_MUL_SCA = 2.6049265215988271e-06f;
const float RES_ADD_UP = -1.9808944489341229e-04f;
const float COEF_2 = 8.3330497145652771e-03f;
const float COEF_3 = -1.6666658222675323e-01f;

const float COS_PI_DOWN = 1.57079637050628662e+00f;
const float COS_PI_RESDOWN_ADDS_NEG = -4.37113900018937e-08f;
const float COS_CLIP_LOW = -1.0f;
const float COS_CLIP_HIGH = 1.0f;
} // namespace trig
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_UTILS_UTILS_H
