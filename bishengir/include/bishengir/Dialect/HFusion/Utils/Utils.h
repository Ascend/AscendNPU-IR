//===- Utils.h ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
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

Value createEmptyLikeAnyShaped(OpBuilder &b, Location loc, Value a, Value bval);

Operation *createCmpOp(OpBuilder &b, Location loc, Value lhs, Value rhs,
                       CompareFn cmpFn);

Operation *createVandOp(OpBuilder &b, Location loc, Value lhs, Value rhs);
Operation *createLinalgElemwiseBinaryOp(OpBuilder &b, Location loc,
                                        linalg::BinaryFn fn, Value lhs,
                                        Value rhs);
Operation *createHFusionElemwiseBinaryOp(OpBuilder &b, Location loc,
                                         hfusion::BinaryFn fn, Value lhs,
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

bool isSimtOps(Operation *op);

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

/// Cast `src` value to the specified element type and rounding mode.
///
/// `src` can be either tensor or scalar.
/// If it's a scalar, casting is done by arith dialect ops.
/// If it's a tensor, casting is done by `hfusion.cast` op. If `dst` is not
/// provided, the init value is a `tensor.empty` op. Otherwise, it's written
/// to `dst`.
Value castTo(OpBuilder &builder, Value src, Type targetElemType,
             hfusion::RoundMode roundMode,
             std::optional<Value> dst = std::nullopt,
             bool enableOverflow = true, bool enableSaturate = false,
             hfusion::TypeFn castIntegerType = hfusion::TypeFn::cast_signed,
             hfusion::UnsignedMode unsignedMode = hfusion::UnsignedMode::SI2SI);

/// Cast `src` value to the specified element type.
/// Select rounding mode inside.
Value castTo(OpBuilder &builder, Value src, Type targetElemType,
             hfusion::TypeFn castIntegerType = hfusion::TypeFn::cast_signed);

// Checks if a linalg op has only one element
bool isSingleElementLinalgOp(linalg::LinalgOp op);

// Checks if a linalg op has only zero element
bool isZeroElementLinalgOp(linalg::LinalgOp op);

bool isFillOp(Operation *op);

// Check if the operation is certain ones (fill, transpose)
// that can be fused into a matmul
bool opCanFuseIntoMatmul(Operation *op);

// Check if tensor is a linalg.fill op with zero value or a tensor.empty op
bool isZeroOrEmptyTensor(Value op);

// Check if only unit dimensions are flattened
bool isOnlyUnitDimFlattened(ArrayRef<int64_t> oldShape,
                            ArrayRef<int64_t> newShape);
// Check if operations is in cube scope
bool isInCubeScope(Operation *op);

bool isFP8(Type type, Builder builder);

bool shouldUseTileReductionUsingForV2(OpBuilder &builder, Operation *Op);

namespace util {
const static std::string saturateSrcUnsigned = "saturate_src_unsigned";
const static std::string saturateDstUnsigned = "saturate_dst_unsigned";

constexpr static unsigned int VL = 256;
constexpr static unsigned int BL = VL / 8;
const static int vectorBlockSizeBit = 256;
const static int srcNumPerRepeatOfVBRCBIntrin = 8;

constexpr static unsigned int INTR_BYTES_PER_BLOCK = 32;
constexpr static unsigned int INTR_BYTES_PER_REPEAT = 256;
constexpr static unsigned int VNCHWCONV_INTR_BYTES_PER_REPEAT = 512;

bool isFromFunctionArg(mlir::Value v);

/// Deduce Alignment information for DPS Op's init operand.
///
/// If operand has memref semantic, we try to deduce the information from the
/// memref type. Otherwise, we look for annotations on the tied result value. If
/// there is conflicting annotations, a warning is produced.
hivm::AlignKind deduceAlignmentForDPSInitOperand(OpOperand &operand);

hivm::AlignKind deduceAlignmentForMemRefType(MemRefType vecType);

bool hasDynamicShapeOperand(Operation *op);

enum ArchType : int64_t {
  IsMemBased = 0,
  IsRegBased = 1,
};

bool hasUnpropagateableCase(Operation *op, bool skipScope);

} // namespace util
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_UTILS_UTILS_H
