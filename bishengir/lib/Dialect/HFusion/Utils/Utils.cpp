//===-----------------------------Utils.cpp--------------------------------===//
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

#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>
#include <unordered_set>

using namespace mlir;
using namespace mlir::hfusion;

#define DEBUG_TYPE "hfusion-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::utils::debugger;

/// int n:
///   (1) b = a + 2^(n-1)
///   (2) d = c % 2^(n)
///   (3) res = d - 2^(n-1)
/// uint n: d = c % 2^(n)
Value hfusion::OverflowProcess(OpBuilder &builder, Value src,
                               Type targetElemType) {
  if (!getElementTypeOrSelf(targetElemType).isInteger()) {
    llvm_unreachable("unsupport int type");
  }
  uint32_t bits = targetElemType.getIntOrFloatBitWidth();
  double MAXINT = static_cast<double>(1ULL << bits);
  double NEGINTOFF = -static_cast<double>(1ULL << (bits - 1));
  double POSINTOFF = static_cast<double>(1ULL << (bits - 1));

  Location loc = src.getLoc();
  Type srcElemType = getElementTypeOrSelf(src);
  Value targetTensor =
      utils::createEmptyOpWithTargetElemType(builder, loc, src, srcElemType);

  Value tmp = targetTensor;

  if (targetElemType.isSignlessInteger()) {
    ///  (1) b = a + 2^(n-1)
    auto posOffset = builder.create<arith::ConstantOp>(
        loc, srcElemType, builder.getFloatAttr(srcElemType, POSINTOFF));
    auto addOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            builder, loc, linalg::BinaryFn::add,
            ValueRange{src, posOffset->getResults()[0]}, targetTensor);
    tmp = addOp->getResults()[0];
  }
  /// (2) d = c % 2^(n)
  auto modCoff = builder.create<arith::ConstantOp>(
      loc, srcElemType, builder.getFloatAttr(srcElemType, MAXINT));
  auto modOp =
      hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                              hfusion::BinaryFnAttr>(
          builder, loc, hfusion::BinaryFn::mod,
          ValueRange{tmp, modCoff->getResults()[0]}, targetTensor);
  Value res = modOp->getResults()[0];

  if (targetElemType.isSignlessInteger()) {
    /// (3) d - 2^(n-1)
    auto negOffset = builder.create<arith::ConstantOp>(
        loc, srcElemType, builder.getFloatAttr(srcElemType, NEGINTOFF));
    auto addOp =
        hfusion::createBinaryOp<linalg::ElemwiseBinaryOp, linalg::BinaryFn,
                                linalg::BinaryFnAttr>(
            builder, loc, linalg::BinaryFn::add,
            ValueRange{modOp->getResults()[0], negOffset->getResults()[0]},
            targetTensor);
    res = addOp->getResults()[0];
  }
  return res;
}

// TODO: Refactor ArithToHFusion pass to use this util

Value hfusion::castToIndex(Value v, OpBuilder &opBuilder, bool isUnsigned) {
  return isUnsigned
             ? opBuilder
                   .create<arith::IndexCastUIOp>(v.getLoc(),
                                                 opBuilder.getIndexType(), v)
                   ->getResult(0)
             : opBuilder
                   .create<arith::IndexCastOp>(v.getLoc(),
                                               opBuilder.getIndexType(), v)
                   ->getResult(0);
}

Value hfusion::castIndexTo(Value v, Type t, OpBuilder &opBuilder,
                           bool isUnsigned) {
  return isUnsigned ? opBuilder.create<arith::IndexCastUIOp>(v.getLoc(), t, v)
                          ->getResult(0)
                    : opBuilder.create<arith::IndexCastOp>(v.getLoc(), t, v)
                          ->getResult(0);
}

Operation *hfusion::createCmpOp(PatternRewriter &rewriter, Location loc,
                                Value lhs, Value rhs, CompareFn cmpFn) {
  Type boolType = rewriter.getIntegerType(1);
  auto cmpInit =
      utils::createEmptyOpWithTargetElemType(rewriter, loc, lhs, boolType);
  auto cmpPredicateAttr = rewriter.getAttr<hfusion::CompareFnAttr>(cmpFn);
  auto cmpModeAttr = rewriter.getNamedAttr(
      hfusion::CompareFnAttr::getMnemonic(), cmpPredicateAttr);
  return rewriter.create<hfusion::CompareOp>(
      loc, TypeRange(cmpInit), ValueRange({lhs, rhs}), ValueRange(cmpInit),
      ArrayRef{cmpModeAttr});
}

Operation *hfusion::createVandOp(PatternRewriter &rewriter, Location loc,
                                 Value lhs, Value rhs) {
  auto andEmptyOp = utils::createEmptyOp(rewriter, loc, rhs);
  return hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                 hfusion::BinaryFnAttr>(
      rewriter, loc, hfusion::BinaryFn::vand, ValueRange({lhs, rhs}),
      ValueRange{andEmptyOp});
}

bool isConstantAllOne(Value v) {
  auto type = getElementTypeOrSelf(v);
  if (isa<FloatType>(type)) {
    return false;
  }

  APInt value;
  if (matchPattern(v, m_ConstantInt(&value))) {
    // match true for i1 type, match -1 for i8/i16/i32/i64 type
    return value.isAllOnes() ? true : false;
  }

  auto defineOp = v.getDefiningOp();
  if (!defineOp) {
    return false;
  }

  auto resIndx = cast<OpResult>(v).getResultNumber();
  if (auto fillOp = dyn_cast<linalg::FillOp>(defineOp)) {
    return isConstantAllOne(fillOp.getOperand(resIndx));
  }

  if (auto castOp = dyn_cast<hfusion::CastOp>(defineOp)) {
    return isConstantAllOne(castOp.getOperand(resIndx));
  }

  return false;
}

LogicalResult hfusion::simplifyVxorToVnot(PatternRewriter &rewriter,
                                          hfusion::ElemwiseBinaryOp op) {
  if (isConstantAllOne(op.getOperand(1))) {
    // XOR with all ones values can be optimized as NOT operation
    auto vnotOp = hfusion::createUnaryOp<hfusion::ElemwiseUnaryOp, hfusion::UnaryFn, hfusion::UnaryFnAttr> (
      rewriter, op.getLoc(), hfusion::UnaryFn::vnot,
      ValueRange(op.getOperand(0)), ValueRange(op.getOperand(0))
    );

    rewriter.replaceOp(op, vnotOp);
    return success();
  }

  return failure();
}

void tiling::getCallerInfo(func::FuncOp callee, ModuleOp enclosingModule,
                           DenseMap<func::FuncOp, CallerInfo> &info) {
  std::optional<SymbolTable::UseRange> maybeUses =
      callee.getSymbolUses(enclosingModule);
  if (!maybeUses.has_value())
    return;

  for (SymbolTable::SymbolUse use : maybeUses.value()) {
    func::CallOp callSite = cast<func::CallOp>(use.getUser());
    auto callerOp = callSite->getParentOfType<func::FuncOp>();
    assert(callerOp != nullptr && "Caller should not be empty!");
    auto &callerInfo = info[callerOp];
    callerInfo.caller = callerOp;
    callerInfo.callerOriginalArgNumber = callerOp.getNumArguments();
    callerInfo.callee = callee;
    callerInfo.callSites.push_back(callSite);
  }
}

SmallVector<Value> tiling::getCalleeTilingArguments(func::FuncOp callee,
                                                    func::CallOp callSite) {
  SmallVector<Value> tilingOperands;
  for (const auto [idx, operand] : llvm::enumerate(callSite.getOperands())) {
    if (hacc::utils::isTilingArg(callee, idx))
      tilingOperands.push_back(operand);
  }
  return tilingOperands;
}

LogicalResult tiling::doFixCallSite(tiling::CallerInfo &callerInfo,
                                    tiling::CallSiteBuilderInfo &builderInfo,
                                    DenseMap<Operation *, Operation *> &irMap,
                                    OpBuilder &opBuilder) {
  for (func::CallOp callSite : callerInfo.callSites) {
    LDBG("fixing call site: " << *callSite);
    auto newArgs = builderInfo.argBuilderFn(callSite, opBuilder);
    auto tilingArgs =
        tiling::getCalleeTilingArguments(callerInfo.callee, callSite);
    // If the arguments in the callee is the result of calling host tiling func,
    // check the validity.
    for (auto arg : tilingArgs) {
      auto calcTilingOp = arg.getDefiningOp<func::CallOp>();
      if (calcTilingOp && failed(checkCallCalcTilingWithTilingOperands(
                              calcTilingOp, tilingArgs))) {
        return failure();
      }
    }
    opBuilder.setInsertionPoint(callSite);
    if (failed(builderInfo.siteBuilderFn(callSite, opBuilder, newArgs, irMap)))
      return failure();
  }
  return success();
}

LogicalResult tiling::callSiteBuilderFnForTilingModification(
    func::CallOp callSite, OpBuilder &opBuilder,
    const SmallVector<Value> &newArguments,
    DenseMap<Operation *, Operation *> &irMap) {
  func::CallOp newCallSite = opBuilder.create<func::CallOp>(
      callSite.getLoc(), callSite.getResultTypes(), callSite.getCallee(),
      newArguments);
  LDBG("Generated new call site:\n" << *newCallSite);
  irMap.insert(std::make_pair(callSite, newCallSite));
  return success();
}

LogicalResult
tiling::checkCallCalcTilingWithTilingOperands(Operation *calcTilingOp,
                                              ArrayRef<Value> tilingOperands) {
  assert(!tilingOperands.empty() && isa<func::CallOp>(calcTilingOp));
  assert(calcTilingOp->getNumResults() == tilingOperands.size());
  for (auto [idx, res] : llvm::enumerate(calcTilingOp->getResults())) {
    if (res != tilingOperands[idx]) {
      return calcTilingOp->emitError(
          "Calc tiling order and usage inconsistency");
    }
    if (!res.getType().isInteger(64)) {
      return calcTilingOp->emitError("Non i64 calculate tiling return type");
    }
  }
  return success();
}

LogicalResult tiling::verifyTilingFunc(func::FuncOp &tilingFunc) {
  // verify tiling func's return type
  for (auto [idx, res] :
       llvm::enumerate(tilingFunc.getFunctionType().getResults())) {
    if (!res.isInteger(64))
      return tilingFunc.emitError("Non i64 calculate tiling return type");

    // Check that the result is annotated with tiling key/tiling data
    // attributes.
    auto resTypeAttr = tilingFunc.getResultAttrOfType<hacc::KernelArgTypeAttr>(
        idx, hacc::KernelArgTypeAttr::name);
    if (!resTypeAttr)
      return tilingFunc.emitError("Result is not annotated with attributes");

    auto resultType = resTypeAttr.getArgType();
    // This is the constraint that the first tiling data has to be a tiling key.
    if (idx == 0 && resultType != hacc::KernelArgType::kTilingKey)
      return tilingFunc.emitError("The first result is not a tiling key");

    if (idx != 0 && resultType != hacc::KernelArgType::kTilingData)
      return tilingFunc.emitError("Result is not a tiling data");
  }
  return success();
}

LogicalResult
tiling::deviceFuncsMatchTilingFunc(SmallVector<func::FuncOp> &deviceFuncs,
                                   func::FuncOp &tilingFunc) {
  if (failed(tiling::verifyTilingFunc(tilingFunc)))
    return failure();

  // verify the number of tiling data and the signature of all the device funcs
  auto goldenDeviceFuncTy = deviceFuncs.front().getFunctionType();
  for (auto [idx, deviceFunc] : llvm::enumerate(deviceFuncs)) {
    if (deviceFunc.getFunctionType() != goldenDeviceFuncTy)
      return deviceFunc.emitError("Device funcs' signature inconsistency");

    int tilingDataDeviceCount = 0;
    for (auto [idxArg, devArg] : llvm::enumerate(deviceFunc.getArguments())) {
      if (hacc::utils::isTilingArg(deviceFunc, idxArg)) {
        tilingDataDeviceCount++;
        if (!devArg.getType().isInteger(64))
          return deviceFunc.emitError("Non i64 device tiling data args");
      }
    }
    if (tilingDataDeviceCount != static_cast<int>(tilingFunc.getNumResults())) {
      return tilingFunc.emitError("Calc tiling order and usage inconsistency");
    }
  }
  return success();
}

bool hfusion::isReshapeOp(Operation *op) {
  if (!op)
    return false;
  return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op);
}

bool hfusion::isReshapeOrSliceOp(Operation *op) {
  if (!op)
    return false;
  return hfusion::isReshapeOp(op) || reshape_utils::isSlicingOp(op);
}

bool hfusion::isTensorManipulationOp(Operation *op) {
  if (!op)
    return false;
  return isa<tensor::PadOp, tensor::ConcatOp, tensor::ExtractSliceOp,
             tensor::InsertSliceOp, hfusion::InterleaveOp,
             hfusion::DeinterleaveOp>(op);
}

bool hfusion::isMatmulOps(Operation *op) {
  return isa<linalg::MatmulOp>(op) || isa<linalg::MatmulTransposeAOp>(op) ||
         isa<linalg::MatmulTransposeBOp>(op);
}

Value hfusion::getReshapeSource(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getSrc(); })
      .Case([](tensor::CollapseShapeOp collapse) { return collapse.getSrc(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported reshape op");
        return Value();
      });
}

Value hfusion::getReshapeResult(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getResult(); })
      .Case(
          [](tensor::CollapseShapeOp collapse) { return collapse.getResult(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported reshape op");
        return Value();
      });
}

Value hfusion::getReshapeOrSliceSource(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getSrc(); })
      .Case([](tensor::CollapseShapeOp collapse) { return collapse.getSrc(); })
      .Case([](tensor::ExtractSliceOp extract) { return extract.getSource(); })
      .Case([](tensor::InsertSliceOp insert) { return insert.getSource(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported reshape or slice op");
        return Value();
      });
}

Value hfusion::getReshapeOrSliceResult(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getResult(); })
      .Case(
          [](tensor::CollapseShapeOp collapse) { return collapse.getResult(); })
      .Case([](tensor::ExtractSliceOp extract) { return extract.getResult(); })
      .Case([](tensor::InsertSliceOp insert) { return insert.getResult(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported reshape or slice op");
        return Value();
      });
}

// TODO: refactor this function with ReshapeAnalyzer
Value hfusion::traceReshapeOrSliceSingleProducerOrSelf(Value input) {
  auto maybeValue = hfusion::traceReshapeOrSliceSingleProducer(input);
  if (succeeded(maybeValue))
    return maybeValue.value();
  return input;
}

// TODO: refactor this function with ReshapeAnalyzer
FailureOr<Value> hfusion::traceReshapeOrSliceSingleProducer(Value input) {
  LDBG("Tracing reshape single producer for " << input);
  if (isa<BlockArgument>(input)) {
    LDBG("Input is a block argument");
    return failure();
  }

  auto result = cast<OpResult>(input);
  auto *definingOp = result.getOwner();
  if (!hfusion::isReshapeOrSliceOp(definingOp)) {
    LDBG("Defining op is not reshape");
    return failure();
  }

  auto reshapeSource = hfusion::getReshapeOrSliceSource(definingOp);
  return hfusion::traceReshapeOrSliceSingleProducerOrSelf(reshapeSource);
}

// TODO: this function is deprecated, refactor with ReshapeAnalyzer
SmallVector<Operation *> hfusion::getReshapeOrSliceOpProduceTrace(Value input) {
  Operation *curOp = input.getDefiningOp();
  SmallVector<Operation *> trace;
  while (hfusion::isReshapeOrSliceOp(curOp)) {
    trace.push_back(curOp);
    Value reshapeSrc = hfusion::getReshapeOrSliceSource(curOp);
    if (isa<BlockArgument>(reshapeSrc)) {
      break;
    }
    curOp = reshapeSrc.getDefiningOp();
  };
  return trace;
}

// TODO: this function is deprecated, refactor with ReshapeAnalyzer
Value hfusion::traceReshapeSingleProducerOrSelf(Value input) {
  auto maybeValue = hfusion::traceReshapeSingleProducer(input);
  if (succeeded(maybeValue))
    return maybeValue.value();
  return input;
}

// TODO: this function is deprecated, refactor with ReshapeAnalyzer
FailureOr<Value> hfusion::traceReshapeSingleProducer(Value input) {
  LDBG("Tracing reshape single producer for " << input);
  if (isa<BlockArgument>(input)) {
    LDBG("Input is a block argument");
    return failure();
  }

  auto result = cast<OpResult>(input);
  auto *definingOp = result.getOwner();
  if (!hfusion::isReshapeOp(definingOp)) {
    LDBG("Defining op is not reshape");
    return failure();
  }

  auto reshapeSource = hfusion::getReshapeSource(definingOp);
  return hfusion::traceReshapeSingleProducerOrSelf(reshapeSource);
}

// TODO: this function is deprecated, refactor with ReshapeAnalyzer
SmallVector<Operation *> hfusion::getReshapeOpProduceTrace(Value input) {
  Operation *curOp = input.getDefiningOp();
  SmallVector<Operation *> trace;
  while (hfusion::isReshapeOp(curOp)) {
    trace.push_back(curOp);
    Value reshapeSrc = hfusion::getReshapeSource(curOp);
    if (isa<BlockArgument>(reshapeSrc)) {
      break;
    }
    curOp = reshapeSrc.getDefiningOp();
  };
  return trace;
}

// TODO: this function is deprecated, refactor with ReshapeAnalyzer
Value hfusion::traceReshapeOrSliceSingleConsumerOrSelf(Value input) {
  auto maybeValue = hfusion::traceReshapeOrSliceSingleConsumer(input);
  if (succeeded(maybeValue))
    return maybeValue.value();
  return input;
}

// TODO: this function is deprecated, refactor with ReshapeAnalyzer
FailureOr<Value> hfusion::traceReshapeOrSliceSingleConsumer(Value input) {
  LDBG("Tracing reshape or slice single consumer for " << input);
  auto reshapeUsers =
      llvm::make_filter_range(input.getUsers(), [&](Operation *user) {
        return hfusion::isReshapeOrSliceOp(user);
      });
  if (!llvm::hasSingleElement(reshapeUsers)) {
    LDBG("Input has none or more than one reshape users");
    return failure();
  }

  auto *singleReshape = *(reshapeUsers.begin());
  auto result = hfusion::getReshapeOrSliceResult(singleReshape);
  return hfusion::traceReshapeOrSliceSingleConsumerOrSelf(result);
}

// TODO: this function is deprecated, refactor with ReshapeAnalyzer
Value hfusion::traceReshapeOrSliceOnlyOneUserOrSelf(Value input) {
  auto maybeValue = hfusion::traceReshapeOrSliceOnlyOneUser(input);
  if (succeeded(maybeValue))
    return maybeValue.value();
  return input;
}

// TODO: this function is deprecated, refactor with ReshapeAnalyzer
FailureOr<Value> hfusion::traceReshapeOrSliceOnlyOneUser(Value input) {
  LDBG("Tracing reshape or slice only one user for " << input);
  auto nonDimUsers =
      llvm::make_filter_range(input.getUsers(), [&](Operation *user) {
        return !llvm::isa<tensor::DimOp>(user);
      });
  if (!llvm::hasSingleElement(nonDimUsers)) {
    LDBG("Input has none or more than one user");
    return failure();
  }
  auto *user = *(nonDimUsers.begin());
  auto reshapeUser = hfusion::isReshapeOrSliceOp(user);
  if (!reshapeUser) {
    LDBG("Input single user is not reshape");
    return failure();
  }

  auto result = hfusion::getReshapeOrSliceResult(user);
  return hfusion::traceReshapeOrSliceOnlyOneUserOrSelf(result);
}

std::optional<FusionKind> hfusion::tryGetFusionKind(func::FuncOp func) {
  auto fusionKindAttr =
      func->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
  if (!fusionKindAttr)
    return std::nullopt;
  return fusionKindAttr.getFusionKind();
}

void hfusion::trySetFusionKind(func::FuncOp func,
                               const FusionKind &fusionKind) {
  if (func->hasAttr(FusionKindAttr::name)) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Function already has a fusionKind, replacing with: "
                   << fusionKind << "\n";);
  }
  func->setAttr(FusionKindAttr::name,
                FusionKindAttr::get(func->getContext(), fusionKind));
  return;
}

namespace mlir {
namespace hfusion {
namespace reshape_utils {

bool isInitOp(Operation *op) { return mlir::reshape_utils::isInitOp(op); }

bool isReshapingOp(Operation *op) {
  return mlir::reshape_utils::isReshapingOp(op);
}

bool isSlicingOp(Operation *op) { return mlir::reshape_utils::isSlicingOp(op); }

bool isArgOp(Operation *op) { return mlir::reshape_utils::isArgOp(op); }

bool isStopPropagatable(Operation *op) {
  return mlir::reshape_utils::isStopPropagatable(op);
}

bool isOutOp(Operation *op) { return mlir::reshape_utils::isOutOp(op); }

bool isUnsupportedOp(Operation *op) {
  return mlir::reshape_utils::isUnsupportedOp(op);
}

bool isSkippableOp(Operation *op) {
  return mlir::reshape_utils::isSkippableOp(op);
}

bool isExplicitlyAllowedCollapseOp(Operation *op) {
  return mlir::reshape_utils::isExplicitlyAllowedCollapseOp(op);
}

bool isContainerAllocator(Operation *op) {
  return mlir::reshape_utils::isContainerAllocator(op);
}

bool isMarkedAsElementwiseOp(Operation *op) {
  // This would handle scalar as well
  return mlir::reshape_utils::isMarkedAsElementwiseOp(op) ||
         isa_and_present<hfusion::ElemwiseUnaryOp, hfusion::ElemwiseBinaryOp,
                         hfusion::CompareOp, hfusion::CastOp, hfusion::SelectOp,
                         hfusion::BitcastOp>(op);
}

bool isZeroDimensionOp(Operation *op) {
  return mlir::reshape_utils::isZeroDimensionOp(op);
}

bool isMarkedAsElementwiseUnaryOp(Operation *op) {
  // This would handle scalar as well
  return mlir::reshape_utils::isMarkedAsElementwiseUnaryOp(op) ||
         isa_and_present<hfusion::CastOp>(op);
}

bool isAllParallelOp(Operation *op) {
  return mlir::reshape_utils::isAllParallelOp(op);
}

// TODO: Need to refactor this.
bool isLegalOp(Operation *op) {
  return isa<hfusion::ElemwiseUnaryOp, hfusion::ElemwiseBinaryOp,
             hfusion::CompareOp, hfusion::CastOp, hfusion::SelectOp,
             hfusion::ReduceWithIndexOp>(op) ||
         mlir::reshape_utils::isLegalOp(op);
}

bool isReturnOp(Operation *op) { return mlir::reshape_utils::isReturnOp(op); }

} // namespace reshape_utils

namespace util {

hivm::AlignKind deduceAlignmentForDPSInitOperand(OpOperand &operand) {
  Value operandValue = operand.get();
  MemRefType maybeMemRefType = dyn_cast<MemRefType>(operandValue.getType());
  if (maybeMemRefType)
    return deduceAlignmentForMemRefType(maybeMemRefType);

  // Try deduce alignment kind for tensor.
  hivm::AlignKind alignKind{hivm::AlignKind::UNKNOWN};
  auto owner = dyn_cast<DestinationStyleOpInterface>(operand.getOwner());
  if (!owner)
    return alignKind;

  // If tied result is tagged with alignment info, return it as it is.
  Value tiedResult = owner.getTiedOpResult(&operand);
  auto markOpsWithAlignmentInfo =
      llvm::make_filter_range(tiedResult.getUsers(), [](Operation *user) {
        return isa<annotation::MarkOp>(user) &&
               user->hasAttrOfType<hivm::AlignKindAttr>(
                   hivm::AlignKindAttr::name);
      });
  if (markOpsWithAlignmentInfo.empty())
    return alignKind;

  auto alignmentInfo =
      llvm::map_to_vector<1>(markOpsWithAlignmentInfo, [](Operation *markOp) {
        return markOp
            ->getAttrOfType<hivm::AlignKindAttr>(hivm::AlignKindAttr::name)
            .getValue();
      });
  if (!llvm::all_equal(alignmentInfo)) {
    LDBG("WARNING: Conflicting alignment annotation for operand #"
         << operand.getOperandNumber() << " in " << *owner);
    return hivm::AlignKind::UNKNOWN;
  }
  return alignmentInfo.front();
}

hivm::AlignKind deduceAlignmentForMemRefType(MemRefType vecType) {
  Type eleType = vecType.getElementType();
  int eleSize = static_cast<int>(eleType.getIntOrFloatBitWidth() / 8);

  hivm::AlignKind alignKind{hivm::AlignKind::UNKNOWN};
  int64_t toCheck{0};

  StridedLayoutAttr dstLayout =
      dyn_cast<StridedLayoutAttr>(vecType.getLayout());
  if (dstLayout) {
    ArrayRef<int64_t> strides = dstLayout.getStrides();
    if (strides.size() <
        2) { // if strides is less than 2, alignment is impossible
      return hivm::AlignKind::UNKNOWN;
    }

    toCheck = strides[strides.size() - 2]; // get the 2nd last strides
  } else {
    int rank = vecType.getRank();
    if (rank == 0) {
      return hivm::AlignKind::UNKNOWN;
    }

    toCheck = vecType.getDimSize(rank - 1);
  }

  if (toCheck != ShapedType::kDynamic) {
    auto isAlignedToBlock = [](int eleNum, int eleSize) {
      return eleNum * eleSize % BL == 0;
    };
    if (isAlignedToBlock(toCheck, eleSize)) {
      alignKind = hivm::AlignKind::ALIGN;
    } else {
      alignKind = hivm::AlignKind::UNALIGNED;
    }
  } else {
    alignKind = hivm::AlignKind::UNKNOWN;
  }

  return alignKind;
}

bool hasDynamicShapeOperand(Operation *op) {
  for (const auto operand : op->getOperands()) {
    if (const TensorType &tensorType = dyn_cast<TensorType>(operand.getType()))
      if (ShapedType::isDynamicShape(tensorType.getShape()))
        return true;
  }
  return false;
}

} // namespace util
} // namespace hfusion
} // namespace mlir

void hfusion::setInsertionPointBeforeOrAfter(OpBuilder &builder, Value &value,
                                             bool isAfter) {
  if (BlockArgument blockArg = dyn_cast<BlockArgument>(value)) {
    LLVM_DEBUG(llvm::dbgs() << "here set\n";);
    // If it's a block argument, set insertion point to the start of the block
    builder.setInsertionPointToStart(blockArg.getOwner());
    return;
  }
  Operation *definingOp = value.getDefiningOp();
  if (definingOp != nullptr) {
    if (isAfter)
      builder.setInsertionPointAfter(definingOp);
    else
      builder.setInsertionPoint(definingOp);
    return;
  }
  LLVM_DEBUG(
      llvm::dbgs() << "Warning: Non-block argument with no defining op\n");
  Region *parentRegion = value.getParentRegion();
  if (parentRegion == nullptr) {
    return;
  }
  Operation *parentOp = parentRegion->getParentOp();
  if (parentOp == nullptr) {
    return;
  }
  if (!parentOp->getRegions().empty() && !parentOp->getRegion(0).empty()) {
    builder.setInsertionPointToStart(&parentOp->getRegion(0).front());
  }
}

void hfusion::setInsertionPointAfterValue(OpBuilder &builder, Value &value) {
  setInsertionPointBeforeOrAfter(builder, value, true);
}

void hfusion::setInsertionPointBeforeValue(OpBuilder &builder, Value &value) {
  setInsertionPointBeforeOrAfter(builder, value, false);
}

/// If the function argument is used as the dps init operand
/// of linalg/hfusion ops, and the tied result value is returned from from
/// the function, return its result index.
/// This function will also consider the following cases:
///   1) The input argument is reshaped before use
///   2) The result tied to the init operand is reshaped before return
///
/// If the function argument is "tied to" multiple return values, only
/// the first index will be returned.
///
/// For example:
/// ```mlir
///    func.func @foo(%arg0, %arg1)
///      %ret0:N = linalg.ops ins(...) outs(%arg1, ...)
///      func.return %some_value, %ret#0
///  ```
/// The result is 1 (start counting from zero).
std::optional<int64_t> hfusion::getFuncArgTiedResultReturnIdx(
    BlockArgument &ba, bool &funcArgIsReshaped, bool &funcResultIsReshaped) {
  auto maybeArgReshaped = hfusion::traceReshapeOrSliceSingleConsumerOrSelf(ba);
  if (hfusion::isReshapeOrSliceOp(maybeArgReshaped.getDefiningOp()))
    funcArgIsReshaped = true;

  for (OpOperand &use : maybeArgReshaped.getUses()) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!linalgOp)
      continue;

    if (!linalgOp.isDpsInit(&use))
      continue;

    // Check to see if tied result is used by `func.return`
    auto tiedResult = linalgOp.getTiedOpResult(&use);
    auto reshapeOrSelf =
        hfusion::traceReshapeOrSliceSingleConsumerOrSelf(tiedResult);
    if (hfusion::isReshapeOrSliceOp(reshapeOrSelf.getDefiningOp()))
      funcResultIsReshaped = true;

    auto maybeOperandOfReturnOp =
        llvm::find_if(reshapeOrSelf.getUses(), [&](OpOperand &operand) {
          return isa<func::ReturnOp>(operand.getOwner());
        });
    if (maybeOperandOfReturnOp == reshapeOrSelf.getUses().end())
      continue;

    // Return the index of the operand in `func.return`
    return static_cast<int64_t>(maybeOperandOfReturnOp->getOperandNumber());
  }
  return std::nullopt;
}

/// Use `operand`'s shape information to create an `tensor.empty` op
/// with the exact same shape.
tensor::EmptyOp
hfusion::createEmptyOpWithSameShape(OpBuilder &rewriter, Value operand,
                                    SmallPtrSet<Operation *, 4> &newOps,
                                    Location loc) {
  auto tensorType = cast<TensorType>(operand.getType());
  ArrayRef<int64_t> staticShapes = tensorType.getShape();
  llvm::SmallVector<Value, 2> dynamicSizes;
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      auto dynDim = rewriter.create<tensor::DimOp>(loc, operand, i);
      newOps.insert(dynDim.getOperation());
      dynamicSizes.push_back(dynDim);
    }
  }
  return rewriter.create<tensor::EmptyOp>(
      loc, staticShapes, tensorType.getElementType(), dynamicSizes);
}

/// Use `result`'s reified shape information to create an `tensor.empty` op
/// with the exact same shape.
tensor::EmptyOp
createEmptyOpWithReifiedShape(OpBuilder &rewriter, Operation *definingOp,
                              Value result, SmallPtrSet<Operation *, 4> &newOps,
                              Location loc) {
  // Compute the reified return shape of the target `result`
  auto reifiableOp = cast<ReifyRankedShapedTypeOpInterface>(definingOp);
  ReifiedRankedShapedTypeDims reifiedReturnShapes;
  if (failed(reifiableOp.reifyResultShapes(rewriter, reifiedReturnShapes))) {
    return nullptr;
  }
  auto resultIndex = cast<OpResult>(result).getResultNumber();
  auto foldResult = reifiedReturnShapes[resultIndex];

  // Use the reified shape to create an `tensor.empty` op
  auto tensorType = cast<TensorType>(result.getType());
  SmallVector<int64_t> staticShapes = llvm::to_vector(tensorType.getShape());
  SmallVector<Value, 2> dynamicSizes;

  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      Value dynDimValue;
      OpFoldResult ofr = foldResult[i];
      auto maybeConst = getConstantIntValue(ofr);
      if (maybeConst.has_value()) {
        dynDimValue = rewriter.create<arith::ConstantOp>(
            loc, llvm::cast<TypedAttr>(ofr.get<Attribute>()));
      } else {
        dynDimValue = dyn_cast<Value>(foldResult[i]);
      }
      newOps.insert(dynDimValue.getDefiningOp());
      dynamicSizes.push_back(dynDimValue);
    }
  }
  return rewriter.create<tensor::EmptyOp>(
      loc, staticShapes, tensorType.getElementType(), dynamicSizes);
}

hfusion::LoadOp hfusion::createCacheRead(OpBuilder &rewriter, Value operand,
                                         Location loc) {
  SmallPtrSet<Operation *, 4> newOps;
  auto emptyOp =
      hfusion::createEmptyOpWithSameShape(rewriter, operand, newOps, loc);
  auto cachedOp = rewriter.create<hfusion::LoadOp>(loc, ValueRange{operand},
                                                   ValueRange{emptyOp});
  newOps.insert(emptyOp);
  newOps.insert(cachedOp);
  operand.replaceAllUsesExcept(cachedOp.getResult(0), newOps);
  return cachedOp;
}
// No need to keep track of visited nodes
// we know the graph with only reshapes will be a tree
bool traceReshapeReachesReturn(Value value) {
  for (Operation *user : value.getUsers()) {
    if (isa<func::ReturnOp>(user)) {
      return true;
    }
  }
  for (Operation *user : value.getUsers()) {
    if (isReshapeOp(user)) {
      Value result = getReshapeResult(user);
      if (traceReshapeReachesReturn(result)) {
        return true;
      }
    }
  }
  return false;
}

SmallPtrSet<Operation *, 4> getOutputOnlyExceptions(OpResult result,
                                                    CacheWriteOptions options) {
  SmallPtrSet<Operation *, 4> exceptions;
  /// If the result is reshaped before return, then the cached result is
  /// only used to replace the original op in the reshape op being returned.
  ///
  /// Before cache write:
  ///
  ///   %result = ...
  ///   some_use(%result)
  ///   %reshaped = reshape(%result)
  ///   %reshaped1 = reshape(%reshaped)
  ///   func.return %reshaped1
  ///
  /// After cache write:
  ///
  ///   %result = ...
  ///   %cached = ...
  ///   %reshaped = reshape(%cached)
  ///   %reshaped1 = reshape(%reshaped)
  ///   func.return %reshaped1

  /// Otherwise, the cached result is only used to replace the original op
  /// in `func.return` op.
  ///
  /// Before cache write:
  ///
  ///   %result = ...
  ///   some_use(%result)
  ///   func.return %result
  ///
  /// After cache write:
  ///
  ///   %result = ...
  ///   %cached = ...
  ///   some_use(%result)
  ///   func.return %cached

  llvm::for_each(result.getUsers(), [&exceptions, &options](Operation *user) {
    if (isa<func::ReturnOp>(user))
      return;
    if (hfusion::isReshapeOp(user)) {
      // if we have a reshape produce trace then no need to trace again
      if (options.reshapeTrace.has_value() && !options.reshapeTrace->empty()) {
        const auto &traceChain = options.reshapeTrace.value();
        if (traceChain.back() == user)
          return;
        exceptions.insert(user);
        return;
      }
      Value reshapeRes = getReshapeResult(user);
      if (traceReshapeReachesReturn(reshapeRes))
        return;
    }
    exceptions.insert(user);
  });
  return exceptions;
}

/// This imitates replaceAllUsesExcept but duplicates the ops instead of
/// replacing inplace. The whole use tree until return is duplicated. This is
/// especially useful when handling identical return operands Before cache
/// write:
///
///   %result = ...
///   %reshaped = reshape(%result)
///   func.return %reshaped, %reshaped
///
/// After cache write:
///
///   %result = ...
///   %cached1 = ...
///   %cached2 = ...
///   %reshaped = reshape(%result) <- can be removed later
///   %reshaped1 = reshape(%cached1)
///   %reshaped2 = reshape(%cached2)
///   func.return %reshaped1, %reshaped2
///
/// If we don't duplicate (error):
///
///   %result = ...
///   %cached1 = ...
///   %cached2 = ...  <- should be used
///   %reshaped = reshape(%cached1)
///   func.return %reshaped, %reshaped
void duplicateAllUsesExcept(OpBuilder &rewriter, Value initialOldValue,
                            Value initialNewValue,
                            const SmallPtrSetImpl<Operation *> &exceptions) {
  std::queue<std::pair<Value, Value>> workQueue;
  workQueue.emplace(initialOldValue, initialNewValue);
  while (!workQueue.empty()) {
    auto [currentOld, currentNew] = workQueue.front();
    workQueue.pop();
    for (auto &use : llvm::make_early_inc_range(currentOld.getUses())) {
      Operation *oldOp = use.getOwner();
      if (isa<func::ReturnOp>(oldOp)) {
        use.assign(currentNew);
        continue;
      }
      if (exceptions.contains(oldOp) || isa<hfusion::StoreOp>(oldOp))
        continue;
      rewriter.setInsertionPoint(oldOp);
      Operation *newOp = rewriter.clone(*oldOp);
      newOp->setOperand(use.getOperandNumber(), currentNew);
      workQueue.emplace(oldOp->getResult(0), newOp->getResult(0));
    }
  }
}

FailureOr<hfusion::StoreOp>
hfusion::createCacheWrite(OpBuilder &rewriter, OpResult result,
                          CacheWriteOptions options) {
  auto *definingOp = result.getOwner();
  bool isLinalg = isa<linalg::LinalgOp>(definingOp);
  // TODO: support more types of tensor op as cache write
  bool isTensor = isTensorManipulationOp(definingOp);
  if (!isLinalg && !isTensor) {
    return {};
  }

  Location loc = definingOp->getLoc();
  OpBuilder::InsertionGuard guard(rewriter);

  // If the cache write mode is output-only ...
  SmallPtrSet<Operation *, 4> exceptions;
  if (options.outputOnly) {
    exceptions = getOutputOnlyExceptions(result, options);
  }

  hfusion::StoreOp cachedOp;
  tensor::EmptyOp emptyOp;
  if (options.cacheWriteToOutputInit && isLinalg) {
    // only support cache write to init for linalg op
    auto linalgOp = cast<linalg::LinalgOp>(definingOp);
    auto initOperand =
        linalgOp.getDpsInitOperand(result.getResultNumber())->get();
    // Input:
    //   %ret = linalg.op ins(...) outs(%init)
    //
    // After performing cache write to %ret:
    //   %dim = tensor.dim %init
    //   %empty = tensor.empty(%dim)
    //   %ret = linalg.op ins(...) outs(%empty)
    //   hfusion.store ins(%ret) outs(%init)
    rewriter.setInsertionPoint(definingOp);
    // for dynamic shape scenario, need to use `initOperand` to create
    // tensor.dim ops.
    //
    // [Optimization] If the defining op is `ReifyRankedShapedTypeOpInterface`,
    // we can use the result's reified shape to create the `tensor.empty`
    // instead. This leaves more room for optimzation in dyanmic shape scenarios
    // because the `tensor.dim` can propagate upwards.
    if (isa<ReifyRankedShapedTypeOpInterface>(definingOp)) {
      emptyOp = createEmptyOpWithReifiedShape(rewriter, definingOp, result,
                                              exceptions, loc);
    } else {
      emptyOp =
          createEmptyOpWithSameShape(rewriter, initOperand, exceptions, loc);
    }
    linalgOp.setDpsInitOperand(result.getResultNumber(), emptyOp);
    rewriter.setInsertionPointAfter(definingOp);
    cachedOp = rewriter.create<hfusion::StoreOp>(loc, ValueRange{result},
                                                 ValueRange{initOperand});
  } else {
    // Input:
    //   %ret = linalg.op ins(...) outs(%init)
    //
    // After performing cache write to %ret:
    //   %ret = linalg.op ins(...) outs(%init)
    //   %dim = tensor.dim %ret
    //   %empty = tensor.empty(%dim)
    //   hfusion.store ins(%ret) outs(%empty)
    rewriter.setInsertionPointAfter(definingOp);
    emptyOp = createEmptyOpWithSameShape(rewriter, result, exceptions, loc);
    cachedOp = rewriter.create<hfusion::StoreOp>(loc, ValueRange{result},
                                                 ValueRange{emptyOp});
  }
  exceptions.insert(emptyOp);
  exceptions.insert(cachedOp);
  duplicateAllUsesExcept(rewriter, result, cachedOp->getResult(0), exceptions);
  return cachedOp;
}

Value hfusion::ClipInput(PatternRewriter &rewriter, Location loc, Value input,
                         double upperBound, double lowerBound) {
  auto resInit = mlir::utils::createEmptyOp(rewriter, loc, input);
  // 1. minInit = vmin(input, upperBound)
  auto elementType = getElementTypeOrSelf(input);
  arith::ConstantOp upperBoundOp = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getFloatAttr(elementType, upperBound));
  auto minOp =
      hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                              hfusion::BinaryFnAttr>(
          rewriter, loc, hfusion::BinaryFn::minf,
          ValueRange{input, upperBoundOp->getResults()[0]}, resInit);
  // 2. maxInit = vmax(minInit, lowerBound)
  arith::ConstantOp lowerBoundOp = rewriter.create<arith::ConstantOp>(
      loc, elementType, rewriter.getFloatAttr(elementType, lowerBound));

  auto maxOp =
      hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                              hfusion::BinaryFnAttr>(
          rewriter, loc, hfusion::BinaryFn::maxf,
          ValueRange{minOp->getResults()[0], lowerBoundOp->getResults()[0]},
          resInit);
  return maxOp->getResults()[0];
}

BitVector hfusion::eraseFuncArgsWithAttr(func::FuncOp &funcOp,
                                         SmallVector<NamedAttribute> &attrs) {
  BitVector indicesToErase(funcOp.getNumArguments());
  for (auto argIndex : llvm::seq<int>(0, funcOp.getNumArguments())) {
    auto arg = funcOp.getArgument(argIndex);
    for (auto attr : attrs) {
      auto attrValue = funcOp.getArgAttr(argIndex, attr.getName());
      if (!attrValue)
        continue;
      if (attrValue == attr.getValue() && arg.getUsers().empty()) {
        indicesToErase.set(argIndex);
        break;
      }
    }
  }
  funcOp.eraseArguments(indicesToErase);
  return indicesToErase.flip();
}

BitVector hfusion::eraseFuncArgsExceptAttr(func::FuncOp &funcOp,
                                           NamedAttribute &attr) {
  BitVector indicesToErase(funcOp.getNumArguments());
  for (auto argIndex : llvm::seq<int>(0, funcOp.getNumArguments())) {
    auto arg = funcOp.getArgument(argIndex);
    auto attrValue = funcOp.getArgAttr(argIndex, attr.getName());
    if (!attrValue && arg.getUsers().empty()) {
      indicesToErase.set(argIndex);
    }
    if (attrValue != attr.getValue() && arg.getUsers().empty()) {
      indicesToErase.set(argIndex);
    }
  }
  funcOp.eraseArguments(indicesToErase);
  return indicesToErase.flip();
}

SmallVector<Value> hfusion::computeExtractCollapsedIndices(
    const SmallVector<ReassociationIndices> &reassociation,
    OperandRange &inputIndices, function_ref<Value(int idx)> getDimSize,
    OpBuilder &builder, Location loc) {
  SmallVector<Value> outputIndices;
  int idx = 0;
  for (const auto &group : reassociation) {
    if (group.size() == 1) {
      outputIndices.push_back(inputIndices[idx]);
      idx++;
      continue;
    }
    Value accumulator = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value multiplier = builder.create<arith::ConstantIndexOp>(loc, 1);
    // Process indices in the group from right to left.
    // if its dynamic then need to get mixes indices
    // Example:
    // indexing:       2      3       4   5
    // oldMixedSize:   3    | %dim2 | 8 | %dim1
    // extractindices: %ex4 | 1     | 5 | %extract3
    // newIndex = %extract3 + %dim1 * 5 + (8 * %dim1) * 1 + (%dim2 * 8 *
    // %dim1) * %ex4 Create computation using arith here
    // Accumulate indices from right to left
    for (int i = (int)(group.size()) - 1; i >= 0; --i) {
      Value dimSize = getDimSize(idx + i);
      // Mul the previously multiplier and the current coordinate
      Value mulResult =
          builder.create<arith::MulIOp>(loc, inputIndices[idx + i], multiplier);
      // Accumulate it into one integer
      accumulator = builder.create<arith::AddIOp>(loc, accumulator, mulResult);
      // multiply with the grid dimension size
      multiplier = builder.create<arith::MulIOp>(loc, multiplier, dimSize);
    }
    outputIndices.push_back(accumulator);
    idx += static_cast<int>(group.size());
  }
  return outputIndices;
}

std::optional<ArrayAttr> hfusion::getSymbolicTensor(Type tensorType) {
  auto rankedTensorType = dyn_cast<RankedTensorType>(tensorType);
  if (!rankedTensorType)
    return std::nullopt;
  auto encoding = rankedTensorType.getEncoding();
  if (!encoding || !isa<ArrayAttr>(encoding)) {
    return std::nullopt;
  }
  return cast<ArrayAttr>(encoding);
}

/// Used for tiling interface.
/// The current arange definition specifies the value for each location within a
/// tensor: arange[i, j...] = offset + i * stride[0] + j * stride[1]...
///
/// Tiling operation will give a list of offsets to use for each `linalg.index`
/// position, i.e. tile(arange[i, j...]) = arange[i+ioff, j+joff...]. We want to
/// add the offsets back into the offset of the operation, therefore
/// arange.offset += stride[0]*ioff + stride[1]*joff...
void hfusion::offsetArangeOp(OpBuilder &builder, Operation *tiledOp,
                             ArrayRef<OpFoldResult> offsets) {
  Location loc = tiledOp->getLoc();
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(tiledOp);
  // NOTE: This should only be called after a `if constexpr
  // (std::is_same<ArangeOp...`, so it should be safe to cast here.
  auto arange = cast<ArangeOp>(tiledOp);

  // Compute the additional offset introduced by tiling as
  // sum over dims {stride[dim]*offsets[dim]}
  Value finalOffset;
  for (auto offsetStride : llvm::zip(offsets, arange.getStrides())) {
    OpFoldResult offset = std::get<0>(offsetStride);
    if (!offset)
      continue;
    Value offsetVal;
    auto intVal = getConstantIntValue(offset);
    if (intVal.has_value())
      offsetVal = builder.create<arith::ConstantIndexOp>(loc, intVal.value());
    else if (auto val = dyn_cast<Value>(offset)) {
      assert(val.getType() == builder.getIndexType());
      offsetVal = val;
    } else
      llvm_unreachable("Expecting integer attribute or value as offset");
    Value stride = std::get<1>(offsetStride);
    Value dimOffset =
        builder.createOrFold<arith::MulIOp>(loc, offsetVal, stride);
    if (finalOffset)
      finalOffset =
          builder.createOrFold<arith::AddIOp>(loc, dimOffset, finalOffset);
    else
      finalOffset = dimOffset;
  }

  // If arange already has an offset, add on top of that.
  if (Value offsetVal = arange.getOffset())
    arange.getOffsetMutable().assign(
        builder.createOrFold<arith::AddIOp>(loc, offsetVal, finalOffset));
  else {
    // If no offset exists, we need to add to arguments, also add the block
    // argument to the body, in addition to adding the offset to the result
    // within the body.
    arange.getOffsetMutable().append(finalOffset);
    Block *body = arange.getBody();
    BlockArgument offsetArg =
        body->insertArgument(body->args_begin(), builder.getIndexType(), loc);
    Operation *resultOp = cast<linalg::YieldOp>(body->getTerminator())
                              .getValues()
                              .front()
                              .getDefiningOp();
    if (auto indexCast = dyn_cast<arith::IndexCastOp>(resultOp)) {
      assert(indexCast.getIn().getType() == builder.getIndexType());
      builder.setInsertionPoint(resultOp);
      indexCast.getInMutable().assign(
          builder.create<arith::AddIOp>(loc, indexCast.getIn(), offsetArg));
    } else {
      assert(resultOp->getResultTypes().front() == builder.getIndexType());
      builder.setInsertionPointAfter(resultOp);
      resultOp->replaceAllUsesWith(builder.create<arith::AddIOp>(
          loc, resultOp->getResult(0), offsetArg));
    }
  }
}

/// used for div and cast the results to certain data type with certain round
/// mode e.g.
///  div(x_i32, y_i32, f32, TRUNC)
/// equals
///  x_f32 = cast(x_i32) -> f32
///  y_f32 = cast(y_i32) -> f32
///  div = x_f32 / y_f32
///  res = cast<TRUNC>(div) -> f32
Value hfusion::divWithRoundMode(OpBuilder &builder, Location loc, Type resType,
                                Value src0, Value src1, Value resTensor,
                                hfusion::RoundMode roundingMode,
                                std::optional<Operation **> divOp) {
  Type elemType = getElementTypeOrSelf(src0.getType());
  Value castF32X = src0;
  Value castF32Y = src1;

  // step 1: x_f32 = cast(x) -> f32
  //         y_f32 = cast(y) -> f32
  if (!elemType.isF32()) {
    castF32X = hfusion::castTo(builder, src0, builder.getF32Type());
    castF32Y = hfusion::castTo(builder, src1, builder.getF32Type());
  }

  // step 2: div_f32 = x_f32 / y_f32
  auto divInit = utils::createEmptyOpWithTargetElemType(builder, loc, resTensor,
                                                        builder.getF32Type());

  auto divF32 = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                        linalg::BinaryFn, linalg::BinaryFnAttr>(
      builder, loc, linalg::BinaryFn::div, ValueRange{castF32X, castF32Y},
      ValueRange(divInit));

  if (divOp != std::nullopt) {
    (*divOp.value()) = divF32;
  }

  // step3: res = cast<RoundingMode>(div) -> resType
  if (resType.isInteger(8) || resType.isInteger(16)) {
    // cast from f32 to i32 then to resType
    Value resI32 = hfusion::castTo(builder, divF32->getResults()[0],
                                   builder.getI32Type(), roundingMode);
    Value res = hfusion::castTo(builder, resI32, resType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    return res;
  }

  // cast directly to resType
  Value res =
      hfusion::castTo(builder, divF32->getResults()[0], resType, roundingMode);
  return res;
}

Value hfusion::divWithRoundModeAndCastType(OpBuilder &builder, Location loc, Type resType,
                                Value src0, Value src1, Value resTensor,
                                hfusion::RoundMode roundingMode, hfusion::TypeFn castIntegerType,
                                std::optional<Operation **> divOp) {
  Type elemType = getElementTypeOrSelf(src0.getType());
  Value castF32X = src0;
  Value castF32Y = src1;

  // step 1: x_f32 = cast(x) -> f32
  //         y_f32 = cast(y) -> f32
  if (!elemType.isF32()) {
    castF32X = hfusion::castTo(builder, src0, builder.getF32Type(), castIntegerType);
    castF32Y = hfusion::castTo(builder, src1, builder.getF32Type(), castIntegerType); 
  }

  // step 2: div_f32 = x_f32 / y_f32
  auto divInit = utils::createEmptyOpWithTargetElemType(builder, loc, resTensor,
                                                        builder.getF32Type());

  auto divF32 = hfusion::createBinaryOp<linalg::ElemwiseBinaryOp,
                                        linalg::BinaryFn, linalg::BinaryFnAttr>(
      builder, loc, linalg::BinaryFn::div, ValueRange{castF32X, castF32Y},
      ValueRange(divInit));

  if (divOp != std::nullopt) {
    (*divOp.value()) = divF32;
  }

  // step3: res = cast<RoundingMode>(div) -> resType
  if (resType.isInteger(8) || resType.isInteger(16)) {
    // cast from f32 to i32 then to resType
    Value resI32 = hfusion::castTo(builder, divF32->getResults()[0],
                                   builder.getI32Type(), roundingMode);
    Value res = hfusion::castTo(builder, resI32, resType,
                                hfusion::RoundMode::TRUNCWITHOVERFLOW);
    return res;
  }

  // cast directly to resType
  Value res =
      hfusion::castTo(builder, divF32->getResults()[0], resType, roundingMode);
  return res;
}
