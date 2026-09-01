//===- TensorResToOutParams.cpp - Move tensor results to function params --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/Support/Debug.h"

#include <queue>

#define DEBUG_TYPE "hfusion-tensor-results-to-out-params"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_TENSORRESTOOUTPARAMS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace {

constexpr static llvm::StringLiteral kHACCNewlyInsertedArgAttrName =
    "hacc.newly_inserted";

static DictionaryAttr createHACCOuputArgAttrs(MLIRContext *ctx,
                                              unsigned outputIdx) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(NamedAttribute(
      StringAttr::get(ctx, hacc::KernelArgTypeAttr::name),
      hacc::KernelArgTypeAttr::get(ctx, hacc::KernelArgType::kOutput)));
  attrs.push_back(
      NamedAttribute(StringAttr::get(ctx, hacc::OutputIdxAttr::name),
                     hacc::OutputIdxAttr::get(ctx, outputIdx)));
  return DictionaryAttr::get(ctx, attrs);
}

static void markHACCOutputArgAttr(func::FuncOp func, unsigned argIdx,
                                  unsigned outputIdx) {
  func.setArgAttrs(argIdx,
                   createHACCOuputArgAttrs(func.getContext(), outputIdx));
}

static void markArgAsNewlyInserted(func::FuncOp func, unsigned argIdx) {
  func.setArgAttr(argIdx, kHACCNewlyInsertedArgAttrName,
                  UnitAttr::get(func.getContext()));
}

struct OutputValueInfo {
  /// This is the value that is returned by the function. If the return value is
  /// the result of a reshape op, this is the value before being reshaped.
  Value output;
  /// Traces of reshape/slice ops that the return value goes through.
  SmallVector<Operation *> reshapeTrace;
  /// Operand number of the return value in `func.return` op.
  unsigned outputIdx;
};

static SmallVector<OutputValueInfo>
getReturnValuesInfo(func::ReturnOp returnOp) {
  SmallVector<OutputValueInfo> outputValuesInfo;
  for (auto [outputIdx, v] : llvm::enumerate(returnOp.getOperands())) {
    if (!isa<mlir::TensorType>(v.getType())) {
      returnOp->emitWarning("return operand is not tensor type");
      continue;
    }
    Operation *op = v.getDefiningOp();
    if (!isReshapeOp(op)) {
      outputValuesInfo.push_back({v, {}, static_cast<unsigned>(outputIdx)});
      continue;
    }
    SmallVector<Operation *> trace =
        hfusion::getReshapeOrSliceOpProduceTrace(v);
    assert(!trace.empty() && "reshape produce trace must not be empty");
    Value source = hfusion::getReshapeSource(trace.back());
    outputValuesInfo.push_back(
        {source, trace, static_cast<unsigned>(outputIdx)});
  }
  return outputValuesInfo;
}

using NeedInsertAtIdx = std::pair<bool, unsigned>;
/// Find the function argument index to insert the new output argument.
/// If the function already has the target output argument, don't need to insert
/// new argument. Just directly return the index.
///
/// Invariant: the function's input/output arguments are in ascending order.
/// TODO: Create interface and refactor.
static NeedInsertAtIdx insertOutputToFuncArgs(func::FuncOp func,
                                              unsigned outputIdx) {
  std::optional<unsigned> inputEndIdx;
  std::optional<unsigned> outputEndIdx;
  for (unsigned argIdx = 0; argIdx < func.getNumArguments(); argIdx++) {
    /// This is to handle the following case:
    /// ```mlir
    ///   func.func @foo(%arg0 {input_idx = 0}, %arg1 {not_input})
    /// ```
    /// If the target return index is 0, we want to insert at position 1, which
    /// directly follows the last input argument.
    auto maybeInputIdx = hacc::getHACCInputIdx(func, argIdx);
    if (maybeInputIdx.has_value())
      inputEndIdx = *maybeInputIdx;

    auto maybeOutputIdx = hacc::getHACCOuputIdx(func, argIdx);
    if (!maybeOutputIdx.has_value())
      continue;

    /// This is to handle the following case:
    /// ```mlir
    ///   func.func @foo(%arg0, %arg1 {output_idx = 0}, %arg2)
    /// ```
    /// If the target return index is 1, we want to insert at position 2, which
    /// directly follows the last output argument.
    outputEndIdx = argIdx;

    if (outputIdx == *maybeOutputIdx)
      return {/*needInsert=*/false, argIdx};

    if (outputIdx < *maybeOutputIdx)
      return {/*needInsert=*/true, argIdx};
  }
  return {/*needInsert=*/true, outputEndIdx.has_value() ? *outputEndIdx + 1
                               : inputEndIdx.has_value()
                                   ? *inputEndIdx + 1
                                   : func.getNumArguments()};
}

static SetVector<unsigned> getNewlyInsertedArgs(func::FuncOp func) {
  SetVector<unsigned> result;
  for (unsigned idx = 0; idx < func.getNumArguments(); idx++) {
    if (func.getArgAttr(idx, kHACCNewlyInsertedArgAttrName))
      result.insert(idx);
  }
  return result;
}

static void removeNewlyInsertedArgAttr(func::FuncOp func) {
  for (unsigned idx = 0; idx < func.getNumArguments(); idx++) {
    func.removeArgAttr(idx, kHACCNewlyInsertedArgAttrName);
  }
}

struct TensorResToOutParamsPass
    : public impl::TensorResToOutParamsBase<TensorResToOutParamsPass> {
public:
  explicit TensorResToOutParamsPass(const TensorResToOutParamsOptions &options)
      : TensorResToOutParamsBase(options) {}
  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() final;

private:
  using OutputArg2ReturnIdx = std::pair<BlockArgument, unsigned>;

  struct CallSiteBuildingInfo {
    func::FuncOp caller;
    func::FuncOp callee;
    func::CallOp oldCall;
    bool callerManageResource;
    SmallVector<OutputValueInfo> callerReturnValuesInfo;
  };

private:
  //===--------------------------------------------------------------------===//
  // Methods for performing res to out to a target function.
  //===--------------------------------------------------------------------===//

  /// Perform result to out params for the input function. Return true if the
  /// function signature was modified.
  bool performResultToOutParams(func::FuncOp func);

  /// Get the value to replace the current dps init value.
  ///
  /// If the reshape trace is empty, the replacement value is simply a new
  /// block argument. Otherwise, it's computed by reverse the
  /// expandOp/collapseOp provided in `trace`.
  Value getInitValueReplacement(func::FuncOp func, Value dpsInitV,
                                const OutputValueInfo &info);

  //===--------------------------------------------------------------------===//
  // Methods for fixing call site after performing res to out.
  //===--------------------------------------------------------------------===//

  /// The main entry point to fix the call sites of all res-to-out-ed functions.
  LogicalResult fixCallSitesAfterResultToOutParams();

  /// Generate a new call site for the callee that has been res-to-out-ed.
  std::optional<func::CallOp> getFixedCallSite(func::FuncOp caller,
                                               func::FuncOp callee,
                                               func::CallOp oldCall);

  /// Get the new call site operand at index `operandIdx` for the callee's
  /// res-to-out-ed argument.
  ///
  /// The call site operand could be a `tensor.empty` or a block argument,
  /// depending on the resource management scheme and whether the call's result
  /// is returned.
  Value getResToOutCallOperand(unsigned operandIdx,
                               const CallSiteBuildingInfo &info);

  //===--------------------------------------------------------------------===//
  // Common utility functions.
  //===--------------------------------------------------------------------===//

  /// If the function's output argument number equals its return value num, it
  /// is in the correct form that tensor-results-to-out-params expects.
  /// Return true if the input function satisfy this condition.
  bool isResultToOutParamsFormat(func::FuncOp func, Diagnostic &diag,
                                 bool resToOutForFuncWithNoOutput = false);

  BlockArgument getOrInsertNewOutputArgument(func::FuncOp func,
                                             unsigned outputIdx,
                                             RankedTensorType newArgType);

  BlockArgument insertNewOutputArgument(func::FuncOp func, unsigned argIndex,
                                        Type argType, DictionaryAttr argAttrs,
                                        Location argLoc);

  Value insertNewTensorEmpty(OpBuilder &opBuilder, Type newOperandType,
                             Location loc);

private:
  std::queue<func::FuncOp> modifiedFuncs;
  DenseSet<StringAttr> includeSymbolSet;
  DenseSet<StringRef> funcWithNewlyInsertedArgs;
};

LogicalResult TensorResToOutParamsPass::initialize(MLIRContext *context) {
  for (const std::string &symbol : includeSymbols)
    includeSymbolSet.insert(StringAttr::get(context, symbol));
  return success();
}

BlockArgument TensorResToOutParamsPass::getOrInsertNewOutputArgument(
    func::FuncOp func, unsigned outputIdx, RankedTensorType newArgType) {
  auto [needInsert, argIndex] = insertOutputToFuncArgs(func, outputIdx);
  if (needInsert)
    return insertNewOutputArgument(
        func, argIndex, newArgType,
        createHACCOuputArgAttrs(func.getContext(), outputIdx), func.getLoc());

  return func.getArgument(argIndex);
}

BlockArgument TensorResToOutParamsPass::insertNewOutputArgument(
    func::FuncOp func, unsigned argIndex, Type argType, DictionaryAttr argAttrs,
    Location argLoc) {
  func.insertArgument(argIndex, argType, argAttrs, func.getLoc());
  markArgAsNewlyInserted(func, argIndex);
  funcWithNewlyInsertedArgs.insert(func.getSymName());
  return func.getArgument(argIndex);
}

Value TensorResToOutParamsPass::insertNewTensorEmpty(OpBuilder &opBuilder,
                                                     Type newOperandType,
                                                     Location loc) {
  auto tensorType = cast<RankedTensorType>(newOperandType);
  return utils::createStaticShapeEmptyOp(opBuilder, loc, tensorType);
}

Value TensorResToOutParamsPass::getInitValueReplacement(
    func::FuncOp func, Value dpsInitV, const OutputValueInfo &info) {
  BlockArgument newArg = getOrInsertNewOutputArgument(
      func, info.outputIdx,
      cast<RankedTensorType>(func.getFunctionType().getResult(info.outputIdx)));
  OpBuilder builder(func.getContext());
  builder.setInsertionPoint(dpsInitV.getDefiningOp());
  return tensor::reshape_utils::getReverseReshapedValue(builder, newArg,
                                                        info.reshapeTrace);
}

bool TensorResToOutParamsPass::performResultToOutParams(func::FuncOp func) {
  if (hacc::utils::isHost(func))
    // If the function is a host function and there is un-outlined dps op,
    // we don't do res-to-out.
    return false;

  LDBG("Perform res to out for func: " << *func);
  func::ReturnOp returnOp = utils::getAssumedUniqueReturnOp(func);
  assert(returnOp && "function without a return op");
  LDBG("Return op is: " << *returnOp);

  bool modified = false;
  for (const auto &returnValueInfo : getReturnValuesInfo(returnOp)) {
    const Value &maybeDpsResult = returnValueInfo.output;
    Operation *maybeDpsOp = maybeDpsResult.getDefiningOp();
    if (!maybeDpsOp || !isa<DestinationStyleOpInterface>(maybeDpsOp))
      continue;

    OpResult dpsResult = cast<OpResult>(maybeDpsResult);
    // Update specified init of target retDefOp, use `initIdx` to update the
    // right init value when handling multiple return values.
    auto dpsOp = cast<DestinationStyleOpInterface>(maybeDpsOp);
    unsigned int initIdx = dpsResult.getResultNumber();
    OpOperand *initV = dpsOp.getDpsInitOperand(initIdx);

    auto initSource = traceReshapeOrSliceSingleProducerOrSelf(initV->get());
    if (auto ba = dyn_cast<BlockArgument>(initSource)) {
      // If result to out param is already done, make sure to add the argument
      // attribute as well.
      if (dyn_cast<func::FuncOp>(ba.getOwner()->getParentOp()))
        markHACCOutputArgAttr(func, ba.getArgNumber(),
                              returnValueInfo.outputIdx);

      continue;
    }
    Value replacement =
        getInitValueReplacement(func, initV->get(), returnValueInfo);
    dpsOp.setDpsInitOperand(initIdx, replacement);
    modified = true;
  }

  if (modified)
    LDBG("After res to out: " << *func);

  return modified;
}

LogicalResult TensorResToOutParamsPass::fixCallSitesAfterResultToOutParams() {
  while (!modifiedFuncs.empty()) {
    func::FuncOp func = modifiedFuncs.front();
    modifiedFuncs.pop();
    LDBG("Fix call sites of @" << func.getSymName());

    DenseMap<func::FuncOp, tiling::CallerInfo> workList;
    tiling::getCallerInfo(func, getOperation(), workList);

    DenseSet<Operation *> processed;
    while (!workList.empty()) {
      auto &[caller, callerInfo] = *(workList.begin());
      if (processed.contains(caller))
        continue;

      auto callee = callerInfo.callee;
      for (auto oldCall : callerInfo.callSites) {
        auto newCall = getFixedCallSite(caller, callee, oldCall);
        if (!newCall.has_value())
          return failure();

        oldCall.replaceAllUsesWith(newCall.value());
        oldCall->erase();
        LDBG("new call site is: " << *newCall);
        Diagnostic diag(func.getLoc(), DiagnosticSeverity::Remark);
        if (!isResultToOutParamsFormat(caller, diag)) {
          // In order to guarantee return result num always match output num,
          // perform res-to-out on caller func.s
          LDBG("try performing res to out for caller func");
          bool modified = performResultToOutParams(caller);
          if (modified) {
            LDBG("modified caller func");
            modifiedFuncs.push(caller);
          }
        }
      }
      removeNewlyInsertedArgAttr(callee);
      tiling::getCallerInfo(caller, getOperation(), workList);
      processed.insert(caller);
      workList.erase(caller);
    }
  }
  return success();
}

Value TensorResToOutParamsPass::getResToOutCallOperand(
    unsigned operandIdx, const CallSiteBuildingInfo &info) {
  func::FuncOp caller = info.caller;
  func::FuncOp callee = info.callee;
  func::CallOp oldCall = info.oldCall;
  bool callerManageResource = info.callerManageResource;
  auto callerReturnValuesInfo = info.callerReturnValuesInfo;

  auto maybeOuputIdx = hacc::getHACCOuputIdx(callee, operandIdx);
  assert(maybeOuputIdx.has_value());
  if (!maybeOuputIdx.has_value()) {
    return {};
  }
  // Check to see if the result of the `func.call` is returned by the caller.
  auto callResult = oldCall->getResult(*maybeOuputIdx);
  std::optional<unsigned> maybeCallerOutputIdx;
  for (const auto &callerReturnValueInfo : callerReturnValuesInfo) {
    if (callResult == callerReturnValueInfo.output) {
      maybeCallerOutputIdx = callerReturnValueInfo.outputIdx;
      LDBG("callerReturnValueInfo.outputIdx: "
           << callerReturnValueInfo.outputIdx);
      break;
    }
  }
  bool callResultIsReturned = maybeCallerOutputIdx.has_value();

  RankedTensorType newOperandType;
  if (callResultIsReturned) {
    // If the `func.call` result is returned by the caller, the new operand
    // type should be consistent with the caller's return type.
    newOperandType = cast<RankedTensorType>(
        caller.getFunctionType().getResult(*maybeCallerOutputIdx));
  } else {
    // If the `func.call` result is not returned by the caller, then the new
    // operand type is just the callee's argument type.
    newOperandType =
        cast<RankedTensorType>(callee.getArgument(operandIdx).getType());
  }

  if (callerManageResource && !newOperandType.hasStaticShape()) {
    callee->emitError(
        "do not support dynamic shape when caller is managing host resource");
    signalPassFailure();
    return {};
  }

  OpBuilder opBuilder(oldCall);
  /// Rules on using `tensor.empty` to fix call site:
  ///   1) if call op's result is not used as return value, always fix with
  ///       `tensor.empty`
  ///   2) if caller func manages host resource, fix with `tensor.empty`
  bool fixWithEmptyTensor = callResultIsReturned ? callerManageResource : true;
  Value newOperand;
  if (fixWithEmptyTensor) {
    newOperand =
        insertNewTensorEmpty(opBuilder, newOperandType, oldCall.getLoc());
  } else {
    newOperand = getOrInsertNewOutputArgument(caller, *maybeCallerOutputIdx,
                                              newOperandType);
  }
  if (!callResultIsReturned)
    return newOperand;

  return tensor::reshape_utils::getReverseReshapedValue(
      opBuilder, newOperand,
      callerReturnValuesInfo[*maybeCallerOutputIdx].reshapeTrace);
}

std::optional<func::CallOp> TensorResToOutParamsPass::getFixedCallSite(
    func::FuncOp caller, func::FuncOp callee, func::CallOp oldCall) {
  const SetVector<unsigned> &calleeNewlyInsertedArgs =
      getNewlyInsertedArgs(callee);
  SmallVector<Value> oldOperands(oldCall.getOperands());

  // Gather information needed to construct the call site operands.
  CallSiteBuildingInfo info{
      caller, callee, oldCall,
      /*callerManageResource=*/
      hacc::utils::isHost(caller) && enableManageHostResources,
      /*callerReturnValuesInfo=*/
      getReturnValuesInfo(utils::getAssumedUniqueReturnOp(caller))};

  // Construct the new operands.
  SmallVector<Value> newOperands;
  SetVector<unsigned> callerNewlyInsertedArgs;
  unsigned oldArgIdx = 0;
  for (unsigned newArgIdx = 0; newArgIdx < callee.getNumArguments();
       newArgIdx++) {
    // If the current operand doesn't correspond to a newly inserted arg, just
    // use the old operand.
    if (!calleeNewlyInsertedArgs.contains(newArgIdx)) {
      assert(oldArgIdx < oldOperands.size());
      newOperands.push_back(oldOperands[oldArgIdx]);
      oldArgIdx++;
      continue;
    }
    newOperands.push_back(getResToOutCallOperand(newArgIdx, info));
  }

  OpBuilder opBuilder(oldCall);
  return opBuilder.create<func::CallOp>(oldCall.getLoc(), callee, newOperands);
}

bool TensorResToOutParamsPass::isResultToOutParamsFormat(
    func::FuncOp func, Diagnostic &diag, bool resToOutForFuncWithNoOutput) {
  unsigned int numArguments = func.getNumArguments();
  unsigned int numOutputArgs = 0;
  for (size_t i = 0; i < numArguments; ++i) {
    if (!hacc::utils::isKernelArg(func, i, hacc::KernelArgType::kOutput))
      continue;

    std::optional<unsigned> maybeOutputIdx = hacc::getHACCOuputIdx(func, i);
    if (!maybeOutputIdx || *maybeOutputIdx != numOutputArgs)
      return false;

    numOutputArgs++;
  }
  if (numOutputArgs == 0 && func.getNumResults() != 0) {
    // special handling for func with no output attr
    return resToOutForFuncWithNoOutput;
  }

  if (hacc::utils::isDevice(func) || !hacc::utils::isHost(func)) {
    if (numOutputArgs != func.getNumResults()) {
      diag.attachNote()
          << "output argument number and result number mismatch. outputs: " +
                 std::to_string(numOutputArgs) +
                 ", results: " + std::to_string(func.getNumResults());
      return false;
    }
    return true;
  }
  /// For host functions, we only count the return values coming from
  /// `func.call`, because these are the ops that we're responsible for doing
  /// res-to-out.
  func::ReturnOp returnOp = utils::getAssumedUniqueReturnOp(func);
  assert(returnOp && "function without a return op");
  unsigned int targetResultNumber = 0;
  for (auto returnValue : returnOp.getOperands()) {
    auto srcValue =
        hfusion::traceReshapeOrSliceSingleProducerOrSelf(returnValue);
    auto *definingOp = srcValue.getDefiningOp();
    if (dyn_cast_if_present<func::CallOp>(definingOp)) {
      targetResultNumber++;
      continue;
    }
  }
  if (numOutputArgs != targetResultNumber) {
    diag.attachNote()
        << "output argument number and result number mismatch. outputs: " +
               std::to_string(numOutputArgs) +
               ", results: " + std::to_string(targetResultNumber);
    return false;
  }
  return true;
}

void TensorResToOutParamsPass::runOnOperation() {
  bool applyToAll = includeSymbolSet.empty();
  ModuleOp mod = getOperation();
  mod->walk([&](func::FuncOp func) {
    if (applyToAll || includeSymbolSet.contains(func.getSymNameAttr())) {
      bool modified = performResultToOutParams(func);
      if (modified)
        modifiedFuncs.push(func);
    }
  });

  if (failed(fixCallSitesAfterResultToOutParams()))
    signalPassFailure();

  mod->walk([&](func::FuncOp func) {
    if (funcWithNewlyInsertedArgs.contains(func.getSymName())) {
      removeNewlyInsertedArgAttr(func);
      funcWithNewlyInsertedArgs.erase(func.getSymName());
    }
  });

  // If it's not on-demand res-to-out, we need to make sure that the functions
  // are in the correct format.
  if (applyToAll)
    mod->walk([&](func::FuncOp func) {
      // Sanity check if all func in the correct form
      Diagnostic diag(func.getLoc(), DiagnosticSeverity::Error);
      if (!isResultToOutParamsFormat(func, diag, true)) {
        func->emitError() << diag.str();
        signalPassFailure();
      }
    });
}

} // anonymous namespace

std::unique_ptr<Pass> hfusion::createTensorResToOutParamsPass(
    const TensorResToOutParamsOptions &options) {
  return std::make_unique<TensorResToOutParamsPass>(options);
}
