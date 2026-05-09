//===- PackTilingData.cpp ------- Pack Tiling Data Pass -------------------===//
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
//
// This file implements a pass to pack dynamic tiling data into a struct.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "hfusion-pack-tiling-data"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
namespace hfusion {

#define GEN_PASS_DEF_PACKTILINGDATA
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

namespace {

constexpr size_t kTilingKeyPosition = 0;
constexpr size_t kTilingDataBitWidth = 64;

bool isEmptyTilingFunc(func::FuncOp func) {
  if (func.getBlocks().size() != 1)
    return false;

  if (!llvm::hasSingleElement(func.getBlocks().front().getOperations()))
    return false;

  // If the function has only one operation, it must be a terminator (a.k.a.
  // return op). We can check the number of function results instead.
  return func.getNumResults() == 0;
}

std::optional<MemRefType> tryGetPackedTilingType(func::FuncOp tilingFunc) {
  for (unsigned argIdx = 0; argIdx < tilingFunc.getNumArguments(); argIdx++) {
    if (hacc::utils::isKernelArg(tilingFunc, argIdx,
                                 hacc::KernelArgType::kTilingStruct)) {
      return cast<MemRefType>(tilingFunc.getFunctionType().getInput(argIdx));
    }
  }
  return {};
}

static DictionaryAttr getKernelArgTypeAttr(OpBuilder &builder,
                                           hacc::KernelArgType type) {
  return builder.getDictionaryAttr(
      hacc::createHACCKernelArgAttr(builder.getContext(), type));
}

/// By convention, tiling function's name has the following format:
///   {original_kernel_name}_{tiling_function_suffix}
/// If the tiling function name follows this convention, return the
/// original kernel name.
static std::string getOriginalKernelName(const std::string &tilingFuncName) {
  auto tilingFuncNameSuffixPos = tilingFuncName.find(
      hacc::stringifyHostFuncType(hacc::HostFuncType::kTilingFunction));
  if (tilingFuncNameSuffixPos == std::string::npos)
    return tilingFuncName;

  return tilingFuncName.substr(0, tilingFuncNameSuffixPos - 1);
}

/// Try to get the one and only `func.call` op within the `scf.switch` op's
/// block.
static FailureOr<func::CallOp> getUniqueTilingCaseCallInBlock(Block &block) {
  std::optional<func::CallOp> tilingCaseCall = std::nullopt;
  auto caseWalkResult = block.walk([&](func::CallOp call) {
    if (tilingCaseCall.has_value()) {
      call->emitError() << "multiple device function call found within "
                           "same switch block";
      return WalkResult::interrupt();
    }
    tilingCaseCall = call;
    return WalkResult::advance();
  });
  if (caseWalkResult.wasInterrupted())
    return failure();

  return *tilingCaseCall;
}

/// Load the `idx`-th tiling data from the `tilingStruct`.
static Value loadTilingDataFromTilingStruct(unsigned idx, Value tilingStruct,
                                            OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  Value idxInTilingStruct =
      builder.create<arith::ConstantIndexOp>(tilingStruct.getLoc(), idx);
  return builder.create<memref::LoadOp>(tilingStruct.getLoc(), tilingStruct,
                                        ValueRange{idxInTilingStruct});
}

/// Load the tiling data directly from `ptr` (with `!llvm.ptr` type).
static Value loadTilingDataFromPtr(Value ptr, OpBuilder &builder,
                                   Location loc) {
  assert(ptr);
  return builder
      .create<LLVM::LoadOp>(
          loc, SmallVector<Type>{builder.getIntegerType(kTilingDataBitWidth)},
          SmallVector<Value>{ptr})
      ->getResult(0);
}

/// Store `tilingData` to the `idx`-th position in the `tilingStruct`.
static void storeToTilingStruct(unsigned idx, Value tilingStruct,
                                Value tilingData, OpBuilder &builder) {
  OpBuilder::InsertionGuard g(builder);
  Value idxV = builder.create<arith::ConstantIndexOp>(tilingData.getLoc(), idx);
  builder.create<memref::StoreOp>(tilingData.getLoc(), tilingData, tilingStruct,
                                  ValueRange{idxV});
}

/// Structure to hold information on a tiling case group.
///
/// In dynamic shape scenarios, the same kernel can be called multiple times
/// with inputs of different shapes. However, the tiling data calculation
/// and kernel selection need to be done differently each time.
/// Therefore, within the same caller, there could be multiple tiling case
/// groups.
///
/// For example, in the below case, the number of tiling case groups = 2.
///
/// ```mlir
/// func @caller(%arg0: i64 {hacc.tiling_data},
///              %arg1: i64 {hacc.tiling_data},
///              %arg2: i64 {hacc.tiling_data},
///              %arg3: i64 {hacc.tiling_data}) {HOST} {
///
///  ================== tiling case group #1 ==================
///  scf.index_switch
///    case tiling_case_1
//       func.call @foo_tiling_1(%arg0, %arg1)
///    case tiling_case_2
///      func.call @foo_tiling_2(%arg0, %arg1)
///
///  ================== tiling case group #2 ==================
///  scf.index_switch
///    case tiling_case_1
//       func.call @foo_tiling_1(%arg2, %arg3)
///    case tiling_case_2
///      func.call @foo_tiling_2(%arg2, %arg3)
// }
/// ```
///
/// After packing, each tiling case group will have its own tiling struct:
/// ```mlir
/// func @caller(%tiling_struct_0 : memref<2xi64> {hacc.tiling_struct},
///              %tiling_struct_1 : memref<2xi64> {hacc.tiling_struct}) {
///
///  ================== tiling case group #1 ==================
///  scf.index_switch
///    case tiling_case_1
//       func.call @foo_tiling_1(%tiling_struct_0)
///    case tiling_case_2
///      func.call @foo_tiling_2(%tiling_struct_0)
///
///  ================== tiling case group #2 ==================
///  scf.index_switch
///    case tiling_case_1
//       func.call @foo_tiling_1(%tiling_struct_1)
///    case tiling_case_2
///      func.call @foo_tiling_2(%tiling_struct_1)
// }
/// ```
class TilingCaseGroupInfo {
public:
  TilingCaseGroupInfo(StringRef tilingFuncNameIn, func::FuncOp callerFuncIn)
      : tilingFuncName(tilingFuncNameIn), callerFunc(callerFuncIn) {}

  /// Get the caller of this tiling case group.
  func::FuncOp getCaller() { return callerFunc; }

  /// Get the name of the tiling function for this tiling case group.
  StringRef getTilingFuncName() { return tilingFuncName; }

  /// Get the calls to the device functions within this tiling case group.
  SmallVector<func::CallOp> getTilingCaseCalls() { return tilingCaseCalls; }

  /// Get the tiling operands and the non-tiling operands for the tiling case
  /// group.
  std::pair<SmallVector<Value>, SmallVector<Value>>
  getTilingAndNonTilingOperands() {
    assert(callOperandIsTiling.has_value());
    assert(callOperands.has_value());
    SmallVector<Value> tilingOperands;
    SmallVector<Value> nonTilingOperands;
    for (auto [idx, value] : llvm::enumerate(*callOperands)) {
      // For non tiling operands, just record them as they are.
      if (!(*callOperandIsTiling).test(idx)) {
        nonTilingOperands.push_back(value);
        continue;
      }
      tilingOperands.push_back(value);
    }
    return {tilingOperands, nonTilingOperands};
  }

  /// Get the tiling operands for the tiling case group.
  SmallVector<Value> getTilingOperands() {
    assert(callOperandIsTiling.has_value());
    assert(callOperands.has_value());
    auto [tilingOperands, _] = getTilingAndNonTilingOperands();
    return tilingOperands;
  }

  /// Get the tiling function call.
  ///
  /// \Note This function assumes that the caller manages the host resource
  /// and thus the defining op of the tiling operand is the tiling function.
  func::CallOp getTilingFuncCall() {
    assert(callOperandIsTiling.has_value());
    assert(callOperands.has_value());
    auto firstIndex = (*callOperandIsTiling).find_first();
    if (firstIndex == -1)
      return nullptr;
    return (*callOperands)[firstIndex].getDefiningOp<func::CallOp>();
  }

  /// Record (if not already) or verify that the operands matches the record.
  LogicalResult setOrVerifyCallOperands(ArrayRef<Value> operands,
                                        const BitVector &operandIsTiling) {
    // Both should be set at the same time.
    assert(!(callOperands.has_value() ^ callOperandIsTiling.has_value()));
    if (!callOperands.has_value() && !callOperandIsTiling.has_value()) {
      callOperands = llvm::to_vector(operands);
      callOperandIsTiling = operandIsTiling;
      return success();
    }
    for (const auto &[currentValue, storedValue] :
         llvm::zip(operands, *callOperands)) {
      if (currentValue != storedValue)
        return failure();
    }
    return success(*callOperandIsTiling == operandIsTiling);
  }

  /// Record a tiling case call.
  void recordTilingCaseCall(func::CallOp call) {
    tilingCaseCalls.push_back(call);
  }

private:
  StringRef tilingFuncName;
  func::FuncOp callerFunc;
  SmallVector<func::CallOp> tilingCaseCalls;
  std::optional<BitVector> callOperandIsTiling;
  std::optional<SmallVector<Value>> callOperands;
};

struct PackTilingDataPass
    : public impl::PackTilingDataBase<PackTilingDataPass> {
public:
  explicit PackTilingDataPass(const PackTilingDataOptions &options)
      : PackTilingDataBase(options) {}

  LogicalResult initialize(MLIRContext *context) override {
    for (const std::string &symbol : includeSymbols) {
      includeSymbolSet.insert(StringAttr::get(context, symbol));
    }
    return success();
  }

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    if (failed(collectTilingFuncInfo(mod))) {
      mod->emitError() << "failed to collect tiling function information";
      signalPassFailure();
      return;
    }

    if (failed(collectTilingCaseInfo(mod))) {
      mod->emitError() << "failed to collect tiling case information";
      signalPassFailure();
      return;
    }

    for (auto &[tilingFuncName, tilingFuncInfo] : tilingFuncInfos) {
      if (!tilingFuncInfo.needPackTiling)
        continue;

      // Pack tiling for tiling func
      if (failed(packTilingForTilingFunc(tilingFuncInfo))) {
        signalPassFailure();
        return;
      }

      // Pack tiling for device functions
      for (auto deviceFunc : tilingFuncToDeviceFuncs[tilingFuncName]) {
        LDBG(
            "Packing tiling data in device func : " << deviceFunc.getSymName());
        if (failed(packTilingForDeviceFunc(deviceFunc, tilingFuncInfo))) {
          deviceFunc->emitError() << "Fail to modify device function";
          signalPassFailure();
          return;
        }
      }

      LDBG("Fixing call sites of tiling func: " << tilingFuncName);
      if (failed(fixCallSiteOfTilingFunc(tilingFuncInfo))) {
        tilingFuncInfo.tilingFunc.emitError()
            << "Fail to fix call site for tiling function";
        signalPassFailure();
        return;
      }
    }

    // Erase dead and empty tiling function
    bishengir::DeadFunctionEliminationOptions dfeOptions;
    dfeOptions.filterFn = [](FunctionOpInterface funcLike) {
      auto funcOp = dyn_cast<func::FuncOp>(funcLike.getOperation());
      if (!funcOp)
        return false;

      if (!hacc::utils::isHost(funcOp))
        return false;

      auto hostFuncType = hacc::utils::getHostFuncType(funcOp);
      if (!hostFuncType || *hostFuncType != hacc::HostFuncType::kTilingFunction)
        return false;

      return isEmptyTilingFunc(funcOp);
    };
    bishengir::eliminateDeadFunctions(mod, dfeOptions);
  }

private:
  struct TilingFuncInfo {
    /// Get the number of tiling data produced by this tiling function.
    unsigned getTilingStructSize() const {
      return tilingStructType == MemRefType() ? 0
                                              : tilingStructType.getDimSize(0);
    }

    bool needPackTiling{false};
    StringRef tilingFuncName;
    MemRefType tilingStructType{};
    func::FuncOp tilingFunc;
  };

private:
  //===--------------------------------------------------------------------===//
  // Functions for collecting information.
  //===--------------------------------------------------------------------===//

  LogicalResult collectTilingFuncInfo(ModuleOp mod) {
    bool applyToAll = includeSymbolSet.empty();

    // Collect all tiling func info
    llvm::DenseSet<StringRef> emptyTilingFuncNames;
    auto walkResult = mod.walk([&](func::FuncOp funcOp) {
      std::optional<hacc::HostFuncType> maybeHostFuncType =
          hacc::utils::getHostFuncType(funcOp);
      if (!maybeHostFuncType.has_value() ||
          *maybeHostFuncType != hacc::HostFuncType::kTilingFunction)
        return WalkResult::skip();

      TilingFuncInfo &tilingFuncInfo = tilingFuncInfos[funcOp.getSymName()];
      tilingFuncInfo.tilingFuncName = funcOp.getSymName();

      std::optional<MemRefType> packedTilingType =
          tryGetPackedTilingType(funcOp);
      if (packedTilingType.has_value()) {
        tilingFuncInfo.tilingStructType = *packedTilingType;
        return WalkResult::skip();
      }

      if (isEmptyTilingFunc(funcOp)) {
        emptyTilingFuncNames.insert(tilingFuncInfo.tilingFuncName);
        return WalkResult::skip();
      }

      if (failed(tiling::verifyTilingFunc(funcOp)))
        return WalkResult::interrupt();

      if (!applyToAll && !includeSymbolSet.contains(funcOp.getSymNameAttr()))
        return WalkResult::skip();

      tilingFuncInfo.needPackTiling = true;
      // If size of the tiling struct is determined by the number of return
      // results of the tiling function.
      // If tiling key is not packed, the size of tiling struct is the
      // number of results minus one.
      tilingFuncInfo.tilingStructType = MemRefType::get(
          {std::max<int>(0, funcOp.getNumResults() - (packTilingKey ? 0 : 1))},
          IntegerType::get(funcOp.getContext(), kTilingDataBitWidth));
      tilingFuncInfo.tilingFunc = funcOp;
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();

    // Get the tiling function to device function mapping.
    walkResult = mod.walk([&](func::FuncOp deviceFunc) {
      auto tilingFuncAttr = deviceFunc->getAttrOfType<hacc::TilingFunctionAttr>(
          hacc::TilingFunctionAttr::name);
      if (!tilingFuncAttr)
        return WalkResult::skip();

      auto tilingFuncName = tilingFuncAttr.getFuncNameStr();
      tilingFuncToDeviceFuncs[tilingFuncName].push_back(deviceFunc);

      if (emitGetTilingStructSizeFunction) {
        OpBuilder builder(deviceFunc);
        doEmitGetTilingStructSizeFunction(
            builder, tilingFuncName,
            tilingFuncInfos.at(tilingFuncName).getTilingStructSize());
      }

      if (!emptyTilingFuncNames.contains(tilingFuncName))
        return WalkResult::skip();

      // If the tiling function is empty, we can remove the attributes from
      // the device function, because there is no need to call it.
      deviceFunc->removeAttr(hacc::TilingFunctionAttr::name);
      return WalkResult::skip();
    });
    if (walkResult.wasInterrupted())
      return failure();

    // Verify that all the device function matches the tiling function.
    for (auto &[tilingFuncName, deviceFuncs] : tilingFuncToDeviceFuncs) {
      auto tilingFunc = mod.lookupSymbol<func::FuncOp>(tilingFuncName);
      if (!tilingFunc)
        return mod.emitError("Cannot find the tiling function: ")
               << tilingFuncName;

      if (failed(tiling::deviceFuncsMatchTilingFunc(deviceFuncs, tilingFunc)))
        return failure();
    }
    return success();
  }

  LogicalResult collectTilingCaseInfo(ModuleOp mod) {
    // Get all tiling case info
    auto walkResult = mod.walk([&](annotation::MarkOp markOp) {
      if (!utils::isAnnotationWithAttr(markOp, hacc::TilingFunctionAttr::name))
        return WalkResult::skip();

      auto tilingFuncAttr = markOp->getAttrOfType<hacc::TilingFunctionAttr>(
          hacc::TilingFunctionAttr::name);
      if (!tilingFuncAttr) {
        markOp.emitWarning() << "Annotation marking tiling case doesn't have "
                                "tiling func name label";
        return WalkResult::skip();
      }

      auto tilingFuncName = tilingFuncAttr.getFuncNameStr();
      if (!tilingFuncInfos.contains(tilingFuncName))
        return WalkResult::skip();

      auto maybeTilingCaseSwitch =
          markOp.getSrc().getDefiningOp<scf::IndexSwitchOp>();
      if (!maybeTilingCaseSwitch)
        return WalkResult::skip();

      TilingCaseGroupInfo tilingCaseInfo(
          /*tilingFuncNameIn=*/tilingFuncName,
          /*callerFuncIn=*/markOp->getParentOfType<func::FuncOp>());
      LDBG("Tiling func is: " << tilingCaseInfo.getTilingFuncName());
      // Process each tiling case's switch block
      for (auto [idx, caseKey] :
           llvm::enumerate(maybeTilingCaseSwitch.getCases())) {
        Block &caseBlock = maybeTilingCaseSwitch.getCaseBlock(idx);
        FailureOr<func::CallOp> tilingCaseCall =
            getUniqueTilingCaseCallInBlock(caseBlock);

        if (failed(tilingCaseCall))
          return WalkResult::interrupt();

        if (failed(analyzeTilingCaseCallSite(
                *tilingCaseCall, tilingCaseInfo,
                tilingFuncInfos[tilingCaseInfo.getTilingFuncName()])))
          return WalkResult::interrupt();
      }

      tilingFunc2TilingCaseInfo[tilingCaseInfo.getTilingFuncName()].push_back(
          tilingCaseInfo);

      markOp.erase();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();

    return success();
  }

  /// Analyze and record information for the tiling case call within a tiling
  /// case group.
  ///
  /// There could be two scenarios:
  ///   1) The tiling data comes from the caller's function argument
  ///   2) The tiling data comes from a function call to the tiling function
  ///      (i.e., the tiling is managed by the caller)
  ///
  /// Assumptions for both cases:
  ///   1) Within the same tiling case group, the operands at every call site
  ///      should be the same.
  ///   1) The resource management scheme must be consistent if the caller
  ///      contains multiple tiling case groups. In other words, we don't
  ///      allow:
  ///
  ///  ```mlir
  ///  ================== tiling case group #1 ==================
  ///  scf.index_switch
  ///    case tiling_case_1
  //       func.call @foo_tiling_1(%arg0, %arg1)
  ///    case tiling_case_2
  ///      func.call @foo_tiling_2(%arg0, %arg1)
  ///
  ///  ================== tiling case group #2 ==================
  ///  %0:2 = func.call @tiling_func(...)
  ///  scf.index_switch
  ///    case tiling_case_1
  //       func.call @foo_tiling_1(%0#0, %0#1)
  ///    case tiling_case_2
  ///      func.call @foo_tiling_2(%0#0, %0#1)
  ///  ```
  LogicalResult analyzeTilingCaseCallSite(func::CallOp callSite,
                                          TilingCaseGroupInfo &tilingCaseInfo,
                                          TilingFuncInfo &tilingFuncInfo) {
    // Record tiling case call.
    tilingCaseInfo.recordTilingCaseCall(callSite);

    // Verify that the device function called indeed uses the target tiling
    // function
    auto deviceFuncName = callSite.getCallee();
    auto deviceFunc = getOperation().lookupSymbol<func::FuncOp>(deviceFuncName);
    if (!deviceFunc) {
      callSite->emitError()
          << "reference to undefined function '" << deviceFuncName << "'";
      return failure();
    }
    if (!deviceFunc->hasAttr(hacc::TilingFunctionAttr::name)) {
      deviceFunc->emitError() << "called device function doesn't have "
                                 "tiling function name attribute";
      return failure();
    }
    auto calleeTilingFuncName =
        deviceFunc->getAttrOfType<hacc::TilingFunctionAttr>(
            hacc::TilingFunctionAttr::name);
    if (!calleeTilingFuncName || tilingCaseInfo.getTilingFuncName() !=
                                     calleeTilingFuncName.getFuncNameStr()) {
      deviceFunc->emitError()
          << "called device function's tiling function doesn't match";
      return failure();
    }

    func::FuncOp caller = tilingCaseInfo.getCaller();
    // Verify that all function calls within the same tiling case block have
    // the exact same arguments. And that the tiling arguments are the same.
    unsigned numArgs = deviceFunc.getNumArguments();
    BitVector isTilingArg(numArgs, false);
    for (auto [idx, operand] : llvm::enumerate(callSite.getOperands())) {
      if (!hacc::utils::isTilingArg(deviceFunc, idx))
        continue;

      isTilingArg.set(idx);
      // If the current tiling operand is passed into the device function,
      // it means that the resource is not managed by the caller
      bool currentTilingIsManagedByCaller = !isa<BlockArgument>(operand);
      LDBG("current tiling operand " << idx << " is managed by caller: "
                                     << currentTilingIsManagedByCaller);

      auto [iter, isInserted] = resourceIsManagedByCaller.insert(
          {caller, currentTilingIsManagedByCaller});
      if (!isInserted && iter->second != currentTilingIsManagedByCaller)
        return callSite->emitError() << "resource management scheme is "
                                        "inconsistent at this call site";
    }

    if (failed(tilingCaseInfo.setOrVerifyCallOperands(
            llvm::to_vector(callSite.getOperands()), isTilingArg)))
      return callSite->emitError()
             << "tiling case call operands are inconsistent or tiling "
                "operands' indicies are inconsistent";

    // Verify the tiling func call if exists.
    if (resourceIsManagedByCaller.at(caller)) {
      // host function can only be called in host function!
      if (!hacc::utils::isHost(caller))
        return caller->emitError() << "Non host function calling host tiling!";

      auto calcTilingOp = tilingCaseInfo.getTilingFuncCall();
#ifndef NDEBUG
      assert(calcTilingOp);
#else
      if (!calcTilingOp)
        return failure();
#endif
      LDBG("current tiling is calculated by " << *calcTilingOp);
      for (auto operand : tilingCaseInfo.getTilingOperands()) {
        auto currentCalcTilingOp = operand.getDefiningOp<func::CallOp>();
        if (!currentCalcTilingOp || currentCalcTilingOp != calcTilingOp)
          return callSite->emitError()
                 << "tiling operand doesn't come from a tiling function or "
                    "doesn't come from the same tiling function";
      }
      if (failed(tiling::checkCallCalcTilingWithTilingOperands(
              calcTilingOp, tilingCaseInfo.getTilingOperands())))
        return calcTilingOp->emitError() << "failed to check tiling";

      StringRef curTilingFuncName = calcTilingOp.getCallee();
      if (tilingCaseInfo.getTilingFuncName() != curTilingFuncName)
        return calcTilingOp.emitError()
               << "device function and tiling function doesn't match";
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Functions for packing tiling data
  //===--------------------------------------------------------------------===//

  class TilingStruct {
    using tiling_struct_iterator = SmallVectorImpl<Value>::iterator;

  public:
    TilingStruct() = default;
    explicit TilingStruct(Value tilingKey, Value tilingStruct)
        : tilingKeyIsPacked(false), tilingStruct({tilingKey, tilingStruct}) {}

    explicit TilingStruct(Value tilingStruct)
        : tilingKeyIsPacked(true), tilingStruct({tilingStruct}) {}

    void replaceValueWithTilingData(Value val, unsigned tilingIndex,
                                    OpBuilder &builder, IRMapping &mapper) {
      auto replacement = getIntegerTilingData(tilingIndex, builder);
      val.replaceUsesWithIf(replacement, [&](OpOperand &operand) -> bool {
        return operand.get().getType() == replacement.getType();
      });
      mapper.map(val, replacement);
    }

    void storeIntegerTilingData(Value tilingData, unsigned tilingIndex,
                                OpBuilder &builder) {
      // If the tiling key is not packed into the tiling struct, we need to
      // modify the argument to a `llvm.ptr` and store the tiling key directly.
      if (tilingKeyIsPacked) {
        storeToTilingStruct(tilingIndex, tilingStruct.back(), tilingData,
                            builder);
        return;
      }
      if (tilingIndex != kTilingKeyPosition) {
        storeToTilingStruct(tilingIndex - 1, tilingStruct.back(), tilingData,
                            builder);
        return;
      }
      builder.create<LLVM::StoreOp>(tilingData.getLoc(),
                                    /*value=*/tilingData,
                                    /*addr=*/tilingStruct.front());
    }

    tiling_struct_iterator tilingStructBegin() { return tilingStruct.begin(); }
    tiling_struct_iterator tilingStructEnd() { return tilingStruct.end(); }

  private:
    Value getIntegerTilingData(unsigned tilingIndex, OpBuilder &builder) {
      if (tilingKeyIsPacked)
        return loadTilingDataFromTilingStruct(tilingIndex, tilingStruct.back(),
                                              builder);

      if (tilingIndex != kTilingKeyPosition) {
        return loadTilingDataFromTilingStruct(tilingIndex - 1,
                                              tilingStruct.back(), builder);
      }

      return loadTilingDataFromPtr(tilingStruct.front(), builder,
                                   tilingStruct.front().getLoc());
    }

    bool tilingKeyIsPacked;
    // If tiling key is not packed, the tiling key and the tiling struct is
    // separated.
    SmallVector<Value, 2> tilingStruct;
  };

  TilingStruct allocateResourceForTilingStruct(OpBuilder &builder,
                                               TilingFuncInfo &info,
                                               Location loc) {
    auto tilingStruct =
        builder.create<memref::AllocOp>(loc, info.tilingStructType);
    if (packTilingKey)
      return TilingStruct{tilingStruct};

    auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
    auto integerTy = builder.getIntegerType(kTilingDataBitWidth);
    Value one =
        builder.create<LLVM::ConstantOp>(loc, builder.getIntegerType(64), 1);
    Value tilingKeyPtr = builder.create<LLVM::AllocaOp>(
        loc, /*resultType=*/ptrTy, /*elementType=*/integerTy,
        /*arraySize=*/one,
        /*alignment=*/0);
    return TilingStruct{tilingKeyPtr, tilingStruct};
  }

  TilingStruct getTilingStructFromCaller(OpBuilder &builder,
                                         TilingFuncInfo &info,
                                         func::FuncOp caller) {
    // TODO: Use hacc interface to get argument insertion point
    auto insertionPoint = caller.getNumArguments();
    auto tilingStruct = getInsertedTilingStruct(caller, info, insertionPoint);
    if (packTilingKey)
      return TilingStruct{tilingStruct};

    auto tilingKey = getInsertedTilingKey(caller, info, insertionPoint);
    return TilingStruct{tilingKey, tilingStruct};
  }

  /// Pack tiling data in the tiling function.
  ///
  /// Input:
  /// ```mlir
  ///    func.func @tiling_calc(...) {
  ///      ...
  ///      return %td0, %td1, %td2 : i64, i64, i64
  ///    }
  /// ```
  ///
  /// Output:
  /// ```mlir
  ///    func.func @tiling_calc(..., %tiling_struct) {
  ///      ...
  ///      memref.store %tiling_struct[%c0], %td0 : i64
  ///      memref.store %tiling_struct[%c1], %td1 : i64
  ///      memref.store %tiling_struct[%c2], %td2 : i64
  ///      return
  ///    }
  /// ```
  LogicalResult packTilingForTilingFunc(TilingFuncInfo &info) {
    LDBG("Packing tiling data for tiling func: " << info.tilingFuncName);
    func::FuncOp tilingFunc = info.tilingFunc;
    OpBuilder builder(tilingFunc);
    TilingStruct s = getTilingStructFromCaller(builder, info, tilingFunc);

    func::ReturnOp returnOp = utils::getAssumedUniqueReturnOp(tilingFunc);
    builder.setInsertionPoint(returnOp);
    for (unsigned tilingIndex = 0; tilingIndex < returnOp->getNumOperands();
         ++tilingIndex) {
      s.storeIntegerTilingData(returnOp.getOperand(tilingIndex), tilingIndex,
                               builder);
    }
    // Update return op's operand.
    returnOp->setOperands({});
    // Update function signature.
    tilingFunc.setResAttrsAttr({});
    tilingFunc.setFunctionType(tilingFunc.getFunctionType().clone(
        /*inputs=*/tilingFunc.getFunctionBody().getArgumentTypes(),
        /*results=*/{}));
    return success();
  }

  Value getInsertedTilingKey(func::FuncOp funcOp, TilingFuncInfo &info,
                             size_t insertionPoint) {
    OpBuilder builder(funcOp);
    funcOp.insertArgument(
        insertionPoint, LLVM::LLVMPointerType::get(builder.getContext()),
        getKernelArgTypeAttr(builder, hacc::KernelArgType::kTilingKey),
        funcOp.getLoc());
    return funcOp.getArgument(insertionPoint);
  }

  Value getInsertedTilingStruct(func::FuncOp funcOp, TilingFuncInfo &info,
                                size_t insertionPoint) {
    OpBuilder builder(funcOp);
    funcOp.insertArgument(
        insertionPoint, info.tilingStructType,
        getKernelArgTypeAttr(builder, hacc::KernelArgType::kTilingStruct),
        funcOp.getLoc());
    return funcOp.getArgument(insertionPoint);
  }

  /// Replace the tiling data arguments in the device function with a tiling
  /// struct.
  LogicalResult packTilingForDeviceFunc(func::FuncOp &deviceFunc,
                                        TilingFuncInfo &tilingFuncInfo) {
    OpBuilder builder(deviceFunc);
    BitVector isTilingArg(deviceFunc.getNumArguments(), false);
    for (size_t i = 0, e = deviceFunc.getNumArguments(); i != e; ++i) {
      if (hacc::utils::isTilingArg(deviceFunc, i))
        isTilingArg.set(i);
    }
    // Insert new arguments and resize the bit vector.
    TilingStruct s =
        getTilingStructFromCaller(builder, tilingFuncInfo, deviceFunc);
    isTilingArg |= BitVector(deviceFunc.getNumArguments(), false);

    // Replace the use of tiling data argument with values unpacked from
    // tiling struct argument
    builder.setInsertionPointToStart(&deviceFunc.front());

    IRMapping mapper;
    size_t tilingIdx = 0;
    for (size_t i = 0, e = deviceFunc.getNumArguments(); i != e; ++i) {
      if (i >= isTilingArg.size() || !isTilingArg.test(i))
        continue;

      s.replaceValueWithTilingData(deviceFunc.getArgument(i), tilingIdx,
                                   builder, mapper);
      tilingIdx++;
    }
    deviceFunc.eraseArguments(isTilingArg);
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Functions for fixing call sites
  //===--------------------------------------------------------------------===//

  /// Fix the call sites for the tiling cases corresponding to the input
  /// tiling function.
  LogicalResult fixCallSiteOfTilingFunc(TilingFuncInfo &tilingFuncInfo) {
    std::vector<TilingCaseGroupInfo> tilingCases =
        tilingFunc2TilingCaseInfo[tilingFuncInfo.tilingFuncName];
    for (auto &tilingCasesToFix : tilingCases) {
      func::FuncOp callerFunc = tilingCasesToFix.getCaller();
      LDBG("Fixing tiling call sites in: " << callerFunc.getSymName());
      if (failed(fixCallSitesOfTilingCaseInCaller(callerFunc, tilingCasesToFix,
                                                  tilingFuncInfo)))
        return callerFunc->emitError() << "Fail to fix call site in caller";
    }
    return success();
  }

  /// Fix the call sites of the tiling case group within the caller.
  ///
  /// A tiling struct value is constructed to replace the use of tiling data
  /// within the caller.
  /// This includes:
  ///   1) The tiling case call sites
  ///   2) The tiling function call (if resource is managed by caller)
  ///   3) The caller's function argument (if resource is not managed by
  ///   caller)
  LogicalResult
  fixCallSitesOfTilingCaseInCaller(func::FuncOp caller,
                                   TilingCaseGroupInfo &tilingCaseInfo,
                                   TilingFuncInfo &tilingFuncInfo) {
    OpBuilder builder(caller);
    auto [tilingOperands, newDeviceFuncCallOperands] =
        tilingCaseInfo.getTilingAndNonTilingOperands();

    TilingStruct s;
    IRMapping mapper;
    if (resourceIsManagedByCaller.at(caller)) {
      func::CallOp oldTilingFuncCall = tilingCaseInfo.getTilingFuncCall();
      if (!oldTilingFuncCall) {
        llvm::report_fatal_error("oldTilingFuncCall doesn't exist");
        return failure();
      }
      builder.setInsertionPoint(oldTilingFuncCall);
      // If tiling function is managed by caller, allocate a memory for the
      // tiling struct.
      s = allocateResourceForTilingStruct(builder, tilingFuncInfo,
                                          oldTilingFuncCall.getLoc());
      // Generate a new tiling function call.
      SmallVector<Value> newTilingFuncCallOperands(
          oldTilingFuncCall->getOperands());
      // By convention, the last operand of the tiling function is the tiling
      // struct.
      newTilingFuncCallOperands.append(s.tilingStructBegin(),
                                       s.tilingStructEnd());
      builder.create<func::CallOp>(
          oldTilingFuncCall.getLoc(), oldTilingFuncCall.getCalleeAttr(),
          tilingFuncInfo.tilingFunc.getFunctionType().getResults(),
          newTilingFuncCallOperands);

      for (OpResult tilingFuncResult : oldTilingFuncCall.getResults()) {
        if (tilingFuncResult.getUsers().empty())
          continue;

        s.replaceValueWithTilingData(tilingFuncResult,
                                     tilingFuncResult.getResultNumber(),
                                     builder, mapper);
      }
      oldTilingFuncCall->erase();
    } else {
      // Else, add an argument to the caller.
      s = getTilingStructFromCaller(builder, tilingFuncInfo, caller);

      // Remove the tiling key/data arguments in the caller.
      auto tilingCaseCaller = tilingCaseInfo.getCaller();
      std::optional<SymbolTable::UseRange> maybeUses =
          tilingCaseCaller.getSymbolUses(getOperation());
      if (maybeUses.has_value() && !maybeUses->empty())
        return tilingCaseCaller->emitError()
               << "Currently don't support nested calls in host";

      BitVector funcArgToErase(tilingCaseCaller.getNumArguments(), false);
      builder.setInsertionPointToStart(
          &tilingCaseCaller.getFunctionBody().getBlocks().front());
      for (auto [tilingIdx, tilingOperand] : llvm::enumerate(tilingOperands)) {
        auto ba = cast<BlockArgument>(tilingOperand);
        funcArgToErase.set(ba.getArgNumber());
        if (ba.getUsers().empty())
          continue;

        s.replaceValueWithTilingData(ba, tilingIdx, builder, mapper);
      }

      // Modify the tilingCaseCaller.
      tilingCaseCaller.eraseArguments(funcArgToErase);
    }

    // TODO: assume that the tiling struct is added at the end
    newDeviceFuncCallOperands.append(s.tilingStructBegin(),
                                     s.tilingStructEnd());
    llvm::for_each(newDeviceFuncCallOperands,
                   [&](Value &v) { v = mapper.lookupOrDefault(v); });
    // Construct the new tiling case calls collectively.
    // All the func calls belonging to the same tiling case are modified in
    // the same way.
    for (auto oldCall : tilingCaseInfo.getTilingCaseCalls()) {
      builder.setInsertionPoint(oldCall);
      func::CallOp newCall = builder.create<func::CallOp>(
          oldCall.getLoc(), oldCall.getCalleeAttr(), oldCall.getResultTypes(),
          ValueRange(newDeviceFuncCallOperands));
      oldCall.replaceAllUsesWith(newCall);
      oldCall->erase();
    }
    return success();
  }

  void doEmitGetTilingStructSizeFunction(OpBuilder &opBuilder,
                                         StringRef tilingFuncName,
                                         unsigned tilingStructSize) {
    auto getTilingStructSizeFnName = constructHostFunctionName(
        getOriginalKernelName(tilingFuncName.str()),
        hacc::HostFuncType::kGetTilingStructSizeFunction);
    // Bail out if the kernel already has a get tiling struct size function
    if (getOperation().lookupSymbol(getTilingStructSizeFnName))
      return;

    Type returnType = IntegerType::get(&getContext(), 64);
    FunctionType t = FunctionType::get(&getContext(),
                                       /*inputs=*/SmallVector<Type>(),
                                       /*results=*/
                                       SmallVector<Type>{returnType});
    auto getTilingSizeFunc =
        opBuilder.create<func::FuncOp>(opBuilder.getUnknownLoc(),
                                       /*name=*/
                                       getTilingStructSizeFnName,
                                       /*type=*/t);
    Block *entryBlock = getTilingSizeFunc.addEntryBlock();
    hacc::utils::setHost(getTilingSizeFunc);
    hacc::utils::setHostFuncType(
        getTilingSizeFunc, hacc::HostFuncType::kGetTilingStructSizeFunction);

    // Return the tiling size
    opBuilder.setInsertionPointToStart(entryBlock);
    auto tilingStructSizeV = opBuilder.create<arith::ConstantIntOp>(
        opBuilder.getUnknownLoc(), tilingStructSize, returnType);
    opBuilder.create<func::ReturnOp>(opBuilder.getUnknownLoc(),
                                     SmallVector<Value>{tilingStructSizeV});

    // Set `hacc.get_tiling_struct_size` attribute to device functions
    if (!tilingFuncToDeviceFuncs.contains(tilingFuncName))
      return;

    for (auto &func : tilingFuncToDeviceFuncs.at(tilingFuncName))
      func->setAttr(hacc::GetTilingStructSizeFunctionAttr::name,
                    hacc::GetTilingStructSizeFunctionAttr::get(
                        &getContext(), getTilingSizeFunc.getSymName()));
  }

private:
  llvm::StringMap<TilingFuncInfo> tilingFuncInfos;
  llvm::StringMap<std::vector<TilingCaseGroupInfo>> tilingFunc2TilingCaseInfo;
  llvm::StringMap<SmallVector<func::FuncOp>> tilingFuncToDeviceFuncs;
  DenseMap<func::FuncOp, bool> resourceIsManagedByCaller;
  DenseSet<StringAttr> includeSymbolSet;
};

} // namespace

std::unique_ptr<Pass>
createPackTilingDataPass(const PackTilingDataOptions &options) {
  return std::make_unique<PackTilingDataPass>(options);
}

} // namespace hfusion
} // namespace mlir
