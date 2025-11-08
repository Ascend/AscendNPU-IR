//===- HIVMInferFuncCoreType.cpp - CoreType Inference Pass ----------------===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_INFERFUNCCORETYPE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "infer-core-type"

namespace {
/// Inference the core type of each function, based on the operations used
/// inside the function body.
///
/// The inference process will respect the explicitly-annotated core type
/// attributes, not changing them.
struct InferFuncCoreTypePass
    : public impl::InferFuncCoreTypeBase<InferFuncCoreTypePass> {
public:
  enum class CoreTypeConstraint { None = 0, AIC, AIV, MIX };

public:
  using ConstraintMap = std::map<Region *, std::optional<CoreTypeConstraint>>;
  using ModuleCoreTypeMap = std::map<ModuleOp, hivm::TModuleCoreType>;

  void runOnOperation() override {
    const mlir::CallGraph callGraph(getOperation());

    // core type constraits for each function.
    ConstraintMap cmap;

    for (mlir::CallGraphNode *node : llvm::post_order(&callGraph)) {
      if (node->isExternal()) {
        continue;
      }

      auto *region = node->getCallableRegion();
      auto funcOp = region->getParentOfType<hacc::HACCFunction>();
      if (funcOp && funcOp.isHost()) {
        // only infer core type for device func
        continue;
      }

      cmap[region] = calculateConstraint(region, cmap);
    }

    bool fail{false};

    // core type information for enclosing module
    ModuleCoreTypeMap moduleMap;

    for (auto [region, constraintMaybe] : cmap) {
      func::FuncOp func = region->getParentOfType<func::FuncOp>();
      if (!func) {
        // we only have interest in FuncOp
        continue;
      }

      if (!constraintMaybe) {
        llvm::errs() << "InferCoreType Fail! Reason: can not calculate the "
                        "core type constraint for func.func("
                     << cast<SymbolOpInterface>(func.getOperation()).getName()
                     << ")\n";

        fail = true;
        continue;
      }

      CoreTypeConstraint c = constraintMaybe.value();

      hivm::TFuncCoreType e = hivm::TFuncCoreType::MIX;
      hivm::TModuleCoreType inferredModuleCoreType = hivm::TModuleCoreType::MIX;
      if (c == CoreTypeConstraint::AIC) {
        e = hivm::TFuncCoreType::AIC;
        inferredModuleCoreType = hivm::TModuleCoreType::AIC;
      } else if (c == CoreTypeConstraint::AIV) {
        e = hivm::TFuncCoreType::AIV;
        inferredModuleCoreType = hivm::TModuleCoreType::AIV;
      } else if (c == CoreTypeConstraint::None) {
        // temporarily allow functions without core type constraints, and infer
        // them to AIV functions.
        //
        // TODO: reconsider this behavior in future
        e = hivm::TFuncCoreType::AIV;
        inferredModuleCoreType = hivm::TModuleCoreType::AIV;
      }

      func->setAttr(hivm::TFuncCoreTypeAttr::name,
                    hivm::TFuncCoreTypeAttr::get(func->getContext(), e));

      auto parentModule = func->getParentOfType<ModuleOp>();
      if (moduleMap.find(parentModule) == moduleMap.end()) {
        moduleMap.insert({parentModule, inferredModuleCoreType});
      } else {
        auto existingModuleCoreType = moduleMap.at(parentModule);
        if (existingModuleCoreType == hivm::TModuleCoreType::MIX)
          continue;

        if (existingModuleCoreType != inferredModuleCoreType)
          moduleMap[parentModule] = hivm::TModuleCoreType::MIX;
      }
    }

    for (auto [module, coreType] : moduleMap)
      module->setAttr(
          hivm::TModuleCoreTypeAttr::name,
          hivm::TModuleCoreTypeAttr::get(module->getContext(), coreType));

    if (fail) {
      signalPassFailure();
    }
  }

private:
  // e is the core type property of a callee
  void refineConstraint(unsigned &c, hivm::TFuncCoreType e) {
    switch (e) {
    case hivm::TFuncCoreType::AIC:
      c |= static_cast<unsigned>(CoreTypeConstraint::AIC);
      break;
    case hivm::TFuncCoreType::AIV:
      c |= static_cast<unsigned>(CoreTypeConstraint::AIV);
      break;
    case hivm::TFuncCoreType::MIX:
      c |= static_cast<unsigned>(CoreTypeConstraint::MIX);
      break;
    default:
      break;
    }
  };

  // e is the core type property of an op inside a func body
  void refineConstraint(unsigned &c, hivm::TCoreType e) {
    switch (e) {
    case hivm::TCoreType::CUBE:
      c |= static_cast<unsigned>(CoreTypeConstraint::AIC);
      break;
    case hivm::TCoreType::VECTOR:
      c |= static_cast<unsigned>(CoreTypeConstraint::AIV);
      break;
    case hivm::TCoreType::CUBE_AND_VECTOR:
      c |= static_cast<unsigned>(CoreTypeConstraint::MIX);
      break;
    default:
      break;
    }
  };

  // calculate the core type constraint for a FuncOp. r is the body region of
  // the op.
  //
  // cmap is used to the query the constraint of callees inside the region.
  std::optional<CoreTypeConstraint>
  calculateConstraint(Region *r, const ConstraintMap &cmap) {
    unsigned c = static_cast<unsigned>(CoreTypeConstraint::None);
    bool fail = false;

#ifndef NDEBUG
    auto constraintStr = [fail](unsigned e) -> std::string {
      if (fail) {
        return "FAIL!!";
      }

      switch (static_cast<CoreTypeConstraint>(e)) {
      case CoreTypeConstraint::AIC:
        return "AIC";
      case CoreTypeConstraint::AIV:
        return "AIV";
      case CoreTypeConstraint::MIX:
        return "MIX";
      case CoreTypeConstraint::None:
        return "NONE";
      }
    };

    auto funcNameStr = [r]() -> std::string {
      auto symbol = dyn_cast<SymbolOpInterface>(r->getParentOp());
      return symbol ? symbol.getName().data() : "<NONAME>";
    };

    LLVM_DEBUG(llvm::dbgs() << "\n'" << r->getParentOp()->getName() << " : "
                            << funcNameStr() << "\n");
#endif

    // query explicitly annotated core type attribute
    auto coreTypeAttr = dyn_cast_or_null<mlir::hivm::TFuncCoreTypeAttr>(
        r->getParentOp()->getAttr(mlir::hivm::TFuncCoreTypeAttr::name));
    if (coreTypeAttr) {
      CoreTypeConstraint result;
      switch (coreTypeAttr.getFuncCoreType()) {
      case hivm::TFuncCoreType::AIC:
        result = CoreTypeConstraint::AIC;
        break;
      case hivm::TFuncCoreType::AIV:
        result = CoreTypeConstraint::AIV;
        break;
      case hivm::TFuncCoreType::MIX:
        result = CoreTypeConstraint::MIX;
        break;
      default:
        result = CoreTypeConstraint::None;
      }

#ifndef NDEBUG
      LLVM_DEBUG(llvm::dbgs()
                 << "  CoreType Constraint Result(From Annotated Attr): "
                 << constraintStr(static_cast<unsigned>(result)) << "\n");
#endif

      return result;
    }

    // inference the core type by analyzing inferrable ops and function calls
#ifndef NDEBUG
    r->walk([this, &c, &fail, &cmap, &constraintStr
#else
    r->walk([this, &c, &fail, &cmap
#endif
    ](Operation *op) -> mlir::WalkResult {
      if (c == static_cast<unsigned>(CoreTypeConstraint::MIX)) {
        // it's impossible to refine c any more, so interrupt the walk
        return WalkResult::interrupt();
      }

      // op with 'CoretypeInterface' must succeed to inference, otherwise
      // it's an implementation error.
      if (auto infer = dyn_cast<hivm::CoreTypeInterface>(op)) {
        auto coreTypeMaybe = infer.getCoreType();
        if (!coreTypeMaybe) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  -> Fail Reason: single op inference failure\n"
                     << "  -> Failed Op: " << infer << "\n");

          fail = true;
          return WalkResult::interrupt();
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "  -- " << *op << ": " << coreTypeMaybe.value() << "\n");

        refineConstraint(c, coreTypeMaybe.value());
      }

      if (auto call = dyn_cast<mlir::CallOpInterface>(op)) {
        CallInterfaceCallable callee = call.getCallableForCallee();
        if (callee.is<mlir::Value>()) {
          // indirect call is considered as an ill-form, becuase it makes the
          // inference impossible in general.
          LLVM_DEBUG(llvm::dbgs() << "  -> Fail Reason: indirect call\n"
                                  << "  -> Failed Op: " << *op << "\n");

          fail = true;
          return WalkResult::interrupt();
        }

        auto calleeOp =
            dyn_cast<CallableOpInterface>(SymbolTable::lookupNearestSymbolFrom(
                op, callee.get<SymbolRefAttr>()));
        if (!calleeOp) {
          LLVM_DEBUG(llvm::dbgs() << "  -> Fail Reason: unresolved call\n"
                                  << "  -> Failed Op: " << *op << "\n");

          fail = true;
          return WalkResult::interrupt();
        }

        // if callee is explicityly annotated, it's done
        mlir::hivm::TFuncCoreTypeAttr calleeCoreTypeAttr =
            dyn_cast_or_null<mlir::hivm::TFuncCoreTypeAttr>(
                calleeOp->getAttr(mlir::hivm::TFuncCoreTypeAttr::name));
        if (calleeCoreTypeAttr) {
          refineConstraint(c, calleeCoreTypeAttr.getFuncCoreType());
          return WalkResult::advance();
        }

        // need to query cmap to decide callee's core type
        Region *calleeRegion = calleeOp.getCallableRegion();

        if (!calleeRegion) {
          LLVM_DEBUG(llvm::dbgs() << "  -> Fail Reason: call an external func "
                                     "with unknown core type\n"
                                  << "  -> Failed Op: " << *op << "\n");

          fail = true;
        } else if (!cmap.count(calleeRegion)) {
          LLVM_DEBUG(llvm::dbgs() << "  -> Fail Reason: "
                                     "recursive call loop\n"
                                  << "  -> Failed Op: " << *op << "\n");

          fail = true;
        } else if (!cmap.at(calleeRegion)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  -> Fail Reason: callee inference failure\n"
                     << "  -> Failed Op: " << *op << "\n");

          fail = true;
        }

        if (fail) {
          return WalkResult::interrupt();
        }

        LLVM_DEBUG(llvm::dbgs() << "  -- " << *op << ": "
                                << constraintStr(static_cast<unsigned>(
                                       cmap.at(calleeRegion).value()))
                                << "\n");

        c |= static_cast<unsigned>(cmap.at(calleeRegion).value());
      }

      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs()
               << "  CoreType Constraint Result: " << constraintStr(c) << "\n");

    if (fail) {
      return std::nullopt;
    }

    return static_cast<CoreTypeConstraint>(c);
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createInferFuncCoreTypePass() {
  return std::make_unique<InferFuncCoreTypePass>();
}
