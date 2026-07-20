//===- Utility.cpp - RegBase compile utility --------------------*- C++ -*-===//
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

#include "bishengir/Tools/bishengir-compile/regbase/Utility.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

namespace bishengir {
namespace regbase {
namespace {

bool shouldUseSoftDotScale(ModuleOp module) {
  bool useSoftDotScale = false;
  module.walk<WalkOrder::PreOrder>(
      [&](mlir::hfusion::MatMulMxOp matmulMxOp) -> WalkResult {
        auto aType =
            mlir::cast<ShapedType>(matmulMxOp->getOperand(0).getType());
        auto shape = aType.getShape();
        if (shape.size() > 1 && shape[1] == 32)
          useSoftDotScale = true;
        return WalkResult::interrupt();
      });
  return useSoftDotScale;
}

static bool isSharedWithDownstreamToolchain(StringRef argName) {
  auto &opts = llvm::cl::getRegisteredOptions();
  auto it = opts.find(argName);
  if (it == opts.end())
    return true;
  // Compare category by name "Options Shared with HIVMC" — equivalent to
  // the pointer comparison in BiShengIRCompileMainConfig but avoids a
  // link-time dependency from the regbase library on BiShengIRCompileLib.
  return llvm::any_of(it->second->Categories, [](const cl::OptionCategory *cat) {
    return cat->getName() == "Options Shared with HIVMC";
  });
}

} // namespace

LogicalResult inferMixedCV(ModuleOp &module,
                           BiShengIRCompileMainConfig &config) {
  if (module.walk([](mlir::scope::ScopeOp) {
        return mlir::WalkResult::interrupt();
      }).wasInterrupted())
    return success();

  auto funcs = llvm::make_filter_range(
      module.getOps<func::FuncOp>(),
      [](const func::FuncOp &func) { return func->hasAttr("mix_mode"); });
  if (funcs.empty()) {
    module.emitWarning()
        << "[WARNING] No function with attribute mix_mode found in this module";
    return success();
  }

  if (shouldUseSoftDotScale(module)) {
    config.setEnableMixedCV(false);
    return success();
  }

  auto first =
      (*funcs.begin())->getAttrOfType<StringAttr>("mix_mode").getValue();
  if (!llvm::all_of(funcs, [&](const func::FuncOp &func) {
        return func->getAttrOfType<StringAttr>("mix_mode").getValue() == first;
      }))
    return failure();

  config.setEnableMixedCV(first != StringRef{"aiv"} ||
                          config.shouldEnableMixedCV());
  return success();
}

LogicalResult inferLayoutOptimization(ModuleOp &module,
                                      BiShengIRCompileMainConfig &config) {
  module.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (isa<linalg::BatchMatmulOp>(op)) {
      config.setEnableLayoutOptimization(false);
      return WalkResult::interrupt();
    }
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      auto callee = callOp.getCallee();
      if (callee.starts_with("triton_print")) {
        config.setEnableLayoutOptimization(false);
        return WalkResult::interrupt();
      }
    }
    if (isa<scope::ScopeOp>(op)) {
      config.setEnableLayoutOptimization(false);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success();
}

std::vector<std::string>
filterRegBaseForwardedHIVMCOptions(ArrayRef<std::string> originalCLArgs) {
  std::vector<std::string> result;
  bool skipNextArg = false;
  for (const auto &arg : originalCLArgs) {
    if (skipNextArg) {
      skipNextArg = false;
      continue;
    }

    StringRef argRef(arg);
    if (!argRef.starts_with("-"))
      continue;
    if (argRef == "-o" || argRef == "--output") {
      skipNextArg = true;
      continue;
    }
    if (argRef.starts_with("-o=") || argRef.starts_with("--output="))
      continue;
    if (argRef.starts_with("--hivmc-args"))
      continue;
    if (argRef.contains("enable-lir-compile"))
      continue;
    result.push_back(arg);
  }
  return result;
}

std::vector<std::string>
filterSharedHIVMCOptions(ArrayRef<std::string> options) {
  std::vector<std::string> result;
  for (const std::string &arg : options) {
    StringRef argRef = arg;
    if (!argRef.starts_with("-"))
      continue;

    SmallVector<StringRef> parts;
    argRef.split(parts, '=');
    if (parts.empty())
      continue;

    std::string trimArg = parts[0].trim().ltrim('-').str();
    if (!isSharedWithDownstreamToolchain(trimArg))
      continue;
    result.push_back(arg);
  }
  return result;
}

bool hasCLIArg(ArrayRef<std::string> args, StringRef argName) {
  return llvm::any_of(args, [&](const std::string &arg) {
    StringRef argRef = arg;
    if (!argRef.starts_with("-"))
      return false;
    return argRef.ltrim('-').split('=').first == argName;
  });
}

} // namespace regbase
} // namespace bishengir
