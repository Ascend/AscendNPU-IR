//===- InjectIRInstrumentation.cpp - Pass-based IR injection --------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
//
//===----------------------------------------------------------------------===//

#include "bishengir/Transforms/InjectIRInstrumentation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

using namespace mlir;

namespace {

/// Next call index for each pass and operation pair.
static std::map<std::string, int> passOpCountMap;

static std::string getPassIdName(Pass *pass) {
  StringRef argument = pass->getArgument();
  return argument.empty() ? pass->getName().str() : argument.str();
}

static std::string getOperationIdName(Operation *op) {
  if (isa<ModuleOp>(op))
    return "module";
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    StringRef name = func.getSymName();
    return name.empty() ? "anonymous" : name.str();
  }
  return op->getName().getStringRef().str();
}

static std::string getPassExecutionId(Pass *pass, Operation *op, bool update) {
  std::string key = getPassIdName(pass) + "/" + getOperationIdName(op);
  auto [it, inserted] = passOpCountMap.try_emplace(key, -1);
  if (update)
    ++it->second;
  return llvm::formatv("{0}/{1}", key, it->second).str();
}

/// Replaces the full module body and attributes, not only matching functions.
/// Full replacement is required when the source and destination pipelines
/// produce different function symbol sets.
static LogicalResult replaceModuleWithFile(ModuleOp module,
                                           const std::string &filePath) {
  if (!llvm::sys::fs::exists(filePath))
    return module.emitError() << "inject IR file does not exist: " << filePath;

  ParserConfig config(module.getContext());
  auto loadedModule = parseSourceFile<ModuleOp>(filePath, config);
  if (!loadedModule)
    return module.emitError()
           << "failed to parse inject IR file: " << filePath;

  ModuleOp loaded = loadedModule.get();
  module->setAttrs(loaded->getAttrs());
  module->setLoc(loaded.getLoc());
  module.getBodyRegion().takeBody(loaded.getBodyRegion());

  if (failed(verify(module)))
    return module.emitError()
           << "injected module failed verification: " << filePath;
  return success();
}

static LogicalResult runInjection(Pass *pass, Operation *op,
                                  const std::string &spec) {
  if (spec.empty())
    return success();

  size_t separator = spec.find('@');
  if (separator == std::string::npos || separator == 0 ||
      separator == spec.size() - 1)
    return op->emitError()
           << "inject-ir: expected pass-id@file-path, got: " << spec;

  std::string currentId = getPassExecutionId(pass, op, /*update=*/false);
  if (currentId != spec.substr(0, separator))
    return success();

  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<ModuleOp>();
  if (!module)
    return op->emitError("inject-ir: cannot find the parent module");

  std::string filePath = spec.substr(separator + 1);
  if (failed(replaceModuleWithFile(module, filePath)))
    return failure();

  llvm::outs() << "[InjectIR] replaced module at " << currentId << " from "
               << filePath << "\n";
  llvm::outs().flush();
  return success();
}

} // namespace

void bishengir::InjectIRInstrumentation::runBeforePass(Pass *pass,
                                                       Operation *op) {
  std::string id = getPassExecutionId(pass, op, /*update=*/true);
  (void)runInjection(pass, op, injectIrBefore);
  if (printPassId) {
    llvm::outs() << "[PassID] " << id << "\n";
    llvm::outs().flush();
  }
}

void bishengir::InjectIRInstrumentation::runAfterPass(Pass *pass,
                                                      Operation *op) {
  (void)runInjection(pass, op, injectIrAfter);
}
