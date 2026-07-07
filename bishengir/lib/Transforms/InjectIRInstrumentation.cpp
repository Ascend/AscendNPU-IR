//===- InjectIRInstrumentation.cpp - BiShengIR Pass Instrumentation -*- C++-*-===//

//===-------------------------------------------------------------------------===//

#include "bishengir/Transforms/InjectIRInstrumentation.h"
#include "bishengir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <map>
#include <string>

#define DEBUG_TYPE "inject-ir-pass-instrument"
#define LDBG(X) LLVM_DEBUG({ llvm::dbgs() << X << "\n"; })

using namespace mlir;
using namespace bishengir;

namespace {

/// Next 0-based call_index per (pass_id_name, op_name). Updated in InjectIR
/// runBeforePass via getPassExecutionId(..., /*update=*/true).
static std::map<std::string, int> passOpCountMap;

/// Print passOpCountMap (when -debug-only=inject-ir-pass-instrument).
static void printPassOpCountMap(llvm::StringRef extraInfo = "") {
  LLVM_DEBUG({
    llvm::dbgs() << "---------- passOpCountMap ----------\n";
    if (!extraInfo.empty()) {
      llvm::dbgs() << extraInfo << "\n";
    }
    for (const auto &p : passOpCountMap) {
      llvm::dbgs() << "  " << p.first << " -> " << p.second << "\n";
    }
    llvm::dbgs() << "-----------------------------------\n";
  });
}

/// String used for pass in PassID: CLI argument if set, else pass display name.
static std::string getPassIdName(Pass *pass) {
  StringRef arg = pass->getArgument();
  return arg.empty() ? pass->getName().str() : arg.str();
}

/// Replaces matching functions in the module with bodies from the loaded file
static LogicalResult replaceModuleWithFile(ModuleOp module, const std::string &filePath) {
  if (!llvm::sys::fs::exists(filePath)) {
    module.emitError() << "inject IR file does not exist: " << filePath;
    return failure();
  }
  ParserConfig config(module.getContext());
  auto loadedRef = parseSourceFile<ModuleOp>(filePath, config);
  if (!loadedRef) {
    module.emitError() << "failed to parse inject IR file: " << filePath;
    return failure();
  }
  ModuleOp loadedModule = loadedRef.get();
  for (func::FuncOp loadedFunc : loadedModule.getBody()->getOps<func::FuncOp>()) {
    StringRef name = loadedFunc.getSymName();
    if (name.empty()) {
      continue;
    }
    func::FuncOp currentFunc = module.lookupSymbol<func::FuncOp>(name);
    if (!currentFunc) {
      continue;
    }
    Region &currentBody = currentFunc.getBody();
    Region &loadedBody = loadedFunc.getBody();
    while (!currentBody.empty())
      currentBody.front().erase();
    IRMapping mapper;
    loadedBody.cloneInto(&currentBody, mapper);
    /// update function type if mismatch.
    currentFunc.setFunctionType(loadedFunc.getFunctionType());
  }
  return success();
}

/// Run inject-ir for one spec (pass_id@file_path). No-op if spec empty.
static LogicalResult runInjectIRForSpec(Pass *pass, Operation *op, const std::string &spec,
                                        bool update) {
  if (spec.empty()) {
    return success();
  }

  size_t atPos = spec.find('@');
  if (atPos == std::string::npos || atPos == 0 || atPos == spec.length() - 1) {
    op->emitError() << "inject-ir: invalid format, expected pass_id@file_path, got: " << spec
                    << "\n";
    return failure();
  }
  std::string targetId = spec.substr(0, atPos);
  std::string filePath = spec.substr(atPos + 1);
  std::string currentId = getPassExecutionId(pass, op, update);
  LDBG("currentId: " << currentId << ", targetId: " << targetId);
  if (currentId.empty() || currentId != targetId) {
    return success();
  }
  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module) {
    module = op->getParentOfType<ModuleOp>();
  }
  if (!module) {
    op->emitError() << "inject-ir: cannot get module for operation";
    return failure();
  }
  if (failed(replaceModuleWithFile(module, filePath))) {
    op->emitError() << "inject-ir: replace failed for file: " << filePath;
    return failure();
  }
  return success();
}

} // namespace

std::string bishengir::getOpNameForPassId(Operation *op) {
  if (isa<ModuleOp>(op)) {
    return "module";
  }
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    auto name = func.getSymName();
    return name.empty() ? "anonymous" : name.str();
  }
  return op->getName().getStringRef().str();
}

std::string bishengir::getPassExecutionId(Pass *pass, Operation *op, bool update) {
  std::string passIdName = getPassIdName(pass);
  std::string opName = getOpNameForPassId(op);
  std::string key = passIdName + "/" + opName;
  if (passOpCountMap.find(key) == passOpCountMap.end()) {
    // initialize to -1 for the first execution of this pass/op pair.
    passOpCountMap[key] = -1;
  }
  if (update) {
    passOpCountMap[key]++;
  }
  int count = passOpCountMap[key];
  return llvm::formatv("{0}/{1}/{2}", passIdName, opName, count).str();
}

void bishengir::InjectIRInstrumentation::runBeforePass(Pass *pass, Operation *op) {
  std::string id = getPassExecutionId(pass, op, /*update=*/true);
  // Run inject-ir-before and propagate errors
  if (failed(runInjectIRForSpec(pass, op, injectIrBefore, /*update=*/false))) {
    // Error is already reported via emitError, nothing more to do here
  }
  if (printPassId) {
    llvm::outs() << "[PassID] " << id << "\n";
    llvm::outs().flush();
  }
}

void bishengir::InjectIRInstrumentation::runAfterPass(Pass *pass, Operation *op) {
  // Run inject-ir-after and propagate errors
  if (failed(runInjectIRForSpec(pass, op, injectIrAfter, /*update=*/false))) {
    // Error is already reported via emitError, nothing more to do here
  }
  printPassOpCountMap("passOpCount");
}