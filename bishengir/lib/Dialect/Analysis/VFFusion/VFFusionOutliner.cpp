//===- VFFusionOutliner.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/VFFusion/VFFusionOutliner.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/IR/Value.h"

#define DEBUG_TYPE "vf-fusion-outliner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace llvm;

namespace mlir {
namespace analysis {

void cloneFuncAttrs(func::FuncOp from, func::FuncOp to,
                    ArrayRef<StringRef> _ignoreAttrNames = {}) {
  // Core func op attributes that define the function signature and symbol.
  // These are handled separately during cloning or already set by the builder.
  const auto funcAttrNames = from.getAttributeNames();
  DenseSet<StringRef> ignoreAttrNames{_ignoreAttrNames.begin(),
                                      _ignoreAttrNames.end()};
  ignoreAttrNames.insert(funcAttrNames.begin(), funcAttrNames.end());
  for (const NamedAttribute &attr : from->getAttrs()) {
    if (ignoreAttrNames.contains(attr.getName()))
      continue;
    to->setAttr(attr.getName(), attr.getValue());
  }
}

FailureOr<func::FuncOp> createFusedFunction(func::FuncOp funcOp,
                                            VFFusionBlock &fusedBlock,
                                            OpBuilder &builder) {
  SmallVector<Operation *> vectorOps(fusedBlock.getOps());
#ifndef NDEBUG
  for (Operation *op : vectorOps)
    LDBG("outlining " << op->getName());
#endif

  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  const std::string prefixFunctionName = funcOp.getSymName().str() + "_fused";

  SetVector<Operation *> ops(vectorOps.begin(), vectorOps.end());
  SmallVector<Value> inputs(fusedBlock.recomputeInputs().takeVector());
  SmallVector<Type> outTypes;

  for (Value out : fusedBlock.recomputeOutputs())
    outTypes.push_back(out.getType());

  builder.setInsertionPoint(funcOp);
  FunctionType funcTy = builder.getFunctionType(TypeRange(inputs), outTypes);

  func::FuncOp newFuncOp =
      builder.create<func::FuncOp>(funcOp->getLoc(), prefixFunctionName, funcTy,
                                   SmallVector<NamedAttribute>());
  SymbolTable symbolTable(moduleOp);
  auto funcName =
      symbolTable.renameToUnique(newFuncOp, SmallVector<SymbolTable *>());
  if (failed(funcName))
    return failure();

  cloneFuncAttrs(
      funcOp, newFuncOp,
      mlir::hacc::stringifyEnum(mlir::hacc::HACCToLLVMIRTranslateAttr::ENTRY));

  // set fused function to be private so that it can be inlined and removed
  // safely
  newFuncOp.setVisibility(SymbolTable::Visibility::Private);
  return newFuncOp;
}

// classic logic of FunctionOp creation.
FailureOr<func::FuncOp> VFFusionOutliner::outline(func::FuncOp funcOp,
                                                  VFFusionBlock &fusedBlock,
                                                  OpBuilder &builder) {
  LDBG("Outlining fused block");
  auto maybeNewFuncOp = createFusedFunction(funcOp, fusedBlock, builder);
  if (failed(maybeNewFuncOp))
    return failure();

  func::FuncOp newFuncOp = maybeNewFuncOp.value();

  // Create function body
  Block *entryBB = newFuncOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBB);

  // Clone operations and replace usages
  LDBG("Pushing outlined operations");
  for (auto [oldIn, newIn] : llvm::zip_equal(
           fusedBlock.recomputeInputs().takeVector(), entryBB->getArguments()))
    mapFromCloning.map(oldIn, newIn);

  SetVector<Operation *> newOps;
  for (auto op : fusedBlock.getOps()) {
    newOps.insert(builder.clone(*op, mapFromCloning));
    LDBG("Cloning " << *op);
  }

  SmallVector<Value> outs;
  for (Value out : fusedBlock.recomputeOutputs())
    outs.push_back(mapFromCloning.lookup(out));

  builder.create<func::ReturnOp>(newFuncOp.getLoc(), outs);
  utils::eraseTriviallyDeadOps(newOps.getArrayRef());
  LDBG("Created FuncOp for fused kernel\n" << *newFuncOp);
  return newFuncOp;
}

// classic logic of FunctionOp creation.
LogicalResult VFFusionOutliner::createInvoke(func::FuncOp fusedFunction,
                                             VFFusionBlock &fusedBlock,
                                             OpBuilder &builder) {
  LDBG("Replacing invoke with callOp");
  SmallVector<Operation *> vectorOps(fusedBlock.getOps());
  SetVector<Operation *> ops(vectorOps.begin(), vectorOps.end());

  // last ops of the fusedBlock
  builder.setInsertionPointAfter(vectorOps.back());
  func::CallOp callOp =
      builder.create<func::CallOp>(vectorOps.back()->getLoc(), fusedFunction,
                                   fusedBlock.recomputeInputs().takeVector());

  // replace all uses with callOp
  // based on the mapping
  LDBG("Replacing uses");
  for (auto [oldOut, newOut] :
       llvm::zip_equal(fusedBlock.recomputeOutputs(), callOp->getResults())) {
    ((Value)oldOut).replaceAllUsesWith(newOut);
  }

  for (Operation *op : reverse(vectorOps))
    op->erase();
  LDBG(*callOp->getParentOfType<ModuleOp>());
  return success();
}

} // namespace analysis
} // namespace mlir
