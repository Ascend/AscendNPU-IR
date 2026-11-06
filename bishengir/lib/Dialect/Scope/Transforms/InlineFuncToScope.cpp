//===- InlineFuncToScope.cpp - Inline function calls into scopes ---------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
// This file implements a pass to inline function calls into scope.scope
// ops.
//
// One-shot-bufferize processes functions callee-first: a function is
// bufferized before its call sites are known, so its tensor arguments fall
// back to a fully-dynamic layout. Inlining the callee body into a scope.scope
// before bufferization removes the function boundary entirely: the ScopeOp
// bufferization interface derives result buffer types from the region
// contents, so the layout at the former call site stays precise. A subsequent
// outline-scope pass restores the function form, now with an exact signature.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "inline-func-to-scope"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_INLINEFUNCTOSCOPE
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::impl;
using namespace mlir::scope;

namespace {

// Replace `callOp` with a scope.scope holding an inlined copy of the
// single-block callee body. Callee arguments map to the call operands.
LogicalResult inlineCall(func::CallOp callOp, func::FuncOp callee) {
  // scope.scope is single-block: only single-block callee bodies can be
  // inlined without control-flow restructuring.
  if (!callee.getBody().hasOneBlock()) {
    return callOp.emitWarning()
           << "skipping inlining: callee has multiple blocks";
  }

  OpBuilder builder(callOp);
  auto scopeOp =
      builder.create<scope::ScopeOp>(callOp.getLoc(), callOp.getResultTypes());

  // Attributes are split so that the outline-scope round trip restores them
  // at the right level:
  // - Function-level attributes (hivm.vector_function, hivm.func_core_type,
  //   no_inline, ...) go onto the scope and are restored on the outlined
  //   function.
  // - Call-site annotations (e.g. ptc_simdvf) belong to the caller: they are
  //   recorded in `call_attrs_vec` and unwrapped onto the recreated call.
  // The call's `callee` symbol reference is dropped: keeping it would pin the
  // dead callee and block its erasure below; the original name is kept as a
  // plain string in `sym_func_name` instead.
  // Discardable attributes exclude the inherent ones (callee, sym_name,
  // function_type, ...); `no_inline` is inherent on both ops and carried
  // over explicitly.
  MLIRContext *ctx = callOp.getContext();
  SmallVector<NamedAttribute> callAttrs(callOp->getDiscardableAttrs());
  if (callOp.getNoInline())
    callAttrs.emplace_back(callOp.getNoInlineAttrName(), builder.getUnitAttr());
  if (!callAttrs.empty())
    scopeOp->setAttr(CallAttrsVecAttr::name,
                     CallAttrsVecAttr::get(ctx, callAttrs));
  // callee attrs
  for (NamedAttribute attr : callee->getDiscardableAttrs())
    scopeOp->setAttr(attr.getName(), attr.getValue());
  if (callee.getNoInline())
    scopeOp->setAttr(callee.getNoInlineAttrName(), builder.getUnitAttr());
  // Mark the scope so that outline-scope{outline-marked-scopes-only} outlines
  // it back.
  scopeOp->setAttr(OutlineAttr::getMnemonic(), builder.getUnitAttr());
  scopeOp->setAttr(SymFuncNameAttr::name,
                   SymFuncNameAttr::get(ctx, callee.getSymNameAttr()));

  Block &entryBlock = callee.getBody().front();
  IRMapping mapper;
  for (auto [from, to] :
       llvm::zip_equal(callee.getArguments(), callOp.getOperands()))
    mapper.map(from, to);

  // Clone the callee entry block into the scope region, dropping the
  // func.return terminator: its operands move to the scope.return.
  Block *scopeBlock = builder.createBlock(&scopeOp.getRegion());
  builder.setInsertionPointToStart(scopeBlock);
  for (Operation &op :
       llvm::make_range(entryBlock.begin(), std::prev(entryBlock.end())))
    builder.clone(op, mapper);

  auto returnOp = cast<func::ReturnOp>(entryBlock.getTerminator());
  builder.create<scope::ReturnOp>(
      returnOp.getLoc(),
      llvm::to_vector(llvm::map_range(returnOp.getOperands(), [&](Value value) {
        return mapper.lookupOrDefault(value);
      })));

  LDBG("inlined " << callee.getSymName() << " into a scope");

  callOp.replaceAllUsesWith(scopeOp.getResults());
  callOp.erase();
  return success();
}

struct InlineFuncToScopePass
    : public InlineFuncToScopeBase<InlineFuncToScopePass> {
  explicit InlineFuncToScopePass(CalleeFilter calleeFilter)
      : calleeFilter(std::move(calleeFilter)) {}
  void runOnOperation() override;

private:
  CalleeFilter calleeFilter;
};
} // namespace

void InlineFuncToScopePass::runOnOperation() {
  auto moduleOp = getOperation();
  SymbolTable symbolTable(moduleOp);

  SmallVector<func::CallOp> calls;
  moduleOp.walk([&](func::CallOp callOp) {
    auto callee = symbolTable.lookup<func::FuncOp>(callOp.getCallee());
    if (callee && calleeFilter(callee))
      calls.push_back(callOp);
  });

  if (calls.empty())
    return;

  SmallPtrSet<Operation *, 4> inlinedCallees;
  for (func::CallOp callOp : calls) {
    auto callee = symbolTable.lookup<func::FuncOp>(callOp.getCallee());
    if (failed(inlineCall(callOp, callee)))
      continue;
    inlinedCallees.insert(callee);
  }

  // Callees without remaining symbol uses are dead after inlining.
  for (Operation *op : inlinedCallees) {
    auto callee = cast<func::FuncOp>(op);
    if (SymbolTable::symbolKnownUseEmpty(callee, moduleOp))
      callee.erase();
  }
}

std::unique_ptr<Pass>
mlir::scope::createInlineFuncToScopePass(CalleeFilter calleeFilter) {
  return std::make_unique<InlineFuncToScopePass>(std::move(calleeFilter));
}
