//===- ScopeOps.cpp --- Implementation of Scope dialect operations --------===//
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

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::scope;

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

ParseResult ScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  FunctionType functionType;
  if (failed(parser.parseColonType(functionType)))
    return failure();

  result.addTypes(functionType.getResults());

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

  // Parse the body region
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  ScopeOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void ScopeOp::print(OpAsmPrinter &p) {
  p << " : ";
  p.printFunctionalType(SmallVector<Type>(), getResultTypes());
  p << " ";

  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict((*this)->getAttrs());
}

void ScopeOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }
  regions.push_back(RegionSuccessor(&getRegion()));
}

void ScopeOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  // Scope always enters its region (no operands flow in due to NoRegionArguments trait)
  regions.emplace_back(&getRegion());
}

void ScopeOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.emplace_back(1, 1);
}

namespace {

/// Eliminate dead return values from scope.scope operation
/// This pattern removes results from scope.scope that have no uses
struct ScopeOpDeadReturnElimination : public OpRewritePattern<ScopeOp> {
  using OpRewritePattern<ScopeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScopeOp scopeOp,
                                PatternRewriter &rewriter) const final {
    Block &block = scopeOp.getRegion().front();
    auto returnOp = cast<ReturnOp>(block.getTerminator());

    // Identify dead results (results with no uses) using a BitVector
    BitVector isResultDead(scopeOp.getNumResults(), false);
    unsigned numDeadResults = 0;
    for (auto result : llvm::enumerate(scopeOp.getResults())) {
      if (result.value().use_empty()) {
        isResultDead.set(result.index());
        numDeadResults++;
      }
    }
    if (numDeadResults == 0)
      return failure();

    SmallVector<Type> newResultTypes;
    SmallVector<Value> newReturnOperands;
    for (auto [idx, result] : llvm::enumerate(scopeOp.getResults())) {
      if (!isResultDead.test(idx)) {
        newResultTypes.push_back(result.getType());
        newReturnOperands.push_back(returnOp.getOperand(idx));
      }
    }
    
    rewriter.setInsertionPoint(scopeOp);
    auto newScopeOp = rewriter.create<ScopeOp>(
        scopeOp.getLoc(), newResultTypes, ValueRange(), scopeOp->getAttrs());

    // Move the region
    Region &newRegion = newScopeOp.getRegion();
    rewriter.inlineRegionBefore(scopeOp.getRegion(), newRegion, 
                                newRegion.end());
    
    // Update the return operation
    Block &newBlock = newRegion.front();
    auto oldReturnOp = cast<ReturnOp>(newBlock.getTerminator());
    rewriter.setInsertionPoint(oldReturnOp);
    rewriter.replaceOpWithNewOp<ReturnOp>(oldReturnOp, newReturnOperands);
    
    unsigned newResultIdx = 0;
    for (unsigned oldIdx = 0; oldIdx < scopeOp.getNumResults(); ++oldIdx) {
      if (isResultDead[oldIdx]) {
        continue;
      } else {
        rewriter.replaceAllUsesWith(scopeOp.getResult(oldIdx),
                                    newScopeOp.getResult(newResultIdx));
        newResultIdx++;
      }
    }
    rewriter.eraseOp(scopeOp);
    return success();
  }
};

} // namespace

void ScopeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<ScopeOpDeadReturnElimination>(context);
}