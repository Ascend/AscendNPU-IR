//===- TransformOps.cpp - Analysis transform ops implementation ----------===//
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
// This file implements dialect-neutral transform ops used by the Analysis
// schedule infrastructure. The ops only depend on generic dialects (func,
// tensor, linalg).
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/Transforms/TransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::transform;

namespace {

//===----------------------------------------------------------------------===//
// Generic reshape/slice tracing.
// Mirrors hfusion::traceReshapeOrSliceSingleConsumer{,OrSelf} but restricted
// to generic tensor dialect ops so this layer does not depend on HFusion.
//===----------------------------------------------------------------------===//

bool isEmptyLikeTensor(Value op);
bool isEmptyLikeTensor(Value op) {
  Operation *defOp = op.getDefiningOp();
  if (!defOp)
    return false;
  if (isa<tensor::EmptyOp>(defOp))
    return true;
  if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(defOp))
    return isEmptyLikeTensor(collapseOp.getSrc());
  if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(defOp))
    return isEmptyLikeTensor(expandOp.getSrc());
  if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(defOp))
    return isEmptyLikeTensor(extractOp.getSource());
  return false;
}

bool isReshapeOrSliceOp(Operation *op) {
  return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
             tensor::ExtractSliceOp, tensor::InsertSliceOp>(op);
}

Value getReshapeOrSliceResult(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp op) { return op.getResult(); })
      .Case([](tensor::CollapseShapeOp op) { return op.getResult(); })
      .Case([](tensor::ExtractSliceOp op) { return op.getResult(); })
      .Case([](tensor::InsertSliceOp op) { return op.getResult(); })
      .Default([](Operation *op) {
        llvm::report_fatal_error("Unsupported reshape or slice op");
        return Value();
      });
}

Value getReshapeOrSliceSource(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp op) { return op.getSrc(); })
      .Case([](tensor::CollapseShapeOp op) { return op.getSrc(); })
      .Case([](tensor::ExtractSliceOp op) { return op.getSource(); })
      .Case([](tensor::InsertSliceOp op) { return op.getSource(); })
      .Default([](Operation *op) {
        llvm::report_fatal_error("Unsupported reshape or slice op");
        return Value();
      });
}

FailureOr<Value> traceReshapeOrSliceSingleConsumer(Value input);
Value traceReshapeOrSliceSingleConsumerOrSelf(Value input) {
  auto maybeValue = traceReshapeOrSliceSingleConsumer(input);
  if (succeeded(maybeValue))
    return maybeValue.value();
  return input;
}

FailureOr<Value> traceReshapeOrSliceSingleConsumer(Value input) {
  auto reshapeUsers = llvm::make_filter_range(
      input.getUsers(),
      [&](Operation *user) { return isReshapeOrSliceOp(user); });
  if (!llvm::hasSingleElement(reshapeUsers))
    return failure();

  return traceReshapeOrSliceSingleConsumerOrSelf(
      getReshapeOrSliceResult(*reshapeUsers.begin()));
}

FailureOr<Value> traceReshapeOrSliceSingleProducer(Value input);
Value traceReshapeOrSliceSingleProducerOrSelf(Value input) {
  auto maybeValue = traceReshapeOrSliceSingleProducer(input);
  if (succeeded(maybeValue))
    return maybeValue.value();
  return input;
}

FailureOr<Value> traceReshapeOrSliceSingleProducer(Value input) {
  if (isa<BlockArgument>(input))
    return failure();

  auto result = cast<OpResult>(input);
  auto *definingOp = result.getOwner();
  if (!isReshapeOrSliceOp(definingOp))
    return failure();

  return traceReshapeOrSliceSingleProducerOrSelf(
      getReshapeOrSliceSource(definingOp));
}

} // namespace

//===----------------------------------------------------------------------===//
// GetFuncArgumentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
GetFuncArgumentOp::apply(TransformRewriter &rewriter,
                         TransformResults &transformResults,
                         TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires exactly one target handle!";

  auto func = dyn_cast_or_null<func::FuncOp>(*payloadOps.begin());
  if (!func)
    return emitDefiniteFailure()
           << "target handle does not point to `func.func` op";

  Region::BlockArgListType funcArgs = func.getArguments();
  SmallVector<int64_t> operandPositions;
  DiagnosedSilenceableFailure diag = expandTargetSpecification(
      getLoc(), getIsAll(), getIsInverted(), getRawPositionList(),
      func.getNumArguments(), operandPositions);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(func->getLoc())
        << "while considering positions of this payload operation";
    return diag;
  }
  SmallVector<Value> selectedArgs = llvm::map_to_vector(
      operandPositions, [&](int64_t pos) { return Value(funcArgs[pos]); });
  if (getFindReshapeConsumer()) {
    for (auto [idx, v] : llvm::enumerate(selectedArgs)) {
      auto maybeResult = traceReshapeOrSliceSingleConsumer(v);
      if (failed(maybeResult))
        return emitDefiniteFailure()
               << "cannot trace to single reshape consumer for " << v;
      v = maybeResult.value();
    }
  }
  transformResults.setValues(llvm::cast<OpResult>(getOutputs()), selectedArgs);
  return DiagnosedSilenceableFailure::success();
}

void GetFuncArgumentOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure ReverseOp::apply(TransformRewriter &rewriter,
                                             TransformResults &transformResults,
                                             TransformState &state) {
  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  SmallVector<Operation *> reversedOperations = {targets.rbegin(),
                                                 targets.rend()};
  transformResults.set(cast<OpResult>(getResult()), reversedOperations);
  return DiagnosedSilenceableFailure::success();
}

void ReverseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// GetFuncResultOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
GetFuncResultOp::apply(TransformRewriter &rewriter,
                       TransformResults &transformResults,
                       TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires exactly one target handle!";

  auto func = dyn_cast_or_null<func::FuncOp>(*payloadOps.begin());
  if (!func)
    return emitDefiniteFailure()
           << "target handle does not point to `func.func` op";

  func::ReturnOp returnOp = nullptr;
  func->walk([&returnOp](func::ReturnOp op) { returnOp = op; });
  if (!returnOp)
    return emitDefiniteFailure() << "cannot find return op in func!";

  SmallVector<int64_t> operandPositions;
  DiagnosedSilenceableFailure diag = expandTargetSpecification(
      getLoc(), getIsAll(), getIsInverted(), getRawPositionList(),
      func.getNumResults(), operandPositions);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(func->getLoc())
        << "while considering positions of this payload operation";
    return diag;
  }
  SmallVector<Value> selectedResult =
      llvm::map_to_vector(operandPositions, [&](int64_t pos) {
        return returnOp->getOpOperand(pos).get();
      });
  if (getFindReshapeProducer()) {
    for (auto [idx, v] : llvm::enumerate(selectedResult)) {
      auto maybeResult = traceReshapeOrSliceSingleProducer(v);
      if (failed(maybeResult))
        return emitDefiniteFailure()
               << "cannot trace to single reshape producer for " << v;
      v = maybeResult.value();
    }
  }
  transformResults.setValues(llvm::cast<OpResult>(getOutputs()),
                             selectedResult);
  return DiagnosedSilenceableFailure::success();
}

void GetFuncResultOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}


//===---------------------------------------------------------------------===//
// MatchAncestorOfOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MatchAncestorOfOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  llvm::StringSet<> strs;
  if (getOps().has_value())
    strs.insert(getOps()->getAsValueRange<StringAttr>().begin(),
                getOps()->getAsValueRange<StringAttr>().end());

  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps)) {
    return emitDefiniteFailure("requires exactly one target handle");
  }

  auto childOps = state.getPayloadOps(getChild());
  if (!llvm::hasSingleElement(childOps)) {
    return emitDefiniteFailure("requires exactly one child handle");
  }
  Operation *childOp = *childOps.begin();
  // Build dominance info from enclosing function
  func::FuncOp enclosingFunc = childOp->getParentOfType<func::FuncOp>();
  DominanceInfo domInfo(enclosingFunc);

  SmallVector<Operation *> res;
  bool incorrectNumOperandTypes = false;
  auto matchFun = [&](Operation *op) {
    if (getOps().has_value() && !strs.contains(op->getName().getStringRef()))
      return;

    // Interfaces cannot be matched by name, just by ID.
    // So we specifically encode the interfaces we care about for this op.
    if (getInterface().has_value()) {
      auto iface = getInterface().value();
      if (iface == transform::MatchInterfaceEnum::LinalgOp &&
          !isa<linalg::LinalgOp>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::TilingInterface &&
          !isa<TilingInterface>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::LoopLikeInterface &&
          !isa<LoopLikeOpInterface>(op))
        return;
    }

    // Check if all specified attributes match.
    if (getOpAttrs().has_value()) {
      DictionaryAttr opAttrs = getOpAttrs().value();
      for (NamedAttribute attr : opAttrs) {
        if (attr.getName() == getInterfaceAttrName() ||
            attr.getName() == getOpsAttrName())
          continue;
        if (!op->hasAttr(attr.getName()))
          return;
        if (op->getAttr(attr.getName()) != attr.getValue())
          return;
      }
    }

    // Check if at least one of the optional attributes match.
    if (getOptionalOpAttrs().has_value() &&
        !getOptionalOpAttrs().value().empty()) {
      DictionaryAttr optionalOpAttrs = getOptionalOpAttrs().value();
      if (llvm::none_of(optionalOpAttrs, [&](NamedAttribute attr) {
            if (!op->hasAttr(attr.getName()))
              return false;

            if (op->getAttr(attr.getName()) != attr.getValue())
              return false;

            return true;
          }))
        return;
    }

    if (getFilterResultType().has_value()) {
      Type t = getFilterResultType().value();
      if (op->getNumResults() != 1 || op->getResultTypes().front() != t)
        return;
    }

    if (getFilterOperandTypes().has_value()) {
      ArrayAttr types = getFilterOperandTypes().value();
      auto operandTypes = op->getOperandTypes();

      if (types.size() == 1) {
        // All the operands must be equal to the specified type
        auto typeattr = dyn_cast<TypeAttr>(getFilterOperandTypes().value()[0]);
        auto t = cast<Type>(typeattr.getValue());
        if (!llvm::all_of(op->getOperandTypes(),
                          [&](Type operandType) { return operandType == t; }))
          return;
      } else {
        // The operand types must match all the types in the list (in the same
        // order in with they are specified)
        if (types.size() != operandTypes.size()) {
          incorrectNumOperandTypes = true;
          return;
        }

        for (auto [attr, operandType] :
             llvm::zip_equal(getFilterOperandTypes().value(), operandTypes)) {
          auto typeattr = cast<TypeAttr>(attr);
          auto type = cast<Type>(typeattr.getValue());
          if (type != operandType)
            return;
        }
      }
    }

    if (!domInfo.properlyDominates(op, childOp))
      return;

    // All constraints are satisfied.
    res.push_back(op);
    return;
  };

  (*payloadOps.begin())->walk(matchFun);
  if (incorrectNumOperandTypes)
    return emitDefiniteFailure("If filter_operand_types contains more than a "
                               "type, then it must contain as much types as "
                               "the number of operands in the target ops");
  results.set(cast<OpResult>(getResult()), res);
  return DiagnosedSilenceableFailure::success();
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         Value child,
                                         ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addOperands(child);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result,
                                         TypeRange resultTypes, Value target,
                                         Value child,
                                         ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addOperands(child);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(resultTypes);
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         Value child, ArrayAttr ops,
                                         DictionaryAttr op_attrs,
                                         DictionaryAttr optional_op_attrs) {
  result.addOperands(target);
  result.addOperands(child);
  if (ops)
    result.addAttribute(MatchAncestorOfOp::getOpsAttrName(result.name), ops);
  if (op_attrs)
    result.addAttribute(MatchAncestorOfOp::getOpAttrsAttrName(result.name),
                        op_attrs);
  if (optional_op_attrs)
    result.addAttribute(
        MatchAncestorOfOp::getOptionalOpAttrsAttrName(result.name),
        optional_op_attrs);
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         Value child, ArrayAttr ops,
                                         DictionaryAttr op_attrs) {
  result.addOperands(target);
  result.addOperands(child);
  if (ops)
    result.addAttribute(MatchAncestorOfOp::getOpsAttrName(result.name), ops);
  if (op_attrs)
    result.addAttribute(MatchAncestorOfOp::getOpAttrsAttrName(result.name),
                        op_attrs);
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}


//===----------------------------------------------------------------------===//
// ExtendedLoopOutlineOp
//===----------------------------------------------------------------------===//

namespace {

inline SmallVector<scf::ForOp> collectLoops(const SmallVector<Value> &targets,
                                            transform::TransformState &state) {
  SmallVector<scf::ForOp> loops;
  DominanceInfo domInfo;
  for (Value target : targets) {
    if (state.getPayloadOps(target).empty()) {
      assert(false && "payload op is empty.");
    }
    Operation *loop = *state.getPayloadOps(target).begin();
    assert(llvm::hasSingleElement(state.getPayloadOps(target)) &&
           "expect single element.");
    loops.push_back(dyn_cast<scf::ForOp>(loop));
  }
  llvm::sort(loops, [&domInfo](scf::ForOp a, scf::ForOp b) {
    return domInfo.properlyDominates(a, b, false);
  });
  return loops;
}

inline void getResultsUsedBelow(const SmallVector<Operation *> &ops,
                                Operation *below, SmallVector<Value> &results) {
  DominanceInfo domInfo;
  for (Operation *op : ops) {
    for (Value res : op->getResults()) {
      for (OpOperand &use : res.getUses()) {
        if (domInfo.properlyDominates(below, use.getOwner(), false)) {
          results.push_back(res);
          break;
        }
      }
    }
  }
}

static Operation *buildCopyOpForValue(Location loc, Value from,
                                      transform::TransformRewriter &rewriter) {
  auto rankedTy = dyn_cast<RankedTensorType>(from.getType());
  if (!rankedTy)
    return nullptr;
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(rewriter, loc, from);
  Value empty =
      rewriter.create<tensor::EmptyOp>(loc, sizes, rankedTy.getElementType());
  if (isEmptyLikeTensor(from))
    return empty.getDefiningOp();
  return rewriter.create<linalg::CopyOp>(loc, from, empty);
}

static void
duplicateReusedValuesForSCFForOp(SmallVector<scf::ForOp> loops,
                                 transform::TransformRewriter &rewriter) {
  DenseSet<Value> set;
  for (auto loop : loops) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(loop);
    auto loc = loop.getLoc();
    for (OpOperand &iterArg : loop.getInitArgsMutable()) {
      auto value = iterArg.get();
      if (!set.contains(value)) {
        set.insert(value);
      } else {
        if (Operation *newOp = buildCopyOpForValue(loc, value, rewriter))
          iterArg.assign(newOp->getResult(0));
      }
    }
    loop.getBody()->walk([&](Operation *op) {
      for (unsigned i = 0; i < op->getOperands().size(); i++) {
        Value value = op->getOperand(i);
        if (set.contains(value)) {
          if (Operation *newOp = buildCopyOpForValue(loc, value, rewriter))
            op->setOperand(i, newOp->getResult(0));
        }
      }
    });
  }
}

inline SmallVector<Operation *>
collectAllDefiningOpsOfForOp(scf::ForOp consumer) {
  SmallVector<Operation *> definingOps;
  for (Value val : consumer.getInitArgs()) {
    if (val.getDefiningOp())
      definingOps.push_back(val.getDefiningOp());
  }
  for (Value val : consumer.getOperands()) {
    if (val.getDefiningOp())
      definingOps.push_back(val.getDefiningOp());
  }
  visitUsedValuesDefinedAbove(consumer->getRegion(0), consumer->getRegion(0),
                              [&definingOps](OpOperand *operand) {
                                definingOps.push_back(
                                    operand->get().getDefiningOp());
                              });
  return definingOps;
}

// For pair of loops: {producer_loop, consumer_loop},
// 1. recursivly collect users of producer_loop until meet consumer_loop,
//    into prevUsers Set.
// 2. recursivly collect consumer_loop definingOp, until meet producer_loop,
//    if exists in prevUsers, push into intermediateOps vector.
// 3. sort them with dominance order.
inline void
collectAndGroupProducerUsers(scf::ForOp producer, scf::ForOp consumer,
                             SmallVector<Operation *> &intermediateOps,
                             SmallVector<Operation *> &nonIntermediateOps) {
  DominanceInfo domInfo;
  if (!domInfo.properlyDominates(producer, consumer, false)) {
    return;
  }

  SmallVector<Operation *> stack = {producer};
  DenseSet<Operation *> users;
  DenseSet<Operation *> definingOps;

  while (!stack.empty()) {
    Operation *cur = stack.pop_back_val();
    for (Operation *curUser : cur->getUsers()) {
      if (users.find(curUser) == users.end() &&
          domInfo.properlyDominates(curUser, consumer, false)) {
        users.insert(curUser);
        stack.push_back(curUser);
      }
    }
  }

  stack.append(collectAllDefiningOpsOfForOp(consumer));

  while (!stack.empty()) {
    Operation *cur = stack.pop_back_val();
    if (users.find(cur) != users.end()) {
      definingOps.insert(cur);
      for (Value operand : cur->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          stack.push_back(defOp);
        }
      }
    }
  }

  intermediateOps = {definingOps.begin(), definingOps.end()};
  llvm::stable_sort(intermediateOps, [&domInfo](Operation *a, Operation *b) {
    return domInfo.properlyDominates(a, b, false);
  });

  for (Operation *item : users) {
    if (definingOps.find(item) == definingOps.end()) {
      nonIntermediateOps.push_back(item);
    }
  }
  llvm::stable_sort(nonIntermediateOps, [&domInfo](Operation *a, Operation *b) {
    return domInfo.properlyDominates(a, b, false);
  });
}

inline void
collectAndGroupOperations(const SmallVector<scf::ForOp> &loops,
                          transform::TransformState &state,
                          SmallVector<Operation *> &opsToOutline,
                          SmallVector<Operation *> &opsToAdjustPositon) {
  scf::ForOp prev = nullptr;
  for (auto cur : loops) {
    if (prev) {
      SmallVector<Operation *> intermediateOps;
      collectAndGroupProducerUsers(prev, cur, intermediateOps,
                                   opsToAdjustPositon);
      opsToOutline.append(intermediateOps);
    }

    opsToOutline.push_back(cur);
    prev = cur;
  }
}

inline void adjustOperationPositionsAfter(
    const SmallVector<Operation *> &opsToAdjustPositon, Operation *after,
    transform::TransformRewriter &rewriter) {
  DominanceInfo domInfo;
  OpBuilder::InsertionGuard g(rewriter);
  for (Operation *op : llvm::reverse(opsToAdjustPositon)) {
    rewriter.setInsertionPointAfter(after);
    Operation *newOp = rewriter.clone(*op);
    bool _;
    rewriter.replaceUsesWithIf(
        op->getResults(), newOp->getResults(),
        [&after, &domInfo](OpOperand &operand) -> bool {
          return domInfo.properlyDominates(after, operand.getOwner(), false);
        },
        &_);
    rewriter.eraseOp(op);
  }
}

inline void initAndOutlineOpsIntoRegion(
    const SmallVector<Operation *> &ops, const SmallVector<Value> &origResults,
    const SmallVector<Type> &yieldResultsTypes,
    SmallVector<Value> &yieldResults, scf::ExecuteRegionOp &executeRegionOp,
    Operation *&symbolTableOp, transform::TransformRewriter &rewriter) {
  IRMapping mapper;
  OpBuilder::InsertionGuard g(rewriter);
  for (auto riter = ops.rbegin(); riter != ops.rend(); riter++) {
    Operation *op = *riter;
    if (!executeRegionOp) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(op);
      executeRegionOp = rewriter.create<scf::ExecuteRegionOp>(
          op->getLoc(), yieldResultsTypes);
      symbolTableOp = SymbolTable::getNearestSymbolTable(op);
      executeRegionOp.getRegion().emplaceBlock();
    }

    rewriter.setInsertionPointToStart(&executeRegionOp.getRegion().back());

    Operation *clonedOp = nullptr;
    if (isa<scf::ForOp>(op)) {
      clonedOp = rewriter.cloneWithoutRegions(*op);
      Region &clonedRegion = clonedOp->getRegions().front();
      assert(clonedRegion.empty() && "expected empty region");

      rewriter.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                                  clonedRegion.end());
    } else {
      clonedOp = rewriter.clone(*op);
    }

    mapper.map(op->getResults(), clonedOp->getResults());

    bool _;
    rewriter.replaceOpUsesWithIf(
        op, clonedOp->getResults(),
        [&](OpOperand &use) {
          return executeRegionOp->isAncestor(use.getOwner());
        },
        &_);
  }

  for (auto item : origResults) {
    yieldResults.push_back(mapper.lookup(item));
  }

  rewriter.setInsertionPointToEnd(&executeRegionOp.getRegion().back());
  rewriter.create<scf::YieldOp>(executeRegionOp.getLoc(), yieldResults);

  rewriter.replaceAllUsesWith(origResults, executeRegionOp->getResults());
}

} // namespace

// Extract all result types for return.
DiagnosedSilenceableFailure
transform::ExtendedLoopOutlineOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  SmallVector<Operation *> functions;
  SmallVector<Operation *> calls;
  DenseMap<Operation *, SymbolTable> symbolTables;

  SmallVector<Value> targets = getTargets();
  SmallVector<scf::ForOp> loops = collectLoops(targets, state);

  // When outline loop as VF, some reused values inside this loop will cause
  // memref.alloc and memref.copy which is illegal inside VF after
  // bufferization. Here we duplicate these reused values to avoid this. See
  // issue:
  // https://codehub-y.huawei.com/CompilerKernel/BiShengKernel/BiSheng/issues/3395
  duplicateReusedValuesForSCFForOp(loops, rewriter);

  SmallVector<Operation *> ops;
  SmallVector<Operation *> opsToAdjustPositon;
  collectAndGroupOperations(loops, state, ops, opsToAdjustPositon);
  adjustOperationPositionsAfter(opsToAdjustPositon, loops.back(), rewriter);

  scf::ExecuteRegionOp executeRegionOp = nullptr;
  Operation *symbolTableOp = nullptr;

  SmallVector<Type> yieldResultsTypes;
  SmallVector<Value> yieldResults;
  SmallVector<Value> origResults;

  getResultsUsedBelow(ops, loops.back(), origResults);
  for (Value origResult : origResults) {
    yieldResultsTypes.push_back(origResult.getType());
  }

  initAndOutlineOpsIntoRegion(ops, origResults, yieldResultsTypes, yieldResults,
                              executeRegionOp, symbolTableOp, rewriter);

  for (Operation *op : llvm::reverse(ops)) {
    if (op->use_empty())
      rewriter.eraseOp(op);
  }

  if (!executeRegionOp) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "failed to outline";
    return diag;
  }

  // Build the symbol table *before* outlineSingleBlockRegion below inserts
  // the new func::FuncOp under its raw, possibly-already-taken name:
  // SymbolTable's constructor asserts on any existing name collision, while
  // SymbolTable::insert() (used below) safely auto-renames on collision.
  SymbolTable *symbolTable = nullptr;
  if (symbolTableOp) {
    symbolTable = &symbolTables.try_emplace(symbolTableOp, symbolTableOp)
                       .first->getSecond();
  }

  func::CallOp call;
  FailureOr<func::FuncOp> outlined = outlineSingleBlockRegion(
      rewriter, executeRegionOp->getLoc(), executeRegionOp.getRegion(),
      getFuncName(), &call);

  if (failed(outlined)) {
    return emitDefaultDefiniteFailure(executeRegionOp);
  }

  if (symbolTable) {
    symbolTable->insert(*outlined);
    call.setCalleeAttr(FlatSymbolRefAttr::get(*outlined));
  }

  functions.push_back(*outlined);
  calls.push_back(call);
  results.set(cast<OpResult>(getFunction()), functions);
  results.set(cast<OpResult>(getCall()), calls);

  return DiagnosedSilenceableFailure::success();
}


//===----------------------------------------------------------------------===//
// Helper functions for MergeProducerExtractUsesOp
//===----------------------------------------------------------------------===//

static bool isValidSliceOpInContainingOp(tensor::ExtractSliceOp sliceOp,
                                         scf::ForOp containingOp) {
  if (!sliceOp || !containingOp->isProperAncestor(sliceOp)) {
    return false;
  }

  auto staticStrides = sliceOp.getStaticStrides();
  if (llvm::count_if(staticStrides, [](int64_t s) { return s != 1; }) > 0) {
    // only handle extract slice with stride 1
    return false;
  }

  return true;
}

static void
findCommonAncesterForOps(SmallVector<tensor::ExtractSliceOp> &extractUserOps,
                         SmallVector<scf::ForOp> &commonAncesterForOps,
                         scf::ForOp containingForOp) {
  assert(extractUserOps.size() > 1);
  scf::ForOp ancestorForOp = extractUserOps[0]->getParentOfType<scf::ForOp>();
  while (containingForOp->isAncestor(ancestorForOp)) {
    bool isAncestorOfAll = true;
    for (size_t i = 1; i < extractUserOps.size(); ++i) {
      if (!ancestorForOp->isAncestor(extractUserOps[i])) {
        isAncestorOfAll = false;
        break;
      }
    }
    if (isAncestorOfAll)
      break;
    ancestorForOp = ancestorForOp->getParentOfType<scf::ForOp>();
  }
  while (ancestorForOp && containingForOp->isAncestor(ancestorForOp)) {
    commonAncesterForOps.push_back(ancestorForOp);
    ancestorForOp = ancestorForOp->getParentOfType<scf::ForOp>();
  }
}

static bool hasSameOffset(unsigned i, OpFoldResult offset,
                          SmallVector<tensor::ExtractSliceOp> extractUserOps) {
  for (auto extractUserOp : extractUserOps) {
    SmallVector<OpFoldResult> offsets = extractUserOp.getMixedOffsets();
    if (offsets[i] != offset)
      return false;
  }
  return true;
}

// If there are multiple consumers of producer in containing op, only fuse
// producer once into the first extractSliceOp of producer. For example:
//   %1 = linalg.generic    // producer op
//   scf.for                // containing op
//     %2 = tensor.extract_slice %1
//     %3 = linalg.generic ins(%2)
//     %4 = tensor.extract_slice %1
//     %5 = linalg.generic ins(%4)
// after merge these two extract uses,there is only one consumer of producer:
//   %1 = linalg.generic    // producer op
//   scf.for                // containing op
//     %2 = tensor.extract_slice %1
//     %3 = tensor.extract_slice %2
//     %4 = linalg.generic ins(%3)
//     %5 = tensor.extract_slice %2
//     %6 = linalg.generic ins(%5)
// another example of there are nested scf.for:
//   %1 = linalg.generic    // producer op
//   scf.for                // containing op
//     scf.for
//       %2 = tensor.extract_slice %1
//       %3 = linalg.generic ins(%2)
//     scf.for
//       %4 = tensor.extract_slice %1
//       %5 = linalg.generic ins(%4)
// after merge these two extract uses:
//   %1 = linalg.generic    // producer op
//   scf.for                // containing op
//     %2 = tensor.extract_slice %1
//     scf.for
//       %3 = tensor.extract_slice %2
//       %4 = linalg.generic ins(%3)
//     scf.for
//       %5 = tensor.extract_slice %2
//       %6 = linalg.generic ins(%5)
static void
findAndMergeMultipleExtractUses(transform::TransformRewriter &rewriter,
                                Operation *producerOp,
                                scf::ForOp containingForOp) {
  if (producerOp->getNumResults() != 1)
    return;

  for (Value res : producerOp->getResults()) {
    SmallVector<tensor::ExtractSliceOp> extractUserOps;
    for (auto user : res.getUsers()) {
      auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (isValidSliceOpInContainingOp(extractSliceOp, containingForOp))
        extractUserOps.push_back(extractSliceOp);
    }
    if (extractUserOps.size() <= 1)
      continue;

    SmallVector<scf::ForOp> commonAncesterForOps;
    findCommonAncesterForOps(extractUserOps, commonAncesterForOps,
                             containingForOp);
    assert(commonAncesterForOps.size() >= 1);
    DenseSet<Value> inductionVarsOfCommonAncesterForOps;
    for (scf::ForOp commonAncesterForOp : commonAncesterForOps) {
      inductionVarsOfCommonAncesterForOps.insert(
          commonAncesterForOp.getInductionVar());
    }

    scf::ForOp innermostCommonAncesterForOp = commonAncesterForOps[0];
    llvm::sort(extractUserOps, [&](Operation *op1, Operation *op2) {
      Operation *current1 = op1;
      while (auto anscestorForOp = current1->getParentOfType<scf::ForOp>()) {
        if (anscestorForOp == innermostCommonAncesterForOp)
          break;
        current1 = anscestorForOp.getOperation();
      }
      Operation *current2 = op2;
      while (auto anscestorForOp = current2->getParentOfType<scf::ForOp>()) {
        if (anscestorForOp == innermostCommonAncesterForOp)
          break;
        current2 = anscestorForOp.getOperation();
      }
      return current1->isBeforeInBlock(current2);
    });

    // Build new extract slice op inside innermost commonAncesterForOp, which
    // will be used by all inner extract slice ops.
    tensor::ExtractSliceOp firstExtractUserOp = extractUserOps[0];
    SmallVector<OpFoldResult> offsets = firstExtractUserOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = firstExtractUserOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = firstExtractUserOp.getMixedStrides();
    SmallVector<OpFoldResult> newOuterOffsets;
    SmallVector<OpFoldResult> newOuterSizes;
    OpFoldResult zeroAttr = rewriter.getIndexAttr(0);
    SmallVector<bool> hasSameOffsetForAllExtractUserOps(offsets.size(), false);

    Operation *insertionPoint = firstExtractUserOp;
    while (auto anscestorForOp =
               insertionPoint->getParentOfType<scf::ForOp>()) {
      if (anscestorForOp == innermostCommonAncesterForOp)
        break;
      insertionPoint = anscestorForOp.getOperation();
    }
    rewriter.setInsertionPoint(insertionPoint);

    Location loc = firstExtractUserOp.getLoc();
    auto resShapedType = cast<ShapedType>(res.getType());
    for (unsigned i = 0; i < offsets.size(); i++) {
      if (inductionVarsOfCommonAncesterForOps.contains(
              llvm::dyn_cast_if_present<Value>(offsets[i])) &&
          hasSameOffset(i, offsets[i], extractUserOps)) {
        newOuterOffsets.push_back(offsets[i]);
        newOuterSizes.push_back(sizes[i]);
        hasSameOffsetForAllExtractUserOps[i] = true;
        continue;
      }
      newOuterOffsets.push_back(zeroAttr);
      int64_t inputShape = resShapedType.getShape()[i];
      // For a dynamic dimension getShape() returns ShapedType::kDynamic
      // (INT64_MIN), and will be truncated to 0.
      if (ShapedType::isDynamic(inputShape))
        newOuterSizes.push_back(rewriter.createOrFold<tensor::DimOp>(
            loc, res, static_cast<int64_t>(i)));
      else
        newOuterSizes.push_back(rewriter.getIndexAttr(inputShape));
    }

    tensor::ExtractSliceOp newOuterExtractSliceOp =
        rewriter.create<tensor::ExtractSliceOp>(loc, res, newOuterOffsets,
                                                newOuterSizes, strides);

    // Build new inner extract slice op for every extractUserOp
    for (tensor::ExtractSliceOp extractUserOp : extractUserOps) {
      SmallVector<OpFoldResult> userOffsets = extractUserOp.getMixedOffsets();
      SmallVector<OpFoldResult> userSizes = extractUserOp.getMixedSizes();
      SmallVector<OpFoldResult> userStrides = extractUserOp.getMixedStrides();
      SmallVector<OpFoldResult> newInnerOffsets;
      SmallVector<OpFoldResult> newInnerSizes;

      for (unsigned i = 0; i < userOffsets.size(); i++) {
        if (hasSameOffsetForAllExtractUserOps[i]) {
          newInnerOffsets.push_back(zeroAttr);
          newInnerSizes.push_back(userSizes[i]);
          continue;
        }
        newInnerOffsets.push_back(userOffsets[i]);
        newInnerSizes.push_back(userSizes[i]);
      }
      rewriter.setInsertionPoint(extractUserOp);
      auto newInnerExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
          extractUserOp.getLoc(), newOuterExtractSliceOp.getResult(),
          newInnerOffsets, newInnerSizes, userStrides);
      rewriter.replaceOp(extractUserOp, newInnerExtractSliceOp);
    }
  }
  containingForOp->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });
}

//===----------------------------------------------------------------------===//
// MergeProducerExtractUsesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
MergeProducerExtractUsesOp::apply(TransformRewriter &rewriter,
                                  TransformResults &transformResults,
                                  TransformState &state) {
  auto producerOps = state.getPayloadOps(getProducerOp());
  auto containingOps = state.getPayloadOps(getContainingOp());
  if (!llvm::hasSingleElement(containingOps)) {
    return emitDefiniteFailure()
           << "requires exactly one containing_op handle (got "
           << llvm::range_size(containingOps) << ")";
  }
  Operation *containingOp = *containingOps.begin();
  auto containingForOp = dyn_cast<scf::ForOp>(*containingOps.begin());
  if (!containingOp) {
    Diagnostic diag(containingOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "could not merge extract uses in " << *containingOp;
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  for (Operation *producerOp : producerOps) {
    // merge all extract uses to make sure producer op is fused only once.
    findAndMergeMultipleExtractUses(rewriter, producerOp, containingForOp);
  }

  return DiagnosedSilenceableFailure::success();
}

void MergeProducerExtractUsesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getProducerOpMutable(), effects);
  onlyReadsHandle(getContainingOpMutable(), effects);
  modifiesPayload(effects);
}


#define GET_OP_CLASSES
#include "bishengir/Dialect/Analysis/Transforms/TransformOps.cpp.inc"
