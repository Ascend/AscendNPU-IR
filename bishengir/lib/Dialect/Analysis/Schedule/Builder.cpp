//===--------- Builder.cpp - Transform op wrapper for schedules --*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/Schedule/Builder.h"

#include "bishengir/Dialect/Analysis/Transforms/TransformOps.h"
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bishengir-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Builder] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::schedule;

namespace {
std::string stringifyCanonicalizationPatternKind(
    schedule::detail::CanonicalizationPatternKind kind) {
  switch (kind) {
  case schedule::detail::CanonicalizationPatternKind::kSimplifyTrivialLoops:
    return "SimplifyTrivialLoops";
  case schedule::detail::CanonicalizationPatternKind::
      kFoldTransposeWithTranspose:
    return "FoldTransposeWithTranspose";
  }
  llvm_unreachable("unknown canonicalization pattern kind");
}

ArrayAttr getMatchOps(const schedule::detail::Identifier &identifier,
                      OpBuilder &opBuilder) {
  if (identifier.getIdentifierKind() != IdentifierType::kOperation)
    return ArrayAttr();

  auto *operationIdentifier =
      dyn_cast_or_null<schedule::detail::OperationIdentifier>(&identifier);
  assert(operationIdentifier);
  return opBuilder.getArrayAttr(
      {opBuilder.getStringAttr(operationIdentifier->getName())});
}

DictionaryAttr getMatchOpAttrs(const schedule::detail::Identifier &identifier,
                               OpBuilder &opBuilder, bool required) {
  if (identifier.getIdentifierKind() != IdentifierType::kAttribute)
    return DictionaryAttr();

  auto *attributeIdentifier =
      dyn_cast_or_null<schedule::detail::AttributeIdentifier>(&identifier);
  assert(attributeIdentifier);
  return attributeIdentifier->getAttrs(opBuilder, required);
}

} // namespace

namespace mlir {
namespace schedule {

//===----------------------------------------------------------------------===//
// Handle materialization.
//===----------------------------------------------------------------------===//

Value ScheduleBuilder::getValue(ValueHandle *handle, OpBuilder &opBuilder) {
  if (handle == nullptr) {
    llvm::report_fatal_error("cannot get value from nullptr handle");
  }
  if (auto *h = dyn_cast<RegularValueHandle>(handle)) {
    return h->get();
  }
  if (auto *h = dyn_cast<NamedValueHandle>(handle)) {
    if (h->getStatus() == HandleStatus::kNeedsRematch) {
      Value matched = h->get(getTransformSeqHandle(), opBuilder);
      if (h->getNeedsReverse())
        matched = createReverseOp(matched, opBuilder);
      h->setHandle(matched);
    }
    return h->get(getTransformSeqHandle(), opBuilder);
  }
  // FuncArgHandle relies on the dialect-neutral `func.get_func_argument`
  // transform op (Analysis/Transforms), so it is dispatched here directly.
  if (auto *h = dyn_cast<FuncArgHandle>(handle)) {
    return h->get(getFuncValue(opBuilder), opBuilder);
  }
  llvm::report_fatal_error("Not implemented!");
}

SmallVector<Value> ScheduleBuilder::getValues(const ValueHandles &handles,
                                              OpBuilder &opBuilder) {
  return llvm::map_to_vector(handles, [this, &opBuilder](ValueHandle *handle) {
    return this->getValue(handle, opBuilder);
  });
}

std::pair<SmallVector<int64_t>, SmallVector<Value>>
ScheduleBuilder::unpackFoldResults(ValueHandleFoldResults &values,
                                   OpBuilder &opBuilder) {
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  for (auto &v : values) {
    auto maybeConstInteger = v.getConstInteger();
    if (maybeConstInteger) {
      staticSizes.push_back(maybeConstInteger.value());
      continue;
    }
    staticSizes.push_back(ShapedType::kDynamic);
    std::optional<ValueHandle *> maybeHandle = v.getValueHandle();
    assert(maybeHandle && "invalid handle");
    dynamicSizes.push_back(getValue(*maybeHandle, opBuilder));
  }
  return {staticSizes, dynamicSizes};
}

//===----------------------------------------------------------------------===//
// Matching and annotating.
//===----------------------------------------------------------------------===//

Value ScheduleBuilder::getFuncValue(OpBuilder &opBuilder) {
  auto matchTarget = getTransformSeqHandle();
  return opBuilder.create<transform::MatchOp>(
      matchTarget.getLoc(), matchTarget,
      ArrayRef<StringRef>({func::FuncOp::getOperationName()}));
}

ValueHandle *ScheduleBuilder::getFuncHandle(OpBuilder &opBuilder) {
  return record<RegularValueHandle>(getFuncValue(opBuilder), opBuilder);
}

Value ScheduleBuilder::createReverseOp(Value target, OpBuilder &opBuilder) {
  return opBuilder.create<transform::ReverseOp>(
      target.getLoc(),
      /*result=*/TypeRange{opBuilder.getType<transform::AnyOpType>()},
      /*target=*/target);
}

Value ScheduleBuilder::matchByIdentifier(Value target,
                                         const Identifier &identifier,
                                         OpBuilder &opBuilder,
                                         const MatchOptions &options) {
  auto ops = getMatchOps(identifier, opBuilder);
  auto requiredOpAttrs =
      getMatchOpAttrs(identifier, opBuilder, /*required=*/true);
  auto optionalOpAttrs =
      getMatchOpAttrs(identifier, opBuilder, /*required=*/false);
  Value matchResult;
  if (!options.childHandleOrValue.has_value()) {
    matchResult = opBuilder
                      .create<transform::MatchOp>(
                          target.getLoc(), /*target=*/target,
                          /*ops=*/ops, requiredOpAttrs, optionalOpAttrs)
                      .getResults();
  } else {
    std::variant<ValueHandle *, Value> val = options.childHandleOrValue.value();
    Value childValue;
    if (std::holds_alternative<Value>(val))
      childValue = std::get<Value>(val);
    else if (std::holds_alternative<ValueHandle *>(val)) {
      auto *valHandle = std::get<ValueHandle *>(val);
      childValue = getValue(valHandle, opBuilder);
    } else {
      llvm::report_fatal_error("Not implemented!");
    }
    matchResult =
        opBuilder
            .create<transform::MatchAncestorOfOp>(
                target.getLoc(), /*target=*/target, /*child=*/childValue,
                /*ops=*/ops, requiredOpAttrs, optionalOpAttrs)
            .getResults();
  }

  if (options.needsReverse)
    matchResult = createReverseOp(matchResult, opBuilder);

  return matchResult;
}

void ScheduleBuilder::annotateByAttr(Value target, StringRef attrName,
                                     OpBuilder &opBuilder) {
  opBuilder.create<transform::AnnotateOp>(
      target.getLoc(),
      /*target=*/target,
      /*name=*/opBuilder.getStringAttr(attrName),
      /*param=*/Value{});
}

Value ScheduleBuilder::mergeHandles(
    const SmallVectorImpl<Value> &handles,
    transform::TransformHandleTypeInterface handleType, OpBuilder &opBuilder) {
  assert(!handles.empty());
  return opBuilder.create<transform::MergeHandlesOp>(handles.front().getLoc(),
                                                     /*result=*/handleType,
                                                     /*target=*/handles);
}

ValueHandles ScheduleBuilder::splitHandle(ValueHandle *handle, size_t splitSize,
                                          OpBuilder &opBuilder) {
  Value handleValue = getValue(handle, opBuilder);
  auto results = opBuilder
                     .create<transform::SplitHandleOp>(
                         handleValue.getLoc(),
                         /*handle=*/handleValue,
                         /*numResultHandles=*/static_cast<int64_t>(splitSize))
                     .getResults();
  return llvm::map_to_vector(results, [this, &opBuilder](Value result) {
    auto ptr = record<RegularValueHandle>(result, opBuilder);
    return static_cast<ValueHandle *>(ptr);
  });
}

ResultRange ScheduleBuilder::createForEachOp(Value target,
                                             TypeRange resultTypes,
                                             RegionBuilderFn regionBuilder,
                                             OpBuilder &opBuilder) {
  OpBuilder::InsertionGuard guard(opBuilder);
  auto foreach =
      opBuilder.create<transform::ForeachOp>(target.getLoc(),
                                             /*results=*/resultTypes,
                                             /*target=*/ValueRange{target},
                                             /*with_zip_shortest=*/false);
  Region &body = foreach.getBody();
  Block *block = opBuilder.createBlock(
      &body, /*insertPt=*/{}, {opBuilder.getType<transform::AnyOpType>()},
      {foreach.getLoc()});
  ImplicitLocOpBuilder b(foreach.getLoc(), opBuilder);
  regionBuilder(b, *block);
  transform::ForeachOp::ensureTerminator(body, opBuilder, foreach.getLoc());
  return foreach->getResults();
}

ValueHandle *ScheduleBuilder::getOpsWithName(StringRef opName,
                                             OpBuilder &opBuilder,
                                             const MatchOptions &options) {
  return getOpsWithIdentifier(OperationIdentifier(opName), opBuilder, options);
}

ValueHandle *ScheduleBuilder::getOpsWithAttr(StringRef attrName,
                                             OpBuilder &opBuilder,
                                             Attribute attrValue,
                                             const MatchOptions &options) {
  return getOpsWithIdentifier(AttributeIdentifier(attrName, attrValue),
                              opBuilder, options);
}

ValueHandle *ScheduleBuilder::getOpsWithAttrs(
    const SmallVector<NamedAttribute> &requiredAttrs, OpBuilder &opBuilder,
    const SmallVector<NamedAttribute> &optionalAttrs,
    const MatchOptions &options) {
  DenseMap<StringRef, Attribute> requiredAttrsMap;
  for (auto namedAttr : requiredAttrs)
    requiredAttrsMap.insert({namedAttr.getName(), namedAttr.getValue()});

  DenseMap<StringRef, Attribute> optionalAttrsMap;
  for (auto namedAttr : optionalAttrs)
    optionalAttrsMap.insert({namedAttr.getName(), namedAttr.getValue()});

  return getOpsWithIdentifier(
      AttributeIdentifier(requiredAttrsMap, optionalAttrsMap), opBuilder,
      options);
}

ValueHandle *
ScheduleBuilder::getOpsWithIdentifier(const Identifier &identifier,
                                      OpBuilder &opBuilder,
                                      const MatchOptions &options) {
  assert(identifier.getIdentifierKind() != IdentifierType::kUnknown);
  // For named handles, there is no need to construct a new handle everytime as
  // the name should be unique. Directly fetch the handle if possible.
  std::optional<NamedValueHandle *> maybeHandle =
      tryFetchRecord<NamedValueHandle>(identifier);
  if (maybeHandle.has_value())
    return (*maybeHandle);

  auto matchTarget = getTransformSeqHandle();
  auto targetOps =
      matchByIdentifier(matchTarget, identifier, opBuilder, options);
  // Don't need to annotate because the ops are match by op name.
  return record<NamedValueHandle>(
      targetOps, opBuilder,
      NamedValueHandleArgs{identifier.getUniqueIdentifier(),
                           identifier.getIdentifierKind(),
                           /*needsAnnotate=*/false,
                           /*needsReverse=*/false,
                           /*isNameUnique=*/true});
}

//===----------------------------------------------------------------------===//
// Transform op wrappers.
//===----------------------------------------------------------------------===//

ScheduleBuilder::ForallTilingResult
ScheduleBuilder::tileUsingForAll(ValueHandles &targets,
                                 int64_t staticNumThreads, ArrayAttr mapping,
                                 OpBuilder &opBuilder) {
  ValueHandles loopHandles;
  for (auto *targetHandle : targets) {
    auto targetValue = getValue(targetHandle, opBuilder);
    auto forAllOp = opBuilder.create<transform::TileUsingForallOp>(
        targetValue.getLoc(),
        /*target=*/targetValue,
        /*staticNumThreads=*/ArrayRef<int64_t>({staticNumThreads}),
        /*odsArg2=*/transform::NumThreadsSpec{},
        /*mapping=*/mapping);
    loopHandles.emplace_back(record<NamedValueHandle>(
        forAllOp.getForallOp(), opBuilder,
        NamedValueHandleArgs{kTiledForAllTagName, IdentifierType::kAttribute}));
    // Update original handle to hold the tiled op.
    targetHandle->setHandle(forAllOp.getTiledOp());
  }
  return ForallTilingResult{/*loops=*/loopHandles};
}

ScheduleBuilder::ForTilingResult ScheduleBuilder::tileUsingFor(
    ValueHandles &targets, ValueHandleFoldResults &tileSizes,
    OpBuilder &opBuilder, ArrayRef<int64_t> interchangeAxis) {
  auto mapFn = [this, &opBuilder](Value tiledLoop) -> ValueHandle * {
    return record<NamedValueHandle>(
        tiledLoop, opBuilder,
        NamedValueHandleArgs{kTiledForTagName, IdentifierType::kAttribute});
  };
  auto [staticTileSizes, dynamicTileSizes] =
      unpackFoldResults(tileSizes, opBuilder);
  SmallVector<bool> scalableSizes(tileSizes.size(), false);
  SmallVector<Type> outputTypes(
      llvm::count_if(staticTileSizes,
                     [](int64_t tileSize) { return tileSize != 0; }),
      opBuilder.getType<transform::AnyOpType>());

  SmallVector<ValueHandles> loopHandles;
  for (auto *targetHandle : targets) {
    auto targetValue = getValue(targetHandle, opBuilder);
    auto forOp = opBuilder.create<transform::TileUsingForOp>(
        targetValue.getLoc(),
        /*tiled_linalg_op=*/opBuilder.getType<transform::AnyOpType>(),
        /*loops=*/outputTypes,
        /*target=*/targetValue,
        /*dynamic_sizes=*/dynamicTileSizes,
        /*static_sizes=*/staticTileSizes,
        /*interchange=*/interchangeAxis,
        /*scalable_sizes=*/scalableSizes);
    loopHandles.emplace_back(llvm::map_to_vector(forOp.getLoops(), mapFn));
    // Update original handle to hold the tiled op.
    targetHandle->setHandle(forOp.getTiledLinalgOp());
    LDBG("tileUsingFor result for " << targetValue);
    LDBG("dynamic tile size: ");
#ifndef NDEBUG
    for (auto dynamicTileSize : dynamicTileSizes) {
      LDBG(dynamicTileSize);
    }
    for (auto forLoop : forOp.getLoops()) {
      LDBG(forLoop);
    }
#endif
    LDBG(forOp.getTiledLinalgOp());
    LDBG("tileUsingFor result end");
  }
  return ForTilingResult{/*loops=*/loopHandles};
}

ValueHandle *ScheduleBuilder::fuseLoops(ValueHandles &loops,
                                        OpBuilder &opBuilder) {
  assert(!std::empty(loops) && "Should fuse more than one loops");
  auto *fusedLoopHandle = loops.front();
  auto fusedLoopValue = getValue(fusedLoopHandle, opBuilder);
  fusedLoopHandle->invalidate();

  for (auto *nextLoopHandle : llvm::drop_begin(loops)) {
    auto nextLoopValue = getValue(nextLoopHandle, opBuilder);
    fusedLoopValue =
        opBuilder
            .create<transform::LoopFuseSiblingOp>(
                nextLoopValue.getLoc(),
                /*fused_loop=*/opBuilder.getType<transform::AnyOpType>(),
                /*target=*/fusedLoopValue,
                /*source=*/nextLoopValue)
            .getFusedLoop();
    nextLoopHandle->invalidate();
  }
  return record<NamedValueHandle>(
      fusedLoopValue, opBuilder,
      NamedValueHandleArgs{kFusedLoopTagName, IdentifierType::kAttribute});
}

ValueHandles ScheduleBuilder::fuseLoopsForEachDim(
    ArrayRef<ValueHandles> tiledLoopsForEachDim, OpBuilder &builder) {
  ValueHandles fusedLoops;
  for (ValueHandles currentDimTiledLoops : tiledLoopsForEachDim) {
    auto loopsToFuse = llvm::to_vector(llvm::make_filter_range(
        currentDimTiledLoops,
        [](const ValueHandle *vh) { return vh != nullptr; }));
    if (loopsToFuse.empty())
      continue;

    llvm::for_each(loopsToFuse, [](ValueHandle *vh) {
      vh->setStatus(HandleStatus::kNeedsRematch);
    });
    fusedLoops.push_back(fuseLoops(loopsToFuse, builder));
  }
  return fusedLoops;
}

ValueHandle *ScheduleBuilder::coalesceLoops(ValueHandle *outerMostLoop,
                                            OpBuilder &opBuilder) {
  // Apply canonicalize before coalescing to move invariants out of loop.
  applyPatterns(
      getFuncHandle(opBuilder),
      /*patterns=*/
      SmallVector<TransformPatternKind>{TransformPatternKind::CANONICALIZATION},
      opBuilder,
      /*disablePatterns=*/
      SmallVector<CanonicalizationPatternKind>{
          CanonicalizationPatternKind::kSimplifyTrivialLoops});

  auto outerMostLoopValue = getValue(outerMostLoop, opBuilder);
  outerMostLoop->invalidate();

  auto coalescedLoopValue = opBuilder.create<transform::LoopCoalesceOp>(
      outerMostLoopValue.getLoc(),
      /*transformed=*/opBuilder.getType<transform::AnyOpType>(),
      /*target=*/outerMostLoopValue);

  return record<NamedValueHandle>(
      coalescedLoopValue, opBuilder,
      NamedValueHandleArgs{kCoalescedLoopTagName, IdentifierType::kAttribute});
}

void ScheduleBuilder::applyCanonicalization(OpBuilder &opBuilder) {
  auto matchTarget = getTransformSeqHandle();
  matchTarget = opBuilder
                    .create<transform::ApplyRegisteredPassOp>(
                        matchTarget.getLoc(),
                        /*result=*/opBuilder.getType<transform::AnyOpType>(),
                        /*target=*/matchTarget,
                        /*pass_name=*/opBuilder.getStringAttr("canonicalize"))
                    .getResult();
  resetAllHandles();
  setTransformSeqHandle(matchTarget);
}

void ScheduleBuilder::applyCSE(OpBuilder &opBuilder) {
  auto matchTarget = getTransformSeqHandle();
  matchTarget = opBuilder
                    .create<transform::ApplyRegisteredPassOp>(
                        matchTarget.getLoc(),
                        /*result=*/opBuilder.getType<transform::AnyOpType>(),
                        /*target=*/matchTarget,
                        /*pass_name=*/opBuilder.getStringAttr("cse"))
                    .getResult();
  resetAllHandles();
  setTransformSeqHandle(matchTarget);
}

void ScheduleBuilder::applyPatterns(
    ValueHandle *target, const SmallVector<TransformPatternKind> &patterns,
    OpBuilder &opBuilder,
    const SmallVector<CanonicalizationPatternKind> &disablePatterns) {
  bool applyCSE = false;
  auto bodyBuilderFn = [&patterns, &applyCSE](OpBuilder &p, Location loc) {
    llvm::for_each(patterns, [&p, &loc, &applyCSE](TransformPatternKind k) {
      switch (k) {
      case TransformPatternKind::CSE:
        applyCSE = true;
        break;
      case TransformPatternKind::CANONICALIZATION:
        p.create<transform::ApplyCanonicalizationPatternsOp>(loc);
        break;
      case TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE:
        p.create<transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp>(
            loc);
        break;
      case TransformPatternKind::RESOLVE_RANKED_SHAPED_TYPE_RESULT_DIMS:
        p.create<transform::ApplyResolveRankedShapedTypeResultDimsPatternsOp>(
            loc);
        break;
      }
    });
  };
  Value targetValue = getValue(target, opBuilder);
  auto applyPatternsOp = opBuilder.create<transform::ApplyPatternsOp>(
      targetValue.getLoc(),
      /*target=*/targetValue,
      /*bodyBuilder=*/bodyBuilderFn);

  // Set apply CSE
  applyPatternsOp.setApplyCse(applyCSE);

  // Add disable patterns
  SmallVector<Attribute> stringifiedDisablePatterns;
  for (auto k : disablePatterns) {
    stringifiedDisablePatterns.push_back(
        opBuilder.getStringAttr(stringifyCanonicalizationPatternKind(k)));
  }

  // Disable FoldTransposeWithTranspose patten in auto-schedule.
  stringifiedDisablePatterns.push_back(
      opBuilder.getStringAttr(stringifyCanonicalizationPatternKind(
          CanonicalizationPatternKind::kFoldTransposeWithTranspose)));

  applyPatternsOp.setDisablePatternsAttr(
      opBuilder.getArrayAttr(stringifiedDisablePatterns));
  target->invalidate();
}

//===----------------------------------------------------------------------===//
// Handle recording.
//===----------------------------------------------------------------------===//

NamedValueHandle ScheduleBuilder::recordImpl(Value target, OpBuilder &opBuilder,
                                             const NamedValueHandleArgs &args) {
  // If the identifier type is operation name, then it's already unique.
  std::string uniqueName =
      args.isNameUnique ? args.name.str()
                        : getHandleRecord()->getAndRecordAttrName(args.name);

  if (args.needsReverse)
    target = createReverseOp(target, opBuilder);

  if (args.needsAnnotate)
    opBuilder.create<transform::AnnotateOp>(
        target.getLoc(),
        /*target=*/target,
        /*name=*/opBuilder.getStringAttr(uniqueName),
        /*param=*/Value{});

  return NamedValueHandle(target, uniqueName, args.type, HandleStatus::kValid,
                          args.needsReverse);
}

RegularValueHandle
ScheduleBuilder::recordImpl(Value target,
                            [[maybe_unused]] OpBuilder &opBuilder) {
  return RegularValueHandle(target, HandleStatus::kValid);
}

FuncArgHandle ScheduleBuilder::recordImpl(Value target,
                                          [[maybe_unused]] OpBuilder &opBuilder,
                                          size_t funcArgNum) {
  return FuncArgHandle(target, funcArgNum, HandleStatus::kValid);
}

//===----------------------------------------------------------------------===//
// Loop helpers (bishengir SCF transform ops).
//===----------------------------------------------------------------------===//

ScheduleBuilder::LoopTileResult
ScheduleBuilder::tileLoop(ValueHandle *targetLoop,
                          ValueHandleFoldResult tileSize, OpBuilder &opBuilder,
                          const LoopTileOptions &options) {
  auto targetLoopValue = getValue(targetLoop, opBuilder);
  targetLoop->invalidate();

  size_t numLoopsAfterTiling = 2;
  SmallVector<Type> resulTypes(numLoopsAfterTiling,
                               opBuilder.getType<transform::AnyOpType>());

  int64_t staticTileSize;
  SmallVector<Value> dynamicTileSizes;
  if (tileSize.getConstInteger().has_value()) {
    staticTileSize = tileSize.getConstInteger().value();
  } else {
    staticTileSize = ShapedType::kDynamic;
    if (auto *h = dyn_cast<FuncArgHandle>(tileSize.getValueHandle().value()))
      dynamicTileSizes.push_back(h->get(getFuncValue(opBuilder), opBuilder));
    else
      dynamicTileSizes.push_back(tileSize.getValueHandle().value()->get());
  }

  auto loopTileOp = opBuilder.create<transform::LoopTileOp>(
      targetLoopValue.getLoc(),
      /*loops=*/resulTypes,
      /*target=*/targetLoopValue,
      /*dynamic_size=*/dynamicTileSizes,
      /*static_sizes=*/opBuilder.getDenseI64ArrayAttr({staticTileSize}),
      /*is_npart_mode=*/
      opBuilder.getBoolAttr(options.mode == LoopTileMode::kNPartMode),
      /*$is_reorder_mode=*/opBuilder.getBoolAttr(options.isReorderMode));

  auto results = loopTileOp.getLoops();
  return LoopTileResult{
      /*outerLoop=*/record<NamedValueHandle>(
          results[0], opBuilder,
          NamedValueHandleArgs{kTiledForTagName, IdentifierType::kAttribute}),
      /*innerLoop=*/record<NamedValueHandle>(
          results[1], opBuilder,
          NamedValueHandleArgs{kTiledForTagName, IdentifierType::kAttribute})};
}

void ScheduleBuilder::normalizeLoop(ValueHandle *targetLoop,
                                    OpBuilder &opBuilder) {
  auto targetLoopValue = getValue(targetLoop, opBuilder);
  auto normalizedLoop = opBuilder.create<transform::LoopNormalizeOp>(
      targetLoopValue.getLoc(),
      /*transformed=*/opBuilder.getType<transform::AnyOpType>(),
      /*target=*/targetLoopValue);
  targetLoop->setHandle(normalizedLoop);
}

ValueHandle *
ScheduleBuilder::mapForToForall(ValueHandle *targetLoop, OpBuilder &opBuilder,
                                const MapForToForallOptions &options) {
  Value loopValue = getValue(targetLoop, opBuilder);
  auto forallValue = opBuilder.create<transform::ForToForallOp>(
      loopValue.getLoc(),
      /*forallOp=*/
      opBuilder.getType<transform::AnyOpType>(),
      /*for_op=*/loopValue,
      /*mapping=*/options.mapping.has_value()
          ? opBuilder.getArrayAttr({options.mapping.value()})
          : ArrayAttr(),
      /*annotate_only=*/opBuilder.getBoolAttr(options.annotateOnly));

  if (options.annotateOnly)
    return targetLoop;

  targetLoop->invalidate();
  return record<NamedValueHandle>(
      forallValue, opBuilder,
      NamedValueHandleArgs{kForallLoopTagName, IdentifierType::kAttribute,
                           /*needsAnnotate=*/true});
}

} // namespace schedule
} // namespace mlir
