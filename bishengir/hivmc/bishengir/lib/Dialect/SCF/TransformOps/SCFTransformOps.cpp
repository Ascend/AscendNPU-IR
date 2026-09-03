//===- SCFTransformOps.cpp - Implementation of SCF transformation ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "bishengir/Dialect/SCF/Transforms/Transform.h"
#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "bishengir-scf-transform-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
static constexpr llvm::StringLiteral kMappingAttrName = "mapping";
} // namespace

using namespace mlir;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// ParallelLoopTileOp
//===----------------------------------------------------------------------===//

static SmallVector<int64_t> extractFromI64ArrayAttr(Attribute attr) {
  if (ArrayAttr array = dyn_cast<ArrayAttr>(attr))
    return llvm::to_vector<4>(
        llvm::map_range(array, [](Attribute a) -> int64_t {
          return cast<IntegerAttr>(a).getInt();
        }));
  if (DenseI64ArrayAttr array = dyn_cast<DenseI64ArrayAttr>(attr))
    return SmallVector<int64_t>(array.asArrayRef());

  llvm_unreachable("Unsupported type for conversion to an integer array.");
}

DiagnosedSilenceableFailure
ParallelLoopTileOp::apply(TransformRewriter &rewriter,
                          TransformResults &results, TransformState &state) {
  auto payload = state.getPayloadOps(getTarget());
  scf::ParallelOp loop = dyn_cast<scf::ParallelOp>(*payload.begin());

  SmallVector<int64_t, 4> tileSizes = extractFromI64ArrayAttr(getStaticSizes());
  for (auto tileSize : tileSizes)
    if (tileSize == 0) {
      return emitDefiniteFailure() << "tile size cannot be 0";
    }

  std::pair<scf::ParallelOp, scf::ParallelOp> result;
  // noMinMaxBounds wired to false
  result = tileParallelLoop(loop, tileSizes, false);
  results.set(getOperation()->getOpResult(0), {result.first});
  results.set(getOperation()->getOpResult(1), {result.second});

  return DiagnosedSilenceableFailure::success();
}

void ParallelLoopTileOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : ";
  p.printFunctionalType(TypeRange(getOperand().getType()),
                        getResults().getTypes());
}

ParseResult ParallelLoopTileOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::UnresolvedOperand targetOperand;
  if (parser.parseOperand(targetOperand) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  FunctionType trailingType;
  SMLoc typeLoc;
  if (parser.getCurrentLocation(&typeLoc) ||
      parser.parseColonType(trailingType)) {
    return failure();
  }
  if (trailingType.getNumInputs() != 1)
    return parser.emitError(typeLoc) << "expected one input type";

  result.addTypes(trailingType.getResults());
  if (parser.resolveOperand(targetOperand, trailingType.getInput(0),
                            result.operands))
    return failure();
  return success();
}

void ParallelLoopTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LoopTileOp
//===----------------------------------------------------------------------===//

/// Loop is normalized if lb = 0, step = 1
static bool isLoopNormalized(const OpFoldResult &lb, const OpFoldResult &step) {
  auto isEqual = [](const OpFoldResult &result, int64_t val) {
    auto intValue = getConstantIntValue(result);
    return intValue.has_value() && intValue == val;
  };
  return isEqual(lb, 0) && isEqual(step, 1);
}

namespace mlir::transform {
/// Appends a dynamic tile size value to the provided tile sizes vector.
///
/// This function handles two different payload types for transform values:
/// 1. AnyValueType: Extracts the payload value directly from the transform
/// state
/// 2. Operation payload: Extracts a single-result operation and uses its result
///
/// @param transformValue The transform value containing the dynamic tile size
///                       payload. Can be either an AnyValueType or an operation
///                       payload.
/// @param tileSizes      Result reference where the extracted tile size index
///                       will be appended.
/// @param state          The transform state containing the payload mappings
///                       for the transform value.
inline DiagnosedSilenceableFailure
appendDynTileSize(Value transformValue, SmallVector<Value, 2> &tileSizes,
                  TransformRewriter &rewriter, TransformState &state,
                  const Location &loc) {
  // omitting the `isa<ParamType>(transformValue.getType())` case for now
  if (isa<transform::AnyValueType>(transformValue.getType())) {
    auto dynSizePayloads =
        llvm::to_vector(state.getPayloadValues(transformValue));
    if (!llvm::hasSingleElement(dynSizePayloads)) {
      return emitDefiniteFailure(loc)
             << "expect only 1 tile size operation payload";
    }
    Value dynSizePayloadValue = getValueOrCreateCastToIndexLike(
        rewriter, loc, rewriter.getIndexType(), *dynSizePayloads.begin());
    tileSizes.push_back(dynSizePayloadValue);
  } else if (isa<transform::AnyOpType>(transformValue.getType())) {
    auto dynSizePayloads = llvm::to_vector(state.getPayloadOps(transformValue));
    if (!llvm::hasSingleElement(dynSizePayloads)) {
      return emitDefiniteFailure(loc)
             << "expect only 1 tile size operation payload";
    }
    Operation *dynSizePayloadOp = *dynSizePayloads.begin();
    if (dynSizePayloadOp->getNumResults() != 1) {
      return emitDefiniteFailure(loc) << "expect a single result";
    }
    Value dynSizePayloadValue = getValueOrCreateCastToIndexLike(
        rewriter, loc, rewriter.getIndexType(), dynSizePayloadOp->getResult(0));
    tileSizes.push_back(dynSizePayloadValue);
  } else {
    return emitDefiniteFailure(loc) << "Unsupported payload value";
  }
  return DiagnosedSilenceableFailure::success();
}
} // namespace mlir::transform

/// IR before tiling:
///   for i : l to u step s
///     use(i)
///
/// IR after tiling loop :
///
/// When it is factor mode with factor tile size x:
///   for i.o : l to u step x
///    for i.i : 0 to min(u - i.o, x) step s
///      use(i.o + i.i)
///
/// When it is npart mode with npart tile size x:
///   for i.o 0 to x step 1
///     for i.i 0 to min(ceilDiv(u, x), u - i.o*(ceilDiv(u, x))) step 1
///       use(i.o*(ceilDiv(u, x)) + i.i)
///
/// When it is factor and reorder mode with factor tile size x:
///   for i.o 0 to x step 1
///     for i.i 0 to (u-i.o) step x
///       use(i.o + i.i)
DiagnosedSilenceableFailure
LoopTileOp::apply(TransformRewriter &rewriter,
                  TransformResults &transformResults, TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires exactly one target handle!";

  auto loopOp = dyn_cast<scf::ForOp>(*payloadOps.begin());
  if (!loopOp)
    return emitDefiniteFailure() << "expect only scf ForOp target";
  rewriter.setInsertionPoint(loopOp);

  ArrayRef<int64_t> staticSizes = getStaticSizes();
  operand_range dynSizeHandles = getDynamicSizes();

  SmallVector<Value, 2> tileSizes;
  tileSizes.reserve(staticSizes.size());
  for (auto t : llvm::enumerate(staticSizes)) {
    if (t.value() == ShapedType::kDynamic) {
      Value transformValue = dynSizeHandles[t.index()];
      if (auto appendResult = appendDynTileSize(transformValue, tileSizes,
                                                rewriter, state, getLoc());
          !appendResult.succeeded()) {
        return appendResult;
      }
      continue;
    }
    // Process and check static tile size
    if (t.value() < 0) {
      return emitDefiniteFailure() << "expect a positive integer tile size";
    }
    tileSizes.push_back(
        rewriter.create<arith::ConstantIndexOp>(getLoc(), t.value()));
  }

  if (tileSizes.size() != 1)
    return emitDefiniteFailure()
           << "currently, only support only a single tile size";

  auto loc = loopOp.getLoc();
  auto originalLowerBound = loopOp.getLowerBound();
  auto originalUpperBound = loopOp.getUpperBound();
  auto originalStep = loopOp.getStep();
  auto iterArgs = loopOp.getInits();

  bool isNPartMode = getIsNpartMode();
  bool isReorderMode = getIsReorderMode();
  if (isNPartMode && isReorderMode) {
    return emitDefiniteFailure()
           << "npart tiling mode does not support reorder mode now";
  }

  if (!loopOp.getSingleLowerBound() || !loopOp.getSingleStep())
    return emitDefiniteFailure() << "failed to get single lower bound or step";

  if (isNPartMode || isReorderMode) {
    // npart mode and reorder mode only support normalized loop
    if (!isLoopNormalized(*loopOp.getSingleLowerBound(),
                          *loopOp.getSingleStep()))
      return emitDefiniteFailure()
             << "npart or reorder tiling mode only support normalized loop";
  }

  Value tileSize = tileSizes[0];
  scf::ForOp outerLoop;
  scf::ForOp innerLoop;
  if (!isNPartMode) {
    if (!isReorderMode) {
      // factor mode, not reorder mode

      // create outer loop
      Value outerUpperBound = originalUpperBound;
      Value outerStep = tileSize;
      outerLoop = rewriter.create<scf::ForOp>(
          loc, originalLowerBound, outerUpperBound, outerStep, iterArgs);
      auto outerInductor = outerLoop.getInductionVar();

      // create inner loop
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      auto innerLowBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto exprA = rewriter.getAffineSymbolExpr(0);
      auto exprB = rewriter.getAffineSymbolExpr(1);
      auto exprC = rewriter.getAffineSymbolExpr(2);

      // (exprA - exprB, exprC)
      AffineMap upperBoundMap =
          AffineMap::get(0, 3, {exprA - exprB, exprC}, rewriter.getContext());

      // min(u - i.o, x)
      Value innerUpperBound = rewriter.create<affine::AffineMinOp>(
          loc, upperBoundMap,
          ValueRange{originalUpperBound, outerInductor, tileSize});
      innerLoop = rewriter.create<scf::ForOp>(loc, innerLowBound,
                                              innerUpperBound, originalStep,
                                              outerLoop.getRegionIterArgs());
      innerLoop.getRegion().takeBody(loopOp.getRegion());
      auto innerInductor = innerLoop.getInductionVar();
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      auto newInductor =
          rewriter.create<arith::AddIOp>(loc, outerInductor, innerInductor);
      rewriter.replaceAllUsesExcept(innerInductor, newInductor, newInductor);
    } else {
      // factor mode, reorder mode

      // create outer loop
      Value outerUpperBound = tileSize;
      Value outerStep = originalStep;
      outerLoop = rewriter.create<scf::ForOp>(
          loc, originalLowerBound, outerUpperBound, outerStep, iterArgs);
      auto outerInductor = outerLoop.getInductionVar();

      // create inner loop
      rewriter.setInsertionPointToStart(outerLoop.getBody());
      auto innerLowBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto exprA = rewriter.getAffineSymbolExpr(0);
      auto exprB = rewriter.getAffineSymbolExpr(1);

      // (exprA - exprB)
      AffineMap upperBoundMap =
          AffineMap::get(0, 2, {exprA - exprB}, rewriter.getContext());
      // (u - i.o)
      Value innerUpperBound = rewriter.create<affine::AffineApplyOp>(
          loc, upperBoundMap, ValueRange{originalUpperBound, outerInductor});
      Value innerStep = tileSize;
      innerLoop =
          rewriter.create<scf::ForOp>(loc, innerLowBound, innerUpperBound,
                                      innerStep, outerLoop.getRegionIterArgs());
      innerLoop.getRegion().takeBody(loopOp.getRegion());
      auto innerInductor = innerLoop.getInductionVar();
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      auto newInductor =
          rewriter.create<arith::AddIOp>(loc, outerInductor, innerInductor);
      rewriter.replaceAllUsesExcept(innerInductor, newInductor, newInductor);
    }
  } else {
    // npart mode
    if (isReorderMode) {
      llvm_unreachable("unsupport npart reorder mode now");
    }

    Value outerUpperBound = tileSize;
    Value outerStep = originalStep;
    outerLoop = rewriter.create<scf::ForOp>(
        loc, originalLowerBound, outerUpperBound, outerStep, iterArgs);
    auto outerInductor = outerLoop.getInductionVar();

    // create inner loop
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    auto innerLowBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto exprA = rewriter.getAffineSymbolExpr(0);
    auto exprB = rewriter.getAffineSymbolExpr(1);
    auto exprC = rewriter.getAffineSymbolExpr(2);

    // (exprA - exprB * ceilDiv(exprA, exprC), ceilDiv(exprA, exprC))
    AffineMap upperBoundMap = AffineMap::get(
        0, 3, {exprA - exprB * exprA.ceilDiv(exprC), exprA.ceilDiv(exprC)},
        rewriter.getContext());

    // min(u - i.o * ceilDiv(u, x), ceilDiv(u, x))
    Value innerUpperBound = rewriter.create<affine::AffineMinOp>(
        loc, upperBoundMap,
        ValueRange{originalUpperBound, outerInductor, tileSize});
    innerLoop = rewriter.create<scf::ForOp>(loc, innerLowBound, innerUpperBound,
                                            originalStep,
                                            outerLoop.getRegionIterArgs());
    innerLoop.getRegion().takeBody(loopOp.getRegion());
    auto innerInductor = innerLoop.getInductionVar();
    rewriter.setInsertionPointToStart(innerLoop.getBody());

    // ceilDiv(u, x)
    auto step = affine::makeComposedAffineApply(
        rewriter, loc, exprA.ceilDiv(exprB), {originalUpperBound, tileSize});
    // i.o * ceilDiv(u, x)
    outerInductor = rewriter.create<arith::MulIOp>(loc, outerInductor, step);

    auto newInductor =
        rewriter.create<arith::AddIOp>(loc, outerInductor, innerInductor);
    rewriter.replaceAllUsesExcept(innerInductor, newInductor, newInductor);
  }

  // construct outer loop body yield
  rewriter.setInsertionPointAfter(innerLoop);
  auto innerResults = innerLoop.getResults();
  if (innerResults.empty()) {
    outerLoop.ensureTerminator(outerLoop.getRegion(), rewriter, loc);
  } else {
    rewriter.create<scf::YieldOp>(loc, innerResults);
  }

  // substitute orig loop results by outer loop results
  loopOp.getResults().replaceAllUsesWith(outerLoop);
  rewriter.eraseOp(loopOp);

  // set TransformResults
  transformResults.set(cast<OpResult>(getLoops()[0]),
                       SmallVector<Operation *>{outerLoop.getOperation()});
  transformResults.set(cast<OpResult>(getLoops()[1]),
                       SmallVector<Operation *>{innerLoop.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

ParseResult LoopTileOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  DenseI64ArrayAttr staticSizes;
  FunctionType functionalType;
  llvm::SMLoc operandLoc;

  if (parser.parseOperand(target) || parser.getCurrentLocation(&operandLoc) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(functionalType))
    return ParseResult::failure();

  size_t numExpectedLoops = static_cast<size_t>(
      staticSizes.size() - llvm::count(staticSizes.asArrayRef(), 0));
  if (functionalType.getNumResults() != numExpectedLoops + 1) {
    return parser.emitError(parser.getNameLoc())
           << "expected " << (numExpectedLoops + 1) << " result type(s)";
  }
  if (functionalType.getNumInputs() != dynamicSizes.size() + 1) {
    return parser.emitError(operandLoc)
           << "expected " << (dynamicSizes.size() + 1) << " operand type(s)";
  }
  if (parser.resolveOperand(target, functionalType.getInputs().front(),
                            result.operands) ||
      parser.resolveOperands(dynamicSizes,
                             functionalType.getInputs().drop_front(),
                             operandLoc, result.operands)) {
    return failure();
  }

  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  result.addTypes(functionalType.getResults());

  return success();
}

void LoopTileOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes());

  SmallVector<StringRef, 2> elidedAttrs;
  Builder odsBuilder(getContext());
  Attribute attr = getIsNpartModeAttr();
  if (attr && (attr == odsBuilder.getBoolAttr(false)))
    elidedAttrs.push_back(getIsNpartModeAttrName());

  attr = getIsReorderModeAttr();
  if (attr && (attr == odsBuilder.getBoolAttr(false)))
    elidedAttrs.push_back(getIsReorderModeAttrName());

  elidedAttrs.push_back(getStaticSizesAttrName());

  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  p << " : ";
  p.printFunctionalType(getOperands().getTypes(), getResults().getTypes());
}

void LoopTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getDynamicSizesMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ForToFoallOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
ForToForallOp::apply(TransformRewriter &rewriter,
                     TransformResults &transformResults,
                     TransformState &state) {
  auto payloadOps = state.getPayloadOps(getForOp());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires exactly one target handle!";

  auto loopOp = dyn_cast<scf::ForOp>(*payloadOps.begin());
  if (!loopOp)
    return emitDefiniteFailure() << "expect only scf.for target!";

  if (getAnnotateOnly()) {
    if (getMapping().has_value())
      loopOp->setAttr(kMappingAttrName, *getMapping());
    loopOp->setAttr(utils::kMapForToForallAttrName, rewriter.getUnitAttr());
    transformResults.set(cast<OpResult>(getForallOp()), {loopOp});
    return DiagnosedSilenceableFailure::success();
  }

  scf::ForallOp maybeResult = nullptr;
  auto diag = scf::utils::mapForToForallImpl(rewriter, loopOp, getMapping(),
                                             maybeResult);
  if (!diag.succeeded())
    return diag;

  assert(maybeResult);
  rewriter.replaceOp(loopOp, maybeResult);
  transformResults.set(cast<OpResult>(getForallOp()), {maybeResult});
  return DiagnosedSilenceableFailure::success();
}

void ForToForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getForOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LoopNormalizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LoopNormalizeOp::apply(TransformRewriter &rewriter,
                                  TransformResults &transformResults,
                                  TransformState &state) {
  SetVector<Operation *> normalized;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    MLIRContext *ctx = target->getContext();
    IRRewriter b(ctx);
    b.setInsertionPoint(target);
    scf::ForOp loop = dyn_cast<scf::ForOp>(target);
    if (!isa<arith::ConstantOp>(loop.getLowerBound().getDefiningOp())) {
      emitError("currently don't support normalizing loops with "
                "dynamic lower bound!");
      normalized.insert(target);
      continue;
    }
    if (cast<IntegerAttr>(
            cast<arith::ConstantOp>(loop.getLowerBound().getDefiningOp())
                .getValue())
            .getInt() != 0) {
      emitError("currently don't support normalizing loops with "
                "non-zero lower bound!");
      normalized.insert(target);
      continue;
    }
    Value oldStep = loop.getStep();
    auto oldStepAsIndexOp = oldStep.getDefiningOp<arith::ConstantIndexOp>();
    if (oldStepAsIndexOp && oldStepAsIndexOp.value() == 1) {
      // terminate early if loop is already normalized
      normalized.insert(target);
      continue;
    }
    bishengir::normalizeLoop(b, loop, oldStep);
    normalized.insert(target);
  }
  transformResults.set(cast<OpResult>(getResult()), normalized.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

mlir::scf::ForOp bishengir::scf::normalizeToIndex(PatternRewriter &rewriter,
                                                  mlir::scf::ForOp op) {
  auto loc = op.getLoc();
  auto getLowerBoundType = op.getLowerBound().getType();
  auto getUpperBoundType = op.getUpperBound().getType();
  auto getStepType = op.getStep().getType();
  if (getLowerBoundType.isIndex() && getUpperBoundType.isIndex() &&
      getStepType.isIndex())
    return op;

  auto indexType = rewriter.getIndexType();
  auto newLowerBound =
      rewriter.create<arith::IndexCastOp>(loc, indexType, op.getLowerBound())
          .getOut();
  auto newUpperBound =
      rewriter.create<arith::IndexCastOp>(loc, indexType, op.getUpperBound())
          .getOut();
  auto newStep =
      rewriter.create<arith::IndexCastOp>(loc, indexType, op.getStep())
          .getOut();
  auto oldIV = op.getInductionVar();
  DenseSet<Operation *> setOps(oldIV.getUsers().begin(),
                               oldIV.getUsers().end());

  auto newLoop = rewriter.create<mlir::scf::ForOp>(loc, newLowerBound,
                                                   newUpperBound, newStep);
  auto newIV = newLoop.getInductionVar();
  Block *innermostBlock = newLoop.getBody();
  rewriter.eraseOp(op.getBody()->getTerminator());
  rewriter.inlineBlockBefore(op.getBody(), innermostBlock,
                             innermostBlock->getTerminator()->getIterator(),
                             newIV);
  rewriter.setInsertionPointToStart(newLoop.getBody());
  auto intIV =
      rewriter.create<arith::IndexCastOp>(loc, getLowerBoundType, newIV)
          .getOut();
  rewriter.replaceAllUsesExcept(newIV, intIV, intIV.getDefiningOp());
  rewriter.eraseOp(op);
  return newLoop;
}

void LoopNormalizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class SCFTransformDialectExtension
    : public transform::TransformDialectExtension<
          SCFTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.cpp.inc"

void bishengir::scf::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<SCFTransformDialectExtension>();
}
