//===---------------LoopRestructureArangeOptimization.cpp------------------===//
//----------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt    for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"

#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <queue>

#define DEBUG_TYPE "loop-restructure-arange-optimization"

namespace bishengir {
namespace triton {
#define GEN_PASS_DEF_LOOPRESTRUCTUREARANGEOPTIMIZATION
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::arith;

/// Data structure to track store-load patterns
struct StoreLoadPattern {
  Operation *storeOp = nullptr;
  Operation *loadOp = nullptr;
  llvm::DenseSet<Operation *> dependentOps;
  int64_t embeddingSize = -1;
  SmallVector<Operation *, 4> maskOps;
  SmallVector<Operation *, 4> constantOps;
};

/// Try to extract a single int64_t from an Attribute that might be a
/// DenseIntElementsAttr (splat or single element) or an IntegerAttr.
static std::optional<int64_t> extractIntFromAttr(Attribute a) {
  if (auto i = mlir::dyn_cast<IntegerAttr>(a)) {
    return i.getValue().getSExtValue();
  }
  if (auto dense = mlir::dyn_cast<DenseElementsAttr>(a)) {
    if (mlir::isa<IntegerType>(dense.getType().getElementType())) {
      if (dense.isSplat()) {
        Attribute v = dense.getSplatValue<Attribute>();
        if (auto ia = mlir::dyn_cast<IntegerAttr>(v))
          return ia.getValue().getSExtValue();
      } else if (dense.getNumElements() == 1) {
        auto it = dense.value_begin<llvm::APInt>();
        return (*it).getSExtValue();
      }
    }
  }
  return std::nullopt;
}

/// Collect dependent ops from storeOp by walking backwards (operands defs).
/// This is used to find the block of ops that can be moved.
static void
findDependentOpsFromStore(triton::StoreOp storeOp,
                          llvm::DenseSet<Operation *> &dependentOps) {

  if (!storeOp)
    return;

  std::queue<Value> worklist;

  // include the store itself
  dependentOps.insert(storeOp.getOperation());

  //  include all store operands
  for (Value v : storeOp->getOperands()) {
    if (v)
      worklist.push(v);
  }

  // walk backwards including all ops of define ops
  while (!worklist.empty()) {
    Value v = worklist.front();
    worklist.pop();

    Operation *def = v.getDefiningOp();
    if (!def || !dependentOps.insert(def).second)
      continue;

    for (Value operand : def->getOperands()) {
      if (operand)
        worklist.push(operand);
    }
  }
}

/// Collect the mask chain and its constants for debug diagnostics. Access-width
/// inference is performed separately so unrelated constants cannot be mistaken
/// for last-dimension bounds.
static void findMaskAndConstantOps(Operation *storeOp,
                                   StoreLoadPattern &pattern) {
  if (!storeOp)
    return;

  // Helper to determine if a Value looks like a mask (tensor<...xi1>)
  SmallVector<Operation *, 8> starts;

  // Get mask from storeOp
  if (auto ttStoreOp = dyn_cast<triton::StoreOp>(storeOp)) {
    Value mask = ttStoreOp.getMask();
    if (mask) {
      if (Operation *def = mask.getDefiningOp())
        starts.push_back(def);
    }
  }

  // From start nodes walk mask-chain upwards; collect mask ops and constants
  const size_t MAX_VISIT = 1024;
  DenseSet<Operation *> visited;
  std::queue<Operation *> work;

  for (Operation *s : starts)
    if (s)
      work.push(s);

  while (!work.empty() && visited.size() < MAX_VISIT) {
    Operation *cur = work.front();
    work.pop();
    if (!cur || visited.count(cur))
      continue;

    visited.insert(cur);
    // add the visited op as part of mask chain
    pattern.maskOps.push_back(cur);

    // Check operands safely
    for (Value in : cur->getOperands()) {
      if (!in)
        continue;

      Operation *ddef = in.getDefiningOp();
      if (!ddef)
        continue; // skip values without defining op

      // Record constant operations
      if (auto constOp = dyn_cast<arith::ConstantOp>(ddef)) {
        pattern.constantOps.push_back(constOp);
      } else if (!visited.count(ddef)) {
        work.push(ddef);
      }
    }
  }
}

/// Map an axis backwards through a reshape that only inserts or removes unit
/// dimensions. More general reshapes do not preserve enough axis provenance
/// for this pass to associate a mask bound with an access dimension.
static std::optional<unsigned>
mapAxisThroughUnitDimReshape(RankedTensorType srcType,
                             RankedTensorType resultType, unsigned resultAxis) {
  ArrayRef<int64_t> srcShape = srcType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultAxis >= resultShape.size() || resultShape[resultAxis] == 1)
    return std::nullopt;

  SmallVector<unsigned> srcNonUnitAxes;
  SmallVector<unsigned> resultNonUnitAxes;
  SmallVector<int64_t> srcNonUnitSizes;
  SmallVector<int64_t> resultNonUnitSizes;
  for (auto [axis, size] : llvm::enumerate(srcShape)) {
    if (size <= 0)
      return std::nullopt;
    if (size != 1) {
      srcNonUnitAxes.push_back(axis);
      srcNonUnitSizes.push_back(size);
    }
  }
  for (auto [axis, size] : llvm::enumerate(resultShape)) {
    if (size <= 0)
      return std::nullopt;
    if (size != 1) {
      resultNonUnitAxes.push_back(axis);
      resultNonUnitSizes.push_back(size);
    }
  }
  if (srcNonUnitSizes != resultNonUnitSizes)
    return std::nullopt;

  auto it = llvm::find(resultNonUnitAxes, resultAxis);
  if (it == resultNonUnitAxes.end())
    return std::nullopt;
  return srcNonUnitAxes[std::distance(resultNonUnitAxes.begin(), it)];
}

/// Map an axis backwards through shape-only Triton operations. Return no axis
/// when the source is invariant along that result axis or when the reshape is
/// not a unit-dimension reshape.
static std::optional<unsigned> mapAxisToShapeOperand(Operation *op,
                                                     unsigned resultAxis) {
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto srcType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!resultType || !srcType || resultAxis >= resultType.getRank())
    return std::nullopt;

  if (isa<triton::BroadcastOp>(op)) {
    if (srcType.getRank() != resultType.getRank() ||
        srcType.getDimSize(resultAxis) != resultType.getDimSize(resultAxis) ||
        srcType.getDimSize(resultAxis) <= 1)
      return std::nullopt;
    return resultAxis;
  }
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
    unsigned expandAxis = expand.getAxis();
    if (resultAxis == expandAxis)
      return std::nullopt;
    return resultAxis > expandAxis ? resultAxis - 1 : resultAxis;
  }
  if (isa<triton::ReshapeOp>(op))
    return mapAxisThroughUnitDimReshape(srcType, resultType, resultAxis);
  return std::nullopt;
}

/// Return the extent only if the value still denotes [0, extent) interpreted
/// unsigned. In particular, casts must preserve values, not just shapes.
static std::optional<int64_t>
getRangeExtentAlongAxis(Value value, unsigned axis, unsigned depth = 0) {
  if (depth >= 64)
    return std::nullopt;
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto makeRange = dyn_cast<triton::MakeRangeOp>(def)) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (type && type.getRank() == 1 && axis == 0 && makeRange.getStart() == 0 &&
        makeRange.getEnd() > 0)
      return makeRange.getEnd();
    return std::nullopt;
  }
  if (isa<triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp>(def)) {
    auto srcAxis = mapAxisToShapeOperand(def, axis);
    return srcAxis ? getRangeExtentAlongAxis(def->getOperand(0), *srcAxis,
                                             depth + 1)
                   : std::nullopt;
  }
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(def)) {
    auto extent = getRangeExtentAlongAxis(def->getOperand(0), axis, depth + 1);
    auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
    if (!extent || !type || !APInt(64, *extent - 1).isIntN(type.getWidth()))
      return std::nullopt;
    if (isa<arith::ExtSIOp>(def)) {
      auto srcType =
          cast<IntegerType>(getElementTypeOrSelf(def->getOperand(0).getType()));
      if (!APInt(64, *extent - 1).isSignedIntN(srcType.getWidth()))
        return std::nullopt;
    }
    return extent;
  }
  // Index casts need a target-dependent index-width proof. Unknown casts are
  // not transparent; another conjunct may still supply a valid bound.
  return std::nullopt;
}

/// Extract an upper bound only when a comparison is proven to constrain the
/// requested tensor axis.
// TODO: Extend this provenance analysis to other predicates and offset index
// arithmetic. Until then, leave those patterns unchanged instead of guessing
// from unrelated constants in the mask DAG.
static std::optional<int64_t> extractAxisBound(arith::CmpIOp cmp,
                                               unsigned axis) {
  Value index;
  Value bound;
  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    index = cmp.getLhs();
    bound = cmp.getRhs();
    break;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    index = cmp.getRhs();
    bound = cmp.getLhs();
    break;
  default:
    return std::nullopt;
  }

  auto constant = bound.getDefiningOp<arith::ConstantOp>();
  auto cmpType = dyn_cast<RankedTensorType>(cmp.getType());
  auto extent = getRangeExtentAlongAxis(index, axis);
  if (!constant || !cmpType || axis >= cmpType.getRank() || !extent)
    return std::nullopt;
  auto indexType = dyn_cast<IntegerType>(getElementTypeOrSelf(index.getType()));
  // A signed comparison observes the high half of an unsigned range as
  // negative, so it is not an upper bound on the original coordinates.
  if ((cmp.getPredicate() == arith::CmpIPredicate::slt ||
       cmp.getPredicate() == arith::CmpIPredicate::sgt) &&
      (!indexType ||
       !APInt(64, *extent - 1).isSignedIntN(indexType.getWidth())))
    return std::nullopt;

  auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
  auto maybeBound = extractIntFromAttr(constant.getValue());
  if (!dense || !maybeBound)
    return std::nullopt;
  int64_t value = *maybeBound;
  if (value <= 0 || !llvm::isPowerOf2_64(value) ||
      cmpType.getDimSize(axis) < value)
    return std::nullopt;
  return value;
}

/// Trace the final store mask backwards while retaining the axis that
/// corresponds to the store's last dimension. For a conjunction, any proven
/// upper bound remains valid and the tightest one determines the useful width.
static std::optional<int64_t> extractMaskAxisBound(Value mask, unsigned axis,
                                                   unsigned depth = 0) {
  if (depth >= 64)
    return std::nullopt;
  Operation *def = mask.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto cmp = dyn_cast<arith::CmpIOp>(def))
    return extractAxisBound(cmp, axis);
  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    auto lhs = extractMaskAxisBound(andOp.getLhs(), axis, depth + 1);
    auto rhs = extractMaskAxisBound(andOp.getRhs(), axis, depth + 1);
    if (lhs && rhs)
      return std::min(*lhs, *rhs);
    return lhs ? lhs : rhs;
  }
  if (isa<triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp>(def)) {
    auto srcAxis = mapAxisToShapeOperand(def, axis);
    if (srcAxis)
      return extractMaskAxisBound(def->getOperand(0), *srcAxis, depth + 1);
  }
  return std::nullopt;
}

/// Infer the useful width of the store's last dimension. Bounds on other
/// dimensions are deliberately ignored even if their numeric values happen to
/// be powers of two.
static bool extractEmbeddingSize(StoreLoadPattern &pattern) {
  auto store = dyn_cast_or_null<triton::StoreOp>(pattern.storeOp);
  if (!store || !store.getMask())
    return false;
  auto maskType = dyn_cast<RankedTensorType>(store.getMask().getType());
  if (!maskType || maskType.getRank() == 0)
    return false;
  auto bound = extractMaskAxisBound(store.getMask(), maskType.getRank() - 1);
  if (!bound)
    return false;
  pattern.embeddingSize = *bound;
  return true;
}

/// Replace only the last dimension if it equals oldLast; else
/// return original type. This is used to change ops return shapes from the old
/// size to the new shrink size from our grouping

// TODO: assumes the size of load is linked to the last dim, add other dims
// later
static Type replaceLastDimIfMatches(Type type, int64_t oldLast,
                                    int64_t newLast) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    ArrayRef<int64_t> shape = tensorType.getShape();
    if (!shape.empty()) {
      int64_t last = shape.back();
      if (last == oldLast) {
        SmallVector<int64_t> newShape(shape.begin(), shape.end());
        newShape.back() = newLast;
        return RankedTensorType::get(newShape, tensorType.getElementType());
      }
    }
  }
  return type;
}

/// Check every operation that will be copied, including scalar and equal-width
/// dependencies. OperationState cloning below does not copy regions/successors.
static bool
canCloneDependencies(const SmallVectorImpl<const StoreLoadPattern *> &group) {
  for (auto *pattern : group) {
    for (Operation *op : pattern->dependentOps) {
      if (op->getNumRegions() || op->getNumSuccessors())
        return false;
      if (auto load = dyn_cast<triton::LoadOp>(op)) {
        if (load.getIsVolatile())
          return false;
      } else if (!isa<triton::StoreOp>(op) && !isMemoryEffectFree(op)) {
        // Ordinary reads can be copied at their original position. Volatile
        // reads, atomics and unknown side effects must not be duplicated.
        return false;
      }
    }
  }
  return true;
}

/// The cloner changes every matching last dimension, not just the mask's
/// range. Prove that these changes all implement the same last-axis prefix
/// slice before mutating anything. Equal dimension sizes alone are not axis
/// provenance (e.g. a row range expanded from 32 to 32x1 must not shrink).
// TODO: Carry explicit axis/slice mappings into the cloner to support more
// general reshapes and independent axes instead of rejecting those graphs.
static bool
canCloneLastAxisPrefix(const SmallVectorImpl<const StoreLoadPattern *> &group,
                       int64_t oldSize) {
  auto changes = [oldSize](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && tensor.getRank() > 0 &&
           tensor.getShape().back() == oldSize;
  };
  llvm::DenseSet<Operation *> ops;
  for (auto *pattern : group) {
    ops.insert(pattern->dependentOps.begin(), pattern->dependentOps.end());
    ops.insert(pattern->storeOp);
  }
  for (Operation *op : ops) {
    bool changesOperand = llvm::any_of(op->getOperandTypes(), changes);
    bool changesResult = llvm::any_of(op->getResultTypes(), changes);
    // make_range is cloned by its end attribute as well as its tensor size.
    if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
      if ((changesResult || range.getEnd() == oldSize) &&
          (range.getStart() != 0 || range.getEnd() != oldSize))
        return false;
      continue;
    }
    if (!changesOperand && !changesResult)
      continue;
    // A tensor block argument cannot be sliced by inserting a broadcast.
    for (Value operand : op->getOperands())
      if (changes(operand.getType()) &&
          (!operand.getDefiningOp() || !ops.contains(operand.getDefiningOp())))
        return false;
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!dense || !dense.isSplat())
        return false;
      continue;
    }
    if (isa<triton::SplatOp>(op))
      continue;
    if (isa<triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp>(op)) {
      auto result = cast<RankedTensorType>(op->getResult(0).getType());
      auto src = cast<RankedTensorType>(op->getOperand(0).getType());
      // A broadcast may introduce the sliced axis from an invariant source.
      if (isa<triton::BroadcastOp>(op) && !changesOperand &&
          src.getShape().back() == 1)
        continue;
      auto srcAxis = mapAxisToShapeOperand(op, result.getRank() - 1);
      if (!changesOperand || !changesResult || !srcAxis ||
          *srcAxis != src.getRank() - 1)
        return false;
      continue;
    }
    // Pointwise operations commute with the same prefix slice on each tensor
    // operand/result. Do not assume this for reductions, transposes or regions.
    if (!op->hasTrait<OpTrait::Elementwise>() &&
        !isa<triton::AddPtrOp, triton::LoadOp, triton::StoreOp>(op))
      return false;
    RankedTensorType tensorType;
    for (Type type :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        if (tensorType && tensor.getShape() != tensorType.getShape())
          return false;
        tensorType = tensor;
      }
    }
  }
  return true;
}

/// Look through the group's dependent operations to find a broadcast op result
/// shape and return the last-dimension size if available. If multiple different
/// last-dims are discovered, prefer the largest. If none can be found,
/// return std::nullopt. This is to find the old size that fits all the old
/// loads/stores

// TODO: this is a simply way to find the old size given the test cases I have
// right now, but this might not work for all cases, need to find better algo
// later
static std::optional<int64_t> findLastDimFromBroadcasts(
    const SmallVectorImpl<const StoreLoadPattern *> &group) {
  int64_t best = -1;
  for (auto *pattern : group) {
    if (!pattern)
      continue;
    for (Operation *dep : pattern->dependentOps) {
      if (!dep)
        continue;
      // only consider broadcast like ops
      if (!isa<triton::BroadcastOp, triton::ExpandDimsOp>(dep))
        continue;
      // check result tensor shapes.
      for (Value res : dep->getResults()) {
        auto rt = dyn_cast<RankedTensorType>(res.getType());
        if (!rt)
          continue;
        ArrayRef<int64_t> shape = rt.getShape();
        if (shape.empty())
          continue;
        int64_t last = shape.back();
        // keep the largest valid one
        if (last > 0 && last > best)
          best = last;
      }
    }
  }
  if (best > 0)
    return best;
  return std::nullopt;
}

/// Ensure a Value with a ranked tensor whose last dim == oldSize is turned
/// into a Value with last-dim == newSize. If the value is already mapped in
/// valueMapping, return that mapped value. Otherwise, insert a tt.broadcast
/// that produces the adjusted type

// TODO: assumes size affect only the last dim of ops
static Value ensureValueWithNewLastDim(
    Value v, int64_t oldSize, int64_t newSize, PatternRewriter &rewriter,
    Location loc, llvm::DenseMap<Value, Value> &valueMapping,
    llvm::DenseMap<Operation *, Operation *> &clonedOpsMap, int groupID) {
  if (!v)
    return v;

  Type t = v.getType();
  auto rt = mlir::dyn_cast<RankedTensorType>(t);
  if (!rt)
    return v;

  ArrayRef<int64_t> shape = rt.getShape();
  if (shape.empty())
    return v;

  int64_t last = shape.back();
  if (last != oldSize)
    return v;

  // If this value is already produced by a cloned producer, return the clone
  // since we also check the inputs.
  if (Operation *def = v.getDefiningOp()) {
    if (clonedOpsMap.count(def)) {
      Operation *clonedProducer = clonedOpsMap[def];
      unsigned idx = mlir::cast<OpResult>(v).getResultNumber();
      Value mapped = clonedProducer->getResult(idx);
      valueMapping[v] = mapped;
      return mapped;
    }
  }
  Type elem = rt.getElementType();

  // New tensor type with last dim replaced
  SmallVector<int64_t> newShape(shape.begin(), shape.end());
  newShape.back() = newSize;
  RankedTensorType newType = RankedTensorType::get(newShape, elem);

  // create a triton.broadcast that produces the requested shape.
  // Do not move the caller's clone away from its original memory-order point.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(v);
  auto bcast = rewriter.create<triton::BroadcastOp>(loc, newType, v);
  bcast.getOperation()->setAttr(
      "group_id", IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                                   APInt(32, groupID, /*isSigned=*/true)));
  Value newV = bcast.getResult();
  valueMapping[v] = newV;
  return newV;
}

/// Clone operations for a group using OperationState approach
static Operation *
cloneConstantOp(PatternRewriter &rewriter, Location loc,
                arith::ConstantOp constOp, int64_t oldSize, int64_t newSize,
                int groupID,
                llvm::DenseMap<Operation *, Operation *> &clonedOpsMap) {
  OperationState state(loc, constOp->getName());
  // Copy all attributes except "value"
  for (auto attr : constOp->getAttrs()) {
    if (attr.getName() == "value")
      continue;
    state.addAttribute(attr.getName(), attr.getValue());
  }
  // update to new result types
  SmallVector<Type, 1> newResultTypes;
  for (Type t : constOp->getResultTypes()) {
    newResultTypes.push_back(replaceLastDimIfMatches(t, oldSize, newSize));
  }
  state.addTypes(newResultTypes);
  // Rebuild/adjust the "value" attribute so its tensor type matches
  // newResultTypes[0]
  Attribute valAttr = constOp.getValue();
  Attribute newValAttr = valAttr;
  // Preserve every element of unchanged constants, including row offsets on
  // unrelated axes. Only resized constants need their value rebuilt; preflight
  // restricts those constants to splats.
  if (!newResultTypes.empty() && newResultTypes[0] != constOp.getType()) {
    auto newResRT = cast<RankedTensorType>(newResultTypes[0]);
    auto dense = dyn_cast<DenseElementsAttr>(valAttr);
    assert(dense && dense.isSplat() &&
           "preflight must only allow resizing splat constants");
    newValAttr =
        SplatElementsAttr::get(newResRT, dense.getSplatValue<Attribute>());
  }
  if (newValAttr)
    state.addAttribute("value", newValAttr);
  Operation *cloned = rewriter.create(state);
  cloned->setAttr("group_id", rewriter.getI32IntegerAttr(groupID));
  clonedOpsMap[constOp] = cloned;
  return cloned;
}

static void
cloneGroupOperations(PatternRewriter &rewriter, Location loc,
                     ArrayRef<Operation *> ordered, Operation *insertionPoint,
                     int64_t oldSize, int64_t newSize,
                     llvm::DenseMap<Operation *, Operation *> &clonedOpsMap,
                     int groupID, llvm::DenseMap<Value, Value> &valueMapping) {

  // Insert cloned ops before insertionPoint
  rewriter.setInsertionPoint(insertionPoint);
  for (Operation *op : ordered) {
    if (!op)
      continue;
    // Special case: triton.make_range
    if (auto mk = dyn_cast<triton::MakeRangeOp>(op)) {
      int64_t start = mk.getStart();
      int64_t end = mk.getEnd();
      // Only adjust when the original end matches oldSize (we want newSize)
      if (end == oldSize) {
        OperationState state(loc, op->getName());
        // Copy all attributes except "start" and "end" (we'll set them below)
        for (auto attr : op->getAttrs()) {
          if (attr.getName() == "start" || attr.getName() == "end")
            continue;
          state.addAttribute(attr.getName(), attr.getValue());
        }
        // Recreate start and end with updated values (preserve start, update
        // end)
        state.addAttribute(
            "start",
            IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                             APInt(32, start, /*isSigned=*/true)));
        state.addAttribute(
            "end", IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                                    APInt(32, newSize, /*isSigned=*/true)));
        for (Type t : op->getResultTypes())
          state.addTypes(replaceLastDimIfMatches(t, oldSize, newSize));
        // Create the adjusted make_range op
        Operation *newMk = rewriter.create(state);
        // Attach group id
        newMk->setAttr(
            "group_id",
            IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                             APInt(32, groupID, /*isSigned=*/true)));
        // Record mapping and continue
        clonedOpsMap[op] = newMk;
        valueMapping[mk.getResult()] = newMk->getResult(0);
        continue;
      }
    }

    // Special-case: arith.constant "value" attribute needs to match new result
    // type
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      Operation *cloned = cloneConstantOp(rewriter, loc, constOp, oldSize,
                                          newSize, groupID, clonedOpsMap);

      // Update value mapping
      for (auto it : llvm::zip(constOp->getResults(), cloned->getResults())) {
        valueMapping[std::get<0>(it)] = std::get<1>(it);
      }
      continue;
    }

    // General: clone ops but make sure input and return are correct shapes
    OperationState state(loc, op->getName());
    // Copy attributes
    state.addAttributes(op->getAttrs());
    // Remap operands
    for (Value v : op->getOperands()) {
      Value mapped = v;
      if (valueMapping.count(v)) {
        mapped = valueMapping[v];
      } else {
        mapped = ensureValueWithNewLastDim(v, oldSize, newSize, rewriter, loc,
                                           valueMapping, clonedOpsMap, groupID);
      }
      state.addOperands(mapped);
    }
    // Update result types
    for (Type t : op->getResultTypes()) {
      state.addTypes(replaceLastDimIfMatches(t, oldSize, newSize));
    }
    // Create the cloned op from the prepared state
    Operation *cloned = rewriter.create(state);
    // Attach group_id on cloned op
    cloned->setAttr("group_id", IntegerAttr::get(
                                    IntegerType::get(rewriter.getContext(), 32),
                                    APInt(32, groupID, /*isSigned=*/true)));
    clonedOpsMap[op] = cloned;
    // Map results
    for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
      valueMapping[std::get<0>(it)] = std::get<1>(it);
    }
  }
}

/// - Start with one group per embedding size (sorted)

/// - digit = # of groups
///   1)  if groups >= intial grouping (one group per embedding
///       size) dont do anything,
///   2)  if groups < intial grouping: merge the smallest two adjacent groups
///       together

///  If digit = 9 it means Greedy Algo grouping:
/// - Compute target = size of the largest group (do not touch that group)
/// - While there exists groups with size < target, pick the smallest group,
///   merge it with the neighbor (left or right) that makes the merged size
///   come closest to target (tie-break by key difference). Repeat until no
///   merges possible or all groups >= target.

// TODO: this logic can be changed, play around with algo and target size
struct GroupStruct {
  int64_t key;
  SmallVector<const StoreLoadPattern *, 4> members;
  int count() const { return (int)members.size(); }
};

static void greedyBalanceGroups(SmallVector<GroupStruct, 16> &groups) {
  // Find largest group size (target)
  int target = 0;
  for (auto &g : groups)
    target = std::max(target, g.count());

  LLVM_DEBUG({
    llvm::dbgs() << "Largest number of elements in a group (target) = "
                 << target << "\n";
  });

  // Greedy merge loop:
  bool changed = true;
  while (changed) {
    changed = false;
    // find the smallest group with count < target
    int smallestIdx = -1;
    int smallestCount = INT_MAX;
    for (unsigned i = 0; i < groups.size(); ++i) {
      int c = groups[i].count();
      if (c < target && c < smallestCount) {
        smallestCount = c;
        smallestIdx = static_cast<int>(i);
      }
    }
    if (smallestIdx == -1)
      break;

    // try merge smallestIdx with left or right neighbor
    int bestNeighbor = -1;
    int bestMergedCount = -1;
    int bestKeyDiff = INT_MAX;
    int left = smallestIdx - 1;
    int right = smallestIdx + 1;

    // Precompute maxAllowed as integer
    int maxAllowed = (target * 3) / 2;
    // Consider both neighbors with early continue checks to reduce nesting.
    for (int nb : {left, right}) {
      // skip out-of-range neighbors immediately
      if (nb < 0 || nb >= static_cast<int>(groups.size()))
        continue;
      int merged = groups[smallestIdx].count() + groups[nb].count();
      // Prevent merging if the merged group is far larger than target.
      if (merged > maxAllowed)
        continue;
      int penalty = std::abs(merged - target);
      int keyDiff;
      if (groups[smallestIdx].key == -1 || groups[nb].key == -1) {
        keyDiff = INT_MAX / 2;
      } else {
        keyDiff = static_cast<int>(
            std::llabs(groups[smallestIdx].key - groups[nb].key));
      }
      if (bestNeighbor == -1) {
        // first valid candidate
        bestNeighbor = nb;
        bestMergedCount = merged;
        bestKeyDiff = keyDiff;
        continue;
      }
      int currentBestPenalty = std::abs(bestMergedCount - target);
      if (penalty < currentBestPenalty ||
          (penalty == currentBestPenalty && keyDiff < bestKeyDiff)) {
        bestNeighbor = nb;
        bestMergedCount = merged;
        bestKeyDiff = keyDiff;
      }
    }
    if (bestNeighbor == -1) {
      // no neighbor to merge with
      break;
    }
    // Merge smaller into neighbor (keep order: merge into the neighbor's slot
    // and erase the other to keep adjacency)
    int a = std::min(smallestIdx, bestNeighbor);
    int b = std::max(smallestIdx, bestNeighbor);
    // merge groups[a] and groups[b] into groups[a]
    groups[a].members.append(groups[b].members.begin(),
                             groups[b].members.end());
    // update key: prefer known key, else average
    int64_t k1 = groups[a].key;
    int64_t k2 = groups[b].key;
    if (k1 == -1 && k2 != -1)
      groups[a].key = k2;
    else if (k2 == -1 && k1 != -1)
      groups[a].key = k1;
    else
      groups[a].key = (k1 + k2) / 2;
    // erase b
    groups.erase(groups.begin() + b);
    changed = true;
  }
}

static void digitControlledMergeGroups(SmallVector<GroupStruct, 16> &groups,
                                       int desiredGroups) {
  // For desiredGroups > 1: if already small enough, do nothing.
  if ((int)groups.size() <= desiredGroups)
    return;

  // Repeatedly merge adjacent pair whose merged size is smallest
  while ((int)groups.size() > desiredGroups) {
    int bestIdx = -1;
    int bestMergedCount = INT_MAX;
    int bestKeyDiff = INT_MAX;

    for (int i = 0; i + 1 < (int)groups.size(); ++i) {
      int merged = groups[i].count() + groups[i + 1].count();
      int keyDiff;
      if (groups[i].key == -1 || groups[i + 1].key == -1)
        keyDiff = INT_MAX / 2;
      else
        keyDiff = (int)std::llabs(groups[i].key - groups[i + 1].key);

      if (merged < bestMergedCount ||
          (merged == bestMergedCount && keyDiff < bestKeyDiff)) {
        bestIdx = i;
        bestMergedCount = merged;
        bestKeyDiff = keyDiff;
      }
    }

    if (bestIdx == -1) {
      // Nothing to merge (shouldn't happen), break to avoid infinite loop.
      break;
    }

    // Merge bestIdx and bestIdx+1 into bestIdx
    groups[bestIdx].members.append(groups[bestIdx + 1].members.begin(),
                                   groups[bestIdx + 1].members.end());
    int64_t k1 = groups[bestIdx].key;
    int64_t k2 = groups[bestIdx + 1].key;
    if (k1 == -1 && k2 != -1)
      groups[bestIdx].key = k2;
    else if (k2 == -1 && k1 != -1)
      groups[bestIdx].key = k1;
    else
      groups[bestIdx].key = (k1 + k2) / 2;

    groups.erase(groups.begin() + bestIdx + 1);
  }
}

static void groupAndBalancePatterns(
    llvm::ArrayRef<StoreLoadPattern> patterns,
    SmallVectorImpl<SmallVector<const StoreLoadPattern *, 4>> &outGroups,
    int digit) {

  // Map from key to bucket
  llvm::DenseMap<int64_t, SmallVector<const StoreLoadPattern *, 4>> buckets;
  for (const auto &p : patterns) {
    int64_t key = p.embeddingSize;
    buckets[key].push_back(&p);
  }

  // Sorted keys
  SmallVector<int64_t, 16> keys;
  for (auto &kv : buckets)
    keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());

  // Build initial groups
  SmallVector<GroupStruct, 16> groups;
  for (int64_t k : keys) {
    GroupStruct g;
    g.key = k;
    g.members = buckets[k];
    groups.push_back(std::move(g));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Initial groups by embedding size: (" << groups.size()
                 << " groups)\n";
    for (size_t i = 0; i < groups.size(); ++i) {
      llvm::dbgs() << " group[" << i << "] key=" << groups[i].key
                   << " count=" << groups[i].count() << "\n";
      for (auto *p : groups[i].members) {
        llvm::dbgs() << "  - store: ";
        if (p && p->storeOp)
          p->storeOp->print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";
        llvm::dbgs() << "\n";
      }
    }
  });

  if (groups.empty()) {
    outGroups.clear();
    return;
  }
  if (digit != 9) {
    // Compute what GREEDY would produce (on a copy) and print it so user can
    // see.
    SmallVector<GroupStruct, 16> greedyGroups = groups;
    greedyBalanceGroups(greedyGroups);
    LLVM_DEBUG({
      llvm::dbgs() << "*** GREEDY OPTIMAL SUGGESTS " << greedyGroups.size()
                   << " groups *** \n \n";
    });
  }

  if (digit == 9) {
    LLVM_DEBUG({ llvm::dbgs() << "RUNNING GREEDY \n"; });
    // Call helper for original greedy balancing
    greedyBalanceGroups(groups);
  } else {
    // Call helper for digit-controlled merging (1..8)
    digitControlledMergeGroups(groups, digit);

    // After user-controlled merging, print resulting groups so user sees the
    // final grouping that followed their request.
    LLVM_DEBUG({
      llvm::dbgs() << "USER-SPECIFIED merging to " << digit
                   << " groups produced " << groups.size() << " groups:\n";
    });
  }

  // Convert to output
  outGroups.clear();
  for (auto &g : groups) {
    SmallVector<const StoreLoadPattern *, 4> v;
    v.append(g.members.begin(), g.members.end());
    outGroups.push_back(std::move(v));
  }

  // sort by smaller amount of store, from testing this order gives slightly
  // better performance
  // TODO: play around with it
  std::sort(outGroups.begin(), outGroups.end(),
            [](const SmallVector<const StoreLoadPattern *, 4> &a,
               const SmallVector<const StoreLoadPattern *, 4> &b) {
              // Primary: smaller group first
              if (a.size() != b.size())
                return a.size() < b.size();

              // Tie-breaker: bigger embedding size/key first
              int64_t keyA = a.empty() ? -1 : a.front()->embeddingSize;
              int64_t keyB = b.empty() ? -1 : b.front()->embeddingSize;

              return keyA > keyB;
            });

  LLVM_DEBUG({
    llvm::dbgs() << "Balanced groups: (" << outGroups.size() << " groups)\n";
    for (size_t gi = 0; gi < outGroups.size(); ++gi) {
      auto &g = outGroups[gi];
      int64_t approxKey = (g.empty() ? -1 : g.front()->embeddingSize);
      llvm::dbgs() << " group[" << gi << "] approxKey=" << approxKey
                   << " size=" << g.size() << "\n";
      for (auto *p : g) {
        llvm::dbgs() << "  - store: ";
        if (p && p->storeOp)
          p->storeOp->print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";
        llvm::dbgs() << "\n";
      }
    }
  });
}

// Find the old group size

// TODO: currently this method/algo works for the cases I tested so far but may
// need to be generalized later
static int64_t
computeGroupOldSize(const SmallVectorImpl<const StoreLoadPattern *> &group,
                    triton::FuncOp parentFunc) {
  // find largest last-dim from broadcasts in dependentOps
  if (auto maybe = findLastDimFromBroadcasts(group)) {
    int64_t sz = *maybe;
    // Try to find a make_range op that had that end value in the function.
    // We don't need to return the op any more, so just return the size.
    Operation *foundMk = nullptr;
    parentFunc.walk([&](triton::MakeRangeOp mk) {
      if (mk.getEnd() == sz) {
        foundMk = mk.getOperation();
      }
    });
    if (foundMk)
      return sz;
  }
  return -1;
}

class LoopRestructureArangeOptimizationPass
    : public impl::LoopRestructureArangeOptimizationBase<
          LoopRestructureArangeOptimizationPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::StringRef myPassName = this->getArgument();
    int digit = mlir::triton::util::getPassColumnDigit(module, myPassName);
    if (digit == 0)
      return;
    if (digit == 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "digit == 1: single group requested; "
                 << "original is already one group -> no changes made. \n");
      return;
    }
    for (Operation &op : module.getOps()) {
      auto func = dyn_cast<triton::FuncOp>(op);
      if (!func)
        continue;
      if (failed(processFunction(func, digit))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult processFunction(triton::FuncOp func, int digit) {
    LLVM_DEBUG(llvm::dbgs()
               << "Processing function: " << func.getName() << "\n");

    // Collect store-load patterns

    // TODO: currently only support store a load value (cat). Only this pattern
    // we define as an independent block of ops we can move/split loops, but we
    // can expand it, any kernel with multiple stores can be split up to
    // seperated indepedent blocks
    SmallVector<StoreLoadPattern, 16> patterns;
    func.walk<WalkOrder::PreOrder>([&](triton::StoreOp storeOp) {
      Value storedVal;

      if (storeOp->getNumOperands() >= 2)
        storedVal = storeOp->getOperand(1);
      else
        WalkResult::advance();

      Operation *def = storedVal.getDefiningOp();
      if (!def)
        return;

      // Case 1: store(load)
      if (isa<triton::LoadOp>(def)) {

        StoreLoadPattern p;
        p.storeOp = storeOp;
        p.loadOp = def;

        findDependentOpsFromStore(storeOp, p.dependentOps);
        findMaskAndConstantOps(storeOp, p);
        extractEmbeddingSize(p);

        patterns.push_back(std::move(p));
        return;
      }

      // Case 2: store(broadcast(load))
      if (isa<triton::BroadcastOp>(def)) {
        Operation *innerLoad = nullptr;

        for (Value v : def->getOperands()) {
          if (Operation *d = v.getDefiningOp()) {
            if (isa<triton::LoadOp>(d)) {
              innerLoad = d;
              break;
            }
          }
        }

        if (!innerLoad)
          return;

        StoreLoadPattern p;
        p.storeOp = storeOp;
        p.loadOp = innerLoad;

        findDependentOpsFromStore(storeOp, p.dependentOps);
        findMaskAndConstantOps(storeOp, p);
        extractEmbeddingSize(p);

        patterns.push_back(std::move(p));
        return;
      }
    });

    if (patterns.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No store-load patterns found\n");
      return success();
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Found " << patterns.size() << " store-load patterns\n");

    // Dump all patterns for debugging
    LLVM_DEBUG({
      llvm::dbgs() << "Dumping all patterns:\n";
      for (auto &p : patterns) {
        llvm::dbgs() << "=== StoreLoadPattern ===\n";

        llvm::dbgs() << " storeOp: ";
        if (p.storeOp)
          p.storeOp->print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";

        llvm::dbgs() << "\n loadOp: ";
        if (p.loadOp)
          p.loadOp->print(llvm::dbgs());
        else
          llvm::dbgs() << "<null>";

        llvm::dbgs() << "\n embeddingSize: " << p.embeddingSize << "\n";

        llvm::dbgs() << " maskOps (" << p.maskOps.size() << "):\n";
        for (Operation *m : p.maskOps) {
          if (m)
            m->print(llvm::dbgs());
          else
            llvm::dbgs() << "<null>";
          llvm::dbgs() << "\n";
        }

        llvm::dbgs() << " constantOps (" << p.constantOps.size() << "):\n";
        for (Operation *c : p.constantOps) {
          if (c)
            c->print(llvm::dbgs());
          else
            llvm::dbgs() << "<null>";
          llvm::dbgs() << "\n";
        }

        llvm::dbgs() << " dependentOps (" << p.dependentOps.size() << "):\n";
        for (Operation *d : p.dependentOps) {
          if (d)
            d->print(llvm::dbgs());
          else
            llvm::dbgs() << "<null>";
          llvm::dbgs() << "\n";
        }

        llvm::dbgs() << "=======================\n";
      }
    });

    for (auto &p : patterns) {
      if (p.embeddingSize <= 0) {
        LLVM_DEBUG(llvm::dbgs() << "SKIP PASS: Embedding size <= 0 \n");
        return success();
      }
    }

    // Group patterns by embedding size and balance them with simple greedy
    // merge
    SmallVector<SmallVector<const StoreLoadPattern *, 4>, 8> groups;
    groupAndBalancePatterns(patterns, groups, digit);
    LLVM_DEBUG(llvm::dbgs() << "Grouped into " << groups.size() << " groups\n");

    // Find a representative scf.for (if any).
    scf::ForOp foundFor = nullptr;
    int loopCount = 0;
    func.walk([&](scf::ForOp f) {
      loopCount++;
      foundFor = f;
      return WalkResult::advance();
    });
    // For now only work with at most one loop.
    if (loopCount > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "SKIPPING: More than 1 loop is not supported yet \n");
      return success();
    }

    // Axis-aware mask analysis above identifies a last-dimension bound, while
    // old-size recovery remains a separate conservative analysis. Keep an
    // explicit shrink-only invariant between the two results. Expanding a
    // tensor type without expanding flattened producers such as tt.make_range
    // would create invalid reshape operations.
    //
    // Validate every group before mutation; unsupported dependency graphs
    // retain the existing whole-function fallback.
    // Preserve the existing behavior when an unchanged group is present next
    // to a shrinkable group: equality is a no-op, not an unsupported
    // expansion. Only skip for all-equal groups after checking every group.
    bool allSizesMatch = !groups.empty();
    for (const auto &group : groups) {
      if (group.empty())
        continue;
      if (!canCloneDependencies(group)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "SKIP PASS: unsupported dependency cloning\n");
        return success();
      }
      int64_t newSize = 0;
      for (auto *p : group)
        newSize = std::max(newSize, p->embeddingSize);
      int64_t oldSize = computeGroupOldSize(group, func);
      if (oldSize < 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "SKIP PASS: unable to infer group oldSize\n");
        return success();
      }
      if (oldSize < newSize) {
        LLVM_DEBUG(llvm::dbgs()
                   << "SKIP PASS: group oldSize=" << oldSize
                   << " is smaller than newSize=" << newSize << "\n");
        return success();
      }
      if (oldSize != newSize) {
        if (!canCloneLastAxisPrefix(group, oldSize)) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "SKIP PASS: dependent graph is not a last-axis slice\n");
          return success();
        }
        allSizesMatch = false;
      }
    }
    if (allSizesMatch) {
      LLVM_DEBUG(llvm::dbgs()
                 << "SKIP PASS: oldSize == newSize for all groups\n");
      return success();
    }
    if (loopCount == 1) {
      for (const auto &p : patterns) {
        Operation *storeParent = p.storeOp->getParentOp();
        bool insideLoop = false;
        // Walk u the parent chain to check if store is nested in foundFor
        while (storeParent && storeParent != func) {
          if (storeParent == foundFor.getOperation()) {
            insideLoop = true;
            break;
          }
          storeParent = storeParent->getParentOp();
        }
        if (!insideLoop) {
          LLVM_DEBUG(llvm::dbgs()
                     << "SKIPPING: Store pattern not inside the single loop: "
                     << p.storeOp << "\n");
          return success();
        }
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "ALL" << patterns.size()
                 << " store patterns verified insid single loop \n");
    }
    return processGroupsInPlace(func, groups);
  }

  /// Width grouping must not schedule memory operations. Copy each dependency
  /// immediately before its original operation, with independent SSA mappings
  /// per width. Stores stay in source order, loads never cross writes, and loop
  /// placement/iteration order is unchanged even when all pointers may alias.
  LogicalResult processGroupsInPlace(
      triton::FuncOp func,
      SmallVectorImpl<SmallVector<const StoreLoadPattern *, 4>> &groups) {
    SmallVector<llvm::DenseSet<Operation *>, 4> groupOps(groups.size());
    SmallVector<llvm::DenseMap<Operation *, Operation *>, 4> clonedOps(
        groups.size());
    SmallVector<llvm::DenseMap<Value, Value>, 4> valueMappings(groups.size());
    SmallVector<int64_t, 4> oldSizes;
    SmallVector<int64_t, 4> newSizes;
    llvm::DenseSet<Operation *> allOps;
    SmallVector<Operation *> storesToDelete;
    for (auto [gi, group] : llvm::enumerate(groups)) {
      int64_t newSize = 0;
      for (auto *pattern : group) {
        newSize = std::max(newSize, pattern->embeddingSize);
        groupOps[gi].insert(pattern->dependentOps.begin(),
                            pattern->dependentOps.end());
        storesToDelete.push_back(pattern->storeOp);
      }
      allOps.insert(groupOps[gi].begin(), groupOps[gi].end());
      oldSizes.push_back(computeGroupOldSize(group, func));
      newSizes.push_back(newSize);
    }

    // Snapshot before mutation: do not visit newly created copies. Regions
    // are retained in place, not recreated by the regionless cloner.
    SmallVector<Operation *> ordered;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (allOps.contains(op))
        ordered.push_back(op);
    });
    // Textual block order need not follow SSA dominance. Build producer maps
    // before their users, even if the producer's block is printed later.
    // This only orders clone construction: each copy is still inserted at its
    // original operation, so neither CFG nor memory-operation order changes.
    if (!computeTopologicalSorting(ordered))
      return success();
    PatternRewriter rewriter(func.getContext());
    for (Operation *op : ordered) {
      for (unsigned gi = 0; gi < groups.size(); ++gi) {
        if (!groupOps[gi].contains(op))
          continue;
        cloneGroupOperations(rewriter, op->getLoc(), {op}, op, oldSizes[gi],
                             newSizes[gi], clonedOps[gi], gi,
                             valueMappings[gi]);
      }
    }
    for (Operation *store : storesToDelete)
      rewriter.eraseOp(store);
    return success();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createLoopRestructureArangeOptimizationPass() {
  return std::make_unique<LoopRestructureArangeOptimizationPass>();
}

} // namespace triton
} // namespace bishengir
