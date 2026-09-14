//===- RewriteSliceOpToMemoryOps.cpp -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Attempts to optimize tensor.extract_slice ops to tt.load ops and
// tensor.insert_slice ops to tt.store ops
//
// For tensor.extract_slice ops
// If there are only elementwise ops whose operands are the same between the
// tt.load source of the tensor.extract_slice op then emit another smaller load
// and copy the chain of ops (starting form the load op, adjusting the sizes of
// those ops as needed) instead of using tensor.extract_slice
//
// For tensor.insert_slice ops
// If there are only elementwise ops whose operands come from only one operation
// between the result of the tensor.insert_slice op and the tt.store dest then:
//   - copy the chain of ops (starting from the tensor to insert, adjusting the
//     sizes of those ops as needed), and emit a store at the end
//   - If the original store op used tensor of ptrs (tensor<shape x !tt.ptr>),
//     add/modify the mask to avoid storing at the insert tensor location
//   - At the end, remove all store ops that have been fully masked out
//
// If unable to optimize a certain tensor.insert_slice/tensor.extract_slice op,
// leaves it alone
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cstddef>
#include <functional>

namespace bishengir::triton {
#define GEN_PASS_DEF_REWRITESLICEOPTOMEMORYOPS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

enum class CompareResult {
  LT, // less than
  EQ, // equal
  GT, // greater than
  NA  // incomparable
};

bool isSGT(const CompareResult &result) { return result == CompareResult::GT; }

bool isSLT(const CompareResult &result) { return result == CompareResult::LT; }

bool isEQ(const CompareResult &result) { return result == CompareResult::EQ; }

bool isNA(const CompareResult &result) { return result == CompareResult::NA; }

// struct used to track offset locations, used to mark start/ends of slices
struct ValuePlusInt {
  bool onlyInt;
  int64_t intVal = 0;
  Value val = nullptr;

  ValuePlusInt() = default;

  explicit ValuePlusInt(int64_t intVal) : intVal(intVal) { onlyInt = true; }

  // if val is a constant op, stores the int value instead
  explicit ValuePlusInt(Value val) {
    this->val = val;
    if (auto constOp =
            dyn_cast_if_present<arith::ConstantOp>(val.getDefiningOp())) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        intVal = intAttr.getInt();
        onlyInt = true;
        return;
      }
    }
    onlyInt = val == nullptr;
  }

  // if val is a constant op, adds it to the int value instead
  explicit ValuePlusInt(Value val, int intVal) {
    this->val = val;
    if (auto constOp =
            dyn_cast_if_present<arith::ConstantOp>(val.getDefiningOp())) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        intVal = intAttr.getInt() + intVal;
        onlyInt = true;
        return;
      }
    }
    onlyInt = val == nullptr;
    this->intVal = intVal;
  }

  bool isPureInt() const { return onlyInt; }

  int64_t getInt() const { return intVal; }

  Value getVal() const { return val; }

  CompareResult compare(const ValuePlusInt &other) const {
    if (onlyInt ^ other.isPureInt()) {
      // If one value is constant and the other is variable, cannot compare them
      return CompareResult::NA;
    }

    if (onlyInt) {
      int64_t otherInt = other.getInt();
      if (intVal < otherInt) {
        return CompareResult::LT;
      }
      if (intVal == otherInt) {
        return CompareResult::EQ;
      }
      return CompareResult::GT;
    }
    if (getVal() == other.getVal()) {
      int64_t otherInt = other.getInt();
      if (intVal < otherInt) {
        return CompareResult::LT;
      }
      if (intVal == otherInt) {
        return CompareResult::EQ;
      }
      return CompareResult::GT;
    }
    return CompareResult::NA;
  }

  bool isSLT(const ValuePlusInt &other) const {
    return compare(other) == CompareResult::LT;
  }

  bool isSGT(const ValuePlusInt &other) const {
    return compare(other) == CompareResult::GT;
  }

  bool isEQ(const ValuePlusInt &other) const {
    return compare(other) == CompareResult::EQ;
  }

  bool isLTE(const ValuePlusInt &other) const {
    CompareResult result = compare(other);
    return result == CompareResult::EQ || result == CompareResult::LT;
  }

  bool isGTE(const ValuePlusInt &other) const {
    CompareResult result = compare(other);
    return result == CompareResult::EQ || result == CompareResult::GT;
  }

  // Splats the value stored by the ValuePlusInt to the shape in
  // RankedTensorType type
  Value getConstantI32Tensor(OpBuilder &builder, Location loc,
                             RankedTensorType type) {
    Value res;
    if (onlyInt) {
      DenseElementsAttr constAttr =
          DenseElementsAttr::get(type, builder.getI32IntegerAttr(intVal));
      res = builder.create<arith::ConstantOp>(loc, constAttr);
    } else {
      // Note the MLIR Value's stored by the ValuePlusInt struct will have type
      // `index` as tensor.insert_slice/extract_slice ops expect an index type
      // for dynamic offsets
      Type i32Type = builder.getI32Type();
      Value splatVal = builder.create<arith::IndexCastOp>(loc, i32Type, val);
      if (intVal != 0) {
        Value constIntVal = builder.create<arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(intVal));
        splatVal = builder.create<arith::AddIOp>(loc, splatVal, constIntVal);
      }
      res = builder.create<triton::SplatOp>(loc, type, splatVal);
    }

    return res;
  }
};

// stores info about a insert_slice/extract_slice op
struct SliceInfo {
  int axis = -1;
  int64_t N = 0; // large.dim[axis]
  int64_t S = 0; // small.dim[axis]

  ValuePlusInt offset;
};

// stores what intervals are overriden by insert_slice ops for a specific
// tt.store op
struct MemoryIntervals {
  bool unused = false;
  bool modified = false;
  size_t dimRemaining = 0;
  // stores [start, end) pairs for the insert range of tensor.insert_slice ops
  SmallVector<std::pair<ValuePlusInt, ValuePlusInt>> sliceIntervals;
  // stores the shape of the associated store op
  SmallVector<int64_t> dims;

  void initialize(ArrayRef<int64_t> dims) {
    this->dims = SmallVector<int64_t>(dims);
  }

  // Adds the slice interval, updates unused if the store is now fully overriden
  // by tensor.insert_slice ops
  void trackSlice(size_t dimLeft, ValuePlusInt start, ValuePlusInt end) {
    if (!modified) {
      modified = true;
      dimRemaining = dimLeft;
    } else {
      if (dimRemaining != dimLeft) {
        unused = true;
        return;
      }
    }

    SmallVector<std::pair<ValuePlusInt, ValuePlusInt>> curDim = sliceIntervals;

    SmallVector<std::pair<ValuePlusInt, ValuePlusInt>> newDimIntervals;
    newDimIntervals.reserve(curDim.size());

    bool changed = true;

    // loop that combines intervals until no more changes occur
    while (changed) {
      changed = false;
      for (size_t i = 0; i < curDim.size(); i++) {
        auto [curIntervalStart, curIntervalEnd] = curDim[i];

        if (curIntervalStart.isSGT(end) || curIntervalEnd.isSLT(start)) {
          newDimIntervals.emplace_back(curIntervalStart, curIntervalEnd);
          continue;
        }

        CompareResult cmpStarts = curIntervalStart.compare(start);
        CompareResult cmpEnds = curIntervalEnd.compare(end);

        if (isNA(cmpStarts) || isNA(cmpEnds)) {
          newDimIntervals.emplace_back(curIntervalStart, curIntervalEnd);
          continue;
        }

        ValuePlusInt left;
        ValuePlusInt right;

        if (isSLT(cmpStarts)) {
          left = curIntervalStart;
          changed = true;
        } else {
          left = start;
        }

        if (isSGT(cmpEnds)) {
          right = curIntervalEnd;
          changed = true;
        } else {
          right = end;
        }

        start = left;
        end = right;
      }

      curDim = newDimIntervals;
      newDimIntervals.clear();
    }

    if (start.isSLT(end)) {
      curDim.emplace_back(start, end);
      changed = true;
    }

    sliceIntervals = curDim;
    if (curDim.size() == 1) {
      ValuePlusInt start = curDim[0].first;
      ValuePlusInt end = curDim[0].second;

      bool startEqZero = isEQ(start.compare(ValuePlusInt(0)));
      bool endEqDim = isEQ(end.compare(ValuePlusInt(dims[dimLeft])));

      unused = startEqZero && endEqDim;
    }
  }

  void trackSlice(size_t dimleft, ValuePlusInt start, size_t size) {
    ValuePlusInt end = start;
    end.intVal += size;
    trackSlice(dimleft, start, end);
  }

  bool isUnused() const { return unused; }
};

// returns true if an op's operands are all the same
bool hasSameOperands(Operation *op) {
  Value val = nullptr;
  for (Value operand : op->getOperands()) {
    if (!val) {
      val = operand;
    } else if (val != operand) {
      return false;
    }
  }

  return true;
}

// stores a tree recording the usage of a triton::StoreOp to the stores of its
// results
struct StoreTree {
  SmallVector<triton::StoreOp, 1> storeOps;
  DenseMap<Operation *, SmallVector<Operation *>> childMapping;
  tensor::InsertSliceOp root;

  StoreTree(tensor::InsertSliceOp op) : root(op) {}

  void addStore(triton::StoreOp storeOp) { storeOps.push_back(storeOp); }

  void addChild(Operation *parent, Operation *child) {
    childMapping[parent].push_back(child);
  }

  ArrayRef<Operation *> getChildren(Operation *parent) const {
    if (childMapping.contains(parent)) {
      return childMapping.at(parent);
    }
    return {};
  }

  void clear() {
    storeOps.clear();
    childMapping.clear();
  }

  bool empty() const { return storeOps.empty(); }

  SmallVector<triton::StoreOp, 1> &getStoreOps() { return storeOps; }

  SmallVector<Operation *> &getChildren(Operation *root) {
    return childMapping[root];
  }

  tensor::InsertSliceOp getRoot() { return root; }
};

// given a tensor.insert_slice or tensor.extract_slice op, determines if the
// slice op is supported for this optimization pass and the slice op information
std::optional<SliceInfo> getSliceInfo(Operation *op) {
  SliceInfo info;
  RankedTensorType large;
  RankedTensorType small;
  ArrayRef<int64_t> sizes;
  ArrayRef<int64_t> strides;
  ArrayRef<int64_t> offsets;

  if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    large = extractSliceOp.getSourceType();
    small = extractSliceOp.getResultType();
    sizes = extractSliceOp.getStaticSizes();
    strides = extractSliceOp.getStaticStrides();
    offsets = extractSliceOp.getStaticOffsets();
  } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    large = insertSliceOp.getDestType();
    small = insertSliceOp.getSourceType();
    sizes = insertSliceOp.getStaticSizes();
    strides = insertSliceOp.getStaticStrides();
    offsets = insertSliceOp.getStaticOffsets();
  }
  if (large.getRank() < 1 || large.getRank() != small.getRank()) {
    op->emitError("large and small tensors must have matching rank >= 1; "
                  "got large ")
        << large << " vs small " << small;
    return std::nullopt;
  }
  const int rank = large.getRank();

  // Find the unique differing axis.
  for (int i = 0; i < rank; ++i) {
    if (large.getDimSize(i) != small.getDimSize(i)) {
      if (info.axis != -1) {
        op->emitError("only single-axis slicing is supported; both axis ")
            << info.axis << " and axis " << i
            << " differ between large and small";
        return std::nullopt;
      }
      info.axis = i;
    }
  }

  if (info.axis == -1)
    return info; // no-op: shapes match.

  info.N = large.getDimSize(info.axis);
  info.S = small.getDimSize(info.axis);

  if (static_cast<int>(sizes.size()) != rank ||
      static_cast<int>(strides.size()) != rank ||
      static_cast<int>(offsets.size()) != rank) {
    op->emitError("offsets/sizes/strides arity must match rank ") << rank;
    return std::nullopt;
  }

  for (int i = 0; i < rank; ++i) {
    const int64_t expected = (i == info.axis) ? info.S : large.getDimSize(i);
    if (sizes[i] != expected) {
      op->emitError("size[")
          << i << "] must be " << expected << "; got " << sizes[i];
      return std::nullopt;
    }
  }

  for (int i = 0; i < rank; ++i) {
    if (strides[i] != 1) {
      op->emitError("strides must all be 1; got non-unit stride at axis ") << i;
      return std::nullopt;
    }
  }

  bool isOffsetStatic = true;

  for (int i = 0; i < rank; ++i) {
    if (ShapedType::isDynamic(offsets[i])) {
      isOffsetStatic = false;
      if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
        info.offset =
            ValuePlusInt(insertSliceOp.getMixedOffsets()[i].get<Value>());
      } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
        info.offset =
            ValuePlusInt(extractSliceOp.getMixedOffsets()[i].get<Value>());
      }
    }
    if (i != info.axis && offsets[i] != 0) {
      if (ShapedType::isDynamic(offsets[i])) {
        op->emitError("offset[")
            << i << "] must be 0; got a dynamic offset instead";
      } else {
        op->emitError("offset[") << i << "] must be 0; got " << offsets[i];
      }

      return std::nullopt;
    }
  }
  if (isOffsetStatic) {
    info.offset = ValuePlusInt(offsets[info.axis]);
    if (offsets[info.axis] < 0 || offsets[info.axis] > info.N - info.S) {
      op->emitError("offset at indexed axis ")
          << info.axis << " must be in [0, " << (info.N - info.S)
          << "] for size " << info.S << " on a dim of " << info.N << "; got "
          << offsets[info.axis];
      return std::nullopt;
    }
  }
  return info;
}

// Given a tensor.extract_slice op, finds the path between the extract slice op
// and a tt.load op.
// If the source does not originate from a load, or if some op in
// the middle is not elementwise, or some op in the middle has operands that are
// not the same, returns an empty SmallVector to indicate failure
SmallVector<Operation *> getPathToParentLoad(tensor::ExtractSliceOp op) {
  SmallVector<Operation *> stack;
  stack.push_back(op);
  stack.push_back(op.getSource().getDefiningOp());

  while (true) {
    Operation *cur = stack.back();

    if (!cur) {
      stack.clear();
      break;
    }

    if (isa<triton::LoadOp>(cur)) {
      break;
    }

    if (cur->hasTrait<OpTrait::Elementwise>() && hasSameOperands(cur)) {
      stack.push_back(cur->getOperand(0).getDefiningOp());
    } else {
      stack.clear();
      break;
    }
  }

  return stack;
}

// Given a tensor.insert_slice op, finds the path between the extract slice op
// and a tt.store op.
// If the source does not originate from a load, or if some op in
// the middle is not elementwise, or some op in the middle has operands that are
// not the same, returns an empty StoreTree to indicate failure
StoreTree getPathsToChildStores(tensor::InsertSliceOp op) {
  SmallVector<Operation *> stack;
  StoreTree tree(op);

  stack.push_back(op);

  while (!stack.empty()) {
    Operation *cur = stack.back();
    stack.pop_back();

    if (!cur) {
      continue;
    }

    if (auto storeOp = dyn_cast<triton::StoreOp>(cur)) {
      tree.addStore(storeOp);
      continue;
    }

    bool abort = false;
    for (Operation *user : cur->getUsers()) {
      if (!isa<triton::StoreOp>(user) &&
          !(user->hasTrait<OpTrait::Elementwise>() && hasSameOperands(user))) {
        abort = true;
        break;
      }

      tree.addChild(cur, user);
      stack.push_back(user);
    }

    if (abort) {
      tree.clear();
      break;
    }
  }

  return tree;
}

// Simulates the ptr/mask recalculation to ensure that the ptr/mask
// recalculation will be successful before commiting to IR modifications.
bool canClone(Value cur) {
  SmallVector<Value> stack = {cur};
  while (!stack.empty()) {
    Operation *cur = stack.back().getDefiningOp();
    stack.pop_back();
    bool failed = false;
    TypeSwitch<Operation *>(cur)
        .Case<triton::AddPtrOp>([&](triton::AddPtrOp addPtrOp) {
          stack.push_back(addPtrOp.getPtr());
          stack.push_back(addPtrOp.getOffset());
        })
        .Case<triton::BroadcastOp>([&](triton::BroadcastOp broadcastOp) {
          stack.push_back(broadcastOp.getSrc());
        })
        .Case<triton::ExpandDimsOp>([&](triton::ExpandDimsOp expandDimsOp) {
          stack.push_back(expandDimsOp.getSrc());
        })
        .Case<triton::MakeRangeOp, triton::SplatOp>([](Operation *) { return; })
        .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
          if (auto oldType = dyn_cast<RankedTensorType>(constOp.getType())) {
            auto denseAttr = cast<DenseElementsAttr>(constOp.getValue());
            if (!denseAttr.isSplat()) {
              failed = false;
            }
          }
        })
        .Case<arith::MulIOp, arith::AddIOp, arith::CmpIOp, arith::AndIOp>(
            [&](Operation *op) {
              stack.push_back(op->getOperand(0));
              stack.push_back(op->getOperand(1));
            })
        .Case<arith::ExtSIOp>([&](arith::ExtSIOp extsiOp) {
          if (auto oldType = dyn_cast<RankedTensorType>(extsiOp.getType())) {
            stack.push_back(extsiOp.getIn());
          }
        })
        .Default([&](Operation *) { failed = true; });

    if (failed) {
      return false;
    }
  }

  return true;
}

// Clones the history of a ptr/mask Value. If not possible, returns nullptr.
// In practice, checks using canClone should prevent this function from ever
// needing to return nullptr
Value cloneAndUpdate(
    OpBuilder &builder, Value cur, int64_t sliceAxisIdx,
    ArrayRef<int64_t> expectedShape,
    std::function<Value(OpBuilder &, Location)> rangeGenerator) {
  Operation *definingOp = cur.getDefiningOp();
  if (!definingOp) {
    return nullptr;
  }

  Location loc = definingOp->getLoc();
  Value res = nullptr;

  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(definingOp)) {
    Value splatPtr = cloneAndUpdate(builder, addPtrOp.getPtr(), sliceAxisIdx,
                                    expectedShape, rangeGenerator);
    Value ptrOffset =
        cloneAndUpdate(builder, addPtrOp.getOffset(), sliceAxisIdx,
                       expectedShape, rangeGenerator);
    if (splatPtr && ptrOffset) {
      res = builder.create<triton::AddPtrOp>(loc, splatPtr.getType(), splatPtr,
                                             ptrOffset);
    }
  } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(definingOp)) {
    RankedTensorType srcType = broadcastOp.getSrc().getType();
    Type elementType = srcType.getElementType();
    ArrayRef<int64_t> srcShape = srcType.getShape();

    Value oldSrc = broadcastOp.getSrc();
    Value newSrc;

    SmallVector<int64_t> sourceShape(srcShape);
    if (sliceAxisIdx >= 0 && srcShape[sliceAxisIdx] != 1) {
      sourceShape[sliceAxisIdx] = expectedShape[sliceAxisIdx];
    }

    newSrc = cloneAndUpdate(builder, oldSrc, sliceAxisIdx, sourceShape,
                            rangeGenerator);

    if (newSrc) {
      RankedTensorType newType =
          RankedTensorType::get(expectedShape, elementType);
      res = builder.create<triton::BroadcastOp>(loc, newType, newSrc);
    }
  } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(definingOp)) {
    uint32_t expandAxis = expandDimsOp.getAxis();
    Value oldSrc = expandDimsOp.getSrc();
    SmallVector<int64_t> sourceShape(expectedShape);
    sourceShape.erase(sourceShape.begin() + expandAxis);

    int64_t newSliceAxisIdx = sliceAxisIdx;
    if (expandAxis < sliceAxisIdx) {
      newSliceAxisIdx -= 1;
    } else if (expandAxis == sliceAxisIdx) {
      newSliceAxisIdx = -1;
    }

    Value newSource = cloneAndUpdate(builder, oldSrc, newSliceAxisIdx,
                                     sourceShape, rangeGenerator);
    if (newSource) {
      res = builder.create<triton::ExpandDimsOp>(loc, newSource, expandAxis);
    }
  } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(definingOp)) {
    if (sliceAxisIdx == 0) {
      res = rangeGenerator(builder, loc);
    } else {
      res = cur;
    }
  } else if (auto splatOp = dyn_cast<triton::SplatOp>(definingOp)) {
    RankedTensorType oldType = splatOp.getType();
    ArrayRef<int64_t> oldShape = oldType.getShape();
    Type elementType = oldType.getElementType();

    if (sliceAxisIdx >= 0 &&
        sliceAxisIdx < static_cast<int64_t>(oldShape.size()) &&
        oldShape[sliceAxisIdx] != 1) {
      RankedTensorType newType =
          RankedTensorType::get(expectedShape, elementType);
      res = builder.create<triton::SplatOp>(loc, newType, splatOp.getSrc());
    } else {
      res = cur;
    }
  } else if (auto constantOp = dyn_cast<arith::ConstantOp>(definingOp)) {
    if (RankedTensorType oldType =
            dyn_cast<RankedTensorType>(constantOp.getType())) {
      DenseElementsAttr denseAttr =
          cast<DenseElementsAttr>(constantOp.getValue());

      if (denseAttr.isSplat()) {
        Attribute constantVal = denseAttr.getSplatValue<Attribute>();
        ArrayRef<int64_t> oldShape = oldType.getShape();
        Type elementType = oldType.getElementType();

        if (sliceAxisIdx >= 0 &&
            sliceAxisIdx < static_cast<int64_t>(oldShape.size()) &&
            oldShape[sliceAxisIdx] != 1) {
          RankedTensorType newType =
              RankedTensorType::get(expectedShape, elementType);
          DenseElementsAttr newAttr =
              DenseElementsAttr::get(newType, constantVal);
          res = builder.create<arith::ConstantOp>(loc, newAttr);
        } else {
          res = cur;
        }
      }
    } else {
      res = cur;
    }
  } else if (auto muliOp = dyn_cast<arith::MulIOp>(definingOp)) {
    Value lhs = cloneAndUpdate(builder, muliOp.getLhs(), sliceAxisIdx,
                               expectedShape, rangeGenerator);
    Value rhs = cloneAndUpdate(builder, muliOp.getRhs(), sliceAxisIdx,
                               expectedShape, rangeGenerator);

    if (lhs && rhs) {
      res = builder.create<arith::MulIOp>(loc, lhs, rhs);
    }
  } else if (auto addiOp = dyn_cast<arith::AddIOp>(definingOp)) {
    Value lhs = cloneAndUpdate(builder, addiOp.getLhs(), sliceAxisIdx,
                               expectedShape, rangeGenerator);
    Value rhs = cloneAndUpdate(builder, addiOp.getRhs(), sliceAxisIdx,
                               expectedShape, rangeGenerator);

    if (lhs && rhs) {
      res = builder.create<arith::AddIOp>(loc, lhs, rhs);
    }
  } else if (auto andiOp = dyn_cast<arith::AndIOp>(definingOp)) {
    Value lhs = cloneAndUpdate(builder, andiOp.getLhs(), sliceAxisIdx,
                               expectedShape, rangeGenerator);
    Value rhs = cloneAndUpdate(builder, andiOp.getRhs(), sliceAxisIdx,
                               expectedShape, rangeGenerator);

    if (lhs && rhs) {
      res = builder.create<arith::AndIOp>(loc, lhs, rhs);
    }
  } else if (auto cmpiOp = dyn_cast<arith::CmpIOp>(definingOp)) {
    Value lhs = cloneAndUpdate(builder, cmpiOp.getLhs(), sliceAxisIdx,
                               expectedShape, rangeGenerator);
    Value rhs = cloneAndUpdate(builder, cmpiOp.getRhs(), sliceAxisIdx,
                               expectedShape, rangeGenerator);

    if (lhs && rhs) {
      res = builder.create<arith::CmpIOp>(loc, cmpiOp.getPredicate(), lhs, rhs);
    }
  } else if (auto extsiOp = dyn_cast<arith::ExtSIOp>(definingOp)) {
    if (RankedTensorType oldType =
            dyn_cast<RankedTensorType>(extsiOp.getType())) {
      Value newSrc = cloneAndUpdate(builder, extsiOp.getIn(), sliceAxisIdx,
                                    expectedShape, rangeGenerator);
      if (newSrc) {
        Type elementType = oldType.getElementType();
        RankedTensorType newType =
            RankedTensorType::get(expectedShape, elementType);
        res = builder.create<arith::ExtSIOp>(loc, newType, newSrc);
      }
    } else {
      res = cur;
    }
  }
  return res;
}

// Clones and updates the calculation history for Value cur to fit smallShape,
// given that the slice op has a dynamic offset
Value dynamicOffsetCloneAndUpdate(OpBuilder &builder, Value cur,
                                  unsigned int sliceAxisIdx,
                                  ArrayRef<int64_t> smallShape, Value offset) {
  int64_t newDimSize = smallShape[sliceAxisIdx];
  auto createNewRange = [&](OpBuilder &builder, Location loc) -> Value {
    Type i32Type = builder.getI32Type();
    RankedTensorType type = RankedTensorType::get({newDimSize}, i32Type);
    Value range = builder.create<triton::MakeRangeOp>(loc, type, 0, newDimSize);
    Value intOffset = builder.create<arith::IndexCastOp>(loc, i32Type, offset);
    Value constOffset = builder.create<triton::SplatOp>(loc, type, intOffset);
    Value res = builder.create<arith::AddIOp>(loc, range, constOffset);
    return res;
  };

  return cloneAndUpdate(builder, cur, sliceAxisIdx, smallShape, createNewRange);
}

// Clones and updates the calculation history for Value cur to fit smallShape,
// given that the slice op has a static offset
Value staticOffsetCloneAndUpdate(OpBuilder &builder, Value cur,
                                 unsigned int sliceAxisIdx,
                                 ArrayRef<int64_t> smallShape, int64_t offset) {
  int64_t newDimSize = smallShape[sliceAxisIdx];
  auto createNewRange = [&](OpBuilder &builder, Location loc) -> Value {
    RankedTensorType type =
        RankedTensorType::get({newDimSize}, builder.getI32Type());
    Value range = builder.create<triton::MakeRangeOp>(loc, type, offset,
                                                      offset + newDimSize);
    return range;
  };

  return cloneAndUpdate(builder, cur, sliceAxisIdx, smallShape, createNewRange);
}

// Returns true if ptr can be successfully cloned
// Used as a safeguard before making IR modifications
bool canClonePtr(Value ptr) {
  if (auto makeTensorPtrOp =
          dyn_cast_if_present<triton::MakeTensorPtrOp>(ptr.getDefiningOp())) {
    return true;
  }

  return canClone(ptr);
}

// Returns true if mask can be successfully cloned
// Used as a safeguard before making IR modifications
bool canCloneMask(Value mask) {
  if (!mask) {
    return true;
  }
  return canClone(mask);
}

// Creates a new pointer arg based off of the old ptr's calculation history and
// slice info's target size
Value getNewPtr(OpBuilder &builder, Value ptr, const SliceInfo &info) {
  if (auto makeTensorPtrOp =
          dyn_cast_if_present<triton::MakeTensorPtrOp>(ptr.getDefiningOp())) {
    // if ptr argument comes from a tt.make_tensor_ptr op, just change the
    // offsets and return shape of the op
    Location loc = makeTensorPtrOp->getLoc();
    Value base = makeTensorPtrOp.getBase();
    auto shape = makeTensorPtrOp.getShape();
    auto strides = makeTensorPtrOp.getStrides();
    SmallVector<Value> newOffsets = makeTensorPtrOp.getOffsets();
    auto order = makeTensorPtrOp.getOrder();
    triton::PointerType oldType = makeTensorPtrOp.getType();
    RankedTensorType oldTensorType =
        cast<RankedTensorType>(oldType.getPointeeType());
    Type elementType = oldTensorType.getElementType();
    ArrayRef<int64_t> oldShape = oldTensorType.getShape();

    SmallVector<int64_t> newShape(oldShape);
    newShape[info.axis] = info.S;
    RankedTensorType retTensorType =
        RankedTensorType::get(newShape, elementType);
    triton::PointerType retType =
        triton::PointerType::get(retTensorType, oldType.getAddressSpace());

    Value offsetValue = nullptr;
    if (info.offset.isPureInt()) {
      offsetValue = builder.create<arith::ConstantOp>(
          loc, builder.getI32IntegerAttr(info.offset.getInt()));
    } else {
      offsetValue = info.offset.getVal();
    }

    newOffsets[info.axis] =
        builder.create<arith::AddIOp>(loc, newOffsets[info.axis], offsetValue);
    Value newPtr = builder.create<triton::MakeTensorPtrOp>(
        loc, retType, base, shape, strides, newOffsets, order);
    return newPtr;
  } else {
    // clone ptr calculation history
    SmallVector<int64_t> expectedShape(
        cast<RankedTensorType>(ptr.getType()).getShape());
    expectedShape[info.axis] = info.S;
    if (info.offset.isPureInt()) {
      return staticOffsetCloneAndUpdate(builder, ptr, info.axis, expectedShape,
                                        info.offset.getInt());
    } else {
      return dynamicOffsetCloneAndUpdate(builder, ptr, info.axis, expectedShape,
                                         info.offset.getVal());
    }
  }

  return nullptr;
}

// Creates a new mask arg based off of the old mask's calculation history and
// slice info's target size
Value getNewMask(OpBuilder &builder, Value mask, const SliceInfo &info) {
  if (!mask) {
    return nullptr;
  }
  SmallVector<int64_t> expectedShape(
      cast<RankedTensorType>(mask.getType()).getShape());
  expectedShape[info.axis] = info.S;
  if (info.offset.isPureInt()) {
    return staticOffsetCloneAndUpdate(builder, mask, info.axis, expectedShape,
                                      info.offset.getInt());
  } else {
    return dynamicOffsetCloneAndUpdate(builder, mask, info.axis, expectedShape,
                                       info.offset.getVal());
  }
}

// For a specific tensor.extract_slice op, clone the chain of its usage to its
// load with reduced size given by the slice info, and remove the
// tensor.extract_slice op
LogicalResult cloneLoadHistory(PatternRewriter &rewriter,
                               tensor::ExtractSliceOp op,
                               SmallVector<Operation *> &pathToLoad,
                               const SliceInfo &info) {
  Location loc = op->getLoc();
  triton::LoadOp loadOp = cast<triton::LoadOp>(pathToLoad.back());
  Value ptr = loadOp.getPtr();
  Value mask = loadOp.getMask();

  if (!canClonePtr(ptr) || !canCloneMask(mask)) {
    return failure();
  }

  ArrayRef<int64_t> shape = op.getStaticSizes();
  Value newPtr = getNewPtr(rewriter, ptr, info);

  if (!newPtr) {
    return failure();
  }

  Value res = nullptr;
  if (loadOp.getMask()) {
    Value newMask = getNewMask(rewriter, loadOp.getMask(), info);
    if (!newMask) {
      return failure();
    }
    res = rewriter.create<triton::LoadOp>(loc, newPtr, newMask,
                                          loadOp.getCache(), loadOp.getEvict(),
                                          loadOp.getIsVolatile());
  } else {
    res = rewriter.create<triton::LoadOp>(loc, newPtr, loadOp.getCache(),
                                          loadOp.getEvict(),
                                          loadOp.getIsVolatile());
  }

  // skip iterating over both the load and extract slice op
  for (size_t i = pathToLoad.size() - 2; i >= 1; i--) {
    Operation *opToClone = pathToLoad[i];
    // clone op and update res
    // opToClone has elementwise and OneOperand trait
    Operation *cloned = rewriter.clone(*opToClone);

    rewriter.modifyOpInPlace(cloned, [&]() {
      for (unsigned int i = 0; i < cloned->getNumOperands(); i++) {
        cloned->setOperand(i, res);
      }
      for (Value result : cloned->getResults()) {
        ShapedType prevType = cast<ShapedType>(result.getType());
        ShapedType newType =
            prevType.cloneWith(shape, prevType.getElementType());
        result.setType(newType);
      }
    });

    unsigned int resIdxUsed =
        cast<OpResult>(pathToLoad[i - 1]->getOperand(0)).getResultNumber();

    res = cloned->getResult(resIdxUsed);
  }

  rewriter.replaceOp(op, res);
  return success();
}

// For a specific tensor.insert_slice op, clone the tree of its usage to its
// stores with reduced size given by the slice info, and remove the
// tensor.insert_slice op.
// Also updates the liveness ranges of the store ops after lowering the
// insert_slice op
LogicalResult cloneStoreTreeHistory(
    PatternRewriter &rewriter, SliceInfo &info, StoreTree &storeTree,
    DenseMap<triton::StoreOp, MemoryIntervals> &storeOpLiveness) {
  SmallVector<Operation *> stack;
  SmallPtrSet<Operation *, 4> visited;
  tensor::InsertSliceOp root = storeTree.getRoot();
  ArrayRef<int64_t> smallShape = root.getSource().getType().getShape();
  ArrayRef<int64_t> largeShape = root.getDest().getType().getShape();
  stack.push_back(root);
  visited.insert(root);

  for (triton::StoreOp storeOp : storeTree.getStoreOps()) {
    Value ptr = storeOp.getPtr();
    Value mask = storeOp.getMask();
    if (!canClonePtr(ptr) || !canCloneMask(mask)) {
      return failure();
    }
  }

  Value res;
  DenseMap<Operation *, Operation *> cloneMapping;

  while (!stack.empty()) {
    Operation *cur = stack.back();
    stack.pop_back();

    for (Operation *user : cur->getUsers()) {
      // Determine which specific result of cur is used by user
      unsigned int resIdxUsed;
      Value operand;
      if (isa<triton::StoreOp>(user)) {
        operand = user->getOperand(1);
      } else {
        operand = user->getOperand(0);
      }
      resIdxUsed = cast<OpResult>(operand).getResultNumber();
      if (cloneMapping.contains(cur)) {
        Operation *clone = cloneMapping.at(cur);
        res = clone->getResult(resIdxUsed);
      } else {
        res = root.getSource();
      }

      if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
        rewriter.setInsertionPointAfter(storeOp);
        Location loc = storeOp.getLoc();
        Value oldPtr = storeOp.getPtr();
        Value oldMask = storeOp.getMask();

        Value newPtr = getNewPtr(rewriter, oldPtr, info);
        Value newMask = getNewMask(rewriter, oldMask, info);
        // Create new store
        if (newMask) {
          rewriter.create<triton::StoreOp>(loc, newPtr, res, newMask,
                                           storeOp.getCache(),
                                           storeOp.getEvict());
        } else {
          rewriter.create<triton::StoreOp>(loc, newPtr, res, storeOp.getCache(),
                                           storeOp.getEvict());
        }

        if (!storeOpLiveness.contains(storeOp)) {
          storeOpLiveness[storeOp].initialize(largeShape);
        }

        storeOpLiveness[storeOp].trackSlice(info.axis, info.offset, info.S);
      } else {
        if (!visited.contains(user)) {
          stack.push_back(user);
          visited.insert(user);
        }
        rewriter.setInsertionPointAfter(user);
        Operation *cloned = rewriter.clone(*user);
        cloneMapping[user] = cloned;
        rewriter.modifyOpInPlace(cloned, [&]() {
          for (unsigned int i = 0; i < cloned->getNumOperands(); i++) {
            cloned->setOperand(i, res);
          }

          for (Value result : cloned->getResults()) {
            ShapedType prevType = cast<ShapedType>(result.getType());
            ShapedType newType =
                prevType.cloneWith(smallShape, prevType.getElementType());
            result.setType(newType);
          }
        });
      }
    }
  }

  // replace insert slice with dest - stored the results separately
  rewriter.replaceOp(root, root.getDest());
  return success();
}

// Given a record of what parts of store ops have been made redundant by
// tensor.insert_slice lowering, either add masks to existing store ops or
// remove the store ops entirely
void maskStores(
    const DenseMap<triton::StoreOp, MemoryIntervals> &storeOpLiveness) {
  for (auto pair : storeOpLiveness) {
    triton::StoreOp storeOp = pair.getFirst();
    const MemoryIntervals &intervalInfo = pair.getSecond();

    if (intervalInfo.isUnused()) {
      storeOp->erase();
    } else if (!isa<triton::PointerType>(storeOp.getPtr().getType())) {
      const auto &intervals = intervalInfo.sliceIntervals;
      Location loc = storeOp.getLoc();
      IRRewriter rewriter(storeOp);
      ArrayRef<int64_t> storeShape =
          cast<RankedTensorType>(storeOp.getValue().getType()).getShape();
      Type i32Type = rewriter.getI32Type();
      Type i1Type = rewriter.getI1Type();
      SmallVector<int64_t> curShape = {storeShape[intervalInfo.dimRemaining]};
      RankedTensorType type = RankedTensorType::get(curShape, i32Type);
      Value range = rewriter.create<triton::MakeRangeOp>(
          loc, type, 0, storeShape[intervalInfo.dimRemaining]);
      bool isRange = true;

      Value curMask;

      for (auto [liveStart, liveEnd] : intervals) {
        Value upperbound = liveEnd.getConstantI32Tensor(rewriter, loc, type);
        Value cmpRes = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, range, upperbound);
        if (isRange) {
          curMask = cmpRes;
          isRange = false;
        } else {
          curMask = rewriter.create<arith::OrIOp>(loc, curMask, cmpRes);
        }

        if (!liveStart.isPureInt() || liveStart.getInt() > 0) {
          Value lowerbound =
              liveStart.getConstantI32Tensor(rewriter, loc, type);
          cmpRes = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, range, lowerbound);
          curMask = rewriter.create<arith::AndIOp>(loc, curMask, cmpRes);
        }
      }

      // expand dims to create 1x.xNx..x1xT tensor

      size_t dimensionInsertLoc = 0;
      for (size_t i = 0; i < storeShape.size(); i++) {
        if (i == intervalInfo.dimRemaining) {
          dimensionInsertLoc = i + 1;
          continue;
        }
        if (dimensionInsertLoc == 0) {
          curShape.insert(curShape.begin(), 1);
        } else {
          curShape.push_back(1);
        }

        RankedTensorType curType = RankedTensorType::get(curShape, i1Type);
        curMask = rewriter.create<triton::ExpandDimsOp>(loc, curType, curMask,
                                                        dimensionInsertLoc);
      }

      type = RankedTensorType::get(storeShape, rewriter.getI1Type());

      curMask = rewriter.create<triton::BroadcastOp>(loc, type, curMask);

      // Perform logical not using xor
      DenseElementsAttr trueAttr =
          DenseElementsAttr::get(type, rewriter.getOneAttr(i1Type));
      Value constOnes = rewriter.create<arith::ConstantOp>(loc, trueAttr);
      curMask = rewriter.create<arith::XOrIOp>(loc, curMask, constOnes);

      if (Value oldMask = storeOp.getMask()) {
        curMask = rewriter.create<arith::AndIOp>(loc, curMask, oldMask);
      }

      rewriter.modifyOpInPlace(
          storeOp, [&]() { storeOp.getMaskMutable().assign(curMask); });
    }
  }
}

struct ExtractSliceToLoadPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<SliceInfo> potentialSliceInfo = getSliceInfo(op);
    if (!potentialSliceInfo) {
      // unsupported slice op
      return failure();
    }

    SliceInfo info = *potentialSliceInfo;
    if (info.axis == -1) {
      // no op
      Value source = op.getSource();
      rewriter.replaceOp(op, source);
      return success();
    }

    SmallVector<Operation *> pathToLoad = getPathToParentLoad(op);
    if (pathToLoad.empty()) {
      return failure();
    }

    return cloneLoadHistory(rewriter, op, pathToLoad, info);
  }
};

struct InsertSliceToStorePattern
    : public OpRewritePattern<tensor::InsertSliceOp> {

  DenseMap<triton::StoreOp, MemoryIntervals> &storeOpLiveness;
  InsertSliceToStorePattern(
      MLIRContext *ctx,
      DenseMap<triton::StoreOp, MemoryIntervals> &storeOpLiveness)
      : OpRewritePattern<tensor::InsertSliceOp>(ctx),
        storeOpLiveness(storeOpLiveness) {}

  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<SliceInfo> potentialSliceInfo = getSliceInfo(op);
    if (!potentialSliceInfo) {
      // unsupported slice op
      return failure();
    }

    SliceInfo info = *potentialSliceInfo;
    if (info.axis == -1) {
      // no op
      Value source = op.getSource();
      rewriter.replaceOp(op, source);
      return success();
    }

    StoreTree pathsToStores = getPathsToChildStores(op);
    if (pathsToStores.empty()) {
      return failure();
    }

    return cloneStoreTreeHistory(rewriter, info, pathsToStores,
                                 storeOpLiveness);
  }
};

class RewriteSliceOpToMemoryOpsPass
    : public impl::RewriteSliceOpToMemoryOpsBase<
          RewriteSliceOpToMemoryOpsPass> {
public:
  using RewriteSliceOpToMemoryOpsBase::RewriteSliceOpToMemoryOpsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto *ctx = &getContext();

    DenseMap<triton::StoreOp, MemoryIntervals> storeOpLiveness;

    RewritePatternSet patterns(ctx);
    patterns.add<ExtractSliceToLoadPattern>(ctx);
    patterns.add<InsertSliceToStorePattern>(ctx, storeOpLiveness);

    if (failed(applyPatternsGreedily(mod, std::move(patterns)))) {
      mod.emitError("Unsupported tensor slicing operations found in the "
                    "SIMT kernel");
      signalPassFailure();
      return;
    }

    // based off of store op liveness calculated from lowered
    // tensor.insert_slice ops, add masks to their associated store ops or
    // delete the store ops entirely
    maskStores(storeOpLiveness);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createRewriteSliceOpToMemoryOpsPass() {
  return std::make_unique<RewriteSliceOpToMemoryOpsPass>();
}

} // namespace bishengir::triton
