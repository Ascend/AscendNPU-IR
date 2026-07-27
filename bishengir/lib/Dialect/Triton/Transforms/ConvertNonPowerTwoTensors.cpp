//===- ConvertNonPowerTwoTensors.cpp -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass converts tensors with non-power of 2 sizes to tensors of power of 2
// sizes so triton accepts them. This pass pads tensors until their dimensions
// are powers of 2
// Ignores operations that only act on power of two tensors/tensor pointers (as
// well as non tensor/tensor pointer operations)
//
// Important data structures:
// LocalTensorShapeData: stores the data shape (initial, actual shape of the
// tensor) and the virtual shape (the padded shape of the tensor)
// DenseMap<Value, LocalTensorShapeData>: maps operation tensor results/operands
// to their shape data (only for non power of 2 tensors) DenseMap<Value,
// PotentialPaddingRequirements>: For each non power of 2 tensor value, keeps
// track of:
//   - The Value's immediate users that have a padding requirement (and which
//   operands have which padding requirement)
//   - The Value's downstream users that have a padding requirement (and which
//   operands have which padding requirement)
// DenseMap<Value, SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand*,
// 2>>>>:
//   For each non power of 2 tensor value, keeps track of:
//   - The Value's required paddings that have been chosen to be set at this
//   value in the final codegen stage (TypedAttr)
//   - The Operands that require the specific padding for each padding type
//   (SmallPtrSet<OpOperand*, 2>)
//
// Overall Steps:
// 1: Scan for non power of 2 tensor ops that are not supported by this pass
// 2: For each tt.load/tt.store/tt.atomic_rmw op: if the pointer operand is of
//    type tensor<shape x !tt.ptr<type>> then add a default mask filled with
//    true, other filled with 0 (for load)
//    Otherwise if ptr operand is of type !tt.ptr<tensor<shape x elemment type>,
//    sets boundary check attr
// 3: When slice ops can go out of bounds due to expansion of the non power of 2
//    dim, or when tensor.insert_slice ops that have a non power of 2 dim along
//    the slice axis and expanding them would override previous existing data,
//    split up the tensor.insert_slice/tensor.extract_slice op into power
//    of 2 dimensions along the slice axis
// 4: Create a DenseMap<Value, LocalTensorShapeData> that tracks the data shape
//    and virtual shape of values
// 5: Create a DenseMap<Value, PotentialPaddingRequirements> that tracks the
//    padding requirement of its usage chain
// 6: Create a DenseMap<Value, SmallVector<std::pair<TypedAttr,
//    SmallPtrSet<OpOperand *, 2>>>> (finalize padding choices) based on
//    DenseMap<Value, PotentialPaddingRequirements>
//   - Currently uses a very naive/basic approach, could be optimized
// 7: Replace non power of 2 tensors via preorder traversal (for nested ops,
//    visit outside wrapping op before visiting the inside), top to bottom
//    This guarantees we visit an op's sources before we visit the op itself
//
// Notable implementation details:
// triton::ReduceOp:
//   - No change if the reduce axis was a power of 2
//   - For each input tensor (or 'lane') tries to find an identity for the math
//   calculation
//     For example, if one tensor gets reduced by arith.addf, finds the identity
//     0.0
//       - Limitations: Only supports reduction algorithms that have one
//       operation (and have an identity), and must only access its own
//       accumulator and value argument
//   - If able to find an identity, requests that the tensor corresponding to
//   that reduction algorithm is padded with the identity element
//   - Otherwise, adds the mask tensor as an argument and uses it to mask out
//   unwanted elements
//
// triton::ScanOp:
//   - Very similar implementation to triton::ReduceOp, except:
//     Only care about padding/masking if reverse=true currently
//     - If reverse=false, then the elements at the end are padding and don't
//     affect calculations
//
// triton::LoadOp/triton::StoreOp/triton::AtomicRMWOp:
//   - For load/store op's that accept a tensor<sizex!tt.ptr<>> argument, we add
//   an initial mask (all true)
//      - Note that masks are NOT added if the load/store op accepts a
//      !tt.ptr<tensor> arg
//   - No other changes, these are treated as 'general tensor ops'
//
// triton::ReshapeOp:
//   - If just updating the operand and result shape results in the data being
//   in the wrong locations,
//     instead reshapes to 1D and uses slice ops to move around data before
//     reshaping to result shape
//
// tensor::ExtractSliceOp/tensor::InsertSliceOp:
//   - Supports cases where only the tensors are only sliced along one dimension
//   - Needed for tensor::InsertSliceOp correctness, RewriteSliceOpToTriton.cpp
//   also has this restriction
//   - Sometimes needs to split the insert_slice/extract_slice into multiple
//   slice ops which have powers of two dimensions along the slice axis
//     - slice ops are split when the offset is dynamic or the offset + expanded
//     result axis dim > expanded source axis dim
//        - insert_slice ops are also split when the slice axis of the insert
//        tensor is not a power of 2, and expanding would override original data
//
// arith::ConstantOp:
//   - Note that constant ops in the form `arith.constant dense<[1, 2, 3]>` for
//   example with non power of two dimensions are not supported
//   - This is because of a restriction that requires arith.constant dense ops
//   declared this way to have num elements == num threads per warp
//
// triton::MakeTensorPtrOp:
//   - Just updates the size of the output tensor
//   - When this tensor is loaded, relies on MakeTensorPtrOp's lowering to
//   provide masking/valid ptrs
//
// General Tensor Ops: (see isGeneralTensorOp(Operation* op) for which ops are
// categorized as a general tensor op)
//   - Note that some ops that implement the InferTypeOpInterface are not
//   supported yet
//   - These are ops that implement the InferTypeOpInterface/have the
//   elementwise trait and some manually added ones
//       - For these ops, we just need to update some operand/return types
//   - Ex: elementwise ops, triton::MakeTensorPtrOp, scf::ForOp, triton::LoadOp,
//   triton::StoreOp, etc
//
//   - Sometimes used as a cleanup step to finish modifying more complex ops
//   (Ex: tt.reduce, tt.scan)
//
// Other notes:
// data shape refers to the shape of tensors before the pass
// virtual shape refers to the padded shape of tensors after the pass
// check isSupportedOp(Operation* op) to view which ops are
// supported/unsupported
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <optional>
#include <queue>

namespace bishengir::triton {
#define GEN_PASS_DEF_CONVERTNONPOWERTWOTENSORS
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

// Given a data shape, calculate virtual shape and virtual shape size
std::pair<SmallVector<int64_t>, uint64_t>
expandShape(ArrayRef<int64_t> dataShape) {
  SmallVector<int64_t> dest;
  dest.reserve(dataShape.size());

  uint64_t totalSize = 1;

  for (int64_t dim : dataShape) {
    uint64_t expandedDim = llvm::PowerOf2Ceil(dim);
    totalSize *= expandedDim;

    dest.push_back(expandedDim);
  }

  return std::make_pair(std::move(dest), totalSize);
}

// For a non power of two tensor Value, stores:
//    non power of two size (data shape),
//    padded size (localVirtualShape),
//    num elements in padded tensor (localVirtualShapeSize)
struct LocalTensorShapeData {
  SmallVector<int64_t> localVirtualShape;
  uint64_t localVirtualShapeSize = 0;
  ArrayRef<int64_t> dataShape;

  LocalTensorShapeData() = default;

  explicit LocalTensorShapeData(ArrayRef<int64_t> dataShape)
      : dataShape(dataShape) {
    initializeShape(dataShape);
  }

  void initializeShape(const ArrayRef<int64_t> initialDataShape) {
    localVirtualShape.clear();
    auto [shape, size] = expandShape(initialDataShape);
    localVirtualShape = std::move(shape);
    localVirtualShapeSize = size;
  }
};

// Stores the padding requirements of a non power of two tensor Value and its
// chain of users
struct PotentialPaddingRequirements {
  // Counts the number of select ops that would be saved by having this op's
  // upstream sources padded
  DenseMap<TypedAttr, SmallPtrSet<OpOperand *, 2>> looseReqCounts;

  // Lists strict padding requirements that MUST be handled at this Value, and
  // for which ops
  DenseMap<TypedAttr, SmallPtrSet<OpOperand *, 2>> strictReqs;

  TypedAttr popMostFrequentLooseReq() {
    if (looseReqCounts.size() == 0) {
      return nullptr;
    }

    TypedAttr mostFrequentReq;
    unsigned int mostFrequentReqUses = 0;

    for (const auto &count : looseReqCounts) {
      if (count.second.size() > mostFrequentReqUses) {
        mostFrequentReq = count.first;
        mostFrequentReqUses = count.second.size();
      }
    }

    removeLooseReq(mostFrequentReq);
    return mostFrequentReq;
  }

  void removeLooseReq(TypedAttr req) { looseReqCounts.erase(req); }

  void addLoose(TypedAttr padding, OpOperand *operand) {
    looseReqCounts[padding].insert(operand);
  }

  void addStrict(TypedAttr padding, OpOperand *operand) {
    // Add to padding count
    addLoose(padding, operand);

    strictReqs[padding].insert(operand);
  }
};

// Stores reqs for a Value during padding backwards dataflow analysis
struct AllPaddingRequirements {
  SmallVector<TypedAttr, 2> reqs;

  bool initialized = false;

  ChangeResult setToDefault() {
    if (!initialized) {
      initialized = true;
      return ChangeResult::Change;
    }
    if (reqs.size() == 0) {
      return ChangeResult::NoChange;
    }

    reqs.clear();
    return ChangeResult::Change;
  }

  bool has(TypedAttr attr) {
    for (TypedAttr reqAttr : reqs) {
      if (reqAttr == attr) {
        return true;
      }
    }

    return false;
  }

  // returns ChangeResult::Change if a new attr was inserted,
  // ChangeResult::NoChange otherwise (attr already in list)
  ChangeResult add(TypedAttr attr) {
    if (has(attr)) {
      return ChangeResult::NoChange;
    }
    reqs.push_back(attr);
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const {}

  ChangeResult meet(const AllPaddingRequirements &other) {
    if (!other.initialized) {
      return ChangeResult::NoChange;
    }
    if (!initialized) {
      initialized = true;
      reqs = other.reqs;
      return ChangeResult::Change;
    }

    ChangeResult changed = ChangeResult::NoChange;

    for (TypedAttr attr : other.reqs) {
      if (add(attr) == ChangeResult::Change) {
        changed = ChangeResult::Change;
      }
    }

    return changed;
  }

  static AllPaddingRequirements join(const AllPaddingRequirements &lhs,
                                     const AllPaddingRequirements &rhs) {
    AllPaddingRequirements newReqs;
    newReqs.reqs = lhs.reqs;
    for (TypedAttr attr : rhs.reqs) {
      newReqs.reqs.push_back(attr);
    }
    newReqs.initialized = lhs.initialized || rhs.initialized;
    return newReqs;
  }

  // does not get run
  static AllPaddingRequirements meet(const AllPaddingRequirements &lhs,
                                     const AllPaddingRequirements &rhs) {
    return AllPaddingRequirements();
  }

  bool operator==(const AllPaddingRequirements &other) {
    if (!initialized) {
      return !other.initialized;
    }
    if (!other.initialized) {
      return false;
    }
    return reqs == other.reqs;
  }
};

// Stores data about a tensor::InsertSliceOp or tensor::ExtractSliceOp
struct SliceOpData {
  int axis = -1;
  bool isOffsetStatic = false;
  int64_t offsetVal = -1;
  OpFoldResult offset;
  uint64_t largeAxisDim = 0;
  uint64_t smallAxisDim = 0;
  size_t rank;

  SliceOpData() {}
  SliceOpData(Operation *op, ArrayRef<int64_t> smallDataShape,
              ArrayRef<int64_t> largeDataShape)
      : rank(largeDataShape.size()) {
    ArrayRef<int64_t> offsets;
    axis = -1;
    offsetVal = -1;

    for (size_t i = 0; i < rank; i++) {
      if (largeDataShape[i] != smallDataShape[i]) {
        axis = static_cast<int>(i);
      }
    }

    if (axis == -1) {
      return;
    }

    smallAxisDim = static_cast<uint64_t>(smallDataShape[axis]);
    largeAxisDim = static_cast<uint64_t>(largeDataShape[axis]);

    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      offsets = extractSliceOp.getStaticOffsets();
      offset = extractSliceOp.getMixedOffsets()[axis];
    } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      offsets = insertSliceOp.getStaticOffsets();
      offset = insertSliceOp.getMixedOffsets()[axis];
    }

    if (!ShapedType::isDynamic(offsets[axis])) {
      offsetVal = offsets[axis];
      isOffsetStatic = true;
    }
  }
  SliceOpData(int axis, int64_t offsetVal, OpFoldResult offset,
              int64_t largeAxisDim, int64_t smallAxisDim, size_t rank)
      : axis(axis), offsetVal(offsetVal), offset(offset),
        largeAxisDim(largeAxisDim), smallAxisDim(smallAxisDim), rank(rank) {
    if (offsetVal == -1) {
      isOffsetStatic = false;
    }
  }
};

bool isTensorProducer(Operation *op) {
  return isa<triton::SplatOp, triton::MakeRangeOp, arith::ConstantOp>(op);
}

bool isGenericTensorOp(Operation *op) {
  if (isa<triton::ReduceOp, triton::ScanOp>(op)) {
    return false;
  }

  if (op->hasTrait<OpTrait::Elementwise>()) {
    return true;
  }

  if (isa<triton::AdvanceOp, triton::AtomicRMWOp, triton::BroadcastOp,
          triton::MakeTensorPtrOp>(op)) {
    return true;
  }

  if (isa<scf::ForOp, scf::IfOp>(op)) {
    return true;
  }

  if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op)) {
    return true;
  }

  if (isa<InferTypeOpInterface>(op)) {
    return true;
  }

  return false;
}

bool hasTensorArgs(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (isa<RankedTensorType>(operand.getType())) {
      return true;
    }
  }

  return false;
}

bool hasNonPowerTwoDim(ArrayRef<int64_t> shape) {
  for (int64_t dim : shape) {
    if (!llvm::isPowerOf2_64(dim)) {
      return true;
    }
  }

  return false;
}

RankedTensorType getNonPowerTwoTensorType(Type type) {
  RankedTensorType tensorType = dyn_cast<RankedTensorType>(type);
  if (tensorType && hasNonPowerTwoDim(tensorType.getShape())) {
    return tensorType;
  }
  return nullptr;
}

RankedTensorType getNestedTensorType(Type type) {
  RankedTensorType tensorType = nullptr;
  if (auto pointerType = dyn_cast<triton::PointerType>(type)) {
    if (auto pointeeType =
            dyn_cast<RankedTensorType>(pointerType.getPointeeType())) {
      tensorType = pointeeType;
    }
  } else if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
    tensorType = rankedTensorType;
  }
  return tensorType;
}

RankedTensorType getNonPowerTwoNestedTensorType(Type type) {
  RankedTensorType tensorType = getNestedTensorType(type);
  if (tensorType && hasNonPowerTwoDim(tensorType.getShape())) {
    return tensorType;
  }
  return nullptr;
}

bool isNestedTensorType(Type type) {
  return getNestedTensorType(type) != nullptr;
}

bool isNonPowerTwoTensorOrTensorPtr(Type type) {
  return getNonPowerTwoNestedTensorType(type) != nullptr;
}

bool isTensorPointerLoadStoreOperation(Operation *op) {
  return isa<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp>(op) &&
         isa<triton::PointerType>(op->getOperandTypes()[0]);
}

bool hasTensorRes(Operation *op) {
  for (Value operand : op->getResults()) {
    if (isNestedTensorType(operand.getType())) {
      return true;
    }
  }

  return false;
}

bool isNonPow2TensorOperation(Operation *op) {
  for (Value val : op->getOperands()) {
    if (isNonPowerTwoTensorOrTensorPtr(val.getType())) {
      return true;
    }
  }

  for (Value val : op->getResults()) {
    if (isNonPowerTwoTensorOrTensorPtr(val.getType())) {
      return true;
    }
  }

  if (auto scfForOp = dyn_cast<scf::ForOp>(op)) {
    for (BlockArgument regionArg : scfForOp.getRegionIterArgs()) {
      if (isNonPowerTwoTensorOrTensorPtr(regionArg.getType())) {
        return true;
      }
    }
  }

  return false;
}

Value createTensor(OpBuilder &builder, Location loc,
                   RankedTensorType tensorType,
                   Attribute paddingVal = nullptr) {
  Value tensor;
  Type elementType = tensorType.getElementType();
  if (auto ptrType = dyn_cast<triton::PointerType>(elementType)) {
    Value scalarZero =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
    Value tritonNullptr =
        builder.create<triton::IntToPtrOp>(loc, elementType, scalarZero);
    tensor = builder.create<triton::SplatOp>(loc, tensorType, tritonNullptr);
  } else {
    DenseElementsAttr constAttr;
    if (paddingVal) {
      constAttr = DenseElementsAttr::get(tensorType, paddingVal);
    } else {
      constAttr =
          DenseElementsAttr::get(tensorType, builder.getZeroAttr(elementType));
    }
    tensor = builder.create<arith::ConstantOp>(loc, constAttr);
  }

  return tensor;
}

// Given the data (unpadded) shape of a tensor and the size of its virtual
// (padded) shape, Returns a vector<bool> that is true where there is data when
// flattened, false otherwise
std::vector<bool> getFlattenedDataLocs(ArrayRef<int64_t> dataShape,
                                       uint64_t virtualShapeSize) {
  std::vector<bool> res;
  res.reserve(virtualShapeSize);

  for (uint64_t i = 0; i < virtualShapeSize; i++) {
    bool inShape = true;
    uint64_t rem = i;
    for (int64_t r = static_cast<int64_t>(dataShape.size() - 1); r >= 0; r--) {
      uint64_t fullDim = llvm::PowerOf2Ceil(dataShape[r]);
      uint64_t coord = rem % fullDim;
      rem /= fullDim;

      if (coord >= static_cast<uint64_t>(dataShape[r])) {
        inShape = false;
        break;
      }
    }
    res.push_back(inShape);
  }

  return res;
}

// If the triton::ReshapeOp is complex, returns the data locations of a
// flattened operand and result, otherwise return std::nullopt
std::optional<std::pair<std::vector<bool>, std::vector<bool>>>
getComplexReshapeDataLocs(const LocalTensorShapeData &operandData,
                          const LocalTensorShapeData &resultData) {
  size_t operandIdx = 0;
  size_t resultIdx = 0;
  ArrayRef<int64_t> operandShape = operandData.localVirtualShape;
  ArrayRef<int64_t> resultShape = resultData.localVirtualShape;
  std::vector<bool> sourceDataLocs = getFlattenedDataLocs(
      operandData.dataShape, operandData.localVirtualShapeSize);
  std::vector<bool> resultDataLocs = getFlattenedDataLocs(
      resultData.dataShape, resultData.localVirtualShapeSize);

  std::pair<std::vector<bool>, std::vector<bool>> result{sourceDataLocs,
                                                         resultDataLocs};

  // Checks the dimensions are equal while skipping over dimensions of size 1
  while (resultIdx < resultShape.size() && operandIdx < operandShape.size()) {
    while (operandIdx < operandShape.size() && operandShape[operandIdx] == 1) {
      operandIdx += 1;
    }

    while (resultIdx < resultShape.size() && resultShape[resultIdx] == 1) {
      resultIdx += 1;
    }

    if (operandIdx == operandShape.size() && resultIdx == resultShape.size()) {
      return std::nullopt;
    }

    if (operandShape[operandIdx] != resultShape[resultIdx]) {
      return result;
    }

    operandIdx += 1;
    resultIdx += 1;
  }

  return result;
}

bool isSupportedSliceOp(Operation *op) {
  ArrayRef<int64_t> largeShape;
  ArrayRef<int64_t> smallShape;
  ArrayRef<int64_t> offsets;
  ArrayRef<int64_t> strides;

  if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    largeShape = extractSliceOp.getSource().getType().getShape();
    smallShape = extractSliceOp.getResult().getType().getShape();
    offsets = extractSliceOp.getStaticOffsets();
    strides = extractSliceOp.getStaticStrides();
  } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    largeShape = insertSliceOp.getDest().getType().getShape();
    smallShape = insertSliceOp.getSource().getType().getShape();
    offsets = insertSliceOp.getStaticOffsets();
    strides = insertSliceOp.getStaticStrides();
  }

  int rank = static_cast<int>(largeShape.size());
  int axis = -1;

  if (largeShape.size() < 1 || largeShape.size() != smallShape.size()) {
    op->emitError("large and small tensors must have matching rank >= 1; "
                  "got large ")
        << largeShape << " vs small " << smallShape;
    return false;
  }

  for (int i = 0; i < rank; i++) {
    if (largeShape[i] != smallShape[i]) {
      if (axis != -1) {
        op->emitError("only single-axis slicing is supported; both axis ")
            << axis << " and axis " << i << " differ between large and small";
        return false;
      }
      axis = i;
    }
  }

  for (int i = 0; i < rank; ++i) {
    if (strides[i] != 1) {
      op->emitError("strides must all be 1; got non-unit stride at axis ") << i;
      return false;
    }
  }

  for (int i = 0; i < rank; ++i) {
    if (i != axis && offsets[i] != 0) {
      return false;
    }
  }

  return true;
}

// Determines is an op is supported by this pass
bool isSupportedOp(Operation *op) {
  if (!isNonPow2TensorOperation(op) || !hasTensorRes(op)) {
    return true;
  }

  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    if (!cast<DenseElementsAttr>(constantOp.getValue()).isSplat()) {
      op->emitError("Non splat tensor constant ops must satisfy num elements "
                    "== threads per warp");
      return false;
    }
    return true;
  }

  if (isa<tensor::InsertOp, tensor::ExtractOp>(op)) {
    return true;
  }

  if (isa<tensor::InsertSliceOp, tensor::ExtractSliceOp>(op)) {
    return isSupportedSliceOp(op);
  }

  if (isa<scf::IfOp, scf::ForOp>(op)) {
    return true;
  }

  if (isa<triton::AdvanceOp, triton::AtomicRMWOp, triton::BroadcastOp,
          triton::DotOp, triton::ExpandDimsOp, triton::GatherOp,
          triton::HistogramOp, triton::LoadOp, triton::MakeRangeOp,
          triton::MakeTensorPtrOp, triton::ReduceOp, triton::ReshapeOp,
          triton::ScanOp, triton::StoreOp, triton::SplatOp, triton::TransOp>(
          op)) {
    return true;
  }

  if (isa<triton::SplitOp, triton::JoinOp>(op)) {
    return false;
  }

  if (op->hasTrait<OpTrait::Elementwise>()) {
    return true;
  }

  return false;
}

// Makes sure all ops are supported by this pass
LogicalResult verifyOps(FuncOp &mod) {
  bool allOpsSupported = true;
  mod->walk([&](Operation *op) {
    if (!isSupportedOp(op)) {
      op->emitError(
          "Non power of two tensor operation is currently unsupported");
      allOpsSupported = false;
    }
  });

  if (allOpsSupported) {
    return success();
  } else {
    return failure();
  }
}

// Searches for triton::LoadOp, triton::StoreOp, and triton::AtmoicRMW ops that
// take in a tensor<!tt.ptr<>> for their pointer operand.
// For load/store ops that have !tt.ptr<tensor<>> pointers, sets boundary check
// attr instead
void addMasksToLoadAndStores(FuncOp &mod) {
  std::queue<Operation *> workQueue;
  std::queue<std::pair<Operation *, RankedTensorType>> tensorPtrWorkQueue;

  mod.walk([&](Operation *op) {
    Type ptrArgType = nullptr;
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      if (loadOp.getMask()) {
        return;
      }
      ptrArgType = loadOp.getPtr().getType();
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      if (storeOp.getMask()) {
        return;
      }
      ptrArgType = storeOp.getPtr().getType();
    } else if (auto atomicRmwOp = dyn_cast<triton::AtomicRMWOp>(op)) {
      if (atomicRmwOp.getMask()) {
        return;
      }
      ptrArgType = atomicRmwOp.getPtr().getType();
    }
    if (ptrArgType) {
      if (RankedTensorType type = getNonPowerTwoNestedTensorType(ptrArgType)) {
        if (isa<triton::PointerType>(ptrArgType)) {
          tensorPtrWorkQueue.emplace(op, type);
        } else {
          workQueue.push(op);
        }
      }
    }
  });

  while (!workQueue.empty()) {
    Operation *cur = workQueue.front();
    workQueue.pop();
    IRRewriter builder(cur);
    Location loc = cur->getLoc();
    Type i1Type = builder.getI1Type();
    TypedAttr trueAttr = builder.getOneAttr(i1Type);

    // create true mask
    RankedTensorType shapeType =
        cast<RankedTensorType>(cur->getOperand(0).getType());

    RankedTensorType resType =
        RankedTensorType::get(shapeType.getShape(), i1Type);
    Value maskOp = createTensor(builder, loc, resType, trueAttr);

    // Replace op
    if (auto loadOp = dyn_cast<triton::LoadOp>(cur)) {
      // Create other value mask
      Value ptr = loadOp.getPtr();
      auto newOp = builder.create<triton::LoadOp>(
          loc, ptr, maskOp, loadOp.getCache(), loadOp.getEvict(),
          loadOp.getIsVolatile());
      builder.replaceOp(cur, newOp);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(cur)) {
      Value ptr = storeOp.getPtr();
      Value val = storeOp.getValue();
      builder.create<triton::StoreOp>(loc, ptr, val, maskOp,
                                      storeOp.getBoundaryCheck(),
                                      storeOp.getCache(), storeOp.getEvict());
      storeOp->erase();
    } else if (auto atomicRmwOp = dyn_cast<triton::AtomicRMWOp>(cur)) {
      Type type = atomicRmwOp.getType();
      RMWOp modifyOp = atomicRmwOp.getAtomicRmwOp();
      MemSemantic semantic = atomicRmwOp.getSem();
      MemSyncScope scope = atomicRmwOp.getScope();
      Value ptr = atomicRmwOp.getPtr();
      Value val = atomicRmwOp.getVal();
      builder.create<triton::AtomicRMWOp>(loc, type, modifyOp, ptr, val, maskOp,
                                          semantic, scope);
      atomicRmwOp->erase();
    }
  }

  // For tt.load/store ops that take a !tt.ptr<tensor>, only set boundary check
  // attrs instead of masking
  while (!tensorPtrWorkQueue.empty()) {
    auto [cur, type] = tensorPtrWorkQueue.front();
    tensorPtrWorkQueue.pop();
    IRRewriter builder(cur);

    SmallVector<int32_t> boundDims;
    for (int64_t i = 0; i < type.getRank(); i++) {
      if (!llvm::isPowerOf2_64(i)) {
        boundDims.push_back(i);
      }
    }
    DenseI32ArrayAttr boundaryCheckAttr =
        DenseI32ArrayAttr::get(mod->getContext(), boundDims);

    if (auto loadOp = dyn_cast<triton::LoadOp>(cur)) {
      loadOp.setBoundaryCheckAttr(boundaryCheckAttr);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(cur)) {
      storeOp.setBoundaryCheckAttr(boundaryCheckAttr);
    }
  }
}

// In the case that either:
//  - A tensor::InsertSliceOp has a non power of 2 dimension on the slice axis
//    - Required to ensure correctness as otherwise extra data will be written
//  - A tensor::ExtractSliceOp has a dynamic offset or the new slice size once
//  padded + offset is larger than the source tensor axis size
//    - Required to prevent out of bound errors
// Split the insert_slice/extract_slice op into power of two
// insert_slice/extract_slice op's (along the slice axis)
void splitSliceOps(FuncOp &mod) {
  std::queue<Operation *> workQueue;
  DenseMap<Operation *, SliceOpData> sliceData;

  mod.walk([&](Operation *op) {
    if (!isNonPow2TensorOperation(op)) {
      return;
    }
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      SliceOpData data(op, insertSliceOp.getSourceType().getShape(),
                       insertSliceOp.getDestType().getShape());
      // if not a no op and small axis dim size is not a power of two, and
      // offset is dynamic, override existing (non padding) data, or insert out
      // of bounds, split up into power of two slices
      if (data.axis != -1 && !llvm::isPowerOf2_64(data.smallAxisDim)) {
        if (!data.isOffsetStatic ||
            data.offsetVal + data.smallAxisDim < data.largeAxisDim ||
            data.offsetVal + llvm::PowerOf2Ceil(data.smallAxisDim) >
                llvm::PowerOf2Ceil(data.largeAxisDim)) {
          workQueue.push(op);
          sliceData[op] = data;
        }
      }
    } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      SliceOpData data(op, extractSliceOp.getResultType().getShape(),
                       extractSliceOp.getSourceType().getShape());
      // if not a no op and offset dynamic and small axis dim size not power of
      // two
      // or offset static and would result in extracting out of bounds,
      // split up into power of two slices
      if (data.axis != -1 && !llvm::isPowerOf2_64(data.smallAxisDim)) {
        if (extractSliceOp.isDynamicOffset(data.axis)) {
          workQueue.push(op);
          sliceData[op] = data;
        } else {
          int64_t offset = extractSliceOp.getStaticOffset(data.axis);
          if (offset + llvm::PowerOf2Ceil(data.smallAxisDim) >
              llvm::PowerOf2Ceil(data.largeAxisDim)) {
            workQueue.push(op);
            sliceData[op] = data;
          }
        }
      }
    }
  });

  while (!workQueue.empty()) {
    Operation *cur = workQueue.front();
    workQueue.pop();
    IRRewriter rewriter(cur);

    Location loc = cur->getLoc();

    const SliceOpData &sliceOpData = sliceData.at(cur);
    const size_t rank = sliceOpData.rank;
    const int axis = sliceOpData.axis;
    const bool isOffsetStatic = sliceOpData.isOffsetStatic;
    const int64_t offsetVal = sliceOpData.offsetVal;
    const OpFoldResult &offset = sliceOpData.offset;

    // Need to split this tensor into smaller tensors along this axis
    SmallVector<int64_t> dimSizes;
    uint64_t rowsLeft = sliceOpData.smallAxisDim;

    // Calculate the sizes of each slice
    while (rowsLeft > 0) {
      uint64_t nextSliceSize;
      if (llvm::isPowerOf2_64(rowsLeft)) {
        nextSliceSize = rowsLeft;
      } else {
        nextSliceSize = llvm::PowerOf2Ceil(rowsLeft) /
                        2; // largest power of two <= rowsLeft
      }
      dimSizes.push_back(nextSliceSize);
      rowsLeft -= nextSliceSize;
    }

    SmallVector<Value> slices;
    slices.reserve(dimSizes.size());

    int64_t curOffset = 0;

    Value source;
    SmallVector<OpFoldResult> newOffsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newSizes;
    SmallVector<OpFoldResult> newStrides(rank, rewriter.getIndexAttr(1));
    OpFoldResult initialExtractOffset;
    int64_t initialExtractOffsetVal;
    bool isInitialExtractOffsetStatic;

    OpFoldResult initialInsertOffset;
    int64_t initialInsertOffsetVal;
    bool isInitialInsertOffsetStatic;

    Value dest;

    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(cur)) {
      source = insertSliceOp.getSource();
      newSizes = insertSliceOp.getMixedSizes();
      initialExtractOffsetVal = 0;
      isInitialExtractOffsetStatic = true;

      dest = insertSliceOp.getDest();
      initialInsertOffset = offset;
      initialInsertOffsetVal = offsetVal;
      isInitialInsertOffsetStatic = isOffsetStatic;
    } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(cur)) {
      source = extractSliceOp.getSource();
      newSizes = extractSliceOp.getMixedSizes();
      initialExtractOffset = offset;
      initialExtractOffsetVal = offsetVal;
      isInitialExtractOffsetStatic = isOffsetStatic;

      RankedTensorType resultType =
          cast<RankedTensorType>(cur->getResult(0).getType());
      dest = createTensor(rewriter, loc, resultType);

      initialInsertOffsetVal = 0;
      isInitialInsertOffsetStatic = true;
    }

    // Interleave extract slice -> insert slice pattern to reduce register
    // pressure
    for (int64_t sliceSize : dimSizes) {
      newSizes[axis] = rewriter.getIndexAttr(sliceSize);

      // Extract slice
      if (isInitialExtractOffsetStatic) {
        newOffsets[axis] =
            rewriter.getIndexAttr(initialExtractOffsetVal + curOffset);
      } else {
        Value curConstOffset = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(curOffset));
        Value newOffset = rewriter.create<arith::AddIOp>(
            loc, curConstOffset, initialExtractOffset.get<Value>());
        newOffsets[axis] = newOffset;
      }
      Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
          loc, source, newOffsets, newSizes, newStrides);

      // Insert slice
      if (isInitialInsertOffsetStatic) {
        newOffsets[axis] =
            rewriter.getIndexAttr(initialInsertOffsetVal + curOffset);
      } else {
        Value curConstOffset = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(curOffset));
        Value newOffset = rewriter.create<arith::AddIOp>(
            loc, curConstOffset, initialInsertOffset.get<Value>());
        newOffsets[axis] = newOffset;
      }

      dest = rewriter.create<tensor::InsertSliceOp>(
          loc, extractedSlice, dest, newOffsets, newSizes, newStrides);

      curOffset += sliceSize;
    }

    rewriter.replaceOp(cur, dest);
  }
}

// Traverses through all ops, checks their operands and results for non power of
// two tensors For each non power of two tensor, calculates its virtual shape
DenseMap<Value, LocalTensorShapeData> populateLocalShapeData(FuncOp &mod) {
  DenseMap<Value, LocalTensorShapeData> shapeMap;

  mod.walk([&](Operation *op) {
    bool calculateAll = false;
    if (isa<tensor::InsertSliceOp, tensor::ExtractSliceOp, triton::HistogramOp>(
            op)) {
      calculateAll = true;
    }

    for (Value val : op->getOperands()) {
      if (RankedTensorType type = getNestedTensorType(val.getType())) {
        if (calculateAll || hasNonPowerTwoDim(type.getShape())) {
          shapeMap[val] = LocalTensorShapeData(type.getShape());
        }
      }
    }

    for (Value val : op->getResults()) {
      if (RankedTensorType type = getNestedTensorType(val.getType())) {
        if (calculateAll || hasNonPowerTwoDim(type.getShape())) {
          shapeMap[val] = LocalTensorShapeData(type.getShape());
        }
      }
    }
  });

  return shapeMap;
}

// Looks at one tensor input in a triton::ReduceOp/triton::ScanOp and its
// corresponding arguments in the reduce block Tries to examine its block to see
// if it has an identity Can find an identity for cases where we have just one
// op acting on the accumulator and next element (adding, multiplying, etc)
// return nullptr if an identity was not found
TypedAttr getReduceOrScanLaneIdentity(Operation *op, uint32_t laneIdx,
                                      uint32_t numLanes) {
  BlockArgument acc;
  BlockArgument next;
  Value laneRes;
  TypeSwitch<Operation *, void>(op)
      .Case<triton::ReduceOp>([&](triton::ReduceOp reduceOp) {
        Block &block = reduceOp.getCombineOp().front();
        acc = block.getArgument(laneIdx);
        next = block.getArgument(laneIdx + numLanes);

        auto returnOp = cast<triton::ReduceReturnOp>(block.getTerminator());
        laneRes = returnOp->getOperand(laneIdx);
      })
      .Case<triton::ScanOp>([&](triton::ScanOp scanOp) {
        Block &block = scanOp.getCombineOp().front();
        acc = block.getArgument(laneIdx);
        next = block.getArgument(laneIdx + numLanes);

        auto returnOp = cast<triton::ScanReturnOp>(block.getTerminator());
        laneRes = returnOp->getOperand(laneIdx);
      })
      .Default([](Operation *) {
        llvm_unreachable(
            "Non tt.reduce/tt.scan op passed to function "
            "getReduceOrScanLaneIdentity in ConvertNonPowerTwoTensors.cpp");
      });

  Operation *foldOp = laneRes.getDefiningOp();
  if ((!foldOp) || foldOp->getNumOperands() != 2) {
    return nullptr;
  }

  bool withinLane =
      (foldOp->getOperand(0) == acc && foldOp->getOperand(1) == next) ||
      (foldOp->getOperand(1) == acc && foldOp->getOperand(0) == next);
  if ((!withinLane) || (!isa<BlockArgument>(foldOp->getOperand(0))) ||
      (!isa<BlockArgument>(foldOp->getOperand(1)))) {
    // using other vars, or we didnt return `arg0 [op] arg1`, abort!
    return nullptr;
  }

  std::optional<TypedAttr> potentialIdentity = arith::getNeutralElement(foldOp);
  if (potentialIdentity) {
    TypedAttr identity = *potentialIdentity;
    if (auto floatIdentity = dyn_cast<FloatAttr>(identity)) {
      if (floatIdentity.getValue().isNaN()) {
        // Replace NAN identities with positive/negative INF to prevent NAN
        // propagation
        OpBuilder builder(op);
        FloatType elementType = cast<FloatType>(floatIdentity.getType());
        if (floatIdentity.getValue().isNegative()) {
          return builder.getFloatAttr(
              elementType, APFloat::getInf(elementType.getFloatSemantics(),
                                           /*Negative=*/true));
        } else {
          return builder.getFloatAttr(
              elementType, APFloat::getInf(elementType.getFloatSemantics(),
                                           /*Negative=*/false));
        }
      }
    }

    return identity;
  }
  return nullptr;
}

// Returns a SmallVector<TypedAttr> containing the identity for the algorithm
// associated with that tensor First tensor's identity is the first entry, etc.
// In the return result, nullptr entries indicate no identity found
SmallVector<TypedAttr> getReduceOrScanOpIdentities(Operation *op) {
  unsigned int numOperands = op->getNumOperands();

  SmallVector<TypedAttr> cur;
  cur.reserve(numOperands);

  for (unsigned int i = 0; i < numOperands; i++) {
    cur.push_back(getReduceOrScanLaneIdentity(op, i, op->getNumOperands()));
  }

  return cur;
}

void expandToSize(SmallVector<TypedAttr> &vec, size_t size) {
  while (vec.size() < size) {
    vec.push_back(nullptr);
  }
}

// Returns a SmallVector of TypedAttr padding requirements, one for each operand
// (non tensor operands will have a nullptr TypedAttr) Operations that do not
// require any padding will just return an empty SmallVector
SmallVector<TypedAttr>
getPaddingRequirements(OpBuilder &builder, Operation *op,
                       const DenseMap<Value, LocalTensorShapeData> &shapeData) {
  SmallVector<TypedAttr> res(op->getNumOperands(), nullptr);
  TypeSwitch<Operation *>(op)
      .Case<triton::ReduceOp>([&](triton::ReduceOp reduceOp) {
        uint32_t reduceAxis = reduceOp.getAxis();
        if (llvm::isPowerOf2_64(shapeData.find(reduceOp->getOperand(0))
                                    ->getSecond()
                                    .dataShape[reduceAxis])) {
          return;
        }
        SmallVector<TypedAttr> paddingAttrs =
            getReduceOrScanOpIdentities(reduceOp);

        if (paddingAttrs.size() != 0) {
          for (size_t i = 0; i < paddingAttrs.size(); i++) {
            if (paddingAttrs[i]) {
              res[i] = paddingAttrs[i];
            }
          }
        }
      })
      .Case<triton::ScanOp>([&](triton::ScanOp scanOp) {
        if (!scanOp.getReverse()) {
          // No padding needed on forward scan op as padding is in the backmost
          // positions
          // TODO - If padding locations are changed, this needs to be removed
          res.clear();
          return;
        }
        uint32_t reduceAxis = scanOp.getAxis();
        if (llvm::isPowerOf2_64(shapeData.find(scanOp->getOperand(0))
                                    ->getSecond()
                                    .dataShape[reduceAxis])) {
          return;
        }
        SmallVector<TypedAttr> paddingAttrs =
            getReduceOrScanOpIdentities(scanOp);

        if (paddingAttrs.size() != 0) {
          for (size_t i = 0; i < paddingAttrs.size(); i++) {
            if (paddingAttrs[i]) {
              res[i] = paddingAttrs[i];
            }
          }
        }
      })
      .Case<triton::LoadOp>([&](triton::LoadOp loadOp) {
        // tt.load mask tensor should be false padded
        expandToSize(res, 3);
        res[1] = builder.getZeroAttr(builder.getI1Type());
      })
      .Case<triton::StoreOp>([&](triton::StoreOp storeOp) {
        // tt.store mask tensor should be false padded
        expandToSize(res, 3);
        res[2] = builder.getZeroAttr(builder.getI1Type());
      })
      .Case<triton::AtomicRMWOp>([&](triton::AtomicRMWOp storeOp) {
        // tt.store mask tensor should be false padded
        expandToSize(res, 3);
        res[2] = builder.getZeroAttr(builder.getI1Type());
      })
      .Case<triton::DotOp>([&](triton::DotOp dotOp) {
        // tt.dot should be zero padded
        TypedAttr paddingAttr = builder.getZeroAttr(
            cast<RankedTensorType>(dotOp.getOperand(0).getType())
                .getElementType());
        for (size_t i = 0; i < res.size() - 1; i++) {
          res[i] = paddingAttr;
        }
      })
      .Default([&](Operation *op) {
        // No padding needed, empty vector
        res.clear();
      });

  return res;
}

// Assumes builder's insertion point is already set
// Given the data shape and virtual shape of an op's result, creates a mask
// which is true where the original elements are in the padded shape Is used
// with arith::SelectOp to add/swap padding
Value createPaddingMask(OpBuilder &builder, Location loc,
                        ArrayRef<int64_t> dataShape,
                        ArrayRef<int64_t> paddedShape) {
  Type i32Type = builder.getI32Type();
  Type i1Type = builder.getI1Type();
  RankedTensorType largeType = RankedTensorType::get(paddedShape, i1Type);

  size_t rank = dataShape.size();
  SmallVector<Value> dimMasks;
  dimMasks.reserve(rank);

  for (size_t i = 0; i < rank; i++) {
    SmallVector<int64_t> curShape = {paddedShape[i]};
    RankedTensorType curType = RankedTensorType::get(curShape, i32Type);
    Value cur =
        builder.create<triton::MakeRangeOp>(loc, curType, 0, paddedShape[i]);
    Value constant = builder.create<arith::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(dataShape[i]));
    Value splatted = builder.create<triton::SplatOp>(loc, curType, constant);

    cur = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, cur,
                                        splatted);

    // We now expand out tensor to be tensor<1x1x...xNx1x1x...x1xT>
    bool insertAtEnd = false;
    for (size_t j = 0; j < rank; j++) {
      if (j == i) {
        insertAtEnd = true;
        continue;
      }

      if (insertAtEnd) {
        curShape.push_back(1);
        curType = RankedTensorType::get(curShape, i1Type);
        cur = builder.create<triton::ExpandDimsOp>(loc, curType, cur,
                                                   curShape.size() - 1);
      } else {
        curShape.insert(curShape.begin(), 1);
        curType = RankedTensorType::get(curShape, i1Type);
        cur = builder.create<triton::ExpandDimsOp>(loc, curType, cur, 0);
      }
    }

    // Just neeed to broadcast now
    cur = builder.create<triton::BroadcastOp>(loc, largeType, cur);
    dimMasks.push_back(cur);
  }

  Value res = dimMasks[0];
  for (size_t i = 1; i < rank; i++) {
    res = builder.create<arith::AndIOp>(loc, res, dimMasks[i]);
  }
  return res;
}

// Assumes sets builder's insertion point before creating mask
// Given the data shape and virtual shape of an op's result, creates a mask
// which is true where the original elements are in the padded shape Is used
// with arith::SelectOp to add/swap padding
Value createPaddingMask(OpBuilder &builder, Operation *op,
                        ArrayRef<int64_t> dataShape,
                        ArrayRef<int64_t> paddedShape,
                        bool setInsertionPointAfter = false) {
  if (setInsertionPointAfter) {
    builder.setInsertionPointAfter(op);
  } else {
    builder.setInsertionPoint(op);
  }
  Location loc = op->getLoc();

  return createPaddingMask(builder, loc, dataShape, paddedShape);
}

// Updates the return shape and iter_args of a scf ForOp
void updateForOp(IRRewriter &rewriter, scf::ForOp op,
                 const DenseMap<Value, LocalTensorShapeData> &shapeMap) {
  ValueRange initialVals = op.getInitArgs();
  rewriter.modifyOpInPlace(op, [&]() {
    Block &block = op.getRegion().front();
    for (size_t i = 1; i < block.getNumArguments(); i++) {
      block.getArgument(i).setType(initialVals[i - 1].getType());
    }

    auto yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());

    for (size_t i = 0; i < op.getNumResults(); i++) {
      Value res = op.getResult(i);
      if (RankedTensorType type =
              getNonPowerTwoNestedTensorType(res.getType())) {
        Type elementType = type.getElementType();
        Value yieldSrc = yieldOp.getOperand(i);
        ArrayRef<int64_t> newShape = shapeMap.at(yieldSrc).localVirtualShape;

        RankedTensorType newTensorType =
            RankedTensorType::get(newShape, elementType);

        if (auto ptrType = dyn_cast<triton::PointerType>(res.getType())) {
          triton::PointerType newPointerType = triton::PointerType::get(
              newTensorType, ptrType.getAddressSpace());
          res.setType(newPointerType);
        } else {
          res.setType(newTensorType);
        }
      }
    }
  });
  return;
}

// Updates the return shape of a scf IfOp
void updateIfOp(IRRewriter &rewriter, scf::IfOp op,
                const DenseMap<Value, LocalTensorShapeData> &shapeMap) {
  rewriter.modifyOpInPlace(op, [&]() {
    auto yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());

    for (size_t i = 0; i < op.getNumResults(); i++) {
      Value res = op.getResult(i);
      if (RankedTensorType type = getNonPowerTwoTensorType(res.getType())) {
        Type elementType = type.getElementType();
        Value yieldSrc = yieldOp.getOperand(i);
        ArrayRef<int64_t> newShape = shapeMap.at(yieldSrc).localVirtualShape;

        RankedTensorType newTensorType =
            RankedTensorType::get(newShape, elementType);

        if (auto ptrType = dyn_cast<triton::PointerType>(res.getType())) {
          triton::PointerType newPointerType = triton::PointerType::get(
              newTensorType, ptrType.getAddressSpace());
          res.setType(newPointerType);
        } else {
          res.setType(newTensorType);
        }
      }
    }
  });
  return;
}

// Updates the return shape and modifies sizes data of a tensor::ExtractSliceOp
void updateExtractSliceOp(IRRewriter &rewriter, tensor::ExtractSliceOp op,
                          const SliceOpData &sliceData,
                          const LocalTensorShapeData &result) {
  Type elementType = op.getResultType().getElementType();
  RankedTensorType resultType =
      RankedTensorType::get(result.localVirtualShape, elementType);
  rewriter.modifyOpInPlace(op, [&]() {
    op.setStaticSizes(result.localVirtualShape);
    op.getResult().setType(resultType);
  });
}

// Updates the return shape and modifies sizes data of a tensor::InsertSliceOp
void updateInsertSliceOp(IRRewriter &rewriter, tensor::InsertSliceOp op,
                         const SliceOpData &sliceData,
                         const LocalTensorShapeData &source,
                         const LocalTensorShapeData &result) {
  Type elementType = op.getDestType().getElementType();
  RankedTensorType destType =
      RankedTensorType::get(result.localVirtualShape, elementType);
  rewriter.modifyOpInPlace(op, [&]() {
    op.setStaticSizes(source.localVirtualShape);
    op.getResult().setType(destType);
  });
}

// Updates the return type of a triton::AdvanceOp to be the same as the input
// tensor ptr
void updateAdvanceOp(IRRewriter &rewriter, triton::AdvanceOp op) {
  rewriter.modifyOpInPlace(
      op, [&]() { op.getResult().setType(op.getPtr().getType()); });
}

// Updates the return shape of a triton BroadcastOp
void updateBroadcastOp(IRRewriter &rewriter, triton::BroadcastOp op,
                       const LocalTensorShapeData &data) {
  Type elementType =
      cast<RankedTensorType>(op.getResult().getType()).getElementType();
  RankedTensorType newType =
      RankedTensorType::get(data.localVirtualShape, elementType);

  rewriter.modifyOpInPlace(op, [&]() { op.getResult().setType(newType); });
}

// Updates the return shape of a triton MakeTensorPtrOp
void updateMakeTensorPtrOp(IRRewriter &rewriter, triton::MakeTensorPtrOp op,
                           const LocalTensorShapeData &data) {
  triton::PointerType oldPtrType = cast<triton::PointerType>(op.getType());
  RankedTensorType oldPtrTensorType =
      cast<RankedTensorType>(oldPtrType.getPointeeType());
  Type elementType = oldPtrTensorType.getElementType();

  RankedTensorType newPtrTensorType =
      RankedTensorType::get(data.localVirtualShape, elementType);
  triton::PointerType newPtrType =
      triton::PointerType::get(newPtrTensorType, oldPtrType.getAddressSpace());

  rewriter.modifyOpInPlace(op, [&]() { op.getResult().setType(newPtrType); });
}

// Updates the return shape of an InferTypeOpInterface op
void updateInferTypeOp(IRRewriter &rewriter, InferTypeOpInterface op) {
  SmallVector<Type> inferredRetTypes;
  inferredRetTypes.reserve(op->getNumResults());

  if (succeeded(op.inferReturnTypes(op->getContext(), op->getLoc(),
                                    op->getOperands(), op->getAttrDictionary(),
                                    op->getPropertiesStorage(),
                                    op->getRegions(), inferredRetTypes))) {

    rewriter.modifyOpInPlace(op, [&]() {
      for (size_t i = 0; i < op->getNumResults(); i++) {
        op->getResult(i).setType(inferredRetTypes[i]);
      }
    });
  }
}

// Updates the return shapes of elementwise ops
void updateElementwiseOp(IRRewriter &rewriter, Operation *op) {
  ArrayRef<int64_t> resShape;
  for (Value operand : op->getOperands()) {
    if (auto operandType = dyn_cast<RankedTensorType>(operand.getType())) {
      resShape = operandType.getShape();
      break;
    }
  }

  rewriter.modifyOpInPlace(op, [&]() {
    for (Value result : op->getResults()) {
      ShapedType prevType = cast<ShapedType>(result.getType());
      ShapedType newType =
          prevType.cloneWith(resShape, prevType.getElementType());
      result.setType(newType);
    }
  });
}

// Used for ops whose only change is updating the return type
// Mostly ops that implement InferTypeOpInterface, but also some manually added
// ops function isGenericTensorOp is used to determine what is a 'general tensor
// op'
void updateGeneralTensorOp(
    IRRewriter &rewriter, Operation *op,
    const DenseMap<Value, LocalTensorShapeData> &shapeMap) {
  TypeSwitch<Operation *, void>(op)
      .Case<scf::ForOp>(
          [&](scf::ForOp forOp) { updateForOp(rewriter, forOp, shapeMap); })
      .Case<scf::IfOp>(
          [&](scf::IfOp op) { updateIfOp(rewriter, op, shapeMap); })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp extractSliceOp) {
        const LocalTensorShapeData &source =
            shapeMap.at(extractSliceOp.getSource());
        const LocalTensorShapeData &result =
            shapeMap.at(extractSliceOp.getResult());
        SliceOpData sliceData(op, result.dataShape, source.dataShape);
        updateExtractSliceOp(rewriter, extractSliceOp, sliceData, result);
      })
      .Case<tensor::InsertSliceOp>([&](tensor::InsertSliceOp insertSliceOp) {
        const LocalTensorShapeData &source =
            shapeMap.at(insertSliceOp.getSource());
        const LocalTensorShapeData &result =
            shapeMap.at(insertSliceOp.getResult());
        SliceOpData sliceData(op, source.dataShape, result.dataShape);
        updateInsertSliceOp(rewriter, insertSliceOp, sliceData, source, result);
      })
      .Case<triton::AdvanceOp>([&](triton::AdvanceOp advanceOp) {
        updateAdvanceOp(rewriter, advanceOp);
      })
      .Case<triton::AtomicRMWOp>([&](triton::AtomicRMWOp atomicRMWOp) {
        updateElementwiseOp(rewriter, op);
      })
      .Case<triton::BroadcastOp>([&](triton::BroadcastOp broadcastOp) {
        const LocalTensorShapeData &data = shapeMap.at(broadcastOp.getResult());
        updateBroadcastOp(rewriter, broadcastOp, data);
      })
      .Case<triton::MakeTensorPtrOp>(
          [&](triton::MakeTensorPtrOp makeTensorPtrOp) {
            const LocalTensorShapeData &data =
                shapeMap.at(makeTensorPtrOp.getResult());
            updateMakeTensorPtrOp(rewriter, makeTensorPtrOp, data);
          })
      .Case<InferTypeOpInterface>([&](InferTypeOpInterface inferOp) {
        updateInferTypeOp(rewriter, inferOp);
      })
      .Default([&](Operation *) {
        if (op->hasTrait<OpTrait::Elementwise>()) {
          updateElementwiseOp(rewriter, op);
        }
      });
}

// Returns true if the reduce op reduction algorithms all have identities
bool reduceOrScanOpNeedsMask(Operation *op) {
  SmallVector<TypedAttr> identities = getReduceOrScanOpIdentities(op);
  for (TypedAttr attr : identities) {
    if (!attr) {
      return true;
    }
  }

  return false;
}

// Use only when it is known that at least one reduce op tensor calculation does
// not have a simple identity If all tensor operand reduction algorithms have a
// simple identity use updateGeneralTensorOp For each tensor operand whose
// reduction algorithm does not have a simple identity, uses a mask to ignore
// those operands
triton::ReduceOp getReplacementReduceOp(IRRewriter &rewriter,
                                        triton::ReduceOp reduceOp, Value mask) {
  SmallVector<TypedAttr> identities = getReduceOrScanOpIdentities(reduceOp);
  // Need to add a mask as another argument to ignore padding values

  rewriter.setInsertionPoint(reduceOp);
  Location loc = reduceOp.getLoc();

  size_t axis = reduceOp.getAxis();
  SmallVector<Value> newOperands(reduceOp->getOperands().begin(),
                                 reduceOp->getOperands().end());
  newOperands.push_back(mask);

  auto newOp = rewriter.create<triton::ReduceOp>(loc, newOperands, axis);

  {
    Block *block = rewriter.createBlock(&(newOp.getCombineOp()));

    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    unsigned numOriginalOperands = reduceOp.getNumOperands();

    // Setting up reduce block arguments
    SmallVector<Type> argElementTypes;
    argElementTypes.reserve(numOriginalOperands + 1);

    for (Value operand : reduceOp.getOperands()) {
      argElementTypes.push_back(
          cast<RankedTensorType>(operand.getType()).getElementType());
    }

    // Element type of mask
    argElementTypes.push_back(rewriter.getI1Type());

    SmallVector<Location> locations;
    locations.assign(argElementTypes.size(), loc);

    // add args twice for both the current value and accumulator args
    block->addArguments(argElementTypes, locations);
    block->addArguments(argElementTypes, locations);

    // map block arguments in the old reduceOp to the new block arguments
    IRMapping mapping;
    Block &oldBlock = reduceOp.getRegion().front();
    for (unsigned i = 0; i < numOriginalOperands; i++) {
      mapping.map(oldBlock.getArgument(i), block->getArgument(i));
    }

    for (unsigned i = numOriginalOperands; i < 2 * numOriginalOperands; i++) {
      mapping.map(
          oldBlock.getArgument(i),
          block->getArgument(i + 1)); // add one to skip the mask operand
    }

    // copy ops using mapping
    for (auto &op : oldBlock.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Use mask now
    Value curMaskVal = block->getArgument(numOriginalOperands);
    Value curMaskAcc = block->getArgument(2 * numOriginalOperands + 1);

    auto oldReturn = cast<triton::ReduceReturnOp>(oldBlock.getTerminator());
    SmallVector<Value> outputs;
    outputs.reserve(numOriginalOperands + 1);
    for (unsigned i = 0; i < numOriginalOperands; i++) {
      TypedAttr identity = identities[i];
      Value laneCombinedRes = mapping.lookup(oldReturn->getOperand(i));

      if (identity) {
        // This operand has an identity, no need to mask
        outputs.push_back(laneCombinedRes);
        continue;
      }
      Value curLaneVal = block->getArgument(i);
      Value curLaneAcc = block->getArgument(numOriginalOperands + 1 + i);

      // This is the value to return if the mask is true at this spot.
      // Using a select op here in case we have only seen padding so far (which
      // means that laneCombinedRes is the result of f(padding, val))
      Value accVal = rewriter.create<arith::SelectOp>(
          loc, curMaskAcc, laneCombinedRes, curLaneVal);

      // Final value
      Value chosenVal =
          rewriter.create<arith::SelectOp>(loc, curMaskVal, accVal, curLaneAcc);
      outputs.push_back(chosenVal);
    }

    // accumulating with or (accumulated mask is true if we have seen a
    // non-padding element so far)
    Value chosenMask =
        rewriter.create<arith::OrIOp>(loc, curMaskVal, curMaskAcc);
    outputs.push_back(chosenMask);

    rewriter.create<triton::ReduceReturnOp>(loc, outputs);
  }

  // triton::ReduceOp implements the InferTypeOpInterface (we use it to update
  // the result type(s))
  updateInferTypeOp(rewriter, newOp);

  return newOp;
}

// Use only when the scan op has reverse=true and at least one scan op tensor
// calculation does not have a simple identity If all tensor operand reduction
// algorithms have a simple identity use updateGeneralTensorOp For each tensor
// operand whose reduction algorithm does not have a simple identity, uses a
// mask to ignore those operands
triton::ScanOp getReplacementScanOp(IRRewriter &rewriter, triton::ScanOp scanOp,
                                    Value mask) {
  SmallVector<TypedAttr> identities = getReduceOrScanOpIdentities(scanOp);
  // Need to add a mask as another argument to ignore padding values

  rewriter.setInsertionPoint(scanOp);
  Location loc = scanOp.getLoc();

  SmallVector<Value> newOperands(scanOp->getOperands().begin(),
                                 scanOp->getOperands().end());
  newOperands.push_back(mask);

  auto newOp = rewriter.create<triton::ScanOp>(
      loc, newOperands, scanOp.getAxis(), scanOp.getReverse());

  {
    Block *block = rewriter.createBlock(&(newOp.getCombineOp()));

    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    unsigned numOriginalOperands = scanOp.getNumOperands();

    // Setting up reduce block arguments
    SmallVector<Type> argElementTypes;
    argElementTypes.reserve(numOriginalOperands + 1);

    for (Value operand : scanOp.getOperands()) {
      argElementTypes.push_back(
          cast<RankedTensorType>(operand.getType()).getElementType());
    }

    // Element type of mask
    argElementTypes.push_back(rewriter.getI1Type());

    SmallVector<Location> locations;
    locations.assign(argElementTypes.size(), loc);

    // add args twice for both the current value and accumulator args
    block->addArguments(argElementTypes, locations);
    block->addArguments(argElementTypes, locations);

    // map block arguments in the old scanOp to the new block arguments
    IRMapping mapping;
    Block &oldBlock = scanOp.getRegion().front();
    for (unsigned i = 0; i < numOriginalOperands; i++) {
      mapping.map(oldBlock.getArgument(i), block->getArgument(i));
    }

    for (unsigned i = numOriginalOperands; i < 2 * numOriginalOperands; i++) {
      mapping.map(
          oldBlock.getArgument(i),
          block->getArgument(i + 1)); // add one to skip the mask operand
    }

    // copy ops using mapping
    for (auto &op : oldBlock.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Use mask now
    Value curMaskVal = block->getArgument(numOriginalOperands);
    Value curMaskAcc = block->getArgument(2 * numOriginalOperands + 1);

    auto oldReturn = cast<triton::ScanReturnOp>(oldBlock.getTerminator());
    SmallVector<Value> outputs;
    outputs.reserve(numOriginalOperands + 1);
    for (unsigned i = 0; i < numOriginalOperands; i++) {
      TypedAttr identity = identities[i];
      Value laneCombinedRes = mapping.lookup(oldReturn->getOperand(i));

      if (identity) {
        // This operand has an identity, no need to mask
        outputs.push_back(laneCombinedRes);
        continue;
      }
      Value curLaneVal = block->getArgument(i);
      Value curLaneAcc = block->getArgument(numOriginalOperands + 1 + i);

      // This is the value to return if the mask is true at this spot.
      // Using a select op here in case we have only seen padding so far (which
      // means that laneCombinedRes is the result of f(padding, val))
      Value accVal = rewriter.create<arith::SelectOp>(
          loc, curMaskAcc, laneCombinedRes, curLaneVal);

      // Final value
      Value chosenVal =
          rewriter.create<arith::SelectOp>(loc, curMaskVal, accVal, curLaneAcc);
      outputs.push_back(chosenVal);
    }

    // accumulating with or (accumulated mask is true if we have seen a
    // non-padding element so far)
    Value chosenMask =
        rewriter.create<arith::OrIOp>(loc, curMaskVal, curMaskAcc);
    outputs.push_back(chosenMask);

    rewriter.create<triton::ScanReturnOp>(loc, outputs);
  }

  // triton::ScanOp implements the InferTypeOpInterface (we use it to update the
  // result type(s))
  updateInferTypeOp(rewriter, newOp);

  return newOp;
}

// Creates a replacement tensor creation op (arith::ConstantOp, triton::SplatOp,
// triton::MakeRangeOp) without any specific padding
Value getReplacementCreationOp(OpBuilder &builder, Operation *tensorCreatorOp,
                               const LocalTensorShapeData &data) {
  Value res;
  Location loc = tensorCreatorOp->getLoc();

  builder.setInsertionPointAfter(tensorCreatorOp);
  TypeSwitch<Operation *, void>(tensorCreatorOp)
      .Case<triton::SplatOp>([&](triton::SplatOp splatOp) {
        Type elementType = splatOp.getType().getElementType();
        res = builder.create<triton::SplatOp>(
            loc, RankedTensorType::get(data.localVirtualShape, elementType),
            splatOp.getSrc());
      })
      .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
        RankedTensorType dataType = cast<RankedTensorType>(constOp.getType());
        Type elementType = dataType.getElementType();
        auto origAttr = cast<DenseElementsAttr>(constOp.getValue());
        RankedTensorType virtualShapeType =
            RankedTensorType::get(data.localVirtualShape, elementType);
        // The constant op must be a splat constant op (otherwise the tensor
        // would not have the same number of elements as number of threads per
        // warp which is a strict requirement)
        Attribute attr = origAttr.getSplatValue<Attribute>();
        res = createTensor(builder, loc, virtualShapeType, attr);
      })
      .Case<triton::MakeRangeOp>([&](triton::MakeRangeOp makeRangeOp) {
        Type elementType = makeRangeOp.getType().getElementType();
        int start = makeRangeOp.getStart();
        res = builder.create<triton::MakeRangeOp>(
            loc, RankedTensorType::get(data.localVirtualShape, elementType),
            start, start + data.localVirtualShapeSize);
      });

  return res;
}

// Updates a simple triton ReshapeOp, which is a reshape op that just
// adds/removes dimensions of size 1
void updateSimpleReshapeOp(IRRewriter &rewriter, triton::ReshapeOp reshapeOp,
                           const LocalTensorShapeData &resultShape) {
  Type elementType = reshapeOp.getResult().getType().getElementType();
  RankedTensorType newType =
      RankedTensorType::get(resultShape.localVirtualShape, elementType);
  rewriter.modifyOpInPlace(reshapeOp,
                           [&]() { reshapeOp.getResult().setType(newType); });
}

// Updates a complex triton ReshapeOp, which is a reshape op that does not just
// add/remove dimensions of size 1 Flattens the operand tensor, then uses
// tensor.extract/insert slice to reposition the data elements Then reshapes
// this tensor to the final shape Very inefficient, so its best to avoid using
// this if possible
Value getReplacementComplexReshapeOp(OpBuilder &builder,
                                     triton::ReshapeOp reshapeOp,
                                     const LocalTensorShapeData &resultData,
                                     const std::vector<bool> &srcValueLocs,
                                     const std::vector<bool> &resValueLocs) {
  Location loc = reshapeOp->getLoc();

  RankedTensorType srcType = reshapeOp.getSrc().getType();
  int64_t srcNumEls = static_cast<int64_t>(srcValueLocs.size());
  int64_t resNumEls = static_cast<int64_t>(resValueLocs.size());
  Type elementType = srcType.getElementType();
  builder.setInsertionPoint(reshapeOp);

  // Reshape source tensor to 1D
  RankedTensorType flattenedSrcType =
      RankedTensorType::get({srcNumEls}, elementType);
  Value flattenedSrc;
  if (srcType.getRank() == 1) {
    flattenedSrc = reshapeOp.getSrc();
  } else {
    flattenedSrc = builder.create<triton::ReshapeOp>(loc, flattenedSrcType,
                                                     reshapeOp.getSrc());
  }

  // Create 1D destination tensor
  RankedTensorType flattenedResType =
      RankedTensorType::get({resNumEls}, elementType);
  Value dest = createTensor(builder, loc, flattenedResType);

  // Extract and insert slices
  int64_t srcIdx = 0;
  int64_t dstIdx = 0;

  SmallVector<OpFoldResult> strides(1, builder.getIndexAttr(1));
  while (srcIdx < srcNumEls && dstIdx < resNumEls) {
    if (!srcValueLocs[srcIdx]) {
      srcIdx += 1;
    }
    if (!resValueLocs[dstIdx]) {
      dstIdx += 1;
    }
    int64_t srcStart = srcIdx;
    int64_t dstStart = dstIdx;

    while (srcIdx < srcNumEls && dstIdx < resNumEls && srcValueLocs[srcIdx] &&
           resValueLocs[dstIdx]) {
      srcIdx += 1;
      dstIdx += 1;
    }

    int64_t sliceSize = srcIdx - srcStart;
    if (sliceSize > 0) {
      int64_t paddedSliceSize =
          static_cast<int64_t>(llvm::PowerOf2Ceil(sliceSize));
      if (dstStart + paddedSliceSize <= resNumEls &&
          srcStart + paddedSliceSize <= srcNumEls) {
        // If we have enough space, just extract and insert a large slice
        // extra elements will be overriden anyways later
        SmallVector<OpFoldResult> offsets(1, builder.getIndexAttr(srcStart));
        SmallVector<OpFoldResult> sizes(1,
                                        builder.getIndexAttr(paddedSliceSize));

        Value extractedSlice = builder.create<tensor::ExtractSliceOp>(
            loc, flattenedSrc, offsets, sizes, strides);
        offsets[0] = builder.getIndexAttr(dstStart);
        dest = builder.create<tensor::InsertSliceOp>(loc, extractedSlice, dest,
                                                     offsets, sizes, strides);
      } else {
        // Not enough space to insert a large slice
        // extract + insert small power of two slices
        SmallVector<Value> extractedSlices;
        int64_t srcOffset = srcStart;
        int64_t dstOffset = dstStart;

        while (sliceSize > 0) {
          int64_t chosenSlice =
              static_cast<int64_t>(llvm::PowerOf2Ceil(sliceSize));
          if (chosenSlice > sliceSize) {
            chosenSlice /= 2;
          }
          sliceSize -= chosenSlice;

          SmallVector<OpFoldResult> offsets(1, builder.getIndexAttr(srcOffset));
          SmallVector<OpFoldResult> sizes(1, builder.getIndexAttr(chosenSlice));
          Value slice = builder.create<tensor::ExtractSliceOp>(
              loc, flattenedSrc, offsets, sizes, strides);
          offsets[0] = builder.getIndexAttr(dstOffset);
          dest = builder.create<tensor::InsertSliceOp>(loc, slice, dest,
                                                       offsets, sizes, strides);

          srcOffset += chosenSlice;
          dstOffset += chosenSlice;
        }
      }
    }
  }
  RankedTensorType resType =
      RankedTensorType::get(resultData.localVirtualShape, elementType);
  Value res = builder.create<triton::ReshapeOp>(loc, resType, dest);

  return res;
}

Value getReplacementHistogramOp(OpBuilder &builder, triton::HistogramOp op,
                                const LocalTensorShapeData &operandData,
                                const LocalTensorShapeData &resultData) {
  Value source = op.getSrc();
  builder.setInsertionPoint(op);
  Location loc = op->getLoc();

  Type elementType = op.getResult().getType().getElementType();
  RankedTensorType resType =
      RankedTensorType::get(resultData.localVirtualShape, elementType);
  Value res;

  if (hasNonPowerTwoDim(operandData.dataShape)) {
    Value mask = op.getMask();
    if (mask) {
      Value paddingMask = createPaddingMask(builder, op, operandData.dataShape,
                                            operandData.localVirtualShape);
      mask = builder.create<arith::AndIOp>(loc, mask, paddingMask);
    } else {
      mask = createPaddingMask(builder, op, operandData.dataShape,
                               operandData.localVirtualShape);
    }
    res = builder.create<triton::HistogramOp>(loc, resType, source, mask);
  } else {
    res = builder.create<triton::HistogramOp>(loc, resType, source);
  }

  return res;
}

struct PaddingLattice : public dataflow::Lattice<AllPaddingRequirements> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PaddingLattice)
  using Lattice::Lattice;
};

// Pushes padding requirements upstream
class BackwardPaddingPopulationAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<PaddingLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      BackwardPaddingPopulationAnalysis)
  BackwardPaddingPopulationAnalysis(
      DataFlowSolver &solver, SymbolTableCollection &symbolTable,
      const DenseMap<Value, LocalTensorShapeData> &shapeData)
      : SparseBackwardDataFlowAnalysis(solver, symbolTable),
        shapeData(shapeData) {}

  DenseMap<Value, PotentialPaddingRequirements> paddingMap;
  const DenseMap<Value, LocalTensorShapeData> &shapeData;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<PaddingLattice *> operands,
                 ArrayRef<const PaddingLattice *> results) override {
    OpBuilder builder(op);

    if (!hasTensorArgs(op)) {
      return success();
    }

    // Power of two tensor/non tensor ops do not need padding, we cannot add
    // padding to a tensor pointer
    if ((!isNonPow2TensorOperation(op)) ||
        isTensorPointerLoadStoreOperation(op)) {
      return success();
    }

    // In general, tries to propagate padding requirements up by pushing the
    // requirements of its results up to tensor operands that share the same
    // element type For some ops though, want to push it to certain operands
    // while ignoring other operands For example for arith::SelectOp, when its
    // operands are boolean tensors, the padding reqs of its result could be
    // applied to the mask operand which is not desired
    TypeSwitch<Operation *>(op)
        .Case<arith::SelectOp>([&](arith::SelectOp selectOp) {
          Type elementType =
              cast<RankedTensorType>(selectOp.getType()).getElementType();
          AllPaddingRequirements &trueTensorLattice = operands[1]->getValue();
          Value trueTensorValue = operands[1]->getAnchor();
          OpOperand *trueTensorOperand = &selectOp->getOpOperand(1);
          AllPaddingRequirements &falseTensorLattice = operands[2]->getValue();
          Value falseTensorValue = operands[2]->getAnchor();
          OpOperand *falseTensorOperand = &selectOp->getOpOperand(2);

          ChangeResult trueLatticeChanged = ChangeResult::NoChange;
          ChangeResult falseLatticeChanged = ChangeResult::NoChange;

          for (const TypedAttr &req : results[0]->getValue().reqs) {
            if (req.getType() == elementType) {
              trueLatticeChanged |= trueTensorLattice.add(req);
              paddingMap[trueTensorValue].addLoose(req, trueTensorOperand);
              falseLatticeChanged |= falseTensorLattice.add(req);
              paddingMap[falseTensorValue].addLoose(req, falseTensorOperand);
            }
          }

          propagateIfChanged(operands[1], trueLatticeChanged);
          propagateIfChanged(operands[2], falseLatticeChanged);
        })
        .Default([&](Operation *op) {
          for (size_t i = 0; i < operands.size(); i++) {
            PaddingLattice *operandLattice = operands[i];
            Value val = operandLattice->getAnchor();
            AllPaddingRequirements &operandReqs = operandLattice->getValue();
            ChangeResult changed = ChangeResult::NoChange;

            for (const PaddingLattice *resLattice : results) {
              if (RankedTensorType type = dyn_cast<RankedTensorType>(
                      operandLattice->getAnchor().getType())) {
                for (const TypedAttr &req : resLattice->getValue().reqs) {
                  if (req.getType() == type.getElementType()) {
                    paddingMap[val].addLoose(req, &op->getOpOperand(i));
                    changed |= operandReqs.add(req);
                  }
                }
              }
            }

            propagateIfChanged(operandLattice, changed);
          }
        });
    SmallVector<TypedAttr> opReqs =
        getPaddingRequirements(builder, op, shapeData);

    if (opReqs.size() == 0) {
      return success();
    }

    // Add this op's required padding
    for (size_t i = 0; i < opReqs.size(); i++) {
      if (opReqs[i]) {
        Value val = op->getOperand(i);
        paddingMap[val].addStrict(opReqs[i], &op->getOpOperand(i));

        ChangeResult changed = operands[i]->getValue().add(opReqs[i]);
        propagateIfChanged(operands[i], changed);
      }
    }

    // Update padding map
    for (const auto &count : paddingMap) {
      paddingMap[count.getFirst()] = count.getSecond();
    }

    return success();
  }

  void visitBranchOperand(OpOperand &operand) override {
    auto branchOp = dyn_cast<BranchOpInterface>(operand.getOwner());
    if (!branchOp) {
      return;
    }

    std::optional<BlockArgument> successorBlockArg =
        branchOp.getSuccessorBlockArgument(operand.getOperandNumber());
    if (!successorBlockArg) {
      return;
    }

    BlockArgument blockArg = *successorBlockArg;
    const PaddingLattice *blockLattice = getLatticeElement(blockArg);
    if ((!blockLattice) || (!blockLattice->getValue().initialized)) {
      return;
    }

    PaddingLattice *srcLattice = getLatticeElement(operand.get());

    ChangeResult changed =
        srcLattice->getValue().meet(blockLattice->getValue());

    propagateIfChanged(srcLattice, changed);
  }

  void visitCallOperand(OpOperand &operand) override {
    auto callOp = dyn_cast<CallOpInterface>(operand.getOwner());
    if (!callOp) {
      return;
    }
    Operation *callable = callOp.resolveCallable();
    if (!callable) {
      return;
    }

    auto callableOp = cast<CallableOpInterface>(callable);
    Region *region = callableOp.getCallableRegion();
    if ((!region) || region->empty()) {
      return;
    }

    unsigned int operandNum = operand.getOperandNumber();
    BlockArgument arg = region->front().getArgument(operandNum);

    const PaddingLattice *argLattice = getLatticeElement(arg);
    if ((!argLattice) || (!argLattice->getValue().initialized)) {
      return;
    }

    PaddingLattice *callerLattice = getLatticeElement(operand.get());

    ChangeResult changed =
        callerLattice->getValue().meet(argLattice->getValue());

    propagateIfChanged(callerLattice, changed);
  }

  void setToExitState(PaddingLattice *lattice) override {
    ChangeResult changed = lattice->getValue().setToDefault();
    propagateIfChanged(lattice, changed);
  }
};

void addOpOperands(
    SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>>> &vec,
    const std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>> &element) {
  for (auto &existingPair : vec) {
    if (existingPair.first == element.first) {
      for (OpOperand *operand : element.second) {
        existingPair.second.insert(operand);
      }
      return;
    }
  }

  vec.push_back(element);
}

// Given the mapping of values to all padding requirements, choose which Value's
// will get padded
// TODO - Currently, the approach replaces the padding with the requirement
// whenever it requires certain padding, regardless of if padding occured
// upstream, etc.
//  Could try to improve
//  For example, if a tensor requires 0 padding (and is 0 padded), and one of
//  its users requires 0 padding, its user will add the 0 padding again In some
//  special cases, the padding might still be preserved (for example adding two
//  zero padded tensors gives a zero padded tensor)
DenseMap<Value, SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>>>>
choosePadding(const DenseMap<Value, PotentialPaddingRequirements> &reqs) {
  DenseMap<Value,
           SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>>>>
      curPadding;
  for (const auto &req : reqs) {
    Value cur = req.getFirst();
    const DenseMap<TypedAttr, SmallPtrSet<OpOperand *, 2>> &strictReqs =
        req.getSecond().strictReqs;
    for (const auto &strictReq : strictReqs) {
      const std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>> &p = strictReq;
      addOpOperands(curPadding[cur], p);
    }
  }

  return curPadding;
}

// Given a Value and a padding, return a new Value that is the original Value
// padded with the desired padding Sets insertion point to after mask
Value setPaddingForValue(OpBuilder &builder, Location loc, Value val,
                         TypedAttr padding, Value paddingMask) {
  RankedTensorType type = cast<RankedTensorType>(val.getType());
  builder.setInsertionPointAfterValue(paddingMask);

  Value tensorPadding = createTensor(builder, loc, type, padding);

  Value selectOp =
      builder.create<arith::SelectOp>(loc, paddingMask, val, tensorPadding);
  return selectOp;
}

// Before replacing the op with the replacement Values, updates the shapeMap and
// chosenPadding map to ensure these keys exist
void replaceOpSafely(
    IRRewriter &rewriter, Operation *op, const ValueRange &replacements,
    DenseMap<Value, LocalTensorShapeData> &shapeMap,
    DenseMap<Value,
             SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>>>>
        &chosenPadding,
    Operation *&newOpPtr) {
  for (size_t i = 0; i < op->getNumResults(); i++) {
    Value res = op->getResult(i);
    Value replacement = replacements[i];
    if (shapeMap.contains(res)) {
      const LocalTensorShapeData shapeData = shapeMap.at(res);
      shapeMap.erase(res);
      shapeMap[replacement] = shapeData;
    }
    if (chosenPadding.contains(res)) {
      const auto paddingData = chosenPadding.at(res);
      chosenPadding.erase(res);
      chosenPadding[replacement] = paddingData;
    }
  }

  rewriter.replaceOp(op, replacements);
  newOpPtr = replacements[0].getDefiningOp();
}

// final ir replacement/rewrite step
void finalCodegen(
    FuncOp &mod, DenseMap<Value, LocalTensorShapeData> &shapeMap,
    DenseMap<Value,
             SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>>>>
        &chosenPadding) {
  std::queue<Operation *> workQueue;

  mod.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isNonPow2TensorOperation(op)) {
      workQueue.push(op);
    }
  });

  IRRewriter rewriter(mod);

  while (!workQueue.empty()) {
    Operation *op = workQueue.front();
    workQueue.pop();

    // Need to keep track of results for iteration later
    // original Operation* op ptr may not be valid after using replaceOp
    Operation *resOp = op;
    if (isTensorProducer(op)) {
      Value val = op->getResult(0);
      const LocalTensorShapeData &shapeData = shapeMap.at(val);
      Value replacement = getReplacementCreationOp(rewriter, op, shapeData);

      replaceOpSafely(rewriter, op, replacement, shapeMap, chosenPadding,
                      resOp);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      // Getting mask
      // all operands are the same shape before (and thus same shape after)
      Value firstOperand = reduceOp.getOperand(0);
      const LocalTensorShapeData &data = shapeMap.at(firstOperand);

      if (!reduceOrScanOpNeedsMask(reduceOp)) {
        updateGeneralTensorOp(rewriter, op, shapeMap);
      } else {
        Value mask = createPaddingMask(rewriter, op, data.dataShape,
                                       data.localVirtualShape);
        triton::ReduceOp res = getReplacementReduceOp(rewriter, reduceOp, mask);
        ValueRange replacements = res->getResults().drop_back();
        replaceOpSafely(rewriter, op, replacements, shapeMap, chosenPadding,
                        resOp);
      }
    } else if (auto scanOp = dyn_cast<triton::ScanOp>(op)) {
      Value firstOperand = scanOp.getOperand(0);
      if (!scanOp.getReverse() || !reduceOrScanOpNeedsMask(scanOp)) {
        updateGeneralTensorOp(rewriter, op, shapeMap);
      } else {
        const LocalTensorShapeData &data = shapeMap.at(firstOperand);
        Value mask = createPaddingMask(rewriter, op, data.dataShape,
                                       data.localVirtualShape);
        triton::ScanOp res = getReplacementScanOp(rewriter, scanOp, mask);
        ValueRange replacements = res->getResults().drop_back();
        replaceOpSafely(rewriter, op, replacements, shapeMap, chosenPadding,
                        resOp);
      }
    } else if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op)) {
      const LocalTensorShapeData &operandData = shapeMap.at(reshapeOp.getSrc());
      const LocalTensorShapeData &resultData =
          shapeMap.at(reshapeOp.getResult());
      std::optional<std::pair<std::vector<bool>, std::vector<bool>>>
          potentialDataLocs =
              getComplexReshapeDataLocs(operandData, resultData);
      if (!potentialDataLocs) {
        updateSimpleReshapeOp(rewriter, reshapeOp, resultData);
      } else {
        std::pair<std::vector<bool>, std::vector<bool>> dataLocs =
            *potentialDataLocs;
        Value replacement = getReplacementComplexReshapeOp(
            rewriter, reshapeOp, resultData, dataLocs.first, dataLocs.second);
        replaceOpSafely(rewriter, op, replacement, shapeMap, chosenPadding,
                        resOp);
      }
    } else if (auto histogramOp = dyn_cast<triton::HistogramOp>(op)) {
      const LocalTensorShapeData &operandData =
          shapeMap.at(histogramOp.getSrc());
      const LocalTensorShapeData &resultData =
          shapeMap.at(histogramOp.getResult());

      Value replacement = getReplacementHistogramOp(rewriter, histogramOp,
                                                    operandData, resultData);
      ValueRange replacements = replacement;
      replaceOpSafely(rewriter, op, replacements, shapeMap, chosenPadding,
                      resOp);
    } else if (isGenericTensorOp(op)) {
      updateGeneralTensorOp(rewriter, op, shapeMap);
    }

    ValueRange results = resOp->getResults();
    ValueRange blockArgs;
    if (!resOp->getRegions().empty() && !resOp->getRegion(0).empty()) {
      blockArgs = resOp->getRegion(0).front().getArguments();
    }

    // TODO - support other ops here or in isGenericTensorOp +
    // updateGeneralTensorOp (add before the isGenericTensorOp check)

    // If any results require padding then apply the padding
    for (Value res : results) {
      if (chosenPadding.contains(res)) {
        const SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>>>
            &allPadding = chosenPadding.at(res);
        const LocalTensorShapeData shapeData = shapeMap.at(res);
        Value mask = createPaddingMask(rewriter, res.getDefiningOp(),
                                       shapeData.dataShape,
                                       shapeData.localVirtualShape, true);
        // Go through pairs of (Padding value, Set of Users)
        for (const auto &paddingUsers : allPadding) {
          // Apply padding
          TypedAttr padding = paddingUsers.first;
          Value replacement = setPaddingForValue(
              rewriter, res.getDefiningOp()->getLoc(), res, padding, mask);
          shapeMap[replacement] = shapeData;

          // Set padding for users
          for (OpOperand *operand : paddingUsers.second) {
            rewriter.modifyOpInPlace(operand->getOwner(),
                                     [&]() { operand->set(replacement); });
          }
        }
      }
    }

    // If any block args require padding then apply the padding
    for (Value arg : blockArgs) {
      if (chosenPadding.contains(arg)) {
        const SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>>>
            &allPadding = chosenPadding.at(arg);
        const LocalTensorShapeData shapeData = shapeMap.at(arg);

        // Insert at start of block instead of before op
        Block *block = arg.getParentBlock();
        Location loc = block->getParentOp()->getLoc();
        rewriter.setInsertionPointToStart(block);
        Value mask = createPaddingMask(rewriter, loc, shapeData.dataShape,
                                       shapeData.localVirtualShape);
        // Go through pairs of (Padding value, Set of Users)
        for (const auto &paddingUsers : allPadding) {
          // Apply padding
          TypedAttr padding = paddingUsers.first;
          Value replacement =
              setPaddingForValue(rewriter, loc, arg, padding, mask);
          shapeMap[replacement] = shapeData;

          // Set padding for users
          for (OpOperand *operand : paddingUsers.second) {
            rewriter.modifyOpInPlace(operand->getOwner(),
                                     [&]() { operand->set(replacement); });
          }
        }
      }
    }
  }
}

class ConvertNonPowerTwoTensorsPass
    : public impl::ConvertNonPowerTwoTensorsBase<
          ConvertNonPowerTwoTensorsPass> {
public:
  using ConvertNonPowerTwoTensorsBase::ConvertNonPowerTwoTensorsBase;
  void runOnOperation() override {
    FuncOp module = getOperation();

    // Make sure that the non power of two tensor ops are supported by this pass
    if (failed(verifyOps(module))) {
      module.emitError("Unsupported non power of two tensor operations found");
      signalPassFailure();
      return;
    }

    // For tt.load, tt.store, tt.atomic_rmw ops that accept a tensor of
    // pointers, add a default mask and other tensor if missing, or set the
    // boundary check attr for load/stores on tensor pointers
    addMasksToLoadAndStores(module);

    // For tensor.insert_slice ops that have a non power of 2 dim on the slice
    // axis, or tensor.extract_slice ops that have a dynamic offset, or offset +
    // expanded slice axis > source tensor slice axis, splits the
    // insert_slice/extract_slice into power of 2 slices along the slice axis
    splitSliceOps(module);

    // For each non power of two tensor operand/result, calculate shape data
    DenseMap<Value, LocalTensorShapeData> shapeMap =
        populateLocalShapeData(module);

    // Determine padding requirements and push them up to their sources
    DataFlowSolver solver;
    SymbolTableCollection symbolTable;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    auto *paddingPropagationAnalysis =
        solver.load<BackwardPaddingPopulationAnalysis>(symbolTable, shapeMap);

    if (failed(solver.initializeAndRun(module))) {
      module.emitError("Error occured trying to perform padding analysis");
      signalPassFailure();
      return;
    }

    DenseMap<Value, PotentialPaddingRequirements> potentialPadding =
        paddingPropagationAnalysis->paddingMap;

    // Choose what padding(s) to use for each Value which needs to have padding
    // TODO - Currently implemented in an inefficient way
    DenseMap<Value,
             SmallVector<std::pair<TypedAttr, SmallPtrSet<OpOperand *, 2>>>>
        chosenPadding = choosePadding(potentialPadding);
    // Replace and generate instructions
    finalCodegen(module, shapeMap, chosenPadding);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertNonPowerTwoTensorsPass() {
  return std::make_unique<ConvertNonPowerTwoTensorsPass>();
}

} // namespace bishengir::triton
