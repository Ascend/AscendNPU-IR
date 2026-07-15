//===- Util.h ---BiShengIR Dialect Uitls-------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_UTILS_UTIL_H
#define BISHENGIR_DIALECT_UTILS_UTIL_H

#include <utility>

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/raw_ostream.h"

#include <numeric>

#define CEIL_FACTOR(x, y) (((x) + ((y)-1)) / (y) * (y))
#define CEIL_DIV(x, y) (((x) + ((y)-1)) / (y))
#define UINT8_WIDTH 8
#define DEBUG_LINE_BEG(m) "===[" #m "]===[BEG]>>>\n"
#define DEBUG_LINE_END(m) "<<<[" #m "]===[END]===\n"

namespace llvm {
template <typename T>
inline raw_ostream &operator<<(raw_ostream &os, const SmallVectorImpl<T> &vec) {
  if (vec.empty())
    return os << "is empty";
  os << "(" << vec.size() << " entries) = [";
  for (size_t i = 0; i < vec.size(); ++i) {
    os << vec[i];
    if (i < vec.size() - 1)
      os << ", ";
  }
  return os << "]";
}
} // namespace llvm
namespace mlir {
namespace utils {
constexpr const uint8_t kBitsToByte = 8;
constexpr static unsigned int INTR_BITS_PER_BYTE = 8;
constexpr static unsigned int INTR_BYTES_PER_BLOCK = 32;
constexpr static unsigned int INTR_BYTES_PER_REPEAT = 256;
constexpr static unsigned int FRACTAL_BLOCK_NUM = 16;
constexpr static int64_t kUBAlignSizeInBits = 32 * 8;
static constexpr llvm::StringLiteral kEnableAutoMarkBufferSize =
    "enable_auto_mark_buffer_size";
static constexpr llvm::StringLiteral kMemrefAsPtr = "memref.memref_as_ptr";
static constexpr llvm::StringLiteral maskOpIdx = "mask_op_idx";
static constexpr llvm::StringLiteral reachedMaskOpsIdx = "reached_mask_ops_idx";
static constexpr llvm::StringLiteral maskBitWidth = "mask_bit_width";
static const llvm::StringLiteral kMapForToForallAttrName = "map_for_to_forall";

namespace debugger {

// Type trait to check if T is an LLVM-style container
template <typename T, typename = void>
struct IsLLVMContainer : public std::false_type {};

template <typename T>
struct IsLLVMContainer<T, std::void_t<decltype(std::declval<T>().begin()),
                                      decltype(std::declval<T>().end()),
                                      decltype(std::declval<T>().size())>>
    : public std::true_type {};

// Type trait to check if T supports indexing
template <typename T, typename = void> struct HasSubscript : public std::false_type {};

template <typename T>
struct HasSubscript<T, std::void_t<decltype(std::declval<T>()[0])>>
    : public std::true_type {};

template <typename T, typename = void> struct IsPrintable : public std::false_type {};

template <typename T>
struct IsPrintable<T, std::void_t<decltype(std::declval<llvm::raw_ostream>() << std::declval<T>())>>
    : public std::true_type {};

template <typename T>
std::string to_string(const T &container, int indent = 0, bool useEndl = false);

template <typename T>
std::string toStrHelper(const T &value, int indent, bool useEndl) {
  if constexpr (IsLLVMContainer<T>::value) {
    return to_string(value, indent + 2, useEndl);
  } else if constexpr (detail::is_pair<T>::value) {
    return "(" + to_string(value.first) + ", " + to_string(value.second) + ")";
  } else if constexpr (IsPrintable<T>::value) {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    os << value;
    os.flush();
    return buffer;
  } else {
    return std::to_string(value);
  }
}
template <typename T>
std::string to_string(const T &container, int indent, bool useEndl) {
  std::ostringstream oss;
  std::string indentation(indent, ' ');

  auto appendEl = [&](const auto &element, bool isLast) {
    if (useEndl)
      oss << indentation;
    oss << toStrHelper(element, indent, useEndl);
    if (!isLast)
      oss << ", ";
    if (useEndl)
      oss << "\n";
  };

  if (useEndl)
    oss << indentation;
  else
    oss << "[";

  if (!container.empty()) {
    if (useEndl)
      oss << "\n";
    if constexpr (HasSubscript<T>::value) {
      for (size_t i = 0; i < container.size(); ++i) {
        appendEl(container[i], i == container.size() - 1);
      }
    } else {
      auto it = container.begin();
      auto end = container.end();
      while (it != end) {
        appendEl(*it, std::next(it) == end);
        ++it;
      }
    }
    if (useEndl)
      oss << indentation;
  }
  oss << "]";
  return oss.str();
}

std::string getPrettyOpName(Operation *op);

} // namespace debugger

// Currently dtype cast rules:
// (1-RINT ) f32 -> f16/bf16/f32
// (2-RINT ) f16/bf16 -> f32
// (3-RINT ) i8 -> f16
// (4-TRUNC) float -> int
// (5-RINT ) int -> float
// (6-RINT ) int -> int
template <typename T> T selectRoundMode(Type inType, Type outType) {
  if (inType.isF32()) {
    if (outType.isF16() || outType.isBF16() || outType.isF32()) {
      return T::RINT;
    }
  }

  if (outType.isF32()) {
    if (inType.isF16() || inType.isBF16()) {
      return T::RINT;
    }
  }

  if (inType.isInteger(8) && outType.isF16()) {
    return T::RINT;
  }

  if (isa<mlir::FloatType>(inType) && outType.isInteger()) {
    return T::TRUNC;
  }

  if (inType.isInteger() && isa<mlir::FloatType>(outType)) {
    return T::RINT;
  }

  if (inType.isInteger() && outType.isInteger()) {
    return T::RINT;
  }
  llvm::report_fatal_error("unsupported type cast.");
}

inline Type getMostElementType(
    SmallVector<Type> types,
    const std::function<bool(const Type &, const Type &)> &comparator) {
  llvm::sort(types, comparator);
  return types.front();
}

/// Return the type that has the smallest bits.
/// \note The input types must be an int or float type.
inline Type getSmallestElementType(SmallVector<Type> types) {
  return getMostElementType(
      std::move(types), [](const Type &lhs, const Type &rhs) -> bool {
        return lhs.getIntOrFloatBitWidth() < rhs.getIntOrFloatBitWidth();
      });
}

/// Return the type that has the largest bits.
/// \note The input types must be an int or float type.
inline Type getLargestElementType(SmallVector<Type> types) {
  return getMostElementType(
      std::move(types), [](const Type &lhs, const Type &rhs) -> bool {
        return lhs.getIntOrFloatBitWidth() > rhs.getIntOrFloatBitWidth();
      });
}
/// Return true if the input operation is `memref.alloc` or `memref.alloca`
bool isAllocLikeOp(Operation *op);

/// Return true if the input value is the SSA result value of `memref.alloc` or
/// `memref.alloca` op.
bool isAllocLikeOp(Value val);

/// Return true if the input operation is `memref.collapse_shape`
bool isCollapseShapeOp(Operation *op);

/// Return true if the input value is the SSA result value of
/// `memref.collapse_shape` op.
bool isCollapseShapeOp(Value val);

/// Return true if the input operation is `memref.expand_shape`
bool isExpandShapeOp(Operation *op);

/// Return true if the input value is the SSA result value of
/// `memref.expand_shape` op.
bool isExpandShapeOp(Value val);

/// Set buffer size to alloc like ops by constructing a new, static-shape
/// alloc. The new alloc is viewed to the original shape.
/// \note Assertion is raised if `op` is not `memref.alloc` or `memref.alloca`
memref::ViewOp createAllocWithSettingBufferSize(Operation *op,
                                                int64_t bufferSize,
                                                RewriterBase &opBuilder);

/// Returns true if input type is a shaped type with known rank.
bool hasRank(const Type &type);

/// Returns the shape rank if exist
std::optional<size_t> getShapeRank(const Type &type);
std::optional<size_t> getShapeRank(const Value &v);

using DimensionShape = SmallVector<int64_t>;
std::optional<std::pair<size_t, DimensionShape>>
getValueShapeInfo(const Value &v);

/// Returns true if input type is shaped.
bool isShaped(const Type &type);

/// Returns true if none of the value is dynamic.
/// \note This should only be applied to shapes/strides/offsets.
bool isFullyStatic(const SmallVector<int64_t> &values);

/// Returns shape of shaped type with known rank.
SmallVector<int64_t> getShape(const Type &type);

/// Get total size of a given array.
std::optional<int64_t> getStaticTotalSize(const ArrayRef<int64_t> &shapes);

/// Get total size in bits given the shape in array and element type.
std::optional<int64_t> getStaticTotalSizeInBits(const ArrayRef<int64_t> &shapes,
                                                Type elemType);

SmallVector<int64_t>
getReassociationMapping(ArrayRef<ReassociationIndices> reassociation);

SmallVector<int64_t> getNewIndexing(ArrayRef<int64_t> oldIndexing,
                                    ArrayRef<int64_t> mapping);

SmallVector<int64_t>
getNewIndexingFullPermutation(ArrayRef<int64_t> oldIndexing,
                              ArrayRef<int64_t> mapping);

void sortReassociation(MutableArrayRef<ReassociationIndices> reassociation);

SmallVector<int64_t> inversePermutation(ArrayRef<int64_t> perm);

/// This gives an arbitrary integer, compress them into [0, |dims|]
SmallVector<int64_t> compressElements(SmallVector<int64_t> dims);

void renumberReassociation(
    MutableArrayRef<ReassociationIndices> newReassociation);

/// Returns true if value is scalar or zero rank tensor or one-size tensor
bool isScalarLike(Value value);

/// Return true if value is ShapedType with size one, e.g. tensor<1x1x1xf32>.
bool isOneSizeShape(Value value);

/// Extract unique scalar value from scalar-like tensor
std::optional<Value> extractScalarValue(PatternRewriter &rewriter, Location loc,
                                        Value src);

/// Return true if op is from arith dialect.
bool isArithOp(Operation *op);

/// Gets a suitable vector size from the element type. Vector size is fixed
/// for regbase; returns an integer representing a vector size.
int64_t getVectorSizeByElementType(Type t);

/// Return true if op is annotation mark op with attr `name`
bool isAnnotationWithAttr(Operation *op, StringRef name);

/// Search the users of value v to find first annotation op with attr `name`.
std::optional<Operation *> getAnnotateOpWithAttr(Value v, StringRef name);

/// Search the users of value v to find all the annotation ops with attr `name`.
SmallVector<Operation *> getAllAnnotateOpsWithAttr(Value v, StringRef name);

/// Search the users of each operand to find the annotation op with attr `name`.
SmallVector<std::optional<Operation *>>
getAnnotateOpWithAttrForEachOperand(const SmallVectorImpl<Value> &operands,
                                    StringRef name);

/// erase trivially deadops from the back to front.
void eraseTriviallyDeadOps(ArrayRef<Operation *> ops);

/// get value according to the indices of every dimension
Value getScalarValue(
    RewriterBase &rewriter, Location loc, Value v,
    std::optional<const llvm::SmallVector<Value> *> indices = std::nullopt);

/// Create memref loadop.
memref::LoadOp createSinglePointLoad(
    RewriterBase &rewriter, Location loc, Value memOper,
    std::optional<llvm::SmallVector<Value>> indexesVec = std::nullopt);

/// Create memref storeop.
memref::StoreOp createSinglePointStore(
    RewriterBase &rewriter, Location loc, Value storeValue, Value memOper,
    std::optional<llvm::SmallVector<Value>> indexesVec = std::nullopt);

/// Create tensor.empty or memref.alloc op with the same type as source
Value createEmptyOp(OpBuilder &builder, Location loc, Value source);

///  Create tensor.empty or memref.alloc op with the same shape as source
///  but with element type targetElemType
Value createEmptyOpWithTargetElemType(
    OpBuilder &builder, Location loc, Value source, Type targetElemType,
    std::optional<MemRefLayoutAttrInterface> layout = std::nullopt);

/// Create a static shape `tensor.empty` op with the `targetTensorType`.
///
/// \Note assertion will be raised if `targetTensorType` does not have static
///       shape.
tensor::EmptyOp createStaticShapeEmptyOp(OpBuilder &builder, Location loc,
                                         TensorType targetTensorType);

/// Returns RankedTensorType with same shape as source tensor with
/// srcTensorType and element type newTensorElementType.
/// e.g. tensor<512xi1> = getTensorTypeWithSameShape(tensor<512xf32>, i1)
RankedTensorType getTensorTypeWithSameShape(Type srcTensorType,
                                            Type newTensorElementType);

/// Return the func::FuncOp called by `callOp`.
template <typename FuncOp_t, typename CallOp_t>
FuncOp_t getCalledFunction(CallOp_t callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp_t>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Create a constant of type 'type' at location 'loc' whose value is 'value'
template <typename T>
Value createConstantOp(OpBuilder &builder, Location loc, Type type, T value) {
  TypedAttr attr;
  if (isa<FloatType>(type))
    attr = builder.getFloatAttr(type, value);
  else if (isa<IntegerType>(type))
    attr = builder.getIntegerAttr(type, static_cast<int64_t>(value));
  else {
    auto vecTy = cast<ShapedType>(type);
    attr = SplatElementsAttr::get(vecTy, value);
  }
  return builder.create<arith::ConstantOp>(loc, attr);
}

/// Implementation is borrowed from
/// `mlir/lib/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.cpp`.
/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp);

bool areShapesAligned(ArrayRef<int64_t> staticShapes, int64_t alignment);
/// Check if op's users all satisfy the condition function.
std::optional<bool>
checkUsersAllWithCondition(Value v, Operation *rootOp,
                           const std::function<bool(Operation *op)> &condFn,
                           const std::function<bool(Operation *op)> &skipFn);

std::optional<bool> checkUsersAllWithConditionImpl(
    Value v, Operation *rootOp,
    const std::function<bool(Operation *op)> &condFn,
    const std::function<bool(Operation *op)> &skipFn);

int checkDefsAllWithCondition(Value v,
                              const std::function<int(Operation *op)> &condFn);

int checkDefsAnyWithCondition(Value v,
                              const std::function<int(Operation *op)> &condFn);

/// Try to trace back the current mermef-typed value to the source values.
SmallVector<Value> tracebackMemRefVec(Value memrefVal);

/// Try to trace back the current mermef-typed value to the target source
/// values.
SmallVector<Value>
tracebackMemRefVecByTargetFn(Value memrefVal,
                             std::function<bool(Value)> targetFn);

/// Try to trace back the current mermef-typed value to the source value.
/// This function always return a value.
Value tracebackMemRef(Value memrefVal);

/// Try to trace back the current mermef-typed value to the source
/// `mermef.alloc`.
/// Return `std::nullopt` if max-iteration is reached, or that the value doesn't
/// originate from a alloc op.
std::optional<memref::AllocOp> tracebackMemRefToAlloc(Value memrefVal);

/// Try to trace back the current mermef-typed value to the source values.
SmallVector<Value> tracebackMemRefAllocAndAlias(Value memrefVal);

void fillAncestorOfOperation(SmallPtrSet<Operation *, 3> &container,
                             Operation *op);

/// Returns all operations of specified type from \p scffor. If \p withNested is false
/// nested scf::ForOp regions are ignored.
template <typename T>
SmallVector<T> collectScfForBodyOperations(scf::ForOp scffor, bool withNested) {
  SmallVector<T> ops;
  for (Region &region : scffor->getRegions()) {
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto opT = dyn_cast<T>(op)) {
        ops.push_back(opT);
      } else if (!withNested && isa<scf::ForOp>(op)) {
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
  return ops;
}

/// Trace back \p value operand dependencies to \p block arguments.
SmallVector<BlockArgument> tracebackOperandsToBlockArguments(Value value, Block* block);

template <typename... StopOpTys>
std::optional<Operation*> valueCalculatedUsingOperationInsideBlockImpl(
  Value value, DenseSet<Value> &visited, Operation *op, Block *block) {

  if (!visited.insert(value).second) {
    return std::nullopt;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getParentBlock() == block) {
      return std::nullopt;
    }
  }

  if (auto *defOp = value.getDefiningOp()) {
    if (defOp == op || isa<StopOpTys...>(defOp)) {
      return defOp;
    }

    for (auto operand : defOp->getOperands()) {
      if (auto foundOp = valueCalculatedUsingOperationInsideBlockImpl<StopOpTys...>(operand, visited, op, block)) {
        return foundOp;
      }
    }
  }

  return std::nullopt;
}

/// Return true if \p value is produced by or transitively uses \p op.
/// Block arguments in \p block are not expanded (loop-carried iter_args are
/// treated as dependency boundaries).
template <typename... StopOpTys>
std::optional<Operation*> valueCalculatedUsingOperationInsideBlock(Value value, Operation *op, Block *block) {
  if (value.getDefiningOp() == op) {
    return op;
  } else {
    DenseSet<Value> visited;
    return valueCalculatedUsingOperationInsideBlockImpl<StopOpTys...>(value, visited, op, block);
  }
}

/// Create tmp buffer or tensor using specified element type.
Value createTmpBufferOrTensorWithTargetType(
    OpBuilder &builder, Location loc, Value source,
    std::optional<Type> targetElemType = std::nullopt,
    std::optional<ArrayRef<int64_t>> targetShape = std::nullopt,
    bool allowDynShapeAlloc = true);

/// Get vector of dims with DimOp for buffer or tensor.
llvm::SmallVector<Value> getTensorOrMemrefShapeDims(PatternRewriter &rewriter,
                                                    Location loc, Value source);

/// Extract the arith value of the arith.constant.
template <typename T> FailureOr<T> getArithConstantOpValue(Value value) {
  auto constOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp());
  if (constOp == nullptr) {
    return failure();
  }
  Attribute valueAttr = constOp.getValue();
  T v;
  if (auto valIntAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    v = static_cast<T>(valIntAttr.getInt());
  } else if (auto valFPAttr = dyn_cast<FloatAttr>(valueAttr)) {
    v = static_cast<T>(valFPAttr.getValueAsDouble());
  } else {
    llvm::report_fatal_error("getArithConstantOpValue supports only IntOrFloat");
  }
  return v;
}

template <typename TensorOrMemRefType,
          typename = typename std::enable_if_t<
              std::is_same_v<TensorOrMemRefType, TensorType> ||
              std::is_same_v<TensorOrMemRefType, MemRefType>>>
SmallVector<int> collectAlignUnits(ArrayRef<int32_t> alignDims,
                                   ArrayRef<int32_t> alignBytes,
                                   TensorOrMemRefType unalignedTy) {
  int rank = unalignedTy.getRank();
  const unsigned bitWidth = unalignedTy.getElementTypeBitWidth();
  SmallVector<int> alignTargets(rank, 1);
  assert(alignBytes.size() == alignDims.size());
  for (size_t i = 0; i < alignDims.size(); ++i) {
    int dim = alignDims[i];
    assert(dim >= 0 && dim < rank);
    int alignBits = alignBytes[i] * utils::kBitsToByte;
    if (bitWidth % alignBits == 0) {
      // If the alignment is smaller than type size, align to itself
      continue;
    }
    assert(alignBits % bitWidth == 0 &&
           "Alignment cannot satisfied by bitwidth");
    alignTargets[dim] =
        static_cast<int>(std::lcm(alignBits / bitWidth, alignTargets[dim]));
  }

  int innerAlignedUnits = 1;
  int shapeAccumulation = 1;
  auto shapes = unalignedTy.getShape();
  SmallVector<int> alignUnits(rank + 1, 1);
  for (int dim = rank - 1; dim >= 0; --dim) {
    // The alignment target forces the INNER dimension to get aligned
    int newAlignedUnits = std::lcm(innerAlignedUnits, alignTargets[dim]);
    if (newAlignedUnits == 0)
      return {};
    if (shapeAccumulation % newAlignedUnits == 0) {
      // already aligned
      alignUnits[dim + 1] = 1;
    } else {
      if (innerAlignedUnits == 0)
        return {}; // should be impossible case (SecA_DivideByZero)
      alignUnits[dim + 1] = newAlignedUnits / innerAlignedUnits;
    }
    innerAlignedUnits = newAlignedUnits;
    if (!ShapedType::isDynamic(shapes[dim])) {
      shapeAccumulation =
          shapeAccumulation * std::lcm(shapes[dim], alignUnits[dim + 1]);
    }
  }
  // The outermost dimension needs no extra alignments
  alignUnits[0] = 1;
  return alignUnits;
}

template <typename IRType, typename CType> bool isConst(TypedAttr v, CType t) {
  if constexpr (std::is_same_v<IRType, FloatAttr>) {
    auto srcTypeAttr = dyn_cast_or_null<FloatAttr>(v);
    return srcTypeAttr == FloatAttr::get(v.getType(), static_cast<double>(t));
  }
  if constexpr (std::is_same_v<IRType, IntegerAttr>) {
    auto srcIntAttr = dyn_cast_or_null<IntegerAttr>(v);
    auto intval = srcIntAttr.getInt();
    return intval == t;
  }
  return false;
}

// get axis kind
hivm::AxisKind getAxisKind(int dim, int rank);

// get axis kind after outlining
hivm::AxisKind getOutlinedAxisKind(int dim, int rank);

BitVector arrayToMask(ArrayRef<int64_t> elements, int maskSize);

std::optional<int64_t> traceToAllocMaxSize(mlir::Value memrefVal);

/// Return a `memref.dim` or `tensor.dim` for the shape of `v` at `dim`.
OpFoldResult getDimOFR(OpBuilder &builder, Location loc, Value v, int64_t dim);
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a `memref.subview` or a `tensor.extract_slice` based on the type of
/// the `source`.
Value getSlice(OpBuilder &b, Location loc, Value source,
               ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
               ArrayRef<OpFoldResult> strides);

bool isAlignedInUB(Type type);

bool isUnstructuredMemAccLoop(Operation *op);

int64_t getNumPerRepeat(Type t);

/// Rewrite loop iter_arg to drop unit dims or to fixed hardware types
template <bool DropUnitDimOnly>
struct ForOpLegalization : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override;
  virtual ~ForOpLegalization() = default;
};

ModuleOp getTopLevelModuleOp(Operation *op);

bool isValidHIVMTileElementType(Type type);

unsigned getHIVMTileSliceMinNumElts(Type type);

bool isValidHIVMTileVectorType(VectorType vType);

bool isValidTwoDimVectorType(VectorType vType);

/// Return true if transfer write op suits for change to StoreWithStride
bool isTransferWriteSuitForStoreWithStride(Operation *op);

bool isValidHIVMTileElementType(Type type);

unsigned getHIVMTileSliceMinNumElts(Type type);

bool isValidHIVMTileVectorType(VectorType vType);

bool isValidTwoDimVectorType(VectorType vType);

/// Return true if transfer write op suits for change to StoreWithStride
bool isTransferWriteSuitForStoreWithStride(Operation *op);

void dumpReassociationIndicesVector(
    const SmallVector<ReassociationIndices> &reassocVec);

} // namespace utils

namespace reshape_utils {

bool isInitOp(Operation *op);

bool isReshapingOp(Operation *op);

bool isSlicingOp(Operation *op);

bool isArgOp(Operation *op);

bool isStopPropagatable(Operation *op);

bool isOutOp(Operation *op);

bool isUnsupportedOp(Operation *op);

bool isSkippableOp(Operation *op);

bool isExplicitlyAllowedCollapseOp(Operation *op);

bool isLegalOp(Operation *op);

bool isReturnOp(Operation *op);

bool isContainerAllocator(Operation *op);

bool isElementwiseOp(Operation *op);

bool isMarkedAsElementwiseOp(Operation *op);

bool isZeroDimensionOp(Operation *op);

bool isMarkedAsElementwiseUnaryOp(Operation *op);

bool isAllParallelOp(Operation *op);

bool areReassociationsCompatible(
    ArrayRef<ReassociationIndices> collapseReassoc,
    ArrayRef<ReassociationIndices> expandReassoc,
    SmallVector<ReassociationIndices> &supposedExpand,
    SmallVector<ReassociationIndices> &supposedCollapse,
    ArrayRef<int64_t> collapseSourceShape, ArrayRef<int64_t> expandShapeResult,
    SmallVector<int64_t> &newExpandShape);

/// Get reassociation of expandShapeOp
/// eg.
///   tensor.expand_shape %arg0 [[0, 1], [2], [3, 4]] : tensor<4x5x3xf32> into
///                                                     tensor<4x1x5x3x1f32>
/// here outRank = 5, expandDims = {1, 4},
/// the result reassociation=[[0, 1], [2], [3, 4]]
SmallVector<SmallVector<int64_t, 2>>
getReAssociation(ArrayRef<int64_t> expandDims, int64_t outRank);

bool isConstIntOne(Value v);

/**
 * @brief Remove unit dimensions (size=1) from a shape vector
 * @details Filters out all elements with value 1 from the input shape,
 *          returning a new shape vector with only non-unit dimensions
 *
 * Shape Transformation Example:
 * - Input:  [2, 1, 3, 1, 5] (shape with unit dims at index 1, 3)
 * - Output: [2, 3, 5] (unit dims removed)
 *
 * Edge Cases:
 * - Input:  [1, 1, 1] → Output: [] (all unit dims)
 * - Input:  [4, 5, 6] → Output: [4, 5, 6] (no unit dims)
 * - Input:  [] → Output: [] (empty shape)
 *
 * @param[in] shape Input shape vector (int64_t dim sizes)
 * @return SmallVector<int64_t> New shape vector with unit dims (size=1) removed
 * @note Does not modify the original input shape vector
 */
SmallVector<int64_t> getSqueezedShape(SmallVectorImpl<int64_t> &type);

/**
 * @brief Multiply two OpFoldResult (OFR) values (constants/Values)
 * @details Optimize multiplication for constant OFRs; create arith::MulIOp for
 * dynamic Value-backed OFRs
 *
 * Multiplication Logic Flow:
 * 1. Constant-Constant (OFR → IntegerAttr):
 *    Input: lhs=IntegerAttr(5), rhs=IntegerAttr(3) → Output: IntegerAttr(15)
 *
 * 2. Constant-Special Case (0/1):
 *    Input: lhs=IntegerAttr(0), rhs=Value(%0) → Output: lhs (IntegerAttr(0))
 *    Input: lhs=IntegerAttr(1), rhs=Value(%0) → Output: rhs (Value(%0))
 *
 * 3. Dynamic-Dynamic (Value-backed OFR):
 *    Input: lhs=Value(%a), rhs=Value(%b) → Output: Value(arith.muli %a, %b :
 * index)
 *
 * 4. Constant-Dynamic Mixed:
 *    Input: lhs=IntegerAttr(2), rhs=Value(%a) → Output: Value(arith.muli
 * (arith.constant 2), %a : index)
 *
 * @param[in] lhs Left-hand side OFR (constant/Value)
 * @param[in] rhs Right-hand side OFR (constant/Value)
 * @param[in,out] b OpBuilder for creating constant/mul ops
 * @param[in] loc Location for new operations (arith::ConstantOp/arith::MulIOp)
 * @return OpFoldResult Result of multiplication (IntegerAttr or Value of
 * arith::MulIOp)
 * @note 1. Prioritizes constant folding (avoids op creation for constant
 * inputs)
 *       2. Handles 0/1 shortcuts to avoid unnecessary ops
 *       3. Converts constant OFRs to arith::ConstantOp for mixed
 * dynamic/constant cases
 *       4. Returns Value-backed OFR for non-constant multiplication
 */
OpFoldResult mulOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     OpBuilder &b, const Location loc);

/**
 * @brief Extract integer value from OpFoldResult (OFR) if it's IntegerAttr
 * @details Check if OFR holds an IntegerAttr and return its int64_t value;
 * return nullopt otherwise
 *
 * Value Extraction Example:
 * - Case 1 (Match):
 *   OFR = IntegerAttr(42) → returns std::optional<int64_t> = 42
 * - Case 2 (No Match):
 *   OFR = Value(%0 : index) → returns std::nullopt
 * - Case 3 (No Match):
 *   OFR = FloatAttr(3.14) → returns std::nullopt
 *
 * @param[in] ofr OpFoldResult to extract integer value from
 * @return std::optional<int64_t> Integer value if OFR is IntegerAttr; nullopt
 * otherwise
 * @note Only handles IntegerAttr (ignores other Attribute types or Value-backed
 * OFR)
 */
std::optional<int64_t> getIntAttr(const OpFoldResult ofr);

/**
 * @brief Multiply two OpFoldResult (OFR) values (constants/Values)
 * @details Optimize multiplication for constant OFRs; create arith::MulIOp for
 * dynamic Value-backed OFRs
 *
 * Multiplication Logic Flow:
 * 1. Constant-Constant (OFR → IntegerAttr):
 *    Input: lhs=IntegerAttr(5), rhs=IntegerAttr(3) → Output: IntegerAttr(15)
 *
 * 2. Constant-Special Case (0/1):
 *    Input: lhs=IntegerAttr(0), rhs=Value(%0) → Output: lhs (IntegerAttr(0))
 *    Input: lhs=IntegerAttr(1), rhs=Value(%0) → Output: rhs (Value(%0))
 *
 * 3. Dynamic-Dynamic (Value-backed OFR):
 *    Input: lhs=Value(%a), rhs=Value(%b) → Output: Value(arith.muli %a, %b :
 * index)
 *
 * 4. Constant-Dynamic Mixed:
 *    Input: lhs=IntegerAttr(2), rhs=Value(%a) → Output: Value(arith.muli
 * (arith.constant 2), %a : index)
 *
 * @param[in] lhs Left-hand side OFR (constant/Value)
 * @param[in] rhs Right-hand side OFR (constant/Value)
 * @param[in,out] b OpBuilder for creating constant/mul ops
 * @param[in] loc Location for new operations (arith::ConstantOp/arith::MulIOp)
 * @return OpFoldResult Result of multiplication (IntegerAttr or Value of
 * arith::MulIOp)
 * @note 1. Prioritizes constant folding (avoids op creation for constant
 * inputs)
 *       2. Handles 0/1 shortcuts to avoid unnecessary ops
 *       3. Converts constant OFRs to arith::ConstantOp for mixed
 * dynamic/constant cases
 *       4. Returns Value-backed OFR for non-constant multiplication
 */
OpFoldResult mulOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     OpBuilder &b, const Location loc);

/**
 * @brief Shrink reassociation indices by removing dropped dimensions
 * @details Adjust reassociation indices to exclude dropped dims and shift
 * remaining indices (compensate for dropped dims)
 *
 * Reassociation Transformation Example:
 * - Input:
 *   reassocIdxVec = [[0,1,2], [3,4]] (original indices for rank=5)
 *   droppedDims = [0, 0, 1, 0, 0] (bit 2 marked as dropped)
 *   shiftTable = [0,0,1,1,1] (precomputed shift for each index)
 *
 * - Step 1 (Filter dropped dim 2):
 *   [[0,1], [3,4]] (remove index 2 from first group)
 *
 * - Step 2 (Shift indices by shiftTable):
 *   [[0,1], [2,3]] (3→2, 4→3; shiftTable[3]=1, shiftTable[4]=1)
 *
 * - Output:
 *   reassocIdxVec = [[0,1], [2,3]] (empty groups are erased)
 *
 * @param[in,out] reassocIdxVec Nested reassociation indices (modified in-place)
 * @param[in] droppedDims BitVector marking dropped dimensions (bit set = dim is
 * dropped)
 *
 * @note 1. shiftTable precomputes cumulative count of dropped dims up to each
 * index
 *       2. Remaining indices are shifted left by number of dropped dims before
 * them
 *       3. Empty reassociation groups are erased from the vector
 *       4. Operation is in-place (modifies input reassocIdxVec directly)
 */
void shrinkReassocIdxByDroppedDims(
    SmallVector<ReassociationIndices> &reassocIdxVec,
    llvm::SmallBitVector &droppedDims);
} // namespace reshape_utils

} // namespace mlir

#endif
