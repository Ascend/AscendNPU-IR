//===------------- ExtraBuffer.cpp ----------------------------------------===//
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

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <queue>

#define DEBUG_TYPE "bishengir-util"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir {

namespace {

SmallVector<Value> tracebackImpl(Value memrefVal) {
  // case 1: v is the iter_arg of a scf.for
  SmallVector<Value> result;
  if (auto arg = dyn_cast<BlockArgument>(memrefVal)) {
    if (arg.getParentRegion() == nullptr) {
      return result;
    }
    if (auto forOp =
            dyn_cast<scf::ForOp>(arg.getParentRegion()->getParentOp())) {
      if (arg.getArgNumber() > 0 &&
          forOp.getInitArgs().size() > arg.getArgNumber() - 1) {
        result.emplace_back(forOp.getInitArgs()[arg.getArgNumber() - 1]);
        result.emplace_back(forOp.getYieldedValues()[arg.getArgNumber() - 1]);
      }
    }
    if (auto whileOp =
            dyn_cast<scf::WhileOp>(arg.getParentRegion()->getParentOp())) {
      result.emplace_back(whileOp.getTiedLoopInit(arg)->get());
    }
  }

  Operation *def = memrefVal.getDefiningOp();
  if (!def) {
    // failed to trace back
    return result;
  }

  // case 2: v is the result of cast-like ops
  //  - memref.cast
  //  - memref.collapse_shape
  //  - memref.expand_shape
  //  - memref.memory_space_cast
  //  - memref.reinterpret_cast
  //  - memref.reshape
  //  - memref.transpose
  if (auto op = dyn_cast<memref::CastOp>(def)) {
    result.emplace_back(op.getSource());
  } else if (auto op = dyn_cast<memref::CollapseShapeOp>(def)) {
    result.emplace_back(op.getSrc());
  } else if (auto op = dyn_cast<memref::ExpandShapeOp>(def)) {
    result.emplace_back(op.getSrc());
  } else if (auto op = dyn_cast<memref::MemorySpaceCastOp>(def)) {
    result.emplace_back(op.getSource());
  } else if (auto op = dyn_cast<memref::ReinterpretCastOp>(def)) {
    result.emplace_back(op.getSource());
  } else if (auto op = dyn_cast<memref::ReshapeOp>(def)) {
    result.emplace_back(op.getSource());
  } else if (auto op = dyn_cast<memref::TransposeOp>(def)) {
    result.emplace_back(op.getIn());
  } else if (auto op = dyn_cast<UnrealizedConversionCastOp>(def)) {
    result.emplace_back(
        op.getOperand(cast<OpResult>(memrefVal).getResultNumber()));
  } else if (auto op = dyn_cast<scf::ForOp>(def)) {
    // trace back memref.alloc support scf.for
    result.emplace_back(
        op.getInitArgs()[cast<OpResult>(memrefVal).getResultNumber()]);
    result.emplace_back(
        op.getYieldedValues()[cast<OpResult>(memrefVal).getResultNumber()]);
  } else if (auto op = dyn_cast<scf::IfOp>(def)) {
    result.emplace_back(op.thenYield()->getOperand(
        cast<OpResult>(memrefVal).getResultNumber()));
    result.emplace_back(op.elseYield()->getOperand(
        cast<OpResult>(memrefVal).getResultNumber()));
  }

  if (!result.empty()) {
    return result;
  }

  // case 3: v is the result of the view-like ops
  //  - memref::view
  //  - memref::subview
  if (auto op = dyn_cast<memref::ViewOp>(def)) {
    result.emplace_back(op.getViewSource());
  } else if (auto op = dyn_cast<memref::SubViewOp>(def)) {
    result.emplace_back(op.getViewSource());
  }

  return result;
}

} // namespace

namespace utils {

Value getScalarValue(RewriterBase &rewriter, Location loc, Value v,
                     std::optional<const llvm::SmallVector<Value> *> indices) {
  if (isa<MemRefType>(v.getType())) {
    if (indices == std::nullopt) {
      auto loadOp = createSinglePointLoad(rewriter, loc, v);
      return loadOp.getResult();
    }
    auto loadOp = createSinglePointLoad(rewriter, loc, v, *(indices.value()));
    return loadOp.getResult();
  }
  return v;
}

memref::LoadOp
createSinglePointLoad(RewriterBase &rewriter, Location loc, Value memOper,
                      std::optional<llvm::SmallVector<Value>> indexesVec) {
  assert(isa<MemRefType>(memOper.getType()));
  auto memShapeDimSize = cast<MemRefType>(memOper.getType()).getShape().size();
  auto constZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  llvm::SmallVector<Value> indexes;
  if (indexesVec.has_value()) {
    indexes = indexesVec.value();
  } else {
    indexes = llvm::SmallVector<Value>(memShapeDimSize, constZero);
  }
  return rewriter.create<memref::LoadOp>(loc, memOper, indexes);
}

memref::StoreOp
createSinglePointStore(RewriterBase &rewriter, Location loc, Value storeValue,
                       Value memOper,
                       std::optional<llvm::SmallVector<Value>> indexesVec) {
  assert(isa<MemRefType>(memOper.getType()));
  auto memShapeDimSize = cast<MemRefType>(memOper.getType()).getShape().size();
  auto constZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  llvm::SmallVector<Value> indexes;
  if (indexesVec.has_value()) {
    indexes = indexesVec.value();
  } else {
    indexes = llvm::SmallVector<Value>(memShapeDimSize, constZero);
  }

  return rewriter.create<memref::StoreOp>(loc, storeValue, memOper, indexes);
}

Value createEmptyOpWithTargetElemType(
    OpBuilder &builder, Location loc, Value source, Type targetElemType,
    std::optional<MemRefLayoutAttrInterface> layout) {
  auto shapedType = cast<ShapedType>(source.getType());
  if (isa<TensorType>(shapedType)) {
#if BISHENGIR_BUILD_STANDALONE_IR_ONLY
    llvm_unreachable("Not implemented");
#else
    // TODO: it should be defined in Dialect/Tensor/IR
    return tensor::createTensorEmptyOpWithTargetElemType(builder, loc, source,
                                                         targetElemType);
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY
  }
  return memref::createMemRefAllocOpWithTargetElemType(
      builder, loc, source, targetElemType, std::move(layout));
}

Value createEmptyOp(OpBuilder &builder, Location loc, Value source) {
  auto shapedType = cast<ShapedType>(source.getType());
  if (isa<TensorType>(shapedType)) {
#if BISHENGIR_BUILD_STANDALONE_IR_ONLY
    llvm_unreachable("Not implemented");
#else
    return tensor::createTensorEmptyOp(builder, loc, source);
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY
  }
  return memref::createMemRefAllocOp(builder, loc, source);
}

tensor::EmptyOp createStaticShapeEmptyOp(OpBuilder &builder, Location loc,
                                         TensorType targetTensorType) {
  assert(targetTensorType.hasStaticShape());
  return builder.create<tensor::EmptyOp>(loc, targetTensorType.getShape(),
                                         targetTensorType.getElementType());
}

func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp) {
  func::ReturnOp returnOp;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

std::optional<bool>
checkUsersAllWithCondition(Value v, Operation *rootOp,
                           const std::function<bool(Operation *op)> &condFn,
                           const std::function<bool(Operation *op)> &skipFn) {
  // Flag initialization is nullopt which means we can't infer flag now
  std::optional<bool> flag = std::nullopt;

  for (auto &use : v.getUses()) {
    auto *op = use.getOwner();
    LLVM_DEBUG(llvm::dbgs() << "[TRACING USERS]" << *op << "\n";);
    if (op == rootOp)
      // When meet rootOp, just ignore it and keep original state
      continue;

    if (condFn(op)) {
      // When meet satisfied op, enable flag to true
      flag = true;
      continue;
    }

    // If op can't satisfy condition and can't be skipped, return false directly
    if (!skipFn(op))
      return false;

    // For all skipped ops, just continue searching its result
    for (auto opRes : op->getResults()) {
      auto resCheck = checkUsersAllWithCondition(opRes, rootOp, condFn, skipFn);
      if (!resCheck.has_value())
        continue;

      if (!resCheck.value())
        return false;

      flag = true;
    }
    if (isa<scf::YieldOp>(op)) {
      auto resNum = use.getOperandNumber();
      Operation *parentOp = op->getParentOp();
      assert(parentOp && "parent op cannot be nullptr");
      auto resCheck = checkUsersAllWithCondition(parentOp->getResult(resNum),
                                                 rootOp, condFn, skipFn);
      if (!resCheck.has_value())
        continue;

      if (!resCheck.value())
        return false;

      flag = true;
    }
  }

  return flag;
}

int checkDefsAllWithCondition(Value v,
                              const std::function<int(Operation *op)> &condFn) {
  int res = 0;
  auto vTy = v.getType();
  if (!(isa<mlir::TensorType>(vTy) || isa<mlir::BaseMemRefType>(vTy))) {
    return res;
  }
  auto defOp = v.getDefiningOp();
  if (defOp == nullptr) {
    return res;
  }
  LLVM_DEBUG(llvm::dbgs() << "[TRACING DEFS]" << *defOp << "\n";);
  res = condFn(defOp);
  if (res < 0) {
    return res;
  }
  for (auto operand : defOp->getOperands()) {
    int cond = checkDefsAllWithCondition(operand, condFn);
    if (cond < 0) {
      return cond;
    }
    if (res < cond) {
      res = cond;
    }
  }
  return res;
}

int checkDefsAnyWithCondition(Value v,
                              const std::function<int(Operation *op)> &condFn) {
  int res = 0;
  auto vTy = v.getType();
  if (!(isa<mlir::TensorType>(vTy) || isa<mlir::BaseMemRefType>(vTy))) {
    return res;
  }
  auto defOp = v.getDefiningOp();
  if (defOp == nullptr) {
    return res;
  }
  LLVM_DEBUG(llvm::dbgs() << "[TRACING DEFS]" << *defOp << "\n";);
  res = condFn(defOp);
  if (res > 0) {
    return res;
  }
  for (auto operand : defOp->getOperands()) {
    int cond = checkDefsAnyWithCondition(operand, condFn);
    if (cond > 0) {
      return cond;
    }
  }
  return res;
}

void fillAncestorOfOperation(SmallPtrSet<Operation *, 3> &container,
                             Operation *op) {
  if (!op)
    return;
  container.insert(op);
  // Propagate castTo
  std::queue<Operation *> workList;
  workList.push(op);
  while (!workList.empty()) {
    Operation *workOp = workList.front();
    workList.pop();
    for (auto opr : workOp->getOperands()) {
      Operation *opOpr = opr.getDefiningOp();
      if (!opOpr)
        continue;
      if (container.contains(opOpr))
        continue;
      container.insert(opOpr);
      workList.push(opOpr);
    }
  }
}

FailureOr<llvm::SmallVector<Value>>
getTensorOrMemrefDynSizes(OpBuilder &builder, Location loc, Value source,
                          std::optional<ArrayRef<int64_t>> targetShape) {
  const bool isMemref = isa<MemRefType>(source.getType());
  const bool isTensor = isa<TensorType>(source.getType());
  if (!isMemref && !isTensor) {
    emitError(loc, "Type of source should be MemRefType or TensorType!");
    return failure();
  }

  llvm::SmallVector<Value> dynSizes;
  ArrayRef<int64_t> shape = targetShape.has_value()
                                ? targetShape.value()
                                : cast<ShapedType>(source.getType()).getShape();

  for (size_t i = 0; i < shape.size(); i++)
    if (ShapedType::isDynamic(shape[i]))
      dynSizes.push_back(getDimValue(builder, loc, source, i));

  return dynSizes;
}

inline bool isPureStatic(ArrayRef<OpFoldResult> mixedValues) {
  return llvm::all_of(mixedValues,
                      [](OpFoldResult x) { return x.is<Attribute>(); });
}

inline void markDynShapeAlloc(OpBuilder &builder, Value source,
                              memref::AllocOp &tmpAllocOp,
                              bool allowDynShapeAlloc) {
  auto srcAlloc = utils::tracebackMemRefToAlloc(source);
  if (!srcAlloc.has_value() || !srcAlloc.value()) {
    if (allowDynShapeAlloc)
      return;
    emitError(tmpAllocOp->getLoc(), "alloc is not found");
    llvm::report_fatal_error("alloc is not found");
    return;
  }
  auto srcAllocMemref = srcAlloc.value().getMemref();
  auto elemType = getElementTypeOrSelf(srcAllocMemref.getType());
  auto srcAllocShape = srcAllocMemref.getType().getShape();
  auto i8TypeWidth = builder.getI8Type().getIntOrFloatBitWidth();
  auto maybeStaticTotalSize =
      utils::getStaticTotalSizeInBits(srcAllocShape, elemType);
  if (!maybeStaticTotalSize.has_value()) {
    if (allowDynShapeAlloc)
      return;
    emitError(tmpAllocOp->getLoc(), "shape has dynamic dimension");
    llvm::report_fatal_error("shape has dynamic dimension");
    return;
  }
  int64_t allocSize =
      maybeStaticTotalSize.value() / static_cast<int64_t>(i8TypeWidth);
  // for dynamic case, set buffer size by annotation.mark op
  auto tmpMarkOp = builder.create<annotation::MarkOp>(tmpAllocOp->getLoc(),
                                                      tmpAllocOp->getResult(0));
  tmpMarkOp->setAttr(hivm::kBufferSizeInByteAttr,
                     builder.getI64IntegerAttr(allocSize));
}

/// Create tmp buffer or tensor using specified element type,
/// if targetElemType is null, then then use source's element type.
Value createTmpBufferOrTensorWithTargetType(
    OpBuilder &builder, Location loc, Value source,
    std::optional<Type> targetElemType,
    std::optional<ArrayRef<int64_t>> targetShape, bool allowDynShapeAlloc) {
  const bool isMemref = isa<MemRefType>(source.getType());
  const bool isTensor = isa<TensorType>(source.getType());
  if (!isMemref && !isTensor) {
    emitError(loc, "Type of source should be MemRefType or TensorType!");
    return nullptr;
  }

  ShapedType srcShapedType = cast<ShapedType>(source.getType());
  if (!targetElemType.has_value()) {
    targetElemType = srcShapedType.getElementType();
  }
  SmallVector<OpFoldResult> bufferSizes =
      isMemref ? memref::getMixedSizes(builder, loc, source)
               : tensor::getMixedSizes(builder, loc, source);
  if (targetShape.has_value()) {
    assert(bufferSizes.size() == targetShape.value().size());
    for (auto [bufferI, tarI] : llvm::zip(bufferSizes, targetShape.value())) {
      if (!ShapedType::isDynamic(tarI))
        bufferI = builder.getIndexAttr(tarI);
    }
  }

  Value tmp;
  if (isMemref) {

    memref::AllocOp tmpAllocOp = builder.create<memref::AllocOp>(
        loc, /*sizes*/ bufferSizes, /*elementType*/ targetElemType.value(),
        /*memorySpace*/ cast<MemRefType>(source.getType()).getMemorySpace());

    if (!isPureStatic(bufferSizes)) {
      // Currently only hfusion pipeline has this tag and allows dyn-sized alloc
      markDynShapeAlloc(builder, source, tmpAllocOp, allowDynShapeAlloc);
    }
    tmp = tmpAllocOp.getResult();
  } else {
    tmp = builder.create<tensor::EmptyOp>(
        loc, /*sizes*/ bufferSizes,
        /*elementType*/ targetElemType.value() /*, encoding = {}*/);
  }
  return tmp;
}

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&builder, &loc, &v, &dim](auto) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&builder, &loc, &v, &dim](auto) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      })
      .Default([&](auto) { return Value(); });
}

OpFoldResult getDimOFR(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  auto type = cast<ShapedType>(v.getType());
  if (!type.hasRank()) {
    llvm_unreachable("Cannot get dim for type with no rank");
    return {};
  }

  if (!type.isDynamicDim(dim))
    return builder.getIndexAttr(type.getDimSize(dim));

  return getDimValue(builder, loc, v, dim);
}

llvm::SmallVector<Value> getTensorOrMemrefShapeDims(PatternRewriter &rewriter,
                                                    Location loc,
                                                    Value source) {
#ifndef NDEBUG
  const bool isMemref = isa<MemRefType>(source.getType());
  const bool isTensor = isa<TensorType>(source.getType());
  assert((isMemref || isTensor) &&
         "Type of source should be MemRefType or TensorType!");
#endif
  auto shapedType = cast<ShapedType>(source.getType());
  llvm::SmallVector<Value> shapeDims;

  auto shape = shapedType.getShape();
  for (size_t i = 0; i < shape.size(); i++)
    shapeDims.push_back(getDimValue(rewriter, loc, source, i));

  return shapeDims;
}

Value getSlice(OpBuilder &b, Location loc, Value source,
               ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
               ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>(
          [&b, &loc, &source, &offsets, &sizes, &strides](auto) -> Value {
            return b
                .create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides)
                ->getResult(0);
          })
      .Case<MemRefType>(
          [&b, &loc, &source, &offsets, &sizes, &strides](auto) -> Value {
            return b
                .create<memref::SubViewOp>(loc, source, offsets, sizes, strides)
                ->getResult(0);
          })
      .Default([](auto) { return nullptr; });
}

hivm::AxisKind getAxisKind(int dim, int rank) {
  if (dim == rank - 1)
    return hivm::AxisKind::LAST;
  if (dim <= 0)
    return hivm::AxisKind::FIRST;
  return hivm::AxisKind::MIDDLE;
}

hivm::AxisKind getOutlinedAxisKind(int dim, int rank) {
  if (rank > 3)
    return getAxisKind(dim + 3 - rank, 3);
  return getAxisKind(dim, rank);
}

} // namespace utils

bool utils::isAllocLikeOp(Value val) {
  return isAllocLikeOp(val.getDefiningOp());
}

bool utils::isAllocLikeOp(Operation *op) {
  if (!op)
    return false;
  return isa<memref::AllocOp>(op) || isa<memref::AllocaOp>(op);
}

bool utils::isCollapseShapeOp(Value val) {
  return isCollapseShapeOp(val.getDefiningOp());
}

bool utils::isCollapseShapeOp(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  return isa<memref::CollapseShapeOp>(op);
}

bool utils::isExpandShapeOp(Value val) {
  return isExpandShapeOp(val.getDefiningOp());
}

bool utils::isExpandShapeOp(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  return isa<memref::ExpandShapeOp>(op);
}

memref::ViewOp
utils::createAllocWithSettingBufferSize(Operation *op, int64_t bufferSize,
                                        RewriterBase &opBuilder) {
  assert(isAllocLikeOp(op));
  OpBuilder::InsertionGuard g(opBuilder);
  opBuilder.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  auto oldType = dyn_cast<MemRefType>(op->getResultTypes().front());
  assert(oldType);
  // Create new alloc with static size.
  auto newMemrefType =
      MemRefType::get({bufferSize}, opBuilder.getI8Type(), mlir::AffineMap{},
                      oldType.getMemorySpace());
  Value newAlloc;
  if (isa<memref::AllocOp>(op)) {
    memref::AllocOp oldOp = cast<memref::AllocOp>(op);
    newAlloc = opBuilder
                   .create<memref::AllocOp>(loc, newMemrefType,
                                            oldOp.getAlignmentAttr())
                   .getMemref();
  } else {
    memref::AllocaOp oldOp = cast<memref::AllocaOp>(op);
    newAlloc = opBuilder
                   .create<memref::AllocaOp>(loc, newMemrefType,
                                             oldOp.getAlignmentAttr())
                   .getMemref();
  }
  // Create view from new alloc to old alloc's sizes and replace its use.
  auto startOffset = opBuilder.create<arith::ConstantIndexOp>(loc, 0);
  auto viewOp = opBuilder.create<memref::ViewOp>(
      loc, oldType, newAlloc, startOffset, op->getOperands());
  return viewOp;
}

// Returns true if input type is a shaped type with known rank.
bool utils::hasRank(const Type &type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return shapedType.hasRank();
  return false;
}

std::optional<size_t> utils::getShapeRank(const Type &type) {
  assert(type && "Type must not be null");
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    assert(shapedType.hasRank() && "ShapedType must have a rank");
    return shapedType.getRank();
  }
  return std::nullopt;
}

std::optional<size_t> utils::getShapeRank(const Value &v) {
  return getShapeRank(v.getType());
}

using DimensionShape = SmallVector<int64_t>;
std::optional<std::pair<size_t, DimensionShape>>
utils::getValueShapeInfo(const Value &v) {
  assert(v && "Value must not be null");
  if (auto shapedType = dyn_cast<ShapedType>(v.getType())) {
    assert(shapedType.hasRank() && "ShapedType must have a rank");
    return std::make_pair(shapedType.getRank(),
                          DimensionShape(shapedType.getShape().begin(),
                                         shapedType.getShape().end()));
  } else if (v.getType().isIntOrFloat() || isa<IndexType>(v.getType())) {
    // Handle scalar types as empty tensor
    return std::make_pair(0, DimensionShape{});
  } else {
    return std::nullopt;
  }
}

bool utils::isShaped(const Type &type) { return isa<ShapedType>(type); }

bool utils::isFullyStatic(const SmallVector<int64_t> &values) {
  return llvm::all_of(values, [](long s) { return s != ShapedType::kDynamic; });
}

SmallVector<int64_t> utils::getShape(const Type &type) {
  return SmallVector<int64_t>(cast<ShapedType>(type).getShape());
}

std::optional<int64_t>
utils::getStaticTotalSize(const ArrayRef<int64_t> &shapes) {
  int64_t totalSize = 1;
  for (const auto &shape : shapes) {
    if (ShapedType::isDynamic(shape)) {
      return std::nullopt;
    }
    totalSize = totalSize * shape;
  }
  return totalSize;
}

std::optional<int64_t>
utils::getStaticTotalSizeInBits(const ArrayRef<int64_t> &shapes,
                                Type elemType) {
  auto totalSize = utils::getStaticTotalSize(shapes);
  if (!totalSize.has_value()) {
    return std::nullopt;
  }
  int64_t elemSizeInBits = elemType.getIntOrFloatBitWidth();
  return totalSize.value() * elemSizeInBits;
}

void utils::sortReassociation(
    MutableArrayRef<ReassociationIndices> reassociation) {
  sort(reassociation,
       [](const auto &a, const auto &b) { return a.front() < b.front(); });
}

[[nodiscard]] SmallVector<int64_t>
utils::getReassociationMapping(ArrayRef<ReassociationIndices> reassociation) {
  // Apply the collapse
  SmallVector<int64_t> mapping(reassociation.back().back() + 1);
  for (const auto &[idx, dim] : llvm::enumerate(reassociation)) {
    for (const auto &reassigned : dim) {
      mapping[reassigned] = static_cast<uint32_t>(idx);
    }
  }
  return mapping;
}

[[nodiscard]] SmallVector<int64_t>
utils::getNewIndexing(ArrayRef<int64_t> oldIndexing,
                      ArrayRef<int64_t> mapping) {
  SmallVector<int64_t> newIndexing;
  newIndexing.reserve(oldIndexing.size());
  for (const auto dim : oldIndexing) {
    newIndexing.push_back(mapping[dim]);
  }
  newIndexing.erase(std::unique(newIndexing.begin(), newIndexing.end()),
                    newIndexing.end());
  return newIndexing;
}

[[nodiscard]] SmallVector<int64_t>
utils::getNewIndexingFullPermutation(ArrayRef<int64_t> oldIndexing,
                                     ArrayRef<int64_t> mapping) {
  // E.g: mapping:     0 1 1 2 3
  // if permutation is 3 0 1 4 2
  // it means [[3], [0, 1], [4], [2]]
  // the new Indexing is [2, 0, 3, 1]
  SmallVector<int64_t> newIndexing;
  int rank = static_cast<int64_t>(oldIndexing.size());
  assert(oldIndexing.size() == mapping.size());
  for (int i = 0; i < rank; i++) {
    if (i > 0 && mapping[i] == mapping[i - 1])
      continue;
    // taking the first index only [3, 0, 4, 2]
    newIndexing.push_back(oldIndexing[i]);
  }

#ifndef NDEBUG
  for (auto &tmp : newIndexing)
    LLVM_DEBUG(llvm::dbgs() << tmp << ", ";);
#endif
  LLVM_DEBUG(llvm::dbgs() << "\n";);
  auto used = newIndexing;
  std::sort(used.begin(), used.end());
  for (auto &idx : newIndexing) {
    idx = std::lower_bound(used.begin(), used.end(), idx) - used.begin();
  }
  return newIndexing;
}

[[nodiscard]] SmallVector<int64_t>
utils::inversePermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> inv(perm.size());
  for (size_t i = 0; i < inv.size(); ++i) {
    inv[perm[i]] = static_cast<int>(i);
  }
  return inv;
}

SmallVector<int64_t> utils::compressElements(SmallVector<int64_t> dims) {
  auto used = llvm::to_vector(dims);
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  for (auto &idx : dims) {
    idx = std::lower_bound(used.begin(), used.end(), idx) - used.begin();
  }
  return dims;
}

void utils::renumberReassociation(
    MutableArrayRef<ReassociationIndices> newReassociation) {
  int shapeCounter = 0;
  for (auto &reassociationIndex : newReassociation) {
    for (auto &shapeIndex : reassociationIndex) {
      shapeIndex = shapeCounter++;
    }
  }
}

#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
bool utils::isScalarLike(Value value) {
  Type type = value.getType();
  std::optional<size_t> rankMaybe = utils::getShapeRank(type);
  // for scalar with no rank
  if (!rankMaybe.has_value()) {
    return type.isIntOrIndexOrFloat();
  }
  // for zero rank tensor like tensor<f32>
  size_t rank = rankMaybe.value();
  if (rank == 0) {
    return true;
  }
  // e.g. dense<1.000000e+00>
  if (mlir::linalg::isSplatDense(value)) {
    return true;
  }
  // for one size tensor like tensor<1x1x1xf32>
  return isOneSizeShape(value);
}
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY

bool utils::isOneSizeShape(Value value) {
  if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    return llvm::all_of(shapedType.getShape(),
                        [](int64_t shape) { return shape == 1; });
  }
  return false;
}

#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
std::optional<Value> utils::extractScalarValue(PatternRewriter &rewriter,
                                               Location loc, Value src) {
  Type type = src.getType();
  if (type.isIntOrIndexOrFloat()) {
    // src already scalar
    return src;
  }

  SmallVector<Value> indices;
  std::optional<size_t> rankMaybe = utils::getShapeRank(type);
  if (!rankMaybe.has_value()) {
    return std::nullopt;
  }
  if (mlir::linalg::isSplatDense(src)) {
    return mlir::linalg::createConstantFromDenseSplat(src, rewriter);
  }
  if (isOneSizeShape(src)) {
    // only extract scalar from one size tensor/memref
    size_t rank = rankMaybe.value();
    Value constZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (size_t i = 0; i < rank; ++i) {
      indices.push_back(constZero);
    }
    Value scalar = rewriter.create<tensor::ExtractOp>(loc, src, indices);
    return scalar;
  }
  return std::nullopt;
}
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY

bool utils::isArithOp(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  mlir::Dialect *dialect = op->getDialect();
  return dialect && dialect->getNamespace() ==
                        mlir::arith::ArithDialect::getDialectNamespace();
}

bool utils::isAnnotationWithAttr(Operation *op, StringRef name) {
  if (!isa<annotation::MarkOp>(op)) {
    return false;
  }

  auto markOp = cast<annotation::MarkOp>(op);
  return markOp.isAnnotatedBy(name);
}

std::optional<Operation *> utils::getAnnotateOpWithAttr(Value v,
                                                        StringRef name) {
  // find the annotation mark op with attr
  auto it = llvm::find_if(v.getUsers(), [&](Operation *user) {
    return utils::isAnnotationWithAttr(user, name);
  });
  if (it == v.getUsers().end()) {
    return std::nullopt;
  }

  return *it;
}

SmallVector<Operation *> utils::getAllAnnotateOpsWithAttr(Value v,
                                                          StringRef name) {
  SmallVector<Operation *> annotateOpsWithAttr;
  for (auto user : v.getUsers()) {
    if (utils::isAnnotationWithAttr(user, name)) {
      annotateOpsWithAttr.push_back(user);
    }
  }
  return annotateOpsWithAttr;
}

SmallVector<std::optional<Operation *>>
utils::getAnnotateOpWithAttrForEachOperand(
    const SmallVectorImpl<Value> &operands, StringRef name) {
  SmallVector<std::optional<Operation *>> maybeMarkOps;
  for (const auto &it : operands) {
    maybeMarkOps.push_back(utils::getAnnotateOpWithAttr(it, name));
  }

  return maybeMarkOps;
}

bool utils::areShapesAligned(ArrayRef<int64_t> staticShapes,
                             int64_t alignment) {
  for (auto &shape : staticShapes) {
    if (ShapedType::isDynamic(shape))
      return false;
    if (shape % alignment != 0)
      return false;
  }
  return true;
}

SmallVector<Value> utils::tracebackMemRefVec(Value memrefVal) {
  return utils::tracebackMemRefVecByTargetFn(
      memrefVal, [](Value val) { return utils::isAllocLikeOp(val); });
}

Value utils::tracebackMemRef(Value memrefVal) {
  SmallVector<Value> memrefValues = utils::tracebackMemRefVec(memrefVal);
  if (memrefValues.empty()) {
    return memrefVal;
  }
  if (memrefValues.size() > 1) {
    LDBG("tracebackMemRef found multiple sources!");
  }
  return memrefValues[0];
}

SmallVector<Value>
utils::tracebackMemRefVecByTargetFn(Value memrefVal,
                                    std::function<bool(Value)> targetFn) {
  SmallVector<Value> memrefValues;
  memrefValues.push_back(memrefVal);
  int loopBound = 256;
  while (!memrefValues.empty() &&
         std::any_of(memrefValues.begin(), memrefValues.end(),
                     [&targetFn](Value &val) { return !targetFn(val); })) {
    Value allocVal;
    for (auto val : memrefValues) {
      if (!targetFn(val)) {
        allocVal = val;
        break;
      }
    }
    auto upward = tracebackImpl(allocVal);
    if (upward.empty()) {
      break;
    }
    auto it = std::find(memrefValues.begin(), memrefValues.end(), allocVal);
    memrefValues.erase(it);
    memrefValues.append(upward.begin(), upward.end());

    // avoid infinite loop
    if (loopBound-- < 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "tracebackMemRef exceeds loopBound(" << loopBound << ")!");
      break;
    }
  }
  return memrefValues;
}

std::optional<memref::AllocOp> utils::tracebackMemRefToAlloc(Value memrefVal) {
  auto tracedValue = utils::tracebackMemRef(memrefVal);
  return utils::isAllocLikeOp(tracedValue)
             ? tracedValue.getDefiningOp<memref::AllocOp>()
             : std::optional<memref::AllocOp>();
}

SmallVector<Value> utils::tracebackMemRefAllocAndAlias(Value memrefVal) {
  return utils::tracebackMemRefVecByTargetFn(memrefVal, [](Value val) {
    return utils::isAllocLikeOp(val) || utils::isCollapseShapeOp(val) ||
           utils::isExpandShapeOp(val);
  });
}

namespace reshape_utils {

bool isInitOp(Operation *op) { return isa<tensor::EmptyOp>(op); }

bool isReshapingOp(Operation *op) {
  return isa<tensor::CollapseShapeOp, tensor::ReshapeOp, tensor::ExpandShapeOp>(
      op);
}

bool isSlicingOp(Operation *op) {
  return isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op);
}

bool isArgOp(Operation *op) {
  return isReshapingOp(op) || isInitOp(op) ||
         isa<arith::ConstantOp, bufferization::ToTensorOp>(op);
}

bool isStopPropagatable(Operation *op) {
  return isInitOp(op) || isa<arith::ConstantOp>(op);
}

bool isOutOp(Operation *op) { return isReshapingOp(op) || isReturnOp(op); }

bool isUnsupportedOp(Operation *op) { return !op->getDialect(); }

bool isSkippableOp(Operation *op) {
  return isOutOp(op) || isArgOp(op) || isUnsupportedOp(op);
}

bool isExplicitlyAllowedCollapseOp(Operation *op) {
  return isa<tensor::ExtractOp, tensor::ConcatOp, tensor::PadOp,
             tensor::ExtractSliceOp, tensor::InsertSliceOp,
             hfusion::InterleaveOp, hfusion::DeinterleaveOp>(op);
}

bool isContainerAllocator(Operation *op) { return isa<tensor::EmptyOp>(op); }

bool isElementwiseOp(Operation *op) {
  if (!isAllParallelOp(op))
    return false;
  auto genericOp = dyn_cast<linalg::LinalgOp>(op);

  LLVM_DEBUG(llvm::dbgs() << *op << "\n";);
  if (llvm::any_of(genericOp.getIndexingMapsArray(),
                   [](AffineMap map) { return !map.isIdentity(); })) {
    return false;
  }
  return true;
}

bool isMarkedAsElementwiseOp(Operation *op) {
  // This would handle scalar as well
  return isa_and_present<linalg::ElemwiseBinaryOp, linalg::ElemwiseUnaryOp,
                         linalg::FillOp>(op);
}

bool isZeroDimensionOp(Operation *op) {
  // This would handle scalar as well
  for (auto opr : op->getOperands()) {
    auto rank = utils::getShapeRank(opr).value_or(0);
    if (rank != 0lu)
      return false;
  }
  return true;
}

bool isMarkedAsElementwiseUnaryOp(Operation *op) {
  // This would handle scalar as well
  return isa_and_present<linalg::ElemwiseUnaryOp, linalg::FillOp>(op);
}

bool isAllParallelOp(Operation *op) {
  // Check if it's a Linalg op with all parallel loops
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    bool isAllParallelLoops =
        (linalgOp.getNumLoops() == linalgOp.getNumParallelLoops());
    return isAllParallelLoops;
  }
  return false;
}

// TODO: Need to refactor this.
bool isLegalOp(Operation *op) {
  if (isa<linalg::MapOp, linalg::FillOp, linalg::GenericOp,
          linalg::ElemwiseBinaryOp, linalg::ElemwiseUnaryOp,
          linalg::BroadcastOp, linalg::ReduceOp, linalg::TransposeOp,
          linalg::MatmulOp, linalg::MatmulTransposeAOp,
          linalg::MatmulTransposeBOp, tensor::ExtractOp>(op)) {
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "Warning: unchecked operation " << *op << "\n");
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    bool isAllParallelLoops =
        linalgOp.getNumLoops() == linalgOp.getNumParallelLoops();
    if (isAllParallelLoops) {
      return true;
    }
  }
  return false;
}

bool isReturnOp(Operation *op) {
  return isa<func::ReturnOp, bufferization::MaterializeInDestinationOp>(op);
}

bool areReassociationsCompatible(
    ArrayRef<ReassociationIndices> collapseReassoc,
    ArrayRef<ReassociationIndices> expandReassoc,
    SmallVector<ReassociationIndices> &supposedExpand,
    SmallVector<ReassociationIndices> &supposedCollapse,
    ArrayRef<int64_t> collapseSourceShape, ArrayRef<int64_t> expandShapeResult,
    SmallVector<int64_t> &newExpandShape) {
  // Check if collapse and expand reassociations are inverses of each other
  if (collapseReassoc.size() != expandReassoc.size())
    return false;
  for (size_t i = 0; i < collapseReassoc.size(); ++i) {
    bool isCollapsing = collapseReassoc[i].size() > 1;
    bool isExpanding = expandReassoc[i].size() > 1;
    if (isCollapsing && isExpanding) {
      return false;
    }
    if (isExpanding) {
      for (auto el : expandReassoc[i]) {
        assert(el >= 0 && static_cast<size_t>(el) < expandShapeResult.size());
        newExpandShape.push_back(expandShapeResult[el]);
        supposedCollapse.push_back({-1});
      }
      supposedExpand.push_back(expandReassoc[i]);
    } else {
      for (auto el : collapseReassoc[i]) {
        newExpandShape.push_back(collapseSourceShape[el]);
        supposedExpand.push_back({-1});
      }
      supposedCollapse.push_back(collapseReassoc[i]);
    }
  }
  return true;
}

SmallVector<SmallVector<int64_t, 2>>
getReAssociation(ArrayRef<int64_t> expandDims, int64_t outRank) {
  std::set<int> expandDimsSet;
  expandDimsSet.insert(expandDims.begin(), expandDims.end());

  SmallVector<SmallVector<int64_t, 2>> retVecVec;
  SmallVector<int64_t, 2> vec;

  // push contiguous expand dims in the head of seq into vec
  int i = 0;
  for (; i < outRank; i++) {
    bool isExpandDim = expandDimsSet.count(i);
    if (isExpandDim) {
      vec.push_back(i);
    } else {
      break;
    }
  }

  // cut the vec if next is unexpand dim or unexisted
  for (; i < outRank; ++i) {
    vec.push_back(i);

    bool nextIsUnExpand = !expandDimsSet.count(i + 1);
    if (nextIsUnExpand) {
      // unexpanded dim
      retVecVec.push_back(vec);
      vec.clear();
    }
  }

  if (!vec.empty()) {
    retVecVec.push_back(vec);
  }
  return retVecVec;
}

} // namespace reshape_utils

BitVector utils::arrayToMask(ArrayRef<int64_t> elements, int maskSize) {
  BitVector ret(maskSize);
  for (auto el : elements) {
    ret.set(el);
  }
  return ret;
}

namespace {

/// Traceback `memrefVal` to its defining memref alloc if possible and return
/// the MemRefType if it has static shape.
std::optional<MemRefType> traceToGetStaticShapedType(mlir::Value memrefVal) {
  Operation *srcOp = memrefVal.getDefiningOp();
  // TODO: Need to confirm the scene where the problem occurred, Consider why
  // tracebackMemRef cannot be processed.
  if (srcOp && dyn_cast<memref::ReinterpretCastOp>(srcOp)) {
    auto srcValue = dyn_cast<memref::ReinterpretCastOp>(srcOp).getSource();
    if (auto extractStridedMetadataOp =
            dyn_cast<memref::ExtractStridedMetadataOp>(
                srcValue.getDefiningOp())) {
      memrefVal = extractStridedMetadataOp.getViewSource();
    }
  }

  auto newMemrefVal = utils::tracebackMemRef(memrefVal);
  if (!newMemrefVal) {
    return std::nullopt;
  }

  auto memrefType = dyn_cast<MemRefType>(newMemrefVal.getType());
  if (!memrefType || !memrefType.hasStaticShape()) {
    return std::nullopt;
  }
  return memrefType;
}
} // namespace

std::optional<int64_t> utils::traceToAllocMaxSize(mlir::Value memrefVal) {
  auto originalMemRefType = dyn_cast<MemRefType>(memrefVal.getType());
  assert(originalMemRefType);
  auto optionalMemrefType = traceToGetStaticShapedType(memrefVal);
  if (!(optionalMemrefType.has_value()))
    return std::nullopt;

  auto memrefType = optionalMemrefType.value();
  int64_t r = 1;
  for (int64_t n : memrefType.getShape()) {
    r *= n;
  }
  int64_t allocSizeInBit =
      r * static_cast<int64_t>(memrefType.getElementTypeBitWidth());
  return allocSizeInBit /
         static_cast<int>(originalMemRefType.getElementTypeBitWidth());
}
} // namespace mlir
