//===------------- Util.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

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
      if (auto *tiedInit = whileOp.getTiedLoopInit(arg)) {
        result.emplace_back(tiedInit->get());
      }
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
  } else if (auto op = dyn_cast<ViewLikeOpInterface>(def)) {
    // TODO: replace instance of view like op interface with this branch
    result.emplace_back(op.getViewSource());
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

  // case 4: v is the result of bufferization ops
  //  - bufferization.to_tensor
  //  - bufferization.to_memref
  if (auto op = dyn_cast<bufferization::ToTensorOp>(def)) {
    result.emplace_back(op.getMemref());
  } else if (auto op = dyn_cast<bufferization::ToMemrefOp>(def)) {
    result.emplace_back(op.getTensor());
  }

  return result;
}

} // namespace

namespace utils {

namespace debugger {
std::string getPrettyOpName(Operation *op) {
  std::string str;
  llvm::raw_string_ostream os(str);

  if (auto callOp = llvm::dyn_cast<mlir::func::CallOp>(op)) {
    os << "func.call @" << callOp.getCallee();
  } else {
    os << op->getName();
  }

  if (op->getNumResults() > 0) {
    os << " (res0: " << op->getResult(0) << ")";
  } else {
    os << " (ptr: " << op << ")";
  }

  return os.str();
}
} // namespace debugger

void eraseTriviallyDeadOps(ArrayRef<Operation *> ops) {
  for (auto I = ops.rbegin(), E = ops.rend(); I != E;) {
    Operation *curOp = *I;
    ++I;
    if (isOpTriviallyDead(curOp))
      curOp->erase();
  }
}

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

Value createEmptyOpWithTargetElemType(OpBuilder &builder, Location loc,
                                      Value source, Type targetElemType) {
  auto shapedType = cast<ShapedType>(source.getType());
  if (isa<TensorType>(shapedType)) {
    return tensor::createTensorEmptyOpWithTargetElemType(builder, loc, source,
                                                         targetElemType);
  }
  return memref::createMemRefAllocOpWithTargetElemType(builder, loc, source,
                                                       targetElemType);
}

Value createEmptyOp(OpBuilder &builder, Location loc, Value source) {
  auto shapedType = cast<ShapedType>(source.getType());
  if (isa<TensorType>(shapedType)) {
    return tensor::createTensorEmptyOp(builder, loc, source);
  }
  return memref::createMemRefAllocOp(builder, loc, source);
}

tensor::EmptyOp createStaticShapeEmptyOp(OpBuilder &builder, Location loc,
                                         TensorType targetTensorType) {
  assert(targetTensorType.hasStaticShape());
  return builder.create<tensor::EmptyOp>(loc, targetTensorType.getShape(),
                                         targetTensorType.getElementType());
}

RankedTensorType getTensorTypeWithSameShape(Type srcTensorType,
                                            Type newTensorElementType) {
  auto rankedTensorType = dyn_cast<RankedTensorType>(srcTensorType);
  assert(rankedTensorType && "Source type is not a mlir::RankedTensorType");
  auto resultTensorShape = rankedTensorType.getShape();
  return mlir::RankedTensorType::get(resultTensorShape, newTensorElementType);
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
      assert(op->getParentOp());
      auto resNum = use.getOperandNumber();
      if (resNum >= op->getParentOp()->getNumResults()) {
        continue;
      }
      auto resCheck = checkUsersAllWithCondition(
          op->getParentOp()->getResult(resNum), rootOp, condFn, skipFn);
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

bool checkUsersAnyWithCondition(
    Value v, Operation *rootOp,
    const std::function<bool(Operation *op)> &condFn,
    const std::function<bool(Operation *op)> &skipFn, DenseSet<Value> &flag) {
  if (flag.contains(v))
    return false;
  flag.insert(v);

  for (auto *op : v.getUsers()) {
    LLVM_DEBUG(llvm::dbgs() << "[TRACING USERS]" << *op << "\n";);

    // When meet rootOp, just ignore it and keep original state
    if (op == rootOp)
      continue;

    // When meet satisfied op, return true
    if (condFn(op))
      return true;

    // If op can't satisfy condition and can't be skipped, return false directly
    if (!skipFn(op))
      return false;

    if (op->hasTrait<OpTrait::IsTerminator>()) {
      Operation *parentOp = op->getParentOp();
      if (!parentOp) {
        continue;
      }
      // like scf.yield
      if (auto branchTerminator =
              dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
        // Find the result of the parentOp corresponding to this value
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          if (op->getOperand(i) != v || i >= parentOp->getNumResults())
            continue;
          if (checkUsersAnyWithCondition(parentOp->getResult(i), rootOp, condFn,
                                         skipFn, flag))
            return true;
        }
      }

      continue;
    }

    // For all skipped ops, just continue searching its result
    for (auto opRes : op->getResults()) {
      if (checkUsersAnyWithCondition(opRes, rootOp, condFn, skipFn, flag))
        return true;
    }
  }

  return false;
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

/// Create memref typed tmp buffer while avoids allocating dynamic shape.
inline Value createTmpMemrefWithTargetType(OpBuilder &builder, Location loc,
                                           Value src, Type targetElemType,
                                           ArrayRef<int64_t> targetShape) {
  assert(llvm::isa<MemRefType>(src.getType()) &&
         "Must be called with memref typed src");
  MemRefType memrefSrcType = llvm::cast<MemRefType>(src.getType());
  assert(memrefSrcType.getRank() == ssize_t(targetShape.size()));

  if (ShapedType::isDynamicShape(targetShape)) {
    // Not allowed to alloc a dynamic shaped buffer. Trace back src to its
    // AllocOp to get a static shape
    std::optional<memref::AllocOp> optAlloc =
        mlir::utils::tracebackMemRefToAlloc(src);
    assert(optAlloc.has_value() &&
           "Unable to trace dynamic shapped memref value to an alloc op");
    memref::AllocOp alloc = optAlloc.value();

    MemRefType tarMemRefType =
        cast<MemRefType>(alloc.getMemref().getType().clone(targetElemType));

    // clone alloc op with new element type and location
    Value fullBuffer = builder.create<memref::AllocOp>(
        loc, /*memrefType*/ tarMemRefType,
        /*dynamicSizes*/ alloc.getDynamicSizes(), alloc.getSymbolOperands(),
        alloc.getAlignmentAttr());

    // The buffer is larger than needed. Use subview to get the original shape
    auto makeRankSizeVector = [&targetShape, &builder](const int64_t value) {
      return SmallVector<OpFoldResult>(targetShape.size(),
                                       builder.getIndexAttr(value));
    };
    // Set sizes equal to target sizes
    SmallVector<OpFoldResult> sizes(targetShape.size());
    for (const auto &[idx, shape] : llvm::enumerate(targetShape)) {
      if (ShapedType::isDynamic(shape))
        sizes[idx] =
            builder.create<memref::DimOp>(loc, /*source*/ src, /*index*/ idx)
                .getResult();
      else
        sizes[idx] = builder.getIndexAttr(shape);
    }
    return builder.create<memref::SubViewOp>(
        loc, /*src*/ fullBuffer, /*offsets*/ makeRankSizeVector(0),
        /*sizes*/ sizes, /*strides*/ makeRankSizeVector(1));
  }
  // The memref type is used to create the alloc, so it shouldn't have any
  // layout.
  auto tarMemRefType =
      MemRefType::get(targetShape, targetElemType, MemRefLayoutAttrInterface{},
                      memrefSrcType.getMemorySpace());
  return builder.create<memref::AllocOp>(loc, /*memrefType*/ tarMemRefType,
                                         /*dynamicSizes*/ ValueRange({}));
}

/// Create tmp buffer or tensor using specified element type,
/// if targetElemType is null, then then use source's element type.
Value createTmpBufferOrTensorWithTargetType(
    OpBuilder &builder, Location loc, Value source,
    std::optional<Type> targetElemType,
    std::optional<SmallVector<int64_t>> targetShape) {
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

  ArrayRef<int64_t> shapeSize =
      targetShape.has_value() ? targetShape.value() : srcShapedType.getShape();

  Value tmp;
  if (isMemref) {
    tmp = createTmpMemrefWithTargetType(builder, loc, source,
                                        targetElemType.value(), shapeSize);
  } else {
    // As memref/tensor is already checked before, here wouldn't return failure.
    auto dynSizes = getTensorOrMemrefDynSizes(builder, loc, source, shapeSize);
    tmp = builder.create<tensor::EmptyOp>(loc, shapeSize,
                                          targetElemType.value(), *dynSizes);
  }
  return tmp;
}

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      })
      .Default([&](Type t) { return Value(); });
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
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b
            .create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                            strides)
            ->getResult(0);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b
            .create<memref::SubViewOp>(loc, source, offsets, sizes, strides)
            ->getResult(0);
      })
      .Default([&](Type t) { return nullptr; });
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

void setAlignUnits(const SmallVectorImpl<int> &alignTargets,
                   SmallVector<int> &alignUnits, ArrayRef<int64_t> shapes,
                   int innerAlignedUnits, int shapeAccumulation,
                   int alignTargetDim, int alignUnitsDim) {
  // The alignment target forces the INNER dimension to get aligned
  int newAlignedUnits =
      std::lcm(innerAlignedUnits, alignTargets[alignTargetDim]);
  if (newAlignedUnits == 0) {
    alignUnits.clear();
    return;
  }
  if (shapeAccumulation % newAlignedUnits == 0) {
    // already aligned
    alignUnits[alignUnitsDim] = 1;
  } else {
    if (innerAlignedUnits == 0) {
      alignUnits.clear();
      return; // should be impossible case (SecA_DivideByZero)
    }
    alignUnits[alignUnitsDim] = newAlignedUnits / innerAlignedUnits;
  }
  innerAlignedUnits = newAlignedUnits;
  if (!ShapedType::isDynamic(shapes[alignTargetDim])) {
    shapeAccumulation = shapeAccumulation * std::lcm(shapes[alignTargetDim],
                                                     alignUnits[alignUnitsDim]);
  }
}

ModuleOp getTopLevelModuleOp(Operation *op) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  while (moduleOp && moduleOp->getParentOp()) {
    auto spec = moduleOp->getAttrOfType<TargetSystemSpecAttr>(
        "dlti.target_system_spec");
    if (spec) {
      return moduleOp;
    }
    moduleOp = moduleOp->getParentOfType<ModuleOp>();
  }
  return moduleOp;
}

int64_t getArgumentIndex(Value value) {
  Block *block = nullptr;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    block = arg.getOwner();
  } else if (auto opResult = dyn_cast<OpResult>(value)) {
    block = opResult.getOwner()->getBlock();
  }

  if (!block)
    return -1;

  auto funcOp = dyn_cast<FunctionOpInterface>(block->getParentOp());
  if (!funcOp)
    return -1;

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() == &funcOp.getFunctionBody().front()) {
      return arg.getArgNumber();
    }
  }

  return -1;
}

bool hasBF16Operand(Operation *op) {
  OperandRange operands = op->getOperands();
  return std::any_of(operands.begin(), operands.end(), [&](Value oper) {
    return isa<BFloat16Type>(getElementTypeOrSelf(oper.getType()));
  });
}

bool isBefore(Operation *before, Operation *after) {
  if (before->getBlock() == after->getBlock()) {
    return before->isBeforeInBlock(after);
  }

  auto afterParentOp = after->getParentOp();
  if (afterParentOp == nullptr) {
    return false;
  }
  return isBefore(before, afterParentOp);
}

int64_t getNumPerBlock(Type t) {
  return INTR_BYTES_PER_BLOCK /
         (getElementTypeOrSelf(t).getIntOrFloatBitWidth() / INTR_BITS_PER_BYTE);
}

int64_t getNumPerRepeat(Type t) {
  return INTR_BYTES_PER_REPEAT /
         (getElementTypeOrSelf(t).getIntOrFloatBitWidth() / INTR_BITS_PER_BYTE);
}

// Gets a suitable vector size from the element type, assuming the type is
// supported by davinci metal. Vector size is fixed for Regbase, returns
// an integer representing a vector size,
int64_t getVectorSizeByElementType(Type t) {
  int factor = t.isInteger(64) ? 2 : 1;
  constexpr unsigned int vectorByteLength = 256;
  constexpr unsigned int byteSize = 8;
  return factor * vectorByteLength / ((int64_t)t.getIntOrFloatBitWidth() / byteSize);
}

template <bool DropUnitDimOnly>
static VectorType getLegalizedVectorType(VectorType source) {
  Type elemTy = source.getElementType();
  if constexpr (DropUnitDimOnly) {
    return hivm::util::trimNonScalableUnitDims(source);
  } else {
    return VectorType::get(
        SmallVector<int64_t>{utils::getVectorSizeByElementType(elemTy)},
        elemTy);
  }
}

template <bool DropUnitDimOnly>
static Value adjustVectorType(PatternRewriter &rewriter, VectorType resultTy,
                              Value src) {
  if constexpr (DropUnitDimOnly)
    // Use shape cast to drop unit dims to exploit the vector dialect fold
    // patterns
    return rewriter.create<vector::ShapeCastOp>(src.getLoc(), resultTy, src);

  // shape_cast cannot cast something like <1xf32> to <64xf32>
  return rewriter
      .create<UnrealizedConversionCastOp>(src.getLoc(), resultTy, src)
      .getResult(0);
}

template <bool DropUnitDimOnly>
LogicalResult ForOpLegalization<DropUnitDimOnly>::matchAndRewrite(
    scf::ForOp op, PatternRewriter &rewriter) const {
  // if the for op has a vector type iter_arg and the shape is not supported
  // by the hardware, we rewrite the shape
  OperandRange iterArgs = op.getInitArgs();
  SmallVector<Value> newIterArgs, newYields;
  SmallVector<unsigned> modified;
  for (unsigned i = 0; i < iterArgs.size(); i++) {
    if (op.getRegionIterArg(i).use_empty())
      continue;
    if (VectorType vecTy = dyn_cast<VectorType>(iterArgs[i].getType())) {
      VectorType adjustedType = getLegalizedVectorType<DropUnitDimOnly>(vecTy);

      if (vecTy.getShape().size() > 1 ||
          adjustedType.getNumElements() != vecTy.getNumElements()) {
        // need to adjust the iter arg
        // do this by making a new iter_arg of the supported type and replace
        // all use of the old iter arg with this new one. Leave the old one
        // for the canonicalizer to clean up.
        modified.push_back(i);

        rewriter.setInsertionPoint(op);
        Value adjustedIterArg = adjustVectorType<DropUnitDimOnly>(
            rewriter, adjustedType, iterArgs[i]);
        newIterArgs.push_back(adjustedIterArg);

        rewriter.setInsertionPoint(op.getBody()->getTerminator());
        Value adjustedYieldedValue = adjustVectorType<DropUnitDimOnly>(
            rewriter, adjustedType, op.getYieldedValues()[i]);
        newYields.push_back(adjustedYieldedValue);
      }
    }
  }

  if (newIterArgs.empty())
    return failure();

  rewriter.setInsertionPointAfter(op);
  NewYieldValuesFn fn =
      [&](OpBuilder &innerBuilder, Location loc,
          ArrayRef<BlockArgument> innerNewBBArgs) -> SmallVector<Value> {
    return newYields;
  };
  scf::ForOp newForOp = cast<scf::ForOp>(
      *op.replaceWithAdditionalYields(rewriter, newIterArgs, false, fn));

  int idx = 0;
  for (unsigned i = 0; i < iterArgs.size(); i++) {
    if (std::find(modified.begin(), modified.end(), i) != modified.end()) {
      rewriter.setInsertionPointAfter(newForOp);
      Value adjustedResult = adjustVectorType<DropUnitDimOnly>(
          rewriter, cast<VectorType>(newForOp.getResult(i).getType()),
          newForOp.getResult(iterArgs.size() + idx));
      rewriter.replaceAllUsesWith(newForOp.getResult(i), adjustedResult);
      rewriter.setInsertionPointToStart(newForOp.getBody());
      Value adjustedArg = adjustVectorType<DropUnitDimOnly>(
          rewriter, cast<VectorType>(newForOp.getRegionIterArg(i).getType()),
          newForOp.getRegionIterArg(iterArgs.size() + idx));
      rewriter.replaceAllUsesWith(newForOp.getRegionIterArg(i), adjustedArg);
      idx++;
    }
  }

  return success();
}

template struct utils::ForOpLegalization<true>;
template struct utils::ForOpLegalization<false>;

} // namespace utils

bool utils::isAllocLikeOp(Value val) {
  return isAllocLikeOp(val.getDefiningOp());
}

bool utils::isAllocLikeOp(Operation *op) {
  if (!op)
    return false;
  return isa<memref::AllocOp>(op) || isa<memref::AllocaOp>(op);
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

bool utils::isOneSizeShape(Value value) {
  if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    return llvm::all_of(shapedType.getShape(),
                        [](int64_t shape) { return shape == 1; });
  }
  return false;
}

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

bool utils::isArithOp(Operation *op) {
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

bool utils::isTransferWriteSuitForStoreWithStride(Operation *op) {
  if (!isa<vector::TransferWriteOp>(op)) {
    return false;
  }
  auto writeOp = cast<vector::TransferWriteOp>(op);
  auto memrefTy = mlir::dyn_cast<MemRefType>(writeOp.getSource().getType());
  LLVM_DEBUG(DBGS() << "transferOp: " << writeOp << "\n");
  if (!memrefTy) {
    return false;
  }
  ArrayRef<int64_t> shape = memrefTy.getShape();
  auto [strides, offset] = getStridesAndOffset(memrefTy);
  if (shape.size() < 2) {
    LLVM_DEBUG(DBGS() << "fail because of shape is 1");
    return false;
  }
  int64_t first_dim = shape[0];
  int64_t last_dim = shape[shape.size() - 1];
  if (first_dim > hivm::util::BLOCK_NUM_PER_VL) {
    LLVM_DEBUG(DBGS() << "fail becasuse of first dim is above 8\n");
    return false;
  }
  if (first_dim == 1) {
    LLVM_DEBUG(
        DBGS()
        << "fail becasuse of first dim is 1, only one block no need vsstb\n");
    return false;
  }
  // check if all the dims between first dim and last dim are 1.
  if (llvm::any_of(llvm::make_range(shape.begin() + 1, shape.end() - 1),
                   [](const auto &dim) { return dim != 1; })) {
    LLVM_DEBUG(DBGS() << "\n fail because of shape dim not 1\n");
    return false;
  }

  // support I8, I16, I32, BF16, F16, F32, F8E4M3FN, F8E5M2
  auto elementType = memrefTy.getElementType();
  bool isSupportedFloat = elementType.isFloat8E4M3FN() ||
                          elementType.isFloat8E5M2() || elementType.isBF16() ||
                          elementType.isF16() || elementType.isF32();
  bool isSupportedInt = elementType.isInteger(32) ||
                        elementType.isInteger(16) || elementType.isInteger(8);
  if (!isSupportedFloat && !isSupportedInt) {
    return false;
  }

  // check if last dim size matches one block which is 256bit.
  auto expectedDim =
      hivm::util::vectorBlockSizeBit / elementType.getIntOrFloatBitWidth();
  if (expectedDim != last_dim) {
    LLVM_DEBUG(DBGS() << "last dim not support\n");
    return false;
  }
  int64_t productLowerDims =
      std::accumulate(shape.begin() + 1, shape.end(), static_cast<int64_t>(1),
                      [](int64_t acc, const auto &dim) { return acc * dim; });
  if (strides[0] == productLowerDims) {
    LLVM_DEBUG(DBGS() << "fail becasuse of stride is continous\n");
    return false;
  }
  LLVM_DEBUG(DBGS() << " matched!! for transferWriteWithStride\n");
  return true;
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

void utils::dumpReassociationIndicesVector(
    const SmallVector<ReassociationIndices> &reassocVec) {
  for (size_t i = 0; i < reassocVec.size(); ++i) {
    std::string name = "reassocVec";
    name += "[" + std::to_string(i) + "]";
    llvm::dbgs() << name << reassocVec[i] << "\n";
  }
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
          linalg::MatmulTransposeBOp, linalg::BatchMatmulOp, tensor::ExtractOp>(
          op)) {
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

/**
 * @brief Generate reassociation indices for ExpandShape/CollapseShape ops
 * @details Create nested reassociation vectors by grouping contiguous expand
 * dims (unit dims) with subsequent non-expand dims in the output rank sequence
 *
 * Reassociation Transformation Examples:
 * - Case 1: expandDims = [1], outRank = 3
 *   Input:  expandDims={1}, outRank=3 (dim 1 is expand dim)
 *   Output: [[0], [1,2]] (group expand dim 1 with non-expand dim 2)
 *
 * - Case 2: expandDims = [0,1], outRank = 4
 *   Input:  expandDims={0,1}, outRank=4 (dims 0/1 are expand dims)
 *   Output: [[0,1,2], [3]] (group leading expand dims 0/1 with dim 2, then dim
 * 3 alone)
 *
 * - Case 3: expandDims = [], outRank = 3
 *   Input:  no expand dims, outRank=3
 *   Output: [[0], [1], [2]] (each dim as independent group)
 *
 * @param[in] expandDims Array of indices marking expand/unit dimensions in
 * output rank
 * @param[in] outRank Total rank (dimension count) of the target shape
 * @return SmallVector<SmallVector<int64_t, 2>> Nested reassociation indices for
 * shape ops
 * @note 1. Expand dims are contiguous leading dims in the loop logic
 *       2. Non-empty residual groups are added to the result at final check
 *       3. Reassociation groups follow MLIR ExpandShape/CollapseShape op
 * conventions
 */
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

bool isConstIntOne(Value v) {
  auto type = getElementTypeOrSelf(v);
  if (type.isIntOrIndex()) {
    if (matchPattern(v, m_One())) {
      return true;
    }
  }
  return false;
}

SmallVector<int64_t> getSqueezedShape(SmallVectorImpl<int64_t> &shape) {
  SmallVector<int64_t> newShape;
  for (int64_t dimSize : shape) {
    if (dimSize != 1) {
      newShape.push_back(dimSize);
    }
  }
  // We do not allow empty shape which means rank 0 shaped value
  if (newShape.empty()) {
    newShape.push_back(1);
  }
  return newShape;
}

std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (ofr.is<Attribute>() && isa<IntegerAttr>(ofr.get<Attribute>()))
    return dyn_cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
  return std::nullopt;
}

OpFoldResult mulOFRs(const OpFoldResult lhs, const OpFoldResult rhs,
                     OpBuilder &b, const Location loc) {
  auto lhsIntAttr = getIntAttr(lhs);
  auto rhsIntAttr = getIntAttr(rhs);

  // both lhs and rhs are constants, return result directly
  if (lhsIntAttr && rhsIntAttr)
    return b.getIndexAttr(lhsIntAttr.value() * rhsIntAttr.value());

  // shortcuts for special cases
  if (lhsIntAttr) {
    if (lhsIntAttr.value() == 0)
      return lhs;
    if (lhsIntAttr.value() == 1)
      return rhs;
  }
  if (rhsIntAttr) {
    if (rhsIntAttr.value() == 0)
      return rhs;
    if (rhsIntAttr.value() == 1)
      return lhs;
  }

  // otherwise, need to create instructions to calculate new attribute value
  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsIntAttr) {
    auto lhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(lhsIntAttr.value()));
    lhsValue = lhsOp.getResult();
  }

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsIntAttr) {
    auto rhsOp =
        b.create<arith::ConstantOp>(loc, b.getIndexAttr(rhsIntAttr.value()));
    rhsValue = rhsOp.getResult();
  }

  auto mulOp = b.create<arith::MulIOp>(loc, lhsValue, rhsValue);
  return mulOp.getResult();
}

void shrinkReassocIdxByDroppedDims(
    SmallVector<ReassociationIndices> &reassocIdxVec,
    llvm::SmallBitVector &droppedDims) {
  size_t rank = droppedDims.size();
  SmallVector<int64_t> shiftTable(rank, 0);
  shiftTable[0] = (droppedDims.test(0) ? 1 : 0);
  for (size_t i = 1; i < rank; ++i) {
    shiftTable[i] = shiftTable[i - 1] + (droppedDims.test(i) ? 1 : 0);
  }
  for (auto it = reassocIdxVec.begin(); it != reassocIdxVec.end();) {
    auto &reassoc = *it;
    size_t writePos = 0;
    for (size_t readPos = 0; readPos < reassoc.size(); ++readPos) {
      int64_t originalIdx = reassoc[readPos];
      if (!droppedDims.test(originalIdx)) {
        reassoc[writePos] = originalIdx - shiftTable[originalIdx];
        ++writePos;
      }
    }
    reassoc.resize(writePos);
    if (reassoc.empty()) {
      it = reassocIdxVec.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace reshape_utils

bool utils::isReduceWithIndex(hivm::ReduceOperation op) {
  return op == hivm::ReduceOperation::min_with_index ||
         op == hivm::ReduceOperation::max_with_index;
}

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

namespace mlir {

namespace triton {
namespace util {

int getPassColumnDigit(Operation *opCtx, llvm::StringRef passName) {
  // get module
  ModuleOp module = nullptr;
  if (isa<ModuleOp>(opCtx) || opCtx->hasAttr(hacc::SIMTModuleAttr::name)) {
    module = cast<ModuleOp>(opCtx);
  } else {
    module = opCtx->getParentOfType<ModuleOp>();
  }

  if (!module) {
    if (auto maybeModule = llvm::dyn_cast_or_null<ModuleOp>(opCtx))
      module = maybeModule;
    else {
      llvm::errs()
          << "[getPassColumnDigit] Warning: enclosing ModuleOp not found\n";
      return 0;
    }
  }

  // get attribute
  Attribute rawAttr = module->getAttr(AttrEnableBishengirSimtOptimizationName);
  if (!rawAttr) {
    return 0;
  }
  std::string digitStr;
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(rawAttr)) {
    llvm::SmallString<32> buf;
    intAttr.getValue().toString(buf, /*Radix=*/10, /*Signed=*/false);
    digitStr.assign(buf.begin(), buf.end());
  } else if (auto strAttr = mlir::dyn_cast<StringAttr>(rawAttr)) {
    digitStr = strAttr.getValue().str();
  } else {
    return 0;
  }

  if (digitStr.empty()) {
    return 0;
  }

  // TODO: add here to control your new passes
  // idx =  0 (ones),  1 (tens),  2 (hundreds)......
  int idx = 0;
  if (passName == "decompose-reduction") {
    // decompose reduction pass is now non-optional
    // the index is kept for backward compatibility
    idx = 0;
  } else if (passName == "optimize-layouts")
    idx = 1;
  else if (passName == "convert-triton-gpu-to-llvm")
    idx = 2;
  else if (passName == "reduce-op")
    idx = 3;
  else if (passName == "optimize-loads")
    idx = 4;
  else if (passName == "loop-restructure-arange-optimization")
    idx = 5;
  else {
    idx = 0;
  }

  if (static_cast<size_t>(idx) >= digitStr.size())
    return 0;

  size_t pos = digitStr.size() - 1 - static_cast<size_t>(idx);
  char c = digitStr[pos];
  return static_cast<int>(c - '0');
}

} // namespace util
} // namespace triton

} // namespace mlir
