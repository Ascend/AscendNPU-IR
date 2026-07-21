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
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRef/IR/MemRefImpl.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#if (!BISHENGIR_BUILD_STANDALONE_IR_ONLY)
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#endif // BISHENGIR_BUILD_STANDALONE_IR_ONLY
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/STLExtras.h"
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
      if (auto *tiedLoopInit = whileOp.getTiedLoopInit(arg)) {
        result.emplace_back(tiedLoopInit->get());
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

void setAlignUnits(const SmallVectorImpl<int> &alignTargets,
                   SmallVector<int> &alignUnits, ArrayRef<int64_t> shapes,
                   int &innerAlignedUnits, int &shapeAccumulation,
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

bool isReduceWithIndex(hivm::ReduceOperation op) {
  return op == hivm::ReduceOperation::min_with_index_left ||
         op == hivm::ReduceOperation::min_with_index_right ||
         op == hivm::ReduceOperation::max_with_index_left ||
         op == hivm::ReduceOperation::max_with_index_right;
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

Value createEmptyOpWithTargetElemType(
    OpBuilder &builder, Location loc, Value source, Type targetElemType,
    std::optional<MemRefLayoutAttrInterface> layout) {
  auto shapedType = cast<ShapedType>(source.getType());
  if (isa<TensorType>(shapedType)) {
#if BISHENGIR_BUILD_STANDALONE_IR_ONLY
    llvm::report_fatal_error("Not implemented");
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
    llvm::report_fatal_error("Not implemented");
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

SmallVector<Value> getAliasValues(Operation *op, OpOperand &operand) {
  SmallVector<Value> values;
  auto resNum = operand.getOperandNumber();
  TypeSwitch<Operation *>(op)
      .Case<scf::IfOp>(
          [&](scf::IfOp ifOp) { values.push_back(ifOp.getResult(resNum)); })
      .Case<scf::ForOp>([&](scf::ForOp forOp) {
        values.push_back(forOp.getResult(resNum));
        values.push_back(forOp.getRegionIterArg(resNum));
      })
      .Case<scf::WhileOp>([&](scf::WhileOp whileOp) {
        values.push_back(whileOp.getResult(resNum));
        values.push_back(whileOp.getBefore().front().getArgument(resNum));
        values.push_back(whileOp.getAfter().front().getArgument(resNum));
      })
      .Default([&](Operation *otherOp) {
        llvm::report_fatal_error("unsupported loop like op!");
      });

  return values;
}

std::optional<bool>
checkUsersAllWithCondition(Value v, Operation *rootOp,
                           const std::function<bool(Operation *op)> &condFn,
                           const std::function<bool(Operation *op)> &skipFn) {
  /*
  A static visited set is needed to solve an infinite recursion issue that could
  happen when loadOp::inferCoreType and checkUsersAllWithCondition keep calling
  each other.
  TODO: rewrite loadOp::inferCoreType so we don't use a static visited set.
   */
  static DenseSet<Value> visited;
  if (!visited.insert(v).second)
    return std::nullopt;
  auto returnedValue =
      checkUsersAllWithConditionImpl(v, rootOp, condFn, skipFn);
  visited.erase(v);
  return returnedValue;
}

std::optional<bool> checkUsersAllWithConditionImpl(
    Value v, Operation *rootOp,
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
    if (isa<scf::YieldOp, scf::ConditionOp>(op)) {
      Operation *parentOp = op->getParentOp();
      assert(parentOp && "parent op cannot be nullptr");
      auto aliasValues = getAliasValues(parentOp, use);
      SmallVector<std::optional<bool>> coreTypes;
      llvm::transform(
          aliasValues, std::back_inserter(coreTypes), [&](Value &v) {
            return checkUsersAllWithCondition(v, rootOp, condFn, skipFn);
          });

      if (llvm::all_of(coreTypes, [&](std::optional<bool> &maybeCoreType) {
            return !maybeCoreType.has_value();
          }))
        continue;

      if (llvm::any_of(coreTypes, [&](std::optional<bool> &maybeCoreType) {
            return maybeCoreType.has_value() && !maybeCoreType.value();
          }))
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

static void
tracebackOperandsToBlockArgumentsImpl(Value value, DenseSet<Value> &visited,
                                      SmallVectorImpl<BlockArgument> &result,
                                      Block *block) {

  if (!visited.insert(value).second) {
    return;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getParentBlock() == block) {
      result.push_back(blockArg);
      return;
    }
  }

  if (auto *defOp = value.getDefiningOp()) {
    for (auto operand : defOp->getOperands()) {
      tracebackOperandsToBlockArgumentsImpl(operand, visited, result, block);
    }
  }
}

SmallVector<BlockArgument> tracebackOperandsToBlockArguments(Value value,
                                                             Block *block) {
  DenseSet<Value> visited;
  SmallVector<BlockArgument> result;
  tracebackOperandsToBlockArgumentsImpl(value, visited, result, block);
  return result;
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
                      [](OpFoldResult x) { return isa<Attribute>(x); });
}

inline void markDynShapeAlloc(OpBuilder &builder, Value source,
                              memref::AllocOp &tmpAllocOp,
                              bool allowDynShapeAlloc) {
  auto srcAlloc = utils::tracebackMemRefToAlloc(source);
  if (!srcAlloc.has_value()) {
    if (allowDynShapeAlloc)
      return;
    emitError(tmpAllocOp->getLoc(), "alloc is not found");
    llvm::report_fatal_error("alloc is not found");
    return;
  }
  auto srcAllocMemref = srcAlloc.value().getMemref();
  auto elemType = getElementTypeOrSelf(tmpAllocOp.getMemref());
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
  auto sourceElemTypeWidth =
      getElementTypeOrSelf(source.getType()).getIntOrFloatBitWidth();
  auto srcAllocElemTypeWidth =
      getElementTypeOrSelf(srcAllocMemref).getIntOrFloatBitWidth();
  allocSize = allocSize * srcAllocElemTypeWidth / sourceElemTypeWidth;
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
    llvm::report_fatal_error("Cannot get dim for type with no rank");
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

bool isAlignedInUB(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    auto shape = shapedType.getShape();
    const int64_t lastDimSizeInBit =
        shape.back() * static_cast<int64_t>(
            shapedType.getElementType().getIntOrFloatBitWidth());
    return lastDimSizeInBit % kUBAlignSizeInBits == 0;
  }
  return type.getIntOrFloatBitWidth() % kUBAlignSizeInBits == 0;
}

bool isUnstructuredMemAccLoop(Operation *op) {
  auto forOp = dyn_cast<scf::ForOp>(op);
  if (!forOp)
    return false;
  return forOp->hasAttr("ExtractedLoadOrStore");
}

int64_t getNumPerRepeat(Type t) {
  unsigned tBits = getElementTypeOrSelf(t).getIntOrFloatBitWidth();
  unsigned tBytes = llvm::divideCeil(tBits, INTR_BITS_PER_BYTE);
  return INTR_BYTES_PER_REPEAT / tBytes;
}

template <bool DropUnitDimOnly>
static VectorType getLegalizedVectorType(VectorType source) {
  Type elemTy = source.getElementType();
  if constexpr (DropUnitDimOnly)
    return hivm::util::trimNonScalableUnitDims(source);
  return VectorType::get(
      SmallVector<int64_t>{utils::getVectorSizeByElementType(elemTy)}, elemTy);
}

template <bool DropUnitDimOnly>
static Value adjustVectorType(PatternRewriter &rewriter, VectorType resultTy,
                              Value src) {
  if constexpr (DropUnitDimOnly)
    return rewriter.create<vector::ShapeCastOp>(src.getLoc(), resultTy, src);
  return rewriter
      .create<UnrealizedConversionCastOp>(src.getLoc(), resultTy, src)
      .getResult(0);
}

template <bool DropUnitDimOnly>
LogicalResult ForOpLegalization<DropUnitDimOnly>::matchAndRewrite(
    scf::ForOp op, PatternRewriter &rewriter) const {
  OperandRange iterArgs = op.getInitArgs();
  SmallVector<Value> newIterArgs, newYields;
  SmallVector<unsigned> modified;
  for (unsigned i = 0; i < iterArgs.size(); i++) {
    if (op.getRegionIterArg(i).use_empty())
      continue;
    if (auto vecTy = dyn_cast<VectorType>(iterArgs[i].getType())) {
      VectorType adjustedType = getLegalizedVectorType<DropUnitDimOnly>(vecTy);
      if (vecTy.getShape().size() > 1 ||
          adjustedType.getNumElements() != vecTy.getNumElements()) {
        modified.push_back(i);
        rewriter.setInsertionPoint(op);
        newIterArgs.push_back(adjustVectorType<DropUnitDimOnly>(
            rewriter, adjustedType, iterArgs[i]));
        rewriter.setInsertionPoint(op.getBody()->getTerminator());
        newYields.push_back(adjustVectorType<DropUnitDimOnly>(
            rewriter, adjustedType, op.getYieldedValues()[i]));
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

int64_t utils::getNumPerBlock(Type t) {
  // Multiply first so i1 (bitWidth=1) does not truncate 1/8 to 0 and divide by
  // zero. Equivalent to 32 / (bitWidth/8) when bitWidth is a multiple of 8.
  return (utils::INTR_BYTES_PER_BLOCK * utils::INTR_BITS_PER_BYTE) /
         getElementTypeOrSelf(t).getIntOrFloatBitWidth();
}

bool utils::isAllocLikeOp(Value val) {
  return isAllocLikeOp(val.getDefiningOp());
}

bool utils::isAllocLikeOp(Operation *op) {
  if (!op)
    return false;
  return isa<memref::AllocOp>(op) || isa<memref::AllocaOp>(op) ||
         isa<bishengir::memref_ext::AllocWorkspaceOp>(op);
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
  memref::ViewOp viewOp;
  auto startOffset = opBuilder.create<arith::ConstantIndexOp>(loc, 0);
  if (isa<memref::AllocOp>(op)) {
    memref::AllocOp oldOp = cast<memref::AllocOp>(op);
    newAlloc = opBuilder
                   .create<memref::AllocOp>(loc, newMemrefType,
                                            oldOp.getAlignmentAttr())
                   .getMemref();
    viewOp = opBuilder.create<memref::ViewOp>(loc, oldType, newAlloc,
                                              startOffset, op->getOperands());
  } else if (isa<memref::AllocaOp>(op)) {
    memref::AllocaOp oldOp = cast<memref::AllocaOp>(op);
    newAlloc = opBuilder
                   .create<memref::AllocaOp>(loc, newMemrefType,
                                             oldOp.getAlignmentAttr())
                   .getMemref();
    viewOp = opBuilder.create<memref::ViewOp>(loc, oldType, newAlloc,
                                              startOffset, op->getOperands());
  } else if (isa<bishengir::memref_ext::AllocWorkspaceOp>(op)) {
    bishengir::memref_ext::AllocWorkspaceOp oldOp =
        cast<bishengir::memref_ext::AllocWorkspaceOp>(op);
    newAlloc = opBuilder
                   .create<bishengir::memref_ext::AllocWorkspaceOp>(
                       loc, newMemrefType, oldOp.getWorkspaceArg(),
                       ValueRange{}, /*offset*/ ValueRange{})
                   .getMemref();
    viewOp =
        opBuilder.create<memref::ViewOp>(loc, oldType, newAlloc, startOffset,
                                         ValueRange{oldOp.getDynamicSize()});
  }
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

int64_t utils::getVectorSizeByElementType(Type t) {
  int factor = t.isInteger(64) ? 2 : 1;
  constexpr unsigned int vectorByteLength = 256;
  constexpr unsigned int byteSize = 8;
  return factor * vectorByteLength /
         ((int64_t)t.getIntOrFloatBitWidth() / byteSize);
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
  if (auto allocOp = tracedValue.getDefiningOp<memref::AllocOp>())
    return allocOp;
  return std::nullopt;
}

std::optional<Value>
utils::tracebackMemRefToAllocOrBlockArgument(Value memrefVal) {
  auto tracedValue = utils::tracebackMemRef(memrefVal);
  return utils::isAllocLikeOp(tracedValue) || isa<BlockArgument>(tracedValue)
             ? tracedValue
             : std::optional<Value>();
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
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  return isa_and_present<linalg::ElemwiseBinaryOp, linalg::ElemwiseUnaryOp,
                         linalg::FillOp>(op);
#else
  return isa_and_present<linalg::FillOp>(op) || isElementwiseOp(op);
#endif
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
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  return isa_and_present<linalg::ElemwiseUnaryOp, linalg::FillOp>(op);
#else
  if (isa_and_present<linalg::FillOp>(op)) {
    return true;
  }
  if (isElementwiseOp(op)) {
    auto genericOp = dyn_cast<linalg::LinalgOp>(op);
    return genericOp.getNumDpsInputs() == 1;
  }
  return false;
#endif
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
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
          linalg::ElemwiseBinaryOp, linalg::ElemwiseUnaryOp,
#endif
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

int64_t utils::getArgumentIndex(Value value) {
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

namespace utils {

bool isValidHIVMTileElementType(Type type) {
  return type.isInteger(1) || type.isInteger(8) || type.isInteger(16) ||
         type.isInteger(32) || type.isInteger(64) || type.isF16() ||
         type.isF32() || type.isBF16() || type.isFloat8E4M3FN() ||
         type.isFloat8E5M2();
}

unsigned getHIVMTileSliceMinNumElts(Type type) {
  assert(isValidHIVMTileElementType(type) && "invalid tile type!");
  unsigned factor = type.isInteger(64) ? 2 : 1;
  return factor * hivm::util::VL_BITS / type.getIntOrFloatBitWidth();
}

bool isValidHIVMTileVectorType(VectorType vType) {
  if (!hivm::util::isOneDimLikeVecType(vType))
    return false;

  auto elemType = vType.getElementType();
  if (!isValidHIVMTileElementType(elemType))
    return false;

  unsigned minNumElts = getHIVMTileSliceMinNumElts(elemType);
  if (vType.getNumElements() > minNumElts)
    return false;

  return true;
}

bool isValidTwoDimVectorType(VectorType vType) {
  if (vType.getRank() < 2)
    return false;
  auto shape = vType.getShape();
  for (int64_t i = 0, e = vType.getRank() - 2; i < e; ++i) {
    if (shape[i] != 1)
      return false;
  }

  auto elemType = vType.getElementType();
  if (!isValidHIVMTileElementType(elemType))
    return false;

  unsigned minNumElts = getHIVMTileSliceMinNumElts(elemType);
  if (vType.getNumElements() > minNumElts)
    return false;

  return true;
}

} // namespace utils

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


void utils::dumpReassociationIndicesVector(
    const SmallVector<ReassociationIndices> &reassocVec) {
  for (size_t i = 0; i < reassocVec.size(); ++i) {
    std::string name = "reassocVec";
    name += "[" + std::to_string(i) + "]";
    llvm::dbgs() << name << reassocVec[i] << "\n";
  }
}
} // namespace mlir

namespace mlir {

namespace triton {
namespace util {
// Create a private llvm.func declaration
LLVM::LLVMFuncOp createLLVMFuncDecl(OpBuilder &b, Location loc, StringRef name,
                                    LLVM::LLVMFunctionType fnTy) {
  auto decl = b.create<LLVM::LLVMFuncOp>(loc, name, fnTy);
  auto haccAlwaysInlineAttr = hacc::stringifyHACCToLLVMIRTranslateAttr(
      hacc::HACCToLLVMIRTranslateAttr::ALWAYS_INLINE);
  decl->setAttr(haccAlwaysInlineAttr, b.getUnitAttr());
  decl->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), b.getUnitAttr());
  decl->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                b.getStringAttr("private"));
  MLIRContext *ctx = decl.getContext();
  decl->setAttr(mlir::hivm::TFuncCoreTypeAttr::name,
                hivm::TFuncCoreTypeAttr::get(ctx, hivm::TFuncCoreType::AIV));
  return decl;
}

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

Value allocateSharedMemory(LLVM::LLVMFuncOp vf, OpBuilder &builder,
                           Location loc) {
  // Try to find the base address from the arguments at first. If it is not
  // exist, the base address of the shared memory starts from 0
  Value sharedMemAddr = nullptr;
  for (auto [idx, value] : llvm::enumerate(vf.getArguments())) {
    auto dictAttr = vf.getArgAttrDict(idx);
    if (!dictAttr)
      continue;

    if (dictAttr.get(hivm::SharedMemoryAttr::name)) {
      sharedMemAddr = value;
      break;
    }
  }

  if (!sharedMemAddr) {
    Value c0 = builder.create<LLVM::ConstantOp>(
        vf->getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(0));
    auto intToPtr = builder.create<LLVM::IntToPtrOp>(
        vf->getLoc(), LLVM::LLVMPointerType::get(builder.getContext(), 6), c0);
    intToPtr->setAttr(hivm::SharedMemoryAttr::name, builder.getUnitAttr());
    sharedMemAddr = intToPtr;
  }

  return sharedMemAddr;
}

} // namespace util
} // namespace triton

} // namespace mlir