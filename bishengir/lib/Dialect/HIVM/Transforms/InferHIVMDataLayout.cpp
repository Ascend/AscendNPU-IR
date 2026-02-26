//===- InferHIVMDataLayout.cpp - Infer Data Layout for HIVM Ops -----------===//
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
#include "bishengir/Dialect/HIVM/Transforms/InferHIVMDataLayout.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>

#define DEBUG_TYPE "hivm-infer-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_INFERHIVMDATALAYOUT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

/// Mapping from <SrcLayout, DstLayout> to LayoutConversionKind.
const std::map<std::pair<hivm::DataLayout, hivm::DataLayout>,
               LayoutConversionKind>
    kSupportedConversion = {
        {{hivm::DataLayout::DOTA_ND, hivm::DataLayout::zN},
         LayoutConversionKind::DOT_ND_TO_zN},
        {{hivm::DataLayout::DOTB_ND, hivm::DataLayout::zN},
         LayoutConversionKind::DOT_ND_TO_zN},
        {{hivm::DataLayout::DOTA_ND, hivm::DataLayout::nZ},
         LayoutConversionKind::DOT_ND_TO_nZ},
        {{hivm::DataLayout::DOTB_ND, hivm::DataLayout::nZ},
         LayoutConversionKind::DOT_ND_TO_nZ},
        {{hivm::DataLayout::DOTC_ND, hivm::DataLayout::zN},
         LayoutConversionKind::DOT_ND_TO_zN},
        {{hivm::DataLayout::ND, hivm::DataLayout::zN},
         LayoutConversionKind::ND_TO_zN},
        {{hivm::DataLayout::ND, hivm::DataLayout::nZ},
         LayoutConversionKind::ND_TO_nZ},
        {{hivm::DataLayout::nZ, hivm::DataLayout::ND},
         LayoutConversionKind::nZ_TO_ND},
        {{hivm::DataLayout::zN, hivm::DataLayout::ND},
         LayoutConversionKind::zN_TO_ND},
};

inline bool isGlobalMemory(Value val) {
  std::optional<AddressSpaceAttr> maybeSpaceAttr = GetBufferSpaceAttr(val);
  return maybeSpaceAttr.has_value() &&
         maybeSpaceAttr->getAddressSpace() == AddressSpace::GM;
}

void convertToBatchND2NZOp(Value src, Value dst, OpBuilder &builder) {
  auto buildLoopBody = [&src, &dst,
                        &builder](llvm::SmallVector<Value> indexes) -> void {
    auto getSub = [&builder, &indexes](Value val) -> Value {
      auto valType = llvm::dyn_cast<ShapedType>(val.getType());

      SmallVector<OpFoldResult> subOffsets(valType.getRank(),
                                           builder.getIndexAttr(0));
      subOffsets[0] = indexes[0];
      auto shape = getValueFromShape(val, builder);
      assert(succeeded(shape));
      SmallVector<OpFoldResult> subSizes = llvm::map_to_vector(
          *shape, [](Value val) { return OpFoldResult(val); });
      subSizes[0] = builder.getIndexAttr(1);
      SmallVector<OpFoldResult> subStrides(valType.getRank(),
                                           builder.getIndexAttr(1));
      auto subviewOp = builder.create<memref::SubViewOp>(
          val.getLoc(), val, subOffsets, subSizes, subStrides);
      SmallVector<ReassociationIndices> reassociation;
      reassociation.push_back({0, 1});
      for (int i = 2; i < valType.getRank(); ++i)
        reassociation.push_back({i});
      auto collapseOp = builder.create<memref::CollapseShapeOp>(
          val.getLoc(), subviewOp.getResult(), reassociation);
      return collapseOp.getResult();
    };

    auto subSrc = getSub(src);
    auto subDst = getSub(dst);

    builder.create<hivm::ND2NZOp>(src.getLoc(), TypeRange{},
                                  /*src=*/subSrc, /*dst=*/subDst,
                                  builder.getUnitAttr());
  };
  std::set<int> loopDims;
  loopDims.insert(0);
  createNestedLoops(builder, src.getLoc(), src, loopDims, buildLoopBody);
}

void convertToND2NZOp(Value src, Value dst, Operation *originalOp,
                      OpBuilder &builder) {
  mlir::Value padValue = nullptr;
  mlir::Value initCondition = nullptr;
  bool hasInitOutBuffer = false;
  if (auto loadOp = dyn_cast_if_present<hivm::LoadOp>(originalOp)) {
    hasInitOutBuffer = loadOp.getInitOutBuffer();
    padValue = hasInitOutBuffer ? loadOp.getPadValue() : nullptr;
    initCondition = loadOp.getInitCondition();
  }
  builder.create<hivm::ND2NZOp>(src.getLoc(), TypeRange{}, src, dst,
                                builder.getUnitAttr(), hasInitOutBuffer,
                                padValue, initCondition);
}

} // namespace

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//
DataLayoutAttr DataLayoutInferAndPropagateHelper::getTargetLayout(Value value) {
  return layout_info_.lookup(value).targetLayout;
}

DataLayoutAttr
DataLayoutInferAndPropagateHelper::getCurrentLayout(Value value) {
  return layout_info_.lookup(value).currentLayout;
}

void DataLayoutInferAndPropagateHelper::map(Value oldValue, Value newValue,
                                            DataLayoutAttr newLayout) {
  rewriteMappingWithLayout_[{oldValue, newLayout}] = newValue;
  (void)updateLayoutIfChanged(newValue, LayoutInfo{newLayout, newLayout});
}

/// Retrieves a value converted to the requested layout.
/// If the value's current layout differs from the requested layout, this
/// function looks up the rewritten value or creates a ConvertLayoutOp.
/// @param value The original value.
/// @param layout The desired data layout.
/// @return The value with the requested layout, or the original value if
///         conversion is not possible or not needed.
Value DataLayoutInferAndPropagateHelper::getValueAs(Value value,
                                                    DataLayoutAttr layout) {
  if (!layout) {
    return value;
  }
  if (!isa<BaseMemRefType>(value.getType())) {
    return value;
  }
  auto layoutIt = layout_info_.find(value);
  // We cannot determine current value's layout, skip.
  if (layoutIt == layout_info_.end()) {
    return value;
  }
  LayoutInfo info = layoutIt->second;
  // If current value's layout is already the requested layout, return.
  // TODO: might wanna check the fractal sizes(?)
  if (info.currentLayout.getDataLayout() == layout.getDataLayout()) {
    return value;
  }
  // Get current value's mapped to its target layout.
  auto rewriteIter = rewriteMappingWithLayout_.find({value, info.targetLayout});
  assert(rewriteIter != rewriteMappingWithLayout_.end());
  Value rewrittenValue = rewriteIter->second;
  // If the rewritten value has the requested layout, return.
  if (info.targetLayout == layout) {
    return rewrittenValue;
  }
  // Otherwise, try to create a ConvertLayoutOp between the rewritten value
  // and the requested layout.
  OpBuilder rewriter(value.getContext());
  rewriter.setInsertionPointAfterValue(value);
  FailureOr<ConvertLayoutOp> status = createLayoutConversion(
      rewrittenValue, {info.targetLayout, layout}, rewriter);
  if (failed(status)) {
    return value;
  }
  return (*status).getResult();
}

bool DataLayoutInferAndPropagateHelper::isConversionValid(
    const LayoutInfo &info) {
  auto currentLayout = info.currentLayout.getDataLayout();
  auto targetLayout = info.targetLayout.getDataLayout();
  auto it =
      kSupportedConversion.find(std::make_pair(currentLayout, targetLayout));
  return it != kSupportedConversion.cend();
}

FailureOr<SmallVector<Value>>
DataLayoutInferAndPropagateHelper::computeTargetLayoutShape(
    SmallVector<Value> currentShape, const LayoutInfo &info, OpBuilder &builder,
    Location loc) {
  if (info.noLayoutConflict())
    return currentShape;
  if (!isConversionValid(info))
    return failure();

  auto conversionKind = kSupportedConversion.at(std::make_pair(
      info.currentLayout.getDataLayout(), info.targetLayout.getDataLayout()));
  SmallVector<int64_t> kBlockSizes;
  auto fractalSizes = info.targetLayout.getFractalSizesArray();
  if (fractalSizes.has_value()) {
    for (int64_t kBlockSize : *fractalSizes) {
 
      kBlockSizes.push_back(kBlockSize);
    }
  }
  std::optional<bool> srcTransposeInfo = info.currentLayout.getTransposeValue();
  SmallVector<Value> transCurrentShape(currentShape);
  if (srcTransposeInfo.has_value() && srcTransposeInfo.value()) {
    auto srcRank = transCurrentShape.size();
    assert(srcRank >= 2);
    transCurrentShape[srcRank - 1] = currentShape[srcRank - 2];
    transCurrentShape[srcRank - 2] = currentShape[srcRank - 1];
  }

  switch (conversionKind) {
  case LayoutConversionKind::DOT_ND_TO_zN:
  case LayoutConversionKind::ND_TO_zN:
    return computeDOTNDToFractalzNShape(transCurrentShape, builder, loc,
                                        kBlockSizes);
  case LayoutConversionKind::DOT_ND_TO_nZ:
  case LayoutConversionKind::ND_TO_nZ:
    return computeDOTNDToFractalnZShape(transCurrentShape, builder, loc,
                                        kBlockSizes);
  default:
    return failure();
  }
}

FailureOr<SmallVector<Value>>
DataLayoutInferAndPropagateHelper::computeTargetLayoutOffset(
    SmallVector<Value> currentOffset, const LayoutInfo &info,
    OpBuilder &builder, Location loc) {
  if (!isConversionValid(info))
    return failure();
  auto conversionKind = kSupportedConversion.at(std::make_pair(
      info.currentLayout.getDataLayout(), info.targetLayout.getDataLayout()));
  SmallVector<int64_t> kBlockSizes;
  auto fractalSizes = info.targetLayout.getFractalSizesArray();
  if (fractalSizes.has_value()) {
    for (int64_t kBlockSize : fractalSizes.value()) {
 
      kBlockSizes.push_back(kBlockSize);
    }
  }
 
  std::optional<bool> srcTransposeInfo = info.currentLayout.getTransposeValue();
  SmallVector<Value> transCurrentOffset(currentOffset);
  if (srcTransposeInfo.has_value() && srcTransposeInfo.value()) {
    auto srcRank = transCurrentOffset.size();
    assert(srcRank >= 2);
    transCurrentOffset[srcRank - 1] = currentOffset[srcRank - 2];
    transCurrentOffset[srcRank - 2] = currentOffset[srcRank - 1];
  }

  switch (conversionKind) {
  case LayoutConversionKind::DOT_ND_TO_nZ:
  case LayoutConversionKind::ND_TO_nZ:
    return computeDOTNDToFractalnZOffset(transCurrentOffset, builder, loc,
                                         kBlockSizes);
  case LayoutConversionKind::DOT_ND_TO_zN:
  case LayoutConversionKind::ND_TO_zN:
    return computeDOTNDToFractalzNOffset(transCurrentOffset, builder, loc,
                                         kBlockSizes);
  default:
    return failure();
  }
}

FailureOr<SmallVector<Value>>
DataLayoutInferAndPropagateHelper::computeDOTNDToFractalzNOffset(
    SmallVector<Value> currentOffset, OpBuilder &builder, Location loc,
    SmallVector<int64_t> kBlockSizes) const {
  OpBuilder::InsertionGuard g(builder);
  uint batchIndexBias = currentOffset.size() - 2;
  // DOT{A/B/C}_ND:   (a, b) -> zN: (b/b0, a/a0, a%a0, b%b0)
  auto symA = builder.getAffineSymbolExpr(0);
  auto symB = builder.getAffineSymbolExpr(1);
  // #mapAOuter = affine_map<(symA, symB) -> (symA / a0)>
  // #mapAInner = affine_map<(symA, symB) ->(symA % a0)>
  auto constA0 = builder.getAffineConstantExpr(kBlockSizes[0]);
  auto mapAOuter = AffineMap::get(0, 2, (symA.floorDiv(constA0)));
  auto mapAInner = AffineMap::get(0, 2, (symA % constA0));

  // #mapBOuter = affine_map<(symA, symB) -> (symB /b0)>
  // #mapBInner = affine_map<(symA, symB) -> (symB % b0)>
  auto constB0 = builder.getAffineConstantExpr(kBlockSizes[1]);
  auto mapBOuter = AffineMap::get(0, 2, (symB.floorDiv(constB0)));
  auto mapBInner = AffineMap::get(0, 2, (symB % constB0));

  Value a = currentOffset[0 + batchIndexBias];
  Value b = currentOffset[1 + batchIndexBias];
  auto aouter =
      builder.create<affine::AffineApplyOp>(loc, mapAOuter, ValueRange{a, b});
  auto ainner =
      builder.create<affine::AffineApplyOp>(loc, mapAInner, ValueRange{a, b});
  auto bouter =
      builder.create<affine::AffineApplyOp>(loc, mapBOuter, ValueRange{a, b});
  auto binner =
      builder.create<affine::AffineApplyOp>(loc, mapBInner, ValueRange{a, b});
  SmallVector<Value> fractalOffset{bouter, aouter, ainner, binner};
  if (batchIndexBias != 0) {
    fractalOffset.insert(fractalOffset.begin(), currentOffset[0]);
  }
  return fractalOffset;
}

FailureOr<SmallVector<Value>>
DataLayoutInferAndPropagateHelper::computeDOTNDToFractalnZOffset(
    SmallVector<Value> currentOffset, OpBuilder &builder, Location loc,
    SmallVector<int64_t> kBlockSizes) const {
  OpBuilder::InsertionGuard g(builder);
  int batchIndexBias = static_cast<int>(currentOffset.size() - 2);
  // DOT{A/B/C}_ND:   (a, b) -> zN: (a/a0, b/b0, b%b0, a%a0)
  auto symA = builder.getAffineSymbolExpr(0);
  auto symB = builder.getAffineSymbolExpr(1);
  // #mapAOuter = affine_map<(symA, symB) -> (symA / a0)>
  // #mapAInner = affine_map<(symA, symB) ->(symA % a0)>
  auto constA0 = builder.getAffineConstantExpr(kBlockSizes[1]);
  auto mapAOuter = AffineMap::get(0, 2, (symA.floorDiv(constA0)));
  auto mapAInner = AffineMap::get(0, 2, (symA % constA0));

  // #mapBOuter = affine_map<(symA, symB) -> (symB /b0)>
  // #mapBInner = affine_map<(symA, symB) -> (symB % b0)>
  auto constB0 = builder.getAffineConstantExpr(kBlockSizes[0]);
  auto mapBOuter = AffineMap::get(0, 2, (symB.floorDiv(constB0)));
  auto mapBInner = AffineMap::get(0, 2, (symB % constB0));

  Value a = currentOffset[0 + batchIndexBias];
  Value b = currentOffset[1 + batchIndexBias];
  auto aouter =
      builder.create<affine::AffineApplyOp>(loc, mapAOuter, ValueRange{a, b});
  auto ainner =
      builder.create<affine::AffineApplyOp>(loc, mapAInner, ValueRange{a, b});
  auto bouter =
      builder.create<affine::AffineApplyOp>(loc, mapBOuter, ValueRange{a, b});
  auto binner =
      builder.create<affine::AffineApplyOp>(loc, mapBInner, ValueRange{a, b});
  SmallVector<Value> fractalOffset{aouter, bouter, binner, ainner};
  if (batchIndexBias != 0) {
    fractalOffset.insert(fractalOffset.begin(), currentOffset[0]);
  }
  return fractalOffset;
}

FailureOr<SmallVector<Value>>
DataLayoutInferAndPropagateHelper::computeDOTNDToFractalzNShape(
    SmallVector<Value> currentShape, OpBuilder &builder, Location loc,
    SmallVector<int64_t> kBlockSizes) const {
  OpBuilder::InsertionGuard g(builder);
  int batchIndexBias = static_cast<int>(currentShape.size() - 2);
  // DOT{A/B/C}_ND:   (a1a0, b1b0) -> zN: (b1, a1, a0, b0)
  auto symA = builder.getAffineSymbolExpr(0);
  auto symB = builder.getAffineSymbolExpr(1);
  auto constA0MinusOne = builder.getAffineConstantExpr(kBlockSizes[0] - 1);
  auto constA0 = builder.getAffineConstantExpr(kBlockSizes[0]);
  // #mapA = affine_map<(symA, symB) -> ((symA+constA0-1)/constA0)>
  auto mapA = AffineMap::get(0, 2, (symA + constA0MinusOne).floorDiv(constA0));
  // #mapB = affine_map<(symA, symB) -> ((symB+constB0-1)/constB0)>
  auto constB0MinusOne = builder.getAffineConstantExpr(kBlockSizes[1] - 1);
  auto constB0 = builder.getAffineConstantExpr(kBlockSizes[1]);
  auto mapB = AffineMap::get(0, 2, (symB + constB0MinusOne).floorDiv(constB0));
  Value a = currentShape[0 + batchIndexBias];
  Value b = currentShape[1 + batchIndexBias];
  auto a1 = builder.create<affine::AffineApplyOp>(loc, mapA, ValueRange{a, b});
  auto b1 = builder.create<affine::AffineApplyOp>(loc, mapB, ValueRange{a, b});
  Value a0 = builder.create<arith::ConstantIndexOp>(loc, kBlockSizes[0]);
  Value b0 = builder.create<arith::ConstantIndexOp>(loc, kBlockSizes[1]);
  SmallVector<Value> fractalShape{b1.getResult(), a1.getResult(), a0, b0};
  if (batchIndexBias != 0) {
    fractalShape.insert(fractalShape.begin(), currentShape[0]);
  }
  return fractalShape;
}

FailureOr<SmallVector<Value>>
DataLayoutInferAndPropagateHelper::computeDOTNDToFractalnZShape(
    SmallVector<Value> currentShape, OpBuilder &builder, Location loc,
    SmallVector<int64_t> kBlockSizes) const {
  OpBuilder::InsertionGuard g(builder);
  uint batchIndexBias = currentShape.size() - 2;
  // DOT{A/B/C}_ND:   (a1a0, b1b0) -> zN: (a1, b1, b0, a0)
  auto symA = builder.getAffineSymbolExpr(0);
  auto symB = builder.getAffineSymbolExpr(1);
  auto constA0MinusOne = builder.getAffineConstantExpr(kBlockSizes[1] - 1);
  auto constA0 = builder.getAffineConstantExpr(kBlockSizes[1]);
  // #mapA = affine_map<(symA, symB) -> ((symA+constA0-1)/constA0)>
  auto mapA = AffineMap::get(0, 2, (symA + constA0MinusOne).floorDiv(constA0));
  // #mapB = affine_map<(symA, symB) -> ((symB+constB0-1)/constB0)>
  auto constB0MinusOne = builder.getAffineConstantExpr(kBlockSizes[0] - 1);
  auto constB0 = builder.getAffineConstantExpr(kBlockSizes[0]);
  auto mapB = AffineMap::get(0, 2, (symB + constB0MinusOne).floorDiv(constB0));
  Value a = currentShape[0 + batchIndexBias];
  Value b = currentShape[1 + batchIndexBias];
  auto a1 = builder.create<affine::AffineApplyOp>(loc, mapA, ValueRange{a, b});
  auto b1 = builder.create<affine::AffineApplyOp>(loc, mapB, ValueRange{a, b});
  Value a0 = builder.create<arith::ConstantIndexOp>(loc, kBlockSizes[1]);
  Value b0 = builder.create<arith::ConstantIndexOp>(loc, kBlockSizes[0]);
  SmallVector<Value> fractalShape{a1.getResult(), b1.getResult(), b0, a0};
  if (batchIndexBias != 0) {
    fractalShape.insert(fractalShape.begin(), currentShape[0]);
  }
  return fractalShape;
}

FailureOr<ConvertLayoutOp>
DataLayoutInferAndPropagateHelper::createLayoutConversion(
    Value currentValue, const LayoutInfo &info, OpBuilder &builder) {
  auto currentLayout = info.currentLayout;
  auto targetLayout = info.targetLayout;
  OpBuilder::InsertionGuard guard(builder);
  Location loc = currentValue.getLoc();
  BaseMemRefType currentType = dyn_cast<BaseMemRefType>(currentValue.getType());
  assert(currentType);
  auto newType = UnrankedMemRefType::get(currentType.getElementType(),
                                         currentType.getMemorySpace());
  // The input value is mapped to a new layout.
  auto conversion =
      builder.create<hivm::ConvertLayoutOp>(loc, newType, currentValue,
                                            /*srcLayout=*/currentLayout,
                                            /*dstLayout=*/targetLayout);
  map(currentValue, conversion.getResult(), targetLayout);
  return conversion;
}

//===----------------------------------------------------------------------===//
// Init Anchor Layout
//===----------------------------------------------------------------------===//

/// Initializes layout information for anchor operations.
/// Walks through the function to find operations implementing
/// OpWithLayoutInterface and extracts their operand layout requirements.
/// Also handles PointerCastOp results and function arguments with GM space.
void DataLayoutInferAndPropagateHelper::initAnchorLayout() {
  LLVM_DEBUG(
      llvm::dbgs()
          << "//===--------------------------------------------===//\n"
             "//===--- Initializing anchor ops' layout info ---===//\n";);
  func_.walk([this](Operation *op) {
    if (auto opWithLayout = dyn_cast<OpWithLayoutInterface>(op)) {
      anchor_ops_.insert(op);
      llvm::SmallDenseMap<Value, DataLayoutAttr> currentLayoutMap =
          opWithLayout.getOperandsCurrentLayout();
      llvm::SmallDenseMap<Value, DataLayoutAttr> targetLayoutMap =
          opWithLayout.getOperandsTargetLayout();
      assert(currentLayoutMap.size() == targetLayoutMap.size());
      for (auto operand : op->getOperands()) {
        if (!isa<MemRefType>(operand.getType()))
          continue;
        // Ops that implement OpWithLayoutInterface should ensure that every
        // operand with memref type has a corresponding data layout.
        auto currentLayout = currentLayoutMap[operand];
        auto targetLayout = targetLayoutMap[operand];
        LLVM_DEBUG(llvm::dbgs() << targetLayout << "\n";);
        // Layout Information is appended on to the root alloc.
        FailureOr<memref::AllocOp> status = getMemRefAlloc(operand);
        if (failed(status)) {
          LLVM_DEBUG(llvm::dbgs() << "  Cannot find root alloc for operand: "
                                  << operand << "\n";);
          continue;
        }
        (void)updateLayoutIfChanged(*status, {currentLayout, targetLayout});
      }
      return WalkResult::advance();
    }
    // Pointer casts with GM space address is considered to have ND layout.
    // This could happen due to supporting indirect memory access.
    if (auto pointerCast = dyn_cast<hivm::PointerCastOp>(op)) {
      for (auto result : pointerCast->getResults()) {
        if (!isGlobalMemory(result))
          continue;

        auto ndLayout =
            DataLayoutAttr::get(result.getContext(), DataLayout::ND);
        (void)updateLayoutIfChanged(result, {ndLayout, ndLayout});
      }
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });
  // Func arguments with GM space address is considered to have ND layout.
  // TODO: Get layout information from func arg attrs.
  for (auto arg : func_.getArguments()) {
    if (!isGlobalMemory(arg))
      continue;

    auto ndLayout = DataLayoutAttr::get(arg.getContext(), DataLayout::ND);
    (void)updateLayoutIfChanged(arg, {ndLayout, ndLayout});
  }
}

//===----------------------------------------------------------------------===//
// Propagate Layout
//===----------------------------------------------------------------------===//

/// Propagates layout information from anchor operations to their users.
/// Uses a worklist algorithm to iteratively propagate layout info through
/// the def-use chain until a fixed point is reached.
void DataLayoutInferAndPropagateHelper::propagateLayout() {
  LLVM_DEBUG(
      llvm::dbgs()
          << "//===----------------------------------------------===//\n"
          << "//=== Propagating anchor ops' layout info to users ===//\n";);
  SmallVector<Value> queue;
  for (auto it : layout_info_)
    queue.push_back(it.first);
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo info = layout_info_.lookup(currentValue);
    queue.pop_back();
    SmallVector<Value> changed = propagateDataLayoutToUsers(currentValue, info);
    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

/// Propagates data layout information to users of a value.
/// Handles various operation types including SelectOp, ForOp, ViewLikeOps,
/// and YieldOp with special handling for scf::IfOp.
/// @param val The value whose layout should be propagated.
/// @param info The layout information to propagate.
/// @return Vector of values whose layouts were changed (for further propagation).
SmallVector<Value>
DataLayoutInferAndPropagateHelper::propagateDataLayoutToUsers(
    Value val, const LayoutInfo &info) {
  SmallVector<Value> changed;
  auto propagateFn = [&](OpOperand &user) -> void {
    Operation *userDefiningOp = user.getOwner();
    return TypeSwitch<Operation *, void>(userDefiningOp)
        .Case<scf::ForOp>([&](scf::ForOp op) {
          Value arg = op.getTiedLoopRegionIterArg(&user);
          Value result = op.getTiedLoopResult(&user);
          updateLayout({arg, result}, info, changed);
        })
        .Case<ViewLikeOpInterface>([&](ViewLikeOpInterface op) {
          updateLayout(op->getResults(), info, changed);
        })
        .Default([&](Operation *op) {
          // Don't need to update Ops that don't have results.
          if (op->getNumResults() == 0)
            return;

          op->emitWarning("Unsupported user for propagating data layout.");
        });
  };
  // Iterate over the users of the val.
  for (OpOperand &user : val.getUses()) {
    // Update the type of the result that corresponds to the operand.
    propagateFn(user);
  }
  return changed;
}

/// Updates the layout information for a value if it has changed.
/// @param value The value to update.
/// @param info The new layout information.
/// @return True if the layout was updated, false if unchanged.
bool DataLayoutInferAndPropagateHelper::updateLayoutIfChanged(
    Value value, const LayoutInfo &info) {

  if (layout_info_.contains(value) && layout_info_.lookup(value) == info) {
    LDBG("Not updating, layout_info_ for " << value << " exists");
    return false;
  }
  layout_info_[value] = info;
  LLVM_DEBUG(llvm::dbgs() << "  Value [" << value << "] Current layout is: "
                          << info.currentLayout
                          << ", Target layout is: "
                          << info.targetLayout << "\n";);
  return true;
}

void DataLayoutInferAndPropagateHelper::updateLayout(
    ValueRange values, const LayoutInfo &info, SmallVector<Value> &changed) {
  for (Value value : values) {
    if (!isa<BaseMemRefType>(value.getType()))
      continue;
    if (updateLayoutIfChanged(value, info))
      changed.push_back(value);
  }
}

//===----------------------------------------------------------------------===//
// Resolve Conflicts
//===----------------------------------------------------------------------===//
void DataLayoutInferAndPropagateHelper::resolveConflicts() {
  LLVM_DEBUG(
      llvm::dbgs()
          << "//===----------------------------------------------===//\n"
          << "//===------ Resolving data layout conflicts -------===//\n";);
  for (Operation *op : anchor_ops_) {
    LLVM_DEBUG(llvm::dbgs()
                   << "  Resolving data layout conflict for anchor op: " << *op
                   << "\n";);
    for (Value currentValue : op->getOperands()) {
      if (!layout_info_.contains(currentValue)) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "  --> Resolving data layout conflict for: "
                              << currentValue << "\n";);
      LayoutInfo info = layout_info_[currentValue];
      if (info.noLayoutConflict()) {
        LLVM_DEBUG(llvm::dbgs() << "        No conflicts found, layout is: "
                                << info.currentLayout << "\n";);
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "        Current layout is: "
                              << info.currentLayout
                              << ", Target layout is: "
                              << info.targetLayout << "\n";);
      if (!isConversionValid(info)) {
        emitError(currentValue.getLoc(),
                  "        Failed to resolve data layout conflicts!\n");
        continue;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//
void DataLayoutInferAndPropagateHelper::rewrite() {
  LLVM_DEBUG(
      llvm::dbgs()
          << "//===--------------------------------------------===//\n"
          << "//===-- Rewriting module with data layout info --===//\n";);
  rewriteRegion(func_->getRegion(0));
}

void DataLayoutInferAndPropagateHelper::rewriteRegion(Region &region) {
  SmallVector<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.back();
    queue.pop_back();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = false;
      SmallVector<Value> results = op.getResults();
      for (Value result : results) {
        // If there is no layout info, skip.
        if (!layout_info_.contains(result))
          continue;
        LayoutInfo info = layout_info_[result];
        // If the layout is already what we want, skip.
        if (info.noLayoutConflict())
          continue;
        needRewrite = true;
      }
      if (needRewrite) {
        LLVM_DEBUG(llvm::dbgs() << "  Rewriting op: " << op << "\n";);
        Operation *newOp = rewriteOp(&op);
        assert(newOp != nullptr && "Failed to rewrite!");
        for (Region &R : newOp->getRegions())
          queue.push_back(&R);
      } else if (isa<memref::CopyOp, hivm::CopyOp, hivm::LoadOp, hivm::StoreOp>(
                     &op)) {
        // Use whitelist instead of copy like interface because there maybe
        // unwanted ops
        LLVM_DEBUG(llvm::dbgs() << "  Remapping op: " << op << "\n";);
        rewriteCopyOp(&op);
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
        auto first_remap = true;
        for (OpOperand &operand : op.getOpOperands()) {
          Value val = operand.get();
          auto targetLayout = getTargetLayout(val);
          if (!targetLayout)
            continue;
          if (first_remap) {
            LLVM_DEBUG(llvm::dbgs() << "  Remapping op: " << op << "\n";);
            first_remap = false;
          }
          LDBG("Get value of val: " << val);
          op.setOperand(operand.getOperandNumber(),
                        getValueAs(val, targetLayout));
        }
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      }
    }
  }
  // Erase ops in reversed order
  for (Operation *op : llvm::reverse(opsToDelete_))
    op->erase();

  LLVM_DEBUG(
      llvm::dbgs()
          << "//===--------------------------------------------===//\n";);
}

Operation *
DataLayoutInferAndPropagateHelper::rewriteMemrefCastOp(memref::CastOp op) {
  auto srcType = op.getType();
  if (srcType.getRank() != 2 && srcType.getRank() != 3)
    llvm_unreachable("Unsupported source shape in mad operand memref cast");
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value src = op.getSource();
  auto layoutInfo = layout_info_.lookup(src);

  // Compute the new shape after layout conversion.
  auto shape = getValueFromShape(src, builder);
  assert(succeeded(shape));
  auto newShape = computeTargetLayoutShape(*shape, layoutInfo, builder, loc);
  if (failed(newShape)) {
    LLVM_DEBUG(llvm::dbgs() << "  Cannot compute new shape.\n";);
    return nullptr;
  }

  auto newRank = (*newShape).size();
  auto newStaticSizes = SmallVector<int64_t>(newRank, ShapedType::kDynamic);
  auto newStridesForMemRefType =
      SmallVector<int64_t>(newRank - 1, ShapedType::kDynamic);
  newStridesForMemRefType.push_back(1);
  Type newResType = MemRefType::get(
      newStaticSizes, srcType.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic,
                             newStridesForMemRefType),
      srcType.getMemorySpace());
  auto rewrittenValue = getValueAs(src, layoutInfo.targetLayout);
  auto newMemrefCastOp =
      builder.create<memref::CastOp>(loc, newResType, rewrittenValue);
  map(op.getResult(), newMemrefCastOp.getResult(), layoutInfo.targetLayout);
  return newMemrefCastOp.getOperation();
}

Operation *DataLayoutInferAndPropagateHelper::rewriteOp(Operation *op) {
  opsToDelete_.insert(op);
  return TypeSwitch<Operation *, Operation *>(op)
      .Case<memref::AllocOp>([&](auto op) { return rewriteAllocOp(op); })
      .Case<memref::SubViewOp>([&](auto op) { return rewriteSubViewOp(op); })
      .Case<memref::CollapseShapeOp>(
          [&](auto op) { return rewriteCollapseShapeOp(op); })
      .Case<scf::ForOp>([&](auto op) { return rewriteForOp(op); })
      .Case<memref::CastOp>([&](auto op) { return rewriteMemrefCastOp(op); })
      .Default([](auto) {
        llvm::report_fatal_error("unexpected op in rewrite");
        return nullptr;
      });
}

Operation *
DataLayoutInferAndPropagateHelper::rewriteAllocOp(memref::AllocOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  auto srcType = op.getType();
  auto layoutInfo = layout_info_.lookup(op.getMemref());
  if (srcType.getRank() != 2 && srcType.getRank() != 3) {
    map(op.getMemref(), op.getMemref(), layoutInfo.targetLayout);
    return op;
  }
  // Extract mix-typed shape from AllocOp.
  auto shape = getValueFromShape(op, builder);
  assert(succeeded(shape));
  // Compute the new shape after layout conversion.

  auto newShape = computeTargetLayoutShape(*shape, layoutInfo, builder, loc);
  if (failed(newShape)) {
    LLVM_DEBUG(llvm::dbgs() << "  Cannot compute new shape.\n";);
    return nullptr;
  }
  auto newRank = (*newShape).size();
  // Create a fully dynamic alloc.
  Type fullyDynamicShapedType = MemRefType::get(
      SmallVector<int64_t>(newRank, ShapedType::kDynamic),
      srcType.getElementType(), builder.getMultiDimIdentityMap(newRank),
      srcType.getMemorySpace());
  auto newAlloc =
      builder.create<memref::AllocOp>(loc, fullyDynamicShapedType, (*newShape),
                                      ValueRange{}, op.getAlignmentAttr());
  map(op.getMemref(), newAlloc.getMemref(), layoutInfo.targetLayout);
  return newAlloc.getOperation();
}

Operation *
DataLayoutInferAndPropagateHelper::rewriteSubViewOp(memref::SubViewOp op) {
  auto src = op.getViewSource();
  auto srcType = op.getSourceType();
  auto layoutInfo = layout_info_.lookup(src);
  if (srcType.getRank() != 2 && srcType.getRank() != 3) {
    map(op.getResult(), op.getResult(), layoutInfo.targetLayout);
    return op;
  }
  int batchIndexBias = (srcType.getRank() == 3 ? 1 : 0);
  auto strideIsOne = [](int64_t stride) { return stride == 1; };
  if (!std::all_of(op.getStaticStrides().begin() + batchIndexBias,
                   op.getStaticStrides().end(), strideIsOne)) {
    LLVM_DEBUG(llvm::dbgs() << "  Currently only support rewriting SubViewOp "
                               "with continuous strides!\n";);
    return nullptr;
  }
  auto rewrittenValue = getValueAs(src, layoutInfo.targetLayout);
  OpBuilder builder(op);
  Location loc = op.getLoc();
  // Extract mix-typed shape from SubViewOp.
  auto shape = getValueFromShape(op, builder);
  assert(succeeded(shape));
  // Create new shape after layout conversion.
  auto newShape = computeTargetLayoutShape(*shape, layoutInfo, builder, loc);
  if (failed(newShape)) {
    LLVM_DEBUG(llvm::dbgs() << "  Cannot compute new shape.\n";);
    return nullptr;
  }
  // Consider drop dimension state
  auto newRank = (*newShape).size();
  auto newStaticSizes = SmallVector<int64_t>(newRank, ShapedType::kDynamic);
  // Result type is memref<?x...x{ElementType}, strided<[?, ?, ?, 1], offset: ?>
  auto newStridesForMemRefType =
      SmallVector<int64_t>(newRank - 1, ShapedType::kDynamic);
  newStridesForMemRefType.push_back(1);
  Type fullyDynamicShapedType = MemRefType::get(
      newStaticSizes, srcType.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic,
                             newStridesForMemRefType),
      srcType.getMemorySpace());
  auto offsets = getValueOrCreateConstantIndexOp(builder, op.getLoc(),
                                                 op.getMixedOffsets());
  auto newOffset = computeTargetLayoutOffset(offsets, layoutInfo, builder, loc);
  if (failed(newOffset)) {
    LLVM_DEBUG(llvm::dbgs() << "  Cannot compute new offset.\n";);
    return nullptr;
  }
  auto newStaticStrides = SmallVector<int64_t>(newRank, 1);
  auto newStaticOffsets = SmallVector<int64_t>(newRank, ShapedType::kDynamic);
  ArrayRef newShapeRef(*newShape);
  ArrayRef newOffsetRef(*newOffset);
  // If batch dimension is `1`, batch offset of subview will always be static 1,
  // so new offsets value collection just skips this state
  if (batchIndexBias && op.getStaticOffsets()[0] == ShapedType::kDynamic)
    newStaticOffsets.front() = ShapedType::kDynamic;

  // consider drop batch dimension state
  if (batchIndexBias && op.getDroppedDims().any()) {
    if (op.getDroppedDims().count() != 1 ||
        op.getDroppedDims().find_first() != 0) {
      op->emitError("only support to drop batch dimension");
      return nullptr;
    }

    newStaticSizes[0] = 1;
    fullyDynamicShapedType = memref::SubViewOp::inferRankReducedResultType(
        ArrayRef(newStaticSizes).drop_front(),
        dyn_cast<MemRefType>(rewrittenValue.getType()), newStaticOffsets,
        newStaticSizes, newStaticStrides);
    newShapeRef = newShapeRef.drop_front();
  }

  auto newSubViewOp = builder.create<memref::SubViewOp>(
      op.getLoc(), TypeRange{fullyDynamicShapedType}, rewrittenValue,
      newOffsetRef, newShapeRef, ValueRange{},
      builder.getDenseI64ArrayAttr(newStaticOffsets),
      builder.getDenseI64ArrayAttr(newStaticSizes),
      builder.getDenseI64ArrayAttr(newStaticStrides));
  map(op.getResult(), newSubViewOp.getResult(), layoutInfo.targetLayout);
  return newSubViewOp;
}

Operation *DataLayoutInferAndPropagateHelper::rewriteForOp(scf::ForOp op) {
  SmallVector<Value> newOperands;
  SmallVector<DataLayoutAttr> newLayouts;
  // Get the remapped operands and their target layouts.
  for (auto [operand, result] : llvm::zip(op.getInitArgs(), op.getResults())) {
    Value convertedOperand = operand;
    DataLayoutAttr layout;
    if (layout_info_.contains(result)) {
      layout = getTargetLayout(result);
      convertedOperand = getValueAs(operand, layout);
    }
    newOperands.push_back(convertedOperand);
    newLayouts.push_back(layout);
  }
  // Construct new ForOp and transfer all operations to the new ForOp.
  OpBuilder rewriter(op);
  auto newForOp = rewriter.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                              op.getUpperBound(), op.getStep(),
                                              newOperands);
  newForOp.getBody()->getOperations().splice(
      newForOp.getBody()->getOperations().begin(),
      op.getBody()->getOperations());
  // Replace old result/iterArg to new ones.
  for (auto [oldResult, newResult, newLayout] :
       llvm::zip(op.getResults(), newForOp.getResults(), newLayouts)) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
    }
    if (!newLayout) {
      continue;
    }
    map(oldResult, newResult, newLayout);
  }
  for (auto [oldArg, newArg, newLayout] : llvm::zip(
           op.getRegionIterArgs(), newForOp.getRegionIterArgs(), newLayouts)) {
    if (oldArg.getType() == newArg.getType()) {
      oldArg.replaceAllUsesWith(newArg);
    }
    if (!newLayout) {
      continue;
    }
    map(oldArg, newArg, newLayout);
  }
  // Induction var is also a block argument that needs to be replaced.
  op.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
  return newForOp.getOperation();
}

Operation *DataLayoutInferAndPropagateHelper::rewriteCollapseShapeOp(
    memref::CollapseShapeOp op) {
  auto src = op.getViewSource();
  auto srcType = op.getSrcType();
  if (srcType.getRank() != 3 || srcType.getShape()[0] != 1)
    op->emitOpError("in infer data layout, memref::CollapseShapeOp should only "
                    "collapse origin batch dimension which size must be one");

  SmallVector<ReassociationIndices> reassociation =
      op.getReassociationIndices();
  if (reassociation.size() + 1 != static_cast<size_t>(srcType.getRank()) ||
      reassociation[0].size() != 2 || reassociation[0][0] != 0 ||
      reassociation[0][1] != 1)
    op->emitOpError("memref::CollapseShapeOp's reassociation could only "
                    "collapse dimension 0(batch axis) and dimension 1");

  auto rewrittenValue = getValueAs(src, getTargetLayout(src));
  OpBuilder builder(op);

  SmallVector<ReassociationIndices> newReassociation;
  newReassociation.push_back({0, 1});
  for (int i = 2;
       i < llvm::dyn_cast<ShapedType>(rewrittenValue.getType()).getRank(); ++i)
    newReassociation.push_back({i});

  auto newCollapseShapeOp = builder.create<memref::CollapseShapeOp>(
      op.getLoc(), rewrittenValue, newReassociation);
  map(op.getResult(), newCollapseShapeOp.getResult(), getTargetLayout(src));

  return newCollapseShapeOp;
}

LogicalResult
DataLayoutInferAndPropagateHelper::tryFoldLayoutConversionIntoCopy(
    Value src, Value dst, Operation *originalOp, OpBuilder &builder) {
  assert(llvm::isa<MemRefType>(src.getType()) &&
         llvm::isa<MemRefType>(dst.getType()));
  if (llvm::dyn_cast<ShapedType>(dst.getType()).getRank() -
          llvm::dyn_cast<ShapedType>(src.getType()).getRank() !=
      2)
    llvm_unreachable("Unsupported operand shape when convert copy to ND2NZ");
  bool batchFlag = (llvm::dyn_cast<ShapedType>(src.getType()).getRank() == 3);

  auto srcLayout = getCurrentLayout(src);
  auto dstLayout = getCurrentLayout(dst);
  if (!srcLayout || !dstLayout) {
    return failure();
  }
  auto conversionKind = kSupportedConversion.find(
      std::make_pair(srcLayout.getDataLayout(), dstLayout.getDataLayout()));
  if (conversionKind == kSupportedConversion.cend()) {
    return failure();
  }
  switch (conversionKind->second) {
  case LayoutConversionKind::ND_TO_nZ:
  case LayoutConversionKind::ND_TO_zN:
    if (batchFlag) {
      convertToBatchND2NZOp(src, dst, builder);
      return success();
    } else {
      convertToND2NZOp(src, dst, originalOp, builder);
      return success();
    }
  default:
    LLVM_DEBUG(llvm::dbgs() << "  No matching HIVM data copy op with "
                               "on-the-fly layout conversion available.\n";);
    return failure();
  }
}

void DataLayoutInferAndPropagateHelper::rewriteCopyOp(mlir::Operation *op) {
  auto copyOp = cast<CopyOpInterface>(op);
  mlir::Value src = copyOp.getSource();
  mlir::Value dst = copyOp.getTarget();
  auto srcTargetLayout = getTargetLayout(src);
  auto dstTargetLayout = getTargetLayout(dst);
  auto rewrittenSrc = getValueAs(src, srcTargetLayout);
  auto rewrittenDst = getValueAs(dst, dstTargetLayout);
  // If we cannot determine the data layout, or src and dst's target layout is
  // the same, simplify replace.
  if (!srcTargetLayout || !dstTargetLayout) {
    op->replaceUsesOfWith(src, rewrittenSrc);
    op->replaceUsesOfWith(dst, rewrittenDst);
    LDBG("cannot determine src or dst layout");
    return;
  }
  if (srcTargetLayout == dstTargetLayout) {
    op->replaceUsesOfWith(src, rewrittenSrc);
    op->replaceUsesOfWith(dst, rewrittenDst);
    LDBG("src and dst has the same layout, no need to rewrite");
    return;
  }
  // Otherwise, we can try to fold the ConvertLayoutOp with HIVM Ops.
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  auto foldResult =
      tryFoldLayoutConversionIntoCopy(rewrittenSrc, rewrittenDst, op, builder);
  LDBG("try to fold layout conversion into copy...");
  if (succeeded(foldResult)) {
    LDBG("successfully folded");
    opsToDelete_.insert(op);
    return;
  }
  // If no such Op exists, we can only map the source op to the target layout
  // by inserting ConvertLayoutOp.
  rewrittenSrc = getValueAs(src, dstTargetLayout);
  op->replaceUsesOfWith(src, rewrittenSrc);
}

namespace {
struct InferHIVMDataLayoutPass
    : public impl::InferHIVMDataLayoutBase<InferHIVMDataLayoutPass> {
  void runOnOperation() override;
};
} // namespace

void InferHIVMDataLayoutPass::runOnOperation() {
  auto funcOp = getOperation();
  if (hacc::utils::isHost(funcOp))
    return;

  auto tFuncCoreTypeAttr = funcOp->getAttrOfType<hivm::TFuncCoreTypeAttr>(
      hivm::TFuncCoreTypeAttr::name);
  if (!tFuncCoreTypeAttr ||
      tFuncCoreTypeAttr.getFuncCoreType() == hivm::TFuncCoreType::AIV)
    return;

  DataLayoutInferAndPropagateHelper helper(funcOp);
  // Init "anchor" ops' data layout information.
  helper.initAnchorLayout();
  // Propagate data layout information to users.
  helper.propagateLayout();
  // Check if data layout conflicts can be resolved.
  helper.resolveConflicts();
  // Rewrite the IR so that all operands are converted to their target layout.
  helper.rewrite();
}

std::unique_ptr<Pass> mlir::hivm::createInferHIVMDataLayoutPass() {
  return std::make_unique<InferHIVMDataLayoutPass>();
}
