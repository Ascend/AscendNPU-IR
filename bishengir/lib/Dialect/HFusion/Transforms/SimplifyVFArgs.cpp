//===----------------------- SimplifyVFArgs.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <cassert>

namespace mlir {
#define GEN_PASS_DEF_SIMPLIFYVFARGS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

bool isZeroIndex(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  return constant && constant.value() == 0;
}

bool isOne(OpFoldResult value) {
  std::optional<int64_t> constant = getConstantIntValue(value);
  return constant && *constant == 1;
}

bool allOnes(ArrayRef<OpFoldResult> values) {
  return llvm::all_of(values, isOne);
}

bool hasAtMostOneNonUnitDim(MemRefType type) {
  if (!type.hasStaticShape())
    return false;

  int64_t nonUnitDims = 0;
  for (int64_t dim : type.getShape()) {
    if (dim != 1)
      ++nonUnitDims;
  }
  return nonUnitDims <= 1;
}

bool isLeadingUnitDims(VectorType type) {
  if (type.getRank() <= 1)
    return false;
  ArrayRef<int64_t> shape = type.getShape();
  return llvm::all_of(shape.drop_back(), [](int64_t dim) { return dim == 1; });
}

bool allInBounds(vector::TransferReadOp readOp) {
  ArrayAttr inBoundsAttr = readOp.getInBoundsAttr();
  if (!inBoundsAttr ||
      static_cast<int64_t>(inBoundsAttr.getValue().size()) !=
          readOp.getVectorType().getRank())
    return false;
  return llvm::all_of(inBoundsAttr.getAsRange<BoolAttr>(),
                      [](BoolAttr attr) { return attr.getValue(); });
}

std::optional<int64_t> getConstantMaskActiveProduct(Value mask) {
  auto constMaskOp = mask.getDefiningOp<vector::ConstantMaskOp>();
  if (!constMaskOp)
    return std::nullopt;

  int64_t product = 1;
  for (Attribute attr : constMaskOp.getMaskDimSizes().getValue()) {
    int64_t dim = cast<IntegerAttr>(attr).getInt();
    if (dim < 0)
      return std::nullopt;
    product *= dim;
  }
  return product;
}

bool hasSingleElementMask(Value mask, int64_t flatSize) {
  if (!mask)
    return false;

  if (auto shapeCast = mask.getDefiningOp<vector::ShapeCastOp>()) {
    auto srcType = dyn_cast<VectorType>(shapeCast.getSource().getType());
    if (srcType && srcType.getRank() == 1 &&
        srcType.getNumElements() == flatSize &&
        srcType.getElementType().isInteger(1)) {
      std::optional<int64_t> active =
          getConstantMaskActiveProduct(shapeCast.getSource());
      return active && *active == 1;
    }
  }

  std::optional<int64_t> active = getConstantMaskActiveProduct(mask);
  return active && *active == 1;
}

Value getOrCreateFlatSingleElementMask(Value mask, int64_t flatSize,
                                       Location loc, OpBuilder &builder) {
  if (auto shapeCast = mask.getDefiningOp<vector::ShapeCastOp>()) {
    auto srcType = dyn_cast<VectorType>(shapeCast.getSource().getType());
    if (srcType && srcType.getRank() == 1 &&
        srcType.getNumElements() == flatSize &&
        srcType.getElementType().isInteger(1)) {
      std::optional<int64_t> active =
          getConstantMaskActiveProduct(shapeCast.getSource());
      if (active && *active == 1)
        return shapeCast.getSource();
    }
  }

  auto flatMaskType = VectorType::get({flatSize}, builder.getI1Type());
  return builder
      .create<vector::ConstantMaskOp>(
          loc, flatMaskType,
          builder.getArrayAttr(builder.getI64IntegerAttr(1)))
      .getResult();
}

bool isRank1I1MemrefExpand(memref::ExpandShapeOp expandOp,
                           MemRefType expectedResultType) {
  auto srcType = dyn_cast<MemRefType>(expandOp.getSrc().getType());
  auto resultType = expandOp.getResultType();
  if (!srcType || resultType != expectedResultType || srcType.getRank() != 1 ||
      !srcType.hasStaticShape() || !resultType.hasStaticShape() ||
      !srcType.getElementType().isInteger(1) ||
      !resultType.getElementType().isInteger(1))
    return false;

  if (srcType.getNumElements() != resultType.getNumElements())
    return false;

  return hasAtMostOneNonUnitDim(resultType);
}

Value materializeIndex(OpFoldResult ofr, Location loc, OpBuilder &builder) {
  if (auto value = dyn_cast<Value>(ofr))
    return value;

  auto attr = cast<IntegerAttr>(dyn_cast<Attribute>(ofr));
  return builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
}

Value linearizeSubviewOffset(memref::SubViewOp subview, OpBuilder &builder) {
  Location loc = subview.getLoc();
  auto srcType = subview.getSourceType();
  ArrayRef<int64_t> shape = srcType.getShape();
  SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();

  Value linearOffset = builder.create<arith::ConstantIndexOp>(loc, 0);
  int64_t stride = 1;
  for (int64_t i = srcType.getRank() - 1; i >= 0; --i) {
    Value offset = materializeIndex(offsets[i], loc, builder);
    if (stride != 1) {
      Value strideValue = builder.create<arith::ConstantIndexOp>(loc, stride);
      offset = builder.create<arith::MulIOp>(loc, offset, strideValue);
    }
    linearOffset = builder.create<arith::AddIOp>(loc, linearOffset, offset);
    stride *= shape[i];
  }
  return linearOffset;
}

Operation *getOnlyUser(Value value) {
  if (!value.hasOneUse())
    return nullptr;
  return *value.user_begin();
}

bool isSupportedI1Subview(memref::SubViewOp subview) {
  auto srcType = subview.getSourceType();
  auto resultType = subview.getType();
  if (!srcType.hasStaticShape() || !resultType.hasStaticShape() ||
      !srcType.getElementType().isInteger(1) ||
      !resultType.getElementType().isInteger(1) ||
      resultType.getRank() <= 1 || resultType.getNumElements() != 1)
    return false;

  return allOnes(subview.getMixedSizes()) && allOnes(subview.getMixedStrides());
}

bool isSupportedI1TransferRead(vector::TransferReadOp readOp,
                               int64_t flatSize) {
  VectorType vectorType = readOp.getVectorType();
  if (!vectorType.getElementType().isInteger(1) ||
      !isLeadingUnitDims(vectorType) ||
      vectorType.getNumElements() != flatSize)
    return false;

  if (!readOp.getPermutationMap().isIdentity() || !allInBounds(readOp) ||
      !llvm::all_of(readOp.getIndices(), isZeroIndex))
    return false;

  if (!hasSingleElementMask(readOp.getMask(), flatSize))
    return false;

  auto shapeCast = dyn_cast_or_null<vector::ShapeCastOp>(
      getOnlyUser(readOp.getResult()));
  if (!shapeCast)
    return false;

  auto flatType = VectorType::get({flatSize}, vectorType.getElementType());
  return shapeCast.getResultVectorType() == flatType;
}

bool canRewriteI1ExpandedArg(BlockArgument arg, int64_t flatSize) {
  for (Operation *user : arg.getUsers()) {
    auto subview = dyn_cast<memref::SubViewOp>(user);
    if (!subview || !isSupportedI1Subview(subview))
      return false;

    for (Operation *subviewUser : subview.getResult().getUsers()) {
      auto readOp = dyn_cast<vector::TransferReadOp>(subviewUser);
      if (!readOp || !isSupportedI1TransferRead(readOp, flatSize))
        return false;
    }
  }
  return !arg.use_empty();
}

void rewriteI1Subview(memref::SubViewOp subview, BlockArgument flatArg,
                      Value flatOffset, int64_t flatSize,
                      OpBuilder &builder) {
  builder.setInsertionPoint(subview);
  Location loc = subview.getLoc();
  SmallVector<OpFoldResult> offsets = {flatOffset};
  SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1)};
  SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1)};
  auto flatSubview = builder.create<memref::SubViewOp>(
      loc, flatArg, offsets, sizes, strides);

  SmallVector<Operation *> users(subview.getResult().getUsers());
  for (Operation *user : users) {
    auto readOp = cast<vector::TransferReadOp>(user);
    auto shapeCast = cast<vector::ShapeCastOp>(getOnlyUser(readOp.getResult()));

    builder.setInsertionPoint(readOp);
    auto flatType =
        VectorType::get({flatSize}, readOp.getVectorType().getElementType());
    Value zero = builder.create<arith::ConstantIndexOp>(readOp.getLoc(), 0);
    Value flatMask = getOrCreateFlatSingleElementMask(
        readOp.getMask(), flatSize, readOp.getLoc(), builder);
    AffineMap permMap =
        AffineMap::getMultiDimIdentityMap(/*numDims=*/1, builder.getContext());
    auto newRead = builder.create<vector::TransferReadOp>(
        readOp.getLoc(), flatType, flatSubview.getResult(), ValueRange{zero},
        AffineMapAttr::get(permMap), readOp.getPadding(), flatMask,
        builder.getBoolArrayAttr({true}));

    shapeCast.replaceAllUsesWith(newRead.getResult());
    shapeCast.erase();
    readOp.erase();
  }

  subview.erase();
}

bool rewriteI1ExpandedArg(func::FuncOp funcOp, unsigned argIdx,
                          MemRefType flatArgType, ModuleOp module,
                          OpBuilder &builder) {
  Block &entryBlock = funcOp.getBody().front();
  BlockArgument oldArg = entryBlock.getArgument(argIdx);
  if (!canRewriteI1ExpandedArg(oldArg, flatArgType.getNumElements()))
    return false;

  auto callSites = funcOp.getSymbolUses(module);
  if (!callSites.has_value())
    return false;

  SmallVector<func::CallOp> calls;
  for (SymbolTable::SymbolUse use : callSites.value()) {
    auto call = dyn_cast<func::CallOp>(use.getUser());
    if (!call || call.getNumOperands() <= argIdx)
      return false;
    auto expandOp =
        call.getOperand(argIdx).getDefiningOp<memref::ExpandShapeOp>();
    if (!expandOp ||
        !isRank1I1MemrefExpand(expandOp, cast<MemRefType>(oldArg.getType())) ||
        expandOp.getSrc().getType() != flatArgType)
      return false;
    calls.push_back(call);
  }

  SmallVector<std::pair<memref::SubViewOp, Value>> subviews;
  for (Operation *user : llvm::make_early_inc_range(oldArg.getUsers())) {
    auto subview = cast<memref::SubViewOp>(user);
    builder.setInsertionPoint(subview);
    subviews.push_back({subview, linearizeSubviewOffset(subview, builder)});
  }

  oldArg.setType(flatArgType);
  for (auto [subview, flatOffset] : subviews)
    rewriteI1Subview(subview, oldArg, flatOffset, flatArgType.getNumElements(),
                     builder);

  SmallVector<Type> inputTypes(funcOp.getFunctionType().getInputs().begin(),
                              funcOp.getFunctionType().getInputs().end());
  inputTypes[argIdx] = flatArgType;
  funcOp.setFunctionType(
      builder.getFunctionType(inputTypes, funcOp.getFunctionType().getResults()));

  for (func::CallOp call : calls) {
    auto expandOp =
        call.getOperand(argIdx).getDefiningOp<memref::ExpandShapeOp>();
    call->setOperand(argIdx, expandOp.getSrc());
  }

  return true;
}

void simplifyI1ExpandedVFArgs(ModuleOp module, func::FuncOp funcOp,
                              OpBuilder &builder) {
  if (!hivm::isVF(funcOp) || funcOp.getBody().empty())
    return;

  auto callSites = funcOp.getSymbolUses(module);
  if (!callSites.has_value())
    return;

  Block &entryBlock = funcOp.getBody().front();
  for (BlockArgument arg : entryBlock.getArguments()) {
    auto argType = dyn_cast<MemRefType>(arg.getType());
    if (!argType || argType.getRank() <= 1 || !argType.hasStaticShape() ||
        !argType.getElementType().isInteger(1))
      continue;

    MemRefType flatArgType;
    bool found = false;
    for (SymbolTable::SymbolUse use : callSites.value()) {
      auto call = dyn_cast<func::CallOp>(use.getUser());
      if (!call || call.getNumOperands() <= arg.getArgNumber()) {
        found = false;
        break;
      }
      auto expandOp =
          call.getOperand(arg.getArgNumber()).getDefiningOp<memref::ExpandShapeOp>();
      if (!expandOp || !isRank1I1MemrefExpand(expandOp, argType)) {
        found = false;
        break;
      }
      auto srcType = cast<MemRefType>(expandOp.getSrc().getType());
      if (!found) {
        flatArgType = srcType;
        found = true;
      } else if (flatArgType != srcType) {
        found = false;
        break;
      }
    }

    if (found)
      (void)rewriteI1ExpandedArg(funcOp, arg.getArgNumber(), flatArgType,
                                 module, builder);
  }
}

struct SimplifyVFArgsPass
    : public impl::SimplifyVFArgsBase<SimplifyVFArgsPass> {
  using SimplifyVFArgsBase<SimplifyVFArgsPass>::SimplifyVFArgsBase;

public:
  void runOnOperation() override;
};
} // namespace

void SimplifyVFArgsPass::runOnOperation() {
  auto mod = getOperation();
  OpBuilder builder(mod.getContext());
  mod->walk([&](func::FuncOp funcOp) {
    if (hivm::isVF(funcOp)) {
      if (funcOp.getBody().empty())
        return;
      simplifyI1ExpandedVFArgs(mod, funcOp, builder);

      SmallVector<int> unusedArgumentInd;
      auto funcType = funcOp.getFunctionType();
      Block &entryBlock = funcOp.getBody().front();
      auto funcInputTypes = funcType.getInputs();
      SmallVector<Type> usedFunctionTypeInputs;
      for (BlockArgument blockArg : entryBlock.getArguments()) {
        int argIdx = blockArg.getArgNumber();
        if (!blockArg.use_empty()) {
          usedFunctionTypeInputs.push_back(funcInputTypes[argIdx]);
          continue;
        }
        unusedArgumentInd.push_back(argIdx);
      }

      if (unusedArgumentInd.empty()) {
        return;
      }

      entryBlock.eraseArguments(
          [&](BlockArgument bArg) { return bArg.use_empty(); });

      auto callSites = funcOp.getSymbolUses(mod);

      assert(callSites.has_value() && "A VF must at least be invoked once");
      FunctionType allUsedFunctionType = FunctionType::get(
          funcOp.getContext(), usedFunctionTypeInputs, funcType.getResults());
      funcOp.setFunctionType(allUsedFunctionType);

      for (SymbolTable::SymbolUse callSite : callSites.value()) {
        func::CallOp call = cast<func::CallOp>(callSite.getUser());
        SmallVector<Value> operands = call->getOperands();
        SmallVector<Value> newOperands;
        for (size_t i = 0; i < operands.size(); ++i) {
          if (!llvm::is_contained(unusedArgumentInd, i)) {
            newOperands.push_back(operands[i]);
          }
        }
        call->setOperands(newOperands);
      }
    }
  });
}

std::unique_ptr<Pass> hfusion::createSimplifyVFArgsPass() {
  return std::make_unique<SimplifyVFArgsPass>();
}
