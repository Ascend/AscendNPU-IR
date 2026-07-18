//===- Utils.cpp - Utilities to support the HIVM dialect ------------------===//
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
//
// This file implements utilities for the HIVM dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <cstdint>
#include <sys/param.h>

#define DEBUG_TYPE "hivm-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {
namespace hivm {

namespace {
/// Find the root memerf alloc for the input block argument.
FailureOr<memref::AllocOp> getMemRefForBlockArgument(BlockArgument bbArg) {
  auto *bbOwner = bbArg.getOwner();
  if (!bbOwner) {
    llvm::report_fatal_error("parentOp doesn't exist");
    return failure();
  }
  auto *bbParentOp = bbOwner->getParentOp();
  if (!bbParentOp)
    return failure();
  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(bbParentOp)) {
    auto *operand = loopOp.getTiedLoopInit(bbArg);
    return getMemRefAlloc(operand->get());
  }
  return bbParentOp->emitError("Unsupported block op type");
}

/// Find the root memerf alloc for the OpResult.
FailureOr<memref::AllocOp> getMemRefForOpResult(OpResult result) {
  return TypeSwitch<Operation *, FailureOr<memref::AllocOp>>(
             result.getDefiningOp())
      .Case<memref::AllocOp>([&](memref::AllocOp op) { return op; })
      // We could pursue view_source of current traced op with
      // viewLikeOpInterface trait.
      .Case<mlir::ViewLikeOpInterface>([&](ViewLikeOpInterface viewLikeOp) {
        return getMemRefAlloc(viewLikeOp.getViewSource());
      })
      .Case<mlir::LoopLikeOpInterface>([&](LoopLikeOpInterface loopOp) {
        Value initSource = loopOp.getInits()[result.getResultNumber()];
        return getMemRefAlloc(initSource);
      })
      .Case<bufferization::ToTensorOp>([&](bufferization::ToTensorOp op) {
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
        return getMemRefAlloc(op.getMemref());
#else
        return getMemRefAlloc(op.getBuffer());
#endif
      })
      .Case<scope::ScopeOp>([&](scope::ScopeOp scopeOp) {
        auto returnOp =
            cast<scope::ReturnOp>(scopeOp.getRegion().front().getTerminator());
        return getMemRefAlloc(
            returnOp->getOperands()[result.getResultNumber()]);
      })
      .Default([&](Operation *op) {
        LDBG("Unsupported op for finding the root alloc.");
        return failure();
      });
}

struct LoopInfo {
  Value inductionVar;
  OpFoldResult lowerBound;
  OpFoldResult upperBound;
  OpFoldResult singleStep;
};

/// Get info of parent and ancestor loop of ptrCastOp.
/// The info is stored from inner to outer parent loop.
std::vector<LoopInfo> getLoopsInfo(LoopLikeOpInterface ptrParentLoop) {
  std::vector<LoopInfo> loopInfoVec;
  LoopLikeOpInterface curOp = ptrParentLoop;
  while (curOp) {
    std::optional<Value> inductionVar = curOp.getSingleInductionVar();
    std::optional<OpFoldResult> lowerBound = curOp.getSingleLowerBound();
    std::optional<OpFoldResult> upperBound = curOp.getSingleUpperBound();
    std::optional<OpFoldResult> singleStep = curOp.getSingleStep();

    assert((inductionVar.has_value() && lowerBound.has_value() &&
            upperBound.has_value() && singleStep.has_value()) &&
           "iv, lb, ub and step shouldn't be null.");

    loopInfoVec.push_back(
        {*inductionVar, *lowerBound, *upperBound, *singleStep});

    curOp = curOp->getParentOfType<LoopLikeOpInterface>();
  }

  return loopInfoVec;
}

/// Index of yielded value where is alias of targetVal.
std::optional<int> getYieldValueIdx(Value targetVal, ValueRange yieldedValues) {
  auto it = std::find(yieldedValues.begin(), yieldedValues.end(), targetVal);
  if (it != yieldedValues.end()) {
    return it - yieldedValues.begin();
  }

  return std::nullopt;
}

/// Flatten loops into one dimension and then modulo modular.
/// for example modular is 2, use all induction var of loops to generate
/// sequence data 0,1,2,3,... then mod 2 to get 0,1,0,1,...
///
/// IR:
/// for i, 0, upperI, step=1:
///   for j, 0, upperJ, step=1:
///     affine.apply #map()[j, i]
///
/// the sequence is: 0,1,2,..., i*upperJ + j, ...
///
/// \return Index value of affineApply
Value createNestedIndexModularUsingLoopInfo(
    OpBuilder &builder, Location loc, const std::vector<LoopInfo> &loopInfoVec,
    int modular) {
  auto ctx = builder.getContext();
  AffineExpr nElems = builder.getAffineConstantExpr(1);
  AffineExpr targetExpr = builder.getAffineConstantExpr(0);
  std::vector<OpFoldResult> symbolValueVec;

  // In order to create affineMap, bind symbols, affineExpr and values are
  // needed. loopInfoVec would be used, loop from inner to outer.
  for (size_t i = 0; i < loopInfoVec.size(); ++i) {
    AffineExpr iv;
    AffineExpr lb;
    AffineExpr ub;
    AffineExpr step;

    // bind symbols
    iv = getAffineSymbolExpr(i * 4, ctx);
    lb = getAffineSymbolExpr(i * 4 + 1, ctx);
    ub = getAffineSymbolExpr(i * 4 + 2, ctx);
    step = getAffineSymbolExpr(i * 4 + 3, ctx);

    // create affineExpr
    targetExpr = targetExpr + (iv - lb).floorDiv(step) * nElems;
    nElems = nElems * ((ub - lb + step - 1).floorDiv(step));

    // values
    auto info = loopInfoVec[i];
    Value ivVal = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getIndexType(), info.inductionVar);
    Value lbVal = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getIndexType(),
        getValueOrCreateConstantIndexOp(builder, loc, info.lowerBound));
    Value ubVal = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getIndexType(),
        getValueOrCreateConstantIndexOp(builder, loc, info.upperBound));
    Value stepVal = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getIndexType(),
        getValueOrCreateConstantIndexOp(builder, loc, info.singleStep));

    symbolValueVec.emplace_back(ivVal);
    symbolValueVec.emplace_back(lbVal);
    symbolValueVec.emplace_back(ubVal);
    symbolValueVec.emplace_back(stepVal);
  }

  if (modular == -1) {
    Value affineApply = mlir::affine::makeComposedAffineApply(
        builder, loc, targetExpr, ArrayRef(symbolValueVec));
    return affineApply;
  }

  targetExpr = targetExpr % modular;

  Value affineApply = mlir::affine::makeComposedAffineApply(
      builder, loc, targetExpr, ArrayRef(symbolValueVec));
  return affineApply;
}

} // namespace

FailureOr<memref::AllocOp> getMemRefAlloc(Value operand) {
  if (auto bbArg = dyn_cast<BlockArgument>(operand)) {
    return getMemRefForBlockArgument(bbArg);
  }
  auto result = dyn_cast<OpResult>(operand);
  assert(result != nullptr);
  return getMemRefForOpResult(result);
}

/// Return all root memref allocs feeding into a value.
/// Recursively traces through scf::IfOp branches and falls back
/// to getMemRefAlloc for simple def-use chains.
SmallVector<Value> getMemRefAllocs(Value operand) {
  // -- if operand comes from scf.if, collect allocs from all branches --
  if (auto ifOp = operand.getDefiningOp<scf::IfOp>()) {
    auto resultNum = cast<OpResult>(operand).getResultNumber(); // which result?
    SmallVector<Value> results;

    for (Region &region : ifOp->getRegions()) {    // then, else
      if (region.empty()) continue;                 // skip absent else
      auto yieldOp = cast<scf::YieldOp>(
          region.front().getTerminator());          // branch's exit point

      // Recurse through the yield operand (handles nested IfOp→IfOp→alloc)
      SmallVector<Value> branchAllocs =
          getMemRefAllocs(yieldOp.getOperand(resultNum));
      if (branchAllocs.empty()) return {};          // any branch fails → bail
      results.append(branchAllocs);
    }

    // All branches must contribute exactly one alloc each
    if (results.size() != ifOp->getRegions().size()) return {};

    // All branches' allocs must have the same type
    Type firstType = results.front().getType();
    for (Value alloc : results)
      if (alloc.getType() != firstType) return {};

    return results;
  }

  // -- fallback: single user case (alloc, subview, for, etc.) --
  FailureOr<memref::AllocOp> alloc = getMemRefAlloc(operand);
  if (failed(alloc)) return {};
  return {Value(*alloc)};                          // wrap in vector
}

// New helper function to get the updated BaseMemRefType
BaseMemRefType getBaseMemRefTypeWithNewScope(BaseMemRefType type,
                                             AddressSpaceAttr targetMemScope) {
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::Builder(memRefType).setMemorySpace(targetMemScope);
  } else if (auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(type)) {
    return UnrankedMemRefType::get(unrankedMemRefType.getElementType(),
                                   targetMemScope);
  }
  llvm_unreachable("Unexpected BaseMemRefType");
  return type;
}

// New helper function to get the updated BaseMemRefType
BaseMemRefType getBaseMemRefTypeWithNewScope(BaseMemRefType type,
                                             unsigned targetMemScope) {
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    auto targetMemScopeAttr = IntegerAttr::get(
        IntegerType::get(type.getContext(), 64), targetMemScope);
    return MemRefType::Builder(memRefType).setMemorySpace(targetMemScopeAttr);
  }
  if (auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(type)) {
    return UnrankedMemRefType::get(unrankedMemRefType.getElementType(),
                                   targetMemScope);
  }
  llvm_unreachable("Unexpected BaseMemRefType");
  return type;
}

void setBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope) {
  Type type = val.getType();
  if (!isa<BaseMemRefType>(type)) {
    return;
  }

  if (auto curMemScope = dyn_cast_if_present<AddressSpaceAttr>(
          dyn_cast<BaseMemRefType>(type).getMemorySpace())) {
    assert(curMemScope == targetMemScope);
    return;
  }

  auto memRefType = cast<BaseMemRefType>(type);
  auto newMemRefType =
      getBaseMemRefTypeWithNewScope(memRefType, targetMemScope);
  val.setType(newMemRefType);
}

void modifyBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope) {
  Type type = val.getType();
  if (!isa<BaseMemRefType>(type)) {
    LDBG("type = " << type << " is not BaseMemRefType\n");
    return;
  }

  auto memRefType = cast<BaseMemRefType>(type);
  auto newMemRefType =
      getBaseMemRefTypeWithNewScope(memRefType, targetMemScope);
  val.setType(newMemRefType);
}

SmallVector<Value>
getValueListFromMixedTypeLists(SmallVector<Value> dynamicValues,
                               ArrayRef<int64_t> staticValues, Location loc,
                               OpBuilder &builder) {
  SmallVector<Value> ret;
  auto dynamicValuesIter = dynamicValues.begin();
  llvm::for_each(staticValues, [&](int64_t val) {
    // Create a constant index op if the static val is not ShapedType::kDynamic
    if (!ShapedType::isDynamic(val)) {
      auto constIndexOp = builder.create<arith::ConstantIndexOp>(loc, val);
      ret.push_back(constIndexOp);
      return;
    }
    // Otherwise, get dynamic value from list.
    assert(dynamicValuesIter != dynamicValues.end());
    ret.push_back(*dynamicValuesIter);
    dynamicValuesIter++;
  });
  return ret;
}

FailureOr<SmallVector<Value>> getValueFromShape(Value currentValue,
                                                OpBuilder &builder) {
  auto currentType = dyn_cast<ShapedType>(currentValue.getType());
  if (!currentType) {
    return failure();
  }
  Location loc = currentValue.getLoc();
  if (auto allocOp = currentValue.getDefiningOp<memref::AllocOp>()) {
    return getValueListFromMixedTypeLists(
        allocOp.getDynamicSizes(), dyn_cast<MemRefType>(currentType).getShape(),
        loc, builder);
  }
  if (auto emptyOp = currentValue.getDefiningOp<tensor::EmptyOp>()) {
    return getValueListFromMixedTypeLists(
        emptyOp.getDynamicSizes(),
        dyn_cast<RankedTensorType>(currentType).getShape(), loc, builder);
  }
  if (auto op = currentValue.getDefiningOp<OffsetSizeAndStrideOpInterface>()) {
    return getValueListFromMixedTypeLists(op.getSizes(), op.getStaticSizes(),
                                          loc, builder);
  }
  if (currentType.hasStaticShape()) {
    return getValueListFromMixedTypeLists(/*No dynamic dimension*/ {},
                                          currentType.getShape(), loc, builder);
  }
  return failure();
}

bool IsAscend910B(Attribute triple) {
  if (!triple) {
    return false;
  }
  return triple ==
         OpBuilder(triple.getContext()).getStringAttr(Ascend910BCubeTriple);
}

uint64_t AlignUp(uint64_t lhs, uint64_t rhs) {
  assert(rhs != 0);
  if (lhs % rhs != 0) {
    lhs += rhs - (lhs % rhs);
  }
  return lhs;
}

std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand) {
  if (!llvm::isa<BaseMemRefType>(operand.getType())) {
    return std::nullopt;
  }
  auto memRefType = cast<BaseMemRefType>(operand.getType());
  auto memorySpace = memRefType.getMemorySpace();
  if (!memorySpace)
    return std::nullopt;
  auto memorySpaceAttr = dyn_cast<AddressSpaceAttr>(memorySpace);
  if (!memorySpaceAttr) {
    return std::nullopt;
  }
  return memorySpaceAttr;
}

bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr) {
  if (!memorySpaceAttr.has_value()) {
    return false;
  }

  if (memorySpaceAttr.value().getAddressSpace() == hivm::AddressSpace::GM) {
    return false;
  }
  if (LocalBufferSpace.count(memorySpaceAttr.value().getAddressSpace())) {
    return true;
  }
  llvm::report_fatal_error("Currently only support (UB | L1 | L0C) allocation");
}

SmallVector<Value> getOpTouchBuffer(Operation *op) {
  SmallVector<Value> touchBuffer;
  touchBuffer.insert(touchBuffer.end(), op->getResults().begin(),
                     op->getResults().end());
  for (OpOperand &operand : op->getOpOperands()) {
    touchBuffer.push_back(operand.get());
  }
  return touchBuffer;
}

bool isOpTouchLocalBuffer(Operation *op) {
  auto touchBuffer = getOpTouchBuffer(op);
  for (Value buffer : touchBuffer) {
    auto bufferSpace = GetBufferSpaceAttr(buffer);
    if (isLocalBuffer(bufferSpace)) {
      return true;
    }
  }
  return false;
}

bool isOpTouchGlobalBuffer(Operation *op) {
  auto touchBuffer = getOpTouchBuffer(op);
  for (Value buffer : touchBuffer) {
    auto bufferSpace = GetBufferSpaceAttr(buffer);
    if (bufferSpace.has_value() &&
        bufferSpace.value().getAddressSpace() == hivm::AddressSpace::GM) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// MapForallToBlocks
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure definiteFailureHelper(
    Operation *target, const Twine &message,
    std::optional<transform::TransformOpInterface> transformOp = std::nullopt) {
  if (transformOp.has_value())
    return transformOp->emitDefiniteFailure() << message;
  return emitDefiniteFailure(target, message);
}

/// Check if given mapping attributes are one of the desired attributes.
static DiagnosedSilenceableFailure checkMappingAttributeTypes(
    scf::ForallOp forallOp,
    std::optional<transform::TransformOpInterface> transformOp) {
  if (!forallOp.getMapping().has_value())
    return definiteFailureHelper(
        forallOp, "scf.forall op requires a mapping attribute", transformOp);

  size_t blockMappingCnt = static_cast<size_t>(
      llvm::count_if(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<HIVMBlockMappingAttr>(attr);
      }));
  size_t subBlockMappingCnt = static_cast<size_t>(
      llvm::count_if(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<HIVMSubBlockMappingAttr>(attr);
      }));
  size_t mappingCnt = blockMappingCnt + subBlockMappingCnt;
  if (mappingCnt != forallOp.getMapping()->size())
    return definiteFailureHelper(
        forallOp, "only support hivm block/sub_block attr", transformOp);

  DenseSet<Attribute> seen;
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (seen.contains(map)) {
      return definiteFailureHelper(
          forallOp,
          "duplicate attribute, cannot map different loops "
          "to the same mapping id",
          transformOp);
    }
    seen.insert(map);
  }

  return DiagnosedSilenceableFailure::success();
}

template <typename MappingAttr>
static bool hasMappingAttr(scf::ForallOp forall) {
  return llvm::any_of(forall.getMappingAttr(),
                      [](Attribute attr) { return isa<MappingAttr>(attr); });
}

static DiagnosedSilenceableFailure verifyHIVMMapping(
    scf::ForallOp forallOp,
    std::optional<transform::TransformOpInterface> transformOp = std::nullopt) {
  // Check the types of the mapping attributes match.
  DiagnosedSilenceableFailure typeRes =
      checkMappingAttributeTypes(forallOp, transformOp);
  if (!typeRes.succeeded())
    return typeRes;

  // Perform other non-types verifications.
  if (!forallOp.isNormalized())
    return definiteFailureHelper(forallOp, "unsupported non-normalized loops",
                                 transformOp);
  if (forallOp.getNumResults() > 0)
    return definiteFailureHelper(
        forallOp, "only bufferized scf.forall can be mapped", transformOp);
  if (!forallOp.getOutputs().empty())
    return definiteFailureHelper(
        forallOp, "does not support scf.forall with shared outputs yet",
        transformOp);

  int64_t maxNumMappingsSupported =
      hasMappingAttr<HIVMSubBlockMappingAttr>(forallOp) ? 2 : 1;
  if (forallOp.getRank() > maxNumMappingsSupported) {
    return definiteFailureHelper(forallOp, "scf.forall with rank > ",
                                 transformOp)
           << maxNumMappingsSupported
           << " does not lower for the specified mapping attribute type";
  }
  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure rewriteNestedForallImpl(
    RewriterBase &rewriter, scf::ForallOp topLevelForall,
    ArrayRef<scf::ForallOp> worklist, ForallRewriteResult &result,
    std::optional<transform::TransformOpInterface> transformOp = std::nullopt) {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = topLevelForall->getLoc();
  rewriter.setInsertionPoint(topLevelForall);
  auto indexTy = rewriter.getIndexType();

  // Get the upper bounds and index variables for block and subblock mappings
  SmallVector<Value> blkIdxBounds;
  SmallVector<Value> blkIdxIV;
  SmallVector<Value> subBlkIdxBounds;
  SmallVector<Value> subBlkIdxIV;

  for (scf::ForallOp forall : worklist) {
    for (auto mapping : llvm::zip_equal(forall.getMappingAttr(),
                                        forall.getUpperBound(rewriter),
                                        forall.getInductionVars())) {
      auto [mappingAttr, upperBoundVal, indvarVal] = mapping;
      if (isa<HIVMBlockMappingAttr>(mappingAttr)) {
        blkIdxBounds.push_back(upperBoundVal);
        blkIdxIV.push_back(indvarVal);
      } else { // verifyHIVMMapping made sure only block mapping and subblock
               // mapping attrs are allowed
        assert(isa<HIVMSubBlockMappingAttr>(mappingAttr) &&
               "should only be sub block mapping");
        subBlkIdxBounds.push_back(upperBoundVal);
        subBlkIdxIV.push_back(indvarVal);
      }
    }
  }

  // Replace forall IV's with delinearized block idx
  if (!blkIdxIV.empty()) {
    Value blkIdx = rewriter.create<arith::IndexCastOp>(
        loc, indexTy, rewriter.create<GetBlockIdxOp>(loc));
    result.mappingId = blkIdx;
    ValueRange ivReplacements =
        rewriter
            .create<affine::AffineDelinearizeIndexOp>(loc, blkIdx, blkIdxBounds)
            .getResults();
    rewriter.replaceAllUsesWith(blkIdxIV, ivReplacements);
  }
  if (!subBlkIdxIV.empty()) {
    Value subBlkIdx = rewriter.create<arith::IndexCastOp>(
        loc, indexTy, rewriter.create<GetSubBlockIdxOp>(loc));
    result.mappingId = subBlkIdx;
    ValueRange ivReplacements = rewriter
                                    .create<affine::AffineDelinearizeIndexOp>(
                                        loc, subBlkIdx, subBlkIdxBounds)
                                    .getResults();
    rewriter.replaceAllUsesWith(subBlkIdxIV, ivReplacements);
  }

  // Move the body of each forall out of its region
  for (scf::ForallOp forall : worklist) {
    auto &targetOpList = forall->getBlock()->getOperations();
    Block *srcBlock = forall.getBody();
    rewriter.eraseOp(srcBlock->getTerminator());
    targetOpList.splice(forall->getIterator(), srcBlock->getOperations());
    rewriter.eraseOp(forall);
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mapForallToBlocksImpl(
    RewriterBase &rewriter, scf::ForallOp forallOp, ForallRewriteResult &result,
    std::optional<transform::TransformOpInterface> transformOp) {
  LDBG("Start mapForallToBlocksImpl");

  // Only allow for top-level forall's so we're addressing a whole nest at a
  // time
  if (forallOp->getParentOfType<scf::ForallOp>())
    return definiteFailureHelper(forallOp, "only valid on top-level forall ops",
                                 transformOp);

  SmallVector<scf::ForallOp> worklist;
  forallOp.walk<WalkOrder::PreOrder>(
      [&worklist](scf::ForallOp op) { worklist.push_back(op); });

  // HIVM-specific verifications.
  for (scf::ForallOp op : worklist) {
    DiagnosedSilenceableFailure diag = verifyHIVMMapping(op, transformOp);
    if (!diag.succeeded())
      return diag;
  }

  // Sort the forall's based on (optional) ordering attribute. If order not
  // present, then will by default be ascending order from outter loops to inner
  std::sort(worklist.begin(), worklist.end(),
            [](scf::ForallOp left, scf::ForallOp right) {
              auto getOrder = [](scf::ForallOp op) -> int {
                for (Attribute mapping : op.getMappingAttr()) {
                  if (auto blkMapping =
                          dyn_cast<HIVMBlockMappingAttr>(mapping))
                    return blkMapping.getOrder().value_or(0);
                }
                // Subblock mapping or no block mapping: order is not dictated.
                return -1;
              };
              return getOrder(left) < getOrder(right);
            });

  return rewriteNestedForallImpl(rewriter, forallOp, worklist, result,
                                 transformOp);
}

bool isSubBlockBindedFor(scf::ForOp op) {
  if (!op->hasAttrOfType<UnitAttr>(kMapForToForallAttrName))
    return false;

  if (!op->hasAttrOfType<ArrayAttr>(kMappingAttrName))
    return false;

  std::optional<ArrayAttr> deviceMappingAttrs =
      op->getAttrOfType<ArrayAttr>(kMappingAttrName);
  if (!deviceMappingAttrs.has_value() &&
      !llvm::hasSingleElement(deviceMappingAttrs.value()))
    return false;

  if (llvm::none_of(deviceMappingAttrs.value(), [](Attribute attr) {
        return isa<HIVMSubBlockMappingAttr>(attr);
      }))
    return false;

  return true;
}

FailureOr<scf::ForOp> findContainingSubblockLoop(Operation *op) {
  auto parentOp = op->getParentOfType<scf::ForOp>();
  if (!parentOp)
    return failure();
  if (isSubBlockBindedFor(parentOp)) {
    return parentOp;
  }
  return findContainingSubblockLoop(parentOp);
}

void removeMarkOpAttr(annotation::MarkOp markOp, ::llvm::StringLiteral attrName,
                      bool removeOp) {
  if (markOp == nullptr) {
    return;
  }
  auto attrDict = markOp->getAttrDictionary();
  if (!attrDict.empty() && attrDict.contains(attrName)) {
    markOp->removeAttr(attrName);
  }

  if (markOp.isAttrEmpty() && removeOp) {
    markOp.erase();
  }
}

void removeMarkOpAttr(annotation::MarkOp markOp, StringRef attrName,
                      RewriterBase &rewriter, bool removeOp) {
  if (markOp == nullptr) {
    return;
  }
  auto attrDict = markOp->getAttrDictionary();
  if (!attrDict.empty() && attrDict.contains(attrName)) {
    rewriter.modifyOpInPlace(markOp, [&]() { markOp->removeAttr(attrName); });
  }

  if (markOp.isAttrEmpty() && removeOp) {
    rewriter.eraseOp(markOp);
  }
}

void removeMarkOpDynamicAttr(annotation::MarkOp markOp, StringRef attrName,
                             PatternRewriter &rewriter) {
  assert(markOp && "markOp shouldn't be null.");
  auto keys = markOp.getKeys();
  auto values = markOp.getValues();
  if (!keys.has_value())
    return;

  SmallVector<Attribute> newKeys;
  SmallVector<Value> newValues;

  auto keysArray = keys.value();

  for (auto [key, value] : llvm::zip_equal(keysArray, values)) {
    StringRef currentKey = mlir::cast<StringAttr>(key).getValue();
    if (currentKey == attrName) {
      LDBG("Removing dynamic annotation: " << attrName);
    } else {
      newKeys.push_back(key);
      newValues.push_back(value);
      LDBG("Keeping dynamic annotation: " << currentKey);
    }
  }

  rewriter.modifyOpInPlace(markOp, [&]() {
    markOp.setKeysAttr(rewriter.getArrayAttr(newKeys));
    markOp.getValuesMutable().assign(newValues);
  });
  if (newKeys.empty()) {
    removeMarkOpAttr(markOp, "keys", rewriter);
  }
}

Value createTmpBufferOrTensorWithShape(PatternRewriter &rewriter, Location loc,
                                       Value source,
                                       SmallVector<int64_t> targetShape) {
#ifndef NDEBUG
  const bool isMemref = isa<MemRefType>(source.getType());
  const bool isTensor = isa<TensorType>(source.getType());
  assert((isMemref || isTensor) &&
         "Type of source should be MemRefType or TensorType!");
#endif
  Value tmp;
  if (isa<MemRefType>(source.getType())) {
    auto srcMemRefType =
        MemRefType::get(targetShape, getElementTypeOrSelf(source.getType()));
    tmp = rewriter.create<memref::AllocOp>(loc, srcMemRefType);
  } else {
    tmp = rewriter.create<tensor::EmptyOp>(
        loc, targetShape, getElementTypeOrSelf(source.getType()));
  }
  return tmp;
}

void getOpUsers(Operation *op, SmallVector<Operation *, 8> &userOps) {
  for (Operation *userOp : op->getUsers()) {
    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp,
            memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp,
            memref::ViewOp, memref::ReinterpretCastOp,
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
            bufferization::ToMemrefOp,
#else
            bufferization::ToBufferOp,
#endif
            bufferization::ToTensorOp>(userOp)) {
      getOpUsers(userOp, userOps);
    } else {
      userOps.push_back(userOp);
    }
  }
}

TModuleCoreTypeAttr getModuleCoreTypeAttr(ModuleOp mod) {
  return dyn_cast_or_null<TModuleCoreTypeAttr>(
      mod->getAttr(TModuleCoreTypeAttr::name));
}

bool isMixModule(ModuleOp mod) {
  auto attr = getModuleCoreTypeAttr(mod);
  return attr && attr.getModuleCoreType() == TModuleCoreType::MIX;
}

bool isAICModule(ModuleOp mod) {
  auto attr = getModuleCoreTypeAttr(mod);
  return attr && attr.getModuleCoreType() == TModuleCoreType::AIC;
}

bool isAIVModule(ModuleOp mod) {
  auto attr = getModuleCoreTypeAttr(mod);
  return attr && attr.getModuleCoreType() == TModuleCoreType::AIV;
}

void setModuleCoreTypeAttr(ModuleOp mod, TModuleCoreType coreType) {
  mod->setAttr(TModuleCoreTypeAttr::name,
               TModuleCoreTypeAttr::get(mod->getContext(), coreType));
}

void removeModuleCoreTypeAttr(ModuleOp mod) {
  mod->removeAttr(TModuleCoreTypeAttr::name);

  for (Operation &op : mod) {
    op.removeAttr(TModuleCoreTypeAttr::name);
  }
}

FailureOr<SmallVector<Operation *>>
traceForPotentialMatrixC(Value v, Block *storeBlock) {
  if (llvm::isa<hivm::MmadL1Op, hivm::BatchMmadL1Op>(v.getDefiningOp())) {
    return SmallVector<Operation *>{v.getDefiningOp()};
  }

  // ToDo:
  // Current supported op in traceback for matrixC
  //   - tensor::ExtractSliceOp
  //   - hivm::VCastOp
  //
  // Need to implement
  //   - quantization
  //   - relu and other activation function
  //   - more shape changing operations
  //
  auto appendCurOp = [](SmallVector<Operation *> vec,
                        Operation *op) -> SmallVector<Operation *> {
    vec.push_back(op);
    return vec;
  };
  if (auto sliceOp = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    auto nextTrace = traceForPotentialMatrixC(sliceOp.getSource(), storeBlock);
    if (succeeded(nextTrace))
      return appendCurOp(*nextTrace, sliceOp);
  }
  if (auto vcastOp = v.getDefiningOp<hivm::VCastOp>()) {
    // FixMe:
    // Current restrictiton is that, cast op to be erased, must be in same
    // block of store operation.
    if (vcastOp->getBlock() != storeBlock)
      return failure();

    auto nextTrace =
        traceForPotentialMatrixC(vcastOp.getSingleSrc(), storeBlock);
    if (succeeded(nextTrace))
      return appendCurOp(*nextTrace, vcastOp);
  }

  // Currently, just relay potential operation inner control flow region.
  if (auto forOp = v.getDefiningOp<scf::ForOp>()) {
    // get result index
    unsigned index = cast<OpResult>(v).getResultNumber();
    Value yieldedValue = forOp.getYieldedValues()[index];
    return traceForPotentialMatrixC(yieldedValue, storeBlock);
  }
  if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
    unsigned index = cast<OpResult>(v).getResultNumber();
    Block &thenBlock = ifOp.getThenRegion().front();
    unsigned thenYieldNum = thenBlock.getTerminator()->getNumOperands();
    if (index < thenYieldNum) {
      Value yieldedValue = thenBlock.getTerminator()->getOperand(index);
      return traceForPotentialMatrixC(yieldedValue, storeBlock);
    } else {
      Value yieldedValue =
          ifOp.getElseRegion().front().getTerminator()->getOperand(
              index - thenYieldNum);
      return traceForPotentialMatrixC(yieldedValue, storeBlock);
    }
  }

  return failure();
}

SmallVector<Value> getTensorDynamicValues(OpBuilder &builder, Location loc,
                                          Value src) {
  SmallVector<Value> dynamicSizes;
  if (auto reifiableOp = dyn_cast<tensor::EmptyOp>(src.getDefiningOp())) {
    ReifiedRankedShapedTypeDims outputShapes;
    if (failed(reifyResultShapes(builder, reifiableOp, outputShapes))) {
      dynamicSizes = tensor::createDynamicDimValues(builder, loc, src);
    } else {
      for (auto dimValue : outputShapes[0]) {
        if (!getConstantIntValue(dimValue).has_value()) {
          dynamicSizes.push_back(
              getValueOrCreateConstantIndexOp(builder, loc, dimValue));
        }
      }
    }
  } else {
    dynamicSizes = tensor::createDynamicDimValues(builder, loc, src);
  }
  return dynamicSizes;
}

Value createAllocLocalWorkSpace(OpBuilder &builder, Location loc,
                                ArrayRef<int64_t> shape, Type elementType) {
  assert(!ShapedType::isDynamicShape(shape) &&
         "AllocWorkspaceOp only supports static shape");
  Type allocWorkspaceType = MemRefType::get(shape, elementType);

  auto allocWorkspaceOp =
      builder.create<bishengir::memref_ext::AllocWorkspaceOp>(
          loc, allocWorkspaceType,
          /*workspaceArg*/ Value(), ValueRange{},
          /*offset*/ ValueRange{});
  return allocWorkspaceOp.getMemref();
}

Value getLocalWorkSpaceTensor(PatternRewriter &rewriter, Location loc,
                              ArrayRef<int64_t> targetShapes,
                              Type elementType) {
#ifndef NDEBUG
  std::optional<int64_t> totalStaticSize =
      utils::getStaticTotalSize(targetShapes);
  // TODO：support dynamic shape.
  assert(totalStaticSize.has_value() && "Can only handle static shape");
#endif

  // 1. Get AllocWorkspaceOp of current block
  Value localWorkSpace = createAllocLocalWorkSpace(
      rewriter, loc, targetShapes, elementType);

  // 2. Use bufferization::ToTensorOp to convert current workspace to tensor
  auto toTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, localWorkSpace, true, true);
  return toTensor;
}

Value createAllocLocalWorkSpace(OpBuilder &builder, Location loc,
                                SmallVector<int64_t> targetShape,
                                SmallVector<Value> dynamicSizes,
                                Type elementType) {
  Type allocWorkspaceType = MemRefType::get(targetShape, elementType);

  auto allocWorkspaceOp =
      builder.create<bishengir::memref_ext::AllocWorkspaceOp>(
          loc, allocWorkspaceType,
          /*workspaceArg*/ Value(), ValueRange{dynamicSizes},
          /*offset*/ ValueRange{});
  return allocWorkspaceOp.getMemref();
}

Value getLocalWorkSpaceTensor(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> targetShape,
    ArrayRef<Value> dynamicShape, Type elementType,
    std::optional<ArrayRef<int64_t>> staticAllocShape) {
  // 1. Get AllocWorkspaceOp of current block
  Value localWorkSpace = createAllocLocalWorkSpace(
      rewriter, loc, SmallVector<int64_t>(targetShape),
      SmallVector<Value>(dynamicShape), elementType);

  // 2. When staticAllocShape is provided, mark static alloc size for dynamic
  // tensor case
  if (staticAllocShape.has_value()) {
    auto localWorkSpaceOp =
        localWorkSpace.getDefiningOp<bishengir::memref_ext::AllocWorkspaceOp>();
    auto maybeStaticTotalSize =
        utils::getStaticTotalSizeInBits(*staticAllocShape, elementType);
    if (!maybeStaticTotalSize.has_value()) {
      emitError(localWorkSpaceOp->getLoc(), "shape has dynamic dimension");
      llvm::report_fatal_error("shape has dynamic dimension");
      return nullptr;
    }
    int64_t allocSizeInByte = maybeStaticTotalSize.value() / utils::kBitsToByte;
    auto tmpMarkOp = rewriter.create<annotation::MarkOp>(
        localWorkSpaceOp->getLoc(), localWorkSpaceOp->getResult(0));
    tmpMarkOp->setAttr(hivm::kBufferSizeInByteAttr,
                       rewriter.getI64IntegerAttr(allocSizeInByte));
  }

  // 3. Use bufferization::ToTensorOp to convert current workspace to tensor
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
  auto toTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, localWorkSpace, true, true);
#else
  // return tensor type
  auto tensorType = RankedTensorType::get(targetShape, elementType);
  auto toTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, tensorType, localWorkSpace, true, true);
#endif
  return toTensor;
}

hivm::CreateSyncBlockLockOp createSyncBlockLockVar(OpBuilder &builder,
                                                   Location loc) {
  SmallVector<int64_t> shape = {1};
  auto elementType = builder.getI64Type();
  Type memrefType = MemRefType::get(shape, elementType);

  auto createSyncBlockLockOp =
      builder.create<hivm::CreateSyncBlockLockOp>(loc, memrefType,
                                                  /*workspaceArg*/ Value());
  return createSyncBlockLockOp;
}

std::vector<std::pair<Value, Value>> getOperationAliasInfo(Operation *op) {
  std::vector<std::pair<Value, Value>> result;
  TypeSwitch<Operation *, void>(op)
      .Case([&](arith::SelectOp op) {
        result.emplace_back(op.getResult(), op.getTrueValue());
        result.emplace_back(op.getResult(), op.getFalseValue());
      })
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
      .Case([&](bufferization::ToMemrefOp op) {
        result.emplace_back(op.getResult(), op.getOperand());
      })
#else
      .Case([&](bufferization::ToBufferOp op) {
        result.emplace_back(op.getResult(), op.getOperand());
      })
#endif
      .Case([&](bufferization::ToTensorOp op) {
        result.emplace_back(op.getResult(), op.getOperand());
      })
      .Case([&](hivm::BitcastOp op) {
        result.emplace_back(op.getResult(), op.getSrc());
      })
      .Case([&](memref::CastOp op) {
        result.emplace_back(op.getResult(), op.getViewSource());
      })
      .Case([&](memref::CollapseShapeOp op) {
        result.emplace_back(op.getResult(), op.getViewSource());
      })
      .Case([&](memref::ExpandShapeOp op) {
        result.emplace_back(op.getResult(), op.getViewSource());
      })
      .Case([&](memref::ExtractStridedMetadataOp op) {
        result.emplace_back(op.getBaseBuffer(), op.getViewSource());
      })
      .Case([&](memref::MemorySpaceCastOp op) {
        result.emplace_back(op.getResult(), op.getOperand());
      })
      .Case([&](memref::ReinterpretCastOp op) {
        result.emplace_back(op.getResult(), op.getViewSource());
      })
      .Case([&](memref::ReshapeOp op) {
        result.emplace_back(op.getResult(), op.getViewSource());
      })
      .Case([&](memref::SubViewOp op) {
        result.emplace_back(op.getResult(), op.getViewSource());
      })
      .Case([&](memref::TransposeOp op) {
        result.emplace_back(op.getResult(), op.getIn());
      })
      .Case([&](memref::ViewOp op) {
        result.emplace_back(op.getResult(), op.getViewSource());
      })
      .Case([&](tensor::CollapseShapeOp op) {
        result.emplace_back(op.getResult(), op.getSrc());
      })
      .Case([&](tensor::ExpandShapeOp op) {
        result.emplace_back(op.getResult(), op.getSrc());
      })
      .Case([&](tensor::ExtractSliceOp op) {
        result.emplace_back(op.getResult(), op.getSource());
      })
      .Case([&](scope::ScopeOp scopeOp) {
        auto returnOp =
            cast<scope::ReturnOp>(scopeOp.getRegion().front().getTerminator());
        for (auto [res, arg] :
             llvm::zip_equal(scopeOp->getResults(), returnOp->getOperands())) {
          result.emplace_back(res, arg);
        }
      });
  return result;
}

std::optional<int64_t> GetBufferBitSize(Value buffer) {
  auto memRefType = cast<MemRefType>(buffer.getType());
  if (!memRefType) {
    return {};
  }
  int64_t bufferConstBitSize =
      static_cast<int64_t>(memRefType.getElementTypeBitWidth());
  for (auto &val : memRefType.getShape()) {
    if (val == ShapedType::kDynamic) {
      return ShapedType::kDynamic;
    }
    bufferConstBitSize *= val;
  }
  return bufferConstBitSize;
}

AlignKind isBrcOpAligned(VBrcOp vbrcOp, int dim, int rank) {
  AxisKind axisKind = utils::getAxisKind(dim, rank);
  AxisKind outlinedAxisKind = utils::getOutlinedAxisKind(dim, rank);
  // Collect the list of memrefType that need to be checked
  std::vector<MemRefType> memrefTypeList = {};
  Type srcType = vbrcOp.getSrc().getType();
  auto dstType = cast<MemRefType>(vbrcOp.getDst().getType());
  unsigned resWidth = dstType.getElementType().getIntOrFloatBitWidth();
  unsigned alignment =
      (util::INTR_BYTES_PER_BLOCK * utils::INTR_BITS_PER_BYTE) / resWidth;
  if (!isScalarLike(srcType))
    memrefTypeList.push_back(cast<MemRefType>(srcType));
  memrefTypeList.push_back(dstType);
  // Collect the list of align kind
  std::vector<AlignKind> alignKindList = {};
  dim = axisKind == AxisKind::LAST ? dstType.getRank() - 2 : dim;
  for (auto memrefType : memrefTypeList) {
    auto layout = dyn_cast<StridedLayoutAttr>(memrefType.getLayout());
    // Return unknown if [dim ... end] has any dynamic shape
    for (auto i = dim + 1; i < rank; i++)
      if (memrefType.isDynamicDim(i))
        return AlignKind::UNKNOWN;
    if (layout &&
        ShapedType::isDynamicShape(layout.getStrides().slice(dim, rank - dim)))
      return AlignKind::UNKNOWN;
    // Transforming the static memrefType to the alignment
    if (layout) {
      alignKindList.push_back(layout.getStrides()[dim] % alignment == 0
                                  ? AlignKind::ALIGN
                                  : AlignKind::UNALIGNED);
    } else {
      int64_t expectedStride = 1;
      for (auto i = dim + 1; i < rank; i++)
        expectedStride *= memrefType.getShape()[i];
      alignKindList.push_back(expectedStride % alignment == 0
                                  ? AlignKind::ALIGN
                                  : AlignKind::UNALIGNED);
    }
  }
  if (outlinedAxisKind == AxisKind::MIDDLE)
    assert(alignKindList.front() == AlignKind::ALIGN &&
           "Middle axis\'s align kind must be \'ALIGN\'");
  if (llvm::all_equal(alignKindList))
    return alignKindList.front();
  return AlignKind::UNKNOWN;
}

void setSubBlockMapping(RewriterBase &rewriter, Operation *loop) {
  rewriter.modifyOpInPlace(loop, [&]() {
    loop->setAttr(kMapForToForallAttrName, UnitAttr::get(loop->getContext()));
    SmallVector<Attribute> MappingNames;
    MappingNames.push_back(HIVMSubBlockMappingAttr::get(loop->getContext(),
                                                        hivm::MappingId::DimX));
    loop->setAttr(kMappingAttrName,
                  ArrayAttr::get(loop->getContext(), MappingNames));
  });
}

LoopLikeOpInterface getParentLoop(Value val) {
  auto *valDefOp = val.getDefiningOp();
  if (!valDefOp)
    llvm::report_fatal_error("val should have defining op.");

  // Firstly, get parent loop
  LoopLikeOpInterface parentLoop =
      valDefOp->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }

  // Need to determine whether val is yielded by the loop.
  auto yieldedValues = parentLoop.getYieldedValues();
  if (yieldedValues.empty())
    return parentLoop;

  auto idxLoopRes = getYieldValueIdx(val, yieldedValues);
  if (idxLoopRes.has_value()) {
    // The val is yielded by loop, so need to find parent of parent loop.
    //
    // Some loop ops (e.g. scf.while) do not expose their results through the
    // LoopLikeOpInterface -- getLoopResults() returns std::nullopt. In that
    // case we cannot follow the value outward through a loop result, so stop
    // here and keep the anchor we already found. (For scf.while the yielded
    // value maps to a before-region iter_arg rather than a loop result, so
    // there is nothing to track further outward anyway.)
    auto loopResults = parentLoop.getLoopResults();
    if (!loopResults || *idxLoopRes >= static_cast<int>(loopResults->size()))
      return parentLoop;
    auto res = (*loopResults)[*idxLoopRes];
    return getParentLoop(res);
  }

  // Need to determine whether val is yielded by if/else.
  auto parentIf = valDefOp->getParentOfType<scf::IfOp>();
  if (!parentIf || parentIf.getResults().empty())
    return parentLoop;

  auto thenYieldOp = parentIf.thenYield();
  auto thenYieldOpers = thenYieldOp.getOperands();

  auto idxThenYielded = getYieldValueIdx(val, thenYieldOpers);
  if (idxThenYielded.has_value()) {
    // The val is yielded by ifOp, need to find parent loop of ifOp's result
    auto res = parentIf.getResults()[*idxThenYielded];
    return getParentLoop(res);
  }

  if (!parentIf.getElseRegion().empty()) {
    auto elseYieldOp = parentIf.elseYield();
    auto elseYieldOpers = elseYieldOp.getOperands();
    auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
    if (idxElseYielded.has_value()) {
      auto res = parentIf.getResults()[*idxElseYielded];
      return getParentLoop(res);
    }
  }

  return parentLoop;
}

Value createNestedIndexModular(OpBuilder &builder, Operation *op, int modular) {
  LoopLikeOpInterface parentLoop = getParentLoop(op->getResult(0));
  assert(parentLoop &&
         " ptrCastOp has no proper parent loop to do multi buffer");

  auto loopInfoVec = getLoopsInfo(parentLoop);

  // Insert at the beginning of the For loop.
  auto forOp = cast<scf::ForOp>(parentLoop.getOperation());
  builder.setInsertionPointToStart(forOp.getBody());
  return createNestedIndexModularUsingLoopInfo(builder, forOp->getLoc(),
                                               loopInfoVec, modular);
}

Value createNestedIndexModular(OpBuilder &builder, LoopLikeOpInterface loopOp,
                               int modular) {
  auto loopInfoVec = getLoopsInfo(loopOp);
  // Insert at the beginning of the For loop.
  auto forOp = dyn_cast<scf::ForOp>(loopOp.getOperation());
  assert(forOp != nullptr);
  builder.setInsertionPointToStart(forOp.getBody());
  return createNestedIndexModularUsingLoopInfo(builder, forOp->getLoc(),
                                               loopInfoVec, modular);
}

Value createNestedIndexForOp(OpBuilder &builder, Operation *operation) {
  LoopLikeOpInterface parentLoop =
      operation->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }
  assert(parentLoop &&
         " ptrCastOp has no proper parent loop to do multi buffer");

  auto loopInfoVec = getLoopsInfo(parentLoop);

  // Insert at the beginning of the For loop.
  auto forOp = cast<scf::ForOp>(parentLoop.getOperation());
  builder.setInsertionPointToStart(forOp.getBody());
  return createNestedIndexModularUsingLoopInfo(builder, forOp->getLoc(),
                                               loopInfoVec, -1);
}

static std::optional<BlockArgument> traceBlockArgument(BlockArgument ba) {
  auto forOp = dyn_cast<scf::ForOp>(ba.getParentBlock()->getParentOp());
  if (!forOp)
    return ba;

  for (const auto &[regionArg, initArg] :
       llvm::zip_equal(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
    if (regionArg == ba)
      return traceValueToBlockArgument(initArg);
  }
  return std::nullopt;
}

std::optional<BlockArgument> traceValueToBlockArgument(Value value) {
  if (auto ba = dyn_cast<BlockArgument>(value))
    return traceBlockArgument(ba);
  if (auto subview = value.getDefiningOp<memref::SubViewOp>())
    return traceValueToBlockArgument(subview.getViewSource());
  if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>())
    return traceValueToBlockArgument(slice.getSource());
  return std::nullopt;
}

SmallVector<unsigned> traceVFWriteOpArgIds(func::CallOp callOp) {
  SymbolRefAttr calleeName = callOp.getCalleeAttr();
  SymbolTableCollection symbolTable;
  auto calleeFunc =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(callOp, calleeName);
  if (!calleeFunc)
    return {};

  func::ReturnOp returnOp;
  calleeFunc.walk([&](func::ReturnOp op) {
    returnOp = op;
    return WalkResult::interrupt();
  });
  if (!returnOp)
    return {};

  SmallVector<unsigned> writeOpArgIds;
  for (Value returned : returnOp->getOperands()) {
    auto maybeWriteOp = traceDefOp<vector::TransferWriteOp>(returned);
    if (!maybeWriteOp.has_value())
      continue;
    auto write = cast<vector::TransferWriteOp>(*maybeWriteOp);
    auto arg = traceValueToBlockArgument(write.getSource());
    if (!arg.has_value())
      continue;
    writeOpArgIds.push_back(arg->getArgNumber());
  }
  return writeOpArgIds;
}

uint32_t getHWAlignBytes(Attribute spaceAttr) {
  auto hivmSpace = dyn_cast<hivm::AddressSpaceAttr>(spaceAttr);
  assert(hivmSpace && "Empty address space attr");
  switch (hivmSpace.getAddressSpace()) {
  case hivm::AddressSpace::UB:
  case hivm::AddressSpace::L1:
    return hivm::util::BL;
  default:
    llvm_unreachable("Unsupported address space");
  }
}

std::optional<uint32_t> getHWAlignBytes(Type t) {
  auto memrefType = dyn_cast<MemRefType>(t);
  if (!memrefType) {
    return std::nullopt;
  }
  auto hwAlignBytes = getHWAlignBytes(memrefType.getMemorySpace());
  return hwAlignBytes;
}

namespace util {
//===----------------------------------------------------------------------===//
// This file contains code from the LLVM Project.
// Original License: Apache License v2.0 with LLVM Exceptions
// Original Copyright: NA
// Original Source:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp
//===----------------------------------------------------------------------===//

/// A more STRICT version of origin `computeCollapsedLayoutMap` in
/// `MemRefOps.cpp`, which do not skip dimensions of size 1.
/// Compute the layout map after collapsing a given source MemRef type
/// with the specified reassociation indices.
static FailureOr<StridedLayoutAttr>
computeCollapsedLayoutMap(MemRefType srcType,
                          ArrayRef<ReassociationIndices> reassociation,
                          bool strict = false) {
  int64_t srcOffset;
  SmallVector<int64_t> srcStrides;
  auto srcShape = srcType.getShape();
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
  if (failed(getStridesAndOffset(srcType, srcStrides, srcOffset)))
#else
  if (failed(srcType.getStridesAndOffset(srcStrides, srcOffset)))
#endif
    return failure();

  // The result stride of a reassociation group is the stride of the last
  // entry of the reassociation. (TODO: Should be the minimum stride in the
  // reassociation because strides are not necessarily sorted. E.g., when
  // using memref.transpose.) Dimensions of size 1 MUST NOT be skipped, their
  // strides are NOT meaningless and MUST NOT have any arbitrary value.
  SmallVector<int64_t> resultStrides;
  resultStrides.reserve(reassociation.size());
  for (const ReassociationIndices &reassoc : reassociation) {
    ArrayRef<int64_t> ref = llvm::ArrayRef(reassoc);
    // CHANGE: DON'T SKIP DIMANSIONS OF SIZE 1 UNDER STRICT MODE
    if (!strict) {
      while (srcShape[ref.back()] == 1 && ref.size() > 1)
        ref = ref.drop_back();
    }
    if (!ShapedType::isDynamic(srcShape[ref.back()]) || ref.size() == 1) {
      resultStrides.push_back(srcStrides[ref.back()]);
    } else {
      // Dynamically-sized dims may turn out to be dims of size 1 at runtime,
      // so the corresponding stride may have to be skipped. (See above
      // comment.) Therefore, the result stride cannot be statically
      // determined and must be dynamic.
      resultStrides.push_back(ShapedType::kDynamic);
    }
  }

  // Validate that each reassociation group is contiguous.
  unsigned resultStrideIndex = resultStrides.size() - 1;
  for (const ReassociationIndices &reassoc : llvm::reverse(reassociation)) {
    auto trailingReassocs = ArrayRef<int64_t>(reassoc).drop_front();
    auto stride = SaturatedInteger::wrap(resultStrides[resultStrideIndex--]);
    for (int64_t idx : llvm::reverse(trailingReassocs)) {
      stride = stride * SaturatedInteger::wrap(srcShape[idx]);

      // Both source and result stride must have the same static value. In
      // that case, we can be sure, that the dimensions are collapsible
      // (because they are contiguous). If `strict = false` (default during op
      // verification), we accept cases where one or both strides are dynamic.
      // This is best effort: We reject ops where obviously non-contiguous
      // dims are collapsed, but accept ops where we cannot be sure
      // statically. Such ops may fail at runtime. See the op documentation
      // for details.
      auto srcStride = SaturatedInteger::wrap(srcStrides[idx - 1]);
      if (strict && (stride.saturated || srcStride.saturated))
        return failure();

      // CHANGE: Dimensions of size 1 MUST NOT be skipped, their strides are
      // NOT meaningless and could NOT have any arbitrary value.
      if (!strict && srcShape[idx - 1] == 1)
        continue;

      if (!stride.saturated && !srcStride.saturated && stride != srcStride)
        return failure();
    }
  }
  return StridedLayoutAttr::get(srcType.getContext(), srcOffset, resultStrides);
}

bool isGuaranteedCollapsibleStrictly(
    MemRefType srcType, ArrayRef<ReassociationIndices> reassociation) {
  // MemRefs with identity layout are always collapsible.
  if (srcType.getLayout().isIdentity())
    return true;

  return succeeded(computeCollapsedLayoutMap(srcType, reassociation,
                                             /*strict=*/true));
}

bool isGuaranteedCollapsibleUnStrictly(
    MemRefType srcType, ArrayRef<ReassociationIndices> reassociation) {
  // MemRefs with identity layout are always collapsible.
  if (srcType.getLayout().isIdentity())
    return true;

  return succeeded(computeCollapsedLayoutMap(srcType, reassociation,
                                             /*strict=*/false));
}

SmallVector<MemRefType> getMemRefTypes(TypeRange types) {
  auto filteredTypes = llvm::make_filter_range(types, [](Type t) {
    if (!isa<MemRefType>(t)) {
      return false;
    };
    auto memrefType = cast<MemRefType>(t);
    if (!memrefType.hasRank() || memrefType.getRank() == 0) {
      return false;
    }

    return true;
  });
  if (filteredTypes.empty()) {
    return {};
  }

  SmallVector<MemRefType> memrefTypes;
  llvm::transform(filteredTypes, std::back_inserter(memrefTypes),
                  [](Type t) { return cast<MemRefType>(t); });

  return memrefTypes;
}

bool isAllSameRank(const SmallVectorImpl<MemRefType> &memrefTypes) {
  if (memrefTypes.empty()) {
    return true;
  }

  if (!memrefTypes[0].hasRank()) {
    return false;
  }

  auto rank = memrefTypes[0].getRank();
  return llvm::all_of(memrefTypes, [&](MemRefType memRefType) {
    return memRefType.getRank() == rank;
  });
}

bool isLastDimContiguous(Value operand) {
  auto operType = operand.getType();
  if (operType.isIntOrIndexOrFloat()) {
    return true;
  }

  auto mem = cast<MemRefType>(operType);
  auto shape = mem.getShape();
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
  auto [strides, offset] = getStridesAndOffset(mem);
#else
  auto [strides, offset] = mem.getStridesAndOffset();
#endif
  assert(!strides.empty() && "strides should not be empty.");

  return *shape.rbegin() == 1 || *strides.rbegin() == 1;
}

bool isGMPointerCastOp(Operation *op) {
  auto pointerCastOp = dyn_cast_or_null<hivm::PointerCastOp>(op);
  if (!pointerCastOp)
    return false;
  auto memorySpaceAttr = hivm::AddressSpaceAttr::getMnemonic();
  auto annotationOp =
      utils::getAnnotateOpWithAttr(pointerCastOp, memorySpaceAttr);
  if (!annotationOp.has_value())
    return false;
  auto markOp = cast<annotation::MarkOp>(annotationOp.value());
  AddressSpaceAttr memSpaceAttr =
      cast<AddressSpaceAttr>(markOp.getStaticAttrValue(memorySpaceAttr));
  return memSpaceAttr.getAddressSpace() == hivm::AddressSpace::GM;
}

bool isSIMTVF(Operation *op) {
  std::optional<VFMode> vfMode = std::nullopt;
  if (const auto vfModeAttr = op->getAttrOfType<VFModeAttr>(VFModeAttr::name)) {
    vfMode = vfModeAttr.getValue();
  }
  return vfMode == hivm::VFMode::SIMT;
}

bool isArgminOrArgmax(ReduceOperation op) {
  return op == ReduceOperation::min_with_index_left ||
         op == ReduceOperation::max_with_index_left ||
         op == ReduceOperation::min_with_index_right ||
         op == ReduceOperation::max_with_index_right;
}

void validateMultiBufferAttr(mlir::DictionaryAttr attrDict) {
  auto attr = attrDict.get(hivm::MultiBufferAttr::name);
  if (!attr)
    return;
  mlir::IntegerAttr intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
  if (!intAttr) {
    llvm::report_fatal_error("MultiBufferAttr illegal!!!");
  }
  int64_t attrValue = intAttr.getValue().getSExtValue();
  if (attrValue < 1) {
    llvm::report_fatal_error("MultiBufferAttr should be >= 1!!");
  }
}

/// Trims non-scalable one dimensions from `oldType` and returns the result
/// type. Copy from
/// mlir/lib/Dialect/Vector/Transforms/VectorTransferOpTransforms.cpp
VectorType trimNonScalableUnitDims(VectorType oldType) {
  SmallVector<int64_t> newShape;
  SmallVector<bool> newScalableDims;
  for (auto [dimIdx, dimSize] : llvm::enumerate(oldType.getShape())) {
    if (dimSize == 1 && !oldType.getScalableDims()[dimIdx])
      continue;
    newShape.push_back(dimSize);
    newScalableDims.push_back(oldType.getScalableDims()[dimIdx]);
  }
  return VectorType::get(newShape, oldType.getElementType(), newScalableDims);
}

bool isOneDimLikeVecType(VectorType vecType) {
  if (vecType.getRank() == 1)
    return true;
  auto shape = vecType.getShape();
  for (int64_t i = 0, e = vecType.getRank() - 1; i < e; ++i) {
    if (shape[i] != 1)
      return false;
  }
  return true;
}

static bool canCombineOneSide(const SmallVectorImpl<MemRefType> &memrefTypes,
                              const ReassociationIndices &curIndexGroup,
                              int indexStartPosition) {
  return llvm::all_of(memrefTypes, [&](const MemRefType &curMemRef) {
    ArrayRef<int64_t> curShapes = curMemRef.getShape();
    int curIndexPosition = indexStartPosition;
    for (int index : curIndexGroup) {
      if (index != curIndexPosition) {
        // do not combine transposed dims
        return false;
      }
      curIndexPosition++;
      if (curShapes[index] != 1) {
        // do not combine dim with shape more than 1
        return false;
      }
    }
    return true;
  });
}

/// if the given index group is not transpose reassociations, and the shape of
/// each memref type corresponding to each index in the reassociation indices
/// group is 1, the given index group can be combined.
static bool canCombine(const SmallVectorImpl<MemRefType> &memrefTypes,
                       const ReassociationIndices &lhs,
                       const ReassociationIndices &rhs, int64_t lhsStartIndex,
                       int64_t rhsStartIndex) {
  // check whether one reassociation index group can be combined to another
  bool anySideCombinable = canCombineOneSide(memrefTypes, rhs, rhsStartIndex) ||
                           canCombineOneSide(memrefTypes, lhs, lhsStartIndex);
  if (!anySideCombinable) {
    return false;
  }
  // check continuous between index groups
  ReassociationIndices combinedIndexGroup = lhs;
  combinedIndexGroup.append(rhs);
  return llvm::all_of(memrefTypes, [&](const MemRefType &curMemRef) {
    return util::isGuaranteedCollapsibleStrictly(curMemRef,
                                                 {combinedIndexGroup});
  });
}

SmallVector<ReassociationIndices> combineReassociationGroups(
    const SmallVectorImpl<MemRefType> &memrefTypes,
    const SmallVectorImpl<ReassociationIndices> &oldReassociations) {
  SmallVector<ReassociationIndices> reassociations(oldReassociations.begin(),
                                                   oldReassociations.end());
  SmallVector<ReassociationIndices> combinedReassociations;
  auto *prevIt = reassociations.begin();
  // used for checking whether contains transposed dim
  int64_t size = static_cast<int64_t>(reassociations.front().size());

  // combine continuous reassociation index groups
  for (auto *curIt = reassociations.begin() + 1; curIt < reassociations.end();
       ++curIt) {
    if (canCombine(memrefTypes, *prevIt, *curIt, size - prevIt->size(), size)) {
      // combine current reassociations into current combined group if
      // possible
      prevIt->append(*curIt);
    } else {
      // otherwise, save current combined group and create a new group
      combinedReassociations.push_back(*prevIt);
      prevIt = curIt;
    }
    size += static_cast<int64_t>(curIt->size());
  }
  // save remained reassociations
  if (!prevIt->empty()) {
    combinedReassociations.push_back(*prevIt);
  }
  return combinedReassociations;
}

SmallVector<ReassociationIndices> getContinuousReassociation(
    const SmallVectorImpl<MemRefType> &memrefTypes,
    const SmallVectorImpl<ReassociationIndices> &reassociations) {
  // no more refine if all dims collapsible within origin reassociation groups
  bool isCollapsedIntraGroup = llvm::all_of(memrefTypes, [&](MemRefType t) {
    return util::isGuaranteedCollapsibleStrictly(t, reassociations);
  });
  if (isCollapsedIntraGroup) {
    auto combinedReassociations =
        combineReassociationGroups(memrefTypes, reassociations);
    return combinedReassociations;
  }

  // refine reassociations into smaller parts for best effort collapse
  SmallVector<ReassociationIndices> refinedReassociations;
  for (const ReassociationIndices &group : reassociations) {
    if (group.empty()) {
      continue;
    }
    ReassociationIndices refinedGroup{group.front()};
    const auto *prevIt = group.begin();
    for (const auto *curIt = prevIt + 1; curIt < group.end();
         ++curIt, ++prevIt) {
      assert(*curIt > *prevIt &&
             "reassociation indices must be ascending order");
      // pairwise check for collapsible
      SmallVector<ReassociationIndices> maybeCollapse{{*prevIt, *curIt}};
      bool pairwiseCollapsed = llvm::all_of(memrefTypes, [&](MemRefType t) {
        return util::isGuaranteedCollapsibleStrictly(t, maybeCollapse);
      });
      if (pairwiseCollapsed) {
        // if prev and cur reasso indices can be collapsed, add to new group
        refinedGroup.push_back(*curIt);
      } else {
        // if cannot collapse, split cur indice to another new group
        refinedReassociations.push_back(refinedGroup);
        refinedGroup.clear();
        refinedGroup.push_back(*curIt);
      }
    }
    if (!refinedGroup.empty()) {
      refinedReassociations.push_back(refinedGroup);
    }
  }

  // further combine inter-group reassociations
  return combineReassociationGroups(memrefTypes, refinedReassociations);
}

SmallVector<ReassociationIndices>
getContinuousReassociation(const SmallVectorImpl<MemRefType> &memrefTypes,
                           ArrayRef<int64_t> reshapeDims,
                           ArrayRef<int64_t> permutations) {
  if (!isAllSameRank(memrefTypes)) {
    LDBG("MemrefTypes of operands have different rank");
    return {};
  }

  // get reassociations by dividing reshape group
  SmallVector<ReassociationIndices> reassociations;
  ReassociationIndices reassociation{0};
  bool prevIsReshape = std::count(reshapeDims.begin(), reshapeDims.end(), 0);
  auto rank = memrefTypes[0].getRank();
  for (int64_t i = 1; i < rank; i++) {
    bool curIsReshape = std::count(reshapeDims.begin(), reshapeDims.end(), i);
    if (curIsReshape == prevIsReshape &&
        (permutations.empty() || permutations[i] == permutations[i - 1] + 1)) {
      reassociation.push_back(i);
    } else {
      reassociations.push_back(reassociation);
      reassociation.clear();
      reassociation.push_back(i);
      prevIsReshape = curIsReshape;
    }
  }
  if (!reassociation.empty()) {
    reassociations.push_back(reassociation);
  }

  // cut the un-continuous ones into different group
  auto continuousReassociations =
      util::getContinuousReassociation(memrefTypes, reassociations);
  assert(!continuousReassociations.empty());
  return continuousReassociations;
}
} // namespace util
} // namespace hivm
} // namespace mlir
