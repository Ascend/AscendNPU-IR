//===- Utils.cpp - Utilities to support the HIVM dialect ------------------===//
//
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
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
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
    llvm_unreachable("parentOp doesn't exist");
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
        return getMemRefAlloc(op.getMemref());
      })
      .Default([&](Operation *op) {
        op->emitOpError("Unsupported op for finding the root alloc.");
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
/// \return IndexCastOp of affineApply
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
    return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                              affineApply);
  }

  targetExpr = targetExpr % modular;

  Value affineApply = mlir::affine::makeComposedAffineApply(
      builder, loc, targetExpr, ArrayRef(symbolValueVec));
  return builder.create<arith::IndexCastOp>(loc, builder.getI1Type(),
                                            affineApply);
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
  llvm_unreachable("Currently only support (UB | L1 | L0C) allocation");
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
              int leftOrder = -1;
              for (Attribute mapping : left.getMappingAttr()) {
                if (auto blkMapping = dyn_cast<HIVMBlockMappingAttr>(mapping)) {
                  leftOrder = blkMapping.getOrder().value_or(0);
                  break;
                }
              }
              for (Attribute mapping : right.getMappingAttr()) {
                if (auto blkMapping = dyn_cast<HIVMBlockMappingAttr>(mapping))
                  return leftOrder < blkMapping.getOrder().value_or(0);
              }
              // Expecting subblock mapping here, where the order is not
              // dictated
              return true;
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

  if (markOp->getAttrDictionary().empty() && removeOp) {
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

  if (markOp->getAttrDictionary().empty() && removeOp) {
    rewriter.eraseOp(markOp);
  }
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
            bufferization::ToMemrefOp, bufferization::ToTensorOp>(userOp)) {
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

bool isLastDimTranspose(hivm::VTransposeOp op) {
  SmallVector<int64_t> transposeLoopDims;
  op.getTransposeLoopDims(transposeLoopDims);
  auto dimSize = op.getNumLoops();
  if (std::find(transposeLoopDims.begin(), transposeLoopDims.end(),
                dimSize - 1) == transposeLoopDims.end()) {
    return false;
  }
  return true;
}

bool isLastTwoAxesTranspose(hivm::VTransposeOp op) {
  if (!isLastDimTranspose(op)) {
    return false;
  }

  SmallVector<int64_t> transposeLoopDims;
  op.getTransposeLoopDims(transposeLoopDims);
  return transposeLoopDims[0] == (transposeLoopDims[1] - 1);
}

Value createAllocLocalWorkSpace(OpBuilder &builder, Location loc,
                                SmallVector<int64_t> shape, Type elementType) {
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
      rewriter, loc, SmallVector<int64_t>(targetShapes), elementType);

  // 2. Use bufferization::ToTensorOp to convert current workspace to tensor
  auto toTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, localWorkSpace, true, true);
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
  if (auto subViewOp = dyn_cast<memref::SubViewOp>(op)) {
    result.emplace_back(subViewOp.getResult(), subViewOp.getViewSource());
  } else if (auto extSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    result.emplace_back(extSliceOp.getResult(), extSliceOp.getSource());
  } else if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(op)) {
    result.emplace_back(collapseShapeOp.getResult(),
                        collapseShapeOp.getViewSource());
  } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(op)) {
    result.emplace_back(expandShapeOp.getResult(),
                        expandShapeOp.getViewSource());
  } else if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
    result.emplace_back(expandShapeOp.getResult(), expandShapeOp.getSrc());
  } else if (auto viewOp = dyn_cast<memref::ViewOp>(op)) {
    result.emplace_back(viewOp.getResult(), viewOp.getViewSource());
  } else if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    result.emplace_back(reinterpretCastOp.getResult(),
                        reinterpretCastOp.getViewSource());
  } else if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(op)) {
    result.emplace_back(reshapeOp.getResult(), reshapeOp.getViewSource());
  } else if (auto castOp = dyn_cast<memref::CastOp>(op)) {
    result.emplace_back(castOp.getResult(), castOp.getViewSource());
  } else if (auto extractStridedMetadataOp =
                 dyn_cast<memref::ExtractStridedMetadataOp>(op)) {
    result.emplace_back(extractStridedMetadataOp.getBaseBuffer(),
                        extractStridedMetadataOp.getViewSource());
  } else if (auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(op)) {
    result.emplace_back(toMemrefOp.getResult(), toMemrefOp.getOperand());
  } else if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
    result.emplace_back(toTensorOp.getResult(), toTensorOp.getOperand());
  } else if (auto bitCastOp = dyn_cast<hivm::BitcastOp>(op)) {
    result.emplace_back(bitCastOp.getResult(), bitCastOp.getSrc());
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    result.emplace_back(selectOp.getResult(), selectOp.getTrueValue());
    result.emplace_back(selectOp.getResult(), selectOp.getFalseValue());
  }
  return result;
}

std::optional<uint32_t> GetBufferSize(Value buffer) {
  auto memRefType = cast<MemRefType>(buffer.getType());
  if (!memRefType)
    return std::nullopt;
  ::llvm::ArrayRef<int64_t> shape = memRefType.getShape();
  uint32_t bufferConstByteSize = memRefType.getElementTypeBitWidth() / 8;
  for (auto &v : shape)
    bufferConstByteSize *= v;
  return bufferConstByteSize;
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
    auto res = parentLoop.getLoopResults().value()[*idxLoopRes];
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

  auto elseYieldOp = parentIf.elseYield();
  auto elseYieldOpers = elseYieldOp.getOperands();
  auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return getParentLoop(res);
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

Value createNestedIndexModular(OpBuilder &builder,
                               LoopLikeOpInterface loopOp,
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

static LogicalResult getCastSrcUnAlignSizeInfo(
    Value src, SmallVector<int64_t> castAlignDims, int64_t bytesFactor,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType srcType = cast<ShapedType>(src.getType());
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // collect unalign info of src if cast src dims are not aligned.
  auto hwAlignBytes = maybeHwAlignBytes.value();
  auto elemTypeBytes = getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
#ifndef NDEBUG
  const int b16InByte = 2;
  const int b32InByte = 4;
  assert((elemTypeBytes == b16InByte || elemTypeBytes == b32InByte) &&
      "Src only supports b32/b16 in cast overflow.");
#endif
  auto shape = cast<ShapedType>(src.getType()).getShape();
  int64_t numElemPerBlock = mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
  int64_t numElemPerBlockForDst = numElemPerBlock * bytesFactor;
  int64_t rank = srcType.getRank();
  // For example (a, b)strides<n1, 1>*i32 cast to (a, b)strides<n2, 1>*i8:
  // 1. (a, b)strides<n1, 1>*i32 view as (a, b*4)strides<n1*4, 1>*i8
  // 2. i8 transpose: Used to separate the high and low bits of int32, make sure
  //    the shape of tranpose is 32*32 aligned.
  // 3. i8 copyubtoub: Take out the lower 8 bits.
  // 4. i8 transpose: Transpose back to get the final cast result, make sure the
  //    shape of tranpose is 32*32 aligned.
  // 5. When n2 is aligned with multiple blocks, need to add another copyubtoub
  //    to adjust it to the target stride.
  if (rank == 1) {
    // The 1D scene is quite special and needs to be converted into a
    // corresponding 2D scene to implement.
    if (!ShapedType::isDynamic(shape[0]) && shape[0] <= numElemPerBlockForDst) {
      hwAlignBytes = static_cast<unsigned>(
        CEIL_FACTOR(shape[0] * bytesFactor, numElemPerBlockForDst) *
        numElemPerBlockForDst);
    } else {
      hwAlignBytes = static_cast<unsigned>(numElemPerBlockForDst *
                                           numElemPerBlockForDst * bytesFactor);
    }
    if (ShapedType::isDynamic(shape[0]) ||
        (shape[0] * elemTypeBytes) % hwAlignBytes != 0) {
      auto srcAlignInfo = std::make_unique<OperAlignInfo>(src, 0, hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
  } else {
#ifndef NDEBUG
    const int supportedCastAlignDimSize = 2;
    assert(castAlignDims.size() == supportedCastAlignDimSize &&
        "When cast rank >= 2, castAlignDims size must be equal to 2");
#endif
    // Align the second axis in castAlignDims.
    if (ShapedType::isDynamic(shape[castAlignDims[1]]) ||
        (shape[castAlignDims[1]] * elemTypeBytes) % hwAlignBytes != 0) {
      auto srcAlignInfo =
          std::make_unique<OperAlignInfo>(src, castAlignDims[1], hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
    // Align the first axis in castAlignDims.
    hwAlignBytes = static_cast<unsigned>(numElemPerBlockForDst * bytesFactor);
    if (ShapedType::isDynamic(shape[castAlignDims[0]]) ||
        (static_cast<uint64_t>(shape[castAlignDims[0]]) * elemTypeBytes) %
        hwAlignBytes !=
        0) {
      auto srcAlignInfo =
          std::make_unique<OperAlignInfo>(src, castAlignDims[0], hwAlignBytes);
      operAlignInfoList->push_back(std::move(srcAlignInfo));
    }
  }
  return success();
}


static LogicalResult getCastDstUnAlignSizeInfo(
    Value dst, SmallVector<int64_t> castAlignDims,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType dstType = cast<ShapedType>(dst.getType());
  auto maybeHwAlignBytes = getHWAlignBytes(dstType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // collect unalign info of dst if cast dst dims are not aligned
  auto hwAlignBytes = maybeHwAlignBytes.value();
  auto elemTypeBytes = getElementTypeOrSelf(dstType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
  assert(elemTypeBytes == 1 && "Dst only supports b8 in cast overflow.");
  auto shape = cast<ShapedType>(dst.getType()).getShape();
  uint64_t numElemPerBlock = mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
  int64_t rank = dstType.getRank();
  // For example (a, b)strides<n1, 1>*i32 cast to (a, b)strides<n2, 1>*i8:
  // 1. (a, b)strides<n1, 1>*i32 view as (a, b*4)strides<n1*4, 1>*i8
  // 2. i8 transpose: Used to separate the high and low bits of int32, make sure
  //    the shape of tranpose is 32*32 aligned.
  // 3. i8 copyubtoub: Take out the lower 8 bits.
  // 4. i8 transpose: Transpose back to get the final cast result, make sure the
  //    shape of tranpose is 32*32 aligned.
  // 5. When n2 is aligned with multiple blocks, need to add another copyubtoub
  //    to adjust it to the target stride.
  if (rank == 1) {
    // The 1D scene is quite special and needs to be converted into a
    // corresponding 2D scene to implement.
    hwAlignBytes = numElemPerBlock * numElemPerBlock;
  }
  for (auto checkDim : castAlignDims) {
    if (ShapedType::isDynamic(shape[checkDim]) ||
        (static_cast<uint64_t>(shape[checkDim]) * elemTypeBytes) %
        hwAlignBytes !=
        0) {
      auto dstAlignInfo =
          std::make_unique<OperAlignInfo>(dst, checkDim, hwAlignBytes);
      operAlignInfoList->push_back(std::move(dstAlignInfo));
    }
  }
  return success();
}

static void collectOpAlignInfo(
    Operation *op, SmallVector<int64_t> checkDims,
    llvm::SmallDenseMap<Value, uint32_t> *alignBytes,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  assert(alignBytes != nullptr);
  for (auto oper : op->getOperands()) {
    auto elemTypeBytes = getElementTypeOrSelf(oper).getIntOrFloatBitWidth() / 8;
    auto shape = cast<ShapedType>(oper.getType()).getShape();
    for (auto checkDim : checkDims) {
      assert(checkDim >= 0 && checkDim < static_cast<int64_t>(shape.size()));
      assert((*alignBytes)[oper] != 0);
      if (ShapedType::isDynamic(shape[checkDim]) ||
          (shape[checkDim] * elemTypeBytes) % (*alignBytes)[oper] != 0) {
        auto operAlignInfo = std::make_unique<OperAlignInfo>(
            oper, checkDim, (*alignBytes)[oper]);
        operAlignInfoList->push_back(std::move(operAlignInfo));
      }
    }
  }
}

LogicalResult getUnAlignSizeInfo(
    VTransposeOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  auto srcType = op.getSrc().getType();
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // get transpose loop dims
  SmallVector<int64_t> transposeLoopDims;
  op.getTransposeLoopDims(transposeLoopDims);

  // collect unalign info of all operands if transpose dims are not aligned
  auto hwAlignBytes = maybeHwAlignBytes.value();
  llvm::SmallDenseMap<Value, uint32_t> operHwAlignBytes;
  for (auto oper : op->getOperands()) {
    operHwAlignBytes[oper] = hwAlignBytes;
  }
  collectOpAlignInfo(op.getOperation(), transposeLoopDims, &operHwAlignBytes,
                     operAlignInfoList);

  auto elemTypeBytes = getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
                       mlir::utils::INTR_BITS_PER_BYTE;
  const int b32InByte = 4;
  if (elemTypeBytes != b32InByte) {
    return success();
  }

  // when it is B32 type, judge if there is one dim that is already double
  // aligned. if not, should choose one dim to do double alignment, e.g.
  // 8x16xf32 or 16x8xf32.
  auto srcShape = cast<ShapedType>(op.getSrc().getType()).getShape();
  bool isAlreadyDoubleAlign = false;
  for (auto transDim : transposeLoopDims) {
    auto alignedSrcDimBytes =
        CEIL_FACTOR(static_cast<uint64_t>(srcShape[transDim]) * elemTypeBytes,
                    hwAlignBytes);
    if (alignedSrcDimBytes % (hwAlignBytes * 2) == 0) {
      isAlreadyDoubleAlign = true;
    }
  }
  if (isAlreadyDoubleAlign) {
    return success();
  }

  // must choose double align dim from two transpose dims
  if (transposeLoopDims.size() != 2) {
    // For B32, do transpose decompose first
    return failure();
  }

  operAlignInfoList->clear();
  // choose transdim 0 as double align dim
  auto srcTrans0AlignInfo = std::make_unique<OperAlignInfo>(
      op.getSrc(), transposeLoopDims[0], hwAlignBytes);
  operAlignInfoList->push_back(std::move(srcTrans0AlignInfo));
  auto srcTrans1AlignInfo = std::make_unique<OperAlignInfo>(
      op.getSrc(), transposeLoopDims[1], hwAlignBytes * 2);
  operAlignInfoList->push_back(std::move(srcTrans1AlignInfo));

  auto dstTrans0AlignInfo = std::make_unique<OperAlignInfo>(
      op.getDst(), transposeLoopDims[0], hwAlignBytes * 2);
  operAlignInfoList->push_back(std::move(dstTrans0AlignInfo));
  auto dstTrans1AlignInfo = std::make_unique<OperAlignInfo>(
      op.getDst(), transposeLoopDims[1], hwAlignBytes);
  operAlignInfoList->push_back(std::move(dstTrans1AlignInfo));
  return success();
}

LogicalResult getUnAlignSizeInfo(
    VCastOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  auto srcType = cast<ShapedType>(op.getSrc()[0].getType());
  auto dstType = cast<ShapedType>(op.getDst()[0].getType());
  auto srcElemTypeBytes =
      getElementTypeOrSelf(srcType).getIntOrFloatBitWidth() /
      mlir::utils::INTR_BITS_PER_BYTE;
  auto dstElemTypeBytes =
      getElementTypeOrSelf(dstType).getIntOrFloatBitWidth() /
      mlir::utils::INTR_BITS_PER_BYTE;
  auto bytesFactor = srcElemTypeBytes / dstElemTypeBytes;

  // Get the cast axis that needs to be aligned.
  SmallVector<int64_t> castAlignDims;
  int64_t rank = srcType.getRank();
  if (rank == 1) {
    castAlignDims.push_back(0);
  } else if (rank >= 2) {
    castAlignDims.push_back(rank - 2);
    castAlignDims.push_back(rank - 1);
  } else {
    llvm_unreachable("cast op rank need lager than 0.");
  }

  // Get the unalign information of the axis corresponding to cast src.
  if (failed(getCastSrcUnAlignSizeInfo(op.getSrc()[0], castAlignDims,
                                       bytesFactor, operAlignInfoList))) {
    return failure();
  }
  // Get the unalign information of the axis corresponding to cast dst.
  if (failed(getCastDstUnAlignSizeInfo(op.getDst()[0], castAlignDims,
                                       operAlignInfoList))) {
    return failure();
  }
  return success();
}

LogicalResult getUnAlignSizeInfo(
    VSortOp op,
    std::vector<std::unique_ptr<OperAlignInfo>> *operAlignInfoList) {
  // get alignment bytes
  ShapedType srcType = cast<ShapedType>(op.getSrc().getType());
  auto maybeHwAlignBytes = getHWAlignBytes(srcType);
  if (!maybeHwAlignBytes.has_value()) {
    return failure();
  }

  // Get the sort axis that needs to be aligned.
  SmallVector<int64_t> sortAlignDims;
  int64_t rank = srcType.getRank();
  sortAlignDims.push_back(rank - 1);

  llvm::SmallDenseMap<Value, uint32_t> operHwAlignBytes;
  for (auto oper : op->getOperands()) {
    ShapedType operType = cast<ShapedType>(oper.getType());
    auto elemTypeBytes =
        getElementTypeOrSelf(operType).getIntOrFloatBitWidth() / 8;
    unsigned int numElemPerBlock =
        mlir::utils::INTR_BYTES_PER_BLOCK / elemTypeBytes;
    operHwAlignBytes[oper] =
        maybeHwAlignBytes.value() * (VBITSORT_NUM_PER_REPEAT / numElemPerBlock);
  }
  collectOpAlignInfo(op.getOperation(), sortAlignDims, &operHwAlignBytes,
                     operAlignInfoList);
  return success();
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
  if (failed(getStridesAndOffset(srcType, srcStrides, srcOffset)))
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
  auto [strides, offset] = getStridesAndOffset(mem);
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

bool isArgminOrArgmax(ReduceOperation op) {
  return op == ReduceOperation::min_with_index_left ||
         op == ReduceOperation::max_with_index_left ||
         op == ReduceOperation::min_with_index_right ||
         op == ReduceOperation::max_with_index_right;
}

static bool shouldMapToUnsigned(IntegerType::SignednessSemantics val,
                         hivm::TypeFn casting) {
  if (hivm::TypeFn::cast_unsigned == casting)
    return true;

  switch (val) {
  case IntegerType::Signless:
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

std::string getTypeName(Location loc, Type type,
                                      hivm::TypeFn casting) {
  std::string unknown = "UNKNOWN";
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (BitWidth(iType.getWidth())) {
    case BitWidth::B1:
      return "bool";
    case BitWidth::B4:
    case BitWidth::B8:
    case BitWidth::B16:
    case BitWidth::B32:
    case BitWidth::B64:
      if (shouldMapToUnsigned(iType.getSignedness(), casting))
        return "uint" + std::to_string(iType.getWidth()) + "_t";
      else
        return "int" + std::to_string(iType.getWidth()) + "_t";
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (BitWidth(fType.getWidth())) {
    case BitWidth::B16:
      if (fType.isF16()) {
        return "half";
      } else if (fType.isBF16()) {
        return "bfloat16_t";
      } else {
        emitError(loc, "unrecognized float type: ") << type;
        return unknown;
      }
    case BitWidth::B32:
      return "float";
    case BitWidth::B64:
      return "double";
    default:
      emitError(loc, "unrecognized float type: ") << type;
      return unknown;
    }
  }
  emitError(loc, "unsupported type: ") << type;
  return unknown;
}
} // namespace util
} // namespace hivm
} // namespace mlir
