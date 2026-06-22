//===- FusibleHelper.cpp - Provide utilities and fusion rules to analyzer -===//
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

#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include <numeric>
#include <queue>

#define DEBUG_TYPE "hfusion-fuse"
namespace mlir {
namespace hfusion {
namespace opfusion {

//===---------------------------------------------------------------------===//
// Utils
//===---------------------------------------------------------------------===//

bool FusibleHelper::isSingleOutlinable(Operation *op) {
  auto pattern = FusibleHelper::getOpPattern(op);
  LLVM_DEBUG(llvm::dbgs() << "Checking op " << static_cast<uint8_t>(pattern)
                          << "\n";);
  return static_cast<uint8_t>(pattern) >=
         static_cast<uint8_t>(opfusion::OpPattern::kElementWise);
}

FusionKind FusibleHelper::getSingleFusionKind(Operation *op) {
  auto pattern = FusibleHelper::getOpPattern(op);

  LLVM_DEBUG(llvm::dbgs() << "Trying to single outline\n";);
  switch (pattern) {
  case opfusion::OpPattern::kElementWise:
  case opfusion::OpPattern::kZeroRankElemwise:
  case opfusion::OpPattern::kInterleave:
  case opfusion::OpPattern::kMidFusionImportantAux:
  case opfusion::OpPattern::kTranspose:
    return FusionKind::PureElemwise;
  case opfusion::OpPattern::kLastAxisReduce:
    return FusionKind::LastAxisPBR;
  case opfusion::OpPattern::kOtherReduce:
    return FusionKind::AnyPBR;
  case opfusion::OpPattern::kMatmul:
    return FusionKind::SingleCube;
  case opfusion::OpPattern::kExtractSlice:
  case opfusion::OpPattern::kLastAxisBroadcast:
  case opfusion::OpPattern::kOtherBroadcast:
    return FusionKind::AnyPB;
  default:
    llvm::report_fatal_error("Invalid operation pattern for outlining");
  }
}

// README: [$FusionType] Means it is for that certain FusionType
FusibleHelper::FusibleHelper(FusionKind fusionKind, bool bufferToOut,
                             int32_t maxHorizontalFusionSize)
    : fusionKind_(fusionKind), moveOutToParam_(bufferToOut),
      maxHorizontalFusion_(
          maxHorizontalFusionSize == -1 ? INT_MAX : maxHorizontalFusionSize) {}

// [General] This is to check whether out tensor allocation
// should be included in the fusion or not.
//
// If it's true, than all out operator's tensor.empty / allocation
// would be left in the caller (not included in the fusion)
bool FusibleHelper::moveOutToParam() const { return moveOutToParam_; }

// [General] This is to check how many max non-dependent function fusion
// should be attempted. -1 to merge all, 0 to separate all.
int32_t FusibleHelper::maxHorizontalFusion() const {
  return maxHorizontalFusion_;
}

// [General] Auxiliary nodes are the arith, etc.
bool FusibleHelper::isPossibleCountingAux(Operation *defOp) {
  if (defOp->getNumResults() == 1 &&
      defOp->getResults()[0].getType().isIntOrIndexOrFloat())
    return true;
  return false;
}

// [General] Auxiliary nodes are the arith, etc.
bool FusibleHelper::isAuxiliary(Operation *op) {
  return getOpPattern(op) == OpPattern::kAuxiliary;
}

bool FusibleHelper::isZeroRankElemwise(Operation *op) {
  return getOpPattern(op) == OpPattern::kZeroRankElemwise;
}

// [General] Auxiliary nodes are the empty etc, etc.
bool FusibleHelper::isBuffer(Operation *op) {
  return getOpPattern(op) == OpPattern::kBuffer;
}

bool FusibleHelper::isFusible(Operation *a, Operation *b) const {
  return isFusible(getOpPattern(a), getOpPattern(b));
}

bool FusibleHelper::isShallowFusion(FusionKind fusionKind) {
  return fusionKind == FusionKind::ShallowCV ||
         fusionKind == FusionKind::ShallowVV;
}

// [MixCV] Node Type Checking
uint8_t FusibleHelper::obtainType(Operation *op) const {
  OpPattern pattern = getOpPattern(op);
  TypePattern returnType = TypePattern::kOpaque;
  switch (pattern) {
  case OpPattern::kElementWise:
  case OpPattern::kZeroRankElemwise:
    returnType = TypePattern::kPureElementWise;
    break;
  case OpPattern::kMatmul:
    returnType = TypePattern::kPureMatmul;
    break;
  default:
    break;
  }
  return static_cast<uint8_t>(returnType);
}

// [MixCV] Adjust type
uint8_t FusibleHelper::adjustType(const uint8_t &typeA, const uint8_t &typeB,
                                  bool isHorizontal) const {
  if (isHorizontal)
    return adjustTypeHorizontal(typeA, typeB);
  return adjustType(typeA, typeB);
}

// [MixCV] adjust type for vertical fusion
uint8_t FusibleHelper::adjustType(const uint8_t &typeA,
                                  const uint8_t &typeB) const {
  TypePattern returnType = TypePattern::kOpaque;
  TypePattern patternA = static_cast<TypePattern>(typeA);
  TypePattern patternB = static_cast<TypePattern>(typeB);
  switch (patternA) {
  case TypePattern::kPureElementWise:
    switch (patternB) {
    case TypePattern::kPureElementWise:
      // (Elwise -> .. -> Elwise) + (Elwise -> .. -> Elwise)
      returnType = TypePattern::kPureElementWise;
      break;
    default:
      break;
    }
    break;
  case TypePattern::kPureMatmul:
  case TypePattern::kSuffixElementWise:
    switch (patternB) {
    case TypePattern::kPureElementWise:
      // (Matmul) + (Elwise -> .. -> Elwise)
      returnType = TypePattern::kSuffixElementWise;
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }
  return static_cast<uint8_t>(returnType);
}

// [MixCV] adjust type for horizontal fusion
uint8_t FusibleHelper::adjustTypeHorizontal(const uint8_t &typeA,
                                            const uint8_t &typeB) const {
  TypePattern returnType = TypePattern::kOpaque;
  TypePattern patternA = static_cast<TypePattern>(typeA);
  TypePattern patternB = static_cast<TypePattern>(typeB);
  if (hasMatmulTypePattern(typeA))
    returnType = patternA;
  if (hasMatmulTypePattern(typeB))
    returnType = patternB;
  return static_cast<uint8_t>(returnType);
}

// [MixCV] the matmul allowed at the beginning only, and followed by pure
// elemwise
bool FusibleHelper::isRestrictedByNodeType(const uint8_t &typeA,
                                           const uint8_t &typeB,
                                           bool isHorizontal) const {
  if (isHorizontal)
    return false;
  return isRestrictedByNodeType(typeA, typeB);
}

// [MixCV] the matmul allowed at the beginning
bool FusibleHelper::isRestrictedByNodeType(const uint8_t &typeA,
                                           const uint8_t &typeB) const {
  // For other than mix cv, no restriction
  if (fusionKind_ != FusionKind::MixCV)
    return false;

  TypePattern patternA = static_cast<TypePattern>(typeA);
  TypePattern patternB = static_cast<TypePattern>(typeB);

  bool restricted = true;
  switch (patternA) {
  case TypePattern::kPureElementWise:
  case TypePattern::kPureMatmul:
    switch (patternB) {
    case TypePattern::kPureElementWise:
      restricted = false;
      break;
    default:
      break;
    }
    break;
  case TypePattern::kSuffixElementWise:
    if (patternB == TypePattern::kPureElementWise) {
      restricted = false;
    }
    break;
  default:
    break;
  }

  return restricted;
}

// [MixCV] To check whether the type pattern has a matmul
bool FusibleHelper::hasMatmulTypePattern(const uint8_t &typePattern) const {
  // For other than mix cv, no restriction
  if (fusionKind_ != FusionKind::MixCV)
    return false;

  TypePattern typeUint = static_cast<TypePattern>(typePattern);
  switch (typeUint) {
  case TypePattern::kPureMatmul:
  case TypePattern::kSuffixElementWise:
    return true;
  default:
    return false;
  }
}

/// [LastAxisPBR and AnyPBR]
/// This is to obtain input rank(max rank) of reduce op
int FusibleHelper::obtainLastReduceRank(Operation *op) const {
  if (getOpPattern(op) == OpPattern::kLastAxisReduce) {
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    size_t inputRank = getMaxRank(linalgOp.getDpsInputs());

    return inputRank;
  }

  // Return magic -1 for other op
  return -1;
}

/// [LastAxisPBR and AnyPBR]
/// This is to obtain reduce op dim
int FusibleHelper::obtainReduceDim(Operation *op) const {
  OpPattern pattern = getOpPattern(op);
  if (pattern != OpPattern::kLastAxisReduce &&
      pattern != OpPattern::kOtherReduce) {
    return -1;
  }
  auto prop =
      llvm::dyn_cast<mlir::DictionaryAttr>(op->getPropertiesAsAttribute());
  auto reduceDims =
      ::llvm::cast<::mlir::DenseI64ArrayAttr>(prop.get("dimensions"));
  if (reduceDims.size() != 1) {
    return -1;
  }
  return static_cast<int>(reduceDims[0]);
}

// [LastAxisPBR and AnyPBR]
// This is to check whether all reduce op operate with same input rank
bool FusibleHelper::isRestrictedByReduceRank(const int &a, const int &b) const {
  // For other than lastpbr/anypbr, no restriction
  if (fusionKind_ != FusionKind::LastAxisPBR &&
      fusionKind_ != FusionKind::AnyPBR)
    return false;

  return (a >= 0 && b >= 0 && a != b);
}

// [LastAxisPBR and AnyPBR]
// This is to check whether all reduce op operate with same reduce dim
bool FusibleHelper::isRestrictedByReduceDim(const int &a, const int &b) const {
  // For other than lastpbr/anypbr, no restriction
  if (fusionKind_ != FusionKind::LastAxisPBR &&
      fusionKind_ != FusionKind::AnyPBR)
    return false;

  return (a >= 0 && b >= 0 && a != b);
}

// Dynamic shapes are not allowed for shallow fusion
// For non shallow fusion, it's restricted if it's dynamic when it's horizontal
bool FusibleHelper::isRestrictedByDynamicShape(Operation *op,
                                               bool horizontal) const {
  if (fusionKind_ == FusionKind::ShallowCV ||
      fusionKind_ == FusionKind::ShallowVV || horizontal)
    return mlir::hfusion::util::hasDynamicShapeOperand(op);
  return false;
}

// [Horizontal Fusion] This is to determine whether horizontal fusion is fusible
bool FusibleHelper::isShapePivot(Operation *op) const {
  switch (fusionKind_) {
  case FusionKind::MixCV:
    return getOpPattern(op) == OpPattern::kMatmul;
  case FusionKind::AnyPB:
  case FusionKind::LastAxisPBR:
  case FusionKind::AnyPBR:
    return getOpPattern(op) == OpPattern::kLastAxisBroadcast ||
           getOpPattern(op) == OpPattern::kOtherBroadcast ||
           isa<linalg::FillOp>(op);
  case FusionKind::PureElemwise:
    return getOpPattern(op) == OpPattern::kElementWise ||
           getOpPattern(op) == OpPattern::kZeroRankElemwise;
  default:
    return false;
  }
  return false;
}
bool isVectorPattern(const OpPattern &pattern) {
  switch (pattern) {
  case OpPattern::kElementWise:
  case OpPattern::kZeroRankElemwise:
  case OpPattern::kExtractSlice:
  case OpPattern::kInsertSlice:
  case OpPattern::kInterleave:
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kLastAxisReduce:
  case OpPattern::kLoadStore:
  case OpPattern::kMidFusionAuxiliary:
  case OpPattern::kMidFusionImportantAux:
  case OpPattern::kOtherBroadcast:
  case OpPattern::kOtherReduce:
  case OpPattern::kReshape:
  case OpPattern::kTranspose:
    return true;
  default:
    return false;
  }
}
bool FusibleHelper::schedulable(Operation *op) const {
  const auto currentPattern = getOpPattern(op);
  if (!isImportantPattern(op))
    return false;
  switch (fusionKind_) {
  case FusionKind::PureElemwise:
    switch (currentPattern) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
      return true;
    default:
      return false;
    }
  case FusionKind::AnyPB:
    switch (currentPattern) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kOtherBroadcast:
      return true;
    default:
      return false;
    }
  case FusionKind::LastAxisPBR:
    switch (currentPattern) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kLastAxisReduce:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kOtherBroadcast:
      return true;
    default:
      return false;
    }
  case FusionKind::AnyPBR:
    switch (currentPattern) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kLastAxisReduce:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kOtherBroadcast:
    case OpPattern::kOtherReduce:
      return true;
    default:
      return false;
    }
  case FusionKind::ShallowCV:
    switch (currentPattern) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kMatmul:
    case OpPattern::kLastAxisReduce:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kOtherBroadcast:
      return true;
    default:
      return false;
    }
  case FusionKind::ShallowVV:
    return isVectorPattern(currentPattern);
  case FusionKind::MixCV:
    switch (currentPattern) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kMatmul:
      return true;
    default:
      return false;
    }
  case FusionKind::Unknown:
    llvm::report_fatal_error("Fusion kind unknown is not scheduling anything");
  default:
    break;
  }
  return false;
}

namespace {
OpPattern classifyReduceOp(DestinationStyleOpInterface reduceOp) {
  auto prop = llvm::dyn_cast<mlir::DictionaryAttr>(
      reduceOp->getPropertiesAsAttribute());
  auto dimensions =
      ::llvm::cast<::mlir::DenseI64ArrayAttr>(prop.get("dimensions"));
  if (dimensions.size() != 1)
    return OpPattern::kOtherReduce;

  const auto &reduceAxis = dimensions[0];
  // TODO: Handle variadic reduce.
  auto initIt = reduceOp.getDpsInits().begin();
  auto init = cast<TypedValue<ShapedType>>(*initIt);
  auto lastAxis = init.getType().getShape().size();
  return (static_cast<size_t>(reduceAxis) == lastAxis)
             ? OpPattern::kLastAxisReduce
             : OpPattern::kOtherReduce;
}

OpPattern classifyBroadcastOp(linalg::BroadcastOp broadcastOp) {
  auto dimensions = broadcastOp.getDimensions();
  if (dimensions.size() != 1)
    return OpPattern::kOtherBroadcast;

  const auto &broadcastAxis = dimensions[0];
  auto lastAxis = broadcastOp.getInput().getType().getShape().size();
  if (static_cast<size_t>(broadcastAxis) == lastAxis)
    return OpPattern::kLastAxisBroadcast;

  LLVM_DEBUG(llvm::dbgs() << "infer kOtherBroadcast\n";);
  return OpPattern::kOtherBroadcast;
}
} // namespace

OpPattern FusibleHelper::getOpPattern(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Checking op pattern " << *op << "\n";);
  if (reshape_utils::isMarkedAsElementwiseOp(op)) {
    if (reshape_utils::isZeroDimensionOp(op)) {
      return OpPattern::kZeroRankElemwise;
    }
    return OpPattern::kElementWise;
  }
  return llvm::TypeSwitch<Operation *, OpPattern>(op)
      .Case<linalg::ReduceOp, hfusion::ReduceWithIndexOp>(classifyReduceOp)
      .Case<linalg::BroadcastOp>(classifyBroadcastOp)
      .Case<linalg::MatmulOp, linalg::MatmulTransposeAOp,
            linalg::MatmulTransposeBOp>(
          [](auto) -> OpPattern { return OpPattern::kMatmul; })
      .Case<arith::ConstantOp>([](arith::ConstantOp constOp) -> OpPattern {
        auto shapedType = dyn_cast<ShapedType>(constOp.getType());
        return shapedType ? OpPattern::kOpaque : OpPattern::kAuxiliary;
      })
      .Case<tensor::EmptyOp, shape::ShapeOfOp, shape::BroadcastOp>(
          [](auto) -> OpPattern { return OpPattern::kBuffer; })
      .Case<hfusion::LoadOp, hfusion::StoreOp>(
          [](auto) -> OpPattern { return OpPattern::kLoadStore; })
      .Case<hfusion::SymbolicDimOp, tensor::DimOp>(
          [](auto) -> OpPattern { return OpPattern::kAuxiliary; })
      .Case<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
          [](auto) -> OpPattern { return OpPattern::kReshape; })
      .Case<tensor::ExtractSliceOp>(
          [](auto) -> OpPattern { return OpPattern::kExtractSlice; })
      .Case<tensor::InsertSliceOp>(
          [](auto) -> OpPattern { return OpPattern::kInsertSlice; })
      .Case<tensor::CastOp, tensor::ExtractOp, shape::ConstShapeOp>(
          [](auto) -> OpPattern { return OpPattern::kMidFusionAuxiliary; })
      .Case<tensor::ConcatOp, tensor::PadOp>(
          [](auto) -> OpPattern { return OpPattern::kMidFusionImportantAux; })
      .Case<mesh::AllReduceOp>(
          [](auto) -> OpPattern { return OpPattern::kAllReduce; })
      .Case<mesh::AllGatherOp>(
          [](auto) -> OpPattern { return OpPattern::kAllGather; })
      .Case<mesh::ReduceScatterOp>(
          [](auto) -> OpPattern { return OpPattern::kReduceScatter; })
      .Case<hfusion::InterleaveOp, hfusion::DeinterleaveOp>(
          [](auto) -> OpPattern { return OpPattern::kInterleave; })
      .Case<linalg::TransposeOp>(
          [](auto) -> OpPattern { return OpPattern::kTranspose; })
      .Default([](Operation *) -> OpPattern { return OpPattern::kOpaque; });
}

bool FusibleHelper::isImportantPattern(const OpPattern &pattern) {
  switch (pattern) {
  case OpPattern::kAllGather:
  case OpPattern::kAllReduce:
  case OpPattern::kElementWise:
  case OpPattern::kExtractSlice:
  case OpPattern::kInterleave:
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kLastAxisReduce:
  case OpPattern::kMatmul:
  case OpPattern::kMidFusionImportantAux:
  case OpPattern::kOtherBroadcast:
  case OpPattern::kOtherReduce:
  case OpPattern::kReduceScatter:
  case OpPattern::kTranspose:
  case OpPattern::kZeroRankElemwise:
    return true;
  case OpPattern::kAuxiliary:
  case OpPattern::kBuffer:
  case OpPattern::kLoadStore:
  case OpPattern::kOpaque:
  case OpPattern::kReshape:
    return false;
  default:
    break;
  }
  return false;
}

bool FusibleHelper::isImportantPattern(Operation *op) {
  return FusibleHelper::isImportantPattern(getOpPattern(op));
}

FusionKind FusibleHelper::getFusionKind() const { return fusionKind_; }

bool FusibleHelper::isFusible(const OpPattern &patternA,
                              const OpPattern &patternB) const {
  switch (fusionKind_) {
  case FusionKind::PureElemwise:
    return isPureElemwiseFusible(patternA, patternB);
  case FusionKind::AnyPB:
    return isAnyPBFusible(patternA, patternB);
  case FusionKind::LastAxisPBR:
    return isLastAxisPBRFusible(patternA, patternB);
  case FusionKind::AnyPBR:
    return isAnyPBRFusible(patternA, patternB);
  case FusionKind::ShallowCV:
    return isShallowCVFusible(patternA, patternB);
  case FusionKind::ShallowVV:
    return isShallowVVFusible(patternA, patternB);
  case FusionKind::MixCV:
    return isMixCVFusible(patternA, patternB);
  case FusionKind::MixC2:
    return isMixC2Fusible(patternA, patternB);
  case FusionKind::SingleCube:
    // single cube is not considered as fusible because there is only one op
    return false;
  default:
    llvm::report_fatal_error("Invalid fusion mode");
  }
} // namespace opfusion

bool FusibleHelper::isPureElemwiseFusible(const OpPattern &patternA,
                                          const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kZeroRankElemwise:
  case OpPattern::kExtractSlice:
  case OpPattern::kInsertSlice:
  case OpPattern::kInterleave:
  case OpPattern::kLoadStore:
  case OpPattern::kMidFusionAuxiliary:
  case OpPattern::kMidFusionImportantAux:
  case OpPattern::kReshape:
  case OpPattern::kTranspose:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kExtractSlice:
    case OpPattern::kInsertSlice:
    case OpPattern::kInterleave:
    case OpPattern::kLoadStore:
    case OpPattern::kMidFusionAuxiliary:
    case OpPattern::kMidFusionImportantAux:
    case OpPattern::kReshape:
    case OpPattern::kTranspose:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusibleHelper::isLastAxisPBRFusible(const OpPattern &patternA,
                                         const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kZeroRankElemwise:
  case OpPattern::kExtractSlice:
  case OpPattern::kInsertSlice:
  case OpPattern::kInterleave:
  case OpPattern::kLoadStore:
  case OpPattern::kMidFusionAuxiliary:
  case OpPattern::kMidFusionImportantAux:
  case OpPattern::kReshape:
  case OpPattern::kTranspose:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kExtractSlice:
    case OpPattern::kInsertSlice:
    case OpPattern::kInterleave:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kLastAxisReduce:
    case OpPattern::kLoadStore:
    case OpPattern::kMidFusionAuxiliary:
    case OpPattern::kMidFusionImportantAux:
    case OpPattern::kOtherBroadcast:
    case OpPattern::kReshape:
    case OpPattern::kTranspose:
      return true;
    default:
      return false;
    }
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kLastAxisReduce:
  case OpPattern::kOtherBroadcast:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kExtractSlice:
    case OpPattern::kInsertSlice:
    case OpPattern::kInterleave:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kLoadStore:
    case OpPattern::kMidFusionAuxiliary:
    case OpPattern::kMidFusionImportantAux:
    case OpPattern::kOtherBroadcast:
    case OpPattern::kReshape:
    case OpPattern::kTranspose:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusibleHelper::isShallowCVFusible(const OpPattern &patternA,
                                       const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kZeroRankElemwise:
  case OpPattern::kExtractSlice:
  case OpPattern::kInsertSlice:
  case OpPattern::kInterleave:
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kLastAxisReduce:
  case OpPattern::kLoadStore:
  case OpPattern::kMatmul:
  case OpPattern::kMidFusionAuxiliary:
  case OpPattern::kMidFusionImportantAux:
  case OpPattern::kOtherBroadcast:
  case OpPattern::kReshape:
  case OpPattern::kTranspose:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kExtractSlice:
    case OpPattern::kInsertSlice:
    case OpPattern::kInterleave:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kLastAxisReduce:
    case OpPattern::kLoadStore:
    case OpPattern::kMatmul:
    case OpPattern::kMidFusionAuxiliary:
    case OpPattern::kMidFusionImportantAux:
    case OpPattern::kOtherBroadcast:
    case OpPattern::kReshape:
    case OpPattern::kTranspose:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusibleHelper::isShallowVVFusible(const OpPattern &patternA,
                                       const OpPattern &patternB) const {
  return isVectorPattern(patternA) && isVectorPattern(patternB);
}

bool FusibleHelper::isMixCVFusible(const OpPattern &patternA,
                                   const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kZeroRankElemwise:
  case OpPattern::kExtractSlice:
  case OpPattern::kInsertSlice:
  case OpPattern::kInterleave:
  case OpPattern::kLoadStore:
  case OpPattern::kMidFusionAuxiliary:
  case OpPattern::kMidFusionImportantAux:
  case OpPattern::kReshape:
  case OpPattern::kTranspose:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kExtractSlice:
    case OpPattern::kInsertSlice:
    case OpPattern::kInterleave:
    case OpPattern::kLoadStore:
    case OpPattern::kMidFusionAuxiliary:
    case OpPattern::kMidFusionImportantAux:
    case OpPattern::kReshape:
    case OpPattern::kTranspose:
      return true;
    default:
      return false;
    }
  case OpPattern::kMatmul:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

// TODO：support combination of any mesh op and matmul op after the template
// library is ready
bool FusibleHelper::isMixC2Fusible(const OpPattern &patternA,
                                   const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kAllGather:
    switch (patternB) {
    case OpPattern::kMatmul:
      return true;
    default:
      return false;
    }
  case OpPattern::kMatmul:
    switch (patternB) {
    case OpPattern::kAllReduce:
    case OpPattern::kReduceScatter:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusibleHelper::isAnyPBFusible(const OpPattern &patternA,
                                   const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kZeroRankElemwise:
  case OpPattern::kExtractSlice:
  case OpPattern::kInsertSlice:
  case OpPattern::kInterleave:
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kLoadStore:
  case OpPattern::kMidFusionAuxiliary:
  case OpPattern::kMidFusionImportantAux:
  case OpPattern::kOtherBroadcast:
  case OpPattern::kReshape:
  case OpPattern::kTranspose:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kExtractSlice:
    case OpPattern::kInsertSlice:
    case OpPattern::kInterleave:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kLoadStore:
    case OpPattern::kMidFusionAuxiliary:
    case OpPattern::kMidFusionImportantAux:
    case OpPattern::kOtherBroadcast:
    case OpPattern::kReshape:
    case OpPattern::kTranspose:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusibleHelper::isAnyPBRFusible(const OpPattern &patternA,
                                    const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kZeroRankElemwise:
  case OpPattern::kExtractSlice:
  case OpPattern::kInsertSlice:
  case OpPattern::kInterleave:
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kLastAxisReduce:
  case OpPattern::kLoadStore:
  case OpPattern::kMidFusionAuxiliary:
  case OpPattern::kMidFusionImportantAux:
  case OpPattern::kOtherBroadcast:
  case OpPattern::kOtherReduce:
  case OpPattern::kReshape:
  case OpPattern::kTranspose:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kZeroRankElemwise:
    case OpPattern::kExtractSlice:
    case OpPattern::kInsertSlice:
    case OpPattern::kInterleave:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kLastAxisReduce:
    case OpPattern::kLoadStore:
    case OpPattern::kMidFusionAuxiliary:
    case OpPattern::kMidFusionImportantAux:
    case OpPattern::kOtherBroadcast:
    case OpPattern::kOtherReduce:
    case OpPattern::kReshape:
    case OpPattern::kTranspose:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

size_t FusibleHelper::getMaxRank(const SmallVector<Value> &operands) {
  return std::accumulate(operands.begin(), operands.end(), 0,
                         [](const size_t &currentMax, auto nextVal) {
                           if (auto nextRank =
                                   utils::getShapeRank(nextVal.getType())) {
                             return std::max(currentMax, *nextRank);
                           }
                           return currentMax;
                         });
}

} // namespace opfusion
} // namespace hfusion
} // namespace mlir