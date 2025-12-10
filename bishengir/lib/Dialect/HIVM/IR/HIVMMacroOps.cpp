//===- HIVMMacroOps.cpp - HIVM Macro ops implementation -------------------===//
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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.cpp.inc"

using namespace mlir;
using namespace mlir::hivm;
namespace {
// Design for 1D bias specially
constexpr size_t kDimOne = 1;
constexpr size_t kDimTwo = 2;
constexpr size_t kDimFour = 4;

FailureOr<size_t> getRankFromShapedTypeValue(Value val) {
  auto valType = dyn_cast<ShapedType>(val.getType());
  if (!valType) {
    return failure();
  }
  return valType.getRank();
}

//===----------------------------------------------------------------------===//
// Utils for Global Mmad Ops
//===----------------------------------------------------------------------===//

template <typename GlobalMmadTy>
LogicalResult verifyTilingParamsForGlobalMmadOps(GlobalMmadTy op) {
  if (op->getTilingParams() &&
      (!op->getProcessSizes().empty() || !op->getBlockSizes().empty() ||
       op->getSwizzleOffset() || op->getSwizzleDirection() ||
       op->getEpiloguePTiles()))
    return op->emitOpError("`TilingParams` and the other explicit tiling "
                           "params cannot be set at the same time");

  const int opBlockSizeConstraints = 3;
  if (!op->getBlockSizes().empty() &&
      op->getBlockSizes().size() != opBlockSizeConstraints)
    return op->emitOpError("The size of Blocksize should be 3. The order is "
                           "blockM, blockN, blockK");

  const int opProcessSizeConstraints = 3;
  if (!op->getProcessSizes().empty() &&
      op->getProcessSizes().size() != opProcessSizeConstraints)
    return op->emitOpError("The size of ProcessSizes should be 3. The order is "
                           "ProcessM, ProcessN, ProcessK");

  return success();
}

template <typename GlobalMmadTy>
LogicalResult verifyDescaleParamsForGlobalMmadOps(GlobalMmadTy op) {
  auto bShape = dyn_cast<ShapedType>(op->getB().getType()).getShape();
  auto channelDim = bShape[1U];
  auto descaleModeAttr = op->getDescaleModeAttr();
  if (!descaleModeAttr)
    return success();

  DescaleMode descaleMode = descaleModeAttr.getValue();
  if (descaleMode == DescaleMode::DescaleNull)
    return success();

  if (!op->getDescale())
    return op->emitOpError(
        "The descaleMode is defined, descale params must be defined!");

  auto descaleShape =
      dyn_cast<ShapedType>(op->getDescale().getType()).getShape();
  if (descaleShape.size() != 1U)
    return op->emitOpError("descale must must be 1D");

  auto descaleDim = descaleShape[0];
  if (!ShapedType::isDynamic(descaleDim) &&
      !ShapedType::isDynamic(channelDim)) {
    if (descaleMode == DescaleMode::DescalePerTensor && descaleDim != 1U)
      return op->emitOpError("The descaleMode is DescalePerTensor, the size of "
                             "descale is equal to 1");

    if (descaleMode == DescaleMode::DescalePerChannel &&
        descaleDim != channelDim)
      return op->emitOpError(
          "The descaleMode is DescalePerChannel, the size of "
          "descale is equal to the col size of B");
  }
  return success();
}

template <typename GlobalMmadTy>
LogicalResult verifyBiasParamsForGlobalMmadOps(GlobalMmadTy op) {
  if (!op->getBias())
    return success();

  auto bShape = dyn_cast<ShapedType>(op->getB().getType()).getShape();
  auto channelDim = bShape[1U];
  auto biasShape = dyn_cast<ShapedType>(op->getBias().getType()).getShape();
  if (biasShape.size() != 1U)
    return op->emitOpError("bias must must be 1D");

  auto biasDim = biasShape[0];
  if (!ShapedType::isDynamic(biasDim) && !ShapedType::isDynamic(channelDim)) {
    if (biasDim != channelDim)
      return op->emitOpError("The size of bias is equal to the col size of B");
  }

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Utils for Local Mmad Ops
//===----------------------------------------------------------------------===//
template <typename LocalMmadTy>
bool isInitConstantForLocalMmadOp(LocalMmadTy *localMatmulOp,
                                  std::optional<bool> cst = std::nullopt) {
  if (!cst.has_value()) {
    return false;
  }
  Value initCond = localMatmulOp->getInitCondition();
  if (llvm::isa<arith::ConstantOp>(initCond.getDefiningOp())) {
    auto cstOp = cast<arith::ConstantOp>(initCond.getDefiningOp());
    std::optional<int64_t> cstInt = getConstantIntValue(cstOp.getValue());
    return (cstInt && ((*cstInt) == cst.value()));
  }
  return false;
}

static std::optional<int64_t> getConstantFromDefine(Value constVal) {
  if (auto constOp = dyn_cast<arith::ConstantOp>(constVal.getDefiningOp())) {
    return getConstantIntValue(constOp.getValue());
  }

  return std::nullopt;
}

// Checks if matmul which is in loop
// clears the C buffer only in the first iteration
template <typename LocalMmadTy>
bool isInitFirstLoopIterForLocalMmadOp(LocalMmadTy *localMatmulOp) {
  Value initCond = localMatmulOp->getInitCondition();
  if (auto cmpOp = dyn_cast<arith::CmpIOp>(initCond.getDefiningOp())) {
    if (auto forOp = cmpOp->template getParentOfType<scf::ForOp>()) {
      auto cmpConst = getConstantFromDefine(cmpOp.getRhs());
      auto forLowerConst = getConstantFromDefine(forOp.getLowerBound());
      
      if (cmpConst.has_value() && forLowerConst.has_value()) {
        return (cmpConst.value() == forLowerConst.value()) && (cmpOp.getLhs() == forOp.getInductionVar());
      }
    }
  }
  return false;
}

Value mlir::hivm::extractMmadBiasFromPotentialUnitDimExpand(Value bias) {
  // It assumes that there only exists expand op in mmad bias defining chain,
  // while other reshape op like collapse op seems unlikely
  if (auto expandShapeOp = bias.getDefiningOp<tensor::ExpandShapeOp>()) {
    auto reassociation = expandShapeOp.getReassociationIndices();
    auto expandedShape = expandShapeOp.getResultType().getShape();
    if (llvm::all_of(reassociation, [&expandedShape](ReassociationIndices cur) {
          uint32_t nonUnitCount =
              llvm::count_if(cur, [&expandedShape](int64_t idx) {
                return expandedShape[idx] != 1;
              });

          return nonUnitCount <= 1;
        })) {
      bias = expandShapeOp.getSrc();
    }
  }

  return bias;
}

//===----------------------------------------------------------------------===//
// MmadL1Op
//===----------------------------------------------------------------------===//

void MmadL1Op::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TypeRange result_tensors, Value a, Value b,
                     Value init_condition, Value real_m, Value real_k,
                     Value real_n, Value c, Value per_channel_bias,
                     UnitAttr a_transpose, UnitAttr b_transpose,
                     UnitAttr enable_HF32) {
  build(odsBuilder, odsState, result_tensors, a, b, init_condition, real_m,
        real_k, real_n, c, /*sync_related_args*/ ValueRange{},
        /*unit_flag_cond*/ Value{}, per_channel_bias, a_transpose, b_transpose,
        enable_HF32, /*unit_flag_mode*/ UnitFlagAttr{});
}

int MmadL1Op::getNumSyncRelatedArgs() { return 7; }

SmallVector<Value>
MmadL1Op::getInputOperands(bool includeSyncRelatedArgs /*=true*/) {
  SmallVector<Value> retOperands;
  retOperands.push_back(getA());
  retOperands.push_back(getB());
  retOperands.push_back(getInitCondition());
  retOperands.push_back(getRealM());
  retOperands.push_back(getRealK());
  retOperands.push_back(getRealN());
  if (getPerChannelBias()) {
    retOperands.push_back(getPerChannelBias());
  }
  if (includeSyncRelatedArgs) {
    auto syncRelatedArgs = getSyncRelatedArgs();
    std::copy(syncRelatedArgs.begin(), syncRelatedArgs.end(),
              std::back_inserter(retOperands));
  }
  return retOperands;
}

LogicalResult MmadL1Op::verify() {
  auto syncRelatedArgs = getSyncRelatedArgs();
  auto numSyncRelatedArgs = getNumSyncRelatedArgs();
  if (!syncRelatedArgs.empty() &&
      syncRelatedArgs.size() != static_cast<size_t>(numSyncRelatedArgs)) {
    return emitOpError() << "sync_related_args should be empty or of size "
                         << numSyncRelatedArgs << " " << syncRelatedArgs;
  }

  return success();
}

static bool isSatisfiedBrcForPerChannel(hivm::VBrcOp brcOp,
                                        Operation *hookOp = nullptr) {
  // TODO: modify for batch matmul later.
  ArrayRef<int64_t> brcDims = brcOp.getBroadcastDims();
  if (brcDims.empty()) {
    return false;
  }
  Value src = brcOp.getSrc();
  // If there exists tensor::ExpandShapeOp with unit reassociation(just expand
  // size one dimension) for broadcast, here just skip this ExpandShapeOp
  if (auto expandShapeOp = src.getDefiningOp<tensor::ExpandShapeOp>())
    src = extractMmadBiasFromPotentialUnitDimExpand(src);

  // As move_l1_to_biasTable could convert fp16 to fp32, here just enable it
  if (auto castOp = src.getDefiningOp<hivm::VCastOp>())
    if (getElementTypeOrSelf(castOp.getSingleSrc().getType()).isF16() &&
        getElementTypeOrSelf(castOp.getSingleDst().getType()).isF32())
      src = castOp.getSingleSrc();

  // If hookOp is defined, it means that IR order of current candidate bias
  // tensor may be not declared before matmul, which would cause dominance
  // confusion. Here is to verify.
  if (hookOp) {
    if (src.getParentBlock() != hookOp->getBlock())
      return false;
    auto *defOp = src.getDefiningOp();
    if (!defOp)
      llvm::report_fatal_error("unhandled case for null defOp");
    if (!defOp->isBeforeInBlock(hookOp))
      return false;
  }

#ifndef NDEBUG
  ShapedType srcVecType = dyn_cast<ShapedType>(src.getType());
  assert(srcVecType);
#endif
  // only brc first dim
  return brcDims.size() == 1 && brcDims[0] == 0;
}

std::optional<Value> getPerChannelOperand(OpOperand &operand) {
  auto traceVbrcDefOp = traceDefOp<hivm::VBrcOp>(operand.get());
  auto traceIfDefOp = traceDefOp<scf::IfOp>(operand.get());
  if (traceIfDefOp.has_value()) {
    auto ifOp = cast<scf::IfOp>(traceIfDefOp.value());
    const unsigned int index = cast<OpResult>(operand.get()).getResultNumber();
    OpOperand &thenYeildOperand =
        ifOp.getThenRegion().front().getTerminator()->getOpOperand(index);
    OpOperand &elseYeildOperand =
        ifOp.getElseRegion().front().getTerminator()->getOpOperand(index);
    auto vbrcThenOperand = getPerChannelOperand(thenYeildOperand);
    auto vbrcElseOperand = getPerChannelOperand(elseYeildOperand);
    if (vbrcThenOperand.has_value() && vbrcElseOperand.has_value() &&
        vbrcThenOperand.value() == vbrcElseOperand.value()) {
      return vbrcThenOperand.value();
    }
  } else {
    if (traceVbrcDefOp.has_value()) {
      auto brcOp = cast<hivm::VBrcOp>(traceVbrcDefOp.value());
      // For normal perChannel pattern, bias user has acted as outC of matmulOp,
      // and there's no need to verify order of bias
      if (isSatisfiedBrcForPerChannel(brcOp))
        return brcOp.getSrc();
    }
  }
  return std::nullopt;
}

static bool isPerChannelPattern(OpOperand &mmOut) {
  return getPerChannelOperand(mmOut).has_value();
}

static bool isPerChannelSplitKPattern(OpOperand &mmOut) {
  Operation *localMatmulOp = mmOut.getOwner();
  if (auto blockArg = dyn_cast_if_present<BlockArgument>(mmOut.get())) {
    if (auto scfForOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      auto correspondForRes = scfForOp.getTiedLoopResult(blockArg);
      if (!(localMatmulOp->getResults()[0].hasOneUse() &&
            isa<scf::YieldOp>(*(localMatmulOp->getResults()[0].user_begin())) &&
            correspondForRes.hasOneUse() &&
            isa<hivm::VAddOp>(*(correspondForRes.user_begin()))))
        return false;
      auto vaddOp = dyn_cast<hivm::VAddOp>(*(correspondForRes.user_begin()));
      assert(vaddOp.getSrc().size() == 2);
      for (Value src : vaddOp.getSrc()) {
        auto vbrcOp = src.getDefiningOp<hivm::VBrcOp>();
        // While anchor is vaddOp after matmul in perChannelSplitK pattern,
        // here use forOp to verify whether bias is defined before matmul
        if (vbrcOp && isSatisfiedBrcForPerChannel(vbrcOp, scfForOp))
          return true;
      }
    }
  }

  return false;
}

static bool isElementwiseAddCrossLoopPattern(OpOperand &mmOut) {
  Operation *localMatmulOp = mmOut.getOwner();
  if (auto blockArg = dyn_cast_if_present<BlockArgument>(mmOut.get())) {
    if (auto scfForOp = dyn_cast_if_present<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      auto correspondYieldVal = scfForOp.getTiedLoopYieldedValue(blockArg)->get();

      return !traceDefOp<tensor::EmptyOp>(correspondYieldVal).has_value();
    }
  }

  return false;
}

/// NoBias:
/// %1 = tensor.empty()
/// mmadL1 dst(%1)

/// PerChannelAdd
/// %1 = vbrc src: (1, n) dst :(m, n)
/// mmadL1 dst(%1)

/// PerChannelAddWithSplitK
/// %init = tensor.empty()
/// %mat = for split k (%iterator = %init) {
///   %acc_mad = mmadL1 dst(%iterator)
///   yield %acc_mad
/// }
/// %bias = vbrc src: (1, n) dst :(m, n)
/// vadd(%mat, %bias)

/// ElementwiseAdd
/// %1 = ops // not 0 const
/// mmadL1 dst(%1)

/// ElementwiseCrossLoopAdd
/// %init = tensor.empty()
/// %mat = for (%iterator = %init) {
///   %acc_mad = mmadL1 dst(%iterator)
///   %vec_res = VectorOp %acc_mad 
//    yield %vec_res
/// }

/// Well, both per-channel modes are optimization and related pattern is a
/// little customized, whatever ElementwiseAdd mode will be final standby for
/// all adding bias scenario
template <typename LocalMmadTy>
MatmulBiasMode getMatmulLikeBiasMode(LocalMmadTy localMatmulOp) {
  OpOperand &matmulOutput = localMatmulOp.getCMutable();

  // Here just traces forward to find satisfied first axis VBrcOp
  if (isPerChannelPattern(matmulOutput))
    return MatmulBiasMode::PerChannelAdd;

  if (isPerChannelSplitKPattern(matmulOutput))
    return MatmulBiasMode::PerChannelAddWithSplitK;

  auto emptyOp = traceDefOp<tensor::EmptyOp>(matmulOutput.get());
  if (!emptyOp.has_value()) {
    return MatmulBiasMode::ElementwiseAdd;
  }

  if (isElementwiseAddCrossLoopPattern(matmulOutput)) {
    return MatmulBiasMode::ElementwiseCrossLoopAdd;
  }

  return MatmulBiasMode::NoBias;
}

FailureOr<DataLayoutAttr> MmadL1Op::getOperandALayout() {
  auto rank = getRankFromShapedTypeValue(getA());
  if (failed(rank)) {
    return failure();
  }
  bool isTranspose = getATranspose().has_value();
  switch (*rank) {
  case kDimTwo:
    return DataLayoutAttr::get(getContext(), DataLayout::DOTA_ND, isTranspose);
  case kDimFour:
    return DataLayoutAttr::get(getContext(),
                               isTranspose ? DataLayout::nZ : DataLayout::zN);
  default:
    return failure();
  }
}

FailureOr<DataLayoutAttr> MmadL1Op::getOperandBLayout() {
  auto rank = getRankFromShapedTypeValue(getB());
  if (failed(rank)) {
    return failure();
  }
  bool isTranspose = getBTranspose().has_value();
  switch (*rank) {
  case kDimTwo:
    return DataLayoutAttr::get(getContext(), DataLayout::DOTB_ND, isTranspose);
  case kDimFour:
    return DataLayoutAttr::get(getContext(),
                               isTranspose ? DataLayout::nZ : DataLayout::zN);
  default:
    return failure();
  }
}

FailureOr<DataLayoutAttr> MmadL1Op::getOperandCLayout() {
  auto rank = getRankFromShapedTypeValue(getC());
  if (failed(rank)) {
    return failure();
  }
  switch (*rank) {
  case kDimTwo:
    return DataLayoutAttr::get(getContext(), DataLayout::DOTC_ND);
  case kDimFour:
    return DataLayoutAttr::get(getContext(), DataLayout::zN);
  default:
    return failure();
  }
}

FailureOr<DataLayoutAttr> MmadL1Op::getOperandBiasLayout() {
  auto rank = getRankFromShapedTypeValue(getPerChannelBias());
  if (failed(rank)) {
    return failure();
  }
  switch (*rank) {
  case kDimOne:
  case kDimTwo:
    return DataLayoutAttr::get(getContext(), DataLayout::ND);
  case kDimFour:
    return DataLayoutAttr::get(getContext(), DataLayout::zN);
  default:
    return failure();
  }
}

bool MmadL1Op::isInitConstant(std::optional<bool> cst) {
  return isInitConstantForLocalMmadOp<MmadL1Op>(this, cst);
}

bool MmadL1Op::isInitFirstLoopIter() {
  return isInitFirstLoopIterForLocalMmadOp<MmadL1Op>(this);
}

void MmadL1Op::setInitCondition(Value init) {
  getInitConditionMutable().assign(init);
}

MatmulBiasMode MmadL1Op::getMatmulBiasMode() {
  return getMatmulLikeBiasMode<MmadL1Op>(*this);
}

bool MmadL1Op::shouldDecomposeBiasByElementAdd() {
  if (this->getMatmulBiasMode() != MatmulBiasMode::ElementwiseAdd ||
      !isInitConstant(false)) {
    // Type of C is not used for accumulating
    return false;
  }

  if (isSingleChainMmadToMmad<MmadL1Op>(*this)) {
    // One of accumulating situation is C to C:
    // the C can be stored in L0c and directly be the init operand of local
    // matmul like op, so no need decomposing by mmad op and additionally vector
    // add.
    return false;
  }

  // The other of accumulating situation is :
  // should decompose local matmul like op with bias to local matmul like op and
  // additional vector add op.
  return true;
}

bool MmadL1Op::shouldDecomposeBiasByCrossLoopElementAdd() {
  if (this->getMatmulBiasMode() != MatmulBiasMode::ElementwiseCrossLoopAdd ||
      !isInitFirstLoopIter()) {
    return false;
  }

  if (isSingleChainCrossLoopMmadToMmad<MmadL1Op>(*this)) {
    // One of accumulating situation is cross loop C to C:
    // the C can be stored in L0c and directly be the init operand of local
    // matmul like op, so no need decomposing by mmad op and additionally vector
    // add.
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// BatchMmadL1Op
//===----------------------------------------------------------------------===//

void BatchMmadL1Op::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TypeRange result_tensors, Value a, Value b,
                          Value init_condition, Value real_m, Value real_k,
                          Value real_n, Value c, Value per_channel_bias,
                          UnitAttr a_transpose, UnitAttr b_transpose,
                          UnitAttr enable_HF32) {
  build(odsBuilder, odsState, result_tensors, a, b, init_condition, real_m,
        real_k, real_n, c, /*sync_related_args*/ ValueRange{},
        /*unit_flag_cond*/ Value{}, per_channel_bias, a_transpose, b_transpose,
        enable_HF32, /*unit_flag_mode*/ UnitFlagAttr{});
}

int BatchMmadL1Op::getNumSyncRelatedArgs() { return 7; }

LogicalResult BatchMmadL1Op::verify() {
  auto syncRelatedArgs = getSyncRelatedArgs();
  auto numSyncRelatedArgs = getNumSyncRelatedArgs();
  if (!syncRelatedArgs.empty() &&
      syncRelatedArgs.size() != static_cast<size_t>(numSyncRelatedArgs)) {
    return emitOpError() << "sync_related_args should be empty or of size "
                         << numSyncRelatedArgs << " " << syncRelatedArgs;
  }

  return success();
}

bool BatchMmadL1Op::isInitConstant(std::optional<bool> cst) {
  return isInitConstantForLocalMmadOp<BatchMmadL1Op>(this, cst);
}


bool BatchMmadL1Op::isInitFirstLoopIter() {
  return isInitFirstLoopIterForLocalMmadOp<BatchMmadL1Op>(this);
}

void BatchMmadL1Op::setInitCondition(Value init) {
  getInitConditionMutable().assign(init);
}

MatmulBiasMode BatchMmadL1Op::getMatmulBiasMode() {
  return getMatmulLikeBiasMode<BatchMmadL1Op>(*this);
}

bool BatchMmadL1Op::shouldDecomposeBiasByElementAdd() {
  if (this->getMatmulBiasMode() != MatmulBiasMode::ElementwiseAdd ||
      !isInitConstant(false)) {
    // Type of C is not used for accumulating
    return false;
  }

  if (isSingleChainMmadToMmad<BatchMmadL1Op>(*this)) {
    // One of accumulating situation is C to C:
    // the C can be stored in L0c and directly be the init operand of local
    // matmul like op, so no need decomposing by mmad op and additionally vector
    // add.
    return false;
  }

  // The other of accumulating situation is :
  // should decompose local matmul like op with bias to local matmul like op and
  // additional vector add op.
  return true;
}

bool BatchMmadL1Op::shouldDecomposeBiasByCrossLoopElementAdd() {
  if (this->getMatmulBiasMode() != MatmulBiasMode::ElementwiseCrossLoopAdd ||
      !isInitFirstLoopIter()) {
    return false;
  }

  if (isSingleChainCrossLoopMmadToMmad<BatchMmadL1Op>(*this)) {
    // One of accumulating situation is cross loop C to C:
    // the C can be stored in L0c and directly be the init operand of local
    // matmul like op, so no need decomposing by mmad op and additionally vector
    // add.
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  if (!(getA() && getB()))
    return emitOpError("matrix A and matrix B must be defined");

  auto AShape = dyn_cast<ShapedType>(getA().getType()).getShape();
  auto BShape = dyn_cast<ShapedType>(getB().getType()).getShape();
  if (AShape.size() != 2U || BShape.size() != 2U)
    return emitOpError("matrix A and matrix B must be 2D");

  if (failed(verifyDescaleParamsForGlobalMmadOps(this)))
    return failure();

  if (failed(verifyBiasParamsForGlobalMmadOps(this)))
    return failure();

  if (failed(verifyTilingParamsForGlobalMmadOps(this)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// MixMatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MixMatmulOp::verify() {
  if (!(getA() && getB()))
    return emitOpError("matrix A and matrix B must be defined");

  auto AShape = dyn_cast<ShapedType>(getA().getType()).getShape();
  auto BShape = dyn_cast<ShapedType>(getB().getType()).getShape();
  if (AShape.size() != 2U || BShape.size() != 2U)
    return emitOpError("matrix A and matrix B must be 2D");

  if (failed(verifyDescaleParamsForGlobalMmadOps(this)))
    return failure();

  if (failed(verifyBiasParamsForGlobalMmadOps(this)))
    return failure();

  if (failed(verifyTilingParamsForGlobalMmadOps(this)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// MixGroupMatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MixGroupMatmulOp::verify() {
  if (!(getA() && getB() && getTokensPerExpert()))
    return emitOpError(
        "matrix A, matrix B and matrix TokensPerExpert must be defined");

  auto AShape = dyn_cast<ShapedType>(getA().getType()).getShape();
  if (AShape.size() != 3U)
    return emitOpError("matrix A must be 3D");

  auto BShape = dyn_cast<ShapedType>(getB().getType()).getShape();
  if (BShape.size() != 2U)
    return emitOpError("matrix B must be 2D");

  auto TokensPerExpertShape =
      dyn_cast<ShapedType>(getTokensPerExpert().getType()).getShape();
  if (TokensPerExpertShape.size() != 1U)
    return emitOpError("matrix TokensPerExpert must be 1D");

  if (failed(verifyDescaleParamsForGlobalMmadOps(this)))
    return failure();

  if (failed(verifyBiasParamsForGlobalMmadOps(this)))
    return failure();

  if (failed(verifyTilingParamsForGlobalMmadOps(this)))
    return failure();

  return success();
}
