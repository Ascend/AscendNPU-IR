//===- HIVMMacroOps.cpp - HIVM Macro ops implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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

template <typename GlobalMmadTy>
std::string getLibraryCallNameForGlobalMmadOps(GlobalMmadTy *mmadOp) {
  std::stringstream ss;
  ss << mmadOp->getOpName().data();

  if (mmadOp->getBias()) {
    ss << "_bias"
       << "_TBIAS"
       << hivm::detail::getTypeName(
              mmadOp->getLoc(), mmadOp->getBias().getType().getElementType());
  } else {
    ss << "_Xbias";
  }

  if (mmadOp->getDescale() && mmadOp->getDescaleMode() &&
      mmadOp->getDescaleMode().value() != hivm::DescaleMode::DescaleNull) {
    switch (mmadOp->getDescaleMode().value()) {
    case hivm::DescaleMode::DescalePerChannel:
      ss << "_descalePerChannel";
      break;
    case hivm::DescaleMode::DescalePerTensor:
      ss << "_descalePerTensor";
      break;
    default:
      llvm_unreachable("Unsupported descale mode");
    }
    ss << "_TDESCALE"
       << hivm::detail::getTypeName(
              mmadOp->getLoc(),
              mmadOp->getDescale().getType().getElementType());
  } else {
    ss << "_Xdescale";
  }

  ss << (mmadOp->getATranspose() ? "_" : "_X") << "transposeA"
     << (mmadOp->getBTranspose() ? "_" : "_X") << "transposeB";

  ss << "_TA"
     << hivm::detail::getTypeName(mmadOp->getLoc(),
                                  mmadOp->getA().getType().getElementType())
     << "_TB"
     << hivm::detail::getTypeName(mmadOp->getLoc(),
                                  mmadOp->getB().getType().getElementType())
     << "_TC"
     << hivm::detail::getTypeName(mmadOp->getLoc(),
                                  mmadOp->getC().getType().getElementType());

  if constexpr (std::is_same_v<GlobalMmadTy, hivm::MixMatmulOp>) {
    for (auto vecIn : mmadOp->getPostVecFuncIns()) {
      ss << "_TV"
         << hivm::detail::getTypeName(mmadOp->getLoc(),
                                      getElementTypeOrSelf(vecIn.getType()));
    }

    for (auto vecIn : mmadOp->getWorkspaceIns()) {
      ss << "_TW"
         << hivm::detail::getTypeName(mmadOp->getLoc(),
                                      getElementTypeOrSelf(vecIn.getType()));
    }
  } else if constexpr (std::is_same_v<GlobalMmadTy, hivm::MixGroupMatmulOp>) {
    const auto &vecOuts = mmadOp->getPostVecFuncOuts();
    assert(vecOuts.size() == 1);
    ss << "_TM"
       << hivm::detail::getTypeName(mmadOp->getLoc(),
                                    getElementTypeOrSelf(vecOuts[0].getType()));
    const auto &vecIns = mmadOp->getPostVecFuncIns();
    const size_t vecInsSizeConstraint = 3;
    if (vecIns.size() != vecInsSizeConstraint)
      llvm::report_fatal_error("internal error: vecInsSizeConstraint is not 3");

    ss << "_TI"
       << hivm::detail::getTypeName(mmadOp->getLoc(),
                                    getElementTypeOrSelf(vecIns[0].getType()));
    ss << "_TO"
       << hivm::detail::getTypeName(mmadOp->getLoc(),
                                    getElementTypeOrSelf(vecIns[1].getType()));
    ss << "_TG"
       << hivm::detail::getTypeName(mmadOp->getLoc(),
                                    getElementTypeOrSelf(vecIns[2].getType()));

    ss << "_TT"
       << hivm::detail::getTypeName(
              mmadOp->getLoc(),
              getElementTypeOrSelf(mmadOp->getTokensPerExpert().getType()));
    for (auto vecIn : mmadOp->getWorkspaceIns()) {
      ss << "_TW"
         << hivm::detail::getTypeName(mmadOp->getLoc(),
                                      getElementTypeOrSelf(vecIn.getType()));
    }
  }

  if (mmadOp->getTilingParams()) {
    ss << "_TT"
       << hivm::detail::getTypeName(
              mmadOp->getLoc(),
              getElementTypeOrSelf(mmadOp->getTilingParams().getType()));
    return ss.str();
  }

  for (auto blockSize : mmadOp->getBlockSizes()) {
    auto str = hivm::util::stringfyConstantIntOpValue(blockSize);
    assert(succeeded(str));
    ss << *str;
  }

  for (auto processSize : mmadOp->getProcessSizes()) {
    auto str = hivm::util::stringfyConstantIntOpValue(processSize);
    assert(succeeded(str));
    ss << *str;
  }

  if (mmadOp->getSwizzleOffset()) {
    auto str =
        hivm::util::stringfyConstantIntOpValue(mmadOp->getSwizzleOffset());
    assert(succeeded(str));
    ss << *str;
  }

  if (mmadOp->getSwizzleDirection()) {
    auto str =
        hivm::util::stringfyConstantIntOpValue(mmadOp->getSwizzleDirection());
    assert(succeeded(str));
    ss << *str;
  }

  if (mmadOp->getEpiloguePTiles()) {
    auto str =
        hivm::util::stringfyConstantIntOpValue(mmadOp->getEpiloguePTiles());
    assert(succeeded(str));
    ss << *str;
  }
  return ss.str();
}

template <typename GlobalMixMatmulTy>
std::string
getLibraryCallNameForGlobalMixMatmulOps(GlobalMixMatmulTy *mixMatmulOp) {
  std::string baseCallName =
      getLibraryCallNameForGlobalMmadOps<GlobalMixMatmulTy>(mixMatmulOp);
  std::stringstream ss;
  ss << baseCallName;
  if (mixMatmulOp->getCommParams()) {
    ss << "_TC"
       << hivm::detail::getTypeName(
              mixMatmulOp->getLoc(),
              getElementTypeOrSelf(mixMatmulOp->getCommParams().getType()));
  }

  // Append core type at the end.
  auto coreType = (*mixMatmulOp)
                      ->template getParentOfType<func::FuncOp>()
                      ->getAttr(TFuncCoreTypeAttr::name);
  auto coreTypeAttr = dyn_cast<hivm::TFuncCoreTypeAttr>(coreType);
  switch (coreTypeAttr.getFuncCoreType()) {
  case hivm::TFuncCoreType::AIV:
    ss << "_mix_aiv";
    break;
  case hivm::TFuncCoreType::AIC:
    ss << "_mix_aic";
    break;
  default:
    llvm_unreachable("Unsupported CoreType");
  }
  return ss.str();
}

llvm::SmallVector<int64_t> getBlockSizes(mlir::Value oper) {
  llvm::SmallVector<int64_t> kBlockSizes;
  auto elementType = getElementTypeOrSelf(oper.getType());
  size_t kBlockSize =
      utils::INTR_BYTES_PER_BLOCK /
      (elementType.getIntOrFloatBitWidth() / utils::kBitsToByte);
  kBlockSizes.push_back(utils::FRACTAL_BLOCK_NUM);
  kBlockSizes.push_back(kBlockSize);
  return kBlockSizes;
}
// Currently, the B8 implementation is aligned with CATLASS constraints.
llvm::SmallVector<int64_t> getBlockSizesB(mlir::Value oper, bool isBTranspose,
                                          bool isA5) {
  llvm::SmallVector<int64_t> kBlockSizes;
  auto elementType = getElementTypeOrSelf(oper.getType());
  size_t elementSize =
      (elementType.getIntOrFloatBitWidth() / utils::kBitsToByte);
  auto kBlockSize = utils::INTR_BYTES_PER_BLOCK /
                    (elementType.getIntOrFloatBitWidth() / utils::kBitsToByte);
  auto factalBlockNum = utils::FRACTAL_BLOCK_NUM;
  if (isA5) {
    factalBlockNum =
        ((elementSize == 1 && !isBTranspose) ? 32 : utils::FRACTAL_BLOCK_NUM);
  }
  kBlockSizes.push_back(factalBlockNum);
  kBlockSizes.push_back(kBlockSize);
  return kBlockSizes;
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
        /*unit_flag_cond*/ ValueRange{}, per_channel_bias, a_transpose,
        b_transpose, enable_HF32, /*unit_flag_mode*/ ArrayAttr{});
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

static bool isPerChannelPattern(OpOperand &mmOut) {
  auto defOp = traceDefOp<hivm::VBrcOp>(mmOut.get());
  if (defOp.has_value()) {
    auto brcOp = cast<hivm::VBrcOp>(defOp.value());
    // For normal perChannel pattern, bias user has acted as outC of matmulOp,
    // and there's no need to verify order of bias
    if (isSatisfiedBrcForPerChannel(brcOp))
      return true;
  }

  return false;
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

/// NoBias:
/// %1 = tensor.empty()
/// mmadL1 dst(%1)

/// %alloc = memref.alloc(): memref<#hivm.address_space<cc>>
/// mmadL1 dst(%alloc)

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

  // When the inits of the mmadL1 op is an alloc on L0C, it means that the user
  // is explicitly controlling buffer reuse on L0C. We treat it as NoBias case
  // because we don't want to decompose it to mmadL1 + add.
  auto allocOp = traceDefOp<memref::AllocOp>(matmulOutput.get());
  hivm::AddressSpace addrSpace{hivm::AddressSpace::Zero};
  if (allocOp.has_value()) {
    auto alloc = cast<memref::AllocOp>(allocOp.value());
    if (auto memSpaceAttr = alloc.getType().getMemorySpace()) {
      addrSpace = dyn_cast<AddressSpaceAttr>(memSpaceAttr).getAddressSpace();
    }
  }
  auto emptyOp = traceDefOp<tensor::EmptyOp>(matmulOutput.get());
  return (addrSpace == hivm::AddressSpace::L0C || emptyOp.has_value())
             ? MatmulBiasMode::NoBias
             : MatmulBiasMode::ElementwiseAdd;
}

llvm::SmallDenseMap<Value, DataLayoutAttr> MmadL1Op::getOperandsTargetLayout() {
  llvm::SmallDenseMap<Value, DataLayoutAttr> valLayoutMap;

  auto operA = getA();
  bool isATranspose = getATranspose().has_value();
  auto aBlockSizes = getBlockSizes(operA);
  auto mALayoutAttr = DataLayoutAttr::get(
      getContext(), isATranspose ? DataLayout::nZ : DataLayout::zN,
      std::nullopt,
      mlir::DenseI64ArrayAttr::get(getContext(), ArrayRef(aBlockSizes)));
  valLayoutMap[operA] = mALayoutAttr;

  auto operB = getB();
  bool isBTranspose = getBTranspose().has_value();
  bool isA5 = hacc::utils::isAscend950(
      this->getOperation()->getParentOfType<ModuleOp>());
  auto bBlockSizes = getBlockSizesB(operB, isBTranspose, isA5);
  auto mBLayoutAttr = DataLayoutAttr::get(
      getContext(), isBTranspose ? DataLayout::nZ : DataLayout::zN,
      std::nullopt,
      mlir::DenseI64ArrayAttr::get(getContext(), ArrayRef(bBlockSizes)));
  valLayoutMap[operB] = mBLayoutAttr;

  llvm::SmallVector<int64_t> cBlockSizes;
  cBlockSizes.push_back(utils::FRACTAL_BLOCK_NUM);
  cBlockSizes.push_back(utils::FRACTAL_BLOCK_NUM);
  auto mCLayoutAttr = DataLayoutAttr::get(
      getContext(), DataLayout::zN, std::nullopt,
      mlir::DenseI64ArrayAttr::get(getContext(), ArrayRef(cBlockSizes)));
  valLayoutMap[getC()] = mCLayoutAttr;

  if (auto bias = getPerChannelBias()) {
    auto biasLayoutAttr = DataLayoutAttr::get(getContext(), DataLayout::ND,
                                              std::nullopt, std::nullopt);
    valLayoutMap[bias] = biasLayoutAttr;
  }
  return valLayoutMap;
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
  case kDimFour: {
    auto shape = cast<MemRefType>(getA().getType()).getShape();
    // When the alloc is four-dimensional, the last two dims should be the
    // fractal block sizes.
    return DataLayoutAttr::get(
        getContext(), isTranspose ? DataLayout::nZ : DataLayout::zN,
        std::nullopt,
        mlir::DenseI64ArrayAttr::get(getContext(),
                                     ArrayRef({shape[2], shape[3]})));
  }
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
  case kDimFour: {
    auto shape = cast<MemRefType>(getB().getType()).getShape();
    // When the alloc is four-dimensional, the last two dims should be the
    // fractal block sizes.
    return DataLayoutAttr::get(
        getContext(), isTranspose ? DataLayout::nZ : DataLayout::zN,
        std::nullopt,
        mlir::DenseI64ArrayAttr::get(getContext(),
                                     ArrayRef({shape[2], shape[3]})));
  }
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

llvm::SmallDenseMap<Value, DataLayoutAttr>
MmadL1Op::getOperandsCurrentLayout() {
  llvm::SmallDenseMap<Value, DataLayoutAttr> valLayoutMap;

  auto aLayoutAttr = getOperandALayout();
  assert(succeeded(aLayoutAttr) && "Cannot get layout for Matrix A");
  valLayoutMap[getDpsInputOperand(0)->get()] = *aLayoutAttr;

  auto bLayoutAttr = getOperandBLayout();
  assert(succeeded(bLayoutAttr) && "Cannot get layout for Matrix B");
  valLayoutMap[getDpsInputOperand(1)->get()] = *bLayoutAttr;

  auto cLayoutAttr = getOperandCLayout();
  assert(succeeded(cLayoutAttr) && "Cannot get layout for Matrix C");
  valLayoutMap[getDpsInitOperand(0)->get()] = *cLayoutAttr;

  if (getPerChannelBias()) {
    auto biasLayoutAttr = getOperandBiasLayout();
    assert(succeeded(biasLayoutAttr) && "Cannot get layout for bias");
    valLayoutMap[getDpsInputOperand(getNumDpsInputs() - 1)->get()] =
        *biasLayoutAttr;
  }
  return valLayoutMap;
}

std::string MmadL1Op::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  auto baseCallName = getOpName().str();
  auto srcTypeName = hivm::detail::getTypeName(
      this->getLoc(), getElementTypeOrSelf(this->getDpsInputs()[0].getType()));
  auto dstTypeName = hivm::detail::getTypeName(
      this->getLoc(), getElementTypeOrSelf(this->getDpsInits()[0].getType()));
  auto transposeA = getATranspose();
  auto transposeB = getBTranspose();
  auto enableHF32 = getEnable_HF32();
  std::string transName = "";
  if (transposeA.has_value()) {
    transName = transName + "_ta";
  }
  if (transposeB.has_value()) {
    transName = transName + "_tb";
  }
  if (enableHF32.has_value()) {
    transName = transName + "_hf32";
  }
  if (getPerChannelBias()) {
    auto biasTypeName = hivm::detail::getTypeName(
        this->getLoc(),
        getElementTypeOrSelf(this->getPerChannelBias().getType()));
    return baseCallName + "_with_" + biasTypeName + "_bias_" + srcTypeName +
           "_to_" + dstTypeName;
  } else {
    return baseCallName + "_" + srcTypeName + "_to_" + dstTypeName + transName;
  }
}

bool MmadL1Op::isInitConstant(std::optional<bool> cst) {
  return isInitConstantForLocalMmadOp<MmadL1Op>(this, cst);
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
        /*unit_flag_cond*/ ValueRange{}, per_channel_bias, a_transpose,
        b_transpose, enable_HF32, /*unit_flag_mode*/ ArrayAttr{});
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

std::string
BatchMmadL1Op::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  llvm_unreachable("this op has no library function");
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

std::string MatmulOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  return getLibraryCallNameForGlobalMmadOps<MatmulOp>(this);
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

std::string
MixMatmulOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  return getLibraryCallNameForGlobalMixMatmulOps<MixMatmulOp>(this);
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

std::string
MixGroupMatmulOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  return getLibraryCallNameForGlobalMixMatmulOps<MixGroupMatmulOp>(this);
}
