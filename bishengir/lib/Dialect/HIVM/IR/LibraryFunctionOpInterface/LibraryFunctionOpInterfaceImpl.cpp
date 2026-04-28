//===- LibraryFunctionOpInterfaceImpl.cpp - library function op impls -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVMDialectExtension.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Interfaces/LibraryFunctionOpInterface.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>

#include "bishengir/Dialect/HIVM/Interfaces/LibraryFunctionOpInterface.cpp.inc"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::hivm::detail;
using namespace mlir::hivm::util;

namespace mlir::hivm {
std::string concatVectorOpLibraryCallName(const std::string &baseCallName,
                                          int rank,
                                          const std::string &elemTypeName) {
  std::stringstream ss;
  ss << baseCallName << "_" << rank << "d"
     << "_" << elemTypeName;
  return ss.str();
}

void pushStringifiedIfSuccessPresent(std::stringstream &ss, Value val) {
  if (!val)
    return;
  FailureOr<std::string> str = stringfyConstantIntOpValue(val);
  assert(succeeded(str));
  ss << *str;
}

int getOpLibraryCallRankImpl(Operation *op, int rank) {
  assert(rank > 0 && "Invalid operand rank.");
  std::optional<int> maxOpRank =
      cast<OpWithLibraryFunction>(op).getOpLibraryMaxRank();
  assert(maxOpRank.has_value() && "Invalid max operand rank.");
  return std::min(rank, maxOpRank.value());
}

template <typename OpTy> std::string getCumOpLibraryCallName(OpTy op) {
  StringRef baseName = op.getOpName();
  ShapedType srcVecType = cast<ShapedType>(op.getSrc().getType());
  Type elemType = srcVecType.getElementType();
  int64_t cumDim = op.getCumDims()[0];
  bool reverse = op.getReverse();

  std::stringstream ss;
  ss << baseName.data() << (cumDim > 0 ? "_ara_" : "_ra_")
     << (reverse ? "reverse_" : "") << getTypeName(op.getLoc(), elemType);
  return ss.str();
}

std::string getVSortOpLibraryCallName(VSortOp sortOp,
                                      std::optional<bool> /*isOpsAligned*/) {
  StringRef baseName = sortOp.getOpName();
  ShapedType srcVecType = cast<ShapedType>(sortOp.getSrc().getType());
  Type elemType = srcVecType.getElementType();

  bool needSortIndex = sortOp.getDst().size() == 2;
  std::stringstream ss;
  ss << baseName.data();
  if (needSortIndex) {
    ss << "_with_index";
  }

  ss << "_1d_" << getTypeName(sortOp.getLoc(), elemType);
  return ss.str();
}

// TODO: use stringifyAddressSpace after the library call names are consistent
std::map<AddressSpace, std::string> kAddressSpace2LibraryName = {
    {AddressSpace::UB, "ubuf"},
    {AddressSpace::GM, "gm"},
    {AddressSpace::L1, "cbuf"}};

std::string getLibraryCallNameForCopyLikeOp(std::string baseCallName,
                                            Type srcType, Type dstType,
                                            Location loc, int rank) {
  auto srcScope = getHIVMAddressSpace(srcType);
  assert(kAddressSpace2LibraryName.find(srcScope) !=
             kAddressSpace2LibraryName.cend() &&
         "Unsupported src address space");
  auto dstScope = getHIVMAddressSpace(dstType);
  assert(kAddressSpace2LibraryName.find(dstScope) !=
             kAddressSpace2LibraryName.cend() &&
         "Unsupported dst address space");
  std::string srcScopeName = kAddressSpace2LibraryName.at(srcScope);
  std::string dstScopeName = kAddressSpace2LibraryName.at(dstScope);
  std::string src2DstName =
      llvm::formatv("{0}_to_{1}", srcScopeName, dstScopeName);

  std::string dataTypeStr = getTypeName(loc, getElementTypeOrSelf(srcType));
  std::string libCallDim = std::to_string(rank) + "d";

  std::string callLibraryName = llvm::formatv(
      "{0}_{1}_{2}_{3}", baseCallName, src2DstName, libCallDim, dataTypeStr);
  return callLibraryName;
}

template <typename OpTy>
std::string getCopyLikeOpLibraryCallName(OpTy op,
                                         std::optional<bool> /*isOpsAligned*/) {
  MemRefType srcMemref = cast<MemRefType>(op.getSrcOperandType());
  assert(srcMemref.getMemorySpace() &&
         "Source should have memory space by now.");
  int64_t rank = srcMemref.getRank();
  auto baseCallName = getLibraryCallNameForCopyLikeOp(
      op.getOpName().str(), op.getSrc().getType(), op.getDst().getType(),
      op.getLoc(), getOpLibraryCallRankImpl(op.getOperation(), rank));
  return baseCallName;
}

std::string getVCmpOpLibraryCallName(VCmpOp concreteOp,
                                     std::optional<bool> /*isOpsAligned*/) {
  StringRef modeName = stringifyCompareMode(concreteOp.getCompareMode());
  std::string baseCallName = concreteOp.getOpName().str();
  if (!(isa<ShapedType>(concreteOp.getSrc()[1].getType())))
    baseCallName = baseCallName + "s";
  baseCallName = baseCallName + "_" + modeName.str();
  Type elemType =
      getElementTypeOrSelf(concreteOp.getDpsInputs().front().getType());
  std::string elemTypeName = getTypeName(concreteOp.getLoc(), elemType);
  int rank = static_cast<int>(concreteOp.getNumLoops());
  return concatVectorOpLibraryCallName(
      baseCallName, getOpLibraryCallRankImpl(concreteOp.getOperation(), rank),
      elemTypeName);
}

std::string getVCastOpLibraryCallName(VCastOp concreteOp,
                                      std::optional<bool> /*isOpsAligned*/) {
  MemRefType srcMemref = cast<MemRefType>(concreteOp.getSingleSrc().getType());
  int rank = srcMemref.getRank();
  auto baseCallName = concreteOp.getOpName().str();
  bool tempBufferCond = srcMemref.getElementType().isInteger(1);
  std::stringstream ss;
  ss << baseCallName << "_" << concreteOp.getCastName(false) << "_"
     << getOpLibraryCallRankImpl(concreteOp.getOperation(), rank) << "d";

  auto srcType = getElementTypeOrSelf(concreteOp.getSrc()[0]);
  auto dstType = getElementTypeOrSelf(concreteOp.getDst()[0]);
  const bool isI16ToI8 = srcType.isInteger(16) && dstType.isInteger(8);
  const bool isI32ToI8 = srcType.isInteger(32) && dstType.isInteger(8);
  const bool isI32ToI16 = srcType.isInteger(32) && dstType.isInteger(16);
  const bool isI64ToI8 = srcType.isInteger(64) && dstType.isInteger(8);
  const bool isI64ToI16 = srcType.isInteger(64) && dstType.isInteger(16);
  const bool isI64ToI32 = srcType.isInteger(64) && dstType.isInteger(32);
  if ((isI32ToI8 || isI16ToI8 || isI32ToI16 || isI64ToI8 || isI64ToI16 ||
       isI64ToI32) &&
      concreteOp.getRoundMode() == hivm::RoundMode::TRUNCWITHOVERFLOW) {
    ss << "_with_overflow";
    bool disableSizeAlignForCast = false;
    if (auto funcOp = concreteOp->getParentOfType<func::FuncOp>()) {
      if (funcOp->hasAttr(hivm::DisableSizeAlignForCastAttr::name)) {
        disableSizeAlignForCast = true;
      }
    }
    // Disable size align for overflow cast only when the cast is I32 to I8 or I16 to I8
    if (disableSizeAlignForCast && (isI32ToI8 || isI16ToI8))
      ss << "_no_size_align";
  } else {
    ss << (tempBufferCond ? "_with_temp" : "_with_mode");
  }
  return ss.str();
}

std::string getVSelOpLibraryCallName(VSelOp concreteOp,
                                     std::optional<bool> /*isOpsAligned*/) {
  std::string baseCallName = concreteOp.getOpName().str();
  Type elemType =
      getElementTypeOrSelf(concreteOp.getDpsInputs().back().getType());
  std::string elemTypeName = getTypeName(concreteOp.getLoc(), elemType);
  int rank = static_cast<int>(concreteOp.getNumLoops());

  // Start off with the assumption of Scalar-Scalar Input
  std::string selectTypeName = "_ss";

  bool src0ScalarType = concreteOp.getSrc()[1].getType().isIntOrFloat();
  bool src1ScalarType = concreteOp.getSrc()[2].getType().isIntOrFloat();
  // If src0 and src1 are scalar types
  if (!src0ScalarType && !src1ScalarType) {
    selectTypeName = "_vv";
    Type condType = getElementTypeOrSelf(concreteOp.getSrc()[0].getType());
    std::string condTypeName = getTypeName(concreteOp.getLoc(), condType);
    elemTypeName = condTypeName + "_" + elemTypeName;
  }

  // If src0 is vector and src1 is scalar
  if (!src0ScalarType && src1ScalarType) {
    selectTypeName = "_vs";
  }

  // Emit error when src0 is scalar and src1 is vector
  if (src0ScalarType && !src1ScalarType) {
    selectTypeName = "_sv";
  }

  return concatVectorOpLibraryCallName(
      baseCallName + selectTypeName,
      getOpLibraryCallRankImpl(concreteOp.getOperation(), rank), elemTypeName);
}

std::string getNZ2NDOpLibraryCallName(NZ2NDOp concreteOp,
                                      std::optional<bool> /*isOpsAligned*/) {
  // check address space
  Type srcType = concreteOp.getSrc().getType();
#ifndef NDEBUG
  AddressSpace srcScope = getHIVMAddressSpace(srcType);
  assert(srcScope == AddressSpace::L1 && "src scope should be L1");
  Type dstType = concreteOp.getDst().getType();
  AddressSpace dstScope = getHIVMAddressSpace(dstType);
  assert(dstScope == AddressSpace::GM && "dst scope should be GM");
#endif
  // get dimensions
  MemRefType srcMemref = cast<MemRefType>(concreteOp.getSrcOperandType());
  std::string srcRankStr = std::to_string(srcMemref.getRank()) + "d";
  MemRefType dstMemref = cast<MemRefType>(concreteOp.getDstOperandType());
  std::string dstRankStr = std::to_string(dstMemref.getRank()) + "d";
  // get data type
  std::string dataTypeStr =
      getTypeName(concreteOp.getLoc(), getElementTypeOrSelf(srcType));
  // make library function name
  return concreteOp.getOpName().str() + "_" + srcRankStr + "_to_" + dstRankStr +
         "_" + dataTypeStr;
}

std::string debugCallNameMangleSuffix(Operation *op) {
  std::string suffix = "";
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return suffix;
  }
  TModuleCoreTypeAttr attr = dyn_cast_or_null<TModuleCoreTypeAttr>(
      moduleOp->getAttr(TModuleCoreTypeAttr::name));
  if (attr && attr.getModuleCoreType() == TModuleCoreType::MIX) {
    // getOpLibraryCallName is called in HIVMToStandard
    // where mix functions have already been splitted.
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return suffix;
    }
    std::optional<TFuncCoreType> funcCoreType = queryFuncCoreType(funcOp);
    if (funcCoreType.has_value()) {
      if (funcCoreType.value() == TFuncCoreType::AIC) {
        suffix = ".cube";
      } else if (funcCoreType.value() == TFuncCoreType::AIV) {
        suffix = ".vector";
      }
    }
  }
  return suffix;
}

std::string getVGatherOpLibraryCallName(VGatherOp concreteOp,
                                        std::optional<bool> isOpsAligned) {
  StringRef baseName = concreteOp.getOpName();
  ShapedType srcVecType = cast<ShapedType>(concreteOp.getSrc().getType());
  Type elemType = srcVecType.getElementType();

  std::stringstream ss;
  ss << baseName.data() << "_1d_" << getTypeName(concreteOp.getLoc(), elemType);
  return ss.str();
}

std::string getVGatherMaskOpLibraryCallName(VGatherMaskOp concreteOp,
                                            std::optional<bool> isOpsAligned) {
  StringRef baseName = concreteOp.getOpName();
  ShapedType srcVecType = cast<ShapedType>(concreteOp.getSrc().getType());
  Type elemType = srcVecType.getElementType();

  std::stringstream ss;
  ss << baseName.data() << "_1d_" << getTypeName(concreteOp.getLoc(), elemType);
  return ss.str();
}
  
std::string getDebugOpLibraryCallName(DebugOp concreteOp,
                                      std::optional<bool> /*isOpsAligned*/) {
  std::string callName = concreteOp.getDebugtype().str();
  auto argTy = concreteOp.getArg().getType();
  if (isa<ShapedType>(argTy)) {
    auto argBufTy = cast<ShapedType>(argTy);
    int rank = argBufTy.getRank();
    std::optional<int> maybeMaxRank =
        cast<OpWithLibraryFunction>(concreteOp.getOperation())
            .getOpLibraryMaxRank();
    assert(maybeMaxRank.has_value());
    int maxOpRank = maybeMaxRank.value();
    if (rank > maxOpRank)
      concreteOp.emitError("DebugOp requires rank <= maxOpRank");
    std::string libCallDim = std::to_string(rank) + "d";
    std::string dataTypeStr =
        getTypeName(concreteOp.getLoc(), argBufTy.getElementType());
    callName += "_" + libCallDim + "_" + dataTypeStr;
    // get and append address space
    // when getOpLibraryCallName is called from HIVMToStandard,
    // address space should only be GM (CUBE or VEC) or UB (VEC)
    if (isa<MemRefType>(argTy)) {
      Attribute argAttr = cast<MemRefType>(argTy).getMemorySpace();
      if (!isa<AddressSpaceAttr>(argAttr))
        concreteOp.emitError("print-to-libcall cannot find mem space");
      AddressSpace argAddrSpace =
          dyn_cast<AddressSpaceAttr>(argAttr).getAddressSpace();
      if (!(argAddrSpace == AddressSpace::GM ||
            argAddrSpace == AddressSpace::UB))
        concreteOp.emitError(
            "print-to-libcall currently only supports GM and UB");
      if (argAddrSpace == AddressSpace::GM)
        callName = callName + "_" + "gm";
      else if (argAddrSpace == AddressSpace::UB)
        callName = callName + "_" + "ubuf";
    } else {
      concreteOp.emitError(
          "LibaryFunctionOpExternalModel<DebugOp>::getOpLibraryCallName should "
          "only be called with memref");
    }
  } else {
    // Note: in this case "_mlir_ciface_" won't be automatically added by
    // mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp
    std::string dataTypeStr = getTypeName(concreteOp.getLoc(), argTy);
    callName = "_mlir_ciface_" + callName + "_scalar_" + dataTypeStr;
    callName += "_gm"; // currently, scalar can choose either gm or ubuf since
                       // they currently call the same core
  }
  return callName + debugCallNameMangleSuffix(concreteOp.getOperation());
}

void updateStringStreamForMixMatmulOp(std::stringstream &ss,
                                      hivm::MixMatmulOp mmadOp) {
  Location mmadLoc = mmadOp.getLoc();
  for (auto vecIn : mmadOp.getPostVecFuncIns()) {
    ss << "_TV" << getTypeName(mmadLoc, getElementTypeOrSelf(vecIn.getType()));
  }

  for (auto vecIn : mmadOp.getWorkspaceIns()) {
    ss << "_TW" << getTypeName(mmadLoc, getElementTypeOrSelf(vecIn.getType()));
  }
}

void updateStringStreamForMixGroupMatmulOp(std::stringstream &ss,
                                           hivm::MixGroupMatmulOp mmadOp) {
  Location mmadLoc = mmadOp.getLoc();

  const auto &vecOuts = mmadOp.getPostVecFuncOuts();
  assert(vecOuts.size() == 1);
  ss << "_TM"
     << getTypeName(mmadLoc, getElementTypeOrSelf(vecOuts[0].getType()));
  const auto &vecIns = mmadOp.getPostVecFuncIns();
  const size_t vecInsSizeConstraint = 3;
  if (vecIns.size() != vecInsSizeConstraint)
    llvm::report_fatal_error("internal error: vecInsSizeConstraint is not 3");

  ss << "_TI"
     << getTypeName(mmadLoc, getElementTypeOrSelf(vecIns[0].getType()));
  ss << "_TO"
     << getTypeName(mmadLoc, getElementTypeOrSelf(vecIns[1].getType()));
  ss << "_TG"
     << getTypeName(mmadLoc, getElementTypeOrSelf(vecIns[2].getType()));

  ss << "_TT"
     << getTypeName(mmadLoc, getElementTypeOrSelf(
                                 mmadOp.getTokensPerExpert().getType()));
  for (auto vecIn : mmadOp.getWorkspaceIns()) {
    ss << "_TW" << getTypeName(mmadLoc, getElementTypeOrSelf(vecIn.getType()));
  }
}

template <typename GlobalMmadTy>
std::string getLibraryCallNameForGlobalMmadOps(GlobalMmadTy mmadOp) {
  Location mmadLoc = mmadOp.getLoc();
  std::stringstream ss;
  ss << mmadOp.getOpName().data();

  ss << (mmadOp.getBias() ? "_" : "_X") << "bias";
  if (mmadOp.getBias())
    ss << "_TBIAS"
       << getTypeName(mmadLoc, mmadOp.getBias().getType().getElementType());

  if (mmadOp.getDescale() && mmadOp.getDescaleMode() &&
      mmadOp.getDescaleMode().value() != hivm::DescaleMode::DescaleNull) {
    switch (mmadOp.getDescaleMode().value()) {
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
       << getTypeName(mmadLoc, mmadOp.getDescale().getType().getElementType());
  } else {
    ss << "_Xdescale";
  }

  ss << (mmadOp.getATranspose() ? "_" : "_X") << "transposeA"
     << (mmadOp.getBTranspose() ? "_" : "_X") << "transposeB";

  ss << "_TA" << getTypeName(mmadLoc, mmadOp.getA().getType().getElementType())
     << "_TB" << getTypeName(mmadLoc, mmadOp.getB().getType().getElementType())
     << "_TC" << getTypeName(mmadLoc, mmadOp.getC().getType().getElementType());

  if constexpr (std::is_same_v<GlobalMmadTy, hivm::MixMatmulOp>) {
    updateStringStreamForMixMatmulOp(ss, mmadOp);
  } else if constexpr (std::is_same_v<GlobalMmadTy, hivm::MixGroupMatmulOp>) {
    updateStringStreamForMixGroupMatmulOp(ss, mmadOp);
  }

  if (mmadOp.getTilingParams()) {
    ss << "_TT"
       << getTypeName(mmadLoc,
                      getElementTypeOrSelf(mmadOp.getTilingParams().getType()));
    return ss.str();
  }

  for (auto blockSize : mmadOp.getBlockSizes())
    pushStringifiedIfSuccessPresent(ss, blockSize);

  for (auto processSize : mmadOp.getProcessSizes())
    pushStringifiedIfSuccessPresent(ss, processSize);

  pushStringifiedIfSuccessPresent(ss, mmadOp.getSwizzleOffset());
  pushStringifiedIfSuccessPresent(ss, mmadOp.getSwizzleDirection());
  pushStringifiedIfSuccessPresent(ss, mmadOp.getEpiloguePTiles());

  return ss.str();
}

template <typename GlobalMixMatmulTy>
std::string
getLibraryCallNameForGlobalMixMatmulOps(GlobalMixMatmulTy mixMatmulOp) {
  std::string baseCallName =
      getLibraryCallNameForGlobalMmadOps<GlobalMixMatmulTy>(mixMatmulOp);
  std::stringstream ss;
  ss << baseCallName;
  if (mixMatmulOp.getCommParams()) {
    ss << "_TC"
       << getTypeName(
              mixMatmulOp.getLoc(),
              getElementTypeOrSelf(mixMatmulOp.getCommParams().getType()));
  }

  // Append core type at the end.
  auto coreType =
      mixMatmulOp->template getParentOfType<func::FuncOp>()->getAttr(
          TFuncCoreTypeAttr::name);
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

} // namespace mlir::hivm

//===----------------------------------------------------------------------===//
// InferMaxRankExternalModel
//===----------------------------------------------------------------------===//

template <typename ConcreteOp>
struct InferMaxRankExternalModel
    : public OpWithLibraryFunction::ExternalModel<
          InferMaxRankExternalModel<ConcreteOp>, ConcreteOp> {
  std::string getOpLibraryCallName(Operation *op,
                                   std::optional<bool> isOpsAligned) const {
    llvm_unreachable("Not implemented");
  }

  std::optional<int> getOpLibraryMaxRank(Operation *op) const {
    return inferOpLibraryMaxRank(op);
  }

  int getOpLibraryCallRank(Operation *op, int rank) const {
    return getOpLibraryCallRankImpl(op, rank);
  }

  int inferOpLibraryMaxRank(Operation *op) const {
    llvm_unreachable("Not implemented");
  }
};

//===----------------------------------------------------------------------===//
// StaticMaxRankExternalModel
//===----------------------------------------------------------------------===//

template <typename ConcreteOp, int MaxRank>
struct StaticMaxRankExternalModel
    : public OpWithLibraryFunction::ExternalModel<
          StaticMaxRankExternalModel<ConcreteOp, MaxRank>, ConcreteOp> {
  std::string getOpLibraryCallName(Operation *op,
                                   std::optional<bool> isOpsAligned) const {
    ConcreteOp concreteOp = cast<ConcreteOp>(op);
    // Op-specific impl.
    if constexpr (std::is_same_v<ConcreteOp, VCumsumOp> ||
                  std::is_same_v<ConcreteOp, VCumprodOp>) {
      return getCumOpLibraryCallName(concreteOp);
    }
    if constexpr (std::is_same_v<ConcreteOp, VSortOp>) {
      return getVSortOpLibraryCallName(concreteOp, isOpsAligned);
    }
    if constexpr (std::is_same_v<ConcreteOp, LoadOp> ||
                  std::is_same_v<ConcreteOp, StoreOp> ||
                  std::is_same_v<ConcreteOp, CopyOp>) {
      return getCopyLikeOpLibraryCallName(concreteOp, isOpsAligned);
    }
    if constexpr (std::is_same_v<ConcreteOp, VCmpOp>) {
      return getVCmpOpLibraryCallName(concreteOp, isOpsAligned);
    }
    if constexpr (std::is_same_v<ConcreteOp, VCastOp>) {
      return getVCastOpLibraryCallName(concreteOp, isOpsAligned);
    }
    if constexpr (std::is_same_v<ConcreteOp, VSelOp>) {
      return getVSelOpLibraryCallName(concreteOp, isOpsAligned);
    }
    if constexpr (std::is_same_v<ConcreteOp, NZ2NDOp>) {
      return getNZ2NDOpLibraryCallName(concreteOp, isOpsAligned);
    }
    if constexpr (std::is_same_v<ConcreteOp, VGatherOp>) {
      return getVGatherOpLibraryCallName(concreteOp, isOpsAligned);
    }
    if constexpr (std::is_same_v<ConcreteOp, VGatherMaskOp>) {
      return getVGatherMaskOpLibraryCallName(concreteOp, isOpsAligned);
    }
    if constexpr (std::is_same_v<ConcreteOp, DebugOp>) {
      return getDebugOpLibraryCallName(concreteOp, isOpsAligned);
    } else {
      // Wrap the default impl. in the else branch because we want to utilize
      // `if constexpr`
      std::string baseCallName = concreteOp.getOpName().str();
      std::string suffix = "";
      if (op->hasTrait<OpTrait::ElementwiseNaryOpTrait<2>::Impl>()) {
        if constexpr (std::is_same_v<ConcreteOp, VDivOp> ||
                      std::is_same_v<ConcreteOp, VSubOp>) {
          bool src0ScalarType =
              concreteOp.getDpsInputOperand(0)->get().getType().isIntOrFloat();
          bool src1ScalarType =
              concreteOp.getDpsInputOperand(1)->get().getType().isIntOrFloat();
          if (!src0ScalarType && src1ScalarType) {
            suffix = "_vs";
          }
          if (src0ScalarType && !src1ScalarType) {
            suffix = "_sv";
          }
          Type elemType =
              getElementTypeOrSelf(concreteOp.getDpsInputs().back().getType());
          std::string elemTypeName = getTypeName(concreteOp.getLoc(), elemType);
          int rank = static_cast<int>(concreteOp.getNumLoops());
          return concatVectorOpLibraryCallName(baseCallName + suffix,
                                               getOpLibraryCallRank(op, rank),
                                               elemTypeName);
        } else {
          if (!(isa<ShapedType>(
                  concreteOp.getDpsInputOperand(1)->get().getType())))
            suffix = "s_vs";
        }
      }
      Type elemType =
          getElementTypeOrSelf(concreteOp.getDpsInits().front().getType());
      std::string elemTypeName = getTypeName(concreteOp.getLoc(), elemType);
      int rank = static_cast<int>(concreteOp.getNumLoops());
      return concatVectorOpLibraryCallName(
          baseCallName + suffix, getOpLibraryCallRank(op, rank), elemTypeName);
    }
  }

  std::optional<int> getOpLibraryMaxRank(Operation *op) const {
    return MaxRank;
  }

  int getOpLibraryCallRank(Operation *op, int rank) const {
    return getOpLibraryCallRankImpl(op, rank);
  }
};

//===----------------------------------------------------------------------===//
// NoMaxRankExternalModel
//===----------------------------------------------------------------------===//

template <typename ConcreteOp>
struct NoMaxRankExternalModel
    : public OpWithLibraryFunction::ExternalModel<
          NoMaxRankExternalModel<ConcreteOp>, ConcreteOp> {
  std::string getOpLibraryCallName(Operation *op,
                                   std::optional<bool> isOpsAligned) const;

  std::optional<int> getOpLibraryMaxRank(Operation *op) const {
    return std::nullopt;
  }

  int getOpLibraryCallRank(Operation *op, int rank) const {
    llvm_unreachable("Not implemented");
  }

  int inferOpLibraryMaxRank(Operation *op) const {
    llvm_unreachable("Not implemented");
  }
};

//===----------------------------------------------------------------------===//
// NoLibCallExternalModel
//===----------------------------------------------------------------------===//

template <typename ConcreteOp>
struct NoLibCallExternalModel
    : public OpWithLibraryFunction::ExternalModel<
          NoLibCallExternalModel<ConcreteOp>, ConcreteOp> {
  std::string getOpLibraryCallName(Operation *op,
                                   std::optional<bool> isOpsAligned) const {
    llvm_unreachable("This op has no library function.");
  }

  std::optional<int> getOpLibraryMaxRank(Operation *op) const {
    llvm_unreachable("This op has no library function.");
  }

  int getOpLibraryCallRank(Operation *op, int rank) const {
    llvm_unreachable("This op has no library function.");
  }

  int inferOpLibraryMaxRank(Operation *op) const {
    llvm_unreachable("This op has no library function.");
  }
};

//===----------------------------------------------------------------------===//
// VBrcOp
//===----------------------------------------------------------------------===//

template <>
int InferMaxRankExternalModel<VBrcOp>::inferOpLibraryMaxRank(
    Operation *operation) const {
  VBrcOp op = cast<VBrcOp>(operation);
  Type srcType = op.getSrc().getType();
  MemRefType dstVecType = cast<MemRefType>(op.getDst().getType());
  int rank = dstVecType.getRank();
  if (isScalarLike(srcType))
    return 2;

  llvm::ArrayRef<int64_t> brcDims = op.getBroadcastDims();
  assert(brcDims.size() == 1 &&
         "broadcast dimensions array is not decomposed yet.");
  int brcIdx = brcDims[0];
  assert(brcIdx >= 0 && brcIdx < rank && "invalid broadcast index");
  bool lastAxis = (brcIdx == (rank - 1));
  // maxOpRank is 2d for lastAxis, 3d for firstAxis and middleAxis,
  return lastAxis ? 2 : 3;
}

template <>
std::string InferMaxRankExternalModel<VBrcOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<VBrcOp>(op);
  static std::map<AxisKind, std::string> axisKindMap = {
      {AxisKind::FIRST, "first"},
      {AxisKind::MIDDLE, "middle"},
      {AxisKind::LAST, "last"},
  };
  static std::map<AlignKind, std::string> alignKindMap = {
      {AlignKind::ALIGN, "align"},
      {AlignKind::UNALIGNED, "unalign"},
      {AlignKind::UNKNOWN, "unknown_align"},
  };
  Type srcType = concreteOp.getSrc().getType();
  Type dstType = concreteOp.getDst().getType();
  MemRefType srcVecType = dyn_cast<MemRefType>(srcType);
  MemRefType dstVecType = cast<MemRefType>(dstType);
  Type elemType = dstVecType.getElementType();
  std::stringstream ss;

  if (getHIVMAddressSpace(dstType) == hivm::AddressSpace::L1) {
    ss << "set_l1_2d_" << getTypeName(op->getLoc(), elemType);
    return ss.str();
  }

  ss << concreteOp.getOpName().data();
  const int dstRank = dstVecType.getRank();
  // get name for scalar in format brc_scalar_##type##_to_##dim##d
  if (!srcVecType) {
    assert(getElementTypeOrSelf(srcType).isIntOrFloat() &&
           "Only support scalar src");
    ss << "_scalar_" << getTypeName(op->getLoc(), srcType) << "_to_"
       << std::min(dstRank, this->inferOpLibraryMaxRank(op)) << "d";
    return ss.str();
  }

  llvm::ArrayRef<int64_t> brcDims = concreteOp.getBroadcastDims();
  assert(brcDims.size() == 1 &&
         "broadcast dimensions array is not decomposed yet");
  auto brcIdx = brcDims[0];
  auto rank = srcVecType.getRank();
  bool isBrcB8LastAxis = elemType.isInteger(8) && brcIdx == rank - 1;
  // get name for 1d vector or brc I8/I64 last axis in format brc_1d_##type##
  if (dstRank == 1 || isBrcB8LastAxis) {
    ss << "_1d_" << getTypeName(op->getLoc(), elemType);
    return ss.str();
  }

  // get name for nd vector
  AxisKind axisKind = utils::getOutlinedAxisKind(brcIdx, rank);
  rank = std::min(static_cast<int64_t>(this->inferOpLibraryMaxRank(op)), rank);

  AlignKind alignKind = deduceAlignmentForMemRefType(dstVecType);
  assert(isOpsAligned.has_value());
  if (*isOpsAligned && axisKind != AxisKind::LAST)
    alignKind = AlignKind::ALIGN;

  ss << "_" << axisKindMap[axisKind] << "_axis_" << alignKindMap[alignKind]
     << "_" << rank << "d_" << getTypeName(op->getLoc(), elemType);
  return ss.str();
}

// //===----------------------------------------------------------------------===//
// // VDeinterleaveOp
// //===----------------------------------------------------------------------===//

template <>
int InferMaxRankExternalModel<VDeinterleaveOp>::inferOpLibraryMaxRank(
    Operation *op) const {
  auto concreteOp = cast<VDeinterleaveOp>(op);
  const int maxDeInterLeaveChannelNum = 2;
  if (concreteOp.getDeInterLeaveChannelNum() > maxDeInterLeaveChannelNum &&
      concreteOp.getIndexMode() == hivm::DeinterleaveMode::CHANNEL_0) {
    // select channel0 from N channels support 2d
    return 2;
  }
  // select channel from 2 channels only support 1d
  return 1;
}

template <>
std::string InferMaxRankExternalModel<VDeinterleaveOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<VDeinterleaveOp>(op);
  auto mode = concreteOp.getIndexMode();
  assert(mode != DeinterleaveMode::ALL_CHANNELS &&
         "There shouldn't exist double mode deinterleave library call"
         "which has been decomposed.");

  assert(mode <= DeinterleaveMode::CHANNEL_1 &&
         "deinterleave mode don't support select this channel");

  StringRef baseName = concreteOp.getOpName();
  ShapedType srcVecType = cast<ShapedType>(concreteOp.getSrc().getType());
  Type elemType = srcVecType.getElementType();

  std::string modeName = stringifyDeinterleaveMode(mode).lower();

  MemRefType srcMemRefType = cast<MemRefType>(concreteOp.getSrc().getType());
  int maxRank = inferOpLibraryMaxRank(concreteOp.getOperation());
  int rank = srcMemRefType.getRank();
  rank = rank <= maxRank ? rank : maxRank;

  std::stringstream ss;
  const int maxDeInterLeaveChannelNum = 2;
  if (concreteOp.getDeInterLeaveChannelNum() > maxDeInterLeaveChannelNum) {
    assert(
        mode == DeinterleaveMode::CHANNEL_0 &&
        "deinterleave mode only support select channel0 when channel num > 2");
    ss << baseName.data() << "_" << modeName
       << "_from_"
          "n_channels_"
       << rank << "d_" << getTypeName(concreteOp.getLoc(), elemType);
    return ss.str();
  }

  ss << baseName.data() << "_" << modeName << "_from_"
     << concreteOp.getDeInterLeaveChannelNum() << "_channels"
     << "_1d_" << getTypeName(concreteOp.getLoc(), elemType);
  return ss.str();
}

//===----------------------------------------------------------------------===//
// FinishDebugOp
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<FinishDebugOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  return "_mlir_ciface_finish_debug" + debugCallNameMangleSuffix(op);
}

//===----------------------------------------------------------------------===//
// FixpipeOp
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<FixpipeOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<FixpipeOp>(op);
  StringRef baseCallName = concreteOp.getOpName();
  // TODO, support 5HD, and other transform mode
  StringRef modeName =
      concreteOp.getDmaMode() == FixpipeDMAMode::NZ2ND ? "nz2nd" : "normal";

  Type srcElemType = getElementTypeOrSelf(concreteOp.getSrcOperandType());
  Type dstElemType = getElementTypeOrSelf(concreteOp.getDstOperandType());
  int srcRank = concreteOp.getSrcOperandType().getRank();
  int dstRank = concreteOp.getDstOperandType().getRank();

  std::stringstream ss;
  ss << baseCallName.data() << "_" << modeName.data() << "_"
     << getTypeName(concreteOp.getLoc(), srcElemType) << "_to_"
     << getTypeName(concreteOp.getLoc(), dstElemType) << "_" << srcRank << "d"
     << "_to_" << dstRank << "d";
  return ss.str();
}

//===----------------------------------------------------------------------===//
// InitDebugOp
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<InitDebugOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  return "_mlir_ciface_init_debug" + debugCallNameMangleSuffix(op);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<MatmulOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  return getLibraryCallNameForGlobalMmadOps<MatmulOp>(cast<MatmulOp>(op));
}

//===----------------------------------------------------------------------===//
// MixGroupMatmulOp
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<MixGroupMatmulOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  return getLibraryCallNameForGlobalMixMatmulOps<MixGroupMatmulOp>(
      cast<MixGroupMatmulOp>(op));
}

//===----------------------------------------------------------------------===//
// MixMatmulOp
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<MixMatmulOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  return getLibraryCallNameForGlobalMixMatmulOps<MixMatmulOp>(
      cast<MixMatmulOp>(op));
}

//===----------------------------------------------------------------------===//
// MmadL1Op
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<MmadL1Op>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<MmadL1Op>(op);
  auto baseCallName = concreteOp.getOpName().str();
  auto srcTypeName =
      getTypeName(concreteOp.getLoc(),
                  getElementTypeOrSelf(concreteOp.getDpsInputs()[0].getType()));
  auto dstTypeName =
      getTypeName(concreteOp.getLoc(),
                  getElementTypeOrSelf(concreteOp.getDpsInits()[0].getType()));
  auto transposeA = concreteOp.getATranspose();
  auto transposeB = concreteOp.getBTranspose();
  auto enableHF32 = concreteOp.getEnable_HF32();
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
  if (concreteOp.getPerChannelBias()) {
    auto biasTypeName = getTypeName(
        concreteOp.getLoc(),
        getElementTypeOrSelf(concreteOp.getPerChannelBias().getType()));
    return baseCallName + "_with_" + biasTypeName + "_bias_" + srcTypeName +
           "_to_" + dstTypeName;
  } else {
    return baseCallName + "_" + srcTypeName + "_to_" + dstTypeName + transName;
  }
}

//===----------------------------------------------------------------------===//
// Conv1DL1Op
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<Conv1DL1Op>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<Conv1DL1Op>(op);
  auto baseCallName = std::string("conv2d_group");

  auto srcTypeName =
      getTypeName(concreteOp.getLoc(),
                  getElementTypeOrSelf(concreteOp.getDpsInputs()[0].getType()));
  auto dstTypeName =
      getTypeName(concreteOp.getLoc(),
                  getElementTypeOrSelf(concreteOp.getDpsInits()[0].getType()));
  return baseCallName + "_" + srcTypeName + "_to_" + dstTypeName;
}

//===----------------------------------------------------------------------===//
// Conv2DL1Op
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<Conv2DL1Op>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<Conv2DL1Op>(op);
  auto baseCallName = std::string("conv2d_group");

  auto srcTypeName =
      getTypeName(concreteOp.getLoc(),
                  getElementTypeOrSelf(concreteOp.getDpsInputs()[0].getType()));
  auto dstTypeName =
      getTypeName(concreteOp.getLoc(),
                  getElementTypeOrSelf(concreteOp.getDpsInits()[0].getType()));
  return baseCallName + "_" + srcTypeName + "_to_" + dstTypeName;
}

//===----------------------------------------------------------------------===//
// ND2NZOp
//===----------------------------------------------------------------------===//

template <>
std::string NoMaxRankExternalModel<ND2NZOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<ND2NZOp>(op);
  std::string callName = concreteOp.getOpName().str();
  for (Operation *nextOp : concreteOp.getDst().getUsers()) {
    if (auto mmadl1Op = llvm::dyn_cast<hivm::MmadL1Op>(nextOp)) {
      if (mmadl1Op.getPerChannelBias() == concreteOp.getDst())
        callName = callName + "_forbias";
    }
  }
  Type eleType = getElementTypeOrSelf(concreteOp.getDpsInputs()[0].getType());
  auto elemTypeName = getTypeName(concreteOp.getLoc(), eleType);
  return callName + "_" + elemTypeName;
}

// //===----------------------------------------------------------------------===//
// // VReduceOp
// //===----------------------------------------------------------------------===//

template <>
std::string InferMaxRankExternalModel<VReduceOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  bool enableVCG = false;
  auto concreteOp = cast<VReduceOp>(op);
  if (concreteOp->getParentOfType<func::FuncOp>()->hasAttr(
          hivm::EnableSavingUbAttr::name)) {
    enableVCG = true;
  }
  StringRef baseName = concreteOp.getOpName();
  MemRefType srcVecType = cast<MemRefType>(concreteOp.getSrc().getType());
  int rank = srcVecType.getRank();
  llvm::ArrayRef<int64_t> reduceDims = concreteOp.getReduceDims();
  assert(reduceDims.size() == 1 &&
         "reduce dimensions array is not decomposed yet");

  bool firstAxis = reduceDims[0] == 0;
  bool lastAxis = reduceDims[0] == rank - 1;
  bool midAxis = !firstAxis && !lastAxis;
  if (firstAxis && (rank > 2) &&
      VReduceOp::isArgminOrArgmax(concreteOp.getArithAttr().getReduceOp())) {
    rank = 2;
  }

  auto reduceOpName =
      stringifyReduceOperation(concreteOp.getArith().getReduceOp());
  std::stringstream ss;
  if (concreteOp.useVectorCrossIntr(lastAxis, rank)) {
    ss << (enableVCG ? "enablevcg_" : "enablevc_");
  }
  ss << baseName.data() << "_" << reduceOpName.str();
  if (concreteOp.getIndices()) {
    ss << "_with_specified_index";
  }
  const int dim3Rank = 3;
  const int maxLastDim = 2;
  if (rank == 1) {
    ss << "_r_";
  } else if ((firstAxis && rank >= dim3Rank) ||
             (midAxis && (rank - reduceDims[0] >= dim3Rank))) {
    ss << "_ra0a1_";
    rank = dim3Rank;
  } else if ((firstAxis && rank < dim3Rank) ||
             (midAxis && (rank - reduceDims[0] < dim3Rank))) {
    ss << "_ra_";
  } else if (lastAxis) {
    ss << "_ar_";
    rank = std::min(rank, maxLastDim);
  }

  Type eleType = srcVecType.getElementType();
  ss << getTypeName(concreteOp.getLoc(), eleType);
  return ss.str();
}

template <>
int InferMaxRankExternalModel<VReduceOp>::inferOpLibraryMaxRank(
    Operation *op) const {
  auto concreteOp = cast<VReduceOp>(op);
  llvm::ArrayRef<int64_t> reduceDims = concreteOp.getReduceDims();
  assert(!reduceDims.empty() && "reduce dimensions array must not be empty.");
  assert(reduceDims.size() == 1 &&
         "reduce dimensions array is not decomposed yet.");
  int reduceIdx = reduceDims[0];
  MemRefType srcVecType = cast<MemRefType>(concreteOp.getSrc().getType());
  int rank = srcVecType.getRank();
  assert(rank > 0 && "invalid MemRefType rank");
  // R: 1d
  if (rank == 1) {
    return 1;
  }
  bool firstAxis = (reduceIdx == 0);
  bool lastAxis = (reduceIdx == rank - 1);
  // RA0A1: 3d
  if (firstAxis && rank >= 3) {
    return 3;
  }
  // RA: 2d; AR: 2d
  if (firstAxis || lastAxis) {
    return 2;
  }
  llvm_unreachable("no support for middle axis reduction");
}

//===----------------------------------------------------------------------===//
// VTransposeOp
//===----------------------------------------------------------------------===//

template <>
int InferMaxRankExternalModel<VTransposeOp>::inferOpLibraryMaxRank(
    Operation *op) const {
  auto concreteOp = cast<VTransposeOp>(op);
  const int maxRank = 3;
  ArrayRef<int64_t> permutation = concreteOp.getPermutation();
  SmallVector<int64_t> transposeAxes =
      hivm::util::getTransposeAxes(permutation);
  return hivm::util::isTransposeWithLastAxis(permutation)
             ? (hivm::util::isTransposeAdjacentAxes(transposeAxes) ? maxRank - 1
                                                                   : maxRank)
             : maxRank;
}

template <>
std::string InferMaxRankExternalModel<VTransposeOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<VTransposeOp>(op);
  ArrayRef<int64_t> permutation = concreteOp.getPermutation();
  const bool isWithLastAxis = isTransposeWithLastAxis(permutation);

  // Currently support three kinds of libs.
  // - transpose 2d lib for last axis transpose, (x, y) to (y, x);
  // - transpose 3d lib for last axis transpose, (x, y, z) to (z, y, x).
  // - transpose 3d lib for non-last axis transpose, (x, y, z) to (y, x, z).
  int dim = this->inferOpLibraryMaxRank(op);
  std::string desc = isWithLastAxis ? "with_last_axis" : "without_last_axis";

  StringRef baseName = concreteOp.getOpName();
  MemRefType srcMemrefType = cast<MemRefType>(concreteOp.getSrc().getType());
  auto elemTypeName =
      getTypeName(concreteOp.getLoc(), srcMemrefType.getElementType());

  std::stringstream ss;
  ss << baseName.data() << "_" << this->getOpLibraryCallRank(op, dim) << "d"
     << "_" << desc << "_" << elemTypeName;
  return ss.str();
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

template <>
int InferMaxRankExternalModel<CustomOp>::inferOpLibraryMaxRank(
    Operation *op) const {
  return cast<CustomOp>(op).getMaxRank();
}

template <>
std::string InferMaxRankExternalModel<CustomOp>::getOpLibraryCallName(
    Operation *op, std::optional<bool> isOpsAligned) const {
  auto concreteOp = cast<CustomOp>(op);

  // TODO: add support for built-in template library
  using InferBuiltinMaxRankFuncTy =
      std::function<std::string(std::optional<bool>)>;
  static const DenseMap<StringRef, InferBuiltinMaxRankFuncTy>
      builtinsCallName{};

  if (concreteOp.isBuiltin()) {
    const auto inferBuiltinCallName = builtinsCallName.at(concreteOp.getName());
    return inferBuiltinCallName(isOpsAligned);
  }

  return concreteOp.getSymbol();
}

#define REGISTER_STATIC_MAX_RANK(OP, MAX_RANK)                                 \
  OP::attachInterface<StaticMaxRankExternalModel<OP, /*MaxRank=*/MAX_RANK>>(   \
      *ctx)

#define REGISTER_INFER_MAX_RANK(OP)                                            \
  OP::attachInterface<InferMaxRankExternalModel<OP>>(*ctx)

#define REGISTER_NO_MAX_RANK(OP)                                               \
  OP::attachInterface<NoMaxRankExternalModel<OP>>(*ctx)

#define REGISTER_NO_LIBRARY_FUNCTION(OP)                                       \
  OP::attachInterface<NoLibCallExternalModel<OP>>(*ctx)

void bishengir::hivm::detail::registerLibraryFunctionOpInterfaceExtension(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HIVMDialect *dialect) {
    // Vector ops
    REGISTER_INFER_MAX_RANK(VBrcOp);
    REGISTER_INFER_MAX_RANK(VReduceOp);
    REGISTER_INFER_MAX_RANK(VDeinterleaveOp);
    REGISTER_INFER_MAX_RANK(VTransposeOp);

    REGISTER_INFER_MAX_RANK(CustomOp);

    REGISTER_STATIC_MAX_RANK(VExpOp, 3);
    REGISTER_STATIC_MAX_RANK(VAbsOp, 3);
    REGISTER_STATIC_MAX_RANK(VLnOp, 3);
    REGISTER_STATIC_MAX_RANK(VReluOp, 3);
    REGISTER_STATIC_MAX_RANK(VRsqrtOp, 3);
    REGISTER_STATIC_MAX_RANK(VSqrtOp, 3);
    REGISTER_STATIC_MAX_RANK(VRecOp, 3);
    REGISTER_STATIC_MAX_RANK(VNotOp, 3);
    REGISTER_STATIC_MAX_RANK(VAddOp, 3);
    REGISTER_STATIC_MAX_RANK(VMulOp, 3);
    REGISTER_STATIC_MAX_RANK(VMulExtOp, 3);
    REGISTER_STATIC_MAX_RANK(VMaxOp, 3);
    REGISTER_STATIC_MAX_RANK(VMinOp, 3);
    REGISTER_STATIC_MAX_RANK(VOrOp, 3);
    REGISTER_STATIC_MAX_RANK(VAndOp, 3);
    REGISTER_STATIC_MAX_RANK(VXorOp, 2);
    REGISTER_STATIC_MAX_RANK(VShLOp, 3);
    REGISTER_STATIC_MAX_RANK(VShROp, 3);
    REGISTER_STATIC_MAX_RANK(VPowOp, 1);
    REGISTER_STATIC_MAX_RANK(VSubOp, 3);
    REGISTER_STATIC_MAX_RANK(VDivOp, 3);
    REGISTER_STATIC_MAX_RANK(VArangeOp, 3);
    REGISTER_STATIC_MAX_RANK(VInterleaveOp, 1);
    REGISTER_STATIC_MAX_RANK(VFlipOp, 1);
    REGISTER_STATIC_MAX_RANK(VMulextendedOp, 1);
    REGISTER_STATIC_MAX_RANK(VGatherOp, 1);
    REGISTER_STATIC_MAX_RANK(VGatherMaskOp, 1);
    REGISTER_STATIC_MAX_RANK(VCumprodOp, 1);
    REGISTER_STATIC_MAX_RANK(VCumsumOp, 2);
    REGISTER_STATIC_MAX_RANK(VSortOp, 1);
    REGISTER_STATIC_MAX_RANK(VCmpOp, 1);
    REGISTER_STATIC_MAX_RANK(VCastOp, 2);
    REGISTER_STATIC_MAX_RANK(VSelOp, 1);
    REGISTER_NO_LIBRARY_FUNCTION(VTanhOp);
    REGISTER_NO_LIBRARY_FUNCTION(VSinOp);
    REGISTER_NO_LIBRARY_FUNCTION(VCosOp);
    REGISTER_NO_LIBRARY_FUNCTION(VErfOp);
    REGISTER_NO_LIBRARY_FUNCTION(VModOp);
    REGISTER_NO_LIBRARY_FUNCTION(VPadOp);
    REGISTER_NO_LIBRARY_FUNCTION(VConcatOp);

    // Dma Ops
    REGISTER_STATIC_MAX_RANK(LoadOp, 3);
    REGISTER_STATIC_MAX_RANK(StoreOp, 3);
    REGISTER_STATIC_MAX_RANK(CopyOp, 3);
    REGISTER_STATIC_MAX_RANK(NZ2NDOp, 2);
    REGISTER_NO_MAX_RANK(FixpipeOp);
    REGISTER_NO_MAX_RANK(ND2NZOp);
    REGISTER_NO_LIBRARY_FUNCTION(AtomicCasOp);
    REGISTER_NO_LIBRARY_FUNCTION(AtomicXchgOp);
    REGISTER_NO_LIBRARY_FUNCTION(AtomicRMWOp);

    // Macro Ops
    REGISTER_NO_MAX_RANK(MmadL1Op);
    REGISTER_NO_MAX_RANK(Conv1DL1Op);
    REGISTER_NO_MAX_RANK(Conv2DL1Op);
    REGISTER_NO_MAX_RANK(MatmulOp);
    REGISTER_NO_MAX_RANK(MixMatmulOp);
    REGISTER_NO_MAX_RANK(MixGroupMatmulOp);
    REGISTER_NO_LIBRARY_FUNCTION(BatchMmadL1Op);

    // Other ops
    REGISTER_STATIC_MAX_RANK(DebugOp, 4);
    REGISTER_NO_MAX_RANK(FinishDebugOp);
    REGISTER_NO_MAX_RANK(InitDebugOp);
  });
}

#undef REGISTER_VECTOR_STATIC_MAX_RANK
