//===- AVEVFOps.cpp - AVE Regbased ops implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include <numeric>
#include <string>

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/HIVMAVE/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::hivmave;

//===----------------------------------------------------------------------===//
// AVE Regbase Op
//===----------------------------------------------------------------------===//

static bool IsAllowedAVEBitsVector(Type type) {
  if (auto vecType = ::mlir::dyn_cast<VectorType>(type)) {
    int factor = vecType.getElementType().isInteger(64) ? 2 : 1;
    return vecType.getRank() <= 1 &&
           vecType.getNumElements() *
                   static_cast<int64_t>(vecType.getElementType().getIntOrFloatBitWidth()) <=
               factor * mlir::hivm::util::VL_BITS;
  }
  return false;
}

static Type getElementType(Type type) {
  auto vecType = ::mlir::dyn_cast<VectorType>(type);
  if (vecType)
    return vecType.getElementType();
  return nullptr;
}

template <typename HIVMAVEOp>
constexpr hivm::TypeFn getTypeDeductionHint() {
  if constexpr (std::is_same_v<HIVMAVEOp, VMaxUIOp>  ||
                std::is_same_v<HIVMAVEOp, VMinUIOp>  ||
                std::is_same_v<HIVMAVEOp, VMaxsUIOp> ||
                std::is_same_v<HIVMAVEOp, VMinsUIOp>) {
    return hivm::TypeFn::cast_unsigned;
  }

  return hivm::TypeFn::cast_signed;
}

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVMAVE/IR/AVEOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Macros to help generate Interface Method `getOpLibraryCallName`
//===----------------------------------------------------------------------===//

#define ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(OP_NAME)                     \
  std::string OP_NAME::getOpLibraryCallName() {                                \
    std::string baseCallName = getOpName().str();                              \
    auto elemType = getElementTypeOrSelf(getOperands()[0].getType());          \
    std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);  \
    std::string libName = baseCallName + "_" + elemTypeName;                   \
    std::replace(libName.begin(), libName.end(), '.', '_');                    \
    return libName;                                                            \
  }

#define ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_IRREGULAR(OP_NAME)           \
  std::string OP_NAME::getOpLibraryCallName() {                                \
    std::string baseCallName = getOpName().str();                              \
    auto elemType = getElementTypeOrSelf(getRes().getType());                  \
    std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);  \
    std ::string libName =                                                     \
        "_mlir_ciface_" + baseCallName + "_" + elemTypeName;                   \
    std::replace(libName.begin(), libName.end(), '.', '_');                    \
    return libName;                                                            \
  }

#define ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(OP_NAME)          \
  std::string OP_NAME::getOpLibraryCallName() {                                \
    std::string baseCallName = getIntrinsicName().str();                       \
    auto elemType = getElementTypeOrSelf(getOperands()[0].getType());          \
    auto typeDeductionHint = getTypeDeductionHint<OP_NAME>();                  \
    std::string elemTypeName =                                                 \
        hivm::detail::getTypeName(getLoc(), elemType, typeDeductionHint);      \
    std::string libName = "_mlir_ciface_" + baseCallName + "_" + elemTypeName; \
    std::replace(libName.begin(), libName.end(), '.', '_');                    \
    return libName;                                                            \
  }

#define ENABLE_CAST_LIKE_OP_LIBRARY_CALL_CONVENTION(OP_NAME)                   \
  std::string OP_NAME::getOpLibraryCallName() {                                \
    std::string baseCallName = "cast";                                         \
    auto elemType0 = getElementTypeOrSelf(getOperands()[0].getType());         \
    auto elemType1 = getElementTypeOrSelf(getResult().getType());              \
    std::string elemTypeName0 =                                                \
        hivm::detail::getTypeName(getLoc(), elemType0);                        \
    std::string elemTypeName1 =                                                \
        hivm::detail::getTypeName(getLoc(), elemType1);                        \
    std::string libName = "_mlir_ciface_" + baseCallName + "_" +               \
                          elemTypeName0 + "_to_" + elemTypeName1;              \
    if (std::string(#OP_NAME) == "VFUIntToFpOp" or                             \
        std::string(#OP_NAME) == "VFExtUIOp")                                  \
      libName = "_mlir_ciface_" + baseCallName + "_" + "u" +                   \
                elemTypeName0 + "_to_" + elemTypeName1;                        \
    else if (std::string(#OP_NAME) == "VFFpToUIntOp")                          \
      libName = "_mlir_ciface_" + baseCallName + "_" +                         \
                elemTypeName0 + "_to_" + "u" + elemTypeName1;                  \
    std::replace(libName.begin(), libName.end(), '.', '_');                    \
    return libName;                                                            \
  }

// VCmp(s) use attribute to construct library function name
#define ENABLE_CMP_OP_LIBRARY_CALL_CONVENTION(OP_NAME)                         \
  std::string OP_NAME::getOpLibraryCallName() {                                \
    std::string baseCallName = getOpName().str();                              \
    auto cmpKind = getCmp();                                                   \
    std::string cmpStrRaw = stringifyCmpType(cmpKind).str();                   \
    std::transform(cmpStrRaw.begin(), cmpStrRaw.end(), cmpStrRaw.begin(),      \
                   [](unsigned char c) { return std::tolower(c); });           \
    std::array<std::string, 4> uops = {"ult", "ugt", "ule", "uge"};            \
    const bool isUnsignedOp =                                                  \
        std::find(uops.begin(), uops.end(), cmpStrRaw) != uops.end();          \
    hivm::TypeFn typeDeductionHint = isUnsignedOp                              \
                                         ? hivm::TypeFn::cast_unsigned         \
                                         : hivm::TypeFn::cast_signed;          \
    auto elemType = getElementTypeOrSelf(getOperands()[0].getType());          \
    std::string elemTypeName =                                                 \
        hivm::detail::getTypeName(getLoc(), elemType, typeDeductionHint);      \
    std::string libName =                                                      \
        "_mlir_ciface_" + baseCallName + "_" + cmpStrRaw + "_" + elemTypeName; \
    std::replace(libName.begin(), libName.end(), '.', '_');                    \
    return libName;                                                            \
  }

std::string VFTruncIOp::getOpLibraryCallName() {
  std::string baseCallName = "cast";
  auto elemType0 = getElementTypeOrSelf(getOperands()[0].getType());
  auto elemType1 = getElementTypeOrSelf(getResult().getType());
  std::string elemTypeName0 =
      hivm::detail::getTypeName(getLoc(), elemType0);
  std::string elemTypeName1 =
      hivm::detail::getTypeName(getLoc(), elemType1);
  std::string libName = "_mlir_ciface_" + baseCallName + "_" +
                        elemTypeName0 + "_to_" + elemTypeName1;
  if (getSat()) {
    if (auto uniAttr = getUni()) {
      auto uni = *uniAttr;
      if (uni == hivm::UnsignedMode::SI2UI) {
        libName = "_mlir_ciface_" + baseCallName + "_" + 
                  elemTypeName0 + "_to_" + "u" + elemTypeName1 + "_sat";
      } else if (uni == hivm::UnsignedMode::UI2SI) {
        libName = "_mlir_ciface_" + baseCallName + "_" + "u" + 
                elemTypeName0 + "_to_" + elemTypeName1 + "_sat";
      } else if (uni == hivm::UnsignedMode::UI2UI) {
        libName = "_mlir_ciface_" + baseCallName + "_" + "u" +
                elemTypeName0 + "_to_" + "u" + elemTypeName1 + "_sat";
      } else if (uni == hivm::UnsignedMode::SI2SI) {
        libName = "_mlir_ciface_" + baseCallName + "_" +
                elemTypeName0 + "_to_" + elemTypeName1 + "_sat";
      }
    }
  }
  std::replace(libName.begin(), libName.end(), '.', '_');
  return libName;
}

std::string VFLoadOp::getOpLibraryCallName() {
  std::string baseCallName = getOpName().str();
  LoadDist pattern = getPattern();
  std::string patternStr = stringifyLoadDist(pattern).str();
  Type elemType = getElementTypeOrSelf(getResult(0).getType());
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);
  mlir::OperandRange indices = getIndices();
  unsigned rank = indices.size();
  std::string libName = baseCallName + "_" + patternStr + "_" + elemTypeName;
  if (pattern != LoadDist::BRC_B64 && (*this)->hasAttr("ave.unaligned_ub_access")) {
    libName += "_unalign_rank" + std::to_string(rank);
  } else {
    libName += "_rank" + std::to_string(rank);
  }
  std::replace(libName.begin(), libName.end(), '.', '_');
  return libName;
}

std::string VFMaskedStoreOp::getOpLibraryCallName() {
  std::string baseCallName = getOpName().str();
  StoreDist pattern = getPattern();
  std::string patternStr = stringifyStoreDist(pattern).str();
  Type elemType = getElementTypeOrSelf(getOperands()[0].getType());
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);
  mlir::OperandRange indices = getIndices();
  unsigned rank = indices.size();
  std::string libName = baseCallName + "_" + patternStr + "_" + elemTypeName;
  if (pattern != StoreDist::ONEPT_B64 && (*this)->hasAttr("ave.unaligned_ub_access")) {
    libName += "_unalign_rank" + std::to_string(rank);
  } else {
    libName += "_rank" + std::to_string(rank);
  }
  std::replace(libName.begin(), libName.end(), '.', '_');
  return libName;
}

std::string VFGatherOp::getOpLibraryCallName() {
  std::string baseCallName = getOpName().str();
  auto resVecTy = mlir::cast<VectorType>(getRes().getType());
  Type elemType = resVecTy.getElementType();
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);
  mlir::OperandRange indices = getIndices();
  unsigned rank = indices.size();
  std::string libName = baseCallName + "_" + elemTypeName + "_rank" + std::to_string(rank);
  std::replace(libName.begin(), libName.end(), '.', '_');
  return libName;
}

static std::string stringifyCombineKind(CombiningKind kind) {
  switch (kind) {
  case CombiningKind::ADD:
    return "vcadd";
  case CombiningKind::MAX:
  case CombiningKind::UMAX:
    return "vcmax";
  case CombiningKind::MIN:
  case CombiningKind::UMIN:
    return "vcmin";
  default:
    std::string message = "Unhandled CombiningKind::" + mlir::hivmave::stringifyCombiningKind(kind).upper() + " in switch";
    llvm_unreachable(message.c_str());
  }
}

// FIXME: CombiningKind::XORI is unsupported by current CCEC version, remove assert and add support in stringifyCombineKind when so
// Hint: execute 'find ~ -type f -name __clang_cce_vector_intrinsics.h'
std::string ReductionOp::getOpLibraryCallName() {
  CombiningKind kind = getKind();
  assert(kind != CombiningKind::XORI);
  std::string baseCallName = stringifyCombineKind(kind);
  auto elemType = getElementTypeOrSelf(getOperands()[0].getType());
  hivm::TypeFn casting = isUnsigned() ? hivm::TypeFn::cast_unsigned
                                      : hivm::TypeFn::cast_signed;
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType, casting);
  std::string libName = "_mlir_ciface_" + baseCallName + "_" + elemTypeName;
  return libName;
}

std::string VFDivOp::getOpLibraryCallName() {
  std::string baseCallName = getOpName().str();
  std::replace(baseCallName.begin(), baseCallName.end(), '.', '_');
  auto elemType = getElementTypeOrSelf(getOperands()[0].getType());
  auto casting = getCast();
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType, casting);
  std::string libName = "_mlir_ciface_" + baseCallName + "_" + elemTypeName;
  return libName;
}

// Ops have MemrefType as input
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VFScatterOp)

// default VectorType Only Ops -- Unary
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFAbsOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFNegOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFNotOp)

// default VectorType Only Ops -- Binary
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFAddOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFMulOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFSubOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFMaxOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFMinOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VMaxSIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VMaxUIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VMinSIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VMinUIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFAndOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFOrOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFXorOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFDivFHPOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFModOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFModUIOp)

// default VectorType Only Ops -- vector-scalar
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFAddsOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFMulsOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFMaxsOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFMinsOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VMinsSIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VMaxsSIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VMinsUIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VMaxsUIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFLRelusOp)

// default VectorType Only Ops -- others
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFShlOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFShrOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFShlsOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFShrsOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFInterleaveOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFDeInterleaveOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFVMULLOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_NON_MEMREF(VFSlideOp)

// cast-like Ops
ENABLE_CAST_LIKE_OP_LIBRARY_CALL_CONVENTION(VFExtSIOp)
ENABLE_CAST_LIKE_OP_LIBRARY_CALL_CONVENTION(VFExtUIOp)
ENABLE_CAST_LIKE_OP_LIBRARY_CALL_CONVENTION(VFFpToSIntOp)
ENABLE_CAST_LIKE_OP_LIBRARY_CALL_CONVENTION(VFFpToUIntOp)
ENABLE_CAST_LIKE_OP_LIBRARY_CALL_CONVENTION(VFSIntToFpOp)
ENABLE_CAST_LIKE_OP_LIBRARY_CALL_CONVENTION(VFUIntToFpOp)

// Cmp Ops
ENABLE_CMP_OP_LIBRARY_CALL_CONVENTION(VFCmpOp)
ENABLE_CMP_OP_LIBRARY_CALL_CONVENTION(VFCmpS)

// default lib name for irregular Ops
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_IRREGULAR(VFVCIOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_IRREGULAR(VFSelectOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_IRREGULAR(VFVpackOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION_IRREGULAR(VFVunpackOp)

// to align with template function call name
std ::string VFBroadcastScalarOp ::getOpLibraryCallName() {
  auto elemType = getElementTypeOrSelf(getSrc().getType());
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);
  return "_mlir_ciface_vbr_" + elemTypeName;
}

std ::string VFBroadcastScalarMaskOp ::getOpLibraryCallName() {
  auto elemType = getElementTypeOrSelf(getOperands()[0].getType());
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);
  return "_mlir_ciface_vdups_" + elemTypeName;
}

std ::string VFBroadcastVectorOp::getOpLibraryCallName() {
  auto elemType = getElementTypeOrSelf(getOperands()[0].getType());
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);
  return "_mlir_ciface_vdup_" + elemTypeName;
}
//===----------------------------------------------------------------------===//
// override Interface Method `getOpLibraryCallInputs`
//===----------------------------------------------------------------------===//

SmallVector<Value> VFSIntToFpOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs = getOperands();
  auto rnd = getRnd();
  int rndCst = rnd.has_value() ? static_cast<int>(rnd.value()) : 0;
  Value rndVal =
      b.create<arith::ConstantOp>(getLoc(), b.getI32IntegerAttr(rndCst));
  inputs.push_back(rndVal);
  return inputs;
}

SmallVector<Value> VFUIntToFpOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs = getOperands();
  auto rnd = getRnd();
  int rndCst = rnd.has_value() ? static_cast<int>(rnd.value()) : 0;
  Value rndVal =
      b.create<arith::ConstantOp>(getLoc(), b.getI32IntegerAttr(rndCst));
  inputs.push_back(rndVal);
  return inputs;
}

SmallVector<Value> VFFpToSIntOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs = getOperands();
  auto rndCst = static_cast<uint32_t>(getRnd());
  Value rndVal =
      b.create<arith::ConstantOp>(getLoc(), b.getI32IntegerAttr(rndCst));
  inputs.push_back(rndVal);
  return inputs;
}

SmallVector<Value> VFFpToUIntOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs = getOperands();
  auto rndCst = static_cast<uint32_t>(getRnd());
  Value rndVal =
      b.create<arith::ConstantOp>(getLoc(), b.getI32IntegerAttr(rndCst));
  inputs.push_back(rndVal);
  return inputs;
}

SmallVector<Value> VFVCIOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs = {getOperand()};
  auto incP = static_cast<uint32_t>(getIncreaseP());
  Value rndVal =
      b.create<arith::ConstantOp>(getLoc(), b.getI32IntegerAttr(incP));
  inputs.push_back(rndVal);
  return inputs;
}

SmallVector<Value> VFBroadcastVectorOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs = getOperands();
  auto rndCst = static_cast<uint32_t>(getLow());
  Value rndVal =
      b.create<arith::ConstantOp>(getLoc(), b.getI32IntegerAttr(rndCst));
  inputs.push_back(rndVal);
  return inputs;
}

SmallVector<Value> VFLoadOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs;
  Value base = getBase();
  inputs.push_back(base);
  // compute linear offset from indices and strides
  Value offset =
      hivmave::computeLinearMemRefOffset(b, getLoc(), base, getIndices(),
                                         b.getI64Type());
  inputs.push_back(offset);

  return inputs;
}

SmallVector<Value> VFMaskedStoreOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs;
  Value base = getBase();
  inputs.push_back(base);
  // compute linear offset from indices and strides
  Value offset =
      hivmave::computeLinearMemRefOffset(b, getLoc(), base, getIndices(),
                                         b.getI64Type());
  inputs.push_back(offset);
  if (getPattern() == StoreDist::ONEPT_B64 || !(*this)->hasAttr("ave.unaligned_ub_access")) {
    inputs.push_back(getMask());
  }
  inputs.push_back(getVal());
  return inputs;
}

SmallVector<Value> VFGatherOp::getOpLibraryCallInputs(OpBuilder &b) {
  SmallVector<Value> inputs;
  inputs.push_back(getBase());
  Value offset =
      hivmave::computeLinearMemRefOffset(b, getLoc(), getBase(), getIndices(),
                                         b.getI64Type());
  inputs.push_back(offset);
  inputs.push_back(getIndexVec());
  inputs.push_back(getMask());
  return inputs;
}

//===----------------------------------------------------------------------===//
// AVEOps Method
//===----------------------------------------------------------------------===//

LogicalResult VFLoadOp::verify() {
  unsigned rank =  static_cast<unsigned>(getMemRefType().getRank());
  if (getIndices().size() != rank) {
    return emitOpError("requires ") << rank << " indices";
  }
  return success();
}

// Simplify the following pattern:
// ```
//  %0 = hivm.hir.ave.pge <b8> <ALL>, %1 : !hivm.mask<64xb8, planar, 0>
//  %1 = hivm.hir.ave.preg.cast %0 :
//     !hivm.mask<64xb8, planar, 0> to !hivm.preg<64xb16, planar, 0>
// ```
// into
// ```
//  %1 = hivm.hidden.pge <b16> <ALL>, %1 : !hivm.mask<64xb16, planar, 0>
// ```
struct SimplifyPregTypeCastPattern : public OpRewritePattern<VFPregTypeCastOp> {
  using OpRewritePattern<VFPregTypeCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VFPregTypeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSrc();
    auto resType = mlir::dyn_cast<AVE_MaskTypeType>(op.getRes().getType());
    auto definingOp = src.getDefiningOp();

    if (isa<VFPgeOp>(definingOp)) {
      auto pgeOp = cast<VFPgeOp>(definingOp);
      rewriter.replaceOpWithNewOp<VFPgeOp>(op, resType, pgeOp.getPatternAttr());
      rewriter.eraseOp(pgeOp);

      return success();
    } else if (isa<VFPltOp>(definingOp)) {
      auto pltOp = cast<VFPltOp>(definingOp);
      rewriter.replaceOpWithNewOp<VFPltOp>(
          op, resType, pltOp.getNewTrueShape().getType(), pltOp.getTrueShape());
      rewriter.eraseOp(pltOp);

      return success();
    } else {
      return failure();
    }
  }
};

void hivmave::VFPregTypeCastOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyPregTypeCastPattern>(context);
}

//===----------------------------------------------------------------------===//
// VFAddsOp canonicalization
//===----------------------------------------------------------------------===//

// Fold identity vadds where scalar is 0:
//   ave.hir.vadds %vec, %cst_0, %mask -> %vec
struct IdentityVFAddsOpPattern : public OpRewritePattern<VFAddsOp> {
  using OpRewritePattern<VFAddsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VFAddsOp op,
                                PatternRewriter &rewriter) const override {
    auto scalar = op.getScalar();
    auto constOp = scalar.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return failure();

    auto constValue = constOp.getValue();

    // Float zero: vadds(vec, 0.0) -> vec
    if (isa<FloatType>(constValue.getType()) &&
        utils::isConst<FloatAttr, float_t>(constValue, 0.0)) {
      rewriter.replaceOp(op, op.getVec());
      return success();
    }

    // Integer zero: vadds(vec, 0) -> vec
    if (isa<IntegerType>(constValue.getType()) &&
        utils::isConst<IntegerAttr, int64_t>(constValue, 0)) {
      rewriter.replaceOp(op, op.getVec());
      return success();
    }

    return failure();
  }
};

void hivmave::VFAddsOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<IdentityVFAddsOpPattern>(context);
}

LogicalResult VFExtSIOp::verify() {
  Type inType = cast<VectorType>(getSrc().getType()).getElementType();
  Type outType = cast<VectorType>(getRes().getType()).getElementType();
  bool supported = true;

  if ((inType.isSignlessInteger(8) && outType.isSignlessInteger(16)) ||
      (inType.isSignlessInteger(16) && outType.isSignlessInteger(32)) ||
      (inType.isSignlessInteger(32) && outType.isSignlessInteger(64)))
    supported = getPart().has_value() && !getPp().has_value();
  else if (inType.isSignlessInteger(8) && outType.isSignlessInteger(32))
    supported = !getPart().has_value() && getPp().has_value();
  else
    return emitOpError("The conversion is not supported.");

  if (supported)
    return success();

  return emitOpError("Improper setting of #part and #pp, please check ISA");
}

LogicalResult VFExtUIOp::verify() {
  Type inType = cast<VectorType>(getSrc().getType()).getElementType();
  Type outType = cast<VectorType>(getRes().getType()).getElementType();
  bool supported = true;

  if ((inType.isSignlessInteger(8) && outType.isSignlessInteger(16)) ||
      (inType.isSignlessInteger(16) && outType.isSignlessInteger(32)) ||
      (inType.isSignlessInteger(32) && outType.isSignlessInteger(64)))
    supported = getPart().has_value() && !getPp().has_value();
  else if (inType.isSignlessInteger(8) && outType.isSignlessInteger(32))
    supported = !getPart().has_value() && getPp().has_value();
  else
    return emitOpError("The conversion is not supported.");

  if (supported)
    return success();

  return emitOpError("Improper setting of #part and #pp, please check ISA");
}

LogicalResult VFTruncIOp::verify() {
  Type inType = cast<VectorType>(getSrc().getType()).getElementType();
  Type outType = cast<VectorType>(getRes().getType()).getElementType();
  bool supported = true;

  if ((inType.isSignlessInteger(16) && outType.isSignlessInteger(8)) ||
      (inType.isSignlessInteger(32) && outType.isSignlessInteger(16)) ||
      (inType.isSignlessInteger(64) && outType.isSignlessInteger(32)))
    supported = getPart().has_value() && !getPp().has_value();
  else if (inType.isSignlessInteger(32) && outType.isSignlessInteger(8))
    supported = !getPart().has_value() && getPp().has_value();
  else
    return emitOpError("The conversion is not supported.");

  if (supported)
    return success();

  return emitOpError("Improper setting of #part and #pp, please check ISA");
}

LogicalResult VFFpToSIntOp::verify() {
  Type inType = cast<VectorType>(getSrc().getType()).getElementType();
  Type outType = cast<VectorType>(getRes().getType()).getElementType();
  bool supported = true;

  if ((inType.isF32() &&
       (outType.isSignlessInteger(64) || outType.isSignlessInteger(16))) ||
      (inType.isF16() && outType.isSignlessInteger(8)) ||
      (inType.isBF16() && outType.isSignlessInteger(32)))
    supported = getPart().has_value() && getSat().has_value();
  else if ((inType.isF32() && outType.isSignlessInteger(32)) ||
           (inType.isF16() && outType.isSignlessInteger(16)))
    supported = !getPart().has_value() && getSat().has_value();
  else if (inType.isF16() && outType.isSignlessInteger(32))
    supported = getPart().has_value() && !getSat().has_value();
  else
    return emitOpError("The conversion is not supported.");

  if (supported)
    return success();

  return emitOpError("Improper setting of #part and #sat, please check ISA");
}

LogicalResult VFSIntToFpOp::verify() {
  Type inType = cast<VectorType>(getSrc().getType()).getElementType();
  Type outType = cast<VectorType>(getRes().getType()).getElementType();
  bool supported = true;

  if ((inType.isSignlessInteger(8) && outType.isF16()) ||
      (inType.isSignlessInteger(16) && outType.isF32()))
    supported = getPart().has_value() && !getRnd().has_value();
  else if ((inType.isSignlessInteger(16) && outType.isF16()) ||
           (inType.isSignlessInteger(32) && outType.isF32()))
    supported = !getPart().has_value() && getRnd().has_value();
  else if (inType.isSignlessInteger(64) && outType.isF32())
    supported = getPart().has_value() && getRnd().has_value();
  else
    return emitOpError("The conversion is not supported.");

  if (supported)
    return success();

  return emitOpError("Improper setting of #part and #rnd, please check ISA");
}

LogicalResult VFUIntToFpOp::verify() {
  Type inType = cast<VectorType>(getSrc().getType()).getElementType();
  Type outType = cast<VectorType>(getRes().getType()).getElementType();
  bool supported = true;

  if ((inType.isSignlessInteger(8) && outType.isF16()))
    supported = getPartAttr() && !getRnd().has_value();
  else if ((inType.isSignlessInteger(64) && outType.isF32()))
    supported = getPartAttr() && getRnd().has_value();
  else
    return emitOpError("The conversion is not supported.");

  if (supported)
    return success();

  return emitOpError("Improper setting of #part and #rnd.");
}
