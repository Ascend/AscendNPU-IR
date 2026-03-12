//===- HIVMVector.cpp - HIVM Vector ops implementation --------------------===//
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <sstream>

using namespace mlir;
using namespace mlir::hivm;

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Macros to help generate `getOpLibraryCallName`
//===----------------------------------------------------------------------===//

#define ENABLE_NO_DEFAULT_OP_LIBRARY_CALL_CONVENTION(OP_NAME)                  \
  std::string OP_NAME::getOpLibraryCallName(                                   \
      [[maybe_unused]] std::optional<bool> isOpsAligned) {                     \
    llvm_unreachable("this op has no library function");                       \
  }

#define ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(OP_NAME)                     \
  std::string OP_NAME::getOpLibraryCallName(                                   \
      [[maybe_unused]] std::optional<bool> isOpsAligned) {                     \
    std::string baseCallName = getOpName().str();                              \
    auto elemType = getElementTypeOrSelf(getDpsInits().front().getType());     \
    std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);  \
    int rank = static_cast<int>(getNumLoops());                                \
    return concatVectorOpLibraryCallName(                                      \
        baseCallName, getOpLibraryCallRank(rank), elemTypeName);               \
  }

#define ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(OP_NAME)              \
  std::string OP_NAME::getOpLibraryCallName(                                   \
      [[maybe_unused]] std::optional<bool> isOpsAligned) {                     \
    std::string baseCallName = getOpName().str();                              \
    if (!(isa<ShapedType>(getSrc()[1].getType())))                             \
      baseCallName = baseCallName + "s_vs";                                    \
    auto elemType = getElementTypeOrSelf(getDpsInits().front().getType());     \
    std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);  \
    int rank = static_cast<int>(getNumLoops());                                \
    return concatVectorOpLibraryCallName(                                      \
        baseCallName, getOpLibraryCallRank(rank), elemTypeName);               \
  }

#define ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION_WITH_EXTENED_SUPPORT( \
    OP_NAME)                                                                   \
  std::string OP_NAME::getOpLibraryCallName(                                   \
      [[maybe_unused]] std::optional<bool> isOpsAligned) {                     \
    std::string baseCallName = getOpName().str();                              \
    Type elemType = getElementTypeOrSelf(getDpsInputs().back().getType());     \
    std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);  \
    int rank = static_cast<int>(getNumLoops());                                \
    std::string selectTypeName = "_vv";                                        \
    bool src0ScalarType = getSrc()[0].getType().isIntOrFloat();                \
    bool src1ScalarType = getSrc()[1].getType().isIntOrFloat();                \
    if (!src0ScalarType && src1ScalarType) {                                   \
      selectTypeName = "_vs";                                                  \
    }                                                                          \
    if (src0ScalarType && !src1ScalarType) {                                   \
      selectTypeName = "_sv";                                                  \
    }                                                                          \
    if (selectTypeName == "_vv") {                                             \
      return concatVectorOpLibraryCallName(                                    \
          baseCallName, getOpLibraryCallRank(rank), elemTypeName);             \
    }                                                                          \
    return concatVectorOpLibraryCallName(baseCallName + selectTypeName,        \
                                         getOpLibraryCallRank(rank),           \
                                         elemTypeName);                        \
  }

std::string concatVectorOpLibraryCallName(const std::string &baseCallName,
                                          int rank,
                                          const std::string &elemTypeName) {
  std::stringstream ss;
  ss << baseCallName << "_" << rank << "d"
     << "_" << elemTypeName;
  return ss.str();
}

// Elemwise Unary Ops
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VExpOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VAbsOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VLnOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VReluOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VRsqrtOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VSqrtOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VRecOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VNotOp)

// Elemwise Binary Ops
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VAddOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VMulOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VMaxOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VMinOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VOrOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VAndOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VXorOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VPowOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VShLOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VShROp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VMulExtOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VModOp)
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION(VModUIOp)

#undef ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION

// Elemwise Binary Ops with Extended Support for VS and SV inputs
ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION_WITH_EXTENED_SUPPORT(VSubOp)
#undef ENABLE_DEFAULT_BINARY_OP_LIBRARY_CALL_CONVENTION_WITH_EXTENED_SUPPORT

// Other Ops
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VInterleaveOp)
ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VFlipOp)
#undef ENABLE_DEFAULT_OP_LIBRARY_CALL_CONVENTION

// Ops with no library call
ENABLE_NO_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VTanhOp)
ENABLE_NO_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VSinOp)
ENABLE_NO_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VCosOp)
ENABLE_NO_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VErfOp)
ENABLE_NO_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VConcatOp)
ENABLE_NO_DEFAULT_OP_LIBRARY_CALL_CONVENTION(VPadOp)
#undef ENABLE_NO_DEFAULT_OP_LIBRARY_CALL_CONVENTION

namespace {
template <typename HIVMOP>
LogicalResult verifyCumOp(HIVMOP op) {
  ArrayRef<int64_t> cumDims = op.getCumDims();
  ShapedType srcType = cast<ShapedType>(op.getSrc().getType());
  if (cumDims.empty()) {
    return op.emitOpError() << "have empty cum dims array";
  }
  if (static_cast<int64_t>(cumDims.size()) > srcType.getRank()) {
    return op.emitOpError() << "have too many indices in the cum dims array";
  }

  ShapedType dstType = cast<ShapedType>(op.getDst().getType());
  std::set<int64_t> cumDimSet;
  for (int64_t idx : cumDims) {
    if (idx < 0 || idx >= dstType.getRank()) {
      return op.emitOpError()
             << "have invalid index '" << idx << "' inside cum dims array";
    }
    if (cumDimSet.find(idx) != cumDimSet.end()) {
      return op.emitOpError()
             << "have duplicate index '" << idx << "' inside cum dims array";
    }
    cumDimSet.insert(idx);
  }

  if (cumDimSet.size() > 1) {
    return op.emitOpError() << "have more than one cumulative dims";
  }
  return success();
}

template <typename CUMOP>
std::string getCumOpLibraryCallName(CUMOP op) {
  StringRef baseName = op.getOpName();
  ShapedType srcVecType = cast<ShapedType>(op.getSrc().getType());
  Type elemType = srcVecType.getElementType();

  std::stringstream ss;
  ss << baseName.data() << "_ra_"
     << hivm::detail::getTypeName(op.getLoc(), elemType);
  return ss.str();
}

} // namespace

//===----------------------------------------------------------------------===//
// Binary/Unary Op build
//===----------------------------------------------------------------------===//

#define ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(OP_NAME)          \
  void OP_NAME::build(OpBuilder &odsBuilder, OperationState &odsState,         \
                      TypeRange result, ValueRange src, ValueRange dst,        \
                      DenseI64ArrayAttr transpose,                             \
                      DenseI64ArrayAttr broadcast) {                           \
    build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,     \
          transpose, broadcast);                                               \
  }

// Vector Binary Op
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VAddOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VMulOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VMinOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VMaxOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VAndOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VOrOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VSubOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VShLOp)
// Vector Unary Op
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VNotOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VAbsOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VLnOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VReluOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VExpOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VRsqrtOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VSqrtOp)
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VRecOp)
#undef ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF

//===----------------------------------------------------------------------===//
// VDivOp
//===----------------------------------------------------------------------===//
void VDivOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange result, ValueRange src, ValueRange dst,
                   bool isSigned, ArrayRef<int64_t> transpose,
                   ArrayRef<int64_t> broadcast) {
  auto transposeAttr = odsBuilder.getDenseI64ArrayAttr(transpose);
  auto broadcastAttr = odsBuilder.getDenseI64ArrayAttr(broadcast);
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/Value(),
        isSigned, transposeAttr, broadcastAttr);
}

//===----------------------------------------------------------------------===//
// VShROp
//===----------------------------------------------------------------------===//
void VShROp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange result, ValueRange src, ValueRange dst,
                   BoolAttr round, DenseI64ArrayAttr transpose,
                   DenseI64ArrayAttr broadcast) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr, round,
        transpose, broadcast);
}

//===----------------------------------------------------------------------===//
// VCmpOp
//===----------------------------------------------------------------------===//

std::string VCmpOp::getOpLibraryCallName(
    [[maybe_unused]] std::optional<bool> isOpsAligned) {
  StringRef modeName = stringifyCompareMode(this->getCompareMode());
  std::string baseCallName = getOpName().str();
  if (!(isa<ShapedType>(getSrc()[1].getType())))
    baseCallName = baseCallName + "s";
  baseCallName = baseCallName + "_" + modeName.str();
  Type elemType = getElementTypeOrSelf(getDpsInputs().front().getType());
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);
  int rank = static_cast<int>(getNumLoops());
  return concatVectorOpLibraryCallName(baseCallName, getOpLibraryCallRank(rank),
                                       elemTypeName);
}

//===----------------------------------------------------------------------===//
// VSelOp
//===----------------------------------------------------------------------===//

std::string VSelOp::getOpLibraryCallName(
    [[maybe_unused]] std::optional<bool> isOpsAligned) {
  std::string baseCallName = getOpName().str();
  Type elemType = getElementTypeOrSelf(getDpsInputs().back().getType());
  std::string elemTypeName = hivm::detail::getTypeName(getLoc(), elemType);
  int rank = static_cast<int>(getNumLoops());

  // Start off with the assumption of Scalar-Scalar Input
  std::string selectTypeName = "_ss";

  bool src0ScalarType = getSrc()[1].getType().isIntOrFloat();
  bool src1ScalarType = getSrc()[2].getType().isIntOrFloat();
  // If src0 and src1 are scalar types
  if (!src0ScalarType && !src1ScalarType) {
    selectTypeName = "_vv";
    Type condType = getElementTypeOrSelf(getSrc()[0].getType());
    std::string condTypeName = hivm::detail::getTypeName(getLoc(), condType);
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
      baseCallName + selectTypeName, getOpLibraryCallRank(rank), elemTypeName);
}

//===----------------------------------------------------------------------===//
// VBrcOp
//===----------------------------------------------------------------------===//

void VBrcOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange result, Value src, Value dst,
                   DenseI64ArrayAttr broadcast_dims) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,
        broadcast_dims);
}

void VBrcOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange result, Value src, Value dst) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,
        ArrayRef<int64_t>{});
}

LogicalResult VBrcOp::verify() {
  mlir::ModuleOp moduleOp = (*this)->getParentOfType<mlir::ModuleOp>();
  if (!hacc::utils::isAscend950(moduleOp)) {
    Type dstType = this->getDst().getType();
    ShapedType dstVecType = cast<ShapedType>(dstType);
    Type eleType = dstVecType.getElementType();
    if (eleType.isFloat8E4M3FN() || eleType.isFloat8E5M2()) {
      return this->emitError("fp8 is not supported.");
    }
  }
  // tmpBuf can be null
  auto tmpBuf = getTempBuffer();
  if (tmpBuf && tmpBuf.getType().getShape().size() != 1) {
    return emitOpError() << "temp_buffer'rank should be one";
  }

  ArrayRef<int64_t> brcDims = this->getBroadcastDims();

  if (ShapedType srcVecType = dyn_cast<ShapedType>(getSrc().getType())) {
    // src is vector type
    if (brcDims.empty()) {
      return emitOpError() << "have empty broadcast dims array";
    }
    if (static_cast<int64_t>(brcDims.size()) > srcVecType.getRank()) {
      return emitOpError()
             << "have too many indices in the broadcast dims array";
    }

    for (int64_t idx : brcDims) {
      if (idx < 0 || idx >= srcVecType.getRank()) {
        return emitOpError() << "have invalid index '" << idx
                             << "' inside broadcast dims array";
      }
      if (srcVecType.getDimSize(idx) != 1) {
        return emitOpError() << "invalid source vector shape, 'SrcVecDim["
                             << idx << "]' != 1\n";
      }
    }
  } else {
    // src is scalar type
    if (!brcDims.empty()) {
      return emitOpError("broadcast dims must be empty for scalar src");
    }
  }

  return success();
}

int VBrcOp::inferOpLibraryMaxRank() {
  Type srcType = this->getSrc().getType();
  MemRefType dstVecType = cast<MemRefType>(this->getDst().getType());
  int rank = dstVecType.getRank();
  if (isScalarLike(srcType))
    return 2;

  llvm::ArrayRef<int64_t> brcDims = this->getBroadcastDims();
  assert(brcDims.size() == 1 &&
         "broadcast dimensions array is not decomposed yet.");
  int brcIdx = brcDims[0];
  assert(brcIdx >= 0 && brcIdx < rank && "invalid broadcast index");
  bool lastAxis = (brcIdx == (rank - 1));
  // maxOpRank is 2d for lastAxis, 3d for firstAxis and middleAxis,
  return lastAxis ? 2 : 3;
}

std::string VBrcOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
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
  Type srcType = this->getSrc().getType();
  Type dstType = this->getDst().getType();
  MemRefType srcVecType = dyn_cast<MemRefType>(srcType);
  MemRefType dstVecType = cast<MemRefType>(dstType);
  std::stringstream ss;

  if (getHIVMAddressSpace(dstType) == hivm::AddressSpace::L1) {
    Type eleType = dstVecType.getElementType();
    ss << "set_l1_2d_" << hivm::detail::getTypeName(this->getLoc(), eleType);
    return ss.str();
  }

  StringRef baseName = this->getOpName();
  const int dstRank = dstVecType.getRank();
  // get name for scalar in format brc_scalar_##type##_to_##dim##d
  if (!srcVecType) {
    int rank = std::min(dstRank, this->inferOpLibraryMaxRank());
    assert(getElementTypeOrSelf(srcType).isIntOrFloat() &&
           "Only support scalar src");
    ss << baseName.data() << "_scalar_"
       << hivm::detail::getTypeName(this->getLoc(), srcType) << "_to_" << rank
       << "d";
    return ss.str();
  }

  llvm::ArrayRef<int64_t> brcDims = this->getBroadcastDims();
  assert(brcDims.size() == 1 &&
         "broadcast dimensions array is not decomposed yet");
  int brcIdx = brcDims[0];
  int64_t rank = srcVecType.getRank();
  bool isBrcB8LastAxis =
      getElementTypeOrSelf(srcType).isInteger(8) && brcIdx == rank - 1;
  // get name for 1d vector or brc I8/I64 last axis in format brc_1d_##type##
  if (srcVecType && (dstRank == 1 || isBrcB8LastAxis)) {
    ss << baseName.data() << "_1d_"
       << hivm::detail::getTypeName(this->getLoc(),
                                    srcVecType.getElementType());
    return ss.str();
  }

  // get name for nd vector
  AxisKind axisKind = utils::getOutlinedAxisKind(brcIdx, rank);
  int64_t maxlibRank = this->inferOpLibraryMaxRank();
  rank = std::min(maxlibRank, rank);

  AlignKind alignKind = hivm::util::deduceAlignmentForMemRefType(dstVecType);
  assert(isOpsAligned.has_value());
  if (*isOpsAligned && axisKind != AxisKind::LAST) {
    alignKind = AlignKind::ALIGN;
  }
  Type elemType = srcVecType.getElementType();

  ss << baseName.data() << "_" << axisKindMap[axisKind] << "_axis_"
     << alignKindMap[alignKind] << "_" << rank << "d_"
     << hivm::detail::getTypeName(this->getLoc(), elemType);
  return ss.str();
}

PIPE VBrcOp::getPipe() {
  Type dstType = this->getDst().getType();
  if (getHIVMAddressSpace(dstType) == hivm::AddressSpace::L1) {
    return PIPE::PIPE_MTE2;
  }
  if (getHIVMAddressSpace(dstType) == hivm::AddressSpace::UB) {
    return PIPE::PIPE_V;
  }
  llvm_unreachable("Unknown PIPE!");
}

//===----------------------------------------------------------------------===//
// VCastOp
//===----------------------------------------------------------------------===//

void VCastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    TypeRange result, ValueRange src, ValueRange dst,
                    hivm::RoundModeAttr round_mode, hivm::TypeFnAttr cast,
                    DenseI64ArrayAttr transpose, DenseI64ArrayAttr broadcast) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,
        round_mode, cast, transpose, broadcast);
}

void VCastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    TypeRange result, ValueRange src, ValueRange dst,
                    hivm::RoundMode round_mode, hivm::TypeFn cast,
                    ArrayRef<int64_t> transpose, ArrayRef<int64_t> broadcast) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,
        round_mode, cast, transpose, broadcast);
}

std::string VCastOp::getCastName(bool withMode = false) {
  std::string castName = "";
  ShapedType srcVcastType = cast<ShapedType>(getSingleSrc().getType());
  ShapedType dstVcastType = cast<ShapedType>(getSingleDst().getType());
  auto srcElemType = srcVcastType.getElementType();
  auto dstElemType = dstVcastType.getElementType();
  hivm::TypeFn casting = this->getCast();
  castName.append(
      hivm::detail::getTypeName(this->getLoc(), srcElemType, casting));
  castName.append("_to_");
  castName.append(
      hivm::detail::getTypeName(this->getLoc(), dstElemType, casting));
  if (withMode) {
    castName.append("_");
    castName.append(stringifyRoundMode((*this).getRoundMode()));
    castName.append("mode");
  }
  return castName;
}

LogicalResult VCastOp::verify() {
  /// considering cast f32 to f16 and cast f16 to i8 both support
  /// round/rint/floor/ceil/trunc modes, so cast f32 to i8 supports these
  /// modes.
  /// considering cast i4 to i16 only supports rint, so cast i4 to i8 only
  /// supports rint mode.

  const std::set<std::string> softSupportedCast{
      "float_to_int8_t_roundmode",
      "float_to_int8_t_rintmode",
      "float_to_int8_t_floormode",
      "float_to_int8_t_ceilmode",
      "float_to_int8_t_truncmode",
      "int4_t_to_int8_t_rintmode",
      "int8_t_to_bool_rintmode",
      "int16_t_to_bool_rintmode",
      "int32_t_to_bool_rintmode",
      "bool_to_int8_t_rintmode",
      "bool_to_float_rintmode",
      "bool_to_half_rintmode",
      "bool_to_int32_t_rintmode",
      "bool_to_float_truncmode",
      "bool_to_half_truncmode",
      "bool_to_bfloat16_t_truncmode",
      "bool_to_int16_t_rintmode",
      "bool_to_int32_t_rintmode",
      "bool_to_uint16_t_rintmode",
      "bool_to_uint32_t_rintmode",
      "bool_to_bfloat16_t_rintmode",
      "half_to_half_ceilmode",
      "half_to_half_floormode",
      "bfloat16_t_to_bfloat16_t_ceilmode",
      "bfloat16_t_to_bfloat16_t_floormode",
      "int16_t_to_int32_t_rintmode",
      "int8_t_to_int32_t_rintmode",
      "int8_t_to_int16_t_rintmode",
      "int32_t_to_int8_t_truncwithoverflowmode",
      "int16_t_to_int8_t_truncwithoverflowmode",
      "int32_t_to_int16_t_truncwithoverflowmode",
      "int64_t_to_int32_t_truncwithoverflowmode",
      "float_to_float8_e4m3_t_rintmode",
      "uint8_t_to_uint32_t_rintmode",
      "uint32_t_to_uint64_t_rintmode"};

  std::string castNameWithMode = getCastName(true);
  // check whether supports the cast operation.
  if (!HWSupportedCast.count(castNameWithMode) &&
      !softSupportedCast.count(castNameWithMode)) {
    return emitOpError() << "currently don't support cast " << castNameWithMode;
  }

  return success();
}

std::string VCastOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  MemRefType srcMemref = cast<MemRefType>(getSingleSrc().getType());
  int rank = srcMemref.getRank();
  auto baseCallName = getOpName().str();
  bool tempBufferCond = srcMemref.getElementType().isInteger(1);
  std::stringstream ss;
  ss << baseCallName << "_" << getCastName() << "_"
     << getOpLibraryCallRank(rank) << "d";

  auto srcType = getElementTypeOrSelf(this->getSrc()[0]);
  auto dstType = getElementTypeOrSelf(this->getDst()[0]);
  const bool isI32ToI8 = srcType.isInteger(32) && dstType.isInteger(8);
  const bool isI16ToI8 = srcType.isInteger(16) && dstType.isInteger(8);
  const bool isI32ToI16 = srcType.isInteger(32) && dstType.isInteger(16);
  const bool isI64ToI32 = srcType.isInteger(64) && dstType.isInteger(32);
  if ((isI32ToI8 || isI16ToI8 || isI32ToI16 || isI64ToI32) &&
      this->getRoundMode() == hivm::RoundMode::TRUNCWITHOVERFLOW) {
    ss << "_with_overflow";
  } else {
    ss << (tempBufferCond ? "_with_temp" : "_with_mode");
  }
  return ss.str();
}

//===----------------------------------------------------------------------===//
// VReduceOp
//===----------------------------------------------------------------------===//

void VReduceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value src, ValueRange dst,
                      hivm::ReduceOpAttr arith, BoolAttr unsignedSrc,
                      DenseI64ArrayAttr reduce_dims) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr, arith,
        unsignedSrc, /*tie_break_left*/ nullptr, reduce_dims);
}

void VReduceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value src, ValueRange dst,
                      hivm::ReduceOpAttr arith, BoolAttr unsignedSrc,
                      BoolAttr tieBreakLeft, DenseI64ArrayAttr reduce_dims) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr, arith,
        unsignedSrc, tieBreakLeft, reduce_dims);
}

LogicalResult VReduceOp::verify() {
  // tmpBuf can be null
  auto tmpBuf = getTempBuffer();
  if (tmpBuf && tmpBuf.getType().getShape().size() != 1) {
    return emitOpError() << "temp_buffer'rank should be one";
  }

  ArrayRef<int64_t> reduceDims = this->getReduceDims();
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  ShapedType dstVecType = cast<ShapedType>(getDstValue().getType());

  if (reduceDims.empty()) {
    return emitOpError() << "have empty reduce dims array";
  }
  if (static_cast<int64_t>(reduceDims.size()) > srcVecType.getRank()) {
    return emitOpError() << "have too many indices in the reduce dims array";
  }

  for (int64_t idx : reduceDims) {
    if (idx < 0 || idx >= dstVecType.getRank()) {
      return emitOpError() << "have invalid index '" << idx
                           << "' inside reduce dims array";
    }
    if (dstVecType.getDimSize(idx) != 1) {
      return emitOpError() << "invalid dst vector shape, 'DstVecDim[" << idx
                           << "]' != 1\n";
    }
  }
  auto arith = getArithAttr();
  if (utils::isReduceWithIndex(arith.getReduceOp())) {
    if (!getDstIndex()) {
      return emitOpError() << "dst index must be defined for min_with_index "
                              "and max_with_index";
    }
    if (!getElementTypeOrSelf(getDstIndex().getType()).isInteger(32)) {
      return emitOpError() << "invalid dst index elemtype";
    }
  } else if (arith.getReduceOp() == hivm::ReduceOperation::xori) {
    if (!getElementTypeOrSelf(srcVecType).isInteger()) {
      return emitOpError() << "invalid elemtype for xori";
    }
  }
  return success();
}

static inline Attribute
getIntegerInitAttr(ReduceOperation reduceKind, IntegerType intType) {
  llvm::APInt initVal;
  unsigned bitWidth = intType.getIntOrFloatBitWidth();
  switch (reduceKind) {
  case ReduceOperation::sum:
  case ReduceOperation::xori:
  case ReduceOperation::ori:
  case ReduceOperation::any:
    initVal = llvm::APInt::getZero(bitWidth);
    break;
  case ReduceOperation::andi:
    initVal = llvm::APInt(bitWidth, -1);
    break;
  case ReduceOperation::min:
  case ReduceOperation::min_with_index:
    initVal = intType.isUnsignedInteger()
                  ? llvm::APInt::getMaxValue(bitWidth)
                  : llvm::APInt::getSignedMaxValue(bitWidth);
    break;
  case ReduceOperation::max:
  case ReduceOperation::max_with_index:
    initVal = intType.isUnsignedInteger()
                  ? llvm::APInt::getMinValue(bitWidth)
                  : llvm::APInt::getSignedMinValue(bitWidth);
    break;
  case ReduceOperation::prod:
    initVal = llvm::APInt(bitWidth, 1);
    break;
  default:
    llvm_unreachable("Unsupported reduce kind.");
    return {};
  };
  IntegerType signlessType = IntegerType::get(intType.getContext(), bitWidth, IntegerType::Signless);
  return IntegerAttr::get(signlessType, initVal);
}

static inline Attribute
getFloatInitAttr(ReduceOperation reduceKind, FloatType floatType) {
  const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
  llvm::APFloat initVal(semantics);
  switch (reduceKind) {
  case ReduceOperation::sum:
    initVal = llvm::APFloat::getZero(semantics);
    break;
  case ReduceOperation::min:
  case ReduceOperation::min_with_index:
    initVal = llvm::APFloat::getInf(semantics);
    break;
  case ReduceOperation::max:
  case ReduceOperation::max_with_index:
    initVal = llvm::APFloat::getInf(semantics, true);
    break;
  case ReduceOperation::prod:
    initVal = llvm::APFloat(semantics, 1);
    break;
  default:
    llvm_unreachable("Unsupported reduce kind.");
    return {};
  };
  return FloatAttr::get(floatType, initVal);
}

Attribute VReduceOp::getInit() {
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  Type eleType = srcVecType.getElementType();
  ReduceOperation arith = getArithAttr().getReduceOp();
  if (eleType.isInteger()) {
    if (this->getUnsignedSrc()) {
      auto intType = mlir::dyn_cast<IntegerType>(eleType);
      Type unsignedEleType = IntegerType::get(intType.getContext(), intType.getWidth(), IntegerType::Unsigned);
      return getIntegerInitAttr(arith, cast<IntegerType>(unsignedEleType));
    } else {
      return getIntegerInitAttr(arith, cast<IntegerType>(eleType));
    }
  } else if (isa<FloatType>(eleType)) {
    return getFloatInitAttr(arith, cast<FloatType>(eleType));
  }
  llvm_unreachable("Unsupported element data type.");
  return {};
}

int VReduceOp::inferOpLibraryMaxRank() {
  llvm::ArrayRef<int64_t> reduceDims = this->getReduceDims();
  assert(!reduceDims.empty() && "reduce dimensions array must not be empty.");
  assert(reduceDims.size() == 1 &&
         "reduce dimensions array is not decomposed yet.");
  int reduceIdx = reduceDims[0];
  MemRefType srcVecType = cast<MemRefType>(this->getSrc().getType());
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

bool VReduceOp::useVectorCrossIntr(bool lastAxis, int rank) {
  // only half and float datatype support VC Intrin
  auto eleType = getElementTypeOrSelf(this->getSrc().getType());
  if (!eleType.isF16() && !eleType.isF32()) {
    return false;
  }
  // For any type of sum/min/max, enable VC Intrin
  hivm::ReduceOperation arithOp = this->getArith().getReduceOp();
  if (arithOp != hivm::ReduceOperation::sum &&
      arithOp != hivm::ReduceOperation::min &&
      arithOp != hivm::ReduceOperation::max) {
    return false;
  }
  // only last-axis min max sum op with fp16 or fp32 type use vector cross
  // intrinsic
  return (lastAxis || rank == 1);
}

std::string VReduceOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  StringRef baseName = this->getOpName();
  MemRefType srcVecType = cast<MemRefType>(this->getSrc().getType());
  int rank = srcVecType.getRank();
  llvm::ArrayRef<int64_t> reduceDims = this->getReduceDims();
  assert(reduceDims.size() == 1 &&
         "reduce dimensions array is not decomposed yet");

  bool firstAxis = reduceDims[0] == 0;
  bool lastAxis = reduceDims[0] == rank - 1;
  bool midAxis = !firstAxis && !lastAxis;
  auto reduceOpName = stringifyReduceOperation(this->getArith().getReduceOp());
  std::stringstream ss;
  ss << (useVectorCrossIntr(lastAxis, rank) ? "enablevc_" : "");
  ss << baseName.data() << "_" << reduceOpName.str();
  if (reduceOpName == "min_with_index" || reduceOpName == "max_with_index") {
    std::optional<bool> maybeIsTieBreakLeft = this->getTieBreakLeft();
    assert(maybeIsTieBreakLeft.has_value());
    bool isTieBreakLeft = maybeIsTieBreakLeft.value();
    ss << "_" << (isTieBreakLeft ? "left" : "right");
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
  ss << hivm::detail::getTypeName(this->getLoc(), eleType);
  return ss.str();
}

//===----------------------------------------------------------------------===//
// VSortOp
//===----------------------------------------------------------------------===//

void VSortOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    TypeRange result, Value src, ValueRange dst,
                    bool descending, int64_t sort_axis) {
  build(odsBuilder, odsState, result, src, dst,
        /*temp_buffer=*/nullptr, descending, sort_axis);
}

std::string VSortOp::getOpLibraryCallName(
    [[maybe_unused]] std::optional<bool> isOpsAligned) {
  StringRef baseName = this->getOpName();
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  Type elemType = srcVecType.getElementType();

  bool needSortIndex = getDst().size() == 2;
  std::stringstream ss;
  ss << baseName.data();
  if (needSortIndex) {
    ss << "_with_index";
  }
  if (srcVecType.getRank() == 1)
    ss << "_1d_" << hivm::detail::getTypeName(this->getLoc(), elemType);
  else if (srcVecType.getRank() == 2)
    ss << "_2d_" << hivm::detail::getTypeName(this->getLoc(), elemType);
  return ss.str();
}

Value VSortOp::getDstValue() { return getDst()[0]; }

Value VSortOp::getDstIndex() {
  assert(getDst().size() == 2 && "there should be 2 operands");
  return getDst()[1];
}

int64_t VSortOp::getSignedSortAxis() {
  return getSortAxisAttr().getValue().getSExtValue();
}

LogicalResult VSortOp::verify() {
  // tmpBuf can be null
  auto tmpBuf = getTempBuffer();
  if (tmpBuf && tmpBuf.getType().getShape().size() != 1) {
    return emitOpError() << "temp_buffer'rank should be one";
  }

  int64_t sortAxis = this->getSignedSortAxis();
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  if (sortAxis != srcVecType.getRank() - 1 && sortAxis != -1) {
    return emitOpError() << "Currently only tail axis sorting is supported";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VTransposeOp
//===----------------------------------------------------------------------===//

LogicalResult VTransposeOp::verify() {
  ArrayRef<int64_t> permutation = this->getPermutation();
  size_t permSize = permutation.size();
  if (permutation.empty()) {
    return emitOpError() << "Permutation array should not be empty.";
  }

  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  if (static_cast<int64_t>(permSize) != srcVecType.getRank()) {
    return emitOpError() << "Permutation size should be equal to src rank";
  }

  int tranposeAxisNum = 0;
  for (int64_t idx : permutation) {
    if (idx < 0 || idx >= srcVecType.getRank()) {
      return emitOpError() << "have invalid index '" << idx
                           << "' inside permutation array";
    }
    if (idx != permutation[idx]) {
      tranposeAxisNum++;
    }
  }
  const int supportedTransposeAxisNum = 2;
  if (tranposeAxisNum != supportedTransposeAxisNum) {
    int rank = srcVecType.getRank();
    int swaps = 0;
    int supportedSwapNum = 2;
    if (rank == 4) {
      llvm::SmallVector<bool, 8> vis(permSize, false);
      for (size_t i = 0; i < permSize; ++i) {
        if (vis[i])
          continue;
        size_t j = i, len = 0;
        while (!vis[j]) {
          vis[j] = 1;
          j = (size_t)permutation[j];
          ++len;
        }
        swaps += (int)len - 1;
      }
    }
    if (rank != 4 || swaps != supportedSwapNum) {
      return emitOpError()
             << "Vtranspose supports only swapping two axes; for rank-4, "
                "also allows permutations equivalent to two swaps (got moved="
             << tranposeAxisNum << ", swaps=" << swaps << ")";
    }
  }

  // Verify elem type and rank of src/dst/res
  ShapedType dstVecType = cast<ShapedType>(getDst().getType());
  if (srcVecType.getElementType() != dstVecType.getElementType()) {
    return emitOpError() << "ElementType of src and dst are not the same";
  }

  if (srcVecType.getRank() != dstVecType.getRank()) {
    return emitOpError() << "Rank of src and dst are not the same";
  }

  if (hasPureTensorSemantics()) {
    auto res = getResult()[0];
    auto resShapedType = cast<ShapedType>(res.getType());
    if (resShapedType.getElementType() != srcVecType.getElementType()) {
      return emitOpError() << "ElementType of src and res are not the same";
    }
    if (resShapedType.getRank() != srcVecType.getRank()) {
      return emitOpError() << "Rank of src and res are not the same";
    }
  }

  return success();
}

LogicalResult
VTransposeOp::setIteratorTypesArray(const IteratorType iteratorType,
                                    const DenseI64ArrayAttr &arrayAttr) {
  assert(iteratorType == hivm::IteratorType::kTranspose);
  getOperation()->setAttr(stringifyIteratorType(iteratorType), arrayAttr);
  return success();
}

int VTransposeOp::inferOpLibraryMaxRank() {
  const int maxRank = 3;
  ArrayRef<int64_t> permutation = this->getPermutation();
  SmallVector<int64_t> transposeAxes =
      hivm::util::getTransposeAxes(permutation);
  return hivm::util::isTransposeWithLastAxis(permutation)
             ? (hivm::util::isTransposeAdjacentAxes(transposeAxes) ? maxRank - 1
                                                                   : maxRank)
             : maxRank;
}

std::string
VTransposeOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  ArrayRef<int64_t> permutation = this->getPermutation();
  const bool isTransposeWithLastAxis =
      hivm::util::isTransposeWithLastAxis(permutation);

  // Currently support three kinds of libs.
  // - transpose 2d lib for last axis transpose, (x, y) to (y, x);
  // - transpose 3d lib for last axis transpose, (x, y, z) to (z, y, x).
  // - transpose 3d lib for non-last axis transpose, (x, y, z) to (y, x, z).
  int dim = inferOpLibraryMaxRank();
  std::string desc =
      isTransposeWithLastAxis ? "with_last_axis" : "without_last_axis";

  StringRef baseName = this->getOpName();
  MemRefType srcMemrefType = cast<MemRefType>(this->getSrc().getType());
  auto elemTypeName =
      hivm::detail::getTypeName(this->getLoc(), srcMemrefType.getElementType());

  std::stringstream ss;
  ss << baseName.data() << "_" << getOpLibraryCallRank(dim) << "d"
     << "_" << desc << "_" << elemTypeName;
  return ss.str();
}

void VTransposeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                         TypeRange result, Value src, Value dst,
                         DenseI64ArrayAttr permutation) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,
        permutation);
}

//===----------------------------------------------------------------------===//
// VArangeOp
//===----------------------------------------------------------------------===//

void VArangeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value dst) {
  SmallVector<Value, 3> strides;
  Value offset = Value();
  VArangeOp::getOffsetFromValue(odsBuilder, odsState.location, offset);
  VArangeOp::getStridesFromValue(odsBuilder, odsState.location, dst, strides);
  build(odsBuilder, odsState, result, dst, offset, strides);
}

void VArangeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value dst, Value offset) {
  SmallVector<Value, 3> strides;
  VArangeOp::getOffsetFromValue(odsBuilder, odsState.location, offset);
  VArangeOp::getStridesFromValue(odsBuilder, odsState.location, dst, strides);
  build(odsBuilder, odsState, result, dst, offset, strides);
}

LogicalResult VArangeOp::verify() {
  // stride should not be empty
  if (this->getStrides().empty())
    return emitOpError() << "stride array should not be empty";

  // number of stide should match the ranke of the dst
  ShapedType dstVecType = cast<ShapedType>(getDst().getType());
  if (dstVecType.getRank() != static_cast<int64_t>(this->getStrides().size()))
    return emitOpError() << "stride array size should match the rank of dst";

  return success();
}

void VArangeOp::getOffsetFromValue(OpBuilder &builder, Location loc,
                                   Value &offset) {
  offset = offset == nullptr
               ? builder.createOrFold<arith::ConstantIndexOp>(loc, 0)
               : offset;
}

void VArangeOp::getStridesFromValue(OpBuilder &builder, Location loc, Value val,
                                    SmallVectorImpl<Value> &strides) {
  auto shapedTy = cast<ShapedType>(val.getType());
  Value constOne = builder.createOrFold<arith::ConstantIndexOp>(loc, 1);
  int rank = shapedTy.getRank();
  // Number of strides equal to number of ranks, fill with one's
  strides.append(rank, constOne);
  // Reverse iterater to fill rank from back to forward
  for (int dim = rank - 1; dim > 0; --dim) {
    Value size;
    if (isa<MemRefType>(shapedTy))
      size = builder.createOrFold<memref::DimOp>(loc, val, dim);
    else if (isa<TensorType>(shapedTy))
      size = builder.createOrFold<tensor::DimOp>(loc, val, dim);
    else
      llvm_unreachable(
          "Expected arange to be initialized with tensor or memref type.");
    strides[dim - 1] =
        builder.createOrFold<arith::MulIOp>(loc, strides[dim], size);
  }
}

std::string VArangeOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  std::string baseCallName = this->getOpName().str();
  Type elemType = getElementTypeOrSelf(this->getDpsInits().front().getType());
  std::string elemTypeName =
      hivm::detail::getTypeName(this->getLoc(), elemType);
  int rank = static_cast<int>(getNumLoops());
  return concatVectorOpLibraryCallName(baseCallName, getOpLibraryCallRank(rank),
                                       elemTypeName);
}

//===----------------------------------------------------------------------===//
// VInterleaveOp
//===----------------------------------------------------------------------===//

void VInterleaveOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TypeRange result, ValueRange src, Value dst) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr);
}

void VInterleaveOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TypeRange result, ValueRange src, Value dst,
                          int64_t interleave_channel_nums) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,
        interleave_channel_nums);
}

LogicalResult VInterleaveOp::verify() {
  auto inputs = getSrc();
  const int supportedTensorSize = 2;
  if (inputs.size() != supportedTensorSize ||
      inputs.size() != getInterleaveChannelNums()) {
    return emitOpError() << "Only support interleave two tensor2";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VDeinterleaveOp
//===----------------------------------------------------------------------===//

std::string
VDeinterleaveOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  auto mode = getIndexMode();
  assert(mode != DeinterleaveMode::ALL_CHANNELS &&
         "There shouldn't exist double mode deinterleave library call"
         "which has been decomposed.");

  assert(mode <= DeinterleaveMode::CHANNEL_1 &&
         "deinterleave mode don't support select this channel");

  StringRef baseName = this->getOpName();
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  Type elemType = srcVecType.getElementType();

  std::string modeName = stringifyDeinterleaveMode(mode).lower();

  MemRefType srcMemRefType = cast<MemRefType>(this->getSrc().getType());
  int maxRank = inferOpLibraryMaxRank();
  int rank = srcMemRefType.getRank();
  rank = rank <= maxRank ? rank : maxRank;

  std::stringstream ss;
  const int maxDeInterLeaveChannelNum = 2;
  if (getDeInterLeaveChannelNum() > maxDeInterLeaveChannelNum) {
    assert(
        mode == DeinterleaveMode::CHANNEL_0 &&
        "deinterleave mode only support select channel0 when channel num > 2");
    ss << baseName.data() << "_" << modeName
       << "_from_"
          "n_channels_"
       << rank << "d_" << hivm::detail::getTypeName(this->getLoc(), elemType);
    return ss.str();
  }

  ss << baseName.data() << "_" << modeName << "_from_"
     << getDeInterLeaveChannelNum() << "_channels"
     << "_1d_" << hivm::detail::getTypeName(this->getLoc(), elemType);
  return ss.str();
}

LogicalResult VDeinterleaveOp::verify() {
  auto outputs = getDst();
  auto mode = getIndexMode();
  if (mode == hivm::DeinterleaveMode::ALL_CHANNELS) {
    if (ssize_t(outputs.size()) != getDeInterLeaveChannelNum()) {
      return emitOpError() << "output num mismatch with channel num";
    }
  } else {
    if (outputs.size() != 1) {
      return emitOpError()
             << "output num for CHANNEL_0 CHANNEL_1 should be one";
    }
  }

  return success();
}

int VDeinterleaveOp::inferOpLibraryMaxRank() {
  const int maxDeInterLeaveChannelNum = 2;
  if (getDeInterLeaveChannelNum() > maxDeInterLeaveChannelNum &&
      getIndexMode() == hivm::DeinterleaveMode::CHANNEL_0) {
    // select channel0 from N channels support 2d
    return 2;
  }
  // select channel from 2 channels only support 1d
  return 1;
}

//===----------------------------------------------------------------------===//
// VXor
//===----------------------------------------------------------------------===//

void VXorOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange result, ValueRange src, ValueRange dst,
                   DenseI64ArrayAttr transpose, DenseI64ArrayAttr broadcast) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,
        transpose, broadcast);
}

//===----------------------------------------------------------------------===//
// VMulExtendedOp
//===----------------------------------------------------------------------===//

void VMulextendedOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           TypeRange result, ValueRange src, ValueRange dst) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr);
}

std::string
VMulextendedOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  StringRef baseName = this->getOpName();
  ShapedType srcVecType = cast<ShapedType>(getSrc()[0].getType());
  Type elemType = srcVecType.getElementType();

  std::stringstream ss;
  ss << baseName.data() << "_1d_"
     << hivm::detail::getTypeName(this->getLoc(), elemType);
  return ss.str();
}

//===----------------------------------------------------------------------===//
// VPowOp
//===----------------------------------------------------------------------===//

void VPowOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TypeRange result, ValueRange src, ValueRange dst) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr);
}

//===----------------------------------------------------------------------===//
// VPadOp
//===----------------------------------------------------------------------===//

// Return a vector of all the static or dynamic values (low/high padding) of
// the op.
SmallVector<OpFoldResult> VPadOp::getMixedPadImpl(ArrayRef<int64_t> staticAttrs,
                                                  ValueRange values) {
  Builder builder(*this);
  SmallVector<OpFoldResult> res;
  unsigned numDynamic = 0;
  unsigned count = staticAttrs.size();
  for (unsigned idx = 0; idx < count; ++idx) {
    if (ShapedType::isDynamic(staticAttrs[idx]))
      res.push_back(values[numDynamic++]);
    else
      res.push_back(builder.getI64IntegerAttr(staticAttrs[idx]));
  }
  return res;
}

SmallVector<OpFoldResult> VPadOp::getMixedLowPad() {
  return getMixedPadImpl(getStaticLow(), getLow());
}

SmallVector<OpFoldResult> VPadOp::getMixedHighPad() {
  return getMixedPadImpl(getStaticHigh(), getHigh());
}

//===----------------------------------------------------------------------===//
// VGatherOp
//===----------------------------------------------------------------------===//

void VGatherOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value src, Value indices, Value dst) {
  build(odsBuilder, odsState, result, src, indices, dst,
        /*temp_buffer=*/nullptr);
}

std::string VGatherOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  StringRef baseName = this->getOpName();
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  Type elemType = srcVecType.getElementType();

  std::stringstream ss;
  ss << baseName.data() << "_1d_"
     << hivm::detail::getTypeName(this->getLoc(), elemType);
  return ss.str();
}

//===----------------------------------------------------------------------===//
// VCumprodOp
//===----------------------------------------------------------------------===//

std::string VCumprodOp::getOpLibraryCallName(std::optional<bool> isOpsAligned) {
  return getCumOpLibraryCallName(*this);
}

LogicalResult VCumprodOp::verify() { return verifyCumOp(*this); }

//===----------------------------------------------------------------------===//
// VCumsumOp
//===----------------------------------------------------------------------===//

std::string VCumsumOp::getOpLibraryCallName(
    [[maybe_unused]] std::optional<bool> isOpsAligned) {
  return getCumOpLibraryCallName(*this);
}

LogicalResult VCumsumOp::verify() { return verifyCumOp(*this); }

//===----------------------------------------------------------------------===//
// VDivOp
//===----------------------------------------------------------------------===//

std::string VDivOp::getOpLibraryCallName(
    [[maybe_unused]] std::optional<bool> isOpsAligned) {
  std::string baseCallName = getOpName().str();
  Type elemType = getElementTypeOrSelf(getDpsInputs().back().getType());
  hivm::TypeFn cast = this->getIsSigned() ? hivm::TypeFn::cast_signed
                                          : hivm::TypeFn::cast_unsigned;
  std::string elemTypeName =
      hivm::detail::getTypeName(getLoc(), elemType, cast);
  int rank = static_cast<int>(getNumLoops());
  std::string selectTypeName = "_vv";
  bool src0ScalarType = getSrc()[0].getType().isIntOrFloat();
  bool src1ScalarType = getSrc()[1].getType().isIntOrFloat();
  if (!src0ScalarType && src1ScalarType) {
    selectTypeName = "_vs";
  }
  if (src0ScalarType && !src1ScalarType) {
    selectTypeName = "_sv";
  }
  if (selectTypeName == "_vv") {
    return concatVectorOpLibraryCallName(
        baseCallName, getOpLibraryCallRank(rank), elemTypeName);
  }
  return concatVectorOpLibraryCallName(
      baseCallName + selectTypeName, getOpLibraryCallRank(rank), elemTypeName);
}