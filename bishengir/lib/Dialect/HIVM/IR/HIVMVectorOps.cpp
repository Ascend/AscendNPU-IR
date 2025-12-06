//===- HIVMVector.cpp - HIVM Vector ops implementation --------------------===//
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
#include "bishengir/Dialect/HIVM/Transforms/AlignBuffer/Util.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::hivm;

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.cpp.inc"

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
ENABLE_VECTOR_BINARY_AND_UNARY_OP_BUILD_WITH_TMPBUFF(VDivOp)
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

std::string VCastOp::getCastName(bool withMode) {
  std::string castName = "";
  ShapedType srcVcastType = cast<ShapedType>(getSingleSrc().getType());
  ShapedType dstVcastType = cast<ShapedType>(getSingleDst().getType());
  auto srcElemType = srcVcastType.getElementType();
  auto dstElemType = dstVcastType.getElementType();
  hivm::TypeFn casting = this->getCast();
  castName.append(
      util::getTypeName(this->getLoc(), srcElemType, casting));
  castName.append("_to_");
  castName.append(
      util::getTypeName(this->getLoc(), dstElemType, casting));
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
      "int64_t_to_int32_t_truncwithoverflowmode"};

  std::string castNameWithMode = getCastName(true);
  // check whether supports the cast operation.
  if (!HWSupportedCast.count(castNameWithMode) &&
      !softSupportedCast.count(castNameWithMode)) {
    return emitOpError() << "currently don't support cast " << castNameWithMode;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VReduceOp
//===----------------------------------------------------------------------===//

void VReduceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value src, ValueRange dst,
                      hivm::ReduceOpAttr arith, DenseI64ArrayAttr reduce_dims) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr, arith,
        reduce_dims, /*indices=*/nullptr);
}

void VReduceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange result, Value src, ValueRange dst,
                      hivm::ReduceOpAttr arith, DenseI64ArrayAttr reduce_dims,
                      Value indices) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr, arith,
        reduce_dims, indices);
}

static LogicalResult verifyVReduceDims(VReduceOp op) {
  SmallVector<int64_t> reduceDims(op.getReduceDims());
  const auto srcVecType = cast<ShapedType>(op.getSrc().getType());
  const auto dstVecType = cast<ShapedType>(op.getDstValue().getType());

  if (reduceDims.empty()) {
    return op.emitOpError() << "have empty reduce dims array";
  }
  if (static_cast<int64_t>(reduceDims.size()) > srcVecType.getRank()) {
    return op.emitOpError() << "have too many indices in the reduce dims array";
  }

  for (int64_t idx : reduceDims) {
    if (idx < 0 || idx >= dstVecType.getRank()) {
      return op.emitOpError()
             << "have invalid index '" << idx << "' inside reduce dims array";
    }
    if (dstVecType.getDimSize(idx) != 1) {
      return op.emitOpError()
             << "invalid dst vector shape, 'DstVecDim[" << idx << "]' != 1\n";
    }
  }

  if (srcVecType.getRank() != dstVecType.getRank()) {
    return op.emitOpError() << "have src and dst with different ranks";
  }
  llvm::sort(reduceDims);
  auto constReduceDims = ArrayRef(reduceDims);
  for (auto reduceDim = constReduceDims.begin(),
            srcDim = srcVecType.getShape().begin(),
            dstDim = dstVecType.getShape().begin();
       srcDim != srcVecType.getShape().end(); ++srcDim, ++dstDim) {
    const auto idx = srcDim - srcVecType.getShape().begin();

    if (reduceDim != constReduceDims.end() && idx == *reduceDim) {
      ++reduceDim;
      continue;
    }

    if (failed(verifyCompatibleDims({*srcDim, *dstDim}))) {
      return op.emitOpError()
             << "have dim at index " << idx << " different between src and dst";
    }
  }

  return success();
}

static LogicalResult verifyVReduceArith(VReduceOp op) {
  const auto srcVecType = cast<ShapedType>(op.getSrc().getType());
  const auto dstVecType = cast<ShapedType>(op.getDstValue().getType());
  auto arith = op.getArithAttr();
  if (VReduceOp::isArgminOrArgmax(arith.getReduceOp())) {
    if (op.getDst().size() != 2)
      return op.emitOpError()
             << "with index should have exactly 2 destination operands";

    if (op.getReduceDims().size() > 1)
      return op.emitOpError()
             << "with index should have exactly 1 reduce dimension";

    if (!op.getDstIndex()) {
      return op.emitOpError() << "dst index must be defined for min_with_index "
                                 "and max_with_index";
    }
    if (failed(verifyCompatibleShape(
            cast<ShapedType>(op.getDstIndex().getType()).getShape(),
            dstVecType.getShape()))) {
      return op.emitOpError()
             << "dst index shape should be compatible with dst value";
    }
    if (!getElementTypeOrSelf(op.getDstIndex().getType()).isInteger(32)) {
      return op.emitOpError() << "invalid dst index elemtype";
    }
  } else {
    if (op.getDst().size() != 1)
      return op.emitOpError() << "should have exactly 1 destination operand";

    if (llvm::is_contained({ReduceOperation::andi, ReduceOperation::ori,
                            ReduceOperation::xori},
                           arith.getReduceOp()) &&
        !getElementTypeOrSelf(srcVecType).isInteger())
      return op.emitOpError() << "elemtype should be an integer";
  }
  return success();
}

LogicalResult VReduceOp::verify() {
  // tmpBuf can be null
  auto tmpBuf = getTempBuffer();
  if (tmpBuf && tmpBuf.getType().getShape().size() != 1) {
    return emitOpError() << "temp_buffer'rank should be one";
  }

  if (failed(verifyVReduceDims(*this)))
    return failure();

  return verifyVReduceArith(*this);
}

Attribute VReduceOp::getInit() {
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  mlir::Type eleType = srcVecType.getElementType();
  mlir::hivm::ReduceOperation arith = getArithAttr().getReduceOp();

  mlir::Type f16Ty = Float16Type::get(getContext());
  mlir::Type f32Ty = Float32Type::get(getContext());
  mlir::Type i8TySL = IntegerType::get(
      getContext(), 8, IntegerType::SignednessSemantics::Signless); // signless
  mlir::Type i8TyS = IntegerType::get(
      getContext(), 8, IntegerType::SignednessSemantics::Signed); // signed
  mlir::Type i8TyU = IntegerType::get(
      getContext(), 8, IntegerType::SignednessSemantics::Unsigned); // unsigned
  mlir::Type i16TySL = IntegerType::get(
      getContext(), 16, IntegerType::SignednessSemantics::Signless); // signless
  mlir::Type i16TyS = IntegerType::get(
      getContext(), 16, IntegerType::SignednessSemantics::Signed); // signed
  mlir::Type i16TyU = IntegerType::get(
      getContext(), 16, IntegerType::SignednessSemantics::Unsigned); // unsigned
  mlir::Type i32TySL = IntegerType::get(
      getContext(), 32, IntegerType::SignednessSemantics::Signless); // signless
  mlir::Type i32TyS = IntegerType::get(
      getContext(), 32, IntegerType::SignednessSemantics::Signed); // signed
  mlir::Type i32TyU = IntegerType::get(
      getContext(), 32, IntegerType::SignednessSemantics::Unsigned); // unsigned
  mlir::Type i64TySL = IntegerType::get(
      getContext(), 64, IntegerType::SignednessSemantics::Signless); // signless
  mlir::Type i64TyS = IntegerType::get(
      getContext(), 64, IntegerType::SignednessSemantics::Signed); // signed
  mlir::Type i64TyU = IntegerType::get(
      getContext(), 64, IntegerType::SignednessSemantics::Unsigned); // unsigned

  llvm::APFloat halfZero = llvm::APFloat::getZero(llvm::APFloat::IEEEhalf());
  llvm::APFloat halfOne(llvm::APFloat::IEEEhalf(), 1);
  llvm::APFloat halfMax = llvm::APFloat::getInf(llvm::APFloat::IEEEhalf());
  llvm::APFloat halfMin =
      llvm::APFloat::getInf(llvm::APFloat::IEEEhalf(), true);

  llvm::APFloat floatZero = llvm::APFloat::getZero(llvm::APFloat::IEEEsingle());
  llvm::APFloat floatOne(llvm::APFloat::IEEEsingle(), 1);
  llvm::APFloat floatMax = llvm::APFloat::getInf(llvm::APFloat::IEEEsingle());
  llvm::APFloat floatMin =
      llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), true);

  auto toPtr = [](mlir::Type ty) { return ty.getAsOpaquePointer(); };

  // a mapping from {arithmatic operator, element type} pair to the initial
  // value
  const std::map<std::pair<mlir::hivm::ReduceOperation, const void *>,
                 std::variant<int8_t, int16_t, int32_t, int64_t, llvm::APFloat>>
      initMap = {
          {{hivm::ReduceOperation::sum, toPtr(f16Ty)}, halfZero},
          {{hivm::ReduceOperation::sum, toPtr(f32Ty)}, floatZero},
          {{hivm::ReduceOperation::sum, toPtr(i16TySL)}, (int16_t)0},
          {{hivm::ReduceOperation::sum, toPtr(i16TyS)}, (int16_t)0},
          {{hivm::ReduceOperation::sum, toPtr(i16TyU)}, (int16_t)0},
          {{hivm::ReduceOperation::sum, toPtr(i32TySL)}, 0},
          {{hivm::ReduceOperation::sum, toPtr(i32TyS)}, 0},
          {{hivm::ReduceOperation::sum, toPtr(i32TyU)}, 0},
          {{hivm::ReduceOperation::sum, toPtr(i64TySL)}, (int64_t)0},
          {{hivm::ReduceOperation::sum, toPtr(i64TyS)}, (int64_t)0},
          {{hivm::ReduceOperation::sum, toPtr(i64TyU)}, (int64_t)0},

          {{hivm::ReduceOperation::min, toPtr(f16Ty)}, halfMax},
          {{hivm::ReduceOperation::min, toPtr(f32Ty)}, floatMax},
          {{hivm::ReduceOperation::min, toPtr(i16TySL)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min, toPtr(i16TyS)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min, toPtr(i16TyU)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min, toPtr(i32TySL)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min, toPtr(i32TyS)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min, toPtr(i32TyU)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min, toPtr(i64TySL)},
           std::numeric_limits<int64_t>::max()},
          {{hivm::ReduceOperation::min, toPtr(i64TyS)},
           std::numeric_limits<int64_t>::max()},
          {{hivm::ReduceOperation::min, toPtr(i64TyU)},
           std::numeric_limits<int64_t>::max()},

          {{hivm::ReduceOperation::max, toPtr(f16Ty)}, halfMin},
          {{hivm::ReduceOperation::max, toPtr(f32Ty)}, floatMin},
          {{hivm::ReduceOperation::max, toPtr(i16TySL)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max, toPtr(i16TyS)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max, toPtr(i16TyU)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max, toPtr(i32TySL)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max, toPtr(i32TyS)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max, toPtr(i32TyU)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max, toPtr(i64TySL)},
           std::numeric_limits<int64_t>::min()},
          {{hivm::ReduceOperation::max, toPtr(i64TyS)},
           std::numeric_limits<int64_t>::min()},
          {{hivm::ReduceOperation::max, toPtr(i64TyU)},
           std::numeric_limits<int64_t>::min()},

          {{hivm::ReduceOperation::prod, toPtr(f16Ty)}, halfOne},
          {{hivm::ReduceOperation::prod, toPtr(f32Ty)}, floatOne},
          {{hivm::ReduceOperation::prod, toPtr(i16TySL)}, (int16_t)1},
          {{hivm::ReduceOperation::prod, toPtr(i16TyS)}, (int16_t)1},
          {{hivm::ReduceOperation::prod, toPtr(i16TyU)}, (int16_t)1},
          {{hivm::ReduceOperation::prod, toPtr(i32TySL)}, 1},
          {{hivm::ReduceOperation::prod, toPtr(i32TyS)}, 1},
          {{hivm::ReduceOperation::prod, toPtr(i32TyU)}, 1},
          {{hivm::ReduceOperation::prod, toPtr(i64TySL)}, (int64_t)1},
          {{hivm::ReduceOperation::prod, toPtr(i64TyS)}, (int64_t)1},
          {{hivm::ReduceOperation::prod, toPtr(i64TyU)}, (int64_t)1},

          {{hivm::ReduceOperation::xori, toPtr(i8TySL)}, (int8_t)0},
          {{hivm::ReduceOperation::xori, toPtr(i8TyS)}, (int8_t)0},
          {{hivm::ReduceOperation::xori, toPtr(i8TyU)}, (int8_t)0},
          {{hivm::ReduceOperation::xori, toPtr(i16TySL)}, (int16_t)0},
          {{hivm::ReduceOperation::xori, toPtr(i16TyS)}, (int16_t)0},
          {{hivm::ReduceOperation::xori, toPtr(i16TyU)}, (int16_t)0},
          {{hivm::ReduceOperation::xori, toPtr(i32TySL)}, 0},
          {{hivm::ReduceOperation::xori, toPtr(i32TyS)}, 0},
          {{hivm::ReduceOperation::xori, toPtr(i32TyU)}, 0},
          {{hivm::ReduceOperation::xori, toPtr(i64TySL)}, (int64_t)0},
          {{hivm::ReduceOperation::xori, toPtr(i64TyS)}, (int64_t)0},
          {{hivm::ReduceOperation::xori, toPtr(i64TyU)}, (int64_t)0},

          {{hivm::ReduceOperation::ori, toPtr(i8TySL)}, (int8_t)0},
          {{hivm::ReduceOperation::ori, toPtr(i8TyS)}, (int8_t)0},
          {{hivm::ReduceOperation::ori, toPtr(i8TyU)}, (int8_t)0},
          {{hivm::ReduceOperation::ori, toPtr(i16TySL)}, (int16_t)0},
          {{hivm::ReduceOperation::ori, toPtr(i16TyS)}, (int16_t)0},
          {{hivm::ReduceOperation::ori, toPtr(i16TyU)}, (int16_t)0},
          {{hivm::ReduceOperation::ori, toPtr(i32TySL)}, 0},
          {{hivm::ReduceOperation::ori, toPtr(i32TyS)}, 0},
          {{hivm::ReduceOperation::ori, toPtr(i32TyU)}, 0},
          {{hivm::ReduceOperation::ori, toPtr(i64TySL)}, (int64_t)0},
          {{hivm::ReduceOperation::ori, toPtr(i64TyS)}, (int64_t)0},
          {{hivm::ReduceOperation::ori, toPtr(i64TyU)}, (int64_t)0},

          {{hivm::ReduceOperation::andi, toPtr(i8TySL)}, (int8_t)-1},
          {{hivm::ReduceOperation::andi, toPtr(i8TyS)}, (int8_t)-1},
          {{hivm::ReduceOperation::andi, toPtr(i8TyU)}, (int8_t)-1},
          {{hivm::ReduceOperation::andi, toPtr(i16TySL)}, (int16_t)-1},
          {{hivm::ReduceOperation::andi, toPtr(i16TyS)}, (int16_t)-1},
          {{hivm::ReduceOperation::andi, toPtr(i16TyU)}, (int16_t)-1},
          {{hivm::ReduceOperation::andi, toPtr(i32TySL)}, -1},
          {{hivm::ReduceOperation::andi, toPtr(i32TyS)}, -1},
          {{hivm::ReduceOperation::andi, toPtr(i32TyU)}, -1},
          {{hivm::ReduceOperation::andi, toPtr(i64TySL)}, (int64_t)-1},
          {{hivm::ReduceOperation::andi, toPtr(i64TyS)}, (int64_t)-1},
          {{hivm::ReduceOperation::andi, toPtr(i64TyU)}, (int64_t)-1},

          {{hivm::ReduceOperation::min_with_index_left, toPtr(f16Ty)}, halfMax},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(f32Ty)},
           floatMax},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i16TySL)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i16TyS)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i16TyU)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i32TySL)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i32TyS)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i32TyU)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i64TySL)},
           std::numeric_limits<int64_t>::max()},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i64TyS)},
           std::numeric_limits<int64_t>::max()},
          {{hivm::ReduceOperation::min_with_index_left, toPtr(i64TyU)},
           std::numeric_limits<int64_t>::max()},

          {{hivm::ReduceOperation::min_with_index_right, toPtr(f16Ty)},
           halfMax},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(f32Ty)},
           floatMax},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i16TySL)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i16TyS)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i16TyU)},
           std::numeric_limits<int16_t>::max()},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i32TySL)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i32TyS)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i32TyU)},
           std::numeric_limits<int32_t>::max()},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i64TySL)},
           std::numeric_limits<int64_t>::max()},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i64TyS)},
           std::numeric_limits<int64_t>::max()},
          {{hivm::ReduceOperation::min_with_index_right, toPtr(i64TyU)},
           std::numeric_limits<int64_t>::max()},

          {{hivm::ReduceOperation::max_with_index_left, toPtr(f16Ty)}, halfMin},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(f32Ty)},
           floatMin},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i16TySL)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i16TyS)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i16TyU)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i32TySL)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i32TyS)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i32TyU)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i64TySL)},
           std::numeric_limits<int64_t>::min()},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i64TyS)},
           std::numeric_limits<int64_t>::min()},
          {{hivm::ReduceOperation::max_with_index_left, toPtr(i64TyU)},
           std::numeric_limits<int64_t>::min()},

          {{hivm::ReduceOperation::max_with_index_right, toPtr(f16Ty)},
           halfMin},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(f32Ty)},
           floatMin},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i16TySL)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i16TyS)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i16TyU)},
           std::numeric_limits<int16_t>::min()},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i32TySL)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i32TyS)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i32TyU)},
           std::numeric_limits<int32_t>::min()},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i64TySL)},
           std::numeric_limits<int64_t>::min()},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i64TyS)},
           std::numeric_limits<int64_t>::min()},
          {{hivm::ReduceOperation::max_with_index_right, toPtr(i64TyU)},
           std::numeric_limits<int64_t>::min()},
      };

  Attribute ret;

  auto key = std::make_pair(arith, toPtr(eleType));
  if (initMap.find(key) != initMap.end()) {
    if (eleType.isInteger(8)) {
      ret = IntegerAttr::get(IntegerType::get(getContext(), 8),
                             std::get<int8_t>(initMap.at(key)));
    } else if (eleType.isInteger(16)) {
      ret = IntegerAttr::get(IntegerType::get(getContext(), 16),
                             std::get<int16_t>(initMap.at(key)));
    } else if (eleType.isInteger(32)) {
      ret = IntegerAttr::get(IntegerType::get(getContext(), 32),
                             std::get<int32_t>(initMap.at(key)));
    } else if (eleType.isInteger(64)) {
      ret = IntegerAttr::get(IntegerType::get(getContext(), 64),
                             std::get<int64_t>(initMap.at(key)));
    } else if (isa<Float16Type>(eleType)) {
      ret = FloatAttr::get(f16Ty, std::get<llvm::APFloat>(initMap.at(key)));
    } else if (isa<Float32Type>(eleType)) {
      ret = FloatAttr::get(f32Ty, std::get<llvm::APFloat>(initMap.at(key)));
    }
  }

  return ret;
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

//===----------------------------------------------------------------------===//
// VTransposeOp
//===----------------------------------------------------------------------===//

bool VTransposeOp::isLastDimTranspose() {
  SmallVector<int64_t> transposeLoopDims;
  getTransposeLoopDims(transposeLoopDims);
  auto dimSize = getNumLoops();
  if (std::find(transposeLoopDims.begin(), transposeLoopDims.end(),
                dimSize - 1) == transposeLoopDims.end()) {
    return false;
  }
  return true;
}

bool VTransposeOp::isLastTwoAxesTranspose() {
  if (!isLastDimTranspose()) {
    return false;
  }

  SmallVector<int64_t> transposeLoopDims;
  getTransposeLoopDims(transposeLoopDims);
  return transposeLoopDims[0] == (transposeLoopDims[1] - 1);
}

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
    return emitOpError() << "Vtranspose only support two axes transpose";
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

  if (getDisableAlign()) {
    if (!isLastTwoAxesTranspose()) {
      return emitOpError()
             << "Disabled allign mode supports only last 2 axes transpose";
    }

    if (hasPureBufferSemantics()) {
      auto srcMemRefType = cast<MemRefType>(getSrc().getType());
      auto dstMemRefType = cast<MemRefType>(getDst().getType());
      auto srcMemSpaceAttr = srcMemRefType.getMemorySpace();
      auto dstMemSpaceAttr = dstMemRefType.getMemorySpace();
      if (srcMemSpaceAttr && dstMemSpaceAttr) {
        std::vector<std::unique_ptr<util::OperAlignInfo>> alignList;
        if (getUnAlignSizeInfo(*this, &alignList).failed()) {
          return failure();
        }

        auto srcShape = srcMemRefType.getShape();

        auto elemTypeBytes = srcMemRefType.getElementTypeBitWidth() /
                             mlir::utils::INTR_BITS_PER_BYTE;

        SmallVector<int64_t> transposeLoopDims;
        getTransposeLoopDims(transposeLoopDims);

        auto firstAlign = alignList[1]->alignBytes[0] / elemTypeBytes;
        auto firstDim = srcShape[transposeLoopDims[0]];

        auto lastAlign = alignList[0]->alignBytes[0] / elemTypeBytes;
        auto lastDim = srcShape[transposeLoopDims[1]];

        if ((firstDim % firstAlign == 0) && (lastDim % lastAlign == 0)) {
          return emitOpError("VTransposeOp supports unaligned mode only for "
                             "tensors with second unaligned dim");
        }

        if (firstDim % (firstAlign * lastAlign)) {
          return emitOpError("VTransposeOp supports unaligned mode only for "
                             "tensors with double-aligned first axis");
        }
      }
    }
  }

  return success();
}

void VTransposeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                         TypeRange result, Value src, Value dst,
                         DenseI64ArrayAttr permutation, bool disable_align) {
  build(odsBuilder, odsState, result, src, dst, /*temp_buffer=*/nullptr,
        permutation, disable_align);
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

LogicalResult VDeinterleaveOp::verify() {
  auto outputs = getDst();
  auto mode = getIndexMode();
  if (mode == hivm::DeinterleaveMode::ALL_CHANNELS) {
    if (outputs.size() != static_cast<size_t>(getDeInterLeaveChannelNum())) {
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

//===----------------------------------------------------------------------===//
// VCumprodOp
//===----------------------------------------------------------------------===//

LogicalResult VCumprodOp::verify() { return verifyCumOp(*this); }

//===----------------------------------------------------------------------===//
// VCumsumOp
//===----------------------------------------------------------------------===//

LogicalResult VCumsumOp::verify() { return verifyCumOp(*this); }

//===----------------------------------------------------------------------===//
// VSortOp
//===----------------------------------------------------------------------===//

void VSortOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    TypeRange result, Value src, ValueRange dst,
                    bool descending, int64_t sort_axis) {
  build(odsBuilder, odsState, result, src, dst,
        /*temp_buffer=*/nullptr, descending, sort_axis);
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

  int64_t sorAxis = this->getSignedSortAxis();
  ShapedType srcVecType = cast<ShapedType>(getSrc().getType());
  if (sorAxis != srcVecType.getRank() - 1 && sorAxis != -1) {
    return emitOpError() << "Currently only tail axis sorting is supported";
  }
  return success();
}
