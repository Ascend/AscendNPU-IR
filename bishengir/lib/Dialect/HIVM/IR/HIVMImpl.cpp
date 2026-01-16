//===------------------ HIVMImpl.cpp - HIVM implementation ----------------===//
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
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "hivm-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir::utils::debugger;

namespace mlir {
using namespace utils;
namespace hivm {
std::optional<int> findIdx(SmallVector<Value> valueVec, Value v) {
  auto it = std::find(valueVec.begin(), valueVec.end(), v);
  if (it != valueVec.end()) {
    return it - valueVec.begin();
  }
  return std::nullopt;
}

static bool isIgnoredOp(Operation *op) { return isa<tensor::DimOp>(op); }

template <typename Container>
static Container filterNonIgnoredOps(const Container &container) {
  auto filteredRange = llvm::make_filter_range(
      container, [](Operation *op) { return !isIgnoredOp(op); });
  return Container(filteredRange.begin(), filteredRange.end());
}

int64_t getUsersNum(Value v) {
  return filterNonIgnoredOps(
             DenseSet<Operation *>(v.getUsers().begin(), v.getUsers().end()))
      .size();
}

bool isLocalMatmulInit(Operation *op, Value v) {
  if (auto mmadL1Op = dyn_cast_if_present<hivm::MmadL1Op>(op)) {
    return mmadL1Op.getC() == v;
  }
  if (auto batchMmadL1Op = dyn_cast_if_present<hivm::BatchMmadL1Op>(op)) {
    return batchMmadL1Op.getC() == v;
  }
  return false;
}

bool traceSingleChainUser(
    Value v, const std::function<bool(Operation *, Value v)> &isMatchedOp) {
  auto users = filterNonIgnoredOps(
      DenseSet<Operation *>(v.getUsers().begin(), v.getUsers().end()));
  LDBG("Here computin for value " << v << " " << users.size());
  if (users.size() != 1)
    return false;
  Operation *curOperation = *users.begin();
  LDBG("Current operation " << *curOperation);
  if (isMatchedOp(curOperation, v))
    return true;

  if (curOperation->getDialect()->getNamespace() ==
      HIVMDialect::getDialectNamespace()) {
    return false;
  }

  if (auto extractSliceOp =
          dyn_cast_if_present<tensor::ExtractSliceOp>(curOperation)) {
    return traceSingleChainUser(extractSliceOp.getResult(), isMatchedOp);
  }

  if (auto insertSliceOp =
          dyn_cast_if_present<tensor::InsertSliceOp>(curOperation)) {
    return traceSingleChainUser(insertSliceOp.getResult(), isMatchedOp);
  }

  if (isa<scf::ForOp>(curOperation)) {
    auto forOp = dyn_cast_if_present<scf::ForOp>(curOperation);
    auto initArgs = forOp.getInitArgs();
    auto it = std::find(initArgs.begin(), initArgs.end(), v);
    int initIndx = it == initArgs.end() ? -1 : it - initArgs.begin();
    if (initIndx >= 0) {
      bool hasTraceMmad = traceSingleChainUser(
          forOp.getRegionIterArgs()[initIndx], isMatchedOp);
      if (getUsersNum(initArgs[initIndx]) == 1 && hasTraceMmad)
        return true;
      return false;
    }
  }

  if (isa<scf::ForOp>(curOperation->getParentOp()) &&
      isa<scf::YieldOp>(curOperation)) {
    auto scfForOp =
        dyn_cast_if_present<scf::ForOp>(curOperation->getParentOp());
    SmallVector<Value> yieldValues =
        llvm::to_vector(scfForOp.getYieldedValues());
    auto idx = findIdx(yieldValues, v);
    if (idx.has_value()) {
      auto forResult = scfForOp.getLoopResults().value()[idx.value()];
      return traceSingleChainUser(forResult, isMatchedOp);
    }
  }

  if (isa<scf::IfOp>(curOperation->getParentOp()) &&
      isa<scf::YieldOp>(curOperation)) {
    auto scfIfOp = dyn_cast_if_present<scf::IfOp>(curOperation->getParentOp());
    SmallVector<Value> thenYieldValues =
        llvm::to_vector(scfIfOp.thenYield().getResults());
    auto idxScfIfThen = findIdx(thenYieldValues, v);
    if (idxScfIfThen.has_value()) {
      auto ifResult = scfIfOp.getResults()[idxScfIfThen.value()];
      return traceSingleChainUser(ifResult, isMatchedOp);
    }

    SmallVector<Value> elseYieldValues =
        llvm::to_vector(scfIfOp.elseYield().getResults());
    auto idxScfIfElse = findIdx(elseYieldValues, v);
    if (idxScfIfElse.has_value()) {
      auto ifResult = scfIfOp.getResults()[idxScfIfElse.value()];
      return traceSingleChainUser(ifResult, isMatchedOp);
    }
  }

  if (curOperation->getResults().size() > 1)
    return false;

  if (!curOperation->getResults().empty()) {
    auto dst = curOperation->getResults()[0];
    if (getUsersNum(dst) != 1)
      return false;
    return traceSingleChainUser(dst, isMatchedOp);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Broadcasting Scalar
//===----------------------------------------------------------------------===//

hivm::VBrcOp brcScalar(RewriterBase &rewriter, Location loc,
                       TypedAttr initValue, Value targetTensor) {
  TypeRange resultTypeRange;
  if (llvm::isa<TensorType>(targetTensor.getType())) {
    assert(targetTensor.getDefiningOp<tensor::EmptyOp>() &&
           "definingOp must be tensor::EmptyOp!");
    auto defOp = targetTensor.getDefiningOp<tensor::EmptyOp>();
    resultTypeRange = TypeRange(defOp.getODSResults(0));
  }
  Value init = rewriter.create<arith::ConstantOp>(loc, initValue);
  auto VBrcOp = rewriter.create<hivm::VBrcOp>(
      loc, resultTypeRange, init, targetTensor,
      rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));
  return VBrcOp;
}

std::optional<Operation *>
createEltwiseOpByAtomicKind(OpBuilder &builder, Location loc,
                            TypeRange resTypeRange, ValueRange src,
                            ValueRange dst, hivm::AtomicKind atomicKind) {
  switch (atomicKind) {
  case hivm::AtomicKind::ADD:
    return builder.create<hivm::VAddOp>(loc, resTypeRange, src, dst);
  case hivm::AtomicKind::MIN:
  case hivm::AtomicKind::UMIN:
    return builder.create<hivm::VMinOp>(loc, resTypeRange, src, dst);
  case hivm::AtomicKind::MAX:
  case hivm::AtomicKind::UMAX:
    return builder.create<hivm::VMaxOp>(loc, resTypeRange, src, dst);
  case hivm::AtomicKind::AND:
    return builder.create<hivm::VAndOp>(loc, resTypeRange, src, dst);
  case hivm::AtomicKind::OR:
    return builder.create<hivm::VOrOp>(loc, resTypeRange, src, dst);
  case hivm::AtomicKind::XOR:
    return builder.create<hivm::VXorOp>(loc, resTypeRange, src, dst);
  default:
    return std::nullopt;
  }
}

std::optional<TFuncCoreType> queryFuncCoreType(Operation *funcOp) {
  if (!funcOp) {
    return std::nullopt;
  }
  if (!isa_and_present<func::FuncOp>(funcOp)) {
    return std::nullopt;
  }

  auto tFuncCoreTypeAttr = funcOp->getAttrOfType<hivm::TFuncCoreTypeAttr>(
      hivm::TFuncCoreTypeAttr::name);
  if (tFuncCoreTypeAttr) {
    return tFuncCoreTypeAttr.getFuncCoreType();
  }
  return std::nullopt;
}

FailureOr<TCoreType> getCoreType(Operation *op) {
  // coretype attribute has the highest priority.
  if (auto coreTypeAttr =
          op->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name)) {
    return coreTypeAttr.getTcoretype();
  }
  // annotation.mark has the second highest priority.
  if (op->getNumResults() > 0) {
    auto res = getAnnotateOpWithAttr(op->getResult(0),
                                     mlir::hivm::TCoreTypeMarkerAttr::name);
    if (res.has_value()) {
      return cast<mlir::hivm::TCoreTypeMarkerAttr>(
                 res.value()->getAttr(mlir::hivm::TCoreTypeMarkerAttr::name))
          .getTcoretype();
    }
  }

  if (auto opCoreType = hivm::detail::queryCoreTypeHelper(op))
    return opCoreType.value();

  if (auto callOp = dyn_cast_or_null<func::CallOp>(op)) {
    auto fnName = callOp.getCallee();
    auto fn =
        op->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(fnName);
    if (!fn) {
      op->emitError() << "reference to undefined function '" << fnName << "'";
      return {};
    }
    auto funcCoreType = queryFuncCoreType(fn);
    if (!funcCoreType.has_value()) {
      op->emitError() << "function core type is unknown for '" << fnName << "'";
      return {};
    }
    return kTFuncCoreType2TCoreType.find(funcCoreType.value())->second;
  }
  if (auto forOp = dyn_cast_or_null<scf::ForOp>(op)) {
    if (auto attr = forOp->getAttr("ExtractedLoadOrStore")) {
      // ExtractedLoadOrStore describes the process of discretely loading
      // scalars on ub.which should be split into aiv kernel
      return TCoreType::VECTOR;
    }
  }
  return TCoreType::CUBE_OR_VECTOR;
}

bool isScalarLike(Type type) {
  ShapedType memrefType = dyn_cast<ShapedType>(type);
  return !memrefType || memrefType.getRank() == 1;
}

bool isIdentityStrides(MemRefType shapedType) {
  auto stridedLayout = dyn_cast<StridedLayoutAttr>(shapedType.getLayout());
  if (!stridedLayout)
    return true;
  return stridedLayout.isIdentity();
}

using AlignInfoMap = SmallVector<int64_t>;
SmallVector<int64_t> getAlignedSizes(ArrayRef<int64_t> baseSizes,
                                     AlignInfoMap &alignInfo) {
  auto rank = baseSizes.size();
  SmallVector<int64_t> alignedSizes(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    alignedSizes[i] =
        static_cast<int64_t>(llvm::divideCeil(baseSizes[i], alignInfo[i])) *
        alignInfo[i];
  }
  return alignedSizes;
}

Type getAnnotationMarkByteAlignment(Value value) {
  SmallVector<Operation *> annotateMarks =
      utils::getAllAnnotateOpsWithAttr(value, StrideAlignDimsAttr::name);

  auto shapedType = cast<ShapedType>(value.getType());
  auto rank = shapedType.getRank();
  AlignInfoMap strideAlignElems(rank, 1);
  auto elemType = getElementTypeOrSelf(shapedType);
  for (auto annotateMark : annotateMarks) {
    auto markOp = cast<annotation::MarkOp>(annotateMark);
    auto alignDims = markOp->getAttrOfType<DenseI32ArrayAttr>(
        hivm::StrideAlignDimsAttr::name);
    auto alignBytes = markOp->getAttrOfType<DenseI32ArrayAttr>(
        hivm::StrideAlignValueInByteAttr::name);
    if (alignDims == nullptr || alignBytes == nullptr || alignDims.empty() ||
        alignBytes.empty() || alignDims.size() != alignBytes.size()) {
      // no stride align if no effective align dims and bytes
      continue;
    }
    for (int i = 0; i < alignBytes.size(); ++i) {
      assert(alignBytes[i] * 8 % elemType.getIntOrFloatBitWidth() == 0);
      auto alignElemNum = alignBytes[i] * 8 /
                          static_cast<int>(elemType.getIntOrFloatBitWidth());
      strideAlignElems[alignDims[i]] =
          std::lcm(strideAlignElems[alignDims[i]], alignElemNum);
    }
  }

  for (int i = 1; i < rank; i++) {
    strideAlignElems[rank - 1 - i] = std::lcm(
        strideAlignElems[rank - 1 - i], strideAlignElems[rank - 1 - i + 1]);
  }

  // check if it is already aligned, if so return orignal type, otherwise
  // set new type with new stride
  auto memrefType = cast<MemRefType>(shapedType);
  bool isAlreadyAligned = true;
// TODO : release different version of ascendnpu ir and remove the macro
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
  auto [strides, offset] = getStridesAndOffset(memrefType);
#else
  auto [strides, offset] = memrefType.getStridesAndOffset();
#endif
  llvm::SmallVector<int64_t> alignedStrides(rank, 1);
  for (int64_t i = 0; i < rank; i++) {
    if (strideAlignElems[i] == 1) {
      alignedStrides[i] = strides[i];
      continue;
    }
    if (ShapedType::isDynamic(strides[i])) {
      isAlreadyAligned = false;
      alignedStrides[i] = ShapedType::kDynamic;
      continue;
    }
    alignedStrides[i] = util::ceilFactor(strides[i], strideAlignElems[i]);
    if (strides[i] != alignedStrides[i]) {
      isAlreadyAligned = false;
    }
  }
  if (isAlreadyAligned) {
    return memrefType;
  }
  auto alignedMemRefType = MemRefType::get(
      shapedType.getShape(), shapedType.getElementType(),
      StridedLayoutAttr::get(memrefType.getContext(), offset, alignedStrides));
  return alignedMemRefType;
}

VCastOp castTo(OpBuilder &builder, Location loc, Value src,
               hivm::RoundModeAttr roundMode, Type targetElemType,
               hivm::TypeFnAttr typeFnAttr) {
  // Create targetTensor
  Value targetTensor =
      createTmpBufferOrTensorWithTargetType(builder, loc, src, targetElemType);

  // cast src to targtElemType
  TypeRange resultTypeRange;
  if (llvm::isa<TensorType>(targetTensor.getType())) {
    assert(targetTensor.getDefiningOp<tensor::EmptyOp>() &&
           "definingOp must be tensor::EmptyOp!");
    auto defOp = targetTensor.getDefiningOp<tensor::EmptyOp>();
    resultTypeRange = TypeRange(defOp.getODSResults(0));
  } else if (isa<MemRefType>(src.getType())) {
    resultTypeRange = TypeRange({});
  } else {
    llvm_unreachable("Cast src is neither in tensor type nor in memref type");
    return nullptr;
  }
  mlir::hivm::VCastOp VCastOp = builder.create<hivm::VCastOp>(
      loc, resultTypeRange, src, targetTensor, roundMode, typeFnAttr);
  return VCastOp;
}

namespace util {
bool isIdentityCollapse(ArrayRef<ReassociationIndices> reassociations) {
  return llvm::all_of(reassociations,
                      [](const ReassociationIndices &indiceGroup) {
                        return indiceGroup.size() <= 1;
                      });
}

bool isTransposeWithLastAxis(ArrayRef<int64_t> permutation) {
  assert(!permutation.empty() && "permutation shouldn't be empty.");
  int64_t idx = static_cast<int64_t>(permutation.size()) - 1;
  return idx != permutation[idx];
}

SmallVector<int64_t> getTransposeAxes(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> transposeAxes;
  for (int64_t idx : permutation) {
    if (idx != permutation[idx]) {
      transposeAxes.push_back(idx);
    }
  }
  return transposeAxes;
}

bool isTransposeAdjacentAxes(SmallVector<int64_t> transposeAxes) {
  assert(!transposeAxes.empty() && "transposeAxes shouldn't be empty.");
  assert(transposeAxes.size() == 2 &&
         "Vtranspose only support two axes transpose.");
  return std::abs(transposeAxes[0] - transposeAxes[1]) == 1;
}

FailureOr<std::string> stringfyConstantIntOpValue(Value value) {
  std::stringstream s;
  auto constantOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp());
  if (constantOp == nullptr) {
    return failure();
  }
  Attribute constantAttr =
      llvm::dyn_cast_if_present<Attribute>(constantOp.getValue());
  auto constantInt = dyn_cast_or_null<IntegerAttr>(constantAttr);
  if (constantInt.getType().isInteger()) {
    s << "_" << std::to_string(constantInt.getInt());
    return s.str();
  }
  return failure();
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

} // namespace util

} // namespace hivm
} // namespace mlir
