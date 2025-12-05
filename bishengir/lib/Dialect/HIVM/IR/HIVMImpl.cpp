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

#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
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
  } else if (auto forOp = dyn_cast_or_null<scf::ForOp>(op)) {
    if (Attribute attr = forOp->getAttr(kPipelinedLoopCoreTypeAttrName)) {
      return cast<TCoreTypeAttr>(attr).getTcoretype();
    } else if (auto attr = forOp->getAttr("ExtractedLoadOrStore")) {
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
               hivm::RoundModeAttr roundMode, Type targetElemType) {
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
      loc, resultTypeRange, src, targetTensor, roundMode);
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

} // namespace util

} // namespace hivm
} // namespace mlir
