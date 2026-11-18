//===- InlineFixpipeUtils.cpp - Shared InlineFixpipe queries -------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "bishengir/Dialect/HIVM/Transforms/InlineFixpipe/InlineFixpipeUtils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/Capability/DMA/Fixpipe/FixpipeCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/IR/Contracts/FixpipePreQuantContract.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/FixpipeCapabilityQuery.h"
#include "bishengir/Dialect/HIVM/Transforms/FixpipeUtils.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace mlir::hivm {

bool isInsideVectorScope(Operation *op) {
  auto scopeOp = op->getParentOfType<scope::ScopeOp>();
  if (!scopeOp)
    return false;
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  return coreTypeAttr && coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR;
}

bool isFixpipeStoreInlinable(hivm::StoreOp storeOp) {
  auto attr = storeOp.getAtomicKindAttr();
  auto kind = attr ? attr.getValue() : hivm::AtomicKind::NONE;
  return kind == AtomicKind::NONE || kind == AtomicKind::ADD ||
         kind == AtomicKind::MAX || kind == AtomicKind::MIN;
}

Operation *getSingleSiftedUser(Value v) {
  Operation *singleUser = nullptr;
  for (Operation *user : v.getUsers()) {
    if (isa<annotation::MarkOp, hivm::DebugOp, tensor::DimOp>(user))
      continue;
    if (singleUser)
      return nullptr;
    singleUser = user;
  }
  return singleUser;
}

/// Fixpipe pre-quant cannot implement integer narrowing casts that disable
/// saturation (e.g. trunc-with-overflow semantics). S322I8 fusion requires an
/// explicit enable_saturate=true on every architecture; absent is not a yes.
static bool isIntegerNarrowingCastInlinable(Type inputType, Type outputType,
                                            Operation *castOp) {
  if (!inputType.isIntOrIndex() || !outputType.isIntOrIndex())
    return true;

  int64_t srcBitWidth = inputType.getIntOrFloatBitWidth();
  int64_t dstBitWidth = outputType.getIntOrFloatBitWidth();
  if (srcBitWidth <= dstBitWidth || outputType.isInteger(1))
    return true;

  auto enableSaturate = castOp->getAttrOfType<BoolAttr>("enable_saturate");
  return enableSaturate && enableSaturate.getValue();
}

static bool isVCastInlinableIntoFixpipe(hivm::VCastOp castOp) {
  auto inputType = getElementTypeOrSelf(castOp.getSrc()[0].getType());
  auto outputType = getElementTypeOrSelf(castOp.getDst()[0].getType());
  return isIntegerNarrowingCastInlinable(inputType, outputType, castOp);
}

/// Collect a single-use chain of VCastOps starting at \p firstCast.
/// The chain always includes at least \p firstCast.
SmallVector<hivm::VCastOp> collectVCastChain(hivm::VCastOp firstCast) {
  SmallVector<hivm::VCastOp> chain;
  hivm::VCastOp cur = firstCast;
  while (true) {
    chain.push_back(cur);
    Value res = cur.getResult()[0];
    Operation *nextUser = getSingleSiftedUser(res);
    auto nextCast = dyn_cast_if_present<hivm::VCastOp>(nextUser);
    if (!nextCast || nextCast.getSrc()[0] != res)
      break;
    cur = nextCast;
  }
  return chain;
}

static std::optional<FixpipePreQuantMode>
getQuantModeForTypes(Type inputType, Type outputType) {
  // Candidates come from the generated pre-quant type contract. InlineFixpipe
  // only fuses type-changing quantizing modes: a signature with several
  // candidates (f32 -> f32 matches both NO_QUANT and QF322F32_PRE) or with
  // only identity candidates is deliberately not fused.
  SmallVector<FixpipePreQuantMode> candidates =
      getFixpipePreQuantCandidates(inputType, outputType);
  // NO_QUANT describes identity signatures and is not an explicit quantizing
  // fusion choice.
  llvm::erase(candidates, FixpipePreQuantMode::NO_QUANT);
  llvm::erase(candidates, FixpipePreQuantMode::QF322F16_PRE);
  if (candidates.size() != 1)
    return std::nullopt;
  FixpipePreQuantMode mode = candidates.front();
  if (mode == FixpipePreQuantMode::QF322F32_PRE)
    return std::nullopt;
  return mode;
}

/// True when \p inputType -> \p outputType is an integer bit-width narrowing.
static bool isIntegerNarrowing(Type inputType, Type outputType) {
  if (!inputType.isIntOrIndex() || !outputType.isIntOrIndex())
    return false;
  if (outputType.isInteger(1))
    return false;
  return inputType.getIntOrFloatBitWidth() > outputType.getIntOrFloatBitWidth();
}

/// Decide pre-quant mode from the overall cast chain: element type of the
/// first cast's input to element type of the last cast's output. Every cast
/// in the chain (and the overall conversion) must be inlinable.
std::optional<FixpipePreQuantMode>
getQuantModeForCastChain(ArrayRef<hivm::VCastOp> castChain,
                         FixpipePreQuantMode preQuant) {
  assert(!castChain.empty() && "cast chain must be non-empty");
  for (hivm::VCastOp castOp : castChain) {
    if (!isVCastInlinableIntoFixpipe(castOp))
      return std::nullopt;
  }

  // Copy out of ArrayRef: getSrc/getDst are non-const accessors.
  hivm::VCastOp firstCast = castChain.front();
  hivm::VCastOp lastCast = castChain.back();
  Type inputType = getElementTypeOrSelf(firstCast.getSrc()[0].getType());
  Type outputType = getElementTypeOrSelf(lastCast.getDst()[0].getType());

  // For overall integer narrowing (e.g. i32->i8):
  // - Integer-narrowing steps must explicitly enable saturation.
  // - Float intermediates (i32->f32->f16->i8) often set enable_saturate=false;
  //   ignore those and instead require the last cast to set
  //   enable_saturate=true explicitly. Absent is not enough.
  if (isIntegerNarrowing(inputType, outputType)) {
    bool sawIntegerNarrowingCast = false;
    for (hivm::VCastOp castOp : castChain) {
      Type castIn = getElementTypeOrSelf(castOp.getSrc()[0].getType());
      Type castOut = getElementTypeOrSelf(castOp.getDst()[0].getType());
      if (!isIntegerNarrowing(castIn, castOut))
        continue;
      sawIntegerNarrowingCast = true;
      auto enableSaturate = castOp->getAttrOfType<BoolAttr>("enable_saturate");
      if (!enableSaturate || !enableSaturate.getValue())
        return std::nullopt;
    }
    if (!sawIntegerNarrowingCast) {
      auto enableSaturate =
          lastCast->getAttrOfType<BoolAttr>("enable_saturate");
      if (!enableSaturate || !enableSaturate.getValue())
        return std::nullopt;
    }
  }

  auto mode = getQuantModeForTypes(inputType, outputType);
  if (preQuant != FixpipePreQuantMode::QF322F32_PRE)
    return mode;

  // Compose scalar PRE scaling with its final conversion, preserving scale.
  auto module = firstCast->getParentOfType<ModuleOp>();
  auto saturate = firstCast->getAttrOfType<BoolAttr>("enable_saturate");
  if (mode != FixpipePreQuantMode::F322F16 || castChain.size() != 1 ||
      firstCast.getRoundMode() != RoundMode::RINT ||
      (saturate && saturate.getValue()) || !module ||
      !hacc::utils::isAscend950(module))
    return std::nullopt;
  return FixpipePreQuantMode::QF322F16_PRE;
}

std::optional<FixpipePreQuantMode> getQuantMode(hivm::VCastOp castOp) {
  return getQuantModeForCastChain(collectVCastChain(castOp));
}

bool isUserTransposeStructurallyInlinable(hivm::FixpipeOp op, Operation *user) {
  auto transpose = dyn_cast<hivm::VTransposeOp>(user);
  if (!transpose || op.getDmaMode() != FixpipeDMAMode::NZ2ND)
    return false;
  ArrayRef<int64_t> permutation = transpose.getPermutation();
  if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
    return false;
  return !hasUnmodeledNz2dnConflict(op);
}

bool isUserTransposeInlinable(hivm::FixpipeOp op, Operation *user) {
  if (!isUserTransposeStructurallyInlinable(op, user))
    return false;
  auto transpose = cast<hivm::VTransposeOp>(user);
  // A memref destination has no result. The destination operand carries the
  // element type for both forms.
  Type dstElementType = getElementTypeOrSelf(transpose.getDst().getType());
  // Query the candidate Fixpipe, not the current one: same src/pre_quant/
  // pre_relu/channel_split, dst element type from the transpose destination,
  // dma_mode forced to NZ2DN.
  using bishengir::npu::FixpipeCapabilityRequestBuilder;
  std::optional<FixpipeCapabilityRequestBuilder> builder =
      FixpipeCapabilityRequestBuilder::from(op);
  if (!builder)
    return false;
  builder->dstElementType(dstElementType);
  builder->dmaMode(FixpipeDMAMode::NZ2DN);
  builder->srcAddressSpace(AddressSpace::L0C);
  builder->dstAddressSpace(AddressSpace::GM);
  std::optional<bishengir::npu::NPUArchitectureKind> architecture =
      lookupModuleArchitecture(op);
  if (!architecture)
    return false;
  auto storage = builder->build();
  return bishengir::npu::queryNativeSupport(*architecture,
                                            storage.getRequest()) ==
         bishengir::npu::NativeSupport::Native;
}

} // namespace mlir::hivm
