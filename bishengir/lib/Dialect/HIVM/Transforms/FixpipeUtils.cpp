//===- FixpipeUtils.cpp - Shared fixpipe utilities ------------------------===//
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

#include "bishengir/Dialect/HIVM/Transforms/FixpipeUtils.h"

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/Capability/DMA/Fixpipe/FixpipeCapabilityBinding.h"
#include "bishengir/Dialect/HIVM/Transforms/FixpipeCapabilityQuery.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::hivm {

/// NZ fractal dest type for L1 fixpipe: [N1, M1, 16, C0].
/// Use ceilDiv so M/N below the fractal tile (e.g. M=1) pad instead of
/// producing a zero-sized dimension (M1 = M/16 == 0).
RankedTensorType computeNz2NzL1DstTypeChannelSplit(RankedTensorType ndType) {
  assert(ndType.getElementType().isF32() && "only support f32");
  auto rank = ndType.getRank();
  int64_t M = ndType.getDimSize(rank - 2);
  int64_t N = ndType.getDimSize(rank - 1);
  const int64_t alignM = 16;
  const int64_t numElemPerBlock = 8;
  int64_t M1 = static_cast<int64_t>(llvm::divideCeil(M, alignM));
  int64_t N1 = static_cast<int64_t>(llvm::divideCeil(N, alignM)) * 2;
  SmallVector<int64_t> shape;
  for (int64_t i = 0; i < rank - 2; i++) {
    shape.push_back(ndType.getDimSize(i));
  }
  shape.push_back(N1);
  shape.push_back(M1);
  shape.push_back(alignM);
  shape.push_back(numElemPerBlock);
  return RankedTensorType::get(shape, ndType.getElementType());
}

/// NZ fractal dest type for L1 fixpipe: [N1, M1, 16, C0].
/// Use ceilDiv so M/N below the fractal tile (e.g. M=1) pad instead of
/// producing a zero-sized dimension (M1 = M/16 == 0).
RankedTensorType computeNz2NzL1DstTypeChannelMerge(RankedTensorType ndType) {
  auto rank = ndType.getRank();
  int64_t M = ndType.getDimSize(rank - 2);
  int64_t N = ndType.getDimSize(rank - 1);
  const int64_t alignM = 16;
  const int64_t numElemPerBlock = mlir::utils::getNumPerBlock(ndType);
  int64_t M1 = static_cast<int64_t>(llvm::divideCeil(M, alignM));
  int64_t N1 = static_cast<int64_t>(llvm::divideCeil(N, numElemPerBlock));
  SmallVector<int64_t> shape;
  for (int64_t i = 0; i < rank - 2; i++) {
    shape.push_back(ndType.getDimSize(i));
  }
  shape.push_back(N1);
  shape.push_back(M1);
  shape.push_back(alignM);
  shape.push_back(numElemPerBlock);
  return RankedTensorType::get(shape, ndType.getElementType());
}

/// Convert an NZ2ND Fixpipe to NZ2NZ. Applies channel_split or channel_merge
/// (and a fractal L1 dest type) when the dest type requires it; otherwise only
/// updates dma_mode in place. Returns the (possibly replaced) FixpipeOp.
///
/// channel_split=true is only applied when that candidate is Native. Plain
/// NZ2NZ / channel_merge still use the historical rewrite: those L1 paths are
/// not fully modeled until address-space joins the schema (W5.b).
static bool nz2nzRequestIsNative(FixpipeOp op, AddressSpace dstSpace,
                                 std::optional<bool> channelSplit) {
  using bishengir::npu::FixpipeCapabilityRequestBuilder;
  std::optional<FixpipeCapabilityRequestBuilder> builder =
      FixpipeCapabilityRequestBuilder::from(op);
  if (!builder)
    return false;
  builder->dmaMode(FixpipeDMAMode::NZ2NZ);
  if (channelSplit)
    builder->channelSplit(*channelSplit);
  builder->srcAddressSpace(AddressSpace::L0C);
  builder->dstAddressSpace(dstSpace);
  std::optional<bishengir::npu::NPUArchitectureKind> architecture =
      lookupModuleArchitecture(op);
  if (!architecture)
    return false;
  auto storage = builder->build();
  return bishengir::npu::queryNativeSupport(*architecture,
                                            storage.getRequest()) ==
         bishengir::npu::NativeSupport::Native;
}

FixpipeOp convertNz2NdFixpipeToNz2Nz(PatternRewriter &rewriter, FixpipeOp op) {
  auto tensorType = cast<RankedTensorType>(op.getDstOperandType());
  bool channelSplit = shouldEnableChannelSplit(tensorType);
  if (!channelSplit && !shouldEnableChannelMerge(tensorType)) {
    // Plain NZ2NZ mode switch is an execution-schema question.
    if (!nz2nzRequestIsNative(op, AddressSpace::L1,
                              /*channelSplit=*/std::nullopt))
      return op;
    rewriter.modifyOpInPlace(op, [&]() {
      op.setDmaModeAttr(FixpipeDMAModeAttr::get(rewriter.getContext(),
                                                FixpipeDMAMode::NZ2NZ));
    });
    return op;
  }

  if (channelSplit &&
      !nz2nzRequestIsNative(op, AddressSpace::L1, /*channelSplit=*/true))
    return op;

  auto dstTy = channelSplit ? computeNz2NzL1DstTypeChannelSplit(tensorType)
                            : computeNz2NzL1DstTypeChannelMerge(tensorType);
  Location loc = op.getLoc();

  rewriter.setInsertionPoint(op);
  Value fixpipeInit =
      rewriter.create<mlir::tensor::EmptyOp>(loc, dstTy, mlir::ValueRange{});
  auto newOp = FixpipeOp::clone(
      rewriter, op, op.getSrc(), fixpipeInit,
      {rewriter.getNamedAttr(op.getDmaModeAttrName(),
                             FixpipeDMAModeAttr::get(rewriter.getContext(),
                                                     FixpipeDMAMode::NZ2NZ)),
       rewriter.getNamedAttr(op.getChannelSplitAttrName(),
                             rewriter.getBoolAttr(channelSplit))});
  rewriter.replaceOp(op, newOp.getResultTensor());
  return newOp;
}

bool isOnRegBasedArch(Operation *op) {
  auto module =
      isa<ModuleOp>(op) ? cast<ModuleOp>(op) : op->getParentOfType<ModuleOp>();
  return module && hacc::utils::isRegBasedArch(module);
}

} // namespace mlir::hivm
