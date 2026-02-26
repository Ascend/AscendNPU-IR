//===- OpLimitation.cpp - Operation limited axes and barrier --------------===//
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
//============================================================================//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#define DEBUG_TYPE "flatten-common"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils;
using namespace mlir::utils::debugger;

// getLimitedAxes implementation macros
#define DEFINE_EMPTY_LIMITED_AXES(OpType)                                      \
  SmallVector<int64_t> OpType::getLimitedAxes() { return {}; }
DEFINE_EMPTY_LIMITED_AXES(VInterleaveOp)
DEFINE_EMPTY_LIMITED_AXES(VDeinterleaveOp)
DEFINE_EMPTY_LIMITED_AXES(CopyOp)
DEFINE_EMPTY_LIMITED_AXES(StoreOp)
DEFINE_EMPTY_LIMITED_AXES(LoadOp)
#undef DEFINE_EMPTY_LIMITED_AXES

#define DEFINE_SIMPLE_LIMITED_AXES(OpType, methodName)                         \
  SmallVector<int64_t> OpType::getLimitedAxes() {                              \
    return llvm::to_vector(methodName());                                      \
  }
DEFINE_SIMPLE_LIMITED_AXES(VBrcOp, getBroadcastDims)
DEFINE_SIMPLE_LIMITED_AXES(VReduceOp, getReduceDims)
DEFINE_SIMPLE_LIMITED_AXES(VCumsumOp, getCumDims)
DEFINE_SIMPLE_LIMITED_AXES(VCumprodOp, getCumDims)
#undef DEFINE_SIMPLE_LIMITED_AXES

SmallVector<int64_t> VTransposeOp::getLimitedAxes() {
  auto permutationArray = getPermutationArray();
  auto rank = permutationArray.size();
  SmallVector<int64_t> permutedDims;
  for (size_t i = 0; i < rank; i++) {
    if (permutationArray[i] != static_cast<int>(i)) {
      permutedDims.push_back(i);
    }
  }
  return permutedDims;
}

SmallVector<int64_t> VConcatOp::getLimitedAxes() {
  SmallVector<int64_t> limitedAxes;
  // get concated axes
  this->getConcatLoopDims(limitedAxes);
  return limitedAxes;
}

SmallVector<int64_t> VPadOp::getLimitedAxes() {
  auto limitedAxes = computeElementwiseLimitation(*this);
  // get padded axes
  this->getPadLoopDims(limitedAxes);
  return limitedAxes;
}

SmallVector<int64_t> VGatherOp::getLimitedAxes() {
  SmallVector<int64_t> limitedAxes;
  // get gathered axes
  this->getGatherLoopDims(limitedAxes);
  return limitedAxes;
}

SmallVector<int64_t> VGatherMaskOp::getLimitedAxes() {
  SmallVector<int64_t> limitedAxes;
  // get gathermask axes
  this->getGatherMaskLoopDims(limitedAxes);
  return limitedAxes;
}

SmallVector<int64_t> VArangeOp::getLimitedAxes() {
  // VArangeOp will not be flattened for now.
  SmallVector<int64_t> limitedAxes(this->getResult().getType().getRank());
  std::iota(limitedAxes.begin(), limitedAxes.end(), 0);
  return limitedAxes;
}

SmallVector<int64_t> VFlipOp::getLimitedAxes() {
  SmallVector<int64_t> limitedAxes = computeElementwiseLimitation(*this);
  // limit for flip_axis dimension
  limitedAxes.push_back(this->getFlipAxis());
  return limitedAxes;
}

SmallVector<int64_t> VMulextendedOp::getLimitedAxes() {
  return computeElementwiseLimitation(*this);
}

namespace mlir::hivm::detail {
SmallVector<int64_t> computeElementwiseLimitation(HIVMStructuredOp op) {
  if (op.existInlineBroadcastLoopDims()) {
    SmallVector<int64_t> broadcastDims;
    op.getBroadcastLoopDims(broadcastDims);
    return broadcastDims;
  }
  if (op.existInlineTransposeLoopDims()) {
    llvm_unreachable("use flatten unit for permutation instead");
  }
  return {};
}
} // namespace mlir::hivm::detail
