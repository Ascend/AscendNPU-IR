//===- LowerToLoops.cpp - HIVM Impl by scalar interface -------------------===//
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

#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/TypeUtilities.h"
#include <algorithm>

#define DEBUG_TYPE "op-lower-to-loops-impl"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;
using namespace utils;

namespace mlir {
namespace hivm {

template <typename HIVMOP, typename SCALAROP>
Value getScalarResult(RewriterBase &rewriter, Location loc,
                      llvm::SmallVector<Value, 4> scalarInputs) {
  auto arithOp = rewriter.create<SCALAROP>(loc, scalarInputs);
  return arithOp.getResult();
}

template <typename HIVMOP>
llvm::SmallVector<Value>
createScalarComputeOp(RewriterBase &rewriter, HIVMOP op,
                      llvm::SmallVector<Value, 4> scalarInputs) {
  llvm::SmallVector<Value> resTensors;
  Value resTensor;
  if constexpr (std::is_same<hivm::VMulOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VMulOp, arith::MulIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VMulExtOp, HIVMOP>::value) {
    auto mulextOp = rewriter.create<arith::MulUIExtendedOp>(
        op.getLoc(), scalarInputs[0], scalarInputs[1]);
    resTensors.push_back(mulextOp.getLow());
    resTensors.push_back(mulextOp.getHigh());
  } else if constexpr (std::is_same<hivm::VModOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VModOp, arith::RemSIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VDivOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VDivOp, arith::DivSIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VAddOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VAddOp, arith::AddIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VSubOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VSubOp, arith::SubIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VMinOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VMinOp, arith::MinSIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VMaxOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VMaxOp, arith::MaxSIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VAbsOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VAbsOp, math::AbsIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VCmpOp, HIVMOP>::value) {
    arith::CmpIPredicate predType;
    switch (op.getCompareMode()) {
    case hivm::CompareMode::LT:
      predType = arith::CmpIPredicate::slt;
      break;
    case hivm::CompareMode::GT:
      predType = arith::CmpIPredicate::sgt;
      break;
    case hivm::CompareMode::LE:
      predType = arith::CmpIPredicate::sle;
      break;
    case hivm::CompareMode::GE:
      predType = arith::CmpIPredicate::sge;
      break;
    case hivm::CompareMode::EQ:
      predType = arith::CmpIPredicate::eq;
      break;
    case hivm::CompareMode::NE:
      predType = arith::CmpIPredicate::ne;
      break;
    }
    resTensor = rewriter
                    .create<arith::CmpIOp>(op.getLoc(), predType,
                                           scalarInputs[0], scalarInputs[1])
                    .getResult();
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VShLOp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VShLOp, arith::ShLIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else if constexpr (std::is_same<hivm::VShROp, HIVMOP>::value) {
    resTensor = getScalarResult<hivm::VShROp, arith::ShRSIOp>(
        rewriter, op.getLoc(), scalarInputs);
    resTensors.push_back(resTensor);
  } else {
    llvm_unreachable("Unsupport op type.");
  }
  return resTensors;
}

template <typename HIVMOP>
void decomposeVectorOpToScalarOpImpl(RewriterBase &rewriter, HIVMOP op) {
  auto buildLoopBody = [&rewriter,
                        &op](llvm::SmallVector<Value> indexes) -> void {
    llvm::SmallVector<Value, 4> scalarInputs;
    hivm::HIVMStructuredOp hivmStructureOp =
        cast<hivm::HIVMStructuredOp>(op.getOperation());
    auto getScalarValueFunc = [&rewriter, &indexes,
                               &hivmStructureOp](OpOperand *operand) -> Value {
      auto inlinedBroadcastableAxes =
          hivmStructureOp.getInlinedBroadcastableAxes(operand);
      SmallVector<Value> newIndexes(indexes);
      auto constZero =
          rewriter.create<arith::ConstantIndexOp>(hivmStructureOp->getLoc(), 0);
      for (auto axis : inlinedBroadcastableAxes) {
        newIndexes[axis] = constZero;
      }
      return mlir::utils::getScalarValue(rewriter, hivmStructureOp->getLoc(),
                                         operand->get(), &newIndexes);
    };

    llvm::transform(
        hivmStructureOp.getHIVMInputOperands(false /*includeExtraBuffer*/),
        std::back_inserter(scalarInputs), getScalarValueFunc);

    llvm::SmallVector<Value> dstIndexes(indexes);
    llvm::SmallVector<Value> resTensors =
        createScalarComputeOp(rewriter, op, scalarInputs);

    if constexpr (std::is_same<hivm::VCmpOp, HIVMOP>::value) {
      resTensors[0] = rewriter.create<arith::ExtUIOp>(
          op.getLoc(), rewriter.getIntegerType(8), resTensors[0]);
    }
    for (size_t i = 0; i < resTensors.size(); ++i) {
      size_t resIndex = i;
      mlir::utils::createSinglePointStore(
          rewriter, op.getLoc(), resTensors[resIndex],
          op.getDpsInits()[resIndex], dstIndexes);
    }
  };

  Value dst = op.getDst()[0];
  MemRefType dstType = dyn_cast<MemRefType>(dst.getType());
  std::set<int> loopDims;
  for (int i = 0; i < dstType.getRank(); i++) {
    loopDims.insert(i);
  }
  createNestedLoops(rewriter, op.getLoc(), dst, loopDims, buildLoopBody);
}

template <typename HIVMOP>
FailureOr<SmallVector<Value>>
decomposeVectorOpToScalarOp(RewriterBase &rewriter, HIVMOP op) {
  Value oper = op->getOperand(0);
  MemRefType operType = dyn_cast<MemRefType>(oper.getType());
  if (!operType) {
    return failure();
  }

  decomposeVectorOpToScalarOpImpl(rewriter, op);
  return SmallVector<Value>{};
}

template <typename HIVMOP>
llvm::SmallVector<Value>
createScalarCumulativeComputeOp(RewriterBase &rewriter, HIVMOP op,
                                llvm::SmallVector<Value, 4> scalarInputs) {
  Value resTensor;
  auto elemType = getElementTypeOrSelf(scalarInputs[0]);
  if constexpr (std::is_same<hivm::VCumsumOp, HIVMOP>::value) {
    resTensor = elemType.isInteger()
                    ? getScalarResult<hivm::VCumsumOp, arith::AddIOp>(
                          rewriter, op.getLoc(), scalarInputs)
                    : getScalarResult<hivm::VCumsumOp, arith::AddFOp>(
                          rewriter, op.getLoc(), scalarInputs);
  } else if constexpr (std::is_same<hivm::VCumprodOp, HIVMOP>::value) {
    resTensor = elemType.isInteger()
                    ? getScalarResult<hivm::VCumprodOp, arith::MulIOp>(
                          rewriter, op.getLoc(), scalarInputs)
                    : getScalarResult<hivm::VCumprodOp, arith::MulFOp>(
                          rewriter, op.getLoc(), scalarInputs);
  } else {
    llvm_unreachable("Unsupport op type.");
  }
  llvm::SmallVector<Value> resTensors;
  resTensors.push_back(resTensor);
  return resTensors;
}

template <typename HIVMOP>
void storeFirstValueOfCumDim(RewriterBase &rewriter,
                             llvm::SmallVector<Value> dstIndexes, HIVMOP op,
                             int64_t cumDim) {
  auto startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

  if (op.getReverse()) {
    auto lastInd = cast<ShapedType>(op.getDst().getType()).getShape()[cumDim] - 1;
    startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), lastInd);
  }

  auto *it = dstIndexes.begin() + cumDim;
  dstIndexes.insert(it, startInd);
  auto loadOp = mlir::utils::createSinglePointLoad(
      rewriter, op.getLoc(), op.getDpsInputs()[0], dstIndexes);
  mlir::utils::createSinglePointStore(rewriter, op.getLoc(), loadOp.getResult(),
                                      op.getDpsInits()[0], dstIndexes);
}

template <typename HIVMOP>
Value getPreviousLoopCumulativeValue(RewriterBase &rewriter,
                                     llvm::SmallVector<Value> indexes,
                                     HIVMOP op, int64_t cumDim) {
  // Get previous index
  llvm::SmallVector<Value> indexInputs;
  // Push back the cumulative calculate dimension loop index.
  indexInputs.push_back(indexes[cumDim]);
  auto constOne = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  indexInputs.push_back(constOne.getResult());
  llvm::SmallVector<Value> loopInputs{indexes};
  if (op.getReverse()) {
    auto arithAddOp = rewriter.create<arith::AddIOp>(op.getLoc(), indexInputs);
    loopInputs[cumDim] = arithAddOp.getResult();
  } else {
    auto arithSubOp = rewriter.create<arith::SubIOp>(op.getLoc(), indexInputs);
    loopInputs[cumDim] = arithSubOp.getResult();
  }

  auto previousLoadOp = mlir::utils::createSinglePointLoad(
      rewriter, op.getLoc(), op.getDpsInits()[0], loopInputs);
  return previousLoadOp.getResult();
}

/// Decompose cumulative vector calculate to scalar loops.
/// e.g.
///   vcumsum ins(%a : memref<?x?x?>) outs(%b : memref<?x?x?>) cum_dims = [2]
/// is decomposed to
///   for i 0 to I               -> not cum dim
///     for j 0 to J             -> not cum dim
///       %b[i,j,0] = %a[i,j,0]
///       for k 1 to K           -> cum dim
///         %b[i,j,k] = %b[i,j,k-1] + %a[i,j,k]
template <typename HIVMOP>
FailureOr<SmallVector<Value>>
decomposeCumVectorOpToScalarOpImpl(RewriterBase &rewriter, HIVMOP op) {
  Value dst = op.getDst();
  int64_t cumDim = op.getCumDims()[0];
  auto buildLoopBody = [&rewriter, &dst, &cumDim,
                        &op](llvm::SmallVector<Value> indexes) -> void {
    auto buildLastLoopBody =
        [&rewriter, &indexes, &cumDim,
         &op](llvm::SmallVector<Value> innerIndexes) -> void {
      // Get loop's index
      auto *it = indexes.begin() + cumDim;
      if (op.getReverse()){
        llvm::SmallVector<Value> fixInd;
        auto lastIndNum = cast<ShapedType>(op.getDst().getType()).getShape()[cumDim] - 1;
        auto lastInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), lastIndNum);
        fixInd.push_back(lastInd.getResult());
        fixInd.push_back(innerIndexes[0]);
        auto fixIndVal = rewriter.create<arith::SubIOp>(op.getLoc(), fixInd);
        indexes.insert(it, fixIndVal.getResult());
      } else {
        indexes.insert(it, innerIndexes[0]);
      }

      auto getScalarValueFunc = [&rewriter, &op,
                                 &indexes](OpOperand *operand) -> Value {
        return mlir::utils::getScalarValue(rewriter, op->getLoc(),
                                           operand->get(), &indexes);
      };

      llvm::SmallVector<Value, 4> scalarInputs;
      auto hivmStructureOp = cast<hivm::HIVMStructuredOp>(op.getOperation());
      llvm::transform(
          hivmStructureOp.getHIVMInputOperands(false /*includeExtraBuffer*/),
          std::back_inserter(scalarInputs), getScalarValueFunc);

      // Push the previous value as another scalar input operand
      auto previousLoopValue =
          getPreviousLoopCumulativeValue(rewriter, indexes, op, cumDim);
      scalarInputs.push_back(previousLoopValue);

      llvm::SmallVector<Value> resTensors =
          createScalarCumulativeComputeOp(rewriter, op, scalarInputs);

      for (size_t i = 0; i < resTensors.size(); ++i) {
        mlir::utils::createSinglePointStore(
            rewriter, op.getLoc(), resTensors[i], op.getDpsInits()[i], indexes);
      }
    };

    storeFirstValueOfCumDim(rewriter, indexes, op, cumDim);
    std::set<int> cumLoopDim;
    cumLoopDim.insert(cumDim);
    createNestedLoops(rewriter, op.getLoc(), dst, cumLoopDim, buildLastLoopBody,
                      1);
  };

  std::set<int> loopDims;
  MemRefType dstType = cast<MemRefType>(dst.getType());
  auto rank = dstType.getRank();
  for (int i = 0; i < rank; i++) {
    if (i != cumDim) {
      loopDims.insert(i);
    }
  }
  createNestedLoops(rewriter, op.getLoc(), dst, loopDims, buildLoopBody);
  return SmallVector<Value>{};
}
} // namespace hivm
} // namespace mlir

//===----------------------------------------------------------------------===//
// Macros to help generate `lowerToLoops`
//===----------------------------------------------------------------------===//

#define ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(OP_NAME)               \
  FailureOr<SmallVector<Value>> OP_NAME::lowerToLoops(RewriterBase &b) {       \
    return decomposeVectorOpToScalarOp(b, *this);                              \
  }

#define ENABLE_CUM_OP_LOWER_TO_LOOPS_IMPLEMENTATION(OP_NAME)                   \
  FailureOr<SmallVector<Value>> OP_NAME::lowerToLoops(RewriterBase &b) {       \
    return decomposeCumVectorOpToScalarOpImpl(b, *this);                       \
  }

ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VMulOp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VAddOp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VSubOp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VMinOp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VMaxOp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VAbsOp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VShLOp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VShROp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VModOp)
ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VDivOp)
#undef ENABLE_DEFAULT_OP_LOWER_TO_LOOPS_IMPLEMENTATION

ENABLE_CUM_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VCumprodOp)
ENABLE_CUM_OP_LOWER_TO_LOOPS_IMPLEMENTATION(VCumsumOp)
#undef ENABLE_CUM_OP_LOWER_TO_LOOPS_IMPLEMENTATION

//===----------------------------------------------------------------------===//
// VCmpOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> VCmpOp::lowerToLoops(RewriterBase &b) {
  decomposeVectorOpToScalarOpImpl(b, *this);
  return SmallVector<Value>{};
}

//===----------------------------------------------------------------------===//
// VDeinterleaveOp
//===----------------------------------------------------------------------===//

namespace mlir::hivm {

void decomposeVDeinterleaveB64ToScalarSingleChannel(
    hivm::VDeinterleaveOp &op, RewriterBase &rewriter,
    llvm::SmallVector<Value> &indices, DeinterleaveMode mode, size_t dstIndex) {
  auto loc = op->getLoc();
  auto index = [&rewriter, &loc](int i) {
    return rewriter.create<arith::ConstantIndexOp>(loc, i);
  };
  llvm::SmallVector<Value> srcIndices;
  auto getScalarValueFunc = [&rewriter, &loc, &srcIndices](Value v) -> Value {
    return getScalarValue(rewriter, loc, v, &srcIndices);
  };
  // step1. calculate the index of every b64 value of CHANNEL_0/CHANNEL_1 result
  // in the input vector(last axis) CHANNEL_0 : i in dst vector -> 2 * i in
  // input vector CHANNEL_1 : i in dst vector -> 2 * i + 1 in input vector
  auto iterTypes = op.getIteratorTypesArray();
  for (size_t i = 0; i < indices.size(); ++i) {
    if (iterTypes[i] == hivm::IteratorType::kDeinterleave) {
      auto mul = rewriter.create<arith::MulIOp>(loc, index(2), indices[i]);
      if (mode == DeinterleaveMode::CHANNEL_0) {
        srcIndices.push_back(mul);
      }
      if (mode == DeinterleaveMode::CHANNEL_1) {
        srcIndices.push_back(
            rewriter.create<arith::AddIOp>(loc, index(1), mul));
      }
      continue;
    }
    srcIndices.push_back(indices[i]);
  }

  // step2. load b64 value from input to scalar according to the index vector
  llvm::SmallVector<Value, 4> scalarInputs;
  llvm::transform(op.getDpsInputs(), std::back_inserter(scalarInputs),
                  getScalarValueFunc);

  // step3. store the b64 value according to the index in dst vector
  // in ALL_CHANNELS mode, there are 2 dst vectors,
  // store CHANNEL_0 result into dst[0] and CHANNEL_1 result into dst[1]
  createSinglePointStore(rewriter, loc, scalarInputs[0], op.getDst()[dstIndex],
                         indices);
}

FailureOr<SmallVector<Value>>
decomposeVDeinterleaveB64ToScalarOpImpl(RewriterBase &rewriter,
                                        VDeinterleaveOp op) {
  // input type need to be int64/uint64
  Value src = op.getSrc();
  auto srcElemType = getElementTypeOrSelf(src);
  if (!srcElemType.isInteger(64)) {
    return failure();
  }

  auto buildLoopBody = [&rewriter,
                        &op](llvm::SmallVector<Value> indices) -> void {
    // ALL_CHANNELS mode
    // need to set the dst index of CHANNEL_1 result
    if (op.getIndexMode() == DeinterleaveMode::ALL_CHANNELS) {
      decomposeVDeinterleaveB64ToScalarSingleChannel(
          op, rewriter, indices, DeinterleaveMode::CHANNEL_0, 0);
      decomposeVDeinterleaveB64ToScalarSingleChannel(
          op, rewriter, indices, DeinterleaveMode::CHANNEL_1, 1);
    }

    // CHANNEL_0 mode
    if (op.getIndexMode() == DeinterleaveMode::CHANNEL_0) {
      decomposeVDeinterleaveB64ToScalarSingleChannel(
          op, rewriter, indices, DeinterleaveMode::CHANNEL_0, 0);
    }

    // CHANNEL_1 mode
    if (op.getIndexMode() == DeinterleaveMode::CHANNEL_1) {
      decomposeVDeinterleaveB64ToScalarSingleChannel(
          op, rewriter, indices, DeinterleaveMode::CHANNEL_1, 0);
    }
  };

  Value dst = op.getDst()[0];
  MemRefType dstMemType = dyn_cast<MemRefType>(dst.getType());
  std::set<int> loopDims;
  for (int i = 0; i < dstMemType.getRank(); i++) {
    loopDims.insert(i);
  }

  createNestedLoops(rewriter, op.getLoc(), dst, loopDims, buildLoopBody);
  return SmallVector<Value>{};
}

} // namespace mlir::hivm

FailureOr<SmallVector<Value>> VDeinterleaveOp::lowerToLoops(RewriterBase &b) {
  return decomposeVDeinterleaveB64ToScalarOpImpl(b, *this);
}

//===----------------------------------------------------------------------===//
// VInterleaveOp
//===----------------------------------------------------------------------===//

namespace mlir::hivm {

FailureOr<SmallVector<Value>>
decomposeVInterleaveOpToScalarOpImpl(RewriterBase &rewriter, VInterleaveOp op) {
  auto buildLoopBody = [&rewriter,
                        &op](llvm::SmallVector<Value> indices) -> void {
    auto getScalarValueFunc = [&rewriter, &op, &indices](Value v) -> Value {
      return getScalarValue(rewriter, op->getLoc(), v, &indices);
    };

    // step1. load the b64 value from op input into scalar
    llvm::SmallVector<Value, 4> scalarInputs;
    llvm::transform(op.getDpsInputs(), std::back_inserter(scalarInputs),
                    getScalarValueFunc);

    auto loc = op->getLoc();
    auto index = [&rewriter, &loc](int i) {
      return rewriter.create<arith::ConstantIndexOp>(loc, i);
    };

    // step2. calculate the indexes of every value in the dst of interleave
    // op(last axis) every index i from the first vector -> 2 * i in the dst
    // vector every index i from the second vector -> 2 * i + 1 in the dst
    // vector
    llvm::SmallVector<Value> dstIndices0;
    llvm::SmallVector<Value> dstIndices1;
    auto iterTypes = op.getIteratorTypesArray();
    for (size_t i = 0; i < indices.size(); ++i) {
      if (iterTypes[i] == hivm::IteratorType::kInterleave) {
        auto mul = rewriter.create<arith::MulIOp>(loc, index(2), indices[i]);
        dstIndices0.push_back(mul);
        dstIndices1.push_back(
            rewriter.create<arith::AddIOp>(loc, index(1), mul));
        continue;
      }
      dstIndices0.push_back(indices[i]);
      dstIndices1.push_back(indices[i]);
    }

    // step3. store the b64 value from scalar according to the index in the
    // dst vector
    createSinglePointStore(rewriter, loc, scalarInputs[0], op.getDst(),
                           dstIndices0);
    createSinglePointStore(rewriter, loc, scalarInputs[1], op.getDst(),
                           dstIndices1);
  };

  Value src = op.getOperand(0);
  MemRefType srcMemType = dyn_cast<MemRefType>(src.getType());
  std::set<int> loopDims;
  for (int i = 0; i < srcMemType.getRank(); i++) {
    loopDims.insert(i);
  }

  createNestedLoops(rewriter, op.getLoc(), src, loopDims, buildLoopBody);
  return SmallVector<Value>{};
}

} // namespace mlir::hivm

FailureOr<SmallVector<Value>> VInterleaveOp::lowerToLoops(RewriterBase &b) {
  return decomposeVInterleaveOpToScalarOpImpl(b, *this);
}

//===----------------------------------------------------------------------===//
// VMulExtOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Value>> VMulExtOp::lowerToLoops(RewriterBase &b) {
  return decomposeVectorOpToScalarOp<VMulExtOp>(b, *this);
}

//===----------------------------------------------------------------------===//
// VReduceOp
//===----------------------------------------------------------------------===//

namespace mlir::hivm {

Value calculateIndexIntegerArgMinArgMax(
    RewriterBase &rewriter, hivm::VReduceOp op,
    llvm::SmallVector<Value, 4> scalarInputs,
    llvm::SmallVector<Value, 4> scalarIndx, arith::CmpIPredicate predType,
    Value resTensor, Value index) {
  // Check if the resulting value was updated to
  // value at current index indicating index needs updating.
  Value resConditional = rewriter
                             .create<arith::CmpIOp>(op.getLoc(), predType,
                                                    resTensor, scalarInputs[0])
                             .getResult();

  // Typecast index value to Int32 and select based on ouptut of CMPI
  auto indval =
      rewriter
          .create<arith::IndexCastOp>(op.getLoc(), rewriter.getI32Type(), index)
          .getResult();
  Value resIndex = rewriter.create<arith::SelectOp>(op.getLoc(), resConditional,
                                                    indval, scalarIndx[0]);
  return resIndex;
}

Value calculateIndexFloatArgMinArgMax(RewriterBase &rewriter,
                                      hivm::VReduceOp op,
                                      llvm::SmallVector<Value, 4> scalarInputs,
                                      llvm::SmallVector<Value, 4> scalarIndx,
                                      arith::CmpFPredicate predType,
                                      Value resTensor, Value index) {
  // Check if the resulting value was updated to
  // value at current index indicating index needs updating.
  Value resConditional = rewriter
                             .create<arith::CmpFOp>(op.getLoc(), predType,
                                                    resTensor, scalarInputs[0])
                             .getResult();

  // Typecast index value to Int32 and select based on ouptut of CMPF
  auto indval =
      rewriter
          .create<arith::IndexCastOp>(op.getLoc(), rewriter.getI32Type(), index)
          .getResult();
  Value resIndex = rewriter.create<arith::SelectOp>(op.getLoc(), resConditional,
                                                    indval, scalarIndx[0]);
  return resIndex;
}

Value calculateResFloatArgMinArgMax(RewriterBase &rewriter, hivm::VReduceOp op,
                                    llvm::SmallVector<Value, 4> scalarInputs,
                                    arith::CmpFPredicate predType,
                                    Value resTensor) {
  // Check if the resulting value was updated to
  // value at current index indicating res needs updating.
  Value resConditional = rewriter
                             .create<arith::CmpFOp>(op.getLoc(), predType,
                                                    resTensor, scalarInputs[0])
                             .getResult();

  Value res = rewriter.create<arith::SelectOp>(op.getLoc(), resConditional,
                                               scalarInputs[0], resTensor);
  return res;
}

template <typename SCALAROP>
std::pair<Value, Value> getIndexTensorForArgmaxArgmin(
    Type elemType, RewriterBase &rewriter, hivm::VReduceOp op,
    llvm::SmallVector<Value, 4> scalarInputs,
    llvm::SmallVector<Value, 4> scalarIndx, Value index,
    arith::CmpIPredicate intPred, arith::CmpFPredicate floatPred) {
  Value resTensor;
  Value resIndex;
  if (elemType.isInteger()) {
    resIndex = calculateIndexIntegerArgMinArgMax(rewriter, op, scalarInputs,
                                                 scalarIndx, intPred,
                                                 scalarInputs[1], index);

    resTensor = getScalarResult<hivm::VMinOp, SCALAROP>(rewriter, op.getLoc(),
                                                        scalarInputs);
  } else {
    resIndex =
        calculateIndexFloatArgMinArgMax(rewriter, op, scalarInputs, scalarIndx,
                                        floatPred, scalarInputs[1], index);
    resTensor = calculateResFloatArgMinArgMax(rewriter, op, scalarInputs,
                                              floatPred, scalarInputs[1]);
  }
  return {resIndex, resTensor};
}

llvm::SmallVector<Value>
createScalarReduceComputeOp(RewriterBase &rewriter, hivm::VReduceOp op,
                            llvm::SmallVector<Value, 4> scalarInputs,
                            llvm::SmallVector<Value, 4> scalarIndx,
                            Value index) {
  Value resTensor;
  Value resIndex;
  auto reduceOpArith = op.getArithAttr();
  auto reduceOpAttr = reduceOpArith.getReduceOp();
  auto elemType = getElementTypeOrSelf(op.getOperandTypes()[0]);
  switch (reduceOpAttr) {
  case hivm::ReduceOperation::min:
    resTensor = getScalarResult<hivm::VMinOp, arith::MinSIOp>(
        rewriter, op.getLoc(), scalarInputs);
    break;
  case hivm::ReduceOperation::max:
    resTensor = getScalarResult<hivm::VMaxOp, arith::MaxSIOp>(
        rewriter, op.getLoc(), scalarInputs);
    break;
  case hivm::ReduceOperation::sum:
    resTensor = getScalarResult<hivm::VAddOp, arith::AddIOp>(
        rewriter, op.getLoc(), scalarInputs);
    break;
  case hivm::ReduceOperation::prod:
    resTensor = getScalarResult<hivm::VMulOp, arith::MulIOp>(
        rewriter, op.getLoc(), scalarInputs);
    break;
  case hivm::ReduceOperation::xori:
    resTensor = rewriter.create<arith::XOrIOp>(op.getLoc(), scalarInputs);
    break;
  case hivm::ReduceOperation::min_with_index_left:
    std::tie(resIndex, resTensor) =
        getIndexTensorForArgmaxArgmin<arith::MinSIOp>(
            elemType, rewriter, op, scalarInputs, scalarIndx, index,
            arith::CmpIPredicate::sgt, arith::CmpFPredicate::OGT);
    break;
  case hivm::ReduceOperation::min_with_index_right:
    std::tie(resIndex, resTensor) =
        getIndexTensorForArgmaxArgmin<arith::MinSIOp>(
            elemType, rewriter, op, scalarInputs, scalarIndx, index,
            arith::CmpIPredicate::sge, arith::CmpFPredicate::OGE);
    break;
  case hivm::ReduceOperation::max_with_index_left:
    std::tie(resIndex, resTensor) =
        getIndexTensorForArgmaxArgmin<arith::MaxSIOp>(
            elemType, rewriter, op, scalarInputs, scalarIndx, index,
            arith::CmpIPredicate::slt, arith::CmpFPredicate::OLT);
    break;
  case hivm::ReduceOperation::max_with_index_right:
    std::tie(resIndex, resTensor) =
        getIndexTensorForArgmaxArgmin<arith::MaxSIOp>(
            elemType, rewriter, op, scalarInputs, scalarIndx, index,
            arith::CmpIPredicate::sle, arith::CmpFPredicate::OLE);
    break;
  default:
    llvm_unreachable("Unsupport Reduction Arith Attr.");
  }
  llvm::SmallVector<Value> resTensors;
  resTensors.push_back(resTensor);

  // Order is important, so we only insert once we have inserted
  // min/max value while running max_with_index/min_with_index.
  if (VReduceOp::isArgminOrArgmax(reduceOpAttr)) {
    resTensors.push_back(resIndex);
  }

  return resTensors;
}

void insertReduceInitialization(RewriterBase &rewriter, hivm::VReduceOp op,
                                const llvm::SmallVector<Value> &indexes) {
  // Get the dstIndexes by changing the index of the reduce axis to 0
  llvm::SmallVector<Value> dstIndexes(indexes);
  auto reduceDims = op.getReduceDims();
  auto constZero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  for (size_t i = 0; i < indexes.size(); i++)
    if (reduceDims[0] == static_cast<int>(i))
      dstIndexes[i] = constZero;

  // Create an IfOp to check whether it is the first iteration
  Value isReduceInitCond = rewriter.create<arith::CmpIOp>(
      op.getLoc(), rewriter.getI1Type(), arith::CmpIPredicate::eq,
      indexes[reduceDims[0]], constZero);
  scf::IfOp scfIfOp = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange(),
                                                 isReduceInitCond, false);

  // Initialize the dst value and index inside the IfOp
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&scfIfOp.getThenRegion().front());

  auto constValueInit = rewriter.create<arith::ConstantOp>(
      op->getLoc(), cast<TypedAttr>(op.getInit()));
  rewriter.create<memref::StoreOp>(op->getLoc(), constValueInit,
                                   op.getDpsInits()[0], dstIndexes);

  auto arith = op.getArithAttr().getReduceOp();
  if (VReduceOp::isArgminOrArgmax(arith)) {
    auto constIndexInit = rewriter.create<arith::ConstantOp>(
        op->getLoc(), IntegerAttr::get(rewriter.getI32Type(), 0));
    rewriter.create<memref::StoreOp>(op->getLoc(), constIndexInit,
                                     op.getDpsInits()[1], dstIndexes);
  }
}

void decomposeVReduceOpToScalarOpImpl(RewriterBase &rewriter, VReduceOp op) {
  auto buildLoopBody = [&rewriter,
                        &op](llvm::SmallVector<Value> indexes) -> void {
    auto getScalarValueFunc = [&rewriter, &op,
                               &indexes](OpOperand *operand) -> Value {
      return getScalarValue(rewriter, op->getLoc(), operand->get(), &indexes);
    };

    llvm::SmallVector<Value, 4> scalarInputs;
    llvm::SmallVector<Value, 4> scalarIndx;
    auto hivmStructureOp = dyn_cast<hivm::HIVMStructuredOp>(op.getOperation());
    assert(hivmStructureOp);
    llvm::transform(
        hivmStructureOp.getHIVMInputOperands(false /*includeExtraBuffer*/),
        std::back_inserter(scalarInputs), getScalarValueFunc);

    // Since the reduce operation src is different from the dst shape,
    // additional indexes are required to obtain the dst value.
    llvm::SmallVector<Value> dstIndexes(indexes);
    auto reduceDims = op.getReduceDims();
    auto constZero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    // update dstIndexes: change index of reduce axis to 0
    dstIndexes[reduceDims[0]] = constZero;

    // Insert the value initialization operations
    insertReduceInitialization(rewriter, op, indexes);

    // push reduce init as another scalar input operand
    if (op.getIndices()) {
      scalarInputs.pop_back();
    }

    if (isa<MemRefType>(op.getDstValue().getType())) {
      auto loadOp = createSinglePointLoad(rewriter, op.getLoc(),
                                          op.getDpsInits()[0], dstIndexes);
      scalarInputs.push_back(loadOp.getResult());
    } else {
      scalarInputs.push_back(op.getDpsInits()[0]);
    }

    auto reduceOpArith = op.getArithAttr();
    auto reduceOpAttr = reduceOpArith.getReduceOp();
    if (VReduceOp::isArgminOrArgmax(reduceOpAttr)) {
      // Load the Index value needed for update across iterations.
      auto loadIndOp = createSinglePointLoad(rewriter, op.getLoc(),
                                             op.getDpsInits()[1], dstIndexes);
      scalarIndx.push_back(loadIndOp.getResult());
    }

    // Depending on reduce dimension axis selecting which index should be
    // picked.
    auto indexVal = indexes.size() > 1 ? reduceDims[0] : 0;

    llvm::SmallVector<Value> resTensors = createScalarReduceComputeOp(
        rewriter, op, scalarInputs, scalarIndx, indexes[indexVal]);

    for (size_t i = 0; i < resTensors.size(); ++i) {	
      createSinglePointStore(rewriter, op.getLoc(), resTensors[i],	
                             op.getDpsInits()[i], dstIndexes);	
    }
  };

  Value dst = op->getOperand(0);
  MemRefType dstType = dyn_cast<MemRefType>(dst.getType());
  std::set<int> loopDims;
  for (int i = 0; i < dstType.getRank(); i++) {
    loopDims.insert(i);
  }
  auto forOps = createNestedLoops(rewriter, op.getLoc(), dst, loopDims, buildLoopBody);
  if (op.getIndices()) {
    if (forOps.size() > 0) {
      rewriter.setInsertionPointAfter(forOps[0]);
    }
    auto loadAndStoreIndexFunc = [&rewriter, &op](llvm::SmallVector<Value> dstIndexes,
                                                  int64_t reduceDim) -> void {
      auto loadedIndex = createSinglePointLoad(rewriter, op.getLoc(),
                                                op.getDpsInits()[1], dstIndexes).getResult();
      auto idxIndex = rewriter.create<arith::IndexCastOp>(op.getLoc(), 
                                                        TypeRange{rewriter.getIndexType()}, 
                                                        ValueRange{loadedIndex}).getResult();
      llvm::SmallVector<Value> idxIndexes(dstIndexes);
      idxIndexes[reduceDim] = idxIndex;
      auto resIndex = createSinglePointLoad(rewriter, op.getLoc(),
                                            op.getIndices(), idxIndexes).getResult();
      createSinglePointStore(rewriter, op.getLoc(), resIndex, op.getDpsInits()[1], dstIndexes);
    };
    
    if (loopDims.size() == 1) {
      auto constZero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      llvm::SmallVector<Value> dstIndexes = {constZero};
      int64_t reduceDim = op.getReduceDims()[0];
      loadAndStoreIndexFunc(dstIndexes, reduceDim);
    } else {
      auto buildGatherLoopBody = [&rewriter, &op,
                                  &loadAndStoreIndexFunc](llvm::SmallVector<Value> indexes) -> void {
        llvm::SmallVector<Value> dstIndexes(indexes);
        int64_t reduceDim = op.getReduceDims()[0];
        auto constZero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        auto *it = dstIndexes.begin() + reduceDim;
        // update dstIndexes: insert 0 to index of reduce axis
        dstIndexes.insert(it, constZero);
        loadAndStoreIndexFunc(dstIndexes, reduceDim);
      };
      loopDims.erase(op.getReduceDims()[0]);
      createNestedLoops(rewriter, op.getLoc(), dst, loopDims, buildGatherLoopBody);
    }
  }
}

} // namespace mlir::hivm

FailureOr<SmallVector<Value>> VReduceOp::lowerToLoops(RewriterBase &b) {
  decomposeVReduceOpToScalarOpImpl(b, *this);
  return SmallVector<Value>{};
}
