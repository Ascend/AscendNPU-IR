//===- DimensionMarker.cpp ------------------------------------------------===//
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

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::utils::debugger;

#define DEBUG_TYPE "dimension-analyzer-marker"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hivm {
namespace detail {

static bool isATransposed(Operation *op) {
  if (auto matmulOp = dyn_cast<hivm::MatmulOp>(op))
    return matmulOp.getATranspose().has_value();
  if (auto matmulOp = dyn_cast<hivm::MixMatmulOp>(op))
    return matmulOp.getATranspose().has_value();
  if (auto matmulOp = dyn_cast<hivm::MmadL1Op>(op))
    return matmulOp.getATranspose().has_value();
  return false;
}

static bool isBTransposed(Operation *op) {
  if (auto matmulOp = dyn_cast<hivm::MatmulOp>(op))
    return matmulOp.getBTranspose().has_value();
  if (auto matmulOp = dyn_cast<hivm::MixMatmulOp>(op))
    return matmulOp.getBTranspose().has_value();
  if (auto matmulOp = dyn_cast<hivm::MmadL1Op>(op))
    return matmulOp.getBTranspose().has_value();
  return false;
}

void DimensionAnalyzer::processBFS() {
  SmallVector<Value> argumentListForBFS;
  LDBG("Argument List for BFS in HIVM:");
  op_->walk([&argumentListForBFS](hivm::LoadOp op) {
    argumentListForBFS.push_back(op.getDst());
  });
  op_->walk([&argumentListForBFS](tensor::EmptyOp op) {
    argumentListForBFS.push_back(op.getResult());
  });
  std::queue<Value> bfsQueue;
  for (const auto &arg : argumentListForBFS) {
    updatePreviousType(arg);
    bfsQueue.push(arg);
  }
  DenseSet<Value> visited(argumentListForBFS.begin(), argumentListForBFS.end());
  combineInferable();

  while (!bfsQueue.empty()) {
    Value current = bfsQueue.front();
    bfsQueue.pop();

    for (Operation *user : current.getUsers()) {
      processOperation(user, current);

      for (Value result : user->getResults()) {
        updatePreviousType(result);
        if (visited.insert(result).second) {
          bfsQueue.push(result);
        }
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        auto yieldParentOp = yieldOp->getParentOp();
        LDBG("Encounter yieldOp. Parent " << *yieldParentOp);
        processOperation(yieldParentOp, current);
        for (Value result : yieldParentOp->getResults()) {
          updatePreviousType(result);
          if (visited.insert(result).second) {
            bfsQueue.push(result);
          }
        }
      }
    }
  }
}

bool DimensionAnalyzer::processOperation(Operation *op, Value current) {
  LDBG("Processing operation: " << *op);
  if (auto vBrcOp = dyn_cast<hivm::VBrcOp>(op)) {
    processVBrcOp(vBrcOp);
  } else if (auto vReduceOp = dyn_cast<hivm::VReduceOp>(op)) {
    processVReduceOp(vReduceOp);
  } else if (auto vTransposeOp = dyn_cast<hivm::VTransposeOp>(op)) {
    processVTransposeOp(vTransposeOp);
  } else if (isa<hivm::MatmulOp, hivm::MixMatmulOp, hivm::MmadL1Op>(op)) {
    processMatmulOp(op, isATransposed(op), isBTransposed(op));
  } else if (auto vGatherOp = dyn_cast<hivm::VGatherOp>(op)) {
    processVGatherOp(vGatherOp);
  } else if (auto vConcatOp = dyn_cast<hivm::VConcatOp>(op)) {
    processVConcatOp(vConcatOp);
  } else if (auto vInterleaveOp = dyn_cast<hivm::VInterleaveOp>(op)) {
    processVInterleaveOp(vInterleaveOp);
  } else if (auto vDeinterleaveOp = dyn_cast<hivm::VDeinterleaveOp>(op)) {
    processVDeinterleaveOp(vDeinterleaveOp);
  } else if (auto vPadOp = dyn_cast<hivm::VPadOp>(op)) {
    processVPadOp(vPadOp);
  } else if (auto vCumsumOp = dyn_cast<hivm::VCumsumOp>(op)) {
    processVCumOp(vCumsumOp);
  } else if (auto vCumprodOp = dyn_cast<hivm::VCumprodOp>(op)) {
    processVCumOp(vCumprodOp);
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    processYieldOp(yieldOp);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    processForOp(forOp);
  } else if (auto expandShapeOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
    processReshapeOp(expandShapeOp);
  } else if (auto collapseShapeOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
    processReshapeOp(collapseShapeOp);
  } else if (isElemwiseNaryOpImpl(op) || isa_and_nonnull<CopyOpInterface>(op) ||
             utils::isAllocLikeOp(op)) {
    processParallelOp(op, current);
  } else {
    return DimensionAnalyzerBase::processOperation(op, current);
  }
  return true;
}

SmallVector<int64_t>
DimensionAnalyzer::getMutatedDims(HIVMStructuredOp hivmOp) const {
  auto allDims = llvm::seq(hivmOp.getNumLoops());
  SetVector<int64_t> mutatedDims(allDims.begin(), allDims.end());
  SmallVector<int64_t> parallelDims;
  hivmOp.getParallelLoopDims(parallelDims);
  mutatedDims.set_subtract(parallelDims);

  return mutatedDims.takeVector();
}

/// By default if merge mutation is not provided, it will be true
/// meaning it will be joined together in collapser union find
/// @see VGatherOp
void DimensionAnalyzer::mergeValues(ArrayRef<Value> inputs,
                                    ArrayRef<Value> outputs,
                                    ArrayRef<int64_t> mutatedDims,
                                    bool mergeMutation) {
  LDBG("Merging value: " << outputs[0]);
  LDBG("Input size: " << inputs.size());
  LDBG("Output size: " << outputs.size());
  LDBG("Mutated dims: " << utils::debugger::to_string(mutatedDims));
  auto outputShape = utils::getShape(outputs[0].getType());
  auto rank = outputShape.size();

  createDummyRefIfNotExist(inputs);
  createDummyRefIfNotExist(outputs);

  auto outputArgs = getArgumentRef(outputs[0]);
  auto joinCollapserIfMergeMutation = [this, &mergeMutation](int a, int b) {
    if (mergeMutation)
      joinCollapser(a, b);
  };
  for (auto input : inputs) {
    auto inputArgs = getArgumentRef(input);
    auto mutatedMask = utils::arrayToMask(mutatedDims, inputArgs.size());
    for (unsigned i = 0; i < rank; ++i) {
      if (mutatedMask[i]) {
        isConnected_[outputArgs[i]].elementKind =
            tensor::reshape_utils::ElementKind::HasMutation;
        LDBG("Mutated index: " << outputArgs[i]);
        joinCollapserIfMergeMutation(outputArgs[i], inputArgs[i]);
      } else {
        joinShape(outputArgs[i], inputArgs[i]);
      }
    }
  }
  for (auto output : drop_begin(outputs)) {
    processValue(outputs[0], output);
  }
}

void DimensionAnalyzer::processVBrcOp(hivm::VBrcOp op) {
  LDBG("Processing VBrcOp " << op);
  Value input = op.getSrc();
  Value output = op.getDst();
  SmallVector<Value> inputs;
  SmallVector<Value> outputs(op.getResult());

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");
  if (!mlir::utils::isScalarLike(input))
    inputs.push_back(input);

  outputs.push_back(output);
  mergeValues(inputs, outputs, getMutatedDims(op));
}

void DimensionAnalyzer::processVReduceOp(hivm::VReduceOp op) {
  LDBG("Processing VReduceOp " << op);
  Value input = op.getSrc();
  SmallVector<Value> outputs(op.getResult());

  assert(outputs.size() <= 1 &&
         "Result size must be 1 if tensor type and 0 if memref type");

  outputs.append(op.getDst().begin(), op.getDst().end());
  mergeValues({input}, outputs, getMutatedDims(op));
}

void DimensionAnalyzer::processVTransposeOp(hivm::VTransposeOp op) {
  LDBG("Processing VTransposeOp " << op);
  Value input = op.getSrc();
  Value output = op.getDst();
  auto perm = op.getPermutation();
  const auto &inputArgs = getArgumentRefOrCreateDummy(input);
  auto newValRef = processPermutation(inputArgs, perm, output);
  initCollapseOrVerify(output, newValRef);
  for (Value result : op->getResults()) {
    processValue(result, output);
  }
}

void DimensionAnalyzer::processVGatherOp(hivm::VGatherOp op) {
  LDBG("Processing VGatherOp " << op);
  auto input = op.getSrc();
  auto indice = op.getIndices();
  auto output = op.getDst();
  SmallVector<Value> outputs(op.getResult());

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");

  outputs.push_back(indice);
  outputs.push_back(output);
  mergeValues({input}, outputs, getMutatedDims(op),
              /*mergeMutation=*/false);
}

void DimensionAnalyzer::processVConcatOp(hivm::VConcatOp op) {
  LDBG("Processing VConcatOp " << op);
  SmallVector<Value> inputs(op.getSrc());
  SmallVector<Value> outputs(op.getResults());

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");

  outputs.push_back(op.getDst());
  mergeValues(inputs, outputs, getMutatedDims(op));
}

void DimensionAnalyzer::processVInterleaveOp(hivm::VInterleaveOp op) {
  LDBG("Processing VInterleaveOp " << op);
  auto output = op.getDst();
  SmallVector<Value> inputs(op.getSrc());
  SmallVector<Value> outputs(op.getResult());

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");

  outputs.push_back(output);
  mergeValues(inputs, outputs, getMutatedDims(op));
}

void DimensionAnalyzer::processVDeinterleaveOp(hivm::VDeinterleaveOp op) {
  LDBG("Processing VDeinterleaveOp " << op);
  auto input = op.getSrc();
  SmallVector<Value> outputs(op.getResult());

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");

  outputs.append(op.getDst().begin(), op.getDst().end());
  mergeValues({input}, outputs, getMutatedDims(op));
}

void DimensionAnalyzer::processVPadOp(hivm::VPadOp op) {
  LDBG("Processing VPadOp " << op);
  auto input = op.getSrc();
  auto output = op.getDst();
  SmallVector<Value> outputs(op.getResult());
  SmallVector<int64_t> paddedIndices;

  op.getPadLoopDims(paddedIndices);

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");

  outputs.push_back(output);
  mergeValues({input}, outputs, paddedIndices);
}

template <typename T, typename>
void DimensionAnalyzer::processVCumOp(T op) {
  if constexpr (std::is_same_v<T, hivm::VCumsumOp>) {
    LDBG("Processing VCumsumOp " << op);
  } else {
    LDBG("Processing VCumprodOp " << op);
  }
  auto input = op.getSrc();
  auto output = op.getDst();
  auto reverse = op.getReverse();
  SmallVector<Value> outputs(op.getResult());

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");

  outputs.push_back(output);
  mergeValues({input}, outputs, getMutatedDims(op), reverse);
}

void DimensionAnalyzer::processYieldOp(scf::YieldOp op) {
  LDBG("Processing YieldOp " << op);
  auto parentOp = op->getParentOp();
  if (!parentOp) {
    llvm::report_fatal_error("YieldOp doesn't have a parent");
  }
  for (auto [parentResult, yieldOpResult] :
       llvm::zip_equal(parentOp->getResults(), op.getOperands())) {
    if (isa<ShapedType>(parentResult.getType()))
      mergeValues({parentResult}, {yieldOpResult});
  }
}

void DimensionAnalyzer::processForOp(scf::ForOp op) {
  LDBG("Processing ForOp " << op);
  for (const auto &[regionArg, initArg] :
       zip_equal(op.getRegionIterArgs(), op.getInitArgs())) {
    createDummyRefIfNotExist({regionArg, initArg});
    processValue(regionArg, initArg);
  }
}

template <typename T, typename>
void DimensionAnalyzer::processReshapeOp(T op) {
  if constexpr (std::is_same_v<T, tensor::ExpandShapeOp>) {
    LDBG("Processing ExpandShapeOp " << op);
  } else {
    LDBG("Processing CollapseShapeOp " << op);
  }
  auto input = op.getSrc();
  auto output = op.getResult();
  auto inputArgs = getArgumentRefOrCreateDummy(input);
  auto outputArgs = getArgumentRefOrCreateDummy(output);
  auto inputShape = utils::getShape(input.getType());
  auto outputShape = utils::getShape(output.getType());
  SmallVector<ReassociationIndices> inputIndices;
  SmallVector<ReassociationIndices> outputIndices;
  if constexpr (std::is_same_v<T, tensor::ExpandShapeOp>) {
    for (size_t i = 0; i < inputArgs.size(); i++)
      inputIndices.push_back({static_cast<int64_t>(i)});
    outputIndices = op.getReassociationIndices();
  } else {
    for (size_t i = 0; i < outputArgs.size(); i++)
      outputIndices.push_back({static_cast<int64_t>(i)});
    inputIndices = op.getReassociationIndices();
  }
  LDBG("Computed input indices: " << utils::debugger::to_string(inputIndices));
  LDBG("Input shape: " << utils::debugger::to_string(inputShape));
  LDBG(
      "Computed output indices: " << utils::debugger::to_string(outputIndices));
  LDBG("Output shape: " << utils::debugger::to_string(outputShape));
  assert(inputIndices.size() == outputIndices.size());
  for (const auto &[inputIdx, outputIdx] :
       zip_equal(inputIndices, outputIndices)) {
    LDBG(utils::debugger::to_string(inputIdx)
         << ' ' << utils::debugger::to_string(outputIdx));
    if (inputIdx.size() == 1 && outputIdx.size() == 1) {
      joinShape(inputArgs[inputIndices.back()[0]], outputArgs[outputIdx[0]]);
      continue;
    }
    // Consider not mutated if and only if there exists exactly 1 nonone
    // dimension for each input and output.
    // for example
    // [1, a, 1] -> [a]
    // [a] -> [a, 1]
    // if a != 1, a is considered to be not mutated
    auto filteredInputIdx = to_vector(make_filter_range(
        inputIdx, [&inputShape](int64_t idx) { return inputShape[idx] != 1; }));
    auto filteredOutputIdx =
        to_vector(make_filter_range(outputIdx, [&outputShape](int64_t idx) {
          return outputShape[idx] != 1;
        }));
    for (auto idx : outputIdx) {
      isConnected_[outputArgs[idx]].elementKind =
          tensor::reshape_utils::ElementKind::HasMutation;
    }
    LDBG("Checking all are mutated: " << utils::debugger::to_string(
             map_to_vector(outputIdx, [&outputArgs](int64_t idx) {
               return outputArgs[idx];
             })));
    if (filteredInputIdx.size() == 1 && filteredOutputIdx.size() == 1) {
      LDBG("One of the dimension is not mutated: "
           << outputArgs[*filteredOutputIdx.begin()]);
      isConnected_[outputArgs[*filteredOutputIdx.begin()]].elementKind =
          tensor::reshape_utils::ElementKind::Unit;
      joinShape(outputArgs[*filteredOutputIdx.begin()],
                inputArgs[*filteredInputIdx.begin()]);
    }
  }
}

void DimensionAnalyzer::combineInferable() {
  DimensionAnalyzerBase::combineInferable();
  for (const auto &arg : argumentList_) {
    auto allocOp = arg.getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      continue;
    LDBG("Combining alloc op " << allocOp);
    auto allocRef = getArgumentRefOrCreateDummy(allocOp.getResult());
    auto mixAllocShape = allocOp.getMixedSizes();
    for (auto [allocIdx, el] : llvm::enumerate(mixAllocShape)) {
      if (!el.is<Value>())
        continue;
      auto dimOp = cast<Value>(el).getDefiningOp<memref::DimOp>();
      if (!dimOp)
        continue;
      LDBG("Found dim op " << dimOp);
      auto constantIndex = dimOp.getConstantIndex();
      auto memrefSource = dimOp.getSource();
      if (!constantIndex.has_value())
        continue;
      auto memrefRef = getArgumentRefOrCreateDummy(memrefSource);
      joinShape(memrefRef[constantIndex.value()], allocRef[allocIdx]);
    }
  }
}

void DimensionAnalyzer::markDimensionKind() {
  op_->walk<WalkOrder::PreOrder>([&](VReduceOp reduceOp) {
    // By default reduce would connect with each other
    LDBG("Trying to mark this reduce op " << reduceOp);
    auto reduceResRef = getArgumentRef(reduceOp.getSrc());
    for (auto reduceDim : reduceOp.getReduceDims()) {
      tilingDimKindMap[solverCollapserElem_->find(reduceResRef[reduceDim])] =
          TilingDimensionKind::Reduce;
    }
  });
  auto processSlice = [this](auto sliceOp) {
    if (!argumentsRefPointer_.contains(sliceOp.getSource()))
      return;
    LDBG("Trying to mark this slice op " << sliceOp);
    llvm::SmallBitVector droppedDimsMask = sliceOp.getDroppedDims();
    SmallVector<int64_t> sliceRef;
    if (isa<tensor::ExtractSliceOp>(sliceOp.getOperation())) {
      sliceRef = getArgumentRef(sliceOp.getSource());
    } else {
      sliceRef = getArgumentRef(sliceOp.getResult());
    }
    for (size_t i = 0; i < sliceRef.size(); ++i) {
      if (droppedDimsMask[i]) {
        tilingDimKindMap[solverCollapserElem_->find(sliceRef[i])] =
            TilingDimensionKind::RankReduced;
      }
    }
  };

  op_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<tensor::InsertSliceOp, tensor::ExtractSliceOp>(op)) {
      if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
        processSlice(insertOp);
      } else if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
        processSlice(extractOp);
      }
    }
  });
}
} // namespace detail
} // namespace hivm
} // namespace mlir
