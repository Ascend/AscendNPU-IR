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

#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

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
  op_->walk([&argumentListForBFS](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](hivm::LoadOp loadOp) {
          argumentListForBFS.push_back(loadOp.getDst());
        })
        .Case([&](tensor::EmptyOp emptyOp) {
          argumentListForBFS.push_back(emptyOp.getResult());
        })
        .Case([&](annotation::MarkOp markOp) {
          if (markOp->hasAttr(hivm::HIVMTightlyCoupledBufferAttr::name)) {
            LDBG(markOp);
            argumentListForBFS.push_back(markOp.getSrc());
          }
        });
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

    for (auto &use : current.getUses()) {
      auto *user = use.getOwner();
      processOperation(user, current);
      if (isa<ShapedType>(current.getType())) {
        createDummyRefIfNotExist({current});
        auto curRef = argumentsRefPointer_.at(current);
        if (auto forOp = dyn_cast<scf::ForOp>(user)) {
          auto regionArg = forOp.getTiedLoopRegionIterArg(&use);
          auto res = forOp.getTiedLoopResult(&use);
          createDummyRefIfNotExist({regionArg, res});
          if (visited.insert(regionArg).second) {
            bfsQueue.push(regionArg);
          }
          solverGroup_->join(curRef, argumentsRefPointer_.at(regionArg));
          solverGroup_->join(curRef, argumentsRefPointer_.at(res));
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
          auto oprNum = use.getOperandNumber();
          auto arg = whileOp.getBeforeArguments()[oprNum];
          createDummyRefIfNotExist({arg});
          if (visited.insert(arg).second) {
            bfsQueue.push(arg);
          }
          solverGroup_->join(curRef, argumentsRefPointer_.at(arg));
        } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
          auto whileOp = cast<scf::WhileOp>(user->getParentOp());
          auto oprNum = use.getOperandNumber() - 1;
          for (auto arg :
               SmallVector<Value>{whileOp.getAfterArguments()[oprNum],
                                  whileOp->getResult(oprNum)}) {
            createDummyRefIfNotExist({arg});
            if (visited.insert(arg).second) {
              bfsQueue.push(arg);
            }
            solverGroup_->join(curRef, argumentsRefPointer_.at(arg));
          }
        } else {
          for (auto res : user->getResults()) {
            if (isa<ShapedType>(res.getType())) {
              createDummyRefIfNotExist({res});
              solverGroup_->join(curRef, argumentsRefPointer_.at(res));
              LDBG(res << " is mapped to "
                       << utils::debugger::to_string(getArgumentRef(res)));
            }
          }
        }
      }

      for (Value result : user->getResults()) {
        updatePreviousType(result);
        if (visited.insert(result).second) {
          bfsQueue.push(result);
        }
      }
      if (isa<scf::YieldOp, scope::ReturnOp>(user)) {
        auto parentOp = user->getParentOp();
        LDBG("Encounter terminator. Parent " << *parentOp);
        processOperation(parentOp, current);
        for (Value result : parentOp->getResults()) {
          updatePreviousType(result);
          if (visited.insert(result).second) {
            bfsQueue.push(result);
          }
        }
        if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
          for (Value init : loopOp.getInits()) {
            updatePreviousType(init);
            if (visited.insert(init).second) {
              bfsQueue.push(init);
            }
          }
        }
      }
    }
  }
}

bool DimensionAnalyzer::processOperation(Operation *op, Value current) {
  LDBG("Processing operation: " << *op);
  return TypeSwitch<Operation *, bool>(op)
      .Case<hivm::VBrcOp>([this](auto op) {
        processVBrcOp(op);
        return true;
      })
      .Case<hivm::VReduceOp>([this](auto op) {
        processVReduceOp(op);
        return true;
      })
      .Case<hivm::VTransposeOp>([this](auto op) {
        processVTransposeOp(op);
        return true;
      })
      .Case<hivm::MatmulOp, hivm::MixMatmulOp, hivm::MmadL1Op>(
          [this](Operation *op) {
            processMatmulOp(op, isATransposed(op), isBTransposed(op));
            return true;
          })
      .Case<hivm::VGatherOp>([this](auto op) {
        processVGatherOp(op);
        return true;
      })
      .Case<hivm::VConcatOp>([this](auto op) {
        processVConcatOp(op);
        return true;
      })
      .Case<hivm::VInterleaveOp>([this](auto op) {
        processVInterleaveOp(op);
        return true;
      })
      .Case<hivm::VDeinterleaveOp>([this](auto op) {
        processVDeinterleaveOp(op);
        return true;
      })
      .Case<hivm::VPadOp>([this](auto op) {
        processVPadOp(op);
        return true;
      })
      .Case<hivm::VCumsumOp>([this](auto op) {
        processVCumOp(op);
        return true;
      })
      .Case<hivm::VCumprodOp>([this](auto op) {
        processVCumOp(op);
        return true;
      })
      .Case<scf::YieldOp>([this](auto op) {
        processYieldOp(op);
        return true;
      })
      .Case<scf::ForOp>([this](auto op) {
        processForOp(op);
        return true;
      })
      .Case<tensor::ExpandShapeOp>([this](auto op) {
        processReshapeOp(op);
        return true;
      })
      .Case<tensor::CollapseShapeOp>([this](auto op) {
        processReshapeOp(op);
        return true;
      })
      .Case<scope::ScopeOp>([this](auto op) {
        processScopeOp(op);
        return true;
      })
      .Case<annotation::MarkOp>([this](auto op) {
        if (op->hasAttr(kTilingDimMappingAttrName)) {
          auto expandShapeOp =
              op.getSrc().template getDefiningOp<tensor::ExpandShapeOp>();
          auto tilingDimMapping = op->template getAttrOfType<DictionaryAttr>(
              kTilingDimMappingAttrName);
          processTilingDimMapping(expandShapeOp, tilingDimMapping);
          return true;
        }
        return false;
      })
      .Default([&](Operation *op) {
        if (isElemwiseNaryOpImpl(op) || isa_and_nonnull<CopyOpInterface>(op) ||
            utils::isAllocLikeOp(op) ||
            isa<memref::MemorySpaceCastOp, bufferization::ToTensorOp,
                bufferization::ToMemrefOp>(op)) {
          processParallelOp(op, current);
          return true;
        }
        return DimensionAnalyzerBase::processOperation(op, current);
      });
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

  assert(outputs.size() <= 2 &&
         "Result size must be 1 or 2 if tensor type and 0 if memref type");

  outputs.append(op.getDst().begin(), op.getDst().end());
  mergeValues({input}, outputs, getMutatedDims(op));
}

void DimensionAnalyzer::processVTransposeOp(hivm::VTransposeOp op) {
  LDBG("Processing VTransposeOp " << op);
  Value input = op.getSrc();
  Value output = op.getDst();
  auto perm = op.getPermutation();
  const auto &inputArgs = getArgumentRefOrCreateDummy(input);
  auto outputArgs = getArgumentRefOrCreateDummy(output);
  for (int i = 0; i < static_cast<int>(inputArgs.size()); ++i) {
    joinCollapser(outputArgs[i], inputArgs[perm[i]]);
  }
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

void DimensionAnalyzer::processVGatherMaskOp(hivm::VGatherMaskOp op) {
  LDBG("Processing VGatherMaskOp " << op);
  auto input = op.getSrc();
  auto mask = op.getMask();
  OperandRange dstRange = op.getDst();
  Value output = dstRange.front();
  SmallVector<Value> outputs(op.getResult());

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");

  outputs.push_back(mask);
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

template <typename T, typename> void DimensionAnalyzer::processVCumOp(T op) {
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
  auto *parentOp = op->getParentOp();
  if (!parentOp) {
    llvm::report_fatal_error("YieldOp doesn't have a parent");
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
    for (auto [beforeArg, yieldOpResult] :
         llvm::zip_equal(whileOp.getBeforeArguments(), op.getOperands())) {
      if (isa<ShapedType>(beforeArg.getType()))
        mergeValues({beforeArg}, {yieldOpResult});
    }
    return;
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

void DimensionAnalyzer::processConditionOp(scf::ConditionOp op) {
  LDBG("Processing ConditionOp " << op);
  auto whileOp = cast<scf::WhileOp>(op->getParentOp());
  for (auto [afterArg, arg] :
       llvm::zip_equal(whileOp.getAfterArguments(), op.getArgs())) {
    if (isa<ShapedType>(afterArg.getType()))
      mergeValues({afterArg}, {arg});
  }
}

void DimensionAnalyzer::processTilingDimMapping(
    tensor::ExpandShapeOp expandShapeOp, DictionaryAttr tilingDimMapping) {
  LDBG("Processing Tiling dim mapping " << expandShapeOp);
  auto src = expandShapeOp.getSrc();
  auto res = expandShapeOp.getResult();
  createDummyRefIfNotExist({src, res});

  auto srcArgs = getArgumentRef(src);
  auto resArgs = getArgumentRef(res);
  for (NamedAttribute dimMappingAttr : tilingDimMapping) {
    int srcDim;
    int resDim = cast<IntegerAttr>(dimMappingAttr.getValue()).getInt();
    llvm::to_integer(dimMappingAttr.getName(), srcDim);
    joinCollapser(srcArgs[srcDim], resArgs[resDim]);
  }
}

template <typename T, typename> void DimensionAnalyzer::processReshapeOp(T op) {
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
      joinShape(inputArgs[inputIdx[0]], outputArgs[outputIdx[0]]);
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

void DimensionAnalyzer::processScopeOp(scope::ScopeOp op) {
  LDBG("Processing ScopeOp " << op);
  auto returnOp = cast<scope::ReturnOp>(op.getRegion().front().getTerminator());
  for (auto [res, ret] :
       llvm::zip_equal(op.getResults(), returnOp.getResults())) {
    createDummyRefIfNotExist({res, ret});
    processValue(res, ret);
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
  auto processSlice = [this](auto sliceOp) {
    if (!argumentsRefPointer_.contains(sliceOp.getSource()))
      return;
    LDBG("Trying to mark this slice op " << sliceOp);
    llvm::SmallBitVector droppedDimsMask = sliceOp.getDroppedDims();
    auto origType = dyn_cast<ShapedType>(sliceOp.getSource().getType());
    auto sliceType = dyn_cast<ShapedType>(sliceOp.getResult().getType());
    auto origRef = getArgumentRefOrCreateDummy(sliceOp.getSource());
    auto sliceRef = getArgumentRefOrCreateDummy(sliceOp.getResult());
    if (isa<tensor::InsertSliceOp>(sliceOp.getOperation())) {
      std::swap(origRef, sliceRef);
      std::swap(origType, sliceType);
    }
    size_t sliceIdx = 0;
    for (size_t i = 0; i < origRef.size(); ++i) {
      if (droppedDimsMask[i]) {
        tilingDimKindMap[solverCollapserElem_->find(origRef[i])] =
            TilingDimensionKind::RankReduced;
        LDBG("Dim " << i << "(" << solverCollapserElem_->find(origRef[i])
                    << ") is marked as RankReduced");
      } else {
        if (isa<tensor::InsertSliceOp>(sliceOp.getOperation()) &&
            sliceType.getDimSize(sliceIdx) == 1) {
          tilingDimKindMap[solverCollapserElem_->find(origRef[i])] =
              TilingDimensionKind::Reduce;
          LDBG("Dim " << i << "(" << solverCollapserElem_->find(origRef[i])
                      << ") is marked as Reduce");
        }
        sliceIdx++;
      }
    }
  };

  op_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto reduceOp = dyn_cast<hivm::VReduceOp>(op)) {
      // By default reduce would connect with each other
      LDBG("Trying to mark this reduce op " << reduceOp);
      auto reduceResRef = getArgumentRef(reduceOp.getSrc());
      for (auto reduceDim : reduceOp.getReduceDims()) {
        tilingDimKindMap[solverCollapserElem_->find(reduceResRef[reduceDim])] =
            TilingDimensionKind::Reduce;
      }
    } else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      processSlice(insertOp);
    } else if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      processSlice(extractOp);
    }
  });
}

void DimensionAnalyzer::markTransposedDims() {
  op_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto vtransposeOp = dyn_cast<hivm::VTransposeOp>(op)) {
      markTransposedDimImpl(vtransposeOp);
    } else if (auto markOp = dyn_cast<annotation::MarkOp>(op);
               markOp && markOp->hasAttr(kTilingDimMappingAttrName)) {
      markTransposedDimImpl(markOp);
    }
  });
}

void DimensionAnalyzer::markTransposedDimImpl(hivm::VTransposeOp op) {
  auto src = op.getSrc();
  auto dst = op.getDst();
  SmallVector<int64_t> srcNonUnitDims;
  SmallVector<int64_t> dstNonUnitDims;
  for (auto dim : utils::getShape(src.getType())) {
    if (dim != 1)
      srcNonUnitDims.push_back(dim);
  }
  for (auto dim : utils::getShape(dst.getType())) {
    if (dim != 1)
      dstNonUnitDims.push_back(dim);
  }
  if (srcNonUnitDims == dstNonUnitDims) {
    return;
  }
  auto srcRef = getArgumentRef(src);
  auto dstRef = getArgumentRef(dst);

  auto perm = op.getPermutation();
  for (auto [dimIdx, parentIdx] : llvm::enumerate(dstRef)) {
    auto srcSolverIdx = solverShapeElem_->find(srcRef[perm[dimIdx]]);
    auto dstSolverIdx = solverShapeElem_->find(parentIdx);
    if (auto it = transposedDimMap.find(srcSolverIdx);
        it != transposedDimMap.end()) {
      LDBG("Successfully moved");
      transposedDimMap[dstSolverIdx] = it->second;
    } else {
      transposedDimMap[dstSolverIdx] = perm[dimIdx];
    }
    LDBG(dstSolverIdx << " is now transposed dim("
                      << transposedDimMap[dstSolverIdx] << ")");
  }
}

void DimensionAnalyzer::markTransposedDimImpl(annotation::MarkOp op) {
  auto expandShapeOp = op.getSrc().getDefiningOp<tensor::ExpandShapeOp>();
  auto tilingDimMapping =
      op->getAttrOfType<DictionaryAttr>(kTilingDimMappingAttrName);
  auto src = expandShapeOp.getSrc();
  auto res = expandShapeOp.getResult();

  auto srcArgs = getArgumentRef(src);
  auto resArgs = getArgumentRef(res);
  for (auto dimMappingAttr : tilingDimMapping) {
    int srcDim;
    int resDim = cast<IntegerAttr>(dimMappingAttr.getValue()).getInt();
    llvm::to_integer(dimMappingAttr.getName(), srcDim);
    srcDim = solverShapeElem_->find(srcArgs[srcDim]);
    resDim = solverShapeElem_->find(resArgs[resDim]);
    LDBG("Checking if transposed dim of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = transposedDimMap.find(srcDim); it != transposedDimMap.end()) {
      LDBG("Successfully moved");
      transposedDimMap[resDim] = it->second;
    }
  }
}

} // namespace detail
} // namespace hivm
} // namespace mlir
