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
#include "bishengir/Dialect/HIVM/Transforms/TileAndBindSubBlock/Helper.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>

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

void DimensionAnalyzer::handleValueGroupForUse(Operation *user, Value current,
                                               OpOperand *use,
                                               unsigned operandNumber) {
  if (!isa<ShapedType>(current.getType()))
    return;
  createDummyRefIfNotExist({current});
  if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    if (use) {
      if (auto regionArg = forOp.getTiedLoopRegionIterArg(use)) {
        joinValueGroup(current, regionArg);
      }
      if (auto res = forOp.getTiedLoopResult(use))
        joinValueGroup(current, res);
    }
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
    if (operandNumber < whileOp.getBeforeArguments().size()) {
      auto arg = whileOp.getBeforeArguments()[operandNumber];
      joinValueGroup(current, arg);
    }
  } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
    if (operandNumber > 0) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      auto idx = operandNumber - 1;
      if (idx < whileOp.getAfterArguments().size()) {
        joinValueGroup(current, whileOp.getAfterArguments()[idx]);
      }
      if (idx < whileOp.getNumResults())
        joinValueGroup(current, whileOp->getResult(idx));
    }
  } else {
    for (auto res : user->getResults()) {
      if (isa<ShapedType>(res.getType())) {
        joinValueGroup(current, res);
        LDBG(res << " is mapped to "
                 << utils::debugger::to_string(getValueDimIndices(res)));
      }
    }
  }

  if (isa<scf::YieldOp, scf::ConditionOp, scope::ReturnOp>(user)) {
    auto parentOp = user->getParentOp();
    if (!parentOp)
      return;
    auto resultIdx = operandNumber;
    if (isa<scf::ConditionOp>(user)) {
      if (resultIdx == 0)
        return;
      resultIdx -= 1;
    }
    if (resultIdx < parentOp->getNumResults()) {
      auto res = parentOp->getResult(resultIdx);
      joinValueGroup(current, res);
    }
  }
}

void DimensionAnalyzer::processBFS() {
  SetVector<Value> argumentListForBFS;
  LDBG("Argument List for BFS in HIVM:");
  op_->walk([&argumentListForBFS](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](hivm::LoadOp loadOp) {
          argumentListForBFS.insert(loadOp.getDst());
        })
        .Case<tensor::EmptyOp, memref::AllocOp>(
            [&](auto op) { argumentListForBFS.insert(op.getResult()); })
        .Case([&](annotation::MarkOp markOp) {
          if (markOp->hasAttr(hivm::HIVMTightlyCoupledBufferAttr::name))
            argumentListForBFS.insert(markOp.getSrc());
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
      handleValueGroupForUse(user, current, &use, use.getOperandNumber());

      for (Value result : user->getResults()) {
        updatePreviousType(result);
        if (visited.insert(result).second) {
          bfsQueue.push(result);
        }
      }
      if (isa<scf::YieldOp, scf::ConditionOp, scope::ReturnOp>(user)) {
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

void DimensionAnalyzer::processPreOrderWalk() {
  SmallVector<Value> argumentListForWalk;
  LDBG("Argument List for PreOrder walk in HIVM:");
  op_->walk([&argumentListForWalk](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](hivm::LoadOp loadOp) {
          argumentListForWalk.push_back(loadOp.getDst());
        })
        .Case([&](tensor::EmptyOp emptyOp) {
          argumentListForWalk.push_back(emptyOp.getResult());
        })
        .Case([&](annotation::MarkOp markOp) {
          if (markOp->hasAttr(hivm::HIVMTightlyCoupledBufferAttr::name))
            argumentListForWalk.push_back(markOp.getSrc());
        });
  });
  for (const auto &arg : argumentListForWalk)
    updatePreviousType(arg);
  combineInferable();

  op_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      auto current = operand.get();
      processOperation(op, current);
      handleValueGroupForUse(op, current, &operand, operand.getOperandNumber());
    }
    for (Value result : op->getResults())
      updatePreviousType(result);
    if (isa<scf::YieldOp, scf::ConditionOp, scope::ReturnOp>(op)) {
      if (auto *parentOp = op->getParentOp()) {
        for (OpOperand &operand : op->getOpOperands())
          processOperation(parentOp, operand.get());
        for (Value result : parentOp->getResults())
          updatePreviousType(result);
        if (auto loopOp = dyn_cast<LoopLikeOpInterface>(parentOp)) {
          for (Value init : loopOp.getInits())
            updatePreviousType(init);
        }
      }
    }
    return WalkResult::advance();
  });
}

bool DimensionAnalyzer::processOperation(Operation *op, Value current) {
  LDBG("Processing operation: " << *op);
  startTransaction(op);
  auto processingResult =
      TypeSwitch<Operation *, bool>(op)
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
          .Case<hivm::MatmulOp, hivm::MixMatmulOp>([this](Operation *op) {
            processMatmulOp(op, isATransposed(op), isBTransposed(op));
            return true;
          })
          .Case<hivm::MmadL1Op>([this](auto op) {
            processMmadL1Op(op, isATransposed(op), isBTransposed(op));
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
          .Case<hivm::VCumsumOp, hivm::VCumprodOp, hivm::VCummaxOp,
                hivm::VCumminOp>([this](auto op) {
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
            if (utils::isAnnotationWithAttr(op, kTilingDimMappingAttrName)) {
              processReshapeOp(op);
            } else {
              processExpandShapeOpLeftmostNonUnit(op);
            }
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
              auto tilingDimMapping =
                  op->template getAttrOfType<DictionaryAttr>(
                      kTilingDimMappingAttrName);
              processTilingDimMapping(expandShapeOp, tilingDimMapping);
              return true;
            }
            return false;
          })
          .Default([&](Operation *op) {
            if (isElemwiseNaryOpImpl(op) ||
                isa_and_nonnull<CopyOpInterface>(op) ||
                utils::isAllocLikeOp(op) ||
                isa<memref::MemorySpaceCastOp, bufferization::ToTensorOp,
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
                    bufferization::ToMemrefOp
#else
                    bufferization::ToBufferOp
#endif
                    >(op)) {
              processParallelOp(op, current);
              return true;
            }
            return DimensionAnalyzerBase::processOperation(op, current);
          });
  LDBG("Finalizing transaction: " << *op);
  if (!finalizeTransaction())
    processingResult = false;
  return processingResult;
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

  auto outputArgs = getValueDimIndices(outputs[0]);
  auto joinCollapserIfMergeMutation = [this, &mergeMutation](int a, int b) {
    if (mergeMutation)
      joinCollapser(a, b);
  };
  for (auto input : inputs) {
    auto inputArgs = getValueDimIndices(input);
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
  createDummyRefIfNotExist({input, output});
  const auto &inputArgs = getValueDimIndices(input);
  auto outputArgs = getValueDimIndices(output);
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
  LDBG("Processing " << op->getName().getStringRef() << " " << op);
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

  auto srcArgs = getValueDimIndices(src);
  auto resArgs = getValueDimIndices(res);
  for (NamedAttribute dimMappingAttr : tilingDimMapping) {
    int srcDim;
    int resDim = cast<IntegerAttr>(dimMappingAttr.getValue()).getInt();
    llvm::to_integer(dimMappingAttr.getName(), srcDim);
    joinCollapser(srcArgs[srcDim], resArgs[resDim]);
  }
}

// [{32}] -> [{2}, 16]
// [{16}] -> [1, {16}]
// [{64}] -> [1, {4}, 16]
void DimensionAnalyzer::processExpandShapeOpLeftmostNonUnit(
    tensor::ExpandShapeOp op) {
  auto input = op.getSrc();
  auto output = op.getResult();
  auto outputType = op.getType();
  createDummyRefIfNotExist({input, output});
  auto inputArgs = getValueDimIndices(input);
  auto outputArgs = getValueDimIndices(output);
  auto reassoc = op.getReassociationIndices();
  SmallVector<std::pair<int64_t, int64_t>> toBeMerged;
  for (auto [inputIdx, indices] : llvm::enumerate(reassoc)) {
    int64_t targetIdx = indices[0];
    for (auto outputIdx : indices) {
      if (outputType.getDimSize(targetIdx) == 1)
        targetIdx = outputIdx;
    }
    if (outputType.getDimSize(targetIdx) % tilingSize != 0)
      continue;
    toBeMerged.emplace_back(targetIdx, inputIdx);
  }

  LDBG("Processing ExpandShapeOp " << op);
  for (auto [outIdx, inIdx] : toBeMerged) {
    LDBG("Connecting " << inIdx << "th input dim with " << outIdx
                       << "th output dim");
    joinCollapser(outputArgs[outIdx], inputArgs[inIdx]);
  }
}

void DimensionAnalyzer::processCollapseShapeOpLeftmostNonUnit(
    tensor::CollapseShapeOp op) {
  auto input = op.getSrc();
  auto output = op.getResult();
  auto inputType = op.getSrcType();
  createDummyRefIfNotExist({input, output});
  auto inputArgs = getValueDimIndices(input);
  auto outputArgs = getValueDimIndices(output);
  auto reassoc = op.getReassociationIndices();
  SmallVector<std::pair<int64_t, int64_t>> toBeMerged;
  for (auto [outputIdx, indices] : llvm::enumerate(reassoc)) {
    int64_t targetIdx = indices[0];
    int64_t targetOrder = indices[0];
    if (auto it =
            transposedDimMap.find(equivalentDsu_->find(inputArgs[targetOrder]));
        it != transposedDimMap.end())
      targetOrder = it->second;
    for (auto inputIdx : indices) {
      auto inputOrder = inputIdx;
      if (auto it = transposedDimMap.find(
              equivalentDsu_->find(inputArgs[inputOrder]));
          it != transposedDimMap.end())
        inputOrder = it->second;
      if (inputType.getDimSize(targetIdx) == 1 ||
          (inputType.getDimSize(inputIdx) != 1 && targetOrder > inputOrder)) {
        targetIdx = inputIdx;
        targetOrder = inputOrder;
      }
    }
    if (inputType.getDimSize(targetIdx) % tilingSize != 0)
      continue;
    toBeMerged.emplace_back(outputIdx, targetIdx);
  }

  LDBG("Processing CollapseShapeOp " << op);
  for (auto [outIdx, inIdx] : toBeMerged) {
    LDBG("Connecting " << inIdx << "th input dim with " << outIdx
                       << "th output dim");
    joinCollapser(outputArgs[outIdx], inputArgs[inIdx]);
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
  createDummyRefIfNotExist({input, output});
  auto inputArgs = getValueDimIndices(input);
  auto outputArgs = getValueDimIndices(output);
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
      joinCollapser(inputArgs[inputIdx[0]], outputArgs[outputIdx[0]]);
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
      joinCollapser(outputArgs[*filteredOutputIdx.begin()],
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

void DimensionAnalyzer::processMmadL1Op(hivm::MmadL1Op op, bool isTransposeA,
                                        bool isTransposeB) {
  Value operandA = op.getA();
  Value operandB = op.getB();
  Value mmadResult = op.getResult(0);
  createDummyRefIfNotExist({operandA, operandB});
  auto dimRefsA = getValueDimIndices(operandA);
  auto dimRefsB = getValueDimIndices(operandB);
  int kAxisIdxA = isTransposeA ? 0 : 1;
  int kAxisIdxB = isTransposeB ? 1 : 0;
  joinShape(dimRefsA[kAxisIdxA], dimRefsB[kAxisIdxB]);
  int mAxisIdxA = isTransposeA ? 1 : 0;
  int nAxisIdxB = isTransposeB ? 0 : 1;
  SmallVector<int64_t> resultDimRefs = {dimRefsA[mAxisIdxA],
                                        dimRefsB[nAxisIdxB]};
  dimIndices_.push_back(resultDimRefs);
  int64_t resultRefIdx = static_cast<int64_t>(dimIndices_.size() - 1);
  initCollapseOrVerify(mmadResult, resultRefIdx);
  for (Value val : op->getResults()) {
    processValue(val, mmadResult);
  }
}

void DimensionAnalyzer::startTransaction(Operation *op) {
  if (processingOperation)
    finalizeTransaction();
  processingOperation = op;
}

bool DimensionAnalyzer::finalizeTransaction() {
  auto *op = processingOperation;
  processingOperation = nullptr;
  SmallVector<int> compParIdx;
  SmallVector<int> compArgIdx;
  DenseSet<int64_t> argIndices;
  compParIdx.reserve(2 * (equivalentUpdates.size() + structuralUpdates.size()));
  compArgIdx.reserve(compParIdx.capacity());
  auto allUpdates = equivalentUpdates;
  allUpdates.append(structuralUpdates);
  if (allUpdates.empty())
    return false;
  for (auto [a, b] : allUpdates) {
    auto parA = structuralDsu_->find(a);
    auto parB = structuralDsu_->find(b);
    compParIdx.push_back(parA);
    compParIdx.push_back(parB);
    compArgIdx.push_back(a);
    compArgIdx.push_back(b);
    argIndices.insert(dimIdxToArgIdx_[a]);
    argIndices.insert(dimIdxToArgIdx_[b]);
    LDBG("Updates: " << a << "(" << parA << ") " << b << "(" << parB << ")");
  }

  llvm::sort(compParIdx);
  compParIdx.erase(llvm::unique(compParIdx), compParIdx.end());
  llvm::sort(compArgIdx);
  compArgIdx.erase(llvm::unique(compArgIdx), compArgIdx.end());

  mlir::detail::SimpleUnionFind transStructureDSU(compParIdx.size());
  mlir::detail::SimpleUnionFind transArgDSU(compArgIdx.size());
  for (auto [a, b] : allUpdates) {
    auto parA = structuralDsu_->find(a);
    auto parB = structuralDsu_->find(b);
    parA = llvm::lower_bound(compParIdx, parA) - compParIdx.begin();
    parB = llvm::lower_bound(compParIdx, parB) - compParIdx.begin();
    LDBG("Merging transStructureDSU: " << a << "(" << parA << ") " << b << "("
                                       << parB << ")");
    transStructureDSU.join(parA, parB);

    a = llvm::lower_bound(compArgIdx, a) - compArgIdx.begin();
    b = llvm::lower_bound(compArgIdx, b) - compArgIdx.begin();
    transArgDSU.join(a, b);
    LDBG("Merging transArgDSU: " << a << " " << b);
  }

  DenseSet<int> invalidMerges;
  SmallVector<DimensionIndex> indicesAfterTransaction(compParIdx.size());
  DimensionIndex repIdx(compArgIdx.size());

  for (int i = 0; i < static_cast<int>(compParIdx.size()); i++) {
    auto pIdx = transStructureDSU.find(i);
    indicesAfterTransaction[pIdx].push_back(compParIdx[i]);
  }
  for (auto &indices : indicesAfterTransaction) {
    for (size_t i = 0; i < indices.size(); i++) {
      for (size_t j = i + 1; j < indices.size(); j++) {
        if (exclusiveDimIdx[indices[i]].contains(indices[j])) {
          invalidMerges.insert(indices[i]);
          invalidMerges.insert(indices[j]);
        }
      }
    }
  }

  indicesAfterTransaction.clear();
  indicesAfterTransaction.resize(compArgIdx.size());
  for (int i = 0; i < static_cast<int>(compArgIdx.size()); i++) {
    auto pIdx = transArgDSU.find(i);
    indicesAfterTransaction[pIdx].push_back(compArgIdx[i]);
  }
  for (auto res : op->getResults()) {
    if (!isa<ShapedType>(res.getType()))
      continue;
    createDummyRefIfNotExist(res);
    for (auto arg : getValueDimIndices(res)) {
      auto parIdx = structuralDsu_->find(arg);
      if (!invalidMerges.contains(parIdx) ||
          !llvm::binary_search(compArgIdx, arg))
        continue;
      parIdx = llvm::lower_bound(compArgIdx, arg) - compArgIdx.begin();
      repIdx[transArgDSU.find(parIdx)] = compArgIdx[parIdx];
    }
  }
  for (const auto &[rIdx, indices] : llvm::enumerate(indicesAfterTransaction)) {
    DimensionIndex invalidIndices;
    for (size_t i = 0; i < indices.size(); i++) {
      auto parIdx = structuralDsu_->find(indices[i]);
      if (repIdx[rIdx] == indices[i] || !invalidMerges.contains(parIdx))
        continue;
      invalidIndices.push_back(indices[i]);
    }
    if (!invalidIndices.empty()) {
      invalidUpdates.emplace_back(repIdx[rIdx], invalidIndices);
      LDBG("Adding invalid updates: ("
           << structuralDsu_->find(repIdx[rIdx]) << ", "
           << utils::debugger::to_string(invalidIndices) << ") ");
    }
  }

  LDBG("argIndices: " << utils::debugger::to_string(argIndices));
  LDBG("compressedParIdx: " << utils::debugger::to_string(compParIdx));
  LDBG("InvalidMerges: " << utils::debugger::to_string(invalidMerges));

  for (auto [a, b] : equivalentUpdates) {
    auto parA = structuralDsu_->find(a);
    auto parB = structuralDsu_->find(b);
    if (!invalidMerges.contains(parA) && !invalidMerges.contains(parB))
      joinShape(a, b);
  }

  for (auto [a, b] : structuralUpdates) {
    auto parA = structuralDsu_->find(a);
    auto parB = structuralDsu_->find(b);
    if (!invalidMerges.contains(parA) && !invalidMerges.contains(parB))
      joinCollapser(a, b);
  }
  equivalentUpdates.clear();
  structuralUpdates.clear();
  return true;
}

void DimensionAnalyzer::combineInferable() {
  DimensionAnalyzerBase::combineInferable();
  for (const auto &arg : argumentList_) {
    auto allocOp = arg.getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      continue;
    LDBG("Combining alloc op " << allocOp);
    createDummyRefIfNotExist({allocOp.getResult()});
    auto allocRef = getValueDimIndices(allocOp.getResult());
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
      createDummyRefIfNotExist({memrefSource});
      auto memrefRef = getValueDimIndices(memrefSource);
      joinShape(memrefRef[constantIndex.value()], allocRef[allocIdx]);
    }
  }
}

void DimensionAnalyzer::markDimensions() {
  auto processSlice = [this](auto sliceOp) {
    if (!valueToDimIndicesIndex_.contains(sliceOp.getSource()))
      return;
    LDBG("Trying to mark this slice op " << sliceOp);
    llvm::SmallBitVector droppedDimsMask = sliceOp.getDroppedDims();
    auto origType = dyn_cast<ShapedType>(sliceOp.getSource().getType());
    auto sliceType = dyn_cast<ShapedType>(sliceOp.getResult().getType());
    createDummyRefIfNotExist({sliceOp.getSource(), sliceOp.getResult()});
    auto origRef = getValueDimIndices(sliceOp.getSource());
    auto sliceRef = getValueDimIndices(sliceOp.getResult());
    if (isa<tensor::InsertSliceOp>(sliceOp.getOperation())) {
      std::swap(origRef, sliceRef);
      std::swap(origType, sliceType);
    }
    size_t sliceIdx = 0;
    for (size_t i = 0; i < origRef.size(); ++i) {
      if (droppedDimsMask[i]) {
        tilingDimKindMapForCollapser[structuralDsu_->find(origRef[i])] =
            TilingDimensionKind::RankReduced;
        tilingDimKindMapForShape[equivalentDsu_->find(origRef[i])] =
            TilingDimensionKind::RankReduced;
        LDBG("Dim " << i << "(" << structuralDsu_->find(origRef[i])
                    << ") is marked as RankReduced");
      } else {
        if (isa<tensor::InsertSliceOp>(sliceOp.getOperation()) &&
            sliceType.getDimSize(sliceIdx) == 1) {
          tilingDimKindMapForCollapser[structuralDsu_->find(origRef[i])] =
              TilingDimensionKind::Reduce;
          tilingDimKindMapForShape[equivalentDsu_->find(origRef[i])] =
              TilingDimensionKind::Reduce;
          LDBG("Dim " << i << "(" << structuralDsu_->find(origRef[i])
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
      auto reduceSrcRef = getValueDimIndices(reduceOp.getSrc());
      auto reduceDstRef = getValueDimIndices(reduceOp.getDst()[0]);
      for (auto reduceDim : reduceOp.getReduceDims()) {
        tilingDimKindMapForCollapser[structuralDsu_->find(
            reduceSrcRef[reduceDim])] = TilingDimensionKind::Reduce;
        tilingDimKindMapForShape[equivalentDsu_->find(
            reduceSrcRef[reduceDim])] = TilingDimensionKind::Reduce;
        tilingDimKindMapForShape[equivalentDsu_->find(
            reduceDstRef[reduceDim])] = TilingDimensionKind::Reduce;
        LDBG("Reduced dim: " << equivalentDsu_->find(reduceSrcRef[reduceDim])
                             << " -> "
                             << equivalentDsu_->find(reduceDstRef[reduceDim]));
      }
    } else if (auto vbrcOp = dyn_cast<hivm::VBrcOp>(op)) {

    } else if (auto mmadOp = dyn_cast<hivm::MmadL1Op>(op)) {
      LDBG("Marking mmadL1 reduction axis: " << mmadOp);
      createDummyRefIfNotExist(mmadOp.getA());
      auto dimRefsA = getValueDimIndices(mmadOp.getA());
      int kAxisIdxA = mmadOp.getATranspose() ? 0 : 1;
      int64_t kAxisRootId = structuralDsu_->find(dimRefsA[kAxisIdxA]);
      tilingDimKindMapForCollapser[kAxisRootId] = TilingDimensionKind::Reduce;
    } else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      processSlice(insertOp);
    } else if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      processSlice(extractOp);
    } else if (auto vtransposeOp = dyn_cast<hivm::VTransposeOp>(op)) {
      markTransposedDim(vtransposeOp);
    } else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      if (!hacc::utils::isRegBasedArch(op->getParentOfType<ModuleOp>()))
        return;
      // FIXME: FIXPIPE intrinsic has the constraint that N has to be multiples
      // of 32 for COL_SPLIT. For compiler to lift this constraint, we need to
      // enhance Stride Align.
      LDBG("Trying to mark this alloc op " << allocOp);
      auto addressAttr = dyn_cast_or_null<hivm::AddressSpaceAttr>(
          allocOp.getType().getMemorySpace());
      auto alloc = allocOp.getMemref();
      if (addressAttr &&
          addressAttr.getAddressSpace() == hivm::AddressSpace::UB &&
          utils::getAnnotateOpWithAttr(alloc,
                                       hivm::HIVMTightlyCoupledBufferAttr::name)
              .has_value()) {
        auto shape = allocOp.getType().getShape();
        if (shape.size() == 2 && shape[1] < 32) {
          createDummyRefIfNotExist(alloc);
          auto args = getValueDimIndices(alloc);
          tilingDimKindMapForCollapser[structuralDsu_->find(args[1])] =
              TilingDimensionKind::InvalidColumnSplit;
          tilingDimKindMapForShape[equivalentDsu_->find(args[1])] =
              TilingDimensionKind::InvalidColumnSplit;
          LDBG("Invalid dim: " << structuralDsu_->find(args[1]) << "("
                               << equivalentDsu_->find(args[1]) << ")");
        }
      }
    }
  });
}

void DimensionAnalyzer::markTransposedDim(hivm::VTransposeOp op) {
  auto src = op.getSrc();
  auto dst = op.getDst();
  auto srcRef = getValueDimIndices(src);
  auto dstRef = getValueDimIndices(dst);
  auto perm = op.getPermutation();
  LDBG("Marking transposed dim: " << op);
  for (auto [dimIdx, parentIdx] : llvm::enumerate(dstRef)) {
    auto srcSolverIdx = equivalentDsu_->find(srcRef[perm[dimIdx]]);
    auto dstSolverIdx = equivalentDsu_->find(parentIdx);
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

void DimensionAnalyzer::transferDimMark() {
  op_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<annotation::MarkOp>([&](auto op) {
          if (op->hasAttr(kTilingDimMappingAttrName)) {
            transferDimMarkImpl(op);
          }
        })
        .Case<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
              tensor::ExtractSliceOp, tensor::InsertSliceOp, hivm::VBrcOp,
              hivm::VTransposeOp>([&](auto op) { transferDimMarkImpl(op); });
  });
}

template <typename IntegerRange>
void DimensionAnalyzer::transferDimMarkImpl(Value input, Value output,
                                            const IntegerRange &mutated) {
  createDummyRefIfNotExist({input, output});
  auto inputArgs = getValueDimIndices(input);
  auto outputArgs = getValueDimIndices(output);
  for (auto idx : mutated) {
    auto srcDim = equivalentDsu_->find(inputArgs[idx]);
    auto resDim = equivalentDsu_->find(outputArgs[idx]);
    LDBG("Checking if transposed dim of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = transposedDimMap.find(srcDim); it != transposedDimMap.end()) {
      LDBG("Successfully moved");
      transposedDimMap[resDim] = it->second;
    }
    LDBG("Checking if dimension kind of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end()) {
      LDBG("Successfully moved: " << static_cast<int>(it->getSecond()));
      tilingDimKindMapForShape[resDim] = it->second;
    }
  }
}

void DimensionAnalyzer::transferDimMarkImpl(annotation::MarkOp op) {
  auto expandShapeOp = op.getSrc().getDefiningOp<tensor::ExpandShapeOp>();
  auto tilingDimMapping =
      op->getAttrOfType<DictionaryAttr>(kTilingDimMappingAttrName);
  auto src = expandShapeOp.getSrc();
  auto res = expandShapeOp.getResult();

  auto srcArgs = getValueDimIndices(src);
  auto resArgs = getValueDimIndices(res);
  for (auto dimMappingAttr : tilingDimMapping) {
    int srcDim;
    int resDim = cast<IntegerAttr>(dimMappingAttr.getValue()).getInt();
    llvm::to_integer(dimMappingAttr.getName(), srcDim);
    srcDim = equivalentDsu_->find(srcArgs[srcDim]);
    resDim = equivalentDsu_->find(resArgs[resDim]);
    LDBG("Checking if transposed dim of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = transposedDimMap.find(srcDim); it != transposedDimMap.end()) {
      LDBG("Successfully moved");
      transposedDimMap[resDim] = it->second;
    }
    LDBG("Checking if dimension kind of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end()) {
      LDBG("Successfully moved");
      tilingDimKindMapForShape[resDim] = it->second;
    }
  }
}

void DimensionAnalyzer::transferDimMarkImpl(tensor::ExpandShapeOp op) {
  auto input = op.getSrc();
  auto output = op.getResult();
  auto outputType = op.getType();
  createDummyRefIfNotExist({input, output});
  auto inputArgs = getValueDimIndices(input);
  auto outputArgs = getValueDimIndices(output);
  auto reassoc = op.getReassociationIndices();
  LDBG("Transferring dimension marks: " << op);
  DenseMap<int64_t, int64_t> dimMap;
  SmallVector<std::pair<int64_t, int64_t>> toBeTransferred;
  for (auto [inputIdx, indices] : llvm::enumerate(reassoc)) {
    int64_t targetIdx = indices[0];
    for (auto outputIdx : indices) {
      if (outputType.getDimSize(targetIdx) == 1)
        targetIdx = outputIdx;
    }
    toBeTransferred.emplace_back(inputIdx, targetIdx);
    dimMap[inputIdx] = static_cast<int64_t>(targetIdx);
  }
  for (auto [inputIdx, targetIdx] : toBeTransferred) {
    LDBG("Dim " << inputIdx << " and dim " << targetIdx << " is mapped");
    auto srcDim = inputArgs[inputIdx];
    auto resDim = outputArgs[targetIdx];
    LDBG("Dim " << inputIdx << " and dim " << targetIdx << " is mapped");
    if (structuralDsu_->find(srcDim) != structuralDsu_->find(resDim))
      continue;
    srcDim = equivalentDsu_->find(srcDim);
    resDim = equivalentDsu_->find(resDim);
    LDBG("Checking if transposed dim of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = transposedDimMap.find(srcDim); it != transposedDimMap.end()) {
      LDBG("Successfully moved");
      transposedDimMap[resDim] = dimMap.at(it->second);
    }
    LDBG("Checking if dimension kind of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end()) {
      LDBG("Successfully moved: " << static_cast<int>(it->getSecond()));
      tilingDimKindMapForShape[resDim] = it->second;
    }
  }
}

void DimensionAnalyzer::transferDimMarkImpl(tensor::CollapseShapeOp op) {
  auto input = op.getSrc();
  auto output = op.getResult();
  auto inputType = op.getSrcType();
  createDummyRefIfNotExist({input, output});
  auto inputArgs = getValueDimIndices(input);
  auto outputArgs = getValueDimIndices(output);
  auto reassoc = op.getReassociationIndices();
  startTransaction(op);
  processCollapseShapeOpLeftmostNonUnit(op);
  finalizeTransaction();
  LDBG("Transferring dimension marks: " << op);
  DenseMap<int64_t, int64_t> dimMap;
  SmallVector<std::pair<int64_t, int64_t>> toBeTransferred;
  for (auto [outputIdx, indices] : llvm::enumerate(reassoc)) {
    int64_t targetIdx = indices[0];
    int64_t targetOrder = indices[0];
    if (auto it =
            transposedDimMap.find(equivalentDsu_->find(inputArgs[targetOrder]));
        it != transposedDimMap.end())
      targetOrder = it->second;
    for (auto inputIdx : indices) {
      auto inputOrder = inputIdx;
      dimMap[inputIdx] = static_cast<int64_t>(outputIdx);
      if (auto it = transposedDimMap.find(
              equivalentDsu_->find(inputArgs[inputOrder]));
          it != transposedDimMap.end())
        inputOrder = it->second;
      if (inputType.getDimSize(targetIdx) == 1 ||
          (inputType.getDimSize(inputIdx) != 1 && targetOrder > inputOrder)) {
        targetIdx = inputIdx;
        targetOrder = inputOrder;
      }
    }
    toBeTransferred.emplace_back(targetIdx, outputIdx);
    dimMap[targetIdx] = static_cast<int64_t>(outputIdx);
  }
  for (auto [targetIdx, outputIdx] : toBeTransferred) {
    LDBG("Dim " << targetIdx << " and dim " << outputIdx << " is mapped");
    auto srcDim = inputArgs[targetIdx];
    auto resDim = outputArgs[outputIdx];
    LDBG("Dim " << targetIdx << " and dim " << outputIdx << " is mapped");
    if (structuralDsu_->find(srcDim) != structuralDsu_->find(resDim))
      continue;
    srcDim = equivalentDsu_->find(srcDim);
    resDim = equivalentDsu_->find(resDim);
    LDBG("Checking if transposed dim of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = transposedDimMap.find(srcDim); it != transposedDimMap.end()) {
      LDBG("Successfully moved");
      transposedDimMap[resDim] = dimMap.at(it->second);
    }
    LDBG("Checking if dimension kind of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end()) {
      LDBG("Successfully moved: " << static_cast<int>(it->getSecond()));
      tilingDimKindMapForShape[resDim] = it->second;
    }
  }
}

void DimensionAnalyzer::transferDimMarkImpl(tensor::ExtractSliceOp op) {
  if (op.getDroppedDims().any())
    return;
  LDBG("Transferring dimension marks: " << op);
  transferDimMarkImpl(op.getSource(), op.getResult(),
                      getExtractOrInsertDim(op));
}

void DimensionAnalyzer::transferDimMarkImpl(tensor::InsertSliceOp op) {
  if (op.getDroppedDims().any())
    return;
  LDBG("Transferring dimension marks: " << op);
  transferDimMarkImpl(op.getSource(), op.getResult(),
                      getExtractOrInsertDim(op));
}

void DimensionAnalyzer::transferDimMarkImpl(hivm::VBrcOp op) {
  if (op->getNumResults() == 0u || op.getSrc().getType().isIntOrIndexOrFloat())
    return;
  LDBG("Transferring dimension marks: " << op);
  transferDimMarkImpl(op.getSrc(), op->getResult(0), op.getBroadcastDims());
}

void DimensionAnalyzer::transferDimMarkImpl(hivm::VTransposeOp op) {
  if (op->getNumResults() == 0u)
    return;
  LDBG("Transferring dimension marks: " << op);
  Value input = op.getSrc();
  Value output = op.getDst();
  auto perm = op.getPermutation();
  createDummyRefIfNotExist({input, output});
  auto inputArgs = getValueDimIndices(input);
  auto outputArgs = getValueDimIndices(output);
  for (int i = 0; i < static_cast<int>(inputArgs.size()); ++i) {
    auto srcDim = equivalentDsu_->find(inputArgs[perm[i]]);
    auto resDim = equivalentDsu_->find(outputArgs[i]);
    LDBG("Checking if dimension kind of " << srcDim << " is moved to "
                                          << resDim);
    if (auto it = tilingDimKindMapForShape.find(srcDim);
        it != tilingDimKindMapForShape.end()) {
      LDBG("Successfully moved: " << static_cast<int>(it->getSecond()));
      tilingDimKindMapForShape[resDim] = it->second;
    }
  }
}

void DimensionAnalyzer::joinShape(int a, int b) {
  if (processingOperation) {
    equivalentUpdates.emplace_back(a, b);
  } else {
    auto parA = structuralDsu_->find(a);
    auto parB = structuralDsu_->find(b);
    DimensionAnalyzerBase::joinShape(a, b);
    if (parA == parB)
      return;
    if (parA == structuralDsu_->find(a))
      std::swap(parA, parB);
    for (auto exIdx : exclusiveDimIdx[parA]) {
      exclusiveDimIdx[parB].insert(exIdx);
      exclusiveDimIdx[exIdx].erase(parA);
      exclusiveDimIdx[exIdx].insert(parB);
      assert(!exclusiveDimIdx[parB].contains(parB));
      assert(!exclusiveDimIdx[exIdx].contains(exIdx));
    }
    exclusiveDimIdx[parA].clear();
  }
}

void DimensionAnalyzer::joinCollapser(int a, int b) {
  if (processingOperation) {
    structuralUpdates.emplace_back(a, b);
  } else {
    auto parA = structuralDsu_->find(a);
    auto parB = structuralDsu_->find(b);
    DimensionAnalyzerBase::joinCollapser(a, b);
    if (parA == parB)
      return;
    if (parA == structuralDsu_->find(a))
      std::swap(parA, parB);
    for (auto exIdx : exclusiveDimIdx[parA]) {
      exclusiveDimIdx[parB].insert(exIdx);
      exclusiveDimIdx[exIdx].erase(parA);
      exclusiveDimIdx[exIdx].insert(parB);
      assert(!exclusiveDimIdx[parB].contains(parB));
      assert(!exclusiveDimIdx[exIdx].contains(exIdx));
    }
    exclusiveDimIdx[parA].clear();
  }
}

} // namespace detail
} // namespace hivm
} // namespace mlir
