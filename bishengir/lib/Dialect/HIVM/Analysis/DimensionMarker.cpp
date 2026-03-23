//===- DimensionMarker.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
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
          solverGroup_->join(curRef, argumentsRefPointer_.at(regionArg));
          solverGroup_->join(curRef, argumentsRefPointer_.at(res));
        } else {
          for (auto res : user->getResults()) {
            if (isa<ShapedType>(res.getType())) {
              createDummyRefIfNotExist({res});
              solverGroup_->join(curRef, argumentsRefPointer_.at(res));
              LDBG(res << " is mapped to " << utils::debugger::to_string(getArgumentRef(res)));
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
  SmallVector<Value> outputs(op.getResult());

  assert(outputs.size() <= 1 &&
         "result size must be 1 if tensor type and 0 if memref type");

  outputs.push_back(output);
  mergeValues({input}, outputs, getMutatedDims(op));
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
    } else if (auto vtransposeOp = dyn_cast<hivm::VTransposeOp>(op)) {
      auto srcRef = getArgumentRef(vtransposeOp.getSrc());
      for (auto[dimIdx, parentIdx] : llvm::enumerate(srcRef)) {
        transposedDimMap[solverCollapserElem_->find(parentIdx)] = dimIdx;
      }
    }
  });
}
} // namespace detail
} // namespace hivm
} // namespace mlir
