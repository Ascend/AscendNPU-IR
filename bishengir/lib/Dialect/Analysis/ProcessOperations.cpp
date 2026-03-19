//===- ProcessOperations.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Analysis/DimensionAnalyzer.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Utils/Util.h"

#include <numeric>

using namespace mlir;
using namespace mlir::utils::debugger;
using namespace mlir::reshape_utils;
using namespace mlir::tensor::reshape_utils;

#define DEBUG_TYPE "dimension-analyzer-hfusion-process-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace detail {

// Step 2: BFS argument list and creating segments
void DimensionAnalyzerBase::processBFS() {
  LLVM_DEBUG(
    llvm::dbgs() << DEBUG_LINE_BEG("Flatten-processBFS");
    llvm::dbgs() << "argumentList_: \n";
  );
  std::queue<Value> bfsQueue;
  for (const auto &arg : argumentList_) {
    LLVM_DEBUG(llvm::dbgs() << arg << "\n");
    updatePreviousType(arg);
    bfsQueue.push(arg);
  }
  DenseSet<Value> visited(argumentList_.begin(), argumentList_.end());
  combineInferable();

  size_t processedCount = 0;
  while (!bfsQueue.empty()) {
    Value current = bfsQueue.front();
    if (options.registerBased) separateMemref(current);
    bfsQueue.pop();

    LLVM_DEBUG({
      llvm::dbgs() << "\n════════════════════════════════════════\n";
      llvm::dbgs() << "BFS Step #" << (++processedCount)
                  << " | Queue Size: " << bfsQueue.size()
                  << "\n";
      llvm::dbgs() << "Processing: [" << current << "]\n";
      if (current.getDefiningOp()) {
        llvm::dbgs() << "Defined by: " << *current.getDefiningOp() << "\n";
      }
      llvm::dbgs() << "════════════════════════════════════════\n";
    });

    SmallVector<Operation*> users(current.user_begin(), current.user_end());

    LLVM_DEBUG({
      if (users.empty()) {
        llvm::dbgs() << "  └─ No users found (leaf node)\n";
      } else {
        llvm::dbgs() << "  └─ Found " << users.size() << " user(s):\n";
      }
    });

    for (size_t i = 0; i < users.size(); ++i) {
      Operation *user = users[i];

      LLVM_DEBUG({
        llvm::dbgs() << "    ├─ User [" << (i+1) << "/" << users.size() << "]:\n";
        llvm::dbgs() << "    │   Operation: " << *user << "\n";
        llvm::dbgs() << "    │   Location: " << user->getLoc() << "\n";
      });

      processOperation(user, current);

      auto results = user->getResults();
      LLVM_DEBUG({
        if (!results.empty()) {
          llvm::dbgs() << "    │   Results (" << results.size() << "):\n";
        }
      });

      for (size_t j = 0; j < results.size(); ++j) {
        Value result = results[j];
        updatePreviousType(result);

        LLVM_DEBUG({
          llvm::dbgs() << "    │   ├─ Result [" << (j+1) << "]: ["
                      << result << "]\n";
        });

        if (visited.insert(result).second) {
          bfsQueue.push(result);
          LLVM_DEBUG({
            llvm::dbgs() << "    │   │   └─ ADDED to BFS queue (new)\n";
          });
        } else {
          LLVM_DEBUG({
            llvm::dbgs() << "    │   │   └─ Already visited (skipped)\n";
          });
        }
      }

      if (auto loopLikeOp = dyn_cast<LoopLikeOpInterface>(user)) {
        LLVM_DEBUG({
          llvm::dbgs() << "    │   └─ Loop-like operation detected\n";
        });

        auto iterArgs = loopLikeOp.getRegionIterArgs();
        LLVM_DEBUG({
          if (!iterArgs.empty()) {
            llvm::dbgs() << "    │       Iteration args (" << iterArgs.size() << "):\n";
          }
        });

        for (size_t k = 0; k < iterArgs.size(); ++k) {
          Value iterArg = iterArgs[k];
          updatePreviousType(iterArg);

          LLVM_DEBUG({
            llvm::dbgs() << "    │       ├─ IterArg [" << (k+1) << "]: ["
                        << iterArg << "]\n";
          });

          if (visited.insert(iterArg).second) {
            bfsQueue.push(iterArg);
            LLVM_DEBUG({
              llvm::dbgs() << "    │       │   └─ ADDED to BFS queue\n";
            });
          } else {
            LLVM_DEBUG({
              llvm::dbgs() << "    │       │   └─ Already visited\n";
            });
          }
        }
      }

      if (auto terminatorOp = dyn_cast<RegionBranchTerminatorOpInterface>(user)) {
        auto parentOp = terminatorOp->getParentOp();

        LLVM_DEBUG({
          llvm::dbgs() << "    │   └─ Region terminator detected\n";
          llvm::dbgs() << "    │       Parent op: " << *parentOp << "\n";
        });

        processOperation(parentOp, current);

        auto parentResults = parentOp->getResults();
        LLVM_DEBUG({
          if (!parentResults.empty()) {
            llvm::dbgs() << "    │       Parent results ("
                        << parentResults.size() << "):\n";
          }
        });

        for (size_t l = 0; l < parentResults.size(); ++l) {
          Value result = parentResults[l];
          if (options.registerBased) separateMemref(result);
          updatePreviousType(result);

          LLVM_DEBUG({
            llvm::dbgs() << "    │       ├─ Result [" << (l+1) << "]: ["
                        << result << "]\n";
          });

          if (visited.insert(result).second) {
            bfsQueue.push(result);
            LLVM_DEBUG({
              llvm::dbgs() << "    │       │   └─ ADDED to BFS queue\n";
            });
          } else {
            LLVM_DEBUG({
              llvm::dbgs() << "    │       │   └─ Already visited\n";
            });
          }
        }
      }

      LLVM_DEBUG({
        if (i == users.size() - 1) {
          llvm::dbgs() << "    └─ End of users for [" << current << "]\n";
        }
      });
    }
  }
  LLVM_DEBUG(
    llvm::dbgs() << DEBUG_LINE_END("Flatten-processBFS");
  );
  LLVM_DEBUG(
    llvm::dbgs() << DEBUG_LINE_BEG("Flatten-After-processBFS");
    llvm::dbgs() << "solverSegments_:\n";
    solverSegments_->dump();
    dumpArgumentsRefPointer();
    dumpArgumentsRef();
    dumpIsConnected();
    llvm::dbgs() << DEBUG_LINE_END("Flatten-After-processBFS");
  );
}

bool DimensionAnalyzerBase::processOperation(Operation *op, Value current) {
  if (auto concatOp = dyn_cast<tensor::ConcatOp>(op)) {
    processConcatOp(concatOp);
  } else if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
    processPadOp(padOp);
  } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    processExtractSliceOp(extractSliceOp);
  } else if (auto insertOp = dyn_cast<tensor::InsertOp>(op)) {
    processInsertOp(insertOp);
  } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    processInsertSliceOp(insertSliceOp);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    processIfOp(ifOp);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    processForOp(forOp);
  } else {
    if (isContainerAllocator(op)) {
      processParallelOp(op, current);
    } else {
      return false;
    }
  }
  return true;
}

void DimensionAnalyzerBase::processIfOp(scf::IfOp op) {
  LDBG("Processing IfOp " << op);

  scf::YieldOp thenYield = op.thenYield();
  for (auto [ifResult, thenOperand] :
       llvm::zip_equal(op.getResults(), thenYield.getOperands())) {
    createDummyRefIfNotExist({thenOperand});
    processValue(ifResult, thenOperand);
  }

  if (!op.elseBlock())
    return;

  scf::YieldOp elseYield = op.elseYield();
  for (auto [ifResult, elseOperand] :
       llvm::zip_equal(op.getResults(), elseYield.getOperands())) {
    createDummyRefIfNotExist({elseOperand});
    processValue(ifResult, elseOperand);
  }
}

void DimensionAnalyzerBase::processForOp(scf::ForOp op) {
  LDBG("Processing ForOp " << op);
  for (const auto &[regionArg, initArg] :
       zip_equal(op.getRegionIterArgs(), op.getInitArgs())) {
    createDummyRefIfNotExist({regionArg, initArg});
    processValue(regionArg, initArg);
  }
  for (const auto &[resultOpr, yieldOpr] :
       zip_equal(op->getResults(), op.getYieldedValues())) {
    createDummyRefIfNotExist({resultOpr, yieldOpr});
    processValue(resultOpr, yieldOpr);
  }
}

void DimensionAnalyzerBase::processParallelOp(Operation *op, Value current) {
  LDBG("Processing parellel op " << *op);
  createDummyRefIfNotExist({current});
  for (Value v : op->getOperands()) {
    processValue(v, current);
  }
  for (Value v : op->getResults()) {
    processValue(v, current);
  }
}

void DimensionAnalyzerBase::processValue(Value v, Value current) {
  if (v == current)
    return;
  LLVM_DEBUG(
    llvm::dbgs() << "Trying to bind two values \n"
                 << "left  = " << v << "\n"
                 << "right = " << current << "\n";
  );
  size_t vRank = utils::getShapeRank(v).value_or(0);
  size_t currentRank = utils::getShapeRank(current).value_or(0);
  // Can actually assert the shape too if the rank is the same
  if (vRank != currentRank)
    return;

  collapsePropagateOrVerify(v, current);
}

size_t
DimensionAnalyzerBase::processDecreasingDimensions(ArrayRef<int64_t> inputArgs,
                                                   ArrayRef<int64_t> dimensions,
                                                   const Value &output) {
  size_t outputRank = utils::getShapeRank(output).value_or(0);
  LLVM_DEBUG(llvm::dbgs() << "\nProcess decreasing dims, output: " << outputRank
                          << "\n");
  assert(inputArgs.size() == outputRank + dimensions.size());
  DimensionIndex outputArgs;
  outputArgs.reserve(outputRank);
  SmallVector<int64_t> sortedDimensions = {dimensions.begin(),
                                           dimensions.end()};
  llvm::sort(sortedDimensions);

  int64_t inputRank = static_cast<int64_t>(inputArgs.size());
  std::vector<bool> isReduced(inputRank, false);
  for (int64_t dim : dimensions) {
    if (dim >= 0 && dim < inputRank) {
      isReduced[dim] = true;
    }
  }

  for (int64_t i = 0; i < inputRank - 1; ++i) {
    if (isReduced[i] != isReduced[i + 1]) {
      disconnect(inputArgs[i], inputArgs[i + 1]);
      LDBG("Disconnected boundary via bit-vector: " << i << " - " << (i + 1));
    }
  }

  const auto *dimPtr = sortedDimensions.begin();
  for (int64_t i = 0; i < static_cast<int64_t>(inputArgs.size()); ++i) {
    if (dimPtr != sortedDimensions.end() && *dimPtr == i) {
      // Skip this dimension as it's being reduced
      ++dimPtr;
    } else {
      outputArgs.push_back(inputArgs[i]);
      LLVM_DEBUG(llvm::dbgs() << outputArgs.back() << " ";);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\n";);

  assert(outputArgs.size() == outputRank);
  argumentsRef_.push_back(std::move(outputArgs));
  LLVM_DEBUG(llvm::dbgs() << "New argumentsRef_ " << argumentsRef_.size() - 1
                          << "\n";);
  return argumentsRef_.size() - 1;
}

size_t DimensionAnalyzerBase::processPermutation(ArrayRef<int64_t> inputArgs,
                                                 ArrayRef<int64_t> perm,
                                                 const Value &output) {
  auto [outputRank, shapeOut] = utils::getValueShapeInfo(output).value_or(
      std::make_pair(0, DimensionIndex{}));
  DimensionIndex outputArgs(outputRank, -1);
  for (int i = 0; i < static_cast<int>(inputArgs.size()); ++i) {
    outputArgs[i] = inputArgs[perm[i]];
  }
  argumentsRef_.push_back(std::move(outputArgs));

  std::vector<int> invPerm(perm.size());
  for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
    invPerm[perm[i]] = i;
  }

  for (int j = 0; j < static_cast<int>(inputArgs.size()) - 1; ++j) {
    int posCurr = invPerm[j];
    int posNext = invPerm[j + 1];

    if (posNext != posCurr + 1) {
      disconnect(inputArgs[j], inputArgs[j + 1]);
      LDBG("Disconnected original neighbors: " << j << " and " << (j + 1));
    }
  }
  return argumentsRef_.size() - 1;
}

void DimensionAnalyzerBase::processMatmulOp(Operation *op, bool isTransposeA,
                                            bool isTransposeB) {
  auto matmulOp = dyn_cast<DestinationStyleOpInterface>(op);

  auto inputs = matmulOp.getDpsInputs();
  assert(matmulOp.getDpsInits().size() == 1);
  Value output = matmulOp.getDpsInits()[0];

  auto arg0 = getArgumentRefOrCreateDummy(inputs[0]);
  auto arg1 = getArgumentRefOrCreateDummy(inputs[1]);

  // When isTransposeA = false, isTransposeB = false:
  // A = [MxK], B = [KxN], C = [MxN]
  // When isTransposeA = true, isTransposeB = false:
  // A = [KxM], B = [KxN], C = [MxN]
  // When isTransposeA = false, isTransposeB = true:
  // A = [MxK], B = [NxK], C = [MxN]
  int mDimIdx = isTransposeA ? 1 : 0;
  int nDimIdx = isTransposeB ? 0 : 1;

  argumentsRef_.push_back({arg0[mDimIdx] /* this refers to the M */,
                           arg1[nDimIdx] /* this refers to the N */});

  initCollapseOrVerify(output, static_cast<int64_t>(argumentsRef_.size() - 1));
  for (Value result : op->getResults()) {
    processValue(result, output);
  }
}

void DimensionAnalyzerBase::processBatchMatmulOp(
    linalg::BatchMatmulOp batchMatmulOp) {
  auto inputs = batchMatmulOp.getDpsInputs();
  assert(batchMatmulOp.getDpsInits().size() == 1);
  Value output = batchMatmulOp.getDpsInits()[0];

  // A = [PxQxR], B = [PxRxS], C = [PxQxS]
  auto arg0 = getArgumentRefOrCreateDummy(inputs[0]);
  auto arg1 = getArgumentRefOrCreateDummy(inputs[1]);
  argumentsRef_.push_back({arg0[0] /* this refers to the P */,
                           arg0[1] /* this refers to the Q */,
                           arg1[2] /* this refers to the S */});

  initCollapseOrVerify(output, static_cast<int64_t>(argumentsRef_.size() - 1));
  for (Value result : batchMatmulOp->getResults())
    processValue(result, output);

  disconnect(arg0[0], arg0[1]);
  disconnect(arg1[0], arg1[1]);
}

void DimensionAnalyzerBase::processConcatOp(tensor::ConcatOp concatOp) {
  LDBG("Processing concat " << concatOp);
  auto dim = concatOp.getDim();
  auto res = concatOp.getResult();
  auto resultShape = utils::getShape(res.getType());
  auto rank = resultShape.size();
  LDBG(res);
  for (auto opr : concatOp.getOperands()) {
    if (utils::getShapeRank(opr.getType()).value_or(0) != resultShape.size())
      continue;
    auto oprArgRef = getArgumentRefOrCreateDummy(opr);
    auto argConcat = getArgumentRefOrCreateDummy(res);
    LDBG(opr << ": " << utils::debugger::to_string(oprArgRef));
    for (unsigned i = 0; i < rank; ++i) {
      if (i != dim) {
        isConnected_[argConcat[i]].elementKind =
            (resultShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation);
        // Shape elem has type inference, can only be joined if the shape is
        // exactly the same
        joinShape(argConcat[i], oprArgRef[i]);
      } else {
        isConnected_[argConcat[i]].elementKind = ElementKind::HasMutation;
        // Can be binded as far as the structure is the same
        joinCollapser(argConcat[i], oprArgRef[i]);
      }
    }
  }
  // No need to disconnect anything, handled by elementKind resolver
}

void DimensionAnalyzerBase::processPadOp(tensor::PadOp padOp) {
  auto padSrc = padOp.getSource();
  auto padHigh = padOp.getStaticHigh();
  auto padLow = padOp.getStaticLow();
  auto padResult = padOp.getResult();
  auto rank = padOp.getType().getRank();
  auto argPadSrc = getArgumentRefOrCreateDummy(padSrc);
  auto argPadResult = getArgumentRefOrCreateDummy(padResult);
  auto srcShape = utils::getShape(padSrc.getType());
  for (int i = 0; i < rank; i++) {
    if (padHigh[i] == 0 && padLow[i] == 0) {
      // Connect;
      joinShape(argPadSrc[i], argPadResult[i]);
      isConnected_[argPadSrc[i]].elementKind =
          srcShape[i] == 1 ? ElementKind::Unit : ElementKind::NoMutation;
      LDBG("From the pad: "
           << static_cast<int64_t>(isConnected_[argPadSrc[i]].elementKind));
    } else {
      joinCollapser(argPadSrc[i], argPadResult[i]);
      isConnected_[argPadSrc[i]].elementKind = ElementKind::HasMutation;
    }
  }
}

template <class T, typename>
void DimensionAnalyzerBase::processSlicingOp(T slicingOp) {
  // Category superview and subview
  auto src = slicingOp.getSource();
  auto res = slicingOp.getResult();
  Value superview, subview;
  if (std::is_same_v<T, tensor::ExtractSliceOp>) {
    superview = src;
    subview = res;
  } else {
    superview = res;
    subview = src;
  }
  SmallVector<OpFoldResult> superviewShape;
  if (auto expandOp =
          superview.template getDefiningOp<tensor::ExpandShapeOp>()) {
    superviewShape = expandOp.getMixedOutputShape();
  } else {
    superviewShape = llvm::map_to_vector(
        utils::getShape(superview.getType()),
        [&slicingOp](int64_t elementShape) -> OpFoldResult {
          return getAsIndexOpFoldResult(slicingOp.getContext(), elementShape);
        });
  }

  auto subviewShape = slicingOp.getMixedSizes();
  auto subviewStride = slicingOp.getMixedStrides();
  // for reduce dim
  auto droppedDims = slicingOp.getDroppedDims();
  auto argRef = getArgumentRef(superview);
  auto rank = superviewShape.size();
  auto superviewRef = getArgumentRef(superview);
  auto subviewRef = getArgumentRef(subview);

  SmallVector<int64_t, 8> subviewShapeSize(superviewShape.size());
  SmallVector<int64_t, 8> subviewStrides(superviewShape.size());
  int64_t subviewAxis = 0;
  for (size_t axis = 0; axis < rank; ++axis) {
    auto subviewStrideVal =
        getConstantIntValue(subviewStride[axis]).value_or(ShapedType::kDynamic);
    auto subviewShapeVal =
        getConstantIntValue(subviewShape[axis]).value_or(ShapedType::kDynamic);
    subviewStrides[axis] = subviewStrideVal;
    subviewShapeSize[axis] = subviewShapeVal;
    if (droppedDims[axis]) {
      LDBG("droppedDims " << droppedDims[axis]);
      continue;
    }
    LDBG("superviewRef " << axis << " subviewRef " << subviewAxis);
    joinCollapser(superviewRef[axis], subviewRef[subviewAxis]);
    subviewAxis++;
  }

  if (rank <= 1)
    return;

  auto getContiguousAxes = [&]() -> BitVector {
    BitVector ret;
    ret.resize(rank, true);

    for (size_t axis = 1; axis < rank; ++axis) {
      //  superview should be continuous for dropped dim
      int64_t staticRes;
      if (droppedDims[axis]) {
        LDBG("droppedDims " << droppedDims[axis]);
        staticRes = 1;
      } else {
        staticRes = getConstantIntValue(subviewShape[axis])
                        .value_or(ShapedType::kDynamic);
      }
      auto staticSrc = getConstantIntValue(superviewShape[axis])
                           .value_or(ShapedType::kDynamic);
      // If it's dynamic its undeterminable
      if (ShapedType::isDynamic(subviewStrides[axis]) ||
          ShapedType::isDynamic(subviewStrides[axis - 1]) ||
          ShapedType::isDynamic(staticRes) ||
          ShapedType::isDynamic(staticSrc)) {
        ret[axis] = false;
        LDBG("ret " << axis << " is " << ret[axis]);
        continue;
      }
      // First dimension is always contiguous
      if (!ret.empty()) {
        ret[0] = true;
      }
      // Check if its contiguous
      if (staticRes * subviewStrides[axis] !=
          staticSrc * subviewStrides[axis - 1])
        ret[axis] = false;
      LDBG("ret " << axis << " is " << ret[axis]);
    }
    return ret;
  };
  auto contiguousMask = getContiguousAxes();
  separateGroup(superview, contiguousMask, subviewShapeSize);
}

void DimensionAnalyzerBase::processInsertOp(tensor::InsertOp insertOp) {
  auto dest = insertOp.getDest();
  auto res = insertOp.getResult();
  createDummyRefIfNotExist({res});
  createDummyRefIfNotExist({dest});
  assert(res.getType().getRank() == dest.getType().getRank());
  collapsePropagateOrVerify(res, dest);
}

void DimensionAnalyzerBase::processInsertSliceOp(
    tensor::InsertSliceOp insertSliceOp) {
  auto dest = insertSliceOp.getDest();
  auto res = insertSliceOp.getResult();
  auto src = insertSliceOp.getSource();
  createDummyRefIfNotExist({res, dest, src});
  assert(res.getType().getRank() == dest.getType().getRank());
  collapsePropagateOrVerify(res, dest);
  LDBG("\nProcessing insert slice ----");
  processSlicingOp(insertSliceOp);
}

void DimensionAnalyzerBase::processExtractSliceOp(
    tensor::ExtractSliceOp extractSliceOp) {
  auto res = extractSliceOp.getResult();
  auto src = extractSliceOp.getSource();
  createDummyRefIfNotExist({res, src});
  LDBG("\nProcessing extract slice ----");
  processSlicingOp(extractSliceOp);
}
} // namespace detail
} // namespace mlir
