//===- DimensionMarker.cpp ------------------------------------------------===//
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

#define DEBUG_TYPE "dimension-analyzer-marker"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace detail {

void DimensionAnalyzerBase::dumpModuleOP() const {
  LLVM_DEBUG(llvm::dbgs() << *op_ << "\n";);
}

void DimensionAnalyzerBase::dumpArgumentsRef() {
  LLVM_DEBUG(
      int numVal = static_cast<int>(argumentsRef_.size()); if (numVal == 0) {
        llvm::dbgs() << "argumentsRef_.back(): [] (empty)\n";
        return;
      } llvm::dbgs() << "[Flatten] argumentsRef_ of size "
                     << numVal << " are:\n";
      for (int vi = 0; vi < numVal; ++vi) {
        auto &shapeIdx = argumentsRef_[vi];
        int rank = static_cast<int>(shapeIdx.size());
        if (rank == 0) {
          return;
        }
        llvm::dbgs() << "argumentsRef_[" << vi << "] of rank " << rank
                     << " = [";
        for (int si = 0; si < rank - 1; ++si) {
          llvm::dbgs() << shapeIdx[si] << ", ";
        }
        llvm::dbgs() << shapeIdx[rank - 1] << "]\n";
      });
}

void DimensionAnalyzerBase::dumpIsConnected() {
  LLVM_DEBUG(
      int numDim = static_cast<int>(isConnected_.size());
      llvm::dbgs() << "[Flatten] isConnected_ of size " << numDim << " are:\n";
      llvm::dbgs()
      << "                  \t[elmentKind, leftConnected, rightConnected]\n";
      for (int di = 0; di < numDim; ++di) {
        auto &connInfo = isConnected_[di];
        llvm::dbgs() << "isConnected_[" << di << "] = \t["
                     << connInfo.elementKind << ", "
                     << (connInfo.leftConnected ? "true" : "false") << ", "
                     << (connInfo.rightConnected ? "true" : "false") << "]\n";
      });
}

void DimensionAnalyzerBase::dumpArgumentsRefPointer() {
  LLVM_DEBUG(auto &map = argumentsRefPointer_;
             llvm::dbgs() << "[Flatten] argumentsRefPointer_(" << map.size()
                          << " entries):\n";
             for (const auto &entry
                  : map) {
               llvm::dbgs() << "[" << entry.second << "]: ";
               if (entry.first) {
                 entry.first.print(llvm::dbgs());
               } else {
                 llvm::dbgs() << "(null)";
               }
               llvm::dbgs() << "\n";
             });
}

bool ExtendedUnionFind::join(int a, int b) {
  assert(shape_[a] != kUndefinedShaped);
  assert(shape_[b] != kUndefinedShaped);
  assert(minParentIndex_.size() == parent_.size());
  assert(std::max(a, b) < ssize_t(parent_.size()));
  a = find(a);
  b = find(b);
  if (a != b) {
    if (parent_[a] > parent_[b])
      std::swap(a, b);
    parent_[a] += parent_[b];
    minIndex[a] = std::min(minIndex[b], minIndex[a]);
    // Derive min parent index
    minParentIndex_[a] = std::min(minParentIndex_[a], minParentIndex_[b]);
    assert(minParentIndex_[a] != kMaxDimPos &&
           "Inconsistency found when assigning parent");
    // Derive shape
    if (shape_[a] != ShapedType::kDynamic &&
        shape_[b] != ShapedType::kDynamic) {
      // if both is static, then verify
      if (shape_[a] != shape_[b])
        return false;
    }
    if (shape_[b] != ShapedType::kDynamic) {
      // if b is static, then propagate to a
      shape_[a] = shape_[b];
    }
    parent_[b] = a;
  }
  return true;
}

void ExtendedUnionFind::allocateMinimum(size_t n) {
  UnionFindBase::allocateMinimum(n);
  if (n + 1 > shape_.size()) {
    shape_.resize(n + 1, kUndefinedShaped);
    minParentIndex_.resize(n + 1, kMaxDimPos);
  }
}

std::pair<DimensionPosition, int64_t>
ExtendedUnionFind::getMinParentAndShapePair(int a) {
  auto parent = find(a);
  return std::make_pair(minParentIndex_[parent], shape_[parent]);
}

/// This function tries to combine extra information for shapes, for example
/// tensor.empty and its tensor.dim components. This is useful for minimum
/// information in dynamic shape context
void DimensionAnalyzerBase::combineInferable() {
  for (const auto &arg : argumentList_) {
    combineEmptyOp(arg);
  }
}

void DimensionAnalyzerBase::combineEmptyOp(Value arg) {
  auto emptyOp = dyn_cast_if_present<tensor::EmptyOp>(arg.getDefiningOp());
  if (!emptyOp) {
    return;
  }

  LDBG("Combining empty op " << emptyOp);
  auto emptyRef = getArgumentRef(emptyOp.getResult());
  auto mixEmptyShape = emptyOp.getMixedSizes();

  for (auto [emptyIdx, el] : llvm::enumerate(mixEmptyShape)) {
    // Skip constant values
    if (getConstantIntValue(el).has_value()) {
      return;
    }

    auto sizeDefiner = cast<Value>(el);
    auto dimOp =
        dyn_cast_if_present<tensor::DimOp>(sizeDefiner.getDefiningOp());
    if (!dimOp) {
      return;
    }

    LDBG("Found dim op " << dimOp);
    linkDimToEmpty(dimOp, emptyRef[emptyIdx]);
  }
}

void DimensionAnalyzerBase::linkDimToEmpty(tensor::DimOp dimOp,
                                           int64_t emptyRefElement) {
  auto constantIndex = dimOp.getConstantIndex();
  if (!constantIndex.has_value()) {
    return;
  }
  // Check if the option is on
  if (!bindUsingTensorDim) {
    return;
  }
  auto tensorSource = dimOp.getSource();
  auto tensorRef = getArgumentRefOrCreateDummy(tensorSource);
  joinShape(tensorRef[constantIndex.value()], emptyRefElement);
}

/// \param values this loops through all value and will create new terminal
/// on-the-fly
void DimensionAnalyzerBase::createDummyRefIfNotExist(ArrayRef<Value> values) {
  for (auto curVal : values) {
    if (argumentsRefPointer_.contains(curVal))
      continue;
    LDBG("[Create] Creating dummy for value: " << curVal);
    // init elements
    auto [rank, shape] = utils::getValueShapeInfo(curVal).value_or(
        std::make_pair(0, DimensionShape{}));
    int startingIdx = allocateArguments(rank, shape);
    argumentsRef_.push_back(DimensionShape(shape));
    std::iota(argumentsRef_.back().begin(), argumentsRef_.back().end(),
              startingIdx);
    LLVM_DEBUG(
      auto arg = argumentsRef_.back();
      auto argSize = arg.size();
      if(arg.size() != 0) {
        llvm::dbgs() << "argumentsRef_.back(): [";
        for (size_t j = 0; j < argSize - 1; ++j) {
          llvm::dbgs() << arg[j] << ", ";
        }
        llvm::dbgs() << arg[argSize-1] << "]\n";
      }
    );
    initCollapseOrVerify(curVal,
                         static_cast<int64_t>(argumentsRef_.size() - 1));
  }
}

void DimensionAnalyzerBase::updatePreviousType(const Value &val) {
  auto curType = dyn_cast<ShapedType>(val.getType());
  if (curType)
    updatePreviousType(val, curType);
  else {
    LLVM_DEBUG(llvm::dbgs() << val << " is not inittable\n";);
  }
}

void DimensionAnalyzerBase::updatePreviousType(const Value &val,
                                               const ShapedType &curType) {
  if (previousType_.contains(val)) {
    assert(previousType_[val] == curType);
  }
  previousType_[val] = curType;
}

// Mark all value that is the result of collapse
void DimensionAnalyzerBase::collapsePropagateOrVerify(Operation *op,
                                                      const Value &refVal) {
  const auto tmpVal = argumentsRefPointer_.at(refVal);
  for (const Value newVal : op->getResults()) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Propagating " << newVal << " " << tmpVal << "\n";);
    if (argumentsRefPointer_.contains(newVal)) {
      solverSegments_->join(argumentsRefPointer_.at(newVal), tmpVal);
      return;
    }
    argumentsRefPointer_[newVal] = tmpVal;
  }
}

void DimensionAnalyzerBase::collapsePropagateOrVerify(const Value &newVal,
                                                      const Value &arg) {
  const auto tmpVal = argumentsRefPointer_.at(arg);
  LLVM_DEBUG(llvm::dbgs() << "Propagating Value [" << newVal << "] with "
                          << "the argIdx = " << tmpVal << ", same as "
                          << "Value [" << arg << "]\n";);
  if (argumentsRefPointer_.contains(newVal)) {
    solverSegments_->join(argumentsRefPointer_.at(newVal), tmpVal);
    return;
  }
  argumentsRefPointer_[newVal] = tmpVal;
}

void DimensionAnalyzerBase::initCollapseOrVerify(const Value &val,
                                                 int64_t refPtr) {
  LLVM_DEBUG(llvm::dbgs() << "Assigning Value [" << val << "] with "
                          << "the argIdx = " << refPtr << "\n";);
  if (argumentsRefPointer_.contains(val)) {
    LLVM_DEBUG(llvm::dbgs() << "Init has been done previously\n";);
    solverSegments_->join(argumentsRefPointer_.at(val), refPtr);
    return;
  }
  argumentsRefPointer_[val] = refPtr;
}

// Step 3: Unifying groups of segments
void DimensionAnalyzerBase::unifyGroups() {
  for (int ref = 0; ref < static_cast<int>(argumentsRef_.size()); ++ref) {
    LLVM_DEBUG(llvm::dbgs() << "\nUnifying ---> " << ref << " \n";);
    auto parIdx = solverSegments_->find(ref);
    if (parIdx == ref)
      continue;

    const auto &par = argumentsRef_[parIdx];
    LDBG("Parent index (" << parIdx << ") - child Index(" << ref << "): "
                          << par.size() << " " << argumentsRef_[ref].size());
    assert(par.size() == argumentsRef_[ref].size());
    for (const auto &[idx, u] : llvm::enumerate(argumentsRef_[ref])) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Unifying " << u << " with " << par[idx] << "\n");
      joinShape(u, par[idx]);
    }
  }
  LLVM_DEBUG(llvm::dbgs() << DEBUG_LINE_BEG("Flatten-After-unifyGroups");
             llvm::dbgs() << "solverSegments_:\n"; solverSegments_->dump();
             llvm::dbgs() << "solverShapeElem_:\n"; solverShapeElem_->dump();
             llvm::dbgs() << "solverCollapserElem_:\n";
             solverCollapserElem_->dump(); dumpArgumentsRefPointer();
             dumpArgumentsRef(); dumpIsConnected();
             llvm::dbgs() << DEBUG_LINE_END("Flatten-After-unifyGroups"););
}

void DimensionAnalyzerBase::propagateConnection(int parent, int child) {
#ifndef NDEBUG
  auto fmt = [](ElementKind e, bool l, bool r) {
    char k = "MNU"[static_cast<int>(e)];
    return std::string(l ? "<-" : "") + k + std::string(r ? "->" : "");
  };

  LDBG("Child: " << child << " "
                 << fmt(isConnected_[child].elementKind,
                        isConnected_[child].leftConnected,
                        isConnected_[child].rightConnected)
                 << ", Parent: " << parent << " "
                 << fmt(isConnected_[parent].elementKind,
                        isConnected_[parent].leftConnected,
                        isConnected_[parent].rightConnected));
#endif

  isConnected_[parent].leftConnected =
      isConnected_[parent].leftConnected && isConnected_[child].leftConnected;
  isConnected_[parent].rightConnected =
      isConnected_[parent].rightConnected && isConnected_[child].rightConnected;
  isConnected_[parent].elementKind = std::min(isConnected_[parent].elementKind,
                                              isConnected_[child].elementKind);

#ifndef NDEBUG
  LDBG("  Result: " << parent << " "
                    << fmt(isConnected_[parent].elementKind,
                           isConnected_[parent].leftConnected,
                           isConnected_[parent].rightConnected));
#endif
}

void DimensionAnalyzerBase::propagateConnection() {
  for (int i = 0; i < argumentTotalLength_; ++i) {
    // Use collapser because the relationship is stronger
    int parent = solverCollapserElem_->find(i);
    propagateConnection(parent, i);
  }
}

void DimensionAnalyzerBase::spreadConnection() {
  LLVM_DEBUG(
      llvm::dbgs() << "[spreadConnection] total number of shape entries = "
                   << argumentTotalLength_ << "\n";
      llvm::dbgs()
      << "[ShapeIdx]: shape, parent, leftConnected, rightConnected\n";);
  for (int i = 0; i < argumentTotalLength_; ++i) {
    isConnected_[i] = isConnected_[solverCollapserElem_->find(i)];
    auto [_, shape] = solverShapeElem_->getMinParentAndShapePair(i);
    // check if this is available in the arguments
    LLVM_DEBUG(llvm::dbgs()
                   << "[" << i << "]: " << shape << ", "
                   << solverCollapserElem_->find(i) << ", "
                   << (isConnected_[i].leftConnected ? "true" : "false") << ","
                   << (isConnected_[i].rightConnected ? "true" : "false")
                   << "\n";);
  }
}

bool DimensionAnalyzerBase::isConnected(int a, int b) {
  if (a + 1 != b)
    return false;
  return isConnected_[a].rightConnected && isConnected_[b].leftConnected;
}

void DimensionAnalyzerBase::joinShape(int a, int b) {
  LDBG("Joining shape bind " << a << " " << b);
  solverShapeElem_->join(a, b);
  solverCollapserElem_->join(a, b);
}

void DimensionAnalyzerBase::joinCollapser(int a, int b) {
  LDBG("Joining collapser bind " << a << " " << b);
  solverCollapserElem_->join(a, b);
}

void DimensionAnalyzerBase::disconnect(int a, int b) {
  LDBG("Disconnecting " << a << " " << b);
  if (0 <= a && a < static_cast<int>(isConnected_.size())) {
    isConnected_[a].rightConnected = false;
    int parentOfA = solverCollapserElem_->find(a);
    isConnected_[parentOfA].rightConnected = false;
  }
  if (0 <= b && b < static_cast<int>(isConnected_.size())) {
    isConnected_[b].leftConnected = false;
    int parentOfB = solverCollapserElem_->find(b);
    isConnected_[parentOfB].leftConnected = false;
  }
}

void DimensionAnalyzerBase::separateGroup(Value val, BitVector contiguousMask,
                                          ArrayRef<int64_t> shape) {
  struct Group {
    bool isAllContiguous;
    bool isAllUnit;
    int leftIndex;
    int rightIndex;
  };
  auto argRef = getArgumentRef(val);
  size_t rank = argRef.size();

  if (rank <= 1)
    return;

  // Initialize groups - one group per dimension
  SmallVector<Group> groups;
  for (size_t i = 0; i < rank; ++i) {
    groups.push_back({/* .isAllContiguous = */ true,
                      /* .isAllUnit = */ (shape[i] == 1),
                      /* .leftIndex = */ static_cast<int>(i),
                      /* .rightIndex = */ static_cast<int>(i)});
  }

  // Helper to merge two groups
  auto mergeGroups = [&contiguousMask](const Group &left,
                                       const Group &right) -> Group {
    bool boundary = contiguousMask[right.leftIndex];
    return {/* .isAllContiguous = */ left.isAllContiguous &&
                right.isAllContiguous && boundary,
            /* .isAllUnit = */ left.isAllUnit && right.isAllUnit,
            /* .leftIndex = */ left.leftIndex,
            /* .rightIndex = */ right.rightIndex};
  };

  // First iteration: merge based on contiguity only
  SmallVector<Group> contiguousGroups;
  contiguousGroups.push_back(groups[0]);

  for (size_t i = 1; i < rank; ++i) {
    Group &lastGroup = contiguousGroups.back();
    const Group &currentGroup = groups[i];

    // Only check contiguity at boundary
    bool boundary = contiguousMask[currentGroup.leftIndex];

    if (boundary) {
      lastGroup = mergeGroups(lastGroup, currentGroup);
    } else {
      contiguousGroups.push_back(currentGroup);
    }
  }

  // Helper to check if two groups can be connected
  auto canConnect = [&argRef, this](const Group &left,
                                    const Group &right) -> bool {
    assert(left.rightIndex + 1 == right.leftIndex && "Groups must be adjacent");
    // Already disconnected, must disconnect
    if (!isConnected(argRef[left.rightIndex], argRef[right.leftIndex])) {
      return false;
    }
    // One is all units - can connect
    if (left.isAllUnit || right.isAllUnit) {
      return true;
    }
    return false;
  };

  // Second iteration: merge considering units and existing connections
  SmallVector<Group> mergedGroups;
  mergedGroups.push_back(contiguousGroups[0]);

  LDBG("Second iteration: merging by units and connections");
  for (size_t i = 1; i < contiguousGroups.size(); ++i) {
    Group &lastGroup = mergedGroups.back();
    const Group &currentGroup = contiguousGroups[i];

    if (canConnect(lastGroup, currentGroup)) {
      lastGroup = mergeGroups(lastGroup, currentGroup);
    } else {
      mergedGroups.push_back(currentGroup);
    }
  }

  // Disconnect dimensions at group boundaries
  for (size_t g = 1; g < mergedGroups.size(); ++g) {
    int leftIdx = mergedGroups[g - 1].rightIndex;
    int rightIdx = mergedGroups[g].leftIndex;
    disconnect(argRef[leftIdx], argRef[rightIdx]);
  }

  LDBG("Finished: " << mergedGroups.size() << " groups total");
}

void DimensionAnalyzerBase::separateMemref(Value memrefVal) {
  auto memrefType = dyn_cast<MemRefType>(memrefVal.getType());
  if (!memrefType)
    return;
  LDBG("separate Memref");
  createDummyRefIfNotExist({memrefVal});

  auto contiguousMask = hivm::detail::getContiguousAxesImpl({memrefType});
  auto shape = memrefType.getShape();
  separateGroup(memrefVal, contiguousMask, shape);
}

} // namespace detail
} // namespace mlir
