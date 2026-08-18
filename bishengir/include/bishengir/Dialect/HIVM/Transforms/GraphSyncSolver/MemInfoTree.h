//===------------- MemInfoTree.h ---- Graph Sync Solver -------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFOTREE_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFOTREE_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/CorePipeInfo.h"
#include "bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/MemInfo.h"

namespace mlir::hivm::syncsolver {

struct MemInfoOccElement {
  Occurrence *occ{nullptr};
  Occurrence *parentOcc{nullptr};
  int64_t occIndex{-1};
  int64_t parentOccIndex{-1};

  MemInfoOccElement(Occurrence *occ, int64_t occIndex)
      : occ(occ), parentOcc(occ), occIndex(occIndex), parentOccIndex(occIndex) {
  }
  MemInfoOccElement(Occurrence *occ, Occurrence *parentOcc, int64_t occIndex,
                    int64_t parentOccIndex)
      : occ(occ), parentOcc(parentOcc), occIndex(occIndex),
        parentOccIndex(parentOccIndex) {}

  bool operator<(const MemInfoOccElement &other) const {
    return occIndex < other.occIndex;
  }
  bool operator>(const MemInfoOccElement &other) const {
    return occIndex > other.occIndex;
  }

  std::string str(int64_t indent = 0) {
    std::string ret = std::string(indent, ' ') + "MemInfoOccElement{";
    ret += "occ: ";
    ret += occ == nullptr ? "null" : "set";
    ret += ", parentOcc: ";
    ret += parentOcc == nullptr ? "null" : "set";
    ret += ", occIndex: " + std::to_string(occIndex) + ", ";
    ret += "parentOccIndex: " + std::to_string(parentOccIndex);
    ret += "}";
    return ret;
  }
};

using MemInfoOccElementList = llvm::SmallVector<MemInfoOccElement>;

struct MemInfoNode {
  CorePipeInfo corePipeInfo;
  MemoryEffect memoryEffect;
  MemInfo rootMemInfo;
  MemInfoOccElementList occElements;

  MemInfoNode(CorePipeInfo corePipeInfo, MemoryEffect memoryEffect,
              MemInfo rootMemInfo)
      : corePipeInfo(corePipeInfo), memoryEffect(memoryEffect),
        rootMemInfo(rootMemInfo) {}

  MemInfoOccElementList::iterator
  lower_bound(const MemInfoOccElement &occElement) {
    return llvm::lower_bound(occElements, occElement);
  }

  void insert(MemInfoOccElement newOccElement) {
    auto it = lower_bound(newOccElement);
    occElements.insert(it, std::move(newOccElement));
  }

  void insert(MemInfoOccElementList newOccElements) {
    if (occElements.size() < newOccElements.size()) {
      std::swap(occElements, newOccElements);
    }
    for (auto &occElement : newOccElements) {
      insert(std::move(occElement));
    }
  }

  std::string str(int64_t indent = 0) {
    std::string ret = std::string(indent, ' ') + "MemInfoNode{";
    ret += "core: " + stringifyTCoreType(corePipeInfo.coreType).str() + ", ";
    ret += "pipe: " + stringifyPIPE(corePipeInfo.pipe).str() + ", ";
    ret += "memoryEffect: " + stringifyMemoryEffect(memoryEffect).str() + ", ";
    ret += "rootMemInfo: " + rootMemInfo.str() + ",\n";
    ret += std::string(indent + 2, ' ') + "occElements: [";
    if (!occElements.empty()) {
      ret += '\n';
      for (auto &element : occElements) {
        ret += element.str(indent + 4) + '\n';
      }
      ret += std::string(indent + 2, ' ');
    }
    ret += "]";
    ret += "}";
    return ret;
  }
};

using MemInfoNodeList = std::vector<MemInfoNode>;

struct MemInfoTree {

  Occurrence *occ{nullptr};
  int64_t occSyncIrIndex{-1};
  llvm::DenseMap<MemoryEffect, llvm::DenseMap<CorePipeInfo, MemInfoNodeList>>
      nodeListMap;

  MemInfoTree(Occurrence *occ, int64_t occSyncIrIndex)
      : occ(occ), occSyncIrIndex(occSyncIrIndex) {}

  std::optional<std::reference_wrapper<MemInfoNodeList>>
  getMemInfoNodeList(CorePipeInfo corePipeInfo, MemoryEffect memoryEffect,
                     bool createIfNotFound = false) {
    auto it1 = nodeListMap.find(memoryEffect);
    if (it1 != nodeListMap.end()) {
      auto it2 = it1->second.find(corePipeInfo);
      if (it2 != it1->second.end()) {
        return std::reference_wrapper<MemInfoNodeList>(it2->second);
      }
    }
    if (createIfNotFound) {
      return std::reference_wrapper<MemInfoNodeList>(
          nodeListMap[memoryEffect][corePipeInfo]);
    }
    return {};
  };

  std::optional<std::reference_wrapper<MemInfoNode>>
  getMemInfoNode(CorePipeInfo corePipeInfo, MemoryEffect memoryEffect,
                 const MemInfo &memInfo, bool createIfNotFound = false) {
    auto nodeList =
        getMemInfoNodeList(corePipeInfo, memoryEffect, createIfNotFound);
    if (!nodeList.has_value()) {
      assert(!createIfNotFound);
      return {};
    }
    for (auto &element : nodeList->get()) {
      if (element.rootMemInfo == memInfo) {
        return std::reference_wrapper<MemInfoNode>(element);
      }
    }
    if (createIfNotFound) {
      nodeList->get().push_back(
          MemInfoNode(corePipeInfo, memoryEffect, memInfo));
      return std::reference_wrapper<MemInfoNode>(nodeList->get().back());
    }
    return {};
  };

  void insert(CorePipeInfo corePipeInfo, MemoryEffect memoryEffect,
              const MemInfo &memInfo, const MemInfoOccElement &occElement) {
    auto node = getMemInfoNode(corePipeInfo, memoryEffect, memInfo,
                               /*createIfNotFound=*/true);
    assert(node.has_value());
    node->get().insert(occElement);
  }

  void insert(const MemInfoNode &otherNode) {
    auto node = getMemInfoNode(otherNode.corePipeInfo, otherNode.memoryEffect,
                               otherNode.rootMemInfo,
                               /*createIfNotFound=*/true);
    assert(node.has_value());
    node->get().insert(otherNode.occElements);
  }

  void insert(const MemInfoNodeList &otherNodeList, Occurrence *parentOcc,
              int64_t parentOccIndex) {
    for (auto copiedOtherNode : otherNodeList) {
      for (auto &occElement : copiedOtherNode.occElements) {
        occElement.parentOcc = parentOcc;
        occElement.parentOccIndex = parentOccIndex;
      }
      insert(copiedOtherNode);
    }
  }

  void merge(const MemInfoTree &other) {
    for (auto &[memoryEffect, map] : other.nodeListMap) {
      for (auto &[corePipeInfo, nodeList] : map) {
        insert(nodeList, other.occ, other.occSyncIrIndex);
      }
    }
  }

  std::string str(int64_t indent = 0) {
    std::string ret;
    ret += std::string(indent, ' ') + "MemInfoTree:\n";
    for (auto &[memoryEffect, map] : nodeListMap) {
      for (auto &[corePipeInfo, nodeList] : map) {
        ret += std::string(indent + 2, ' ');
        ret +=
            "core: " + stringifyTCoreType(corePipeInfo.coreType).str() + ", ";
        ret += "pipe: " + stringifyPIPE(corePipeInfo.pipe).str() + ", ";
        ret += "memoryEffect: " + stringifyMemoryEffect(memoryEffect).str();
        ret += "\n";
        for (auto &node : nodeList) {
          ret += node.str(indent + 4);
          ret += '\n';
        }
      }
    }
    return ret;
  }
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFOTREE_H
