//===- Agglomerate.h - HIVM agglomeration models and engine -----*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_AGGLOMERATE_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_AGGLOMERATE_H

#include "bishengir/Analysis/AgglomerationBlock.h"
#include "bishengir/Analysis/Partition.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace hivm {
struct AgglomerationContext {
  ModuleOp root;
  DictionaryAttr configuration;
};

using AgglomerationBlockIndex = size_t;

using AgglomerationBlockPair =
    std::pair<AgglomerationBlockIndex, AgglomerationBlockIndex>;

struct FusionCandidate {
  AgglomerationBlockPair sources;
  AgglomerationBlock fused;
  Attribute metadata;
};

class PairSelector {
public:
  virtual ~PairSelector() = default;

  virtual FailureOr<SmallVector<FusionCandidate>>
  select(ArrayRef<AgglomerationBlock> blocks,
         const AgglomerationContext &context, std::string &error) const = 0;
};

class FusionCostModel {
public:
  virtual ~FusionCostModel() = default;

  virtual FailureOr<double> evaluate(const AgglomerationBlock &block,
                                     const AgglomerationContext &context,
                                     std::string &error) const = 0;
};

using PairSelectorRef = std::reference_wrapper<const PairSelector>;

LogicalResult registerSelector(StringRef name,
                               std::unique_ptr<PairSelector> selector);

std::optional<PairSelectorRef> lookupSelector(StringRef name);

LogicalResult registerFusionCostModel(StringRef name,
                                      std::unique_ptr<FusionCostModel> model);

using FusionCostModelRef = std::reference_wrapper<const FusionCostModel>;

std::optional<FusionCostModelRef> lookupFusionCostModel(StringRef name);

struct AgglomerationResult {
  double cost;
  Groups groups;
};

FailureOr<AgglomerationBlockList>
buildInitialBlocks(ModuleOp root, ArrayRef<Group> groups, std::string &error);

FailureOr<AgglomerationResult>
agglomerate(ModuleOp root, DictionaryAttr configuration, ArrayRef<Group> groups,
            StringRef selectorName, ArrayRef<StringRef> costModelNames,
            std::string &error);

} // namespace hivm
} // namespace mlir

#endif
