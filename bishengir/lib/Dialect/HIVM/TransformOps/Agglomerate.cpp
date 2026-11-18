//===- Agglomerate.cpp - HIVM agglomeration engine and transform op -------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/Agglomerate.h"
#include "bishengir/Dialect/HIVM/TransformOps/Groups.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <cassert>
#include <cmath>
#include <utility>

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace mlir {
namespace hivm {
namespace {
class PreserveGroupsSelector final : public PairSelector {
public:
  FailureOr<SmallVector<FusionCandidate>> select(ArrayRef<AgglomerationBlock>,
                                                 const AgglomerationContext &,
                                                 std::string &) const override {
    return SmallVector<FusionCandidate>{};
  }
};

class UniformCostModel final : public FusionCostModel {
public:
  FailureOr<double> evaluate(const AgglomerationBlock &,
                             const AgglomerationContext &,
                             std::string &) const override {
    return 1.0;
  }
};

class SelectorRegistry {
public:
  static SelectorRegistry &get() {
    static SelectorRegistry registry;
    return registry;
  }

  LogicalResult add(StringRef name, std::unique_ptr<PairSelector> selector) {
    if (name.empty() || !selector)
      return failure();
    if (selectors.contains(name))
      return failure();
    selectors[name] = std::move(selector);
    return success();
  }

  std::optional<PairSelectorRef> lookup(StringRef name) const {
    auto it = selectors.find(name);
    if (it == selectors.end())
      return std::nullopt;
    return std::cref(*it->second);
  }

private:
  SelectorRegistry() {
    selectors["preserve_groups"] = std::make_unique<PreserveGroupsSelector>();
  }

  llvm::StringMap<std::unique_ptr<PairSelector>> selectors;
};

class CostModelRegistry {
public:
  static CostModelRegistry &get() {
    static CostModelRegistry registry;
    return registry;
  }

  LogicalResult add(StringRef name, std::unique_ptr<FusionCostModel> model) {
    if (name.empty() || !model)
      return failure();
    if (models.contains(name))
      return failure();
    models[name] = std::move(model);
    return success();
  }

  std::optional<FusionCostModelRef> lookup(StringRef name) const {
    auto it = models.find(name);
    if (it == models.end())
      return std::nullopt;
    return std::cref(*it->second);
  }

private:
  CostModelRegistry() {
    models["uniform_cost"] = std::make_unique<UniformCostModel>();
  }

  llvm::StringMap<std::unique_ptr<FusionCostModel>> models;
};

FailureOr<PairSelectorRef> resolveSelector(StringRef name, std::string &error) {
  if (name.empty()) {
    error = "agglomeration requires a selector";
    return failure();
  }
  auto selector = lookupSelector(name);
  if (!selector) {
    error = "unknown selector '" + name.str() + "'";
    return failure();
  }
  return *selector;
}

FailureOr<SmallVector<FusionCostModelRef>>
resolveCostModels(ArrayRef<StringRef> names, std::string &error) {
  if (names.empty()) {
    error = "agglomeration requires at least one cost model";
    return failure();
  }

  SmallVector<FusionCostModelRef> models;
  llvm::StringSet<> seen;
  for (StringRef name : names) {
    if (name.empty()) {
      error = "cost model name must not be empty";
      return failure();
    }
    if (!seen.insert(name).second) {
      error = "duplicate cost model '" + name.str() + "'";
      return failure();
    }
    auto model = lookupFusionCostModel(name);
    if (!model) {
      error = "unknown cost model '" + name.str() + "'";
      return failure();
    }
    models.push_back(*model);
  }
  return models;
}

FailureOr<AgglomerationBlockList>
buildSortedBlocks(ArrayRef<Group> groups, const GroupOrdinals &ordinals,
                  std::string &error) {
  AgglomerationBlockList blocks;
  for (const Group &group : groups) {
    SmallVector<Operation *> members(group.begin(), group.end());
    llvm::sort(members, [&](Operation *lhs, Operation *rhs) {
      return ordinals.lookup(lhs) < ordinals.lookup(rhs);
    });

    AgglomerationBlock block;
    for (Operation *member : members)
      block.add(member);
    if (block.empty()) {
      error = "initial agglomeration block must not be empty";
      return failure();
    }
    blocks.push_back(std::move(block));
  }
  return blocks;
}

FailureOr<func::FuncOp> validateGroupMember(Operation *member,
                                            func::FuncOp expectedFunction,
                                            std::string &error) {
  if (isa<ModuleOp>(member)) {
    error = "group member must not be a module";
    return failure();
  }
  if (isa<func::FuncOp>(member)) {
    error = "group member must not be a function";
    return failure();
  }

  auto function = member->getParentOfType<func::FuncOp>();
  if (!function) {
    error = "group member must be inside a function";
    return failure();
  }
  if (expectedFunction && expectedFunction != function) {
    error = "agglomeration groups must belong to one function";
    return failure();
  }
  return function;
}

bool matchesSourceUnion(const FusionCandidate &candidate,
                        ArrayRef<AgglomerationBlock> blocks) {
  auto [first, second] = candidate.sources;
  llvm::DenseSet<Operation *> expected;
  for (Operation *op : blocks[first].getOps())
    expected.insert(op);
  for (Operation *op : blocks[second].getOps())
    expected.insert(op);

  llvm::DenseSet<Operation *> actual;
  for (Operation *op : candidate.fused.getOps()) {
    if (!expected.contains(op))
      return false;
    actual.insert(op);
  }
  return actual.size() == expected.size();
}

bool hasSingleParentBlock(const AgglomerationBlock &block) {
  ArrayRef<Operation *> ops = block.getOps();
  if (ops.empty())
    return true;
  Block *parent = ops.front()->getBlock();
  if (!parent)
    return false;
  return llvm::all_of(
      ops, [parent](Operation *op) { return op->getBlock() == parent; });
}

bool isInBlockOrder(const AgglomerationBlock &block) {
  ArrayRef<Operation *> ops = block.getOps();
  for (size_t index = 1; index < ops.size(); ++index)
    if (!ops[index - 1]->isBeforeInBlock(ops[index]))
      return false;
  return true;
}

LogicalResult checkCandidateContract(const FusionCandidate &candidate,
                                     ArrayRef<AgglomerationBlock> blocks,
                                     std::string &error) {
  error.clear();
  auto [first, second] = candidate.sources;
  if (first == second) {
    error = "agglomeration candidate contains a duplicate source index";
    return failure();
  }
  if (first >= blocks.size() || second >= blocks.size()) {
    error = "agglomeration candidate source index is out of range";
    return failure();
  }

  if (candidate.fused.empty()) {
    error = "agglomeration candidate fused block is empty";
    return failure();
  }

  if (!hasSingleParentBlock(candidate.fused)) {
    error = "agglomeration candidate fused block must be in one parent block";
    return failure();
  }

  if (!isInBlockOrder(candidate.fused)) {
    error = "agglomeration candidate fused block must be in topological order";
    return failure();
  }

  if (!matchesSourceUnion(candidate, blocks)) {
    error = "agglomeration candidate fused block does not match the union of "
            "its source blocks";
    return failure();
  }
  return success();
}

void normalizeCandidateSources(FusionCandidate &candidate) {
  auto &[first, second] = candidate.sources;
  if (first > second)
    std::swap(first, second);
}

FailureOr<double> evaluateBlockCost(const FusionCostModel &model,
                                    const AgglomerationBlock &block,
                                    const AgglomerationContext &context,
                                    std::string &error) {
  std::string modelError;
  auto cost = model.evaluate(block, context, modelError);
  if (failed(cost)) {
    error = modelError.empty() ? "fusion cost model failed" : modelError;
    return failure();
  }
  if (!std::isfinite(*cost)) {
    error = "fusion cost must be finite";
    return failure();
  }
  if (*cost < 0.0) {
    error = "fusion cost must be non-negative";
    return failure();
  }
  return *cost;
}

FailureOr<double> evaluateAverageDelta(const FusionCandidate &candidate,
                                       ArrayRef<AgglomerationBlock> blocks,
                                       ArrayRef<FusionCostModelRef> models,
                                       const AgglomerationContext &context,
                                       std::string &error) {
  auto [first, second] = candidate.sources;
  double total = 0.0;
  for (const FusionCostModelRef &model : models) {
    auto fusedCost =
        evaluateBlockCost(model.get(), candidate.fused, context, error);
    if (failed(fusedCost))
      return failure();

    auto firstCost =
        evaluateBlockCost(model.get(), blocks[first], context, error);
    if (failed(firstCost))
      return failure();
    auto secondCost =
        evaluateBlockCost(model.get(), blocks[second], context, error);
    if (failed(secondCost))
      return failure();

    double sourceCost = *firstCost + *secondCost;
    if (!std::isfinite(sourceCost)) {
      error = "fusion cost delta must be finite";
      return failure();
    }
    double delta = *fusedCost - sourceCost;
    if (!std::isfinite(delta)) {
      error = "fusion cost delta must be finite";
      return failure();
    }
    total += delta;
    if (!std::isfinite(total)) {
      error = "fusion cost delta must be finite";
      return failure();
    }
  }
  return total / static_cast<double>(models.size());
}

FailureOr<double> evaluatePlanCost(ArrayRef<AgglomerationBlock> blocks,
                                   ArrayRef<FusionCostModelRef> models,
                                   const AgglomerationContext &context,
                                   std::string &error) {
  double total = 0.0;
  for (const FusionCostModelRef &model : models) {
    double sum = 0.0;
    for (const AgglomerationBlock &block : blocks) {
      auto cost = evaluateBlockCost(model.get(), block, context, error);
      if (failed(cost))
        return failure();
      sum += *cost;
    }
    total += sum;
  }
  return total / static_cast<double>(models.size());
}

void applyCandidate(AgglomerationBlockList &blocks,
                    const FusionCandidate &candidate) {
  auto [first, second] = candidate.sources;
  assert(first < second && "candidate sources must be normalized");
  AgglomerationBlockList next;
  for (size_t index = 0; index < blocks.size(); ++index) {
    if (index == first)
      next.push_back(candidate.fused);
    if (index != first && index != second)
      next.push_back(blocks[index]);
  }
  blocks = std::move(next);
}

struct BestCandidate {
  size_t index;
  double delta;
};

FailureOr<std::optional<BestCandidate>>
selectBestCandidate(SmallVector<FusionCandidate> &candidates,
                    ArrayRef<AgglomerationBlock> blocks,
                    ArrayRef<FusionCostModelRef> models,
                    const AgglomerationContext &context, std::string &error) {
  std::optional<BestCandidate> best;
  for (auto [index, candidate] : llvm::enumerate(candidates)) {
    normalizeCandidateSources(candidate);
    if (failed(checkCandidateContract(candidate, blocks, error)))
      return failure();
    auto delta =
        evaluateAverageDelta(candidate, blocks, models, context, error);
    if (failed(delta))
      return failure();
    if (!best || *delta < best->delta)
      best = BestCandidate{index, *delta};
  }
  return best;
}

FailureOr<AgglomerationResult> buildResult(ArrayRef<AgglomerationBlock> blocks,
                                           ArrayRef<FusionCostModelRef> models,
                                           const AgglomerationContext &context,
                                           std::string &error) {
  auto planCost = evaluatePlanCost(blocks, models, context, error);
  if (failed(planCost))
    return failure();
  if (!std::isfinite(*planCost)) {
    error = "plan cost must be finite";
    return failure();
  }
  if (*planCost < 0.0) {
    error = "plan cost must be non-negative";
    return failure();
  }

  AgglomerationResult result;
  result.cost = *planCost;
  for (const AgglomerationBlock &block : blocks)
    result.groups.push_back(
        Group(block.getOps().begin(), block.getOps().end()));
  return result;
}

} // namespace

LogicalResult registerSelector(StringRef name,
                               std::unique_ptr<PairSelector> selector) {
  return SelectorRegistry::get().add(name, std::move(selector));
}

std::optional<PairSelectorRef> lookupSelector(StringRef name) {
  return SelectorRegistry::get().lookup(name);
}

LogicalResult registerFusionCostModel(StringRef name,
                                      std::unique_ptr<FusionCostModel> model) {
  return CostModelRegistry::get().add(name, std::move(model));
}

std::optional<FusionCostModelRef> lookupFusionCostModel(StringRef name) {
  return CostModelRegistry::get().lookup(name);
}

FailureOr<AgglomerationBlockList>
buildInitialBlocks(ModuleOp root, ArrayRef<Group> groups, std::string &error) {
  error.clear();
  auto ordinals = buildGroupOrdinals(root, error);
  if (failed(ordinals))
    return failure();

  llvm::DenseSet<Operation *> seen;
  SmallVector<Operation *> members;
  for (const Group &group : groups) {
    func::FuncOp groupFunction;
    if (group.empty()) {
      error = "initial agglomeration group must not be empty";
      return failure();
    }
    for (Operation *member : group) {
      if (!member) {
        error = "initial agglomeration group contains a null operation";
        return failure();
      }
      if (!ordinals->contains(member)) {
        error = "group member is outside the payload root";
        return failure();
      }
      auto function = validateGroupMember(member, groupFunction, error);
      if (failed(function))
        return failure();
      groupFunction = *function;
      if (!seen.insert(member).second) {
        error = "initial agglomeration blocks must be disjoint";
        return failure();
      }
      members.push_back(member);
    }
  }

  for (Operation *member : members) {
    for (Operation *parent = member->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (seen.contains(parent)) {
        error = "initial agglomeration blocks must not be nested";
        return failure();
      }
    }
  }

  return buildSortedBlocks(groups, *ordinals, error);
}

FailureOr<AgglomerationResult>
agglomerate(ModuleOp root, DictionaryAttr configuration, ArrayRef<Group> groups,
            StringRef selectorName, ArrayRef<StringRef> costModelNames,
            std::string &error) {
  error.clear();
  if (!root) {
    error = "agglomeration root must be a module";
    return failure();
  }
  if (!configuration) {
    error = "agglomeration options must be a dictionary";
    return failure();
  }

  auto selector = resolveSelector(selectorName, error);
  if (failed(selector))
    return failure();
  auto models = resolveCostModels(costModelNames, error);
  if (failed(models))
    return failure();

  auto blocks = buildInitialBlocks(root, groups, error);
  if (failed(blocks))
    return failure();

  AgglomerationContext context{root, configuration};
  PairSelectorRef selectorRef = *selector;

  while (true) {
    if (blocks->size() < 2)
      break;

    std::string selectionError;
    auto candidates =
        selectorRef.get().select(*blocks, context, selectionError);
    if (failed(candidates)) {
      error = selectionError.empty() ? "agglomeration selector failed"
                                     : selectionError;
      return failure();
    }
    if (candidates->empty())
      break;

    auto best =
        selectBestCandidate(*candidates, *blocks, *models, context, error);
    if (failed(best))
      return failure();
    if (!*best || (*best)->delta > 0.0)
      break;

    applyCandidate(*blocks, (*candidates)[(*best)->index]);
  }

  return buildResult(*blocks, *models, context, error);
}

} // namespace hivm
} // namespace mlir

namespace {
FailureOr<ModuleOp> resolveAgglomerationRoot(Operation *user,
                                             ArrayRef<Operation *> roots) {
  if (roots.size() != 1)
    return user->emitOpError() << "expected exactly one payload root";
  auto root = dyn_cast<ModuleOp>(roots.front());
  if (!root)
    return user->emitOpError() << "expected a payload module root";
  return root;
}

FailureOr<DictionaryAttr>
resolveAgglomerationOptions(Operation *user, ArrayRef<Attribute> options) {
  if (options.size() != 1)
    return user->emitOpError() << "expected exactly one options attribute";
  auto configuration = dyn_cast<DictionaryAttr>(options.front());
  if (!configuration)
    return user->emitOpError() << "expected a dictionary options attribute";
  return configuration;
}

FailureOr<GroupsAttr> resolveAgglomerationGroups(Operation *user,
                                                 ArrayRef<Attribute> params) {
  if (params.size() != 1)
    return user->emitOpError() << "expected exactly one groups attribute";
  auto groups = dyn_cast<GroupsAttr>(params.front());
  if (!groups)
    return user->emitOpError() << "expected a #hivm_vectorize.groups attribute";
  return groups;
}

} // namespace

LogicalResult HIVMAgglomerateOp::verify() {
  if (getSelector().empty())
    return emitOpError() << "requires a selector";

  auto cost = getCost();
  if (cost.empty())
    return emitOpError() << "requires at least one cost model";

  llvm::StringSet<> seen;
  for (Attribute attribute : cost) {
    auto name = dyn_cast<StringAttr>(attribute);
    if (!name)
      return emitOpError() << "cost model names must be strings";
    if (name.getValue().empty())
      return emitOpError() << "cost model name must not be empty";
    if (!seen.insert(name.getValue()).second)
      return emitOpError() << "duplicate cost model '" << name.getValue()
                           << "'";
  }
  return success();
}

DiagnosedSilenceableFailure HIVMAgglomerateOp::apply(TransformRewriter &,
                                                     TransformResults &results,
                                                     TransformState &state) {
  auto payloadOps = state.getPayloadOps(getRoot());
  SmallVector<Operation *> roots(payloadOps.begin(), payloadOps.end());
  auto root = resolveAgglomerationRoot(getOperation(), roots);
  if (failed(root))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto configuration = resolveAgglomerationOptions(
      getOperation(), state.getParams(getOptions()));
  if (failed(configuration))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto encodedGroups =
      resolveAgglomerationGroups(getOperation(), state.getParams(getGroups()));
  if (failed(encodedGroups))
    return DiagnosedSilenceableFailure::definiteFailure();

  std::string error;
  auto groups = decodeGroups(*root, *encodedGroups, error);
  if (failed(groups))
    return emitDefiniteFailure() << error;

  SmallVector<StringRef> costNames;
  for (Attribute attribute : getCost())
    costNames.push_back(cast<StringAttr>(attribute).getValue());

  auto result = agglomerate(*root, *configuration, *groups, getSelector(),
                            costNames, error);
  if (failed(result))
    return emitDefiniteFailure() << error;

  auto encoded = encodeGroups(*root, result->groups, error);
  if (failed(encoded))
    return emitDefiniteFailure() << error;

  auto cost = Builder(getContext()).getF64FloatAttr(result->cost);
  auto plan = PlanAttr::get(getContext(), cost, *encoded);
  results.setParams(cast<OpResult>(getPlan()), {plan});
  return DiagnosedSilenceableFailure::success();
}

void HIVMAgglomerateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getRootMutable(), effects);
  onlyReadsHandle(getOptionsMutable(), effects);
  onlyReadsHandle(getGroupsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}
