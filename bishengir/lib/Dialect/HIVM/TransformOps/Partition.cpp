//===- Partition.cpp - HIVM partition models and transport ----------------===//
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

#include "bishengir/Dialect/HIVM/TransformOps/Partition.h"
#include "bishengir/Dialect/HIVM/Interfaces/VectorizableOpInterface.h"
#include "bishengir/Dialect/HIVM/TransformOps/Groups.h"
#include "bishengir/Dialect/HIVM/TransformOps/VectorizeTransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

namespace mlir {
namespace hivm {
namespace {
class SameBlockPartitionModel final : public PartitionModel {
public:
  FailureOr<Groups> apply(ArrayRef<Operation *> group, const PartitionContext &,
                          std::string &error) const override {
    llvm::MapVector<Block *, llvm::DenseSet<Operation *>> members;
    for (auto *member : group) {
      if (!member) {
        error = "SameBlock received a null operation";
        return failure();
      }
      member->walk<WalkOrder::PreOrder>([&](Operation *operation) {
        if (operation->getParentOfType<func::FuncOp>())
          members[operation->getBlock()].insert(operation);
      });
    }
    Groups result;
    for (const auto &[block, selected] : members) {
      Group operations;
      for (auto &operation : *block)
        if (selected.contains(&operation))
          operations.push_back(&operation);
      result.push_back(std::move(operations));
    }
    return result;
  }
};

bool isTensorCompute(Operation &operation) {
  return isVectorizationCompute(operation) && operation.getNumResults() &&
         llvm::all_of(operation.getResultTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

class UnseparatedByBarrierPartitionModel final : public PartitionModel {
public:
  FailureOr<Groups> apply(ArrayRef<Operation *> group, const PartitionContext &,
                          std::string &error) const override {
    Groups result;
    Operation *previous = nullptr;
    for (auto *operation : group) {
      if (!operation) {
        error = "UnseparatedByBarrier received a null operation";
        return failure();
      }
      if (!isTensorCompute(*operation)) {
        previous = nullptr;
        continue;
      }
      if (!previous || previous->getNextNode() != operation)
        result.emplace_back();
      result.back().push_back(operation);
      previous = operation;
    }
    return result;
  }
};

class PartitionRegistry {
public:
  static PartitionRegistry &get() {
    static PartitionRegistry registry;
    return registry;
  }

  LogicalResult add(StringRef name, std::unique_ptr<PartitionModel> model) {
    if (name.empty() || !model)
      return failure();
    if (models.contains(name))
      return failure();
    models[name] = std::move(model);
    return success();
  }

  PartitionModelRef lookup(StringRef name) const {
    auto it = models.find(name);
    if (it == models.end())
      return std::nullopt;
    return std::cref(*it->second);
  }

private:
  PartitionRegistry() {
    models["SameBlock"] = std::make_unique<SameBlockPartitionModel>();
    models["UnseparatedByBarrier"] =
        std::make_unique<UnseparatedByBarrierPartitionModel>();
  }

  llvm::StringMap<std::unique_ptr<PartitionModel>> models;
};

} // namespace

bool isVectorizationCompute(Operation &operation) {
  // These forms have no supported vectorization path yet:
  // vpow is normalize-only or lowered as a library call, and rounded vshr
  // requires backend rounding semantics that the linalg lifting cannot express.
  if (isa<VPowOp>(operation) ||
      (isa<VShROp>(operation) && cast<VShROp>(operation).getRound()))
    return false;
  if (auto vectorizable = dyn_cast<VectorizableOpInterface>(operation))
    if (vectorizable.isVectorizable())
      return true;
  if (auto structured = dyn_cast<HIVMStructuredOp>(operation))
    if (structured.isElemwiseNaryOp())
      return true;
  return isa<VReduceOp, VArangeOp, VBrcOp, VExp2Op, VLog2Op, VLog10Op, VLog1pOp,
             VExpM1Op, VIlogbOp, VTanhOp, VSinOp, VCosOp, VAtanOp, VTanOp,
             VErfOp, VAddOp, VMulOp, VMulExtOp, VMulExtUiOp, VModOp, VModUIOp,
             VLdexpOp, VIsInfOp, VIsNanOp>(operation);
}

LogicalResult registerPartitionModel(StringRef name,
                                     std::unique_ptr<PartitionModel> model) {
  return PartitionRegistry::get().add(name, std::move(model));
}

PartitionModelRef lookupPartitionModel(StringRef name) {
  return PartitionRegistry::get().lookup(name);
}

namespace {
FailureOr<ModuleOp> resolvePartitionRoot(Operation *user,
                                         ArrayRef<Operation *> roots) {
  if (roots.size() != 1)
    return user->emitOpError() << "expected exactly one payload root";
  auto root = dyn_cast<ModuleOp>(roots.front());
  if (!root)
    return user->emitOpError() << "expected a payload module root";
  return root;
}

FailureOr<DictionaryAttr> resolvePartitionOptions(Operation *user,
                                                  ArrayRef<Attribute> options) {
  if (options.size() != 1)
    return user->emitOpError() << "expected exactly one options attribute";
  auto configuration = dyn_cast<DictionaryAttr>(options.front());
  if (!configuration)
    return user->emitOpError() << "expected a dictionary options attribute";
  return configuration;
}

} // namespace

FailureOr<Groups> partition(ModuleOp root, DictionaryAttr configuration,
                            ArrayRef<StringRef> models, std::string &error) {
  if (!root) {
    error = "partition root must be a module";
    return failure();
  }
  if (!configuration) {
    error = "partition options must be a dictionary";
    return failure();
  }
  if (models.empty()) {
    error = "partition requires at least one model";
    return failure();
  }

  PartitionContext context{root, configuration};
  llvm::StringSet<> seen;
  std::string modelError;
  SmallVector<Relation> relations;
  for (StringRef name : models) {
    if (!seen.insert(name).second) {
      error = "duplicate partition model '" + name.str() + "'";
      return failure();
    }
    auto model = lookupPartitionModel(name);
    if (!model) {
      error = "unknown partition model '" + name.str() + "'";
      return failure();
    }
    auto modelRef = *model;
    relations.emplace_back(
        [modelRef, context, &modelError](ArrayRef<Operation *> group) {
          std::string relationError;
          auto groups = modelRef.get().apply(group, context, relationError);
          if (failed(groups))
            modelError = relationError;
          return groups;
        });
  }

  auto groups = applyRelations(root.getOperation(), relations, error);
  if (failed(groups) && !modelError.empty())
    error = modelError;
  return groups;
}

} // namespace hivm
} // namespace mlir

LogicalResult HIVMPartitionOp::verify() {
  auto models = getModels();
  if (!models || models.empty())
    return emitOpError() << "requires at least one model";

  llvm::StringSet<> seen;
  for (Attribute attribute : models) {
    auto name = dyn_cast<StringAttr>(attribute);
    if (!name)
      return emitOpError() << "model names must be strings";
    if (name.getValue().empty())
      return emitOpError() << "model name must not be empty";
    if (!seen.insert(name.getValue()).second)
      return emitOpError() << "duplicate model '" << name.getValue() << "'";
  }
  return success();
}

DiagnosedSilenceableFailure HIVMPartitionOp::apply(TransformRewriter &,
                                                   TransformResults &results,
                                                   TransformState &state) {
  auto payloadOps = state.getPayloadOps(getRoot());
  SmallVector<Operation *> roots(payloadOps.begin(), payloadOps.end());
  auto root = resolvePartitionRoot(getOperation(), roots);
  if (failed(root))
    return DiagnosedSilenceableFailure::definiteFailure();

  auto configuration =
      resolvePartitionOptions(getOperation(), state.getParams(getOptions()));
  if (failed(configuration))
    return DiagnosedSilenceableFailure::definiteFailure();

  SmallVector<StringRef> modelNames;
  for (Attribute attribute : getModels())
    modelNames.push_back(cast<StringAttr>(attribute).getValue());

  std::string error;
  auto groups = hivm::partition(*root, *configuration, modelNames, error);
  if (failed(groups))
    return emitDefiniteFailure() << error;

  auto encoded = encodeGroups(*root, *groups, error);
  if (failed(encoded))
    return emitDefiniteFailure() << error;

  results.setParams(cast<OpResult>(getGroups()), {*encoded});
  return DiagnosedSilenceableFailure::success();
}

void HIVMPartitionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getRootMutable(), effects);
  onlyReadsHandle(getOptionsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}
