//===- Partition.h - HIVM partition models and transport -------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_PARTITION_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_PARTITION_H

#include "bishengir/Analysis/Partition.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
namespace hivm {
struct PartitionContext {
  ModuleOp root;
  DictionaryAttr configuration;
};

class PartitionModel {
public:
  virtual ~PartitionModel() = default;

  virtual FailureOr<Groups> apply(ArrayRef<Operation *> group,
                                  const PartitionContext &context,
                                  std::string &error) const = 0;
};

using PartitionModelRef =
    std::optional<std::reference_wrapper<const PartitionModel>>;

LogicalResult registerPartitionModel(StringRef name,
                                     std::unique_ptr<PartitionModel> model);

PartitionModelRef lookupPartitionModel(StringRef name);

bool isVectorizationCompute(Operation &operation);

FailureOr<Groups> partition(ModuleOp root, DictionaryAttr configuration,
                            ArrayRef<StringRef> models, std::string &error);

} // namespace hivm
} // namespace mlir

#endif
