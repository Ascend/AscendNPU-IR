//===- AgglomerationBlock.h - Block with boundaries -------------*- C++ -*-===//
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

#ifndef BISHENGIR_ANALYSIS_AGGLOMERATIONBLOCK_H
#define BISHENGIR_ANALYSIS_AGGLOMERATIONBLOCK_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>

namespace mlir {
namespace hivm {
class AgglomerationBlock {
public:
  AgglomerationBlock() = default;

  ArrayRef<Operation *> getOps() const { return ops.getArrayRef(); }
  size_t size() const { return ops.size(); }
  bool empty() const { return ops.empty(); }
  bool contains(Operation *op) const { return ops.contains(op); }

  void add(Operation *op);

  void remove(Operation *op);

  const llvm::SetVector<Value> &getInputs() const;

  const llvm::SetVector<Value> &getOutputs() const;

private:
  void ensureBoundaries() const;
  void recomputeBoundaries() const;

  llvm::SetVector<Operation *> ops;
  mutable llvm::SetVector<Value> inputs;
  mutable llvm::SetVector<Value> outputs;
  mutable bool boundariesDirty = true;
};

using AgglomerationBlockList = SmallVector<AgglomerationBlock>;

} // namespace hivm
} // namespace mlir

#endif
