//===- HIVMVectorizeLowering.cpp - HIVM vectorization lowerings -----------===//
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

#include "bishengir/Dialect/HIVM/Transforms/HIVMVectorizeLowering.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::hivm::vectorize {

FailureOr<SmallVector<OperationName>> getExpectedHIVMLowering(Operation *op) {
  using Result = FailureOr<SmallVector<OperationName>>;

  return llvm::TypeSwitch<Operation *, Result>(op)
      .Case<linalg::ElemwiseBinaryOp>([](auto binary) -> Result {
        llvm::StringRef target;
        switch (binary.getFun()) {
        case linalg::BinaryFn::add:
          target = hivm::VAddOp::getOperationName();
          break;
        case linalg::BinaryFn::mul:
          target = hivm::VMulOp::getOperationName();
          break;
        default:
          return failure();
        }
        return SmallVector<OperationName>{
            OperationName(target, binary->getContext())};
      })
      .Default([](Operation *) -> Result { return failure(); });
}

bool isVectorizableTarget(OperationName name) {
  auto registered = name.getRegisteredInfo();
  return registered &&
         registered->hasInterface<hivm::VectorizableOpInterface>();
}

bool hasVectorizableLowering(Operation *op) {
  auto targets = getExpectedHIVMLowering(op);
  if (failed(targets) || targets->empty())
    return false;
  return llvm::all_of(*targets, isVectorizableTarget);
}

} // namespace mlir::hivm::vectorize
