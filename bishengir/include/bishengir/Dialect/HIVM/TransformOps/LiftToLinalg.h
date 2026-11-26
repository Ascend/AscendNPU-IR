//===- LiftToLinalg.h - Lift HIVM vector functions to linalg ----*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_LIFTTOLINALG_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_LIFTTOLINALG_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"

#include <string>

namespace mlir {
namespace hivm {
LogicalResult liftFunctionsToLinalg(ArrayRef<func::FuncOp> functions,
                                    std::string &error);

}
} // namespace mlir

#endif
