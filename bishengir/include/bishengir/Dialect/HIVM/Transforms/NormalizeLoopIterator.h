//===- NormalizeLoopIterator.h - Process memory conflicts in loop iterator-===//
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
#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace hivm {

// As plan memory will consider that iterArg and yield value use same one space
// and if there exists use of iterArg after yield value memory initialization,
// iterArg has been 'dirty'.
// Here consider above state and separate original yield value memory from
// iteration memmory
void populateNormalizeLoopIneratorPattern(RewritePatternSet &patterns);

} // namespace hivm
} // namespace mlir