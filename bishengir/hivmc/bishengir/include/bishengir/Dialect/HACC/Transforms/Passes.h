//===- Passes.h - Pass Entrypoints --------------------------------*- C++-*-==//
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
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_HACC_TRANSFORMS_PASSES_H

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hacc {

#define GEN_PASS_DECL
#include "bishengir/Dialect/HACC/Transforms/Passes.h.inc"

/// Create a pass to rename function name.
std::unique_ptr<Pass> createRenameFuncPass();

/// Create a pass to append target spec. information to the top-level module.
std::unique_ptr<Pass>
createAppendDeviceSpecPass(const AppendTargetDeviceSpecOptions &options = {});

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/HACC/Transforms/Passes.h.inc"

} // namespace hacc
} // namespace mlir

#endif // BISHENGIR_DIALECT_HACC_TRANSFORMS_PASSES_H
