//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
#ifndef BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <functional>

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"
namespace scope {

/// Create a pass to transform operations for SIMT execution.
std::unique_ptr<Pass> createTransformOpForSIMTPass();

/// Create a pass to outline scope within ScopeOp's region
std::unique_ptr<Pass>
createOutlineScopePass(const OutlineScopeOptions &options = {});

/// Create a pass to propagate the SIMT VF mode of outlined functions to
/// their callers.
std::unique_ptr<Pass> createPropagateSIMTModePass();

// Create a pass to inline operations within the scope.scope operation to the
// outer region.
std::unique_ptr<Pass>
createInlineScopePass(const InlineScopeOptions &options = {});

/// Filter deciding whether a callee should be inlined. The default filter
/// accepts every callee.
using CalleeFilter = std::function<bool(func::FuncOp)>;

// Create a pass to inline calls to the functions accepted by `calleeFilter`
// into scope.scope ops, so that one-shot-bufferize sees through the former
// call boundary.
std::unique_ptr<Pass> createInlineFuncToScopePass(
    CalleeFilter calleeFilter = [](func::FuncOp) { return true; });

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Scope/Transforms/Passes.h.inc"
} // namespace scope
} // namespace mlir

#endif // BISHENGIR_DIALECT_SCOPE_TRANSFORMS_PASSES_H
