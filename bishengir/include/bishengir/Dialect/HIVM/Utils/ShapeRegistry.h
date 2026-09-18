//===- ShapeRegistry.h - Loop-shape registry for CV heuristics --*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#ifndef BISHENGIR_DIALECT_HIVM_UTILS_SHAPEREGISTRY_H
#define BISHENGIR_DIALECT_HIVM_UTILS_SHAPEREGISTRY_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace hivm {

/// True when `name` or any substring fingerprints to a registered loop shape.
/// Decorated symbols such as `_mix_aic` / `_mix_aiv` suffixes still match.
/// An empty registry never matches.
bool isLoopShapeRegistered(llvm::StringRef name);

/// Shared gate for registered-kernel CV / preload heuristics.
/// LIT may pass `--bypass-shape-registry` to force the heuristic on.
/// Otherwise `--enable-preload`, a non-zero `--set-workspace-multibuffer`,
/// and a registered kernel name are all required.
/// Covers LCD backup extraction (intra-WI and cross-WI) as well as dest-sink
/// / delay / merge-same-preload heuristics.
inline bool allowLoopShapeHeuristics(bool bypassShapeRegistry,
                                     llvm::StringRef funcName,
                                     bool enablePreload,
                                     int64_t workspaceMultiBuffer) {
  return bypassShapeRegistry ||
         (enablePreload && workspaceMultiBuffer != 0 &&
          isLoopShapeRegistered(funcName));
}

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_UTILS_SHAPEREGISTRY_H
