//===- AivDisabledPasses.cpp - Aiv-mode pass denylist -----------*- C++ -*-===//
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
//
// RegBase (HIVM) entries of the Aiv-mode pass denylist. The generic denylist
// mechanism and the Aiv filter live in bishengir/Pass/PassExecutionPolicy.h;
// this file owns the domain-specific data, kept in its own translation unit so
// it can be reviewed (and unit-tested) without pulling in the pipeline
// construction code.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Pipelines/regbase/Passes.h"
#include "bishengir/Pass/PassExecutionPolicy.h"

namespace mlir {
namespace hivm {
namespace regbase {

void registerAivDisabledPassesForRegbase() {
  // Idempotent: the registry is a set and this runs once per process. Called
  // from the public RegBase pipeline builders, so it does not depend on the
  // linker keeping a file-scope initializer.
  static const bool registered = [] {
    // Passes the RegBase pipeline must not run on a pure-AIV
    // (`mix_mode = "aiv"`) module: cube-side lowering and mix-CV cross-core
    // coordination. Grouped by pipeline area:
    //   - cube pipeline stages: hivm-normalize-matmul, hivm-normalize-convops,
    //     insert-workspace-for-mix-cv, hivm-split-mix-kernel,
    //     hivm-tile-batchmm-into-loop, hivm-insert-fixpipe,
    //     hivm-inline-fixpipe, hivm-combine-optimized-convert-layout;
    //   - mix-CV cross-core sync machinery: hivm-cross-core-gss,
    //     hivm-delayed-cross-core-gss, hivm-insert-anchors-and-backup,
    //     hivm-inject-block-sync, hivm-bind-sub-block;
    //   - CV-pipelining helpers that need a cube side: create-preload,
    //     mark-simt-scope-no-inline;
    //   - debug instrumentation tied to the mix pipeline:
    //     hivm-insert-l12ub-for-debug, hivm-insert-nz2nd-for-debug.
    //
    // Deliberately NOT disabled:
    //   - mark-real-core-type: drives the whole flow and self-gates on the
    //     function core types;
    //   - hivm-insert-cv-tight-coupled-buffer: only active with the
    //     corresponding layout options;
    //   - hivm-insert-load-store-for-mix-cv / hivm-infer-func-core-type:
    //     they still do the AIV-side work in Aiv mode.
    //
    // This is a denylist: a pass added to the pipeline later keeps running in
    // Aiv mode unless it is added here.
    bishengir::registerAivDisabledPasses({
        "hivm-normalize-matmul",
        "hivm-normalize-convops",
        "insert-workspace-for-mix-cv",
        "hivm-split-mix-kernel",
        "hivm-tile-batchmm-into-loop",
        "hivm-insert-fixpipe",
        "hivm-inline-fixpipe",
        "hivm-combine-optimized-convert-layout",
        "hivm-cross-core-gss",
        "hivm-delayed-cross-core-gss",
        "hivm-insert-anchors-and-backup",
        "hivm-inject-block-sync",
        "hivm-bind-sub-block",
        "create-preload",
        "mark-simt-scope-no-inline",
        "hivm-insert-l12ub-for-debug",
        "hivm-insert-nz2nd-for-debug",
    });
    return true;
  }();
  (void)registered;
}

} // namespace regbase
} // namespace hivm
} // namespace mlir
