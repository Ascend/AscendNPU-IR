//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def FoldAllocReshapeOp
    : Pass<"fold-alloc-reshape", "func::FuncOp"> {
  let summary = "Fold alloc and reshape ops";
  let constructor = "mlir::memref::createFoldAllocReshapePass()";
  let dependentDialects = [
    "memref::MemRefDialect",
  ];
}

def MemrefDeadStoreEliminationOp
    : Pass<"memref-dse", "func::FuncOp"> {
  let summary = "Erase dead alloc and stores";
  let constructor = "mlir::memref::createDeadStoreEliminationPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
  ];
}

def RemoveRedundantCopy : Pass<"memref-remove-redundant-copy", "func::FuncOp"> {
  let summary = "Remove memref redundant copy operations";
  let constructor = "mlir::memref::createRemoveRedundantCopyPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
  ];
}

def BindBuffer : Pass<"bind-buffer", "func::FuncOp"> {
  let constructor = "mlir::memref::createBindBufferPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "annotation::AnnotationDialect",
  ];
}

#endif // BISHENGIR_DIALECT_MEMREF_TRANSFORMS_PASSES_TD
