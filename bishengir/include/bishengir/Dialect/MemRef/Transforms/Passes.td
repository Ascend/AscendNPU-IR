//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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