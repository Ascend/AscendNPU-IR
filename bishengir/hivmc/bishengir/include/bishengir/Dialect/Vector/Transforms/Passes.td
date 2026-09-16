//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def VectorTransferLowering
    : Pass<"vector-transfer-lowering", "func::FuncOp"> {
  let summary = "Pass to lower transfer ops to simpler ops like `vector.load` "
                "`vector.store`, `vector.broadcast`";
  let description = [{
    This pass replace all transfer.read and transfer write to vector.load and
    vector.write.
  }];
  let constructor = "mlir::vector::createVectorTransferLoweringPass()";
  let dependentDialects = [
    "vector::VectorDialect"
  ];
}

def PeelLoopsContainingTranspose
    : Pass<"peel-loops-containing-transpose", "func::FuncOp"> {
  let summary = "Peel loops which contains vector.transfer_read which "
                "is transpose load.";
  let constructor =
    "mlir::vector::createPeelLoopsContainingTransposePass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "vector::VectorDialect",
  ];
}

def FoldInsertSliceToTransferWrite : Pass<"fold-insert-slice-to-transfer-write", "func::FuncOp"> {
  let summary = "fold insert-slice to transfer-write to avoid write to empty tensor";
  let constructor = "::mlir::vector::createFoldInsertSliceToTransferWritePass()";
  let dependentDialects = ["tensor::TensorDialect", "arith::ArithDialect",
                           "vector::VectorDialect"];
}


#endif // BISHENGIR_DIALECT_VECTOR_TRANSFORMS_PASSES_TD
