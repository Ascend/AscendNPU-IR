//===- Passes.td - TritonAscendGPUToLLVM passes def. file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRITONASCENDGPUTOLLVM_PASSES_H
#define BISHENGIR_TRITONASCENDGPUTOLLVM_PASSES_H

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TritonAscendGPUToLLVM
//===----------------------------------------------------------------------===//

def AllocateAscendSharedMemory :
    Pass<"allocate-ascend-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation and adds memory barriers";

  let description = [{
    This pass uses the `ModuleAllocation` analysis to:
      - Annotate modules with an attribute with the amount of shared/local
        memory used.
      - Annotate operations with an offset into the total shared/local memory.
  }];
}

#endif // BISHENGIR_TRITONASCENDGPUTOLLVM_PASSES_H
