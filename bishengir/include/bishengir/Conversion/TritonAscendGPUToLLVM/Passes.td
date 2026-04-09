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

def SinkDPXLoad :
    Pass<"sink-dpx-load", "mlir::ModuleOp"> {
  let summary = "Sink ascend_dpx.load ops closer to their first users";

  let description = [{
    This pass reduces register pressure by reordering instructions within basic
    blocks after the TritonAscendGPUToLLVM lowering. The lowering expands tensor
    operations into per-thread scalar operations, processing each tensor op
    independently: all loads for tensor A, then all loads for tensor B, then all
    divs, then all stores. This causes all loaded values to be live
    simultaneously at the point of first use, leading to register spill.

    This pass sinks ascend_dpx.load operations (and their address computation
    dependencies) to just before their first user, naturally interleaving
    load-compute-store chains and dramatically reducing peak register pressure.
  }];
}

#endif // BISHENGIR_TRITONASCENDGPUTOLLVM_PASSES_H
