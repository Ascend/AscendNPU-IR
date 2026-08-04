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

def ConvertDebugOpToAscendDPX :
    Pass<"convert-debug-to-ascenddpx", "mlir::ModuleOp"> {
  let summary = "Unswizzle shared tensors passed to tt.device_print / tt.assert";

  let description = [{
    Converts tensors with swizzled shared memory encodings to natural
    (unswizzled) layout before tt.device_print / tt.assert ops. We use this
    in SIMT mode to read tensors in logical row-major order (for printing
    and assertion checking).
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::TritonDialect",
    "mlir::ascend_dpx::AscendDPXDialect",
  ];
}

def EraseTritonDebugOps :
    Pass<"erase-triton-debug-ops", "mlir::ModuleOp"> {
  let summary = "Erase Triton device print and assert operations";

  let description = [{
    Erases all tt.print and tt.assert operations so disabled SIMT device
    debugging preserves the existing runtime behavior and performance.
  }];
}

#endif // BISHENGIR_TRITONASCENDGPUTOLLVM_PASSES_H
