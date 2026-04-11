//===- TritonOps.td - BishengIR Triton dialect extensions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Extends the upstream Triton dialect with bishengir-specific ops.
// These ops live in the `tt` namespace and are registered into the Triton
// dialect via a DialectRegistry extension (see TritonExtension.h).
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRITON_OPS_TD
#define BISHENGIR_TRITON_OPS_TD

include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// tt.call_scalar
//===----------------------------------------------------------------------===//
// Pre-allocation form of a scalar-mode function call backed by shared memory.
// SIMTFastDiv emits this op; PopulateSharedMemoryOffsetToDPX converts it to
// ascend_dpx.call_scalar once the allocator has assigned an offset.
//
// The shm_desc operand is a memdesc<1xi32> produced by ttg.local_alloc.  It
// conveys the shared-memory resource to the allocator; after allocation the
// "allocation.offset" attribute is attached to the local_alloc and then
// propagated to the replacement ascend_dpx.call_scalar.

def CallScalarOp : Op<Triton_Dialect, "call_scalar",
    [DeclareOpInterfaceMethods<CallOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Scalar-function call with shared memory descriptor";
  let description = [{
    Calls a scalar helper function.

    Example:
    ```mlir
    %res = tt.call_scalar %shm, @simt_div_magic_shift(%divisor)
        : !triton_gpu.memdesc<1xi32, ...>, (i32) -> i32
    ```
  }];

  let arguments = (ins
    TTG_MemDescType:$shm_desc,
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$call_args,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let results = (outs Variadic<AnyType>:$results);

  // Use qualified() so the memdesc type prints as "!triton_gpu.memdesc<...>"
  // rather than just "<...>".
  let assemblyFormat = [{
    $shm_desc `,` $callee `(` $call_args `)` attr-dict
    `:` qualified(type($shm_desc)) `,` functional-type($call_args, results)
  }];
}

#endif // BISHENGIR_TRITON_OPS_TD
