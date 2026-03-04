// RUN: bishengir-opt -transform-interpreter -split-input-file -allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s
// RUN: bishengir-opt -hfusion-simt-loop-tile %s | FileCheck %s

module attributes { transform.with_named_sequence, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @vadd(%arg0: memref<16x32xf16>, %arg1: memref<16x32xf16>, %arg2: memref<16x32xf16>) -> memref<16x32xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    // CHECK: scf.parallel
    // CHECK: scf.parallel
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c16, %c32) step (%c1, %c1) {
      %0 = memref.load %arg0[%arg3, %arg4] : memref<16x32xf16>
      %1 = memref.load %arg1[%arg3, %arg4] : memref<16x32xf16>
      %2 = arith.addf %0, %1 : f16
      memref.store %2, %arg2[%arg3, %arg4] : memref<16x32xf16>
      scf.reduce 
    } {tile}
    return %arg2 : memref<16x32xf16>
  }
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %1 = transform.structured.match ops{["scf.parallel"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.op<"scf.parallel">
    %outer, %inner = transform.parallel_loop.tile %1 { static_sizes = [16, 32] } : (!transform.op<"scf.parallel">) -> (!transform.op<"scf.parallel">, !transform.op<"scf.parallel">)
    transform.yield
  }
}

