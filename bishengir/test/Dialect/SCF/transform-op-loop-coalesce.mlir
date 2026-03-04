// RUN: bishengir-opt -split-input-file -transform-interpreter -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: coalesce_single_loop
// CHECK: scf.for
// CHECK-NOT: scf.for
// CHECK: transform.named_sequence @__transform_main
func.func @coalesce_single_loop(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %cst10 = arith.constant 10 : index
  %cst1_i32 = arith.constant 1 : i32
  %empty = tensor.empty() : tensor<10xi32>

  scf.for %index = %cst0 to %cst10 step %cst1 {
    %load = tensor.extract %arg0[%index] : tensor<10xi32>
    %inc = arith.addi %load, %cst1_i32 : i32
    tensor.insert %inc into %empty[%index] : tensor<10xi32>
  } {coalesce}
  return %empty : tensor<10xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1: (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}
