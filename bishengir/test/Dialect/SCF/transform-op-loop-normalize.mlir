// RUN: bishengir-opt -transform-interpreter -split-input-file -allow-unregistered-dialect -verify-diagnostics %s | FileCheck %s

func.func @normalize_loop(%IN1 : memref<256xf32>, %IN2 : memref<256xf32>, %OUT : memref<256xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[CONST256:.*]] = arith.constant 256 : index
  %c256 = arith.constant 256 : index
  // CHECK: %[[CONST128:.*]] = arith.constant 128 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CONST1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[CONST1_1:.*]] = arith.constant 1 : index
  // CHECK: %[[MAX:.*]] = affine.apply #map()[%[[CONST256]], %[[CONST128]]]
  // CHECK: scf.for {{.*}} = {{.*}} to %[[MAX]] step %[[CONST1_1]] {
  scf.for %i = %c0 to %c256 step %c128 {
    %ub = arith.addi %i, %c128 : index
    scf.for %j = %i to %ub step %c1 {
      %0 = memref.load %IN1[%j] : memref<256xf32>
      %1 = memref.load %IN2[%j] : memref<256xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %OUT[%j] : memref<256xf32>
    }
  } {target}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {target} in %arg0 : (!transform.any_op) -> !transform.any_op
    %loop = transform.loop.normalize %0  : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

