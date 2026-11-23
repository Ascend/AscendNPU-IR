// RUN: bishengir-opt %s -one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s

// A tensor accumulator and an explicit scalar signal are mixed tensor/buffer
// semantics: the signal is already a memref and threads through unchanged while
// C bufferizes as a regular DPS init.

// CHECK-LABEL: func.func @bufferize_mmadL1(
// CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
func.func @bufferize_mmadL1(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>,
    %c: tensor<16x16xf32>, %signal: memref<i32>, %init: i1, %n: index)
    -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  // CHECK: memref.alloc
  // CHECK: hivm.hir.mmadL1
  // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
  %result = hivm.hir.mmadL1
      ins(%a, %b, %init, %n, %n, %n :
          tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
      outs(%c : tensor<16x16xf32>)
      ping_pong_signal(%signal : memref<i32>) -> tensor<16x16xf32>
  return %c, %result : tensor<16x16xf32>, tensor<16x16xf32>
}

// CHECK-LABEL: func.func @bufferize_batchMmadL1(
// CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
func.func @bufferize_batchMmadL1(%a: tensor<1x16x16xf16>, %b: tensor<1x16x16xf16>,
    %c: tensor<1x16x16xf32>, %signal: memref<i32>, %init: i1, %n: index)
    -> (tensor<1x16x16xf32>, tensor<1x16x16xf32>) {
  // CHECK: memref.alloc
  // CHECK: hivm.hir.batchMmadL1
  // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
  %result = hivm.hir.batchMmadL1
      ins(%a, %b, %init, %n, %n, %n :
          tensor<1x16x16xf16>, tensor<1x16x16xf16>, i1, index, index, index)
      outs(%c : tensor<1x16x16xf32>)
      ping_pong_signal(%signal : memref<i32>) -> tensor<1x16x16xf32>
  return %c, %result : tensor<1x16x16xf32>, tensor<1x16x16xf32>
}

// CHECK-LABEL: func.func @bufferize_mmadMxL1(
// CHECK-SAME: %[[SIGNAL:[^ ,]+]]: memref<i32>
func.func @bufferize_mmadMxL1(%a: tensor<16x16xi8>, %b: tensor<16x16xi8>,
    %c: tensor<16x16xf32>, %scale: tensor<16xi8>, %signal: memref<i32>, %init: i1,
    %n: index) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  // CHECK: memref.alloc
  // CHECK: hivm.hir.mmadmxL1
  // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
  %result = hivm.hir.mmadmxL1
      ins(%a, %b, %scale, %scale, %init, %n, %n, %n :
          tensor<16x16xi8>, tensor<16x16xi8>, tensor<16xi8>, tensor<16xi8>, i1,
          index, index, index)
      outs(%c : tensor<16x16xf32>)
      ping_pong_signal(%signal : memref<i32>) -> tensor<16x16xf32>
  return %c, %result : tensor<16x16xf32>, tensor<16x16xf32>
}
