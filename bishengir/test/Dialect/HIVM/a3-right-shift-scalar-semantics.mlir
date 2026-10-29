// RUN: bishengir-opt %s -hivm-lower-to-loops | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910_9391">} {
  // CHECK-LABEL: func.func @shrui
  // CHECK: arith.shrui
  // CHECK-NOT: arith.shrsi
  // CHECK: return
  func.func @shrui(%a: memref<16xi32>, %b: memref<16xi32>, %out: memref<16xi32>) {
    hivm.hir.vshr ins(%a, %b : memref<16xi32>, memref<16xi32>)
      outs(%out : memref<16xi32>) round : false is_signed : false
    return
  }
  // CHECK-LABEL: func.func @shrsi
  // CHECK: arith.shrsi
  // CHECK-NOT: arith.shrui
  // CHECK: return
  func.func @shrsi(%a: memref<16xi32>, %b: memref<16xi32>, %out: memref<16xi32>) {
    hivm.hir.vshr ins(%a, %b : memref<16xi32>, memref<16xi32>)
      outs(%out : memref<16xi32>) round : false is_signed : true
    return
  }
}
