// REQUIRES: regbase
// TODO: enable after migrating CustomOp base + bufferization dialect loading
// RUN: bishengir-opt --inline-scope --split-input-file %s | FileCheck %s

// CHECK: scope.scope
module {
  func.func @simt_scope_ops(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: memref<i32>) {
    scope.scope : () -> () {
      %0 = arith.addi %arg0, %arg1 : tensor<i32>
      %1 = arith.muli %0, %0 : tensor<i32>
      hivm.hir.store ins(%1 : tensor<i32>) outs(%arg2 : memref<i32>)
      scope.return
    } { vector_type = "simt"}
    return
  }
}
