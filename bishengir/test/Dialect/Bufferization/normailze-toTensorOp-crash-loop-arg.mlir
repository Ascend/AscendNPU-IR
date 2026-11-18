// RUN: not --crash bishengir-opt %s -normalize-to-tensor 2>&1 | FileCheck %s

// Negative: a writable to_tensor whose memref operand is a block argument of
// a nested region (here the loop-carried iter_arg of an scf.for) is not a
// valid root: only function-entry arguments, view-like chains ending in an
// allocation and alloc_workspace are accepted, so the write tracing reports
// the unexpected value and aborts.

// CHECK: <block argument> of type 'memref<16xf32>'
// CHECK: LLVM ERROR: unexpected value when check memref written before to_tensor
func.func @to_tensor_of_loop_iter_arg(%arg0: memref<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %mres, %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%m = %arg0, %acc = %arg1) -> (memref<16xf32>, tensor<16xf32>) {
    %t = bufferization.to_tensor %m restrict writable : memref<16xf32>
    scf.yield %m, %t : memref<16xf32>, tensor<16xf32>
  }
  return %acc : tensor<16xf32>
}
