// RUN: bishengir-opt --inline-scope --split-input-file %s | FileCheck %s

// CHECK:   func.func @inline_func(%[[ARG_0:.*]]: tensor<64x128xf32>)
// CHECK-DAG:           %[[END_2:.*]] = arith.constant {debug = 12 : index} 4096 : index
// CHECK-DAG:           %[[INIT:.*]] = arith.constant {debug = 1 : index} 0 : index
// CHECK-DAG:           %[[END_1:.*]] = arith.constant {debug = 2 : index} 64 : index
// CHECK-DAG:           %[[STEP:.*]] = arith.constant {debug = 3 : index} 20 : index
// CHECK:           scf.for %[[_:.*]] = %[[INIT]] to %[[END_1]] step %[[STEP]] {
// CHECK:             scf.for %[[_:.*]] = %[[INIT]] to %[[END_2]] step %[[STEP]] {
// CHECK:               hivm.hir.sync_block_set {debug = 15 : index}
// CHECK:             hivm.hir.sync_block_wait {debug = 20 : index}
// CHECK:             scope.scope : () -> () {
// CHECK:               hivm.hir.sync_block_wait {debug = 26 : index}
// CHECK:               scope.return
// CHECK:             } {core_mode = "vector", no_inline}
// CHECK:           return {debug = 8 : index} %[[ARG_0]] : tensor<64x128xf32>

module {
  func.func @inline_func(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {debug = 0 : index} {
    %c0 = arith.constant {debug = 1 : index} 0 : index
    %c64 = arith.constant {debug = 2 : index} 64 : index
    %c20 = arith.constant {debug = 3 : index} 20 : index
    %0 = scf.for %arg1 = %c0 to %c64 step %c20 iter_args(%arg2 = %arg0) -> (tensor<64x128xf32>) {
      %1 = func.call @inline_func_a(%arg2) : (tensor<64x128xf32>) -> tensor<64x128xf32>
      %2 = func.call @inline_func_b(%1) : (tensor<64x128xf32>) -> tensor<64x128xf32>
      scf.yield %2 : tensor<64x128xf32>
    }
    return {debug = 8 : index} %0 : tensor<64x128xf32>
  }
  func.func private @inline_func_a(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {noinline = false} {
    %c0 = arith.constant {debug = 10 : index} 0 : index
    %c20 = arith.constant {debug = 11 : index} 20 : index
    %c4096 = arith.constant {debug = 12 : index} 4096 : index
    scope.scope : () -> () {
      scf.for %arg1 = %c0 to %c4096 step %c20 {
        hivm.hir.sync_block_set {debug = 15 : index}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
      }
      scope.return
    } {core_mode = "cube"}
    scope.scope : () -> () {
      scope.scope : () -> () {
        hivm.hir.sync_block_wait {debug = 20 : index}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        scope.return
      }
      scope.return
    } {core_mode = "vector"}
    return %arg0 : tensor<64x128xf32>
  }
  func.func private @inline_func_b(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {noinline = false} {
    scope.scope : () -> () {
      hivm.hir.sync_block_wait {debug = 26 : index}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
      scope.return
    } {core_mode = "vector", no_inline}
    return %arg0 : tensor<64x128xf32>
  }
}

// -----

// CHECK:   func.func @extract_multiple_ops(%[[ARG_0:.*]]: tensor<i32>, %[[ARG_1:.*]]: tensor<i32>, %[[ALLOC:.*]]: memref<i32>) {
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG_0]], %[[ARG_1]] : tensor<i32>
// CHECK:           %[[MUL:.*]] = arith.muli %[[ADD]], %[[ADD]] : tensor<i32>
// CHECK:           hivm.hir.store ins(%[[MUL]] : tensor<i32>) outs(%[[ALLOC]] : memref<i32>)
// CHECK:           return
// CHECK:         }
module {
  func.func @extract_multiple_ops(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: memref<i32>) {
    scope.scope : () -> () {
      %0 = arith.addi %arg0, %arg1 : tensor<i32>
      %1 = arith.muli %0, %0 : tensor<i32>
      hivm.hir.store ins(%1 : tensor<i32>) outs(%arg2 : memref<i32>)
      scope.return
    } {hivm.t_core_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

// -----

// CHECK: func.func @inline_func(%[[VAL_0:.*]]: tensor<f32>, %[[VAL_1:.*]]: tensor<i32>) -> (tensor<i32>, tensor<f32>) attributes {debug = 0 : index} {
// CHECK-NOT: scope.scope
// CHECK-NOT: func.func private @inline_func_a(
// CHECK-NOT: scope.scope
// CHECK: return {debug = 8 : index} %[[VAL_1]], %[[VAL_0]] : tensor<i32>, tensor<f32>
module {
  func.func @inline_func(%arg0: tensor<f32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<f32>) attributes {debug = 0 : index} {
    %0:2 = scope.scope : () -> (tensor<f32>, tensor<i32>) {
      %1:2 = func.call @inline_func_a(%arg0, %arg1) : (tensor<f32>, tensor<i32>) -> (tensor<i32>, tensor<f32>)
      scope.return %1#1, %1#0 : tensor<f32>, tensor<i32>
    }
    return {debug = 8 : index} %0#1, %0#0 : tensor<i32>, tensor<f32>
  }
  func.func private @inline_func_a(%arg0: tensor<f32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<f32>) attributes {noinline = false} {
    %0:2 = scope.scope : () -> (tensor<f32>, tensor<i32>) {
      scope.return %arg0, %arg1 : tensor<f32>, tensor<i32>
    }
    %1:2 = scope.scope : () -> (tensor<f32>, tensor<i32>) {
      %2:2 = scope.scope : () -> (tensor<i32>, tensor<f32>) {
        scope.return %0#1, %0#0 : tensor<i32>, tensor<f32>
      }
      scope.return %2#1, %2#0 : tensor<f32>, tensor<i32>
    }
    return %1#1, %1#0 : tensor<i32>, tensor<f32>
  }
}

