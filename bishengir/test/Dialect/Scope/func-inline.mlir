// RUN: bishengir-opt --inline %s | FileCheck %s

// CHECK:   func.func @inline_func(%[[VAL_0:.*]]: tensor<64x128xf16>) {
// CHECK:           %[[C4096:.*]] = arith.constant 4096 : i32
// CHECK:           %[[C0:.*]] = arith.constant 0 : i32
// CHECK:           %[[C64:.*]] = arith.constant 64 : i32
// CHECK:           %[[C20:.*]] = arith.constant 20 : i32
// CHECK:           scf.for %[[FOR_INIT_0:.*]] = %[[C0]] to %[[C64]] step %[[C20]]
// CHECK-NOT: func.func private @func_a
// CHECK:             scope.scope : () -> () {
// CHECK:               scf.for %[[FOR_INIT_1:.*]] = %[[C0]] to %[[C4096]] step %[[C64]]
// CHECK:                 hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK:               scope.return
// CHECK:             } {core_mode = "cube"}
// CHECK:             scope.scope : () -> () {
// CHECK:               scope.scope : () -> () {
// CHECK:                 hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK:                 scope.return
// CHECK:               scope.return
// CHECK:             } {core_mode = "vector"}
// CHECK-NOT: func.func private @func_b
// CHECK:             scope.scope : () -> () {
// CHECK:               hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
// CHECK:               scope.return
// CHECK:             } {core_mode = "vector"}
// CHECK:           return

module {
  func.func @inline_func(%arg0: tensor<64x128xf16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c20_i32 = arith.constant 20 : i32
    scf.for %arg3 = %c0_i32 to %c64_i32 step %c20_i32  : i32 {
      %0 = func.call @inline_func_a(%cst) : (tensor<64x128xf32>) -> tensor<64x128xf32>
      %1 = func.call @inline_func_b(%0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    }
    return
  }
  func.func private @inline_func_a(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4096_i32 = arith.constant 4096 : i32
    scope.scope : () -> () {
      scf.for %arg1 = %c0_i32 to %c4096_i32 step %c64_i32  : i32 {
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
      }
      scope.return
    } {core_mode = "cube"}
    scope.scope : () -> () {
      scope.scope : () -> () {
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
        scope.return
      }
      scope.return
    } {core_mode = "vector"}
    return %arg0 : tensor<64x128xf32>
  }
  func.func private @inline_func_b(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> attributes {noinline = false} {
    scope.scope : () -> () {
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
      scope.return
    } {core_mode = "vector"}
    return %arg0 : tensor<64x128xf32>
  }
}
