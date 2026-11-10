// RUN: bishengir-opt %s -hivm-mark-preload-buffers -outline-scope="outline-marked-scopes-only=true" | FileCheck %s --implicit-check-not=mem_unique

// Same-stage tasks keep source order, and only the inner vector function is outlined.
// CHECK-LABEL: func.func @inplace_tiles
// CHECK: scf.for
// CHECK: annotation.mark {{.*}}hivm.multi_buffer = 3 : i32
// CHECK: hivm.task_name = "load", no_inline
// CHECK: scope.scope
// CHECK: call @
// CHECK-SAME: hivm.vector_function
// CHECK: hivm.preload_num = 1 : i32, hivm.task_name = "double", no_inline
// CHECK: arith.mulf
// CHECK: hivm.preload_num = 1 : i32, hivm.task_name = "square", no_inline
// CHECK: hivm.task_name = "store", no_inline
// CHECK: hivm.cv_pipelined_loop
func.func @inplace_tiles(%gm: memref<?xf32>, %tiles: index)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %tiles step %c1 {
    %tile = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    %offset = arith.muli %i, %c64 : index
    %view = memref.reinterpret_cast %gm to offset: [%offset],
        sizes: [64], strides: [1] : memref<?xf32> to
        memref<64xf32, strided<[1], offset: ?>>
    scope.scope : () -> () {
      hivm.hir.load ins(%view : memref<64xf32, strided<[1], offset: ?>>)
          outs(%tile : memref<64xf32, #hivm.address_space<ub>>)
      scope.return
    } {hivm.task_name = "load", hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.scope : () -> () {
        %x = memref.load %tile[%c0] : memref<64xf32, #hivm.address_space<ub>>
        %y = arith.addf %x, %x : f32
        memref.store %y, %tile[%c0] : memref<64xf32, #hivm.address_space<ub>>
        scope.return
      } {outline, hivm.vector_function}
      scope.return
    } {hivm.task_name = "double", hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %x = memref.load %tile[%c0] : memref<64xf32, #hivm.address_space<ub>>
      %y = arith.mulf %x, %x : f32
      memref.store %y, %tile[%c0] : memref<64xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.task_name = "square", hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      hivm.hir.store ins(%tile : memref<64xf32, #hivm.address_space<ub>>)
          outs(%view : memref<64xf32, strided<[1], offset: ?>>)
      scope.return
    } {hivm.task_name = "store", hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}
