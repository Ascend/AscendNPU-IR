// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s
// RUN: bishengir-opt %s -hivm-reposition-preload-work="bypass-shape-registry=true" -hivm-reposition-preload-work="bypass-shape-registry=true" | FileCheck %s

// Bufferization copies a yielded preload result into the iteration buffer at
// scope.return. Reading the old state in the tail would then see the new state.
// Keep the subtraction before that writeback; its independent exp can move.
// CHECK-LABEL: func.func @preserve_old_state(
// CHECK: %[[PRODUCER:.*]]:2 = scope.scope
// CHECK: hivm.hir.vmax
// CHECK: hivm.hir.vsub
// CHECK-NOT: hivm.hir.vexp
// CHECK: hivm.hir.copy
// CHECK: scope.return
// CHECK: scope.scope
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.vadd
func.func @preserve_old_state(%init: tensor<16xf32>, %q: tensor<16xf32>, %dst: memref<16xf32, #hivm.address_space<cbuf>>, %n: index) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %n step %c1 iter_args(%state = %init) -> tensor<16xf32> {
    %r:2 = scope.scope : () -> (tensor<16xf32>, tensor<16xf32>) {
      %e = tensor.empty() : tensor<16xf32>
      %new = hivm.hir.vmax ins(%state, %q : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      %delta = hivm.hir.vsub ins(%state, %new : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      %alpha = hivm.hir.vexp ins(%delta : tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      hivm.hir.copy ins(%new : tensor<16xf32>) outs(%dst : memref<16xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return %new, %alpha : tensor<16xf32>, tensor<16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<16xf32>
      %use = hivm.hir.vadd ins(%r#1, %r#1 : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %r#0 : tensor<16xf32>
  }
  return %out : tensor<16xf32>
}

// CHECK-LABEL: func.func @preserve_old_state_view(
// CHECK: %[[PRODUCER:.*]]:2 = scope.scope
// CHECK: hivm.hir.vmax
// CHECK: hivm.hir.vsub
// CHECK-NOT: hivm.hir.vexp
// CHECK: hivm.hir.copy
// CHECK: scope.return
// CHECK: scope.scope
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.vadd
func.func @preserve_old_state_view(%init: tensor<16xf32>, %q: tensor<16xf32>, %dst: memref<16xf32, #hivm.address_space<cbuf>>, %n: index) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %n step %c1 iter_args(%state = %init) -> tensor<16xf32> {
    %r:2 = scope.scope : () -> (tensor<16xf32>, tensor<1x16xf32>) {
      %e = tensor.empty() : tensor<16xf32>
      %new = hivm.hir.vmax ins(%state, %q : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      %old_view = tensor.expand_shape %state [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
      %new_view = tensor.expand_shape %new [[0, 1]] output_shape [1, 16] : tensor<16xf32> into tensor<1x16xf32>
      %empty_view = tensor.empty() : tensor<1x16xf32>
      %delta = hivm.hir.vsub ins(%old_view, %new_view : tensor<1x16xf32>, tensor<1x16xf32>) outs(%empty_view : tensor<1x16xf32>) -> tensor<1x16xf32>
      %alpha = hivm.hir.vexp ins(%delta : tensor<1x16xf32>) outs(%empty_view : tensor<1x16xf32>) -> tensor<1x16xf32>
      hivm.hir.copy ins(%new : tensor<16xf32>) outs(%dst : memref<16xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      scope.return %new, %alpha : tensor<16xf32>, tensor<1x16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<1x16xf32>
      %use = hivm.hir.vadd ins(%r#1, %r#1 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%e : tensor<1x16xf32>) -> tensor<1x16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %r#0 : tensor<16xf32>
  }
  return %out : tensor<16xf32>
}


// Splitting after a copy must also preserve reads before state writeback.
// CHECK-LABEL: func.func @preserve_old_state_after_copy(
// CHECK: %[[PRODUCER:.*]]:2 = scope.scope
// CHECK: hivm.hir.vmax
// CHECK: hivm.hir.copy
// CHECK: hivm.hir.vsub
// CHECK-NOT: hivm.hir.vexp
// CHECK: scope.return
// CHECK: scope.scope
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.vadd
func.func @preserve_old_state_after_copy(%init: tensor<16xf32>, %q: tensor<16xf32>, %dst: memref<16xf32, #hivm.address_space<cbuf>>, %n: index) -> tensor<16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %out = scf.for %i = %c0 to %n step %c1 iter_args(%state = %init) -> tensor<16xf32> {
    %r:2 = scope.scope : () -> (tensor<16xf32>, tensor<16xf32>) {
      %e = tensor.empty() : tensor<16xf32>
      %new = hivm.hir.vmax ins(%state, %q : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      hivm.hir.copy ins(%new : tensor<16xf32>) outs(%dst : memref<16xf32, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
      %delta = hivm.hir.vsub ins(%state, %new : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      %alpha = hivm.hir.vexp ins(%delta : tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      scope.return %new, %alpha : tensor<16xf32>, tensor<16xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<16xf32>
      %use = hivm.hir.vadd ins(%r#1, %r#1 : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %r#0 : tensor<16xf32>
  }
  return %out : tensor<16xf32>
}
