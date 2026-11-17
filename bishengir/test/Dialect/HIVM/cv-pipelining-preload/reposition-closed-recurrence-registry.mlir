// RUN: bishengir-opt %s -hivm-reposition-preload-work="enable-preload=true set-workspace-multibuffer=4" | FileCheck %s

// A mixed module must retain closed-recurrence sinking for a registered
// function while leaving the unregistered function's scopes and yields intact.
// CHECK-LABEL: func.func @kernel_sdpa_bwd_q_closed_recurrence
// CHECK: %[[REG_EARLY:.*]]:2 = scope.scope
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.vreduce
// CHECK-NOT: hivm.hir.vmul
// CHECK-NOT: hivm.hir.vadd
// CHECK: scope.return
// CHECK: } {{.*}}hivm.preload_num = 2 : i32
// CHECK: %[[REG_LATE:.*]]:2 = scope.scope
// CHECK: hivm.hir.vmul
// CHECK: hivm.hir.vadd
// CHECK: hivm.hir.vmul
// CHECK: scope.return
// CHECK: } {{.*}}hivm.preload_num = 0 : i32
// CHECK: scf.yield %[[REG_LATE]]#1, %[[REG_LATE]]#0

// CHECK-LABEL: func.func @unregistered_closed_recurrence
// CHECK: %[[OFF_EARLY:.*]]:2 = scope.scope
// CHECK: hivm.hir.vexp
// CHECK: hivm.hir.vreduce
// CHECK: hivm.hir.vmul
// CHECK: hivm.hir.vadd
// CHECK: scope.return
// CHECK: } {{.*}}hivm.preload_num = 2 : i32
// CHECK: %[[OFF_LATE:.*]] = scope.scope
// CHECK: hivm.hir.vmul
// CHECK-NOT: hivm.hir.vadd
// CHECK: scope.return
// CHECK: } {{.*}}hivm.preload_num = 0 : i32
// CHECK: scf.yield %[[OFF_EARLY]]#1, %[[OFF_LATE]]

!row = tensor<16xf32>
!tile = tensor<16x8xf32>

func.func @kernel_sdpa_bwd_q_closed_recurrence(%x: !row, %tile: !tile)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : !row
  %initial = tensor.empty() : !row
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!row, !row) {
    %prod:2 = scope.scope : () -> (!row, !row) {
      %factor = hivm.hir.vexp ins(%x : !row) outs(%empty : !row) -> !row
      %suminit = tensor.empty() : tensor<16x1xf32>
      %sum = hivm.hir.vreduce <sum> ins(%tile : !tile) outs(%suminit : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
      %flat = tensor.collapse_shape %sum [[0, 1]] : tensor<16x1xf32> into !row
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %flat : !row, !row) outs(%empty : !row) -> !row
      scope.return %factor, %next : !row, !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
    %late = scope.scope : () -> !row {
      %scaled = hivm.hir.vmul ins(%acc, %prod#0 : !row, !row) outs(%empty : !row) -> !row
      scope.return %scaled : !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %prod#1, %late : !row, !row
  } {hivm.cv_pipelined_loop}
  return
}

func.func @unregistered_closed_recurrence(%x: !row, %tile: !tile)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : !row
  %initial = tensor.empty() : !row
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!row, !row) {
    %prod:2 = scope.scope : () -> (!row, !row) {
      %factor = hivm.hir.vexp ins(%x : !row) outs(%empty : !row) -> !row
      %suminit = tensor.empty() : tensor<16x1xf32>
      %sum = hivm.hir.vreduce <sum> ins(%tile : !tile) outs(%suminit : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
      %flat = tensor.collapse_shape %sum [[0, 1]] : tensor<16x1xf32> into !row
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %flat : !row, !row) outs(%empty : !row) -> !row
      scope.return %factor, %next : !row, !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
    %late = scope.scope : () -> !row {
      %scaled = hivm.hir.vmul ins(%acc, %prod#0 : !row, !row) outs(%empty : !row) -> !row
      scope.return %scaled : !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %prod#1, %late : !row, !row
  } {hivm.cv_pipelined_loop}
  return
}
