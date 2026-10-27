// A tensor iter-arg is not sunk across preload stages, so the recurrence
// stays in the producer.
// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work="enable-preload=true set-workspace-multibuffer=4" | FileCheck %s --check-prefixes=BEFORE,COMMON
// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work="enable-preload=true set-workspace-multibuffer=4" | bishengir-opt -hivm-sink-exclusive-preload-work="enable-preload=true set-workspace-multibuffer=4" | FileCheck %s --check-prefixes=BEFORE,COMMON
// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work | FileCheck %s --check-prefixes=BEFORE,COMMON
// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work="enable-preload=true set-workspace-multibuffer=0" | FileCheck %s --check-prefixes=BEFORE,COMMON

!row = tensor<16xf32>
!tile = tensor<16x8xf32>
!half_row = tensor<7xf16>
!half_tile = tensor<7x5xf16>
!ub = memref<16xf32, #hivm.address_space<ub>>
!cbuf = memref<16xf32, #hivm.address_space<cbuf>>

// Move a closed recurrence and forward its reduction, preserving stage numbers.
// AFTER-LABEL: func.func @closed_tensor_recurrence
// AFTER: scf.for {{.*}} iter_args(%[[STATE:.*]] = {{.*}}, %[[ACC:.*]] =
// AFTER: %[[PROD:.*]]:2 = scope.scope
// AFTER: %[[FACTOR:.*]] = hivm.hir.vexp
// AFTER: hivm.hir.vreduce <sum>
// AFTER: %[[SUM:.*]] = tensor.collapse_shape
// AFTER-NOT: hivm.hir.vmul
// AFTER-NOT: hivm.hir.vadd
// AFTER: scope.return %[[FACTOR]], %[[SUM]]
// AFTER: } {{{.*}}hivm.preload_num = 2 : i32
// AFTER: scope.scope
// AFTER: } {{{.*}}hivm.loop_core_type = #hivm.tcore_type<CUBE>{{.*}}hivm.preload_num = 1 : i32
// AFTER: %[[LATE:.*]]:2 = scope.scope
// AFTER: %[[SCALE:.*]] = hivm.hir.vmul ins(%[[STATE]], %[[PROD]]#0
// AFTER: %[[NEXT:.*]] = hivm.hir.vadd ins(%[[SCALE]], %[[PROD]]#1
// AFTER: %[[ASCALE:.*]] = hivm.hir.vmul ins(%[[ACC]], %[[PROD]]#0
// AFTER: scope.return %[[ASCALE]], %[[NEXT]]
// AFTER: } {{{.*}}hivm.preload_num = 0 : i32
// AFTER: scf.yield %[[LATE]]#1, %[[LATE]]#0
// BEFORE-LABEL: func.func @closed_tensor_recurrence
// BEFORE: %[[PROD:.*]]:2 = scope.scope
// BEFORE: hivm.hir.vexp
// BEFORE: hivm.hir.vreduce <sum>
// BEFORE: hivm.hir.vmul
// BEFORE: hivm.hir.vadd
// BEFORE: scope.return
// BEFORE: } {{{.*}}hivm.preload_num = 2 : i32
// BEFORE: scope.scope
// BEFORE: } {{{.*}}hivm.loop_core_type = #hivm.tcore_type<CUBE>
// BEFORE: %[[LATE:.*]] = scope.scope
// BEFORE: hivm.hir.vmul
// BEFORE: scf.yield %[[PROD]]#1, %[[LATE]]
func.func @closed_tensor_recurrence(%x: !row,
                                  %tile: !tile)
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

// The recurrence is not closed when a payload reads its old state.
// COMMON-LABEL: func.func @keep_external_state_reader
// COMMON: %[[PROD:.*]]:2 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_external_state_reader(%x: !row)
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
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
      %keep = hivm.hir.vexp ins(%state : !row) outs(%empty : !row) -> !row
      %buf = memref.alloc() : !cbuf
      hivm.hir.copy ins(%keep : !row) outs(%buf : !cbuf)
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

// A yielded state also used by later current-iteration work stays.
// COMMON-LABEL: func.func @keep_current_iteration_use
// COMMON: %[[PROD:.*]]:2 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_current_iteration_use(%x: !row)
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
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
      scope.return %factor, %next : !row, !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
    %late = scope.scope : () -> !row {
      %scaled = hivm.hir.vmul ins(%acc, %prod#0 : !row, !row) outs(%empty : !row) -> !row
      %used = hivm.hir.vadd ins(%scaled, %prod#1 : !row, !row) outs(%empty : !row) -> !row
      scope.return %used : !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %prod#1, %late : !row, !row
  } {hivm.cv_pipelined_loop}
  return
}

// Reading a second iter_arg must not shift coupled recurrence phases.
// COMMON-LABEL: func.func @keep_coupled_recurrences
// COMMON: %[[PROD:.*]]:2 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_coupled_recurrences(%x: !row)
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
      %scale = hivm.hir.vmul ins(%state, %acc : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
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

// Do not move a recurrence across an explicit synchronization boundary.
// COMMON-LABEL: func.func @keep_sync_boundary
// COMMON: %[[PROD:.*]]:2 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_sync_boundary(%x: !row)
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
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_MTE1>] flag = 1
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

// Do not cross an unmodeled control region in the intervening stage.
// COMMON-LABEL: func.func @keep_control_boundary
// COMMON: %[[PROD:.*]]:2 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_control_boundary(%x: !row)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %true = arith.constant true
  %empty = tensor.empty() : !row
  %initial = tensor.empty() : !row
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!row, !row) {
    %prod:2 = scope.scope : () -> (!row, !row) {
      %factor = hivm.hir.vexp ins(%x : !row) outs(%empty : !row) -> !row
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
      scope.return %factor, %next : !row, !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scf.if %true {
      }
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

// Both recurrence phi edges are checked against crossed memory writes.
// COMMON-LABEL: func.func @keep_aliased_state_write
// COMMON: %[[PROD:.*]]:2 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_aliased_state_write(%x: !row, %buf: !ub, %cond: i1)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : !row
  %alias = bufferization.to_tensor %buf restrict writable : !ub
  %view = tensor.cast %alias : !row to !row
  %selected = scf.if %cond -> !row {
    scf.yield %empty : !row
  } else {
    scf.yield %view : !row
  }
  %initial = scf.for %j = %c0 to %c8 step %c1
      iter_args(%seed = %empty) -> !row {
    scf.yield %selected : !row
  }
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!row, !row) {
    %prod:2 = scope.scope : () -> (!row, !row) {
      %factor = hivm.hir.vexp ins(%x : !row) outs(%empty : !row) -> !row
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
      scope.return %factor, %next : !row, !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      hivm.hir.copy ins(%x : !row) outs(%buf : !ub)
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

// Distinct function memrefs may alias; unknown alias is conservative.
// COMMON-LABEL: func.func @keep_unknown_alias_write
// COMMON: %[[PROD:.*]]:2 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_unknown_alias_write(%x: !row, %buf: !ub, %other: !ub)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : !row
  %initial = bufferization.to_tensor %buf restrict writable : !ub
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!row, !row) {
    %prod:2 = scope.scope : () -> (!row, !row) {
      %factor = hivm.hir.vexp ins(%x : !row) outs(%empty : !row) -> !row
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
      scope.return %factor, %next : !row, !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      hivm.hir.copy ins(%x : !row) outs(%other : !ub)
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

// A destination needs an existing shared input, not just empty capacity.
// COMMON-LABEL: func.func @keep_without_shared_consumer
// COMMON: %[[PROD:.*]]:2 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_without_shared_consumer(%x: !row)
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
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
      scope.return %factor, %next : !row, !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
    %late = scope.scope : () -> !row {
      %scaled = hivm.hir.vmul ins(%acc, %x : !row, !row) outs(%empty : !row) -> !row
      scope.return %scaled : !row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %prod#1, %late : !row, !row
  } {hivm.cv_pipelined_loop}
  return
}

// Writes to a disjoint buffer may be crossed, including with scalar inputs.
// AFTER-LABEL: func.func @closed_over_unrelated_write
// AFTER: %[[P:.*]]:2 = scope.scope
// AFTER: hivm.hir.vexp ins(%{{.*}} : memref
// AFTER-NOT: hivm.hir.vmul
// AFTER-NOT: hivm.hir.vadd
// AFTER: scope.return
// AFTER: } {{{.*}}hivm.preload_num = 2 : i32
// AFTER: scope.scope
// AFTER: } {{{.*}}CUBE
// AFTER: %[[L:.*]]:2 = scope.scope
// AFTER: hivm.hir.vmul
// AFTER: hivm.hir.vadd
// AFTER: hivm.hir.vadd
// AFTER: scf.yield %[[L]]#1, %[[L]]#0
// BEFORE-LABEL: func.func @closed_over_unrelated_write
// BEFORE: hivm.hir.vmul
// BEFORE: hivm.hir.vadd
// BEFORE: hivm.hir.vexp ins(%{{.*}} : memref
// BEFORE: } {{{.*}}hivm.preload_num = 2 : i32
func.func @closed_over_unrelated_write(%x: !row,
                                  %tile: !tile, %bias: f32)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : !row
  %initial = tensor.empty() : !row
  %buf = memref.alloc() : !ub
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!row, !row) {
    %prod:2 = scope.scope : () -> (!row, !row) {
      %factor = hivm.hir.vexp ins(%x : !row) outs(%empty : !row) -> !row
      %suminit = tensor.empty() : tensor<16x1xf32>
      %sum = hivm.hir.vreduce <sum> ins(%tile : !tile) outs(%suminit : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
      %flat = tensor.collapse_shape %sum [[0, 1]] : tensor<16x1xf32> into !row
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %biased = hivm.hir.vadd ins(%scale, %bias : !row, f32) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%biased, %flat : !row, !row) outs(%empty : !row) -> !row
      hivm.hir.vexp ins(%buf : !ub) outs(%buf : !ub)
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

// Different dimensions and type use the same recurrence rule.
// AFTER-LABEL: func.func @closed_half_recurrence
// AFTER: %[[P:.*]]:2 = scope.scope
// AFTER: tensor.collapse_shape
// AFTER-NOT: hivm.hir.vmul
// AFTER-NOT: hivm.hir.vadd
// AFTER: scope.return
// AFTER: } {{{.*}}hivm.preload_num = 2 : i32
// AFTER: scope.scope
// AFTER: } {{{.*}}CUBE
// AFTER: %[[L:.*]]:2 = scope.scope
// AFTER: hivm.hir.vmul
// AFTER: hivm.hir.vadd
// AFTER: scope.return
// AFTER: scf.yield %[[L]]#1, %[[L]]#0
// BEFORE-LABEL: func.func @closed_half_recurrence
// BEFORE: hivm.hir.vmul
// BEFORE: hivm.hir.vadd
// BEFORE: } {{{.*}}hivm.preload_num = 2 : i32
func.func @closed_half_recurrence(%x: !half_row,
                                  %tile: !half_tile)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : !half_row
  %initial = tensor.empty() : !half_row
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!half_row, !half_row) {
    %prod:2 = scope.scope : () -> (!half_row, !half_row) {
      %factor = hivm.hir.vexp ins(%x : !half_row) outs(%empty : !half_row) -> !half_row
      %suminit = tensor.empty() : tensor<7x1xf16>
      %sum = hivm.hir.vreduce <sum> ins(%tile : !half_tile) outs(%suminit : tensor<7x1xf16>) reduce_dims = [1] -> tensor<7x1xf16>
      %flat = tensor.collapse_shape %sum [[0, 1]] : tensor<7x1xf16> into !half_row
      %scale = hivm.hir.vmul ins(%state, %factor : !half_row, !half_row) outs(%empty : !half_row) -> !half_row
      %next = hivm.hir.vadd ins(%scale, %flat : !half_row, !half_row) outs(%empty : !half_row) -> !half_row
      scope.return %factor, %next : !half_row, !half_row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
    %late = scope.scope : () -> !half_row {
      %scaled = hivm.hir.vmul ins(%acc, %prod#0 : !half_row, !half_row) outs(%empty : !half_row) -> !half_row
      scope.return %scaled : !half_row
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %prod#1, %late : !half_row, !half_row
  } {hivm.cv_pipelined_loop}
  return
}

// A loop-local view of another iter_arg is a coupled carry too.
// COMMON-LABEL: func.func @keep_other_carry_view
// COMMON: tensor.cast
// COMMON: scope.scope
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_other_carry_view(%x: !row)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : !row
  %initial = tensor.empty() : !row
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!row, !row) {
    %other = tensor.cast %acc : !row to !row
    %prod:2 = scope.scope : () -> (!row, !row) {
      %factor = hivm.hir.vexp ins(%x : !row) outs(%empty : !row) -> !row
      %scale = hivm.hir.vmul ins(%state, %other : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
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

// Returning the same inner value twice is not an exclusive result slice.
// COMMON-LABEL: func.func @keep_duplicate_return
// COMMON: %[[P:.*]]:3 = scope.scope
// COMMON: hivm.hir.vmul
// COMMON: %[[STATE:.*]] = hivm.hir.vadd
// COMMON: scope.return %{{.*}}, %[[STATE]], %[[STATE]]
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
// COMMON: scf.yield %[[P]]#1
func.func @keep_duplicate_return(%x: !row)
    attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %empty = tensor.empty() : !row
  %initial = tensor.empty() : !row
  %out:2 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%state = %initial, %acc = %initial)
      -> (!row, !row) {
    %prod:3 = scope.scope : () -> (!row, !row, !row) {
      %factor = hivm.hir.vexp ins(%x : !row) outs(%empty : !row) -> !row
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
      scope.return %factor, %next, %next : !row, !row, !row
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

// Forward new captures in producer order, independently of DenseSet order.
// AFTER-LABEL: func.func @closed_two_forwarded_inputs
// AFTER: %[[P:.*]]:3 = scope.scope
// AFTER: %[[F:.*]] = hivm.hir.vexp
// AFTER: %[[A:.*]] = tensor.collapse_shape
// AFTER: %[[B:.*]] = tensor.collapse_shape
// AFTER-NOT: hivm.hir.vmul
// AFTER: scope.return %[[F]], %[[A]], %[[B]]
// AFTER: } {{{.*}}hivm.preload_num = 2 : i32
// AFTER: scope.scope
// AFTER: } {{{.*}}CUBE
// AFTER: %[[L:.*]]:2 = scope.scope
// AFTER: hivm.hir.vmul
// AFTER: hivm.hir.vadd ins(%{{.*}}, %[[P]]#1
// AFTER: hivm.hir.vsub ins(%{{.*}}, %[[P]]#2
// AFTER: scf.yield %[[L]]#1, %[[L]]#0
// BEFORE-LABEL: func.func @closed_two_forwarded_inputs
// BEFORE: hivm.hir.vmul
// BEFORE: hivm.hir.vadd
// BEFORE: hivm.hir.vsub
// BEFORE: } {{{.*}}hivm.preload_num = 2 : i32
func.func @closed_two_forwarded_inputs(%x: !row,
                                  %tile: !tile)
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
      %otherinit = tensor.empty() : tensor<16x1xf32>
      %other = hivm.hir.vreduce <max> ins(%tile : !tile) outs(%otherinit : tensor<16x1xf32>) reduce_dims = [1] -> tensor<16x1xf32>
      %otherflat = tensor.collapse_shape %other [[0, 1]] : tensor<16x1xf32> into !row
      %scale = hivm.hir.vmul ins(%state, %factor : !row, !row) outs(%empty : !row) -> !row
      %partial = hivm.hir.vadd ins(%scale, %flat : !row, !row) outs(%empty : !row) -> !row
      %next = hivm.hir.vsub ins(%partial, %otherflat : !row, !row) outs(%empty : !row) -> !row
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

// Producer-local scalar captures are not returned/rematerialized by this rule.
// COMMON-LABEL: func.func @keep_local_scalar_capture
// COMMON: scope.scope
// COMMON: arith.addf
// COMMON: hivm.hir.vmul
// COMMON: hivm.hir.vadd
// COMMON: scope.return
// COMMON: } {{{.*}}hivm.preload_num = 2 : i32
func.func @keep_local_scalar_capture(%x: !row, %scaleval: f32)
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
      %scalar = arith.addf %scaleval, %scaleval : f32
      %scale = hivm.hir.vmul ins(%state, %scalar : !row, f32) outs(%empty : !row) -> !row
      %next = hivm.hir.vadd ins(%scale, %factor : !row, !row) outs(%empty : !row) -> !row
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
