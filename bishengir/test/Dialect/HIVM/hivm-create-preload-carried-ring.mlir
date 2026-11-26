// RUN: bishengir-opt %s -create-preload -verify-each | FileCheck %s

// Identical address lists are not identical state: the carried pointer selects
// the previous iteration's slot, while the producer writes the current slot.
// The guarded result also preserves the initial/final slot for empty/drain ticks.
// CHECK-LABEL: func.func @carried_ring
// CHECK: %[[INIT:.*]] = hivm.hir.pointer_cast
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[PREVIOUS:.*]] = %[[INIT]])
// CHECK: %[[CURRENT:.*]] = hivm.hir.pointer_cast
// CHECK: %[[DELAYED:.*]] = hivm.hir.pointer_cast
// CHECK: %[[NEXT:.*]] = scf.if {{.*}} -> (memref<1xf32, #hivm.address_space<ub>>) {
// CHECK: %[[OLD:.*]] = memref.load %[[PREVIOUS]]
// CHECK: %[[NEW:.*]] = arith.addf %[[OLD]],
// CHECK: memref.store %[[NEW]], %[[CURRENT]]
// CHECK: scf.yield %[[CURRENT]]
// CHECK: } else {
// CHECK: scf.yield %[[PREVIOUS]]
// CHECK: scf.if
// CHECK: memref.load %[[DELAYED]]
// CHECK: scf.yield %[[NEXT]]
// CHECK: %[[FINAL:.*]] = memref.load %[[LOOP]]
// CHECK: return %[[FINAL]]
func.func @carried_ring(%ub: index, %seen: memref<?xf32, #hivm.address_space<gm>>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a0 = arith.constant 0 : i64
  %a1 = arith.constant 64 : i64
  %one = arith.constant 1.0 : f32
  %init = hivm.hir.pointer_cast(%a0, %a1) : memref<1xf32, #hivm.address_space<ub>>
  %result = scf.for %i = %c0 to %ub step %c1
      iter_args(%previous = %init) -> memref<1xf32, #hivm.address_space<ub>> {
    %current = hivm.hir.pointer_cast(%a0, %a1) : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %current {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer}
        : memref<1xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %old = memref.load %previous[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %new = arith.addf %old, %one : f32
      memref.store %new, %current[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %value = memref.load %current[%c0] : memref<1xf32, #hivm.address_space<ub>>
      memref.store %value, %seen[%i] : memref<?xf32, #hivm.address_space<gm>>
      scope.return
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %current : memref<1xf32, #hivm.address_space<ub>>
  }
  %final = memref.load %result[%c0] : memref<1xf32, #hivm.address_space<ub>>
  return %final : f32
}

// A returned view must carry the guarded state, while other stage consumers
// keep views of their own ring slot.
// CHECK-LABEL: func.func @carried_ring_view
// CHECK: %[[VLOOP:.*]] = scf.for {{.*}} iter_args(%[[VPREV:.*]] = {{.*}})
// CHECK: %[[VNEXT:.*]] = scf.if {{.*}} -> (memref<1xf32, #hivm.address_space<ub>>) {
// CHECK: memref.load %[[VPREV]]
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield %[[VPREV]]
// CHECK: scf.yield %[[VNEXT]]
// CHECK: memref.load %[[VLOOP]]
func.func @carried_ring_view(%ub: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a0 = arith.constant 0 : i64
  %a1 = arith.constant 64 : i64
  %initial = hivm.hir.pointer_cast(%a0, %a1) : memref<2xf32, #hivm.address_space<ub>>
  %init = memref.subview %initial[0] [1] [1]
      : memref<2xf32, #hivm.address_space<ub>> to memref<1xf32, #hivm.address_space<ub>>
  %result = scf.for %i = %c0 to %ub step %c1
      iter_args(%previous = %init) -> memref<1xf32, #hivm.address_space<ub>> {
    %current = hivm.hir.pointer_cast(%a0, %a1) : memref<2xf32, #hivm.address_space<ub>>
    annotation.mark %current {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer}
        : memref<2xf32, #hivm.address_space<ub>>
    %next = scope.scope : () -> memref<1xf32, #hivm.address_space<ub>> {
      %view = memref.subview %current[0] [1] [1]
          : memref<2xf32, #hivm.address_space<ub>> to memref<1xf32, #hivm.address_space<ub>>
      %old = memref.load %previous[%c0] : memref<1xf32, #hivm.address_space<ub>>
      memref.store %old, %view[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return %view : memref<1xf32, #hivm.address_space<ub>>
    } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
    scf.yield %next : memref<1xf32, #hivm.address_space<ub>>
  }
  %final = memref.load %result[%c0] : memref<1xf32, #hivm.address_space<ub>>
  return %final : f32
}

func.func @outlined_vf_read(%dst: memref<8x16xf32, #hivm.address_space<ub>>,
                            %src: memref<8x16xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  scf.for %r = %c0 to %c8 step %c1 {
    %s = memref.subview %src[%r, 0] [1, 16] [1, 1]
        : memref<8x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %d = memref.subview %dst[%r, 0] [1, 16] [1, 1]
        : memref<8x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %v = vector.transfer_read %s[%c0, %c0], %cst {in_bounds = [true, true]}
        : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x16xf32>
    vector.transfer_write %v, %d[%c0, %c0] {in_bounds = [true, true]}
        : vector<1x16xf32>, memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

func.func @outlined_vf_write(%in: memref<8x16xf32, #hivm.address_space<ub>>,
                             %out: memref<8x16xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  scf.for %r = %c0 to %c8 step %c1 {
    %s = memref.subview %in[%r, 0] [1, 16] [1, 1]
        : memref<8x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %d = memref.subview %out[%r, 0] [1, 16] [1, 1]
        : memref<8x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %v = vector.transfer_read %s[%c0, %c0], %cst {in_bounds = [true, true]}
        : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x16xf32>
    vector.transfer_write %v, %d[%c0, %c0] {in_bounds = [true, true]}
        : vector<1x16xf32>, memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @carried_ring_vf_writer
// CHECK: %[[INIT_VF:.*]] = hivm.hir.pointer_cast
// CHECK: %[[LOOP_VF:.*]] = scf.for {{.*}} iter_args(%[[PREV_VF:.*]] = %[[INIT_VF]])
// CHECK: %[[CUR_VF:.*]] = hivm.hir.pointer_cast(%c0_i64, %c2624_i64)
// CHECK: %[[DEL_VF:.*]] = hivm.hir.pointer_cast(%c2624_i64, %c0_i64)
// CHECK: %[[NEXT_VF:.*]] = scf.if {{.*}} -> (memref<8x16xf32, #hivm.address_space<ub>>) {
// CHECK: func.call @outlined_vf_read({{.*}}, %[[PREV_VF]])
// CHECK: func.call @outlined_vf_write({{.*}}, %[[CUR_VF]])
// CHECK: scf.yield %[[CUR_VF]]
// CHECK: } else {
// CHECK: scf.yield %[[PREV_VF]]
// CHECK: scf.if
// CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %[[DEL_VF]]
// CHECK: hivm.hir.store ins(%[[COLLAPSED]]
// CHECK: scf.yield %[[NEXT_VF]]
func.func @carried_ring_vf_writer(%ub: i32,
                                  %gm_out: memref<128xf32, #hivm.address_space<gm>>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i64 = arith.constant 0 : i64
  %c512_i64 = arith.constant 512 : i64
  %c2624_i64 = arith.constant 2624 : i64
  %init = hivm.hir.pointer_cast(%c0_i64, %c2624_i64) : memref<8x16xf32, #hivm.address_space<ub>>
  %result = scf.for %i = %c0_i32 to %ub step %c1_i32
      iter_args(%previous = %init) -> (memref<8x16xf32, #hivm.address_space<ub>>) : i32 {
    %current = hivm.hir.pointer_cast(%c0_i64, %c2624_i64) : memref<8x16xf32, #hivm.address_space<ub>>
    annotation.mark %current {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer}
        : memref<8x16xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      %tmp = hivm.hir.pointer_cast(%c512_i64) : memref<8x16xf32, #hivm.address_space<ub>>
      func.call @outlined_vf_read(%tmp, %previous) {hivm.vector_function, no_inline}
          : (memref<8x16xf32, #hivm.address_space<ub>>, memref<8x16xf32, #hivm.address_space<ub>>) -> ()
      func.call @outlined_vf_write(%tmp, %current) {hivm.vector_function, no_inline}
          : (memref<8x16xf32, #hivm.address_space<ub>>, memref<8x16xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {cv_merged_scope, hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %collapsed = memref.collapse_shape %current [[0, 1]]
          : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
      hivm.hir.store ins(%collapsed : memref<128xf32, #hivm.address_space<ub>>)
          outs(%gm_out : memref<128xf32, #hivm.address_space<gm>>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32, no_inline}
    scf.yield %current : memref<8x16xf32, #hivm.address_space<ub>>
  }
  %final_collapsed = memref.collapse_shape %result [[0, 1]]
      : memref<8x16xf32, #hivm.address_space<ub>> into memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.store ins(%final_collapsed : memref<128xf32, #hivm.address_space<ub>>)
      outs(%gm_out : memref<128xf32, #hivm.address_space<gm>>)
  return
}
