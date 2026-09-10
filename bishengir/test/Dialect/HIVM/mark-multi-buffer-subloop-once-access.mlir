// Boundary buffers are skipped when their autoblockify.subloop contains a
// nested compute loop. Streaming buffers inside the nested loop keep their
// mark. A blockify subloop without a nested scf.for also keeps its marks. The
// gate is opt-out via skip-multi-buffer-for-once-per-subloop-access=false.

// RUN: bishengir-opt -allow-unregistered-dialect %s \
// RUN:   -pass-pipeline="builtin.module(                        \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true}))" \
// RUN:   -split-input-file | FileCheck %s
// RUN: bishengir-opt -allow-unregistered-dialect %s \
// RUN:   -pass-pipeline="builtin.module(                        \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true skip-multi-buffer-for-once-per-subloop-access=false}))" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=KEEP

// -----
// CHECK-LABEL: func.func @subloop_with_nested_loop_skips_boundary_marks
func.func @subloop_with_nested_loop_skips_boundary_marks(
    %gm_in : memref<8xf32, #hivm.address_space<gm>>,
    %gm_out : memref<8xf32, #hivm.address_space<gm>>,
    %lb : index, %ub : index) {
  %c1 = arith.constant 1 : index
  scf.for %blk = %lb to %ub step %c1 {
    // Boundary load buffer: GM load directly in the subloop body.
    %init_row = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[INIT_ROW:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK-NOT: annotation.mark %[[INIT_ROW]]
    // KEEP: %[[INIT_ROW:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // KEEP: annotation.mark %[[INIT_ROW]] {hivm.multi_buffer = 2 : i32}
    hivm.hir.load ins(%gm_in : memref<8xf32, #hivm.address_space<gm>>) outs(%init_row : memref<8xf32, #hivm.address_space<ub>>)

    // Boundary accumulator buffer: GM store through the nested loop result,
    // directly in the subloop body.
    %acc = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[ACC:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK-NOT: annotation.mark %[[ACC]]
    // KEEP: %[[ACC:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // KEEP: annotation.mark %[[ACC]] {hivm.multi_buffer = 2 : i32}
    %r = scf.for %i = %lb to %ub step %c1 iter_args(%a = %acc) -> (memref<8xf32, #hivm.address_space<ub>>) {
      // Streaming buffer: GM access inside the nested loop keeps its mark.
      %row = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
      // CHECK: %[[ROW:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
      // CHECK: annotation.mark %[[ROW]] {hivm.multi_buffer = 2 : i32}
      // KEEP: %[[ROW:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
      // KEEP: annotation.mark %[[ROW]] {hivm.multi_buffer = 2 : i32}
      hivm.hir.load ins(%gm_in : memref<8xf32, #hivm.address_space<gm>>) outs(%row : memref<8xf32, #hivm.address_space<ub>>)
      "some_use"(%a) : (memref<8xf32, #hivm.address_space<ub>>) -> ()
      scf.yield %a : memref<8xf32, #hivm.address_space<ub>>
    }
    %sv = memref.subview %r[0] [8] [1] : memref<8xf32, #hivm.address_space<ub>> to memref<8xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.store ins(%sv : memref<8xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%gm_out : memref<8xf32, #hivm.address_space<gm>>)
  } {autoblockify.subloop}
  return
}

// -----
// Multiple blockify subloops without nested loops are handled independently.
// CHECK-LABEL: func.func @multiple_subloops_without_nested_for_keep_marks
func.func @multiple_subloops_without_nested_for_keep_marks(
    %gm_in : memref<8xf32, #hivm.address_space<gm>>, %lb : index, %ub : index) {
  %c1 = arith.constant 1 : index
  scf.for %blk0 = %lb to %ub step %c1 {
    %tmp0 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[TMP0:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[TMP0]] {hivm.multi_buffer = 2 : i32}
    // KEEP: %[[TMP0:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // KEEP: annotation.mark %[[TMP0]] {hivm.multi_buffer = 2 : i32}
    hivm.hir.load ins(%gm_in : memref<8xf32, #hivm.address_space<gm>>) outs(%tmp0 : memref<8xf32, #hivm.address_space<ub>>)
  } {autoblockify.subloop}
  scf.for %blk1 = %lb to %ub step %c1 {
    %tmp1 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[TMP1:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[TMP1]] {hivm.multi_buffer = 2 : i32}
    // KEEP: %[[TMP1:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // KEEP: annotation.mark %[[TMP1]] {hivm.multi_buffer = 2 : i32}
    hivm.hir.load ins(%gm_in : memref<8xf32, #hivm.address_space<gm>>) outs(%tmp1 : memref<8xf32, #hivm.address_space<ub>>)
  } {autoblockify.subloop}
  return
}

// -----
// A sole autoblockify.subloop without a nested scf.for keeps its mark.
// CHECK-LABEL: func.func @sole_subloop_without_nested_for_keeps_mark
func.func @sole_subloop_without_nested_for_keeps_mark(
    %gm_in : memref<8xf32, #hivm.address_space<gm>>, %lb : index, %ub : index) {
  %c1 = arith.constant 1 : index
  scf.for %blk = %lb to %ub step %c1 {
    %tmp = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[TMP:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[TMP]] {hivm.multi_buffer = 2 : i32}
    // KEEP: %[[TMP:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // KEEP: annotation.mark %[[TMP]] {hivm.multi_buffer = 2 : i32}
    func.call @fill_ub(%tmp) : (memref<8xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.load ins(%gm_in : memref<8xf32, #hivm.address_space<gm>>) outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
  } {autoblockify.subloop}
  return
}

func.func private @fill_ub(%buf : memref<8xf32, #hivm.address_space<ub>>) {
  "some_use"(%buf) : (memref<8xf32, #hivm.address_space<ub>>) -> ()
  return
}

// -----
// An opaque external call makes the use graph unmodelable: keep the mark.
// CHECK-LABEL: func.func @subloop_buffer_via_external_call
func.func @subloop_buffer_via_external_call(
    %gm_in : memref<8xf32, #hivm.address_space<gm>>, %lb : index, %ub : index) {
  %c1 = arith.constant 1 : index
  scf.for %blk = %lb to %ub step %c1 {
    %tmp = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[TMP:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[TMP]] {hivm.multi_buffer = 2 : i32}
    // KEEP: %[[TMP:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // KEEP: annotation.mark %[[TMP]] {hivm.multi_buffer = 2 : i32}
    func.call @opaque(%tmp) : (memref<8xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.load ins(%gm_in : memref<8xf32, #hivm.address_space<gm>>) outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
  } {autoblockify.subloop}
  return
}

func.func private @opaque(%buf : memref<8xf32, #hivm.address_space<ub>>)

// -----
// Without an autoblockify.subloop ancestor the gate is inactive.
// CHECK-LABEL: func.func @plain_loop_unaffected
func.func @plain_loop_unaffected(
    %gm_in : memref<8xf32, #hivm.address_space<gm>>, %lb : index, %ub : index) {
  %c1 = arith.constant 1 : index
  scf.for %i = %lb to %ub step %c1 {
    %tmp = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[TMP:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %[[TMP]] {hivm.multi_buffer = 2 : i32}
    // KEEP: %[[TMP:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // KEEP: annotation.mark %[[TMP]] {hivm.multi_buffer = 2 : i32}
    hivm.hir.load ins(%gm_in : memref<8xf32, #hivm.address_space<gm>>) outs(%tmp : memref<8xf32, #hivm.address_space<ub>>)
  }
  return
}
