// RUN: bishengir-opt %s -create-preload -split-input-file -allow-unregistered-dialect | FileCheck %s

// Test 2 scopes with preload_num=1 and preload_num=0, each yielded as iter_arg.

// CHECK-LABEL: func.func @test_two_scopes
func.func @test_two_scopes() -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  %c0_init = arith.constant 0 : i32

  // CHECK: scf.for {{.*}} = %c0{{.*}} to %c18{{.*}} step {{.*}}
  %0:2 = scf.for %i = %c0 to %c16 step %c1 iter_args(%arg0 = %c0_init, %arg1 = %c0_init) -> (i32, i32) : i32 {

    // Stage 0 (preload_num=1): scf.if with cmp guards
    // CHECK: arith.cmpi sge
    // CHECK-NEXT: arith.cmpi slt
    // CHECK-NEXT: arith.andi
    // CHECK-NEXT: scf.if
    %s0 = scope.scope : () -> i32 {
      %r0 = arith.addi %arg0, %i : i32
      scope.return %r0 : i32
    } {no_inline, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}

    // Stage 1 (preload_num=0): scf.if with cmp guards (uses shifted iv)
    // CHECK: arith.cmpi sge
    // CHECK-NEXT: arith.cmpi slt
    // CHECK-NEXT: arith.andi
    // CHECK-NEXT: scf.if
    %s1 = scope.scope : () -> i32 {
      %r1 = arith.addi %arg1, %i : i32
      scope.return %r1 : i32
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}

    scf.yield %s0, %s1 : i32, i32
  }
  // CHECK: return
  return %0#0, %0#1 : i32, i32
}

// -----

// Test that CreatePreload can recognize a preload workspace subview whose
// source is marked by annotation.mark, while the subview itself does not carry
// hivm.preload_workspace. 

module {
  // CHECK-LABEL: func.func @test_preload_workspace_subview_trace_source_mark
  // CHECK-SAME: (%[[WS:arg[0-9]+]]: memref<?xi8, #hivm.address_space<gm>>
  // CHECK-SAME:  %[[DST:arg[0-9]+]]: memref<128x128xf16, #hivm.address_space<gm>>)
  func.func @test_preload_workspace_subview_trace_source_mark(
      %ws: memref<?xi8, #hivm.address_space<gm>>,
      %dst: memref<128x128xf16, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %v1 = arith.constant 1.000000e+00 : f16
    %v2 = arith.constant 2.000000e+00 : f16

    %lb = arith.constant 0 : i32
    %ub = arith.constant 4 : i32
    %step = arith.constant 1 : i32

    %view = memref.view %ws[%c0][]
      : memref<?xi8, #hivm.address_space<gm>>
        to memref<4x128x128xf16, #hivm.address_space<gm>>

    annotation.mark %view {hivm.preload_workspace}
      : memref<4x128x128xf16, #hivm.address_space<gm>>

    scf.for %i = %lb to %ub step %step : i32 {
      scope.scope : () -> () {
        memref.store %v1, %dst[%c0, %c0]
          : memref<128x128xf16, #hivm.address_space<gm>>
        scope.return
      } {
        no_inline,
        hivm.preload_num = 1 : i32,
        hivm.max_preload_num = 2 : i32
      }

      scope.scope : () -> () {
        %subview = memref.subview %view[0, 0, 0] [1, 128, 128] [1, 1, 1]
          : memref<4x128x128xf16, #hivm.address_space<gm>>
            to memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>

        memref.store %v2, %subview[%c0, %c0]
          : memref<128x128xf16, strided<[128, 1]>, #hivm.address_space<gm>>

        scope.return
      } {
        no_inline,
        hivm.preload_num = 0 : i32,
        hivm.max_preload_num = 2 : i32
      }
    }

    return
  }

  // CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C2_I32:.*]] = arith.constant 2 : i32
  // CHECK-DAG: %[[C6_I32:.*]] = arith.constant 6 : i32
  // CHECK-DAG: %[[V2:.*]] = arith.constant 2.000000e+00 : f16

  // Marked workspace root.
  // CHECK: %[[VIEW:.*]] = memref.view %[[WS]][%[[C0_IDX]]][]
  // CHECK-SAME: to memref<4x128x128xf16, #hivm.address_space<gm>>

  // CHECK: scf.for %[[IV:arg[0-9]+]] = {{.*}} to %[[C6_I32]] step {{.*}} : i32 {
  // CHECK:   %[[MAPPED:.*]] = arith.subi %[[IV]], {{.*}} : i32

  // Dynamic preload slot.
  // CHECK:   %[[SLOT_I32:.*]] = arith.remsi {{.*}}, %[[C2_I32]] : i32
  // CHECK:   %[[SLOT:.*]] = arith.index_cast %[[SLOT_I32]] : i32 to index

  // The original subview has no hivm.preload_workspace attr; this verifies
  // that CreatePreload traces the source mark and rewrites offset[0].
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[VIEW]][%[[SLOT]], 0, 0] [1, 128, 128] [1, 1, 1]
  // CHECK-SAME: memref<4x128x128xf16, #hivm.address_space<gm>>
  // CHECK-SAME: memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  // CHECK:   memref.store %[[V2]], %[[SUBVIEW]][%[[C0_IDX]], %[[C0_IDX]]]
  // CHECK: }
  // CHECK: return
}

// -----

module {
  // The preload guard must use the original loop lower bound, not a fixed zero.
  //
  // This test uses a dynamic lower bound %lb. For preload_num = 0 and
  // max_preload_num = 2, CreatePreload maps the original IV to:
  //
  //   mapped_iv = new_iv - step
  //
  // The generated guard must therefore be:
  //
  //   mapped_iv >= %lb && mapped_iv < %ub
  //
  // not:
  //
  //   mapped_iv >= 0 && mapped_iv < %ub

  // CHECK-LABEL: func.func @test_preload_condition_uses_original_lower_bound
  // CHECK-SAME: (%[[LB:arg[0-9]+]]: i32, %[[UB:arg[0-9]+]]: i32, %[[INIT0:arg[0-9]+]]: i32, %[[INIT1:arg[0-9]+]]: i32)
  func.func @test_preload_condition_uses_original_lower_bound(
      %lb: i32, %ub: i32, %init0: i32, %init1: i32) -> (i32, i32) {
    %c1 = arith.constant 1 : i32

    %0:2 = scf.for %i = %lb to %ub step %c1
        iter_args(%arg0 = %init0, %arg1 = %init1) -> (i32, i32) : i32 {

      // preload_num = 1:
      //   old_iv -> new_iv
      %s0 = scope.scope : () -> i32 {
        %r0 = arith.addi %arg0, %i : i32
        scope.return %r0 : i32
      } {
        no_inline,
        hivm.preload_num = 1 : i32,
        hivm.max_preload_num = 2 : i32
      }

      // preload_num = 0:
      //   old_iv -> new_iv - step
      //
      // This scope exposes the old bug where the lower guard was fixed to 0.
      %s1 = scope.scope : () -> i32 {
        %r1 = arith.addi %arg1, %i : i32
        scope.return %r1 : i32
      } {
        no_inline,
        hivm.preload_num = 0 : i32,
        hivm.max_preload_num = 2 : i32
      }

      scf.yield %s0, %s1 : i32, i32
    }

    return %0#0, %0#1 : i32, i32
  }

  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32

  // CHECK: %[[NEW_UB:.*]] = arith.addi %[[UB]], %[[C2]] : i32
  // CHECK: %[[RESULTS:.*]]:2 = scf.for %[[NEW_IV:.*]] = %[[LB]] to %[[NEW_UB]] step %[[C1]] iter_args(%[[ARG0:.*]] = %[[INIT0]], %[[ARG1:.*]] = %[[INIT1]]) -> (i32, i32) : i32 {

  // CHECK: %[[MAPPED_IV:.*]] = arith.subi %[[NEW_IV]], %[[C1]] : i32

  // preload_num = 1: old_iv -> new_iv
  // CHECK: %[[LOWER_DIRECT:.*]] = arith.cmpi sge, %[[NEW_IV]], %[[LB]] : i32
  // CHECK: %[[UPPER_DIRECT:.*]] = arith.cmpi slt, %[[NEW_IV]], %[[UB]] : i32
  // CHECK: %[[COND_DIRECT:.*]] = arith.andi %[[LOWER_DIRECT]], %[[UPPER_DIRECT]] : i1
  // CHECK: %[[IF0:.*]] = scf.if %[[COND_DIRECT]] -> (i32) {
  // CHECK:   %[[R0:.*]] = arith.addi %[[ARG0]], %[[NEW_IV]] : i32
  // CHECK:   scf.yield %[[R0]] : i32
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG0]] : i32
  // CHECK: }

  // preload_num = 0: old_iv -> new_iv - step
  // CHECK: %[[LOWER_SHIFTED:.*]] = arith.cmpi sge, %[[MAPPED_IV]], %[[LB]] : i32
  // CHECK: %[[UPPER_SHIFTED:.*]] = arith.cmpi slt, %[[MAPPED_IV]], %[[UB]] : i32
  // CHECK: %[[COND_SHIFTED:.*]] = arith.andi %[[LOWER_SHIFTED]], %[[UPPER_SHIFTED]] : i1
  // CHECK: %[[IF1:.*]] = scf.if %[[COND_SHIFTED]] -> (i32) {
  // CHECK:   %[[R1:.*]] = arith.addi %[[ARG1]], %[[MAPPED_IV]] : i32
  // CHECK:   scf.yield %[[R1]] : i32
  // CHECK: } else {
  // CHECK:   scf.yield %[[ARG1]] : i32
  // CHECK: }

  // CHECK: scf.yield %[[IF0]], %[[IF1]] : i32, i32
  // CHECK: }
  // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1 : i32, i32
}

// -----

// A scope result that is an arbitrary-depth chain of views rooted at a preload
// local buffer. The preload skip (else) branch must rematerialize each view
// instead of aborting with "Unhandled scope result case".

// CHECK-LABEL: func.func @test_nested_subview_local_buffer_result
func.func @test_nested_subview_local_buffer_result() {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  %a0 = arith.constant 61696 : i64
  %a1 = arith.constant 65792 : i64
  %a2 = arith.constant 69888 : i64
  // CHECK: scf.for
  scf.for %i = %c0 to %c16 step %c1  : i32 {
    // CHECK: %[[BUF:.*]] = hivm.hir.pointer_cast
    %buf = hivm.hir.pointer_cast(%a0, %a1, %a2) : memref<64x256x1xi1, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.preload_local_buffer = 1 : i32, hivm.multi_buffer = 3 : i32} : memref<64x256x1xi1, #hivm.address_space<ub>>
    // CHECK: scf.if
    // CHECK: } else {
    // Each preload mapping has its own rotated pointer cast.
    // CHECK: %[[RE0:.*]] = memref.subview {{.*}}
    // CHECK: %[[RE1:.*]] = memref.subview %[[RE0]]
    // CHECK: scf.yield %[[RE1]]
    %s0 = scope.scope : () -> memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>> {
      %sv0 = memref.subview %buf[0, 0, 0] [64, 128, 1] [1, 1, 1] : memref<64x256x1xi1, #hivm.address_space<ub>> to memref<64x128x1xi1, strided<[256, 1, 1]>, #hivm.address_space<ub>>
      %sv1 = memref.subview %sv0[0, 0, 0] [64, 64, 1] [1, 1, 1] : memref<64x128x1xi1, strided<[256, 1, 1]>, #hivm.address_space<ub>> to memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>>
      "test.use"(%sv1) : (memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>>) -> ()
      scope.return %sv1 : memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>>
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    "test.consume"(%s0) : (memref<64x64xi1, strided<[256, 1]>, #hivm.address_space<ub>>) -> ()
  }
  return
}

// -----

// Each parent loop must use its own declared preload depth. Collecting preload
// numbers module-wide would truncate the second loop to depth 2 and drop stage 3.

// CHECK-LABEL: func.func @test_preload_depth_is_per_loop
func.func @test_preload_depth_is_per_loop() {
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32

  // CHECK-DAG: %[[C18:.*]] = arith.constant 18 : i32
  // CHECK-DAG: %[[C20:.*]] = arith.constant 20 : i32
  // CHECK: scf.for {{.*}} to %[[C18]] step
  scf.for %i = %c0 to %c16 step %c1 : i32 {
    scope.scope : () -> () {
      "test.stage1"(%i) : (i32) -> ()
      scope.return
    } {
      no_inline,
      hivm.preload_num = 1 : i32,
      hivm.max_preload_num = 2 : i32
    }
  }

  // CHECK: scf.for {{.*}} to %[[C20]] step
  // CHECK: "test.stage3"
  scf.for %i = %c0 to %c16 step %c1 : i32 {
    scope.scope : () -> () {
      "test.stage3"(%i) : (i32) -> ()
      scope.return
    } {
      no_inline,
      hivm.preload_num = 3 : i32,
      hivm.max_preload_num = 4 : i32
    }
  }

  return
}

// -----

// A view returned by a scope must use the preload-local buffer selected for
// each mapping, rather than becoming one shared conditional result.

// CHECK-LABEL: func.func @test_preload_local_view_uses_mapping
func.func @test_preload_local_view_uses_mapping() {
  %c0_i64 = arith.constant 0 : i64
  %c128_i64 = arith.constant 128 : i64
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32

  scf.for %i = %c0 to %c4 step %c1 : i32 {
    %buffer = hivm.hir.pointer_cast(%c0_i64, %c128_i64)
      : memref<2x64xi1, #hivm.address_space<ub>>
    annotation.mark %buffer {
      hivm.multi_buffer = 2 : i32,
      hivm.preload_local_buffer = 1 : i32
    } : memref<2x64xi1, #hivm.address_space<ub>>

    %view = scope.scope : () -> memref<1x64xi1, strided<[64, 1]>, #hivm.address_space<ub>> {
      "test.produce"(%i) : (i32) -> ()
      %subview = memref.subview %buffer[0, 0] [1, 64] [1, 1]
        : memref<2x64xi1, #hivm.address_space<ub>>
          to memref<1x64xi1, strided<[64, 1]>, #hivm.address_space<ub>>
      scope.return %subview
        : memref<1x64xi1, strided<[64, 1]>, #hivm.address_space<ub>>
    } {
      no_inline,
      hivm.preload_num = 1 : i32,
      hivm.max_preload_num = 2 : i32
    }

    scope.scope : () -> () {
      "test.consume"(%view)
        : (memref<1x64xi1, strided<[64, 1]>, #hivm.address_space<ub>>) -> ()
      scope.return
    } {
      no_inline,
      hivm.preload_num = 0 : i32,
      hivm.max_preload_num = 2 : i32
    }
  }
  return
}

// CHECK: %[[MAPPING1_BUFFER:.*]] = hivm.hir.pointer_cast
// CHECK: %[[MAPPING0_BUFFER:.*]] = hivm.hir.pointer_cast
// CHECK: scf.if {{.*}} {
// CHECK:   "test.produce"
// CHECK: }
// CHECK: %[[MAPPING0_VIEW:.*]] = memref.subview %[[MAPPING0_BUFFER]]
// CHECK: "test.consume"(%[[MAPPING0_VIEW]])
