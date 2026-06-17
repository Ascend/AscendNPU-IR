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