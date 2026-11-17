// RUN: bishengir-opt %s -hivm-mark-preload-buffers -o %t
// RUN: bishengir-opt %t -hivm-mark-preload-buffers -o %t.again
// RUN: diff %t %t.again
// RUN: FileCheck %s --input-file=%t

// Stage 0 writes through a view, stage 2 reads the root. Three versions share
// one allocation mark, even when the pass runs a second time. Stages 0
// and 1 need two versions. A buffer used only in stage 0 has no mark.
// CHECK-LABEL: @versions
// CHECK: %[[FULL:.*]] = memref.alloc
// CHECK-NEXT: annotation.mark %[[FULL]] {{.*}}hivm.multi_buffer = 3 : i32{{.*}}hivm.preload_local_buffer{{.*}}mem_unique
// CHECK-NEXT: %[[PARTIAL:.*]] = memref.alloc
// CHECK-NEXT: annotation.mark %[[PARTIAL]] {{.*}}hivm.multi_buffer = 2 : i32{{.*}}hivm.preload_local_buffer
// CHECK-NOT: annotation.mark
// CHECK: return
func.func @versions(%n: index, %x: f32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %b = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
    annotation.mark %b {mem_unique} : memref<1xf32, #hivm.address_space<ub>>
    %partial = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
    %same_stage = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
    %view = memref.cast %b : memref<1xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      memref.store %x, %view[%c0] : memref<?xf32, #hivm.address_space<ub>>
      memref.store %x, %partial[%c0] : memref<1xf32, #hivm.address_space<ub>>
      memref.store %x, %same_stage[%c0] : memref<1xf32, #hivm.address_space<ub>>
      %same = memref.load %same_stage[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}
    scope.scope : () -> () {
      %mid = memref.load %partial[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 1 : i32, no_inline}
    scope.scope : () -> () {
      %v = memref.load %b[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
  } {hivm.cv_pipelined_loop}
  return
}

// Only marked CV loops participate in buffer marking.
// CHECK-LABEL: @unmarked_loop
// CHECK-NOT: annotation.mark
// CHECK: return
func.func @unmarked_loop(%n: index, %x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %b = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      memref.store %x, %b[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.preload_num = 2 : i32, hivm.max_preload_num = 3 : i32}
    scope.scope : () -> () {
      %v = memref.load %b[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.preload_num = 0 : i32, hivm.max_preload_num = 3 : i32}
  }
  return
}
