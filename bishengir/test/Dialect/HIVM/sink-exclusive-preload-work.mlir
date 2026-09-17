// RUN: bishengir-opt %s -hivm-sink-exclusive-preload-work | FileCheck %s

func.func @vf_consume(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  return
}

// multi_buffer=1 + preload_local=1 alloc used by one scope is sunk into
// that scope and its mark is deleted. multi_buffer>1 stays at loop scope
// so slots can still rotate.
// CHECK-LABEL: func.func @sink_single_slot_preload_alloc
// CHECK-NOT:     memref.alloc
// CHECK:         scope.scope
// CHECK:           %[[BUF:.*]] = memref.alloc
// CHECK-NOT:       annotation.mark %[[BUF]]
// CHECK:           call @vf_consume(%[[BUF]])
func.func @sink_single_slot_preload_alloc()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 1 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%buf) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// CHECK-LABEL: func.func @keep_rotated_preload_alloc_in_loop
// CHECK:         %[[BUF:.*]] = memref.alloc
// CHECK:         annotation.mark %[[BUF]] {{.*}}multi_buffer = 2{{.*}}preload_local_buffer
// CHECK:         scope.scope
func.func @keep_rotated_preload_alloc_in_loop()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 2 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%buf) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// TCB + mb=1 + preload_local still sinks. The TCB mark is kept;
// only the single-slot preload marks are deleted.
// CHECK-LABEL: func.func @sink_tcb_single_slot_preload_alloc
// CHECK-NOT:     memref.alloc
// CHECK:         scope.scope
// CHECK:           %[[BUF:.*]] = memref.alloc
// CHECK:           annotation.mark %[[BUF]] {{.*}}tightly_coupled_buffer
// CHECK-NOT:       annotation.mark %[[BUF]] {{.*}}preload_local_buffer
// CHECK:           call @vf_consume(%[[BUF]])
func.func @sink_tcb_single_slot_preload_alloc()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {
      effects = ["write", "read"],
      hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>,
      hivm.tiling_dim = 0 : index, tiledAlloc
    } : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 1 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%buf) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}

// A loop-level subview of a single-slot preload alloc follows the alloc
// into the unique consumer scope.
// CHECK-LABEL: func.func @sink_preload_alloc_with_subview
// CHECK-NOT:     memref.alloc
// CHECK:         scope.scope
// CHECK:           %[[BUF:.*]] = memref.alloc
// CHECK-NOT:       annotation.mark %[[BUF]]
// CHECK:           %[[SV:.*]] = memref.subview %[[BUF]]
// CHECK:           call @vf_consume(%[[SV]])
func.func @sink_preload_alloc_with_subview()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    %sv = memref.subview %buf[0, 0] [32, 128] [1, 1] :
        memref<32x128xf32, #hivm.address_space<ub>> to
        memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 1 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%sv) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}
