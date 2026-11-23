// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize-buffers --hivm-v2-split-kernel | FileCheck %s
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_950z --hivm-vectorize-buffers | FileCheck %s --check-prefix=A5

!ub = memref<1xf32, #hivm.address_space<ub>>
!gm = memref<1xf32, #hivm.address_space<gm>>
!indices = memref<1xindex, #hivm.address_space<ub>>
!unsigned = memref<1xui8, #hivm.address_space<ub>>
!wide = memref<1xi64, #hivm.address_space<ub>>
!pred = memref<1xi1, #hivm.address_space<ub>>
module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func private @consume(%tile: !ub, %scale: f32)
      attributes {hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>}

  // The vector side already consumes the scalar. The cube side must not write
  // its temporary once the vector function starts updating that UB slot.
  // CHECK-LABEL: func.func @gate_mix_aic(
  // CHECK-NOT: memref.store
  // CHECK: return
  // CHECK-LABEL: func.func @gate_mix_aiv(
  // CHECK-NOT: memref.store
  // CHECK: call @{{.*}}({{.*}}) {{.*}}hivm.vector_function
  // CHECK: call @consume(
  // A5-LABEL: func.func @gate(
  // A5: memref.store
  func.func @gate(%raw: !gm) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %zero = arith.constant 0 : index
    %p = memref.alloc() : !ub
    %value = memref.load %raw[%zero] : !gm
    memref.store %value, %p[%zero] : !ub
    func.call @consume(%p, %value) {hivm.vector_function} : (!ub, f32) -> ()
    return
  }

  // A vector call may publish the buffer address before the caller writes it.
  // CHECK-LABEL: func.func @after_call_mix_aic(
  // CHECK: memref.store
  // CHECK-LABEL: func.func @after_call_mix_aiv(
  // CHECK: call @consume(
  // CHECK: memref.store
  func.func @after_call(%value: f32) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %zero = arith.constant 0 : index
    %p = memref.alloc() : !ub
    func.call @consume(%p, %value) {hivm.vector_function} : (!ub, f32) -> ()
    memref.store %value, %p[%zero] : !ub
    return
  }

  // The same call may already have published this allocation on a prior trip.
  // CHECK-LABEL: func.func @prior_iteration_mix_aic(
  // CHECK: scf.for
  // CHECK: memref.store
  // CHECK-LABEL: func.func @prior_iteration_mix_aiv(
  // CHECK: memref.store
  // CHECK: call @consume(
  func.func @prior_iteration(%n: index, %value: f32) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %p = memref.alloc() : !ub
    scf.for %i = %zero to %n step %one {
      memref.store %value, %p[%zero] : !ub
      func.call @consume(%p, %value) {hivm.vector_function} : (!ub, f32) -> ()
    }
    return
  }

  // A private pointer escapes to another core via a load. Both local reads
  // see the caller's original scalar store.
  // CHECK-LABEL: func.func @shared_mix_aic(
  // CHECK: memref.store {{.*}} : memref<1xf32, #hivm.address_space<ub>>
  // CHECK: memref.load
  // CHECK-LABEL: func.func @shared_mix_aiv(
  // CHECK: memref.store {{.*}} : memref<1xf32, #hivm.address_space<ub>>
  // CHECK: call @consume(
  func.func @shared(%out: !gm, %v: f32) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %zero = arith.constant 0 : index
    %p = memref.alloc() : !ub
    memref.store %v, %p[%zero] : !ub
    func.call @consume(%p, %v) {hivm.vector_function} : (!ub, f32) -> ()
    %x = memref.load %p[%zero] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : !ub
    memref.store %x, %out[%zero] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : !gm
    return
  }

  // These scalar stores have no native masked single-lane broadcast/store.
  // CHECK-LABEL: func.func @unsupported_types_mix_aiv(
  // CHECK: memref.store {{.*}} : memref<1xindex
  // CHECK: memref.store {{.*}} : memref<1xui8
  // CHECK: memref.store {{.*}} : memref<1xi64
  // CHECK: memref.store {{.*}} : memref<1xi1,
  func.func private @use_scalars(!indices, !unsigned, !wide, !pred)
      attributes {hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>}
  func.func @unsupported_types(%index: index, %unsigned: ui8, %wide: i64, %pred: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %zero = arith.constant 0 : index
    %p = memref.alloc() : !indices
    %u = memref.alloc() : !unsigned
    %w = memref.alloc() : !wide
    %b = memref.alloc() : !pred
    memref.store %index, %p[%zero] : !indices
    memref.store %unsigned, %u[%zero] : !unsigned
    memref.store %wide, %w[%zero] : !wide
    memref.store %pred, %b[%zero] : !pred
    func.call @use_scalars(%p, %u, %w, %b) {hivm.vector_function} : (!indices, !unsigned, !wide, !pred) -> ()
    return
  }

  // CHECK-LABEL: func.func @cube_owned_mix_aic(
  // CHECK: memref.store
  // CHECK-LABEL: func.func @cube_owned_mix_aiv(
  // CHECK-NOT: memref.store
  // CHECK: call @consume(
  func.func @cube_owned(%value: f32) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %zero = arith.constant 0 : index
    %p = memref.alloc() : !ub
    scope.scope : () -> () {
      memref.store %value, %p[%zero] : !ub
      func.call @consume(%p, %value) {hivm.vector_function} : (!ub, f32) -> ()
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
