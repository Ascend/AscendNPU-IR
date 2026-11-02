// RUN: bishengir-opt %s -hivm-bind-sub-block -verify-diagnostics | FileCheck %s

// A 1x1 scalar TCB stays untiled (tiling_dim=-1), so the AIC fixpipe cannot be
// split into a dual destination. If the AIV is still split, one sub-block can
// read a slot the AIC did not write. The pass must roll the pair back to CV1:1
// and limit the AIV store to sub-block 0.
// CHECK-LABEL: func.func @scalar_tcb_fixpipe_cv_aic(
// CHECK: hivm.hir.fixpipe
// CHECK-SAME: outs(%{{.*}} : memref<1x1xf32, #hivm.address_space<ub>>){{$}}
// CHECK-NOT: dual_dst_mode
// CHECK-LABEL: func.func @scalar_tcb_fixpipe_cv_aiv(
// CHECK: scf.if
// CHECK: hivm.hir.store
// CHECK: {limit_sub_block_id0}
// CHECK-NOT: map_for_to_forall
module attributes {hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @scalar_tcb_fixpipe_cv_aic(%a: i32, %b: i32, %c: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %0 = arith.muli %a, %b : i32
    %1 = arith.muli %0, %c : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<1x1xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<1x1xf32, #hivm.address_space<ub>>
    %sa = tensor.empty() : tensor<1x4xf32>
    %sb = tensor.empty() : tensor<4x1xf32>
    %sout = tensor.empty() : tensor<1x1xf32>
    %smmad = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%sa, %sb, %true, %c1, %c4, %c1 : tensor<1x4xf32>, tensor<4x1xf32>, i1, index, index, index) outs(%sout : tensor<1x1xf32>) -> tensor<1x1xf32>
    // expected-warning@+1 {{The tilingDim in AIC does not exist!}}
    hivm.hir.fixpipe ins(%smmad : tensor<1x1xf32>) outs(%alloc : memref<1x1xf32, #hivm.address_space<ub>>)
    return
  }

  func.func @scalar_tcb_fixpipe_cv_aiv(%out: memref<1x1024xi32>, %a: i32, %b: i32, %c: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c0f = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %a, %b : i32
    %1 = arith.muli %0, %c : i32
    annotation.mark %1 {logical_block_num} : i32
    %alloc = memref.alloc() : memref<1x1024xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<1x1024xf32, #hivm.address_space<ub>>
    %scalar = memref.alloc() : memref<1x1xf32, #hivm.address_space<ub>>
    annotation.mark %scalar {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<1x1xf32, #hivm.address_space<ub>>
    %cast = memref.memory_space_cast %alloc : memref<1x1024xf32, #hivm.address_space<ub>> to memref<1x1024xf32>
    %t = bufferization.to_tensor %cast restrict writable : memref<1x1024xf32>
    %scast = memref.memory_space_cast %scalar : memref<1x1xf32, #hivm.address_space<ub>> to memref<1x1xf32>
    %st = bufferization.to_tensor %scast restrict writable : memref<1x1xf32>
    %dot = tensor.extract %st[%c0, %c0] : tensor<1x1xf32>
    %amp = arith.mulf %dot, %c0f : f32
    %e = tensor.empty() : tensor<1x1024xf32>
    %bcast = hivm.hir.vbrc ins(%amp : f32) outs(%e : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %o = tensor.empty() : tensor<1x1024xf32>
    %add = hivm.hir.vadd ins(%t, %bcast : tensor<1x1024xf32>, tensor<1x1024xf32>) outs(%o : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %oi = tensor.empty() : tensor<1x1024xi32>
    %v = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%add : tensor<1x1024xf32>) outs(%oi : tensor<1x1024xi32>) -> tensor<1x1024xi32>
    hivm.hir.store ins(%v : tensor<1x1024xi32>) outs(%out : memref<1x1024xi32>)
    return
  }
}
