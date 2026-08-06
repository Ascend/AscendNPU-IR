// RUN: bishengir-opt %s -hivm-bind-sub-block | FileCheck %s

// CHECK-LABEL: func.func @distributed_dataflow_custom_ops_do_not_bind_to_subblock_zero
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK-NOT: scf.if
// CHECK: hivm.hir.custom {{.*}}aclshmem_wait_int64
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK-NOT: scf.if
// CHECK: hivm.hir.custom {{.*}}aclshmem_signal_wait_until
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK-NOT: scf.if
// CHECK: hivm.hir.custom {{.*}}aclshmem_n_pes
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK-NOT: scf.if
// CHECK: hivm.hir.custom {{.*}}aclshmem_ptr_half
// CHECK-NOT: hivm.hir.get_sub_block_idx
// CHECK-NOT: scf.if
// CHECK: hivm.hir.custom {{.*}}aclshmem_consume_token_half_ptr_1d
func.func @distributed_dataflow_custom_ops_do_not_bind_to_subblock_zero(
    %signal: memref<1xi64>, %signal32: memref<1xi32>, %data: memref<1xf16>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.part_of_mix, mix_mode = "mix"} {
  %rank = arith.constant 0 : i32
  %value = arith.constant 1 : i64
  %token = hivm.hir.custom {hivm.is_distributed,
                             hivm.pipe = #hivm.pipe<PIPE_S>,
                             hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>,
                             hivm.vf_mode = #hivm.vf_mode<SIMD>,
                             symbol = "aclshmem_wait_int64"}
      "dist.aclshmem_wait_int64"
      ins(%signal, %rank, %value : memref<1xi64>, i32, i64) -> i32
  %signalValue = arith.constant 1 : i32
  %signalWait = hivm.hir.custom {hivm.is_distributed,
                                  hivm.pipe = #hivm.pipe<PIPE_S>,
                                  hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>,
                                  hivm.vf_mode = #hivm.vf_mode<SIMD>,
                                  symbol = "aclshmem_signal_wait_until"}
      "dist.aclshmem_signal_wait_until"
      ins(%signal32, %rank, %signalValue : memref<1xi32>, i32, i32) -> i32
  %numPEs = hivm.hir.custom {hivm.is_distributed,
                              hivm.pipe = #hivm.pipe<PIPE_S>,
                              hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>,
                              hivm.vf_mode = #hivm.vf_mode<SIMD>,
                              symbol = "aclshmem_n_pes"}
      "dist.aclshmem_n_pes" -> i32
  %ptr = hivm.hir.custom {hivm.is_distributed,
                           hivm.pipe = #hivm.pipe<PIPE_S>,
                           hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>,
                           hivm.vf_mode = #hivm.vf_mode<SIMD>,
                           symbol = "aclshmem_ptr_half"}
      "dist.aclshmem_ptr_half" ins(%data, %rank : memref<1xf16>, i32) -> memref<1xf16>
  %result = hivm.hir.custom {hivm.is_distributed,
                              hivm.pipe = #hivm.pipe<PIPE_S>,
                              hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>,
                              hivm.vf_mode = #hivm.vf_mode<SIMD>,
                              symbol = "aclshmem_consume_token_half_ptr_1d"}
      "dist.aclshmem_consume_token_half_ptr_1d"
      ins(%ptr, %token : memref<1xf16>, i32) -> memref<1xf16>
  return
}

// -----

// CHECK-LABEL: func.func @other_distributed_custom_op_stays_limited
// CHECK: hivm.hir.get_sub_block_idx
// CHECK: scf.if
// CHECK: hivm.hir.custom {{.*}}aclshmem_barrier_all
// CHECK: } {limit_sub_block_id0}
func.func @other_distributed_custom_op_stays_limited()
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.part_of_mix, mix_mode = "mix"} {
  hivm.hir.custom {hivm.is_distributed,
                    hivm.pipe = #hivm.pipe<PIPE_S>,
                    hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>,
                    hivm.vf_mode = #hivm.vf_mode<SIMD>,
                    symbol = "aclshmem_barrier_all"}
      "dist.aclshmem_barrier_all"
  return
}
