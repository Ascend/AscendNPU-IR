// RUN: bishengir-opt %s -hivm-split-mix-kernel -split-input-file -verify-diagnostics | FileCheck %s

// core_ratio behaviour around the mix split. Kept out of split-mix-kernel.mlir
// because that file is UNSUPPORTED on a published build -- its RUN line pulls in
// SplitMixedIfConditionals and the TCB passes, none of which these cases need.

// Check that core_ratio is also defined in the decl.
module {
  // CHECK-LABEL: func.func private @ratio_mixed_matmul
  // CHECK-SAME: hacc.mix_entry
  // CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 1>
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>

  // CHECK-LABEL: ratio_mixed_matmul_mix_aic({{.*}} hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: ratio_mixed_matmul_mix_aiv({{.*}} hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  func.func @ratio_mixed_matmul(%arg0: tensor<64x64xf16>,
                                %arg1: tensor<64x64xf16>,
                                %arg2: tensor<64x64xf16>,
                                %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %1 = hivm.hir.mix_matmul ins(%arg0, %arg2 : tensor<64x64xf16>, tensor<64x64xf16>)
                         post_vector_func_ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
                         outs(%arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }

  func.func @ratio_host_caller(%arg0: tensor<64x64xf16>,
                               %arg1: tensor<64x64xf16>,
                               %arg2: tensor<64x64xf16>,
                               %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %1 = func.call @ratio_mixed_matmul(%arg0, %arg1, %arg2, %arg3) : (tensor<64x64xf16>,tensor<64x64xf16>,tensor<64x64xf16>,tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }
}
// -----
// Each half holds only its own core's work after the split, so the ratio is
// checked against what the kernel actually needs. 0:1 reserves no cube core.
module {
  // expected-error@+1 {{hivm.core_ratio<0, 1> reserves no cube core, but the kernel has cube work}}
  func.func @cube_work_with_ratio_0_1(%lhs: tensor<8x8xf16>, %rhs: tensor<8x8xf16>,
                                      %dst: tensor<8x8xf16>) -> tensor<8x8xf16>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<0, 1>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %0 = hivm.hir.matmul ins(%lhs, %rhs : tensor<8x8xf16>, tensor<8x8xf16>) outs(%dst : tensor<8x8xf16>) -> tensor<8x8xf16>
    return %0 : tensor<8x8xf16>
  }
}
// -----
// The mirror case, which nothing checked before: 1:0 reserves no vector core.
module {
  // expected-error@+1 {{hivm.core_ratio<1, 0> reserves no vector core, but the kernel has vector work}}
  func.func @vector_work_with_ratio_1_0(%arg0: tensor<64xf32>, %arg1: memref<64xf32>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<1, 0>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    hivm.hir.store ins(%1 : tensor<64xf32>) outs(%arg1 : memref<64xf32>)
    return
  }
}
// -----
// A vector-only kernel at 0:1 must compile: its aic half is empty, so there is
// no cube work to conflict with the ratio. Before the check moved here, the
// pre-split walk inferred MIX for bodies like this and rejected them.
// CHECK-LABEL: func.func @vector_only_ratio_0_1_mix_aic
// CHECK-NOT:     hivm.hir
// CHECK-LABEL: func.func @vector_only_ratio_0_1_mix_aiv
// CHECK:         hivm.hir.vln
module {
  func.func @vector_only_ratio_0_1(%arg0: tensor<64xf32>, %arg1: memref<64xf32>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<0, 1>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %0 = tensor.empty() : tensor<64xf32>
    %1 = hivm.hir.vln ins(%arg0 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    hivm.hir.store ins(%1 : tensor<64xf32>) outs(%arg1 : memref<64xf32>)
    return
  }
}

// -----

// A distributed vector-only kernel: its shmem ops are VECTOR, so they are
// removed from the cube half and retained in the vector half. 0:1 must compile.
// CHECK-LABEL: func.func @distributed_vv_ratio_0_1_mix_aic
// CHECK-NOT:     aclshmemx_barrier_all_vec
// CHECK-LABEL: func.func @distributed_vv_ratio_0_1_mix_aiv
// CHECK:         aclshmemx_barrier_all_vec
module {
  func.func @distributed_vv_ratio_0_1(%counter_ptr: memref<?xi64>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<0, 1>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %cast = memref.reinterpret_cast %counter_ptr to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
    %val = memref.load %cast[%c0] : memref<1xi64, strided<[1]>>
    hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmemx_barrier_all_vec"} "dist.aclshmemx_barrier_all_vec"
    %sum = arith.addi %val, %c1_i64 : i64
    %e = tensor.empty() : tensor<1xi64>
    %ins = tensor.insert %sum into %e[%c0] : tensor<1xi64>
    bufferization.materialize_in_destination %ins in writable %cast : (tensor<1xi64>, memref<1xi64, strided<[1]>>) -> ()
    return
  }
}

// -----

// A distributed operation that requires both cores is retained in the cube
// half and relabelled CUBE there. 0:1 must reject it.
module {
  // expected-error@+1 {{hivm.core_ratio<0, 1> reserves no cube core, but the kernel has cube work}}
  func.func @distributed_cv_ratio_0_1()
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<0, 1>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, libname = "libshmem_device", libpath = "", pure = false, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    return
  }
}
