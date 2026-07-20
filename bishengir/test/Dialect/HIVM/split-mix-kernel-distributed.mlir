// RUN: bishengir-opt %s -hivm-split-mix-kernel -split-input-file -verify-diagnostics | FileCheck %s

// Test inferDistributedCoreType - core type inference for distributed CustomOp in SplitMixKernel pass

// -----
// Test distributed notify op in MIX kernel - should infer core type correctly

module {
  // CHECK-LABEL: distributed_in_mix_mix_aic
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK: hivm.tcore_type = #hivm.tcore_type<CUBE>, symbol = "aclshmem_int16_p"
  // CHECK-LABEL: distributed_in_mix_mix_aiv
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  func.func @distributed_in_mix(%arg0: memref<64x64xf16>,
                                 %arg1: memref<64x64xf16>,
                                 %arg2: memref<64x64xf16>,
                                 %arg3: memref<64x64xf16>,
                                 %arg4: memref<64x64xf16>,
                                 %arg5: memref<64x64xi64>)
                                 attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    // First a matmul (CUBE op)
    hivm.hir.matmul ins(%arg0, %arg1 : memref<64x64xf16>, memref<64x64xf16>) outs(%arg2: memref<64x64xf16>)
    // Then a distributed custom op - with explicit core type
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, symbol = "aclshmem_int16_p"} "dist.aclshmem_int16_p" ins(%arg5, %c1_i64, %c0_i32 : memref<64x64xi64>, i64, i32)
    // Finally a vadd (VECTOR op)
    hivm.hir.vadd ins(%arg3, %arg3 : memref<64x64xf16>, memref<64x64xf16>) outs(%arg4 : memref<64x64xf16>)
    return
  }
}

// -----
// Test distributed notify op before CUBE op - should get VECTOR core type

module {
  // CHECK-LABEL: distributed_in_mix_mix_aic
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: distributed_in_mix_mix_aiv
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  // CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_int16_p"
  func.func @distributed_in_mix(%arg0: memref<64x64xf16>,
                                 %arg1: memref<64x64xf16>,
                                 %arg2: memref<64x64xf16>,
                                 %arg3: memref<64x64xf16>,
                                 %arg4: memref<64x64xf16>,
                                 %arg5: memref<64x64xi64>)
                                 attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    // First a distributed custom op - with explicit core type
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, symbol = "aclshmem_int16_p"} "dist.aclshmem_int16_p" ins(%arg5, %c1_i64, %c0_i32 : memref<64x64xi64>, i64, i32)
    // Then matmul (CUBE op)
    hivm.hir.matmul ins(%arg0, %arg1 : memref<64x64xf16>, memref<64x64xf16>) outs(%arg2: memref<64x64xf16>)
    // Finally a vadd (VECTOR op)
    hivm.hir.vadd ins(%arg3, %arg3 : memref<64x64xf16>, memref<64x64xf16>) outs(%arg4 : memref<64x64xf16>)
    return
  }
}

// -----
// // Test distributed notify op after VECTOR op - should get VECTOR core type

module {
  // CHECK-LABEL: distributed_in_mix_mix_aic
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: distributed_in_mix_mix_aiv
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  // CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_int16_p"
  func.func @distributed_in_mix(%arg0: memref<64x64xf16>,
                                 %arg1: memref<64x64xf16>,
                                 %arg2: memref<64x64xf16>,
                                 %arg3: memref<64x64xf16>,
                                 %arg4: memref<64x64xf16>,
                                 %arg5: memref<64x64xi64>)
                                 attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    // First a matmul (CUBE op)
    hivm.hir.matmul ins(%arg0, %arg1 : memref<64x64xf16>, memref<64x64xf16>) outs(%arg2: memref<64x64xf16>)
    // Then a vadd (VECTOR op)
    hivm.hir.vadd ins(%arg3, %arg3 : memref<64x64xf16>, memref<64x64xf16>) outs(%arg4 : memref<64x64xf16>)
    // Then a distributed custom op - with explicit core type
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, symbol = "aclshmem_int16_p"} "dist.aclshmem_int16_p" ins(%arg5, %c1_i64, %c0_i32 : memref<64x64xi64>, i64, i32)
    return
  }
}

// -----
// Test result-bearing VECTOR-only distributed custom op (e.g. aclshmem_ptr_*).
// When building the AIC kernel the VECTOR op is erased; because it has a result
// but no init/output operand, getOutOperands must synthesize a replacement for
// its uses instead of asserting. Here the result aliases an input memref, so the
// matching input operand is forwarded.

module {
  // CHECK-LABEL: dist_ptr_with_result_mix_aic
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // The VECTOR-only ptr op and its users must be gone from the AIC kernel.
  // CHECK-NOT: aclshmem_ptr_int64
  // CHECK-NOT: hivm.hir.store
  //
  // CHECK-LABEL: dist_ptr_with_result_mix_aiv
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  // The AIV kernel keeps the result-bearing distributed op and its store.
  // CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>{{.*}}symbol = "aclshmem_ptr_int64"
  // CHECK: hivm.hir.store
  func.func @dist_ptr_with_result(%arg0: memref<?xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    // CUBE_AND_VECTOR barrier keeps the func mixed.
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    // VECTOR-only distributed custom op that returns a memref aliasing %arg0.
    %ptr = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_ptr_int64"} "dist.aclshmem_ptr_int64" ins(%arg0, %c1_i32 : memref<?xi64>, i32) -> memref<?xi64>
    %cast = memref.reinterpret_cast %ptr to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
    %e = tensor.empty() : tensor<1xi64>
    %ins = tensor.insert %c1_i64 into %e[%c0] : tensor<1xi64>
    hivm.hir.store ins(%ins : tensor<1xi64>) outs(%cast : memref<1xi64, strided<[1]>>)
    return
  }
}

// -----
// Test distributed custom op with result type that doesn't match any input.
// This exercises the createZeroOrEmptyStub fallback path where the result type
// has no matching input operand. Uses a dummy distributed op for testing purposes.

module {
  // CHECK-LABEL: dist_stub_generation_mix_aic
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // The AIC kernel should have a stub (arith.constant 0) replacing the VECTOR op result.
  // CHECK: %[[STUB:.*]] = arith.constant 0 : i32
  // CHECK-NOT: dummy_dist_op
  // CHECK: tensor.insert{{.*}}%[[STUB]]
  //
  // CHECK-LABEL: dist_stub_generation_mix_aiv
  // CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  // The AIV kernel keeps the distributed op returning i32.
  // CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>{{.*}}symbol = "dummy_dist_op"
  // CHECK: tensor.insert
  func.func @dist_stub_generation(%arg0: memref<4xi64>) -> tensor<4xi32> attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    // CUBE_AND_VECTOR barrier keeps the func mixed.
    hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, symbol = "aclshmem_barrier_all"} "dist.aclshmem_barrier_all"
    // VECTOR-only distributed custom op with inputs memref<4xi64> but returns i32 (rank ID).
    // No operand has type i32 → createZeroOrEmptyStub is called, generates arith.constant 0.
    %rank = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "dummy_dist_op"} "dist.dummy_dist_op" ins(%arg0: memref<4xi64>) -> i32
    %empty = tensor.empty() : tensor<4xi32>
    %result = tensor.insert %rank into %empty[%c0] : tensor<4xi32>
    return %result : tensor<4xi32>
  }
}