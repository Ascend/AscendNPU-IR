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