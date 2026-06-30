// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-mem-scope -split-input-file | FileCheck %s

// Test inferAndPropagateMemScopeForDistributed - memory scope inference for distributed CustomOp

// -----
// Test distributed custom op - should return GM space ptr

// CHECK-LABEL: test_distributed_mem_scope_basic
module {
  func.func @test_distributed_mem_scope_basic(%arg0: memref<128x128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK-DAG: ins(%arg0 : memref<128x128xf32, #hivm.address_space<gm>>) -> memref<128x128xf32, #hivm.address_space<gm>>
    %0 = hivm.hir.custom {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "aclshmem_ptr_float"} "dist.aclshmem_ptr_float" ins(%arg0 : memref<128x128xf32>) -> memref<128x128xf32>
    return
  }
}

// -----
// Test distributed custom op with dist.aclshmem_int32_p - should use GM space ptr as operand

// CHECK-LABEL: test_shmem_int32_scope
module {
  func.func @test_shmem_int32_scope(%arg0: memref<32x32xi64>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    // CHECK-DAG: ins(%arg0, %c1_i64, %c0_i32 : memref<32x32xi64, #hivm.address_space<gm>>, i64, i32)
    hivm.hir.custom {commScope = 2 : i32, hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<CUBE_AND_VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>, sigOp = 1 : i32, symbol = "aclshmem_int64_p"} "dist.aclshmem_int64_p" ins(%arg0, %c1_i64, %c0_i32 : memref<32x32xi64>, i64, i32)
    return
  }
}
