// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// Test that warp shuffle butterfly reduction with num_warps=64 correctly skips
// N=32 (self-shuffle case) and only uses valid offsets: 16, 8, 4, 2, 1.
//
// Bug: When N=32 and numLanes=32, shuffleXor offset=32 wraps to same lane (self-shuffle),
// causing the accumulated value to double instead of exchanging with a peer.
// Fix: Skip butterfly steps where N * interleave >= numLanes.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [64, 1], order = [1, 0]}>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, "ttg.enable-bishengir-simt-optimization" = 900100 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 64 : i32, ttg.shared = 8192 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @reduce_axis_0_64warps
  tt.func public @reduce_axis_0_64warps(%arg0: tensor<64x32xf32, #blocked>) attributes {noinline = false} {
    // CHECK: llvm.mlir.constant(16 : i32) : i32
    // CHECK: ascend_dpx.shfl.bfly
    // CHECK: llvm.mlir.constant(8 : i32) : i32
    // CHECK: ascend_dpx.shfl.bfly
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    // CHECK: ascend_dpx.shfl.bfly
    // CHECK: llvm.mlir.constant(2 : i32) : i32
    // CHECK: ascend_dpx.shfl.bfly
    // CHECK: llvm.mlir.constant(1 : i32) : i32
    // CHECK: ascend_dpx.shfl.bfly
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) {allocation.offset = 0 : i32} : (tensor<64x32xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return
  }
}