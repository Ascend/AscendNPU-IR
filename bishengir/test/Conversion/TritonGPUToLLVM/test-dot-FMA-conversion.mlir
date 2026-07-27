// RUN: bishengir-opt -split-input-file --convert-triton-ascend-gpu-to-llvm %s | FileCheck %s


#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared   = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_a    = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_b    = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
#smem     = #ttg.shared_memory

// M=32, K=16, N=32  –  f32 FMA dot
module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    "ttg.enable-bishengir-simt-optimization" = 111 : i32
} {
  // CHECK-LABEL: @matmul_fma_f32
  tt.func public @matmul_fma_f32(
      %a_ptr : !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %b_ptr : !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %a_m_range = tt.make_range {end = 32 : i32, start = 0 : i32}
                    : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %a_k_range = tt.make_range {end = 16 : i32, start = 0 : i32}
                    : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_m_2d    = tt.expand_dims %a_m_range {axis = 1 : i32}
                    : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
                   -> tensor<32x1xi32, #blocked>
    %a_k_2d    = tt.expand_dims %a_k_range {axis = 0 : i32}
                    : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
                   -> tensor<1x16xi32, #blocked>
    %stride_k  = arith.constant dense<16> : tensor<32x1xi32, #blocked>
    %a_m_off   = arith.muli %a_m_2d, %stride_k : tensor<32x1xi32, #blocked>
    %a_m_broad = tt.broadcast %a_m_off : tensor<32x1xi32, #blocked>
                                       -> tensor<32x16xi32, #blocked>
    %a_k_broad = tt.broadcast %a_k_2d  : tensor<1x16xi32, #blocked>
                                       -> tensor<32x16xi32, #blocked>
    %a_offsets = arith.addi %a_m_broad, %a_k_broad : tensor<32x16xi32, #blocked>
    %a_base    = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<32x16x!tt.ptr<f32>, #blocked>
    %a_ptrs    = tt.addptr %a_base, %a_offsets
                    : tensor<32x16x!tt.ptr<f32>, #blocked>, tensor<32x16xi32, #blocked>


    %b_k_range = tt.make_range {end = 16 : i32, start = 0 : i32}
                    : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %b_n_range = tt.make_range {end = 32 : i32, start = 0 : i32}
                    : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %b_k_2d    = tt.expand_dims %b_k_range {axis = 1 : i32}
                    : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
                   -> tensor<16x1xi32, #blocked>
    %b_n_2d    = tt.expand_dims %b_n_range {axis = 0 : i32}
                    : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
                   -> tensor<1x32xi32, #blocked>
    %stride_n  = arith.constant dense<32> : tensor<16x1xi32, #blocked>
    %b_k_off   = arith.muli %b_k_2d, %stride_n : tensor<16x1xi32, #blocked>
    %b_k_broad = tt.broadcast %b_k_off : tensor<16x1xi32, #blocked>
                                       -> tensor<16x32xi32, #blocked>
    %b_n_broad = tt.broadcast %b_n_2d  : tensor<1x32xi32, #blocked>
                                       -> tensor<16x32xi32, #blocked>
    %b_offsets = arith.addi %b_k_broad, %b_n_broad : tensor<16x32xi32, #blocked>
    %b_base    = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked>
    %b_ptrs    = tt.addptr %b_base, %b_offsets
                    : tensor<16x32x!tt.ptr<f32>, #blocked>, tensor<16x32xi32, #blocked>

    %A = tt.load %a_ptrs : tensor<32x16x!tt.ptr<f32>, #blocked>
    %B = tt.load %b_ptrs : tensor<16x32x!tt.ptr<f32>, #blocked>


    %A_shared = ttg.local_alloc %A {allocation.offset = 0 : i32}
                    : (tensor<32x16xf32, #blocked>) -> !ttg.memdesc<32x16xf32, #shared, #smem>
    %B_shared = ttg.local_alloc %B {allocation.offset = 0 : i32}
                    : (tensor<16x32xf32, #blocked>) -> !ttg.memdesc<16x32xf32, #shared, #smem>

    // CHECK: ascend_dpx.sync_threads
    %A_dot = ttg.local_load %A_shared
                : !ttg.memdesc<32x16xf32, #shared, #smem> -> tensor<32x16xf32, #dot_a>
    %B_dot = ttg.local_load %B_shared
                : !ttg.memdesc<16x32xf32, #shared, #smem> -> tensor<16x32xf32, #dot_b>
    
    // CHECK: llvm.intr.fmuladd
    %C_init = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %D = tt.dot %A_dot, %B_dot, %C_init, inputPrecision = ieee
            : tensor<32x16xf32, #dot_a> * tensor<16x32xf32, #dot_b> -> tensor<32x32xf32, #blocked>

    tt.return
  }
}
