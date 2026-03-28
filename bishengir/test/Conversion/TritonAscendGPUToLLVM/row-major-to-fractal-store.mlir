// RUN: bishengir-opt --convert-triton-ascend-gpu-to-llvm %s | FileCheck %s
//
// Verify that a row-major global load followed by a store into fractal zN
// shared memory lowers correctly through the TritonAscendGPU-to-LLVM pipeline.
//
// The kernel loads a 32x32 f16 tile from global memory using row-major
// pointer arithmetic (offset[m][k] = m * stride + k) and writes it into
// a pre-allocated fractal zN memdesc via ttg.local_store.

#blocked  = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#fractal  = #ttg.fractal_shared<{fractalM0 = 16, fractalN0 = 16, layoutType = "zN"}>
#smem     = #ttg.shared_memory

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 4096 : i32,
    "ttg.enable-bishengir-simt-optimization" = 111 : i32
} {
  // CHECK-LABEL: @load_row_major_store_fractal_zN
  tt.func public @load_row_major_store_fractal_zN(
      %ptr : !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %smem_buf : !ttg.memdesc<32x32xf16, #fractal, #smem, mutable>) {

    // Row-major offset computation: offset[m][k] = m * 32 + k
    %m_range = tt.make_range {end = 32 : i32, start = 0 : i32}
                  : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %k_range = tt.make_range {end = 32 : i32, start = 0 : i32}
                  : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>

    %m_2d    = tt.expand_dims %m_range {axis = 1 : i32}
                  : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
                 -> tensor<32x1xi32, #blocked>
    %k_2d    = tt.expand_dims %k_range {axis = 0 : i32}
                  : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
                 -> tensor<1x32xi32, #blocked>

    %stride  = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %m_off   = arith.muli %m_2d, %stride : tensor<32x1xi32, #blocked>
    %m_broad = tt.broadcast %m_off : tensor<32x1xi32, #blocked>
                                   -> tensor<32x32xi32, #blocked>
    %k_broad = tt.broadcast %k_2d  : tensor<1x32xi32, #blocked>
                                   -> tensor<32x32xi32, #blocked>
    %offsets = arith.addi %m_broad, %k_broad : tensor<32x32xi32, #blocked>

    %base    = tt.splat %ptr : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptrs    = tt.addptr %base, %offsets
                  : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>

    // Load from global memory (row-major, #blocked encoding)
    %data = tt.load %ptrs : tensor<32x32x!tt.ptr<f16>, #blocked>

    // Store into fractal zN shared memory
    // CHECK: ascend_dpx.store {{.*}} : <3>, vector<4xi16>
    ttg.local_store %data, %smem_buf
                : tensor<32x32xf16, #blocked> -> !ttg.memdesc<32x32xf16, #fractal, #smem, mutable>

    tt.return
  }
}
