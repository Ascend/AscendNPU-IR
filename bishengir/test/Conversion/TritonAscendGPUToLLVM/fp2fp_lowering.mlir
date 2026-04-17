// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 900101 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @_kernel_fp2fp(%arg0: tensor<32x64xbf16, #blocked>, %arg1: tensor<32x64xf8E4M3FN, #blocked>) attributes {noinline = false} {

        // CHECK: ascend_dpx.cast {{%[0-9]+}} kind <fp_to_fp> : vector<2xbf16> to vector<2xf8E5M2>
        %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<32x64xbf16, #blocked> -> tensor<32x64xf8E5M2, #blocked>

        // CHECK: ascend_dpx.cast {{%[0-9]+}} kind <fp_to_fp> : vector<2xf8E4M3FN> to vector<2xf32>
        %1 = tt.fp_to_fp %arg1, rounding = rtne : tensor<32x64xf8E4M3FN, #blocked> -> tensor<32x64xf32, #blocked>
        tt.return
    }
}