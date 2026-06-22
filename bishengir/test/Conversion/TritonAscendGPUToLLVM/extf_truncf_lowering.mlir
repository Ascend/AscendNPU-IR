// RUN: bishengir-opt -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.enable-bishengir-simt-optimization" = 900101 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @_kernel_extf_truncf(%arg0: tensor<32x64xbf16, #blocked>, %arg1: tensor<32x64xf16, #blocked>, %arg2: tensor<32x64xf32, #blocked>) attributes {noinline = false} {

        // CHECK: ascend_dpx.cast {{%[0-9]+}} kind <fp_to_fp> : vector<2xbf16> to vector<2xf32>
        %0 = arith.extf %arg0 : tensor<32x64xbf16, #blocked> to tensor<32x64xf32, #blocked>

        // CHECK: ascend_dpx.cast {{%[0-9]+}} kind <fp_to_fp> : vector<2xf16> to vector<2xf32>
        %1 = arith.extf %arg1 : tensor<32x64xf16, #blocked> to tensor<32x64xf32, #blocked>

        // CHECK: ascend_dpx.cast {{%[0-9]+}} kind <fp_to_fp> : vector<2xf32> to vector<2xbf16>
        %2 = arith.truncf %arg2 : tensor<32x64xf32, #blocked> to tensor<32x64xbf16, #blocked>

        // CHECK: ascend_dpx.cast {{%[0-9]+}} kind <fp_to_fp> : vector<2xf32> to vector<2xf16>
        %3 = arith.truncf %arg2 : tensor<32x64xf32, #blocked> to tensor<32x64xf16, #blocked>
        tt.return
    }
}