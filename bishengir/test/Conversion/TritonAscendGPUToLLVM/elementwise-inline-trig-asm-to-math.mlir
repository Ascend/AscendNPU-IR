// RUN: bishengir-opt --convert-triton-ascend-gpu-to-llvm %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>

module attributes {"ttg.enable-bishengir-simt-optimization" = 900101 : i32, "ttg.num-warps" = 16 : i32} {
    // CHECK-LABEL: sinTest
    tt.func @sinTest(%arg0: tensor<64xf32, #blocked>) {
        // CHECK: %{{.*}} = ascend_dpx.sin
        %0 = tt.elementwise_inline_asm "sin.approx.f32 $0, $1;" {constraints = "=f, f", packed_element = 1 : i32, pure = true} %arg0 : tensor<64xf32, #blocked> -> tensor<64xf32, #blocked>
        tt.return

        // CHECK-NOT: %{{.*}} = tt.elementwise_inline_asm
    }

    // CHECK-LABEL: cosTest
    tt.func @cosTest(%arg0: tensor<64xf32, #blocked>) {
        // CHECK: %{{.*}} = ascend_dpx.cos
        %0 = tt.elementwise_inline_asm "cos.approx.f32 $0, $1;" {constraints = "=f, f", packed_element = 1 : i32, pure = true} %arg0 : tensor<64xf32, #blocked> -> tensor<64xf32, #blocked>
        tt.return

        // CHECK-NOT: %{{.*}} = tt.elementwise_inline_asm
    }

    // CHECK-LABEL: atanTest
    tt.func @atanTest(%arg0: tensor<64xf32, #blocked>) {
        // CHECK: %{{.*}} = ascend_dpx.atan
        %0 = tt.elementwise_inline_asm "atan.approx.f32 $0, $1;" {constraints = "=f, f", packed_element = 1 : i32, pure = true} %arg0 : tensor<64xf32, #blocked> -> tensor<64xf32, #blocked>
        tt.return

        // CHECK-NOT: %{{.*}} = tt.elementwise_inline_asm
    }

    // CHECK-LABEL: tanhTest
    tt.func @tanhTest(%arg0: tensor<64xf32, #blocked>) {
        // CHECK: %{{.*}} = ascend_dpx.tanh
        %0 = tt.elementwise_inline_asm "tanh.approx.f32 $0, $1;" {constraints = "=f, f", packed_element = 1 : i32, pure = true} %arg0 : tensor<64xf32, #blocked> -> tensor<64xf32, #blocked>
        tt.return

        // CHECK-NOT: %{{.*}} = tt.elementwise_inline_asm
    }
}