// RUN: bishengir-opt %s -convert-dot-input-to-linear-layout | FileCheck %s

// CHECK-LABEL: @test_dot_to_linear
// Test that dot inputs are converted to FMA-friendly linear layouts.
// After pass:
// - Inputs A and B are converted from #blocked to #linear encoding
// - No shared memory conversions (load optimization worked)
// - The dot op has {fma.converted} attribute

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_op_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, "ttg.enable-bishengir-simt-optimization" = 11 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 32768 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_dot_to_linear(
    %A: tensor<2x32xf16, #blocked>,
    %B: tensor<32x32xf16, #blocked>,
    %C: tensor<2x32xf32, #blocked>
  ) -> tensor<2x32xf32, #blocked> {
    // Convert blocked to dot_op encoding (original IR pattern)
    %A_dot = ttg.convert_layout %A : tensor<2x32xf16, #blocked> -> tensor<2x32xf16, #dot_op_a>
    %B_dot = ttg.convert_layout %B : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #dot_op_b>

    // After pass: inputs converted to linear layout, dot has fma.converted
    // CHECK: ttg.convert_layout %{{.*}} : tensor<2x32xf16, #blocked> -> tensor<2x32xf16, #linear>
    // CHECK: ttg.convert_layout %{{.*}} : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #linear1>
    // CHECK: tt.dot {{.*}} {dot.propagated, fma.converted}
    %D = tt.dot %A_dot, %B_dot, %C : tensor<2x32xf16, #dot_op_a> * tensor<32x32xf16, #dot_op_b> -> tensor<2x32xf32, #blocked>

    tt.return %D : tensor<2x32xf32, #blocked>
  }
}