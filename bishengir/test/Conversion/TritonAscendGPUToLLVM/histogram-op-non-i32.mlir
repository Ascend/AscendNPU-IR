// REQUIRES: bspub_davinci_bishengir
// RUN: bishengir-opt -allocate-shared-memory -convert-triton-ascend-gpu-to-llvm %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {
    "ttg.num-ctas"         = 1 : i32,
    "ttg.num-warps"        = 4 : i32,
    "ttg.threads-per-warp" = 32 : i32,
    ttg.shared             = 0 : i32,
    ttg.target             = "cuda:80",
    hacc.target            = #hacc.target<"Ascend910_9589">,
    "ttg.enable-bishengir-simt-optimization" = 111 : i32
} {
  // CHECK-LABEL: @histogram_non_i32
  tt.func public @histogram_non_i32(
      %arg0 : tensor<256xi8, #blocked>,
      %arg1 : tensor<256xi16, #blocked>,
      %arg2 : tensor<256xi64, #blocked>,
      %mask : tensor<256xi1, #blocked>) {
    // CHECK-DAG: llvm.sext %{{.*}} : i8 to i32
    // CHECK-DAG: llvm.sext %{{.*}} : i16 to i32
    // CHECK-DAG: llvm.trunc %{{.*}} : i64 to i32
    // CHECK-DAG: llvm.icmp "sge"
    // CHECK-DAG: llvm.icmp "slt"
    // CHECK-DAG: llvm.select
    // CHECK-DAG: ascend_dpx.atomic_add
    %0 = tt.histogram %arg0 : tensor<256xi8, #blocked> -> tensor<8xi32, #blocked>
    %1 = tt.histogram %arg1, %mask : tensor<256xi16, #blocked> -> tensor<8xi32, #blocked>
    %2 = tt.histogram %arg2 : tensor<256xi64, #blocked> -> tensor<8xi32, #blocked>
    tt.return
  }
}
