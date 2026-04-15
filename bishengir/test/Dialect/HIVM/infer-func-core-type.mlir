// RUN: bishengir-opt %s -hivm-infer-func-core-type | FileCheck %s

module {
  // CHECK-LABEL: func.func @gather_only(
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
  func.func @gather_only(%base : memref<?xf32>,
                         %indices : tensor<8xi64>,
                         %dst : tensor<8xf32>) -> tensor<8xf32> {
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.gather_load ins(%base : memref<?xf32>,
                                  %indices : tensor<8xi64>,
                                  %c1_i32 : i32)
         outs(%dst : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  // CHECK-LABEL: func.func @matmul_gather_mix(
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  func.func @matmul_gather_mix(%base : memref<?xf32>,
                               %indices : tensor<8xi64>,
                               %gather_dst : tensor<8xf32>,
                               %lhs : tensor<8x8xf16>,
                               %rhs : tensor<8x8xf16>,
                               %matmul_dst : tensor<8x8xf16>)
      -> (tensor<8xf32>, tensor<8x8xf16>) {
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.gather_load ins(%base : memref<?xf32>,
                                  %indices : tensor<8xi64>,
                                  %c1_i32 : i32)
         outs(%gather_dst : tensor<8xf32>) -> tensor<8xf32>
    %1 = hivm.hir.matmul
         ins(%lhs, %rhs : tensor<8x8xf16>, tensor<8x8xf16>)
         outs(%matmul_dst : tensor<8x8xf16>) -> tensor<8x8xf16>
    return %0, %1 : tensor<8xf32>, tensor<8x8xf16>
  }

  // CHECK-LABEL: func.func @matmul_scatter_mix(
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
  func.func @matmul_scatter_mix(%base : memref<?xf32>,
                                %indices : tensor<8xi64>,
                                %data : tensor<8xf32>,
                                %lhs : tensor<8x8xf16>,
                                %rhs : tensor<8x8xf16>,
                                %matmul_dst : tensor<8x8xf16>)
      -> tensor<8x8xf16> {
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.scatter_store ins(%indices : tensor<8xi64>,
                               %data : tensor<8xf32>,
                               %c1_i32 : i32)
                           outs(%base : memref<?xf32>)
    %0 = hivm.hir.matmul
         ins(%lhs, %rhs : tensor<8x8xf16>, tensor<8x8xf16>)
         outs(%matmul_dst : tensor<8x8xf16>) -> tensor<8x8xf16>
    return %0 : tensor<8x8xf16>
  }
}
