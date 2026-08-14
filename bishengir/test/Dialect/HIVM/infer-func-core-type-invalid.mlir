// RUN: bishengir-opt %s -hivm-infer-func-core-type -split-input-file -verify-diagnostics

module {
  // expected-error@+1 {{hivm.core_ratio is only valid on a MIX function, but hivm.func_core_type is AIV}}
  func.func @ratio_on_aiv_func(%base : memref<?xf32>,
                               %indices : tensor<8xi64>,
                               %dst : tensor<8xf32>) -> tensor<8xf32>
      attributes {hivm.core_ratio = #hivm.core_ratio<0, 1>,
                  hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.gather_load ins(%base : memref<?xf32>,
                                  %indices : tensor<8xi64>,
                                  %c1_i32 : i32)
         outs(%dst : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

module {
  // expected-error@+1 {{Invalid core ratio: 7:9, expected one of 0:1, 1:0, 1:1, 1:2}}
  func.func @legacy_array_ratio_out_of_range(%base : memref<?xf32>,
                                             %indices : tensor<8xi64>,
                                             %dst : tensor<8xf32>) -> tensor<8xf32>
      attributes {hivm.core_ratio = array<i32: 7, 9>} {
    %c1_i32 = arith.constant 1 : i32
    %0 = hivm.hir.gather_load ins(%base : memref<?xf32>,
                                  %indices : tensor<8xi64>,
                                  %c1_i32 : i32)
         outs(%dst : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
