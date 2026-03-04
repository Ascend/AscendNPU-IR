// RUN: bishengir-opt %s  | FileCheck %s

// -----
module {
  func.func @matmul_gelu(%arg0: tensor<128x4096xf16>,
                         %arg1: tensor<4096x4096xf16>,
                         %arg2: tensor<128x4096xf16>,
                         %arg3: memref<16xi64>) -> tensor<128x4096xf16>
                          {
    %0 = hivm.hir.matmul ins(%arg0, %arg1 : tensor<128x4096xf16>, tensor<4096x4096xf16>)
                         outs(%arg2 : tensor<128x4096xf16>) -> tensor<128x4096xf16>
    // CHECK: annotation.mark
    annotation.mark %0 keys = ["tiling_params"] values = [%arg3: memref<16xi64>] : tensor<128x4096xf16>
    return %0 : tensor<128x4096xf16>
  }
}
