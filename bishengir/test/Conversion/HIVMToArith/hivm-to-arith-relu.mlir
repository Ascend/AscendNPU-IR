// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_brc_vrelu_arith.maximumf_f16
func.func @test_func_brc_vrelu_arith.maximumf_f16(%arg0:tensor<8x1x8xf16>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.maximumf %[[CST]], %[[brc1]] : tensor<8x8x8xf16>
    hivm.hir.vrelu ins(%arg0 : tensor<8x1x8xf16>) outs(%0:tensor<8x8x8xf16>) broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vrelu_arith.maximumf_f32
func.func @test_func_brc_vrelu_arith.maximumf_f32(%arg0:tensor<8x1x8xf32>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.maximumf %[[CST]], %[[brc1]] : tensor<8x8x8xf32>
    hivm.hir.vrelu ins(%arg0 : tensor<8x1x8xf32>) outs(%0:tensor<8x8x8xf32>) broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vrelu_arith.maxsi_i32
func.func @test_func_brc_vrelu_arith.maxsi_i32(%arg0:tensor<8x1x8xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.maxsi %[[CST]], %[[brc1]] : tensor<8x8x8xi32>
    hivm.hir.vrelu ins(%arg0 : tensor<8x1x8xi32>) outs(%0:tensor<8x8x8xi32>) broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vrelu_arith.maximumf_f16
func.func @test_func_vrelu_arith.maximumf_f16(%arg0:tensor<4x64x32xf16>) -> tensor<4x64x32xf16> {
    %0 = tensor.empty():tensor<4x64x32xf16>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<4x64x32xf16>
    //     CHECK: %[[RET:.*]] = arith.maximumf %[[CST]], {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vrelu ins(%arg0 : tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xf16>)  -> tensor<4x64x32xf16>
    return %0 : tensor<4x64x32xf16>
}

// CHECK-LABEL: func.func @test_func_vrelu_arith.maximumf_f32
func.func @test_func_vrelu_arith.maximumf_f32(%arg0:tensor<4x64x32xf32>) -> tensor<4x64x32xf32> {
    %0 = tensor.empty():tensor<4x64x32xf32>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<4x64x32xf32>
    //     CHECK: %[[RET:.*]] = arith.maximumf %[[CST]], {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vrelu ins(%arg0 : tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xf32>)  -> tensor<4x64x32xf32>
    return %0 : tensor<4x64x32xf32>
}

// CHECK-LABEL: func.func @test_func_vrelu_arith.maxsi_i32
func.func @test_func_vrelu_arith.maxsi_i32(%arg0:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.maxsi %[[CST]], {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vrelu ins(%arg0 : tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>)  -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}