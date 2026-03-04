// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_vrelu_arith.maximumf_f16_
func.func @test_func_vrelu_arith.maximumf_f16_(%arg0:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.maximumf  %[[CST]], {{.*}} : tensor<6x6xf16>
    hivm.hir.vrelu ins(%arg0 : tensor<6x6xf16>) outs(%0:tensor<6x6xf16>)  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vrelu_arith.maximumf_f32_
func.func @test_func_vrelu_arith.maximumf_f32_(%arg0:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.maximumf  %[[CST]], {{.*}} : tensor<6x6xf32>
    hivm.hir.vrelu ins(%arg0 : tensor<6x6xf32>) outs(%0:tensor<6x6xf32>)  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vrelu_arith.maxsi_i32_
func.func @test_func_vrelu_arith.maxsi_i32_(%arg0:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[CST:.*]] = arith.constant dense<0{{.*}}> : tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.maxsi  %[[CST]], {{.*}} : tensor<6x6xi32>
    hivm.hir.vrelu ins(%arg0 : tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}