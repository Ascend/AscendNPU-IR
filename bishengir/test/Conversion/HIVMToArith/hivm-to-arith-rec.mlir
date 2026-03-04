// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_vrec_arith.divf_f16_
func.func @test_func_vrec_arith.divf_f16_(%arg0:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[CST:.*]] = arith.constant dense<1.{{.*}}> : tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.divf  %[[CST]], {{.*}} : tensor<6x6xf16>
    hivm.hir.vrec ins(%arg0 : tensor<6x6xf16>) outs(%0:tensor<6x6xf16>)  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vrec_arith.divf_f32_
func.func @test_func_vrec_arith.divf_f32_(%arg0:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[CST:.*]] = arith.constant dense<1.{{.*}}> : tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.divf  %[[CST]], {{.*}} : tensor<6x6xf32>
    hivm.hir.vrec ins(%arg0 : tensor<6x6xf32>) outs(%0:tensor<6x6xf32>)  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

