// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_vsel_arith.select_si16_i1_
func.func @test_func_vsel_arith.select_si16_i1_(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xsi16>, %arg2:tensor<6x6xsi16>) -> tensor<6x6xsi16> {
    %0 = tensor.empty():tensor<6x6xsi16>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<6x6xi1>, tensor<6x6xsi16>
    hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<6x6xi1>, tensor<6x6xsi16>, tensor<6x6xsi16>) outs(%0:tensor<6x6xsi16>)  -> tensor<6x6xsi16>
    //     CHECK: return %[[RET]] : tensor<6x6xsi16>
    return %0 : tensor<6x6xsi16>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_ui16_i1_
func.func @test_func_vsel_arith.select_ui16_i1_(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xui16>, %arg2:tensor<6x6xui16>) -> tensor<6x6xui16> {
    %0 = tensor.empty():tensor<6x6xui16>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<6x6xi1>, tensor<6x6xui16>
    hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<6x6xi1>, tensor<6x6xui16>, tensor<6x6xui16>) outs(%0:tensor<6x6xui16>)  -> tensor<6x6xui16>
    //     CHECK: return %[[RET]] : tensor<6x6xui16>
    return %0 : tensor<6x6xui16>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_f16_i1_
func.func @test_func_vsel_arith.select_f16_i1_(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xf16>, %arg2:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<6x6xi1>, tensor<6x6xf16>
    hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<6x6xi1>, tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xf16>)  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_si32_i1_
func.func @test_func_vsel_arith.select_si32_i1_(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xsi32>, %arg2:tensor<6x6xsi32>) -> tensor<6x6xsi32> {
    %0 = tensor.empty():tensor<6x6xsi32>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<6x6xi1>, tensor<6x6xsi32>
    hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<6x6xi1>, tensor<6x6xsi32>, tensor<6x6xsi32>) outs(%0:tensor<6x6xsi32>)  -> tensor<6x6xsi32>
    //     CHECK: return %[[RET]] : tensor<6x6xsi32>
    return %0 : tensor<6x6xsi32>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_ui32_i1_
func.func @test_func_vsel_arith.select_ui32_i1_(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xui32>, %arg2:tensor<6x6xui32>) -> tensor<6x6xui32> {
    %0 = tensor.empty():tensor<6x6xui32>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<6x6xi1>, tensor<6x6xui32>
    hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<6x6xi1>, tensor<6x6xui32>, tensor<6x6xui32>) outs(%0:tensor<6x6xui32>)  -> tensor<6x6xui32>
    //     CHECK: return %[[RET]] : tensor<6x6xui32>
    return %0 : tensor<6x6xui32>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_f32_i1_
func.func @test_func_vsel_arith.select_f32_i1_(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xf32>, %arg2:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<6x6xi1>, tensor<6x6xf32>
    hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<6x6xi1>, tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xf32>)  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_i64_i1_
func.func @test_func_vsel_arith.select_i64_i1_(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xi64>, %arg2:tensor<6x6xi64>) -> tensor<6x6xi64> {
    %0 = tensor.empty():tensor<6x6xi64>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<6x6xi1>, tensor<6x6xi64>
    hivm.hir.vsel ins(%arg0, %arg1, %arg2 : tensor<6x6xi1>, tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi64>)  -> tensor<6x6xi64>
    //     CHECK: return %[[RET]] : tensor<6x6xi64>
    return %0 : tensor<6x6xi64>
}

