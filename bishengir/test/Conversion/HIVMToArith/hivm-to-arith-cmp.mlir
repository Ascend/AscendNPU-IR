// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_ne
func.func @test_func_vcmp_arith.cmpf_f16_ne(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf one, {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ne> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_ne
func.func @test_func_vcmp_arith.cmpf_f32_ne(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf one, {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ne> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_ne
func.func @test_func_vcmp_arith.cmpi_i16_ne(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi ne, {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ne> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_ne
func.func @test_func_vcmp_arith.cmpi_i32_ne(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi ne, {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ne> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_ne
func.func @test_func_vcmp_arith.cmpi_i64_ne(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi ne, {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ne> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_eq
func.func @test_func_vcmp_arith.cmpf_f16_eq(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf oeq, {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <eq> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_eq
func.func @test_func_vcmp_arith.cmpf_f32_eq(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf oeq, {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <eq> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_eq
func.func @test_func_vcmp_arith.cmpi_i16_eq(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi eq, {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <eq> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_eq
func.func @test_func_vcmp_arith.cmpi_i32_eq(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi eq, {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <eq> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_eq
func.func @test_func_vcmp_arith.cmpi_i64_eq(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi eq, {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi1>) compare_mode = <eq> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_gt
func.func @test_func_vcmp_arith.cmpf_f16_gt(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf ogt, {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <gt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_gt
func.func @test_func_vcmp_arith.cmpf_f32_gt(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf ogt, {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <gt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_gt
func.func @test_func_vcmp_arith.cmpi_i16_gt(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sgt, {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <gt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_gt
func.func @test_func_vcmp_arith.cmpi_i32_gt(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sgt, {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <gt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_gt
func.func @test_func_vcmp_arith.cmpi_i64_gt(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sgt, {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi1>) compare_mode = <gt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_lt
func.func @test_func_vcmp_arith.cmpf_f16_lt(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf olt, {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <lt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_lt
func.func @test_func_vcmp_arith.cmpf_f32_lt(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf olt, {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <lt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_lt
func.func @test_func_vcmp_arith.cmpi_i16_lt(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi slt, {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <lt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_lt
func.func @test_func_vcmp_arith.cmpi_i32_lt(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi slt, {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <lt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_lt
func.func @test_func_vcmp_arith.cmpi_i64_lt(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi slt, {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi1>) compare_mode = <lt> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i8_ult
func.func @test_func_vcmp_arith.cmpi_i8_ult(%arg0:tensor<4x64x32xi8>,%arg1:tensor<4x64x32xi8>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi ult, {{.*}}, {{.*}} : tensor<4x64x32xi8>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi8>, tensor<4x64x32xi8>) outs(%0:tensor<4x64x32xi1>) compare_mode = <lt> is_signed = false -> tensor<4x64x32xi1>

    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_ge
func.func @test_func_vcmp_arith.cmpf_f16_ge(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf oge, {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ge> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_ge
func.func @test_func_vcmp_arith.cmpf_f32_ge(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf oge, {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ge> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_ge
func.func @test_func_vcmp_arith.cmpi_i16_ge(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sge, {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ge> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_ge
func.func @test_func_vcmp_arith.cmpi_i32_ge(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sge, {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ge> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_ge
func.func @test_func_vcmp_arith.cmpi_i64_ge(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sge, {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi1>) compare_mode = <ge> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_le
func.func @test_func_vcmp_arith.cmpf_f16_le(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf ole, {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <le> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_le
func.func @test_func_vcmp_arith.cmpf_f32_le(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf ole, {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <le> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_le
func.func @test_func_vcmp_arith.cmpi_i16_le(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sle, {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi1>) compare_mode = <le> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_le
func.func @test_func_vcmp_arith.cmpi_i32_le(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sle, {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi1>) compare_mode = <le> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_le
func.func @test_func_vcmp_arith.cmpi_i64_le(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sle, {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vcmp  ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi1>) compare_mode = <le> -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}