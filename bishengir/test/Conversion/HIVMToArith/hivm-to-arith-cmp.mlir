// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_i1_one
func.func @test_func_vcmp_arith.cmpf_f16_i1_one(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf one, {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xi1>) compare_mode = <ne> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_i1_one
func.func @test_func_vcmp_arith.cmpf_f32_i1_one(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf one, {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xi1>) compare_mode = <ne> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_i1_ne
func.func @test_func_vcmp_arith.cmpi_i16_i1_ne(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi ne, {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi1>) compare_mode = <ne> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_i1_ne
func.func @test_func_vcmp_arith.cmpi_i32_i1_ne(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi ne, {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi1>) compare_mode = <ne> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_i1_ne
func.func @test_func_vcmp_arith.cmpi_i64_i1_ne(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi ne, {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi1>) compare_mode = <ne> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_i1_oeq
func.func @test_func_vcmp_arith.cmpf_f16_i1_oeq(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf oeq, {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xi1>) compare_mode = <eq> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_i1_oeq
func.func @test_func_vcmp_arith.cmpf_f32_i1_oeq(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf oeq, {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xi1>) compare_mode = <eq> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_i1_eq
func.func @test_func_vcmp_arith.cmpi_i16_i1_eq(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi eq, {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi1>) compare_mode = <eq> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_i1_eq
func.func @test_func_vcmp_arith.cmpi_i32_i1_eq(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi eq, {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi1>) compare_mode = <eq> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_i1_eq
func.func @test_func_vcmp_arith.cmpi_i64_i1_eq(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi eq, {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi1>) compare_mode = <eq> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_i1_ogt
func.func @test_func_vcmp_arith.cmpf_f16_i1_ogt(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf ogt, {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xi1>) compare_mode = <gt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_i1_ogt
func.func @test_func_vcmp_arith.cmpf_f32_i1_ogt(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf ogt, {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xi1>) compare_mode = <gt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_i1_sgt
func.func @test_func_vcmp_arith.cmpi_i16_i1_sgt(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sgt, {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi1>) compare_mode = <gt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_i1_sgt
func.func @test_func_vcmp_arith.cmpi_i32_i1_sgt(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sgt, {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi1>) compare_mode = <gt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_i1_sgt
func.func @test_func_vcmp_arith.cmpi_i64_i1_sgt(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sgt, {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi1>) compare_mode = <gt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_i1_olt
func.func @test_func_vcmp_arith.cmpf_f16_i1_olt(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf olt, {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xi1>) compare_mode = <lt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_i1_olt
func.func @test_func_vcmp_arith.cmpf_f32_i1_olt(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf olt, {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xi1>) compare_mode = <lt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_i1_slt
func.func @test_func_vcmp_arith.cmpi_i16_i1_slt(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi slt, {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi1>) compare_mode = <lt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_i1_slt
func.func @test_func_vcmp_arith.cmpi_i32_i1_slt(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi slt, {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi1>) compare_mode = <lt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_i1_slt
func.func @test_func_vcmp_arith.cmpi_i64_i1_slt(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi slt, {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi1>) compare_mode = <lt> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_i1_oge
func.func @test_func_vcmp_arith.cmpf_f16_i1_oge(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf oge, {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xi1>) compare_mode = <ge> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_i1_oge
func.func @test_func_vcmp_arith.cmpf_f32_i1_oge(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf oge, {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xi1>) compare_mode = <ge> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_i1_sge
func.func @test_func_vcmp_arith.cmpi_i16_i1_sge(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sge, {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi1>) compare_mode = <ge> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_i1_sge
func.func @test_func_vcmp_arith.cmpi_i32_i1_sge(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sge, {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi1>) compare_mode = <ge> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_i1_sge
func.func @test_func_vcmp_arith.cmpi_i64_i1_sge(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sge, {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi1>) compare_mode = <ge> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f16_i1_ole
func.func @test_func_vcmp_arith.cmpf_f16_i1_ole(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf ole, {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xi1>) compare_mode = <le> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpf_f32_i1_ole
func.func @test_func_vcmp_arith.cmpf_f32_i1_ole(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpf ole, {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xi1>) compare_mode = <le> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i16_i1_sle
func.func @test_func_vcmp_arith.cmpi_i16_i1_sle(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sle, {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi1>) compare_mode = <le> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i32_i1_sle
func.func @test_func_vcmp_arith.cmpi_i32_i1_sle(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sle, {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi1>) compare_mode = <le> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vcmp_arith.cmpi_i64_i1_sle
func.func @test_func_vcmp_arith.cmpi_i64_i1_sle(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.cmpi sle, {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vcmp ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi1>) compare_mode = <le> -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

