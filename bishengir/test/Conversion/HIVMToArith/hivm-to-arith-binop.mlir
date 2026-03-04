// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_vand_i1
func.func @test_func_vand_i1(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xi1>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<6x6xi1>
    hivm.hir.vand ins(%arg0, %arg1 : tensor<6x6xi1>, tensor<6x6xi1>) outs(%0:tensor<6x6xi1>)  -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vand_i8
func.func @test_func_vand_i8(%arg0:tensor<6x6xi8>, %arg1:tensor<6x6xi8>) -> tensor<6x6xi8> {
    %0 = tensor.empty():tensor<6x6xi8>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<6x6xi8>
    hivm.hir.vand ins(%arg0, %arg1 : tensor<6x6xi8>, tensor<6x6xi8>) outs(%0:tensor<6x6xi8>)  -> tensor<6x6xi8>
    //     CHECK: return %[[RET]] : tensor<6x6xi8>
    return %0 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_vand_i16
func.func @test_func_vand_i16(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vand ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>)  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vand_i32
func.func @test_func_vand_i32(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vand ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vand_i64
func.func @test_func_vand_i64(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi64> {
    %0 = tensor.empty():tensor<6x6xi64>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vand ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi64>)  -> tensor<6x6xi64>
    //     CHECK: return %[[RET]] : tensor<6x6xi64>
    return %0 : tensor<6x6xi64>
}

// CHECK-LABEL: func.func @test_func_vor_i1
func.func @test_func_vor_i1(%arg0:tensor<6x6xi1>, %arg1:tensor<6x6xi1>) -> tensor<6x6xi1> {
    %0 = tensor.empty():tensor<6x6xi1>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<6x6xi1>
    hivm.hir.vor ins(%arg0, %arg1 : tensor<6x6xi1>, tensor<6x6xi1>) outs(%0:tensor<6x6xi1>)  -> tensor<6x6xi1>
    //     CHECK: return %[[RET]] : tensor<6x6xi1>
    return %0 : tensor<6x6xi1>
}

// CHECK-LABEL: func.func @test_func_vor_i8
func.func @test_func_vor_i8(%arg0:tensor<6x6xi8>, %arg1:tensor<6x6xi8>) -> tensor<6x6xi8> {
    %0 = tensor.empty():tensor<6x6xi8>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<6x6xi8>
    hivm.hir.vor ins(%arg0, %arg1 : tensor<6x6xi8>, tensor<6x6xi8>) outs(%0:tensor<6x6xi8>)  -> tensor<6x6xi8>
    //     CHECK: return %[[RET]] : tensor<6x6xi8>
    return %0 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_vor_i16
func.func @test_func_vor_i16(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vor ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>)  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vor_i32
func.func @test_func_vor_i32(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vor ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vor_i64
func.func @test_func_vor_i64(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi64> {
    %0 = tensor.empty():tensor<6x6xi64>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vor ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi64>)  -> tensor<6x6xi64>
    //     CHECK: return %[[RET]] : tensor<6x6xi64>
    return %0 : tensor<6x6xi64>
}

// CHECK-LABEL: func.func @test_func_vadd_i16
func.func @test_func_vadd_i16(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.addi {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vadd ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>)  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vadd_i32
func.func @test_func_vadd_i32(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.addi {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vadd ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vadd_f16
func.func @test_func_vadd_f16(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.addf {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vadd ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xf16>)  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vadd_f32
func.func @test_func_vadd_f32(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.addf {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vadd ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xf32>)  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vadd_i64
func.func @test_func_vadd_i64(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi64> {
    %0 = tensor.empty():tensor<6x6xi64>
    //     CHECK: %[[RET:.*]] = arith.addi {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vadd ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi64>)  -> tensor<6x6xi64>
    //     CHECK: return %[[RET]] : tensor<6x6xi64>
    return %0 : tensor<6x6xi64>
}

// CHECK-LABEL: func.func @test_func_vsub_i16
func.func @test_func_vsub_i16(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.subi {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vsub ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>)  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vsub_i32
func.func @test_func_vsub_i32(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.subi {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vsub ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vsub_f16
func.func @test_func_vsub_f16(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.subf {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vsub ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xf16>)  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vsub_f32
func.func @test_func_vsub_f32(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.subf {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vsub ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xf32>)  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vsub_i64
func.func @test_func_vsub_i64(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi64> {
    %0 = tensor.empty():tensor<6x6xi64>
    //     CHECK: %[[RET:.*]] = arith.subi {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vsub ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi64>)  -> tensor<6x6xi64>
    //     CHECK: return %[[RET]] : tensor<6x6xi64>
    return %0 : tensor<6x6xi64>
}

// CHECK-LABEL: func.func @test_func_vmul_i16
func.func @test_func_vmul_i16(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.muli {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vmul ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>)  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vmul_i32
func.func @test_func_vmul_i32(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.muli {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vmul ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vmul_f16
func.func @test_func_vmul_f16(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.mulf {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vmul ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xf16>)  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vmul_f32
func.func @test_func_vmul_f32(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.mulf {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vmul ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xf32>)  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vmul_i64
func.func @test_func_vmul_i64(%arg0:tensor<6x6xi64>, %arg1:tensor<6x6xi64>) -> tensor<6x6xi64> {
    %0 = tensor.empty():tensor<6x6xi64>
    //     CHECK: %[[RET:.*]] = arith.muli {{.*}}, {{.*}} : tensor<6x6xi64>
    hivm.hir.vmul ins(%arg0, %arg1 : tensor<6x6xi64>, tensor<6x6xi64>) outs(%0:tensor<6x6xi64>)  -> tensor<6x6xi64>
    //     CHECK: return %[[RET]] : tensor<6x6xi64>
    return %0 : tensor<6x6xi64>
}

// CHECK-LABEL: func.func @test_func_vdiv_f16
func.func @test_func_vdiv_f16(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.divf {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vdiv ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xf16>) isSigned = true -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vdiv_f32
func.func @test_func_vdiv_f32(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.divf {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vdiv ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xf32>) isSigned = true -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vdiv_i16
func.func @test_func_vdiv_i16(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.divsi {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vdiv ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>) isSigned = true -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vdiv_i32
func.func @test_func_vdiv_i32(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.divsi {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vdiv ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>) isSigned = true -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vdiv_f16_signed
func.func @test_func_vdiv_f16_signed(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.divf {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vdiv ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xf16>) isSigned = false -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vdiv_f32_signed
func.func @test_func_vdiv_f32_signed(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.divf {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vdiv ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xf32>) isSigned = false -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vdiv_i16_signed
func.func @test_func_vdiv_i16_signed(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.divui {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vdiv ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>) isSigned = false -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vdiv_i32_signed
func.func @test_func_vdiv_i32_signed(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.divui {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vdiv ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>) isSigned = false -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vmin_f16
func.func @test_func_vmin_f16(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.minimumf {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vmin ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xf16>)  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vmin_f32
func.func @test_func_vmin_f32(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.minimumf {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vmin ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xf32>)  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vmax_f16
func.func @test_func_vmax_f16(%arg0:tensor<6x6xf16>, %arg1:tensor<6x6xf16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.maximumf {{.*}}, {{.*}} : tensor<6x6xf16>
    hivm.hir.vmax ins(%arg0, %arg1 : tensor<6x6xf16>, tensor<6x6xf16>) outs(%0:tensor<6x6xf16>)  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vmax_f32
func.func @test_func_vmax_f32(%arg0:tensor<6x6xf32>, %arg1:tensor<6x6xf32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.maximumf {{.*}}, {{.*}} : tensor<6x6xf32>
    hivm.hir.vmax ins(%arg0, %arg1 : tensor<6x6xf32>, tensor<6x6xf32>) outs(%0:tensor<6x6xf32>)  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vmodui_arith.remui_i16_
func.func @test_func_vmodui_arith.remui_i16_(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.remui {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vmodui ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>)  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vmodui_arith.remui_i32_
func.func @test_func_vmodui_arith.remui_i32_(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.remui {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vmodui ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vmin_i16
func.func @test_func_vmin_i16(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.minsi {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vmin ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>)  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vmin_i32
func.func @test_func_vmin_i32(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.minsi {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vmin ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vmax_i16
func.func @test_func_vmax_i16(%arg0:tensor<6x6xi16>, %arg1:tensor<6x6xi16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.maxsi {{.*}}, {{.*}} : tensor<6x6xi16>
    hivm.hir.vmax ins(%arg0, %arg1 : tensor<6x6xi16>, tensor<6x6xi16>) outs(%0:tensor<6x6xi16>)  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vmax_i32
func.func @test_func_vmax_i32(%arg0:tensor<6x6xi32>, %arg1:tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.maxsi {{.*}}, {{.*}} : tensor<6x6xi32>
    hivm.hir.vmax ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0:tensor<6x6xi32>)  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vmulext_high
func.func @test_func_vmulext_high(%arg0: tensor<6xi32>, %arg1: tensor<6xi32>) -> tensor<6xi32> {
    %0 = tensor.empty() : tensor<6xi32>
    %1 = tensor.empty() : tensor<6xi32>
    //     CHECK: %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended {{.*}}, {{.*}} : tensor<6xi32>
    hivm.hir.vmulext ins(%arg0, %arg1 : tensor<6xi32>, tensor<6xi32>) outs(%0, %1 : tensor<6xi32>, tensor<6xi32>) -> tensor<6xi32>, tensor<6xi32>
    //     CHECK: return %[[HIGH]] : tensor<6xi32>
    return %1: tensor<6xi32>
}

// CHECK-LABEL: func.func @test_func_vmulext_low
func.func @test_func_vmulext_low(%arg0: tensor<6xi32>, %arg1: tensor<6xi32>) -> tensor<6xi32> {
    %0 = tensor.empty() : tensor<6xi32>
    %1 = tensor.empty() : tensor<6xi32>
    //     CHECK: %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended {{.*}}, {{.*}} : tensor<6xi32>
    hivm.hir.vmulext ins(%arg0, %arg1 : tensor<6xi32>, tensor<6xi32>) outs(%0, %1 : tensor<6xi32>, tensor<6xi32>) -> tensor<6xi32>, tensor<6xi32>
    //     CHECK: return %[[LOW]] : tensor<6xi32>
    return %0: tensor<6xi32>
}

func.func @test_func_vmulext_return_low(%arg0: tensor<6xi32>, %arg1: tensor<6xi32>) -> tensor<6xi32> {
    %0 = tensor.empty() : tensor<6xi32>
    %1 = tensor.empty() : tensor<6xi32>
    //     CHECK: %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended {{.*}}, {{.*}} : tensor<6xi32>
    %2:2 = hivm.hir.vmulext ins(%arg0, %arg1 : tensor<6xi32>, tensor<6xi32>) outs(%0, %1 : tensor<6xi32>, tensor<6xi32>) -> tensor<6xi32>, tensor<6xi32>
    //     CHECK: return %[[LOW]] : tensor<6xi32>
    return %2#0 : tensor<6xi32>
}

func.func @test_func_vmulext_return_high(%arg0: tensor<6xi32>, %arg1: tensor<6xi32>) -> tensor<6xi32> {
    %0 = tensor.empty() : tensor<6xi32>
    %1 = tensor.empty() : tensor<6xi32>
    //     CHECK: %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended {{.*}}, {{.*}} : tensor<6xi32>
    %2:2 = hivm.hir.vmulext ins(%arg0, %arg1 : tensor<6xi32>, tensor<6xi32>) outs(%0, %1 : tensor<6xi32>, tensor<6xi32>) -> tensor<6xi32>, tensor<6xi32>
    //     CHECK: return %[[HIGH]] : tensor<6xi32>
    return %2#1 : tensor<6xi32>
}