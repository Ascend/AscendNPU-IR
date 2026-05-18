// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_brc_vand_arith.andi_i1
func.func @test_func_brc_vand_arith.andi_i1(%arg0:tensor<8x1x8xi1>,%arg1:tensor<8x8x1xi1>) -> tensor<8x8x8xi1> {
    %0 = tensor.empty():tensor<8x8x8xi1>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.andi %[[brc1]], %[[brc2]] : tensor<8x8x8xi1>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<8x1x8xi1>, tensor<8x8x1xi1>) outs(%0:tensor<8x8x8xi1>) broadcast = [0,1,2] -> tensor<8x8x8xi1>
    return %0 : tensor<8x8x8xi1>
}

// CHECK-LABEL: func.func @test_func_vand_arith.andi_i1
func.func @test_func_vand_arith.andi_i1(%arg0:tensor<4x64x32xi1>,%arg1:tensor<4x64x32xi1>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<4x64x32xi1>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<4x64x32xi1>, tensor<4x64x32xi1>) outs(%0:tensor<4x64x32xi1>)  -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_brc_vand_arith.andi_i8
func.func @test_func_brc_vand_arith.andi_i8(%arg0:tensor<8x1x8xi8>,%arg1:tensor<8x8x1xi8>) -> tensor<8x8x8xi8> {
    %0 = tensor.empty():tensor<8x8x8xi8>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi8> -> tensor<8x8x8xi8>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi8> -> tensor<8x8x8xi8>
    //     CHECK: %[[RET:.*]] = arith.andi %[[brc1]], %[[brc2]] : tensor<8x8x8xi8>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<8x1x8xi8>, tensor<8x8x1xi8>) outs(%0:tensor<8x8x8xi8>) broadcast = [0,1,2] -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_vand_arith.andi_i8
func.func @test_func_vand_arith.andi_i8(%arg0:tensor<4x64x32xi8>,%arg1:tensor<4x64x32xi8>) -> tensor<4x64x32xi8> {
    %0 = tensor.empty():tensor<4x64x32xi8>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<4x64x32xi8>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<4x64x32xi8>, tensor<4x64x32xi8>) outs(%0:tensor<4x64x32xi8>)  -> tensor<4x64x32xi8>
    return %0 : tensor<4x64x32xi8>
}

// CHECK-LABEL: func.func @test_func_brc_vand_arith.andi_i16
func.func @test_func_brc_vand_arith.andi_i16(%arg0:tensor<8x1x8xi16>,%arg1:tensor<8x8x1xi16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.andi %[[brc1]], %[[brc2]] : tensor<8x8x8xi16>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<8x1x8xi16>, tensor<8x8x1xi16>) outs(%0:tensor<8x8x8xi16>) broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vand_arith.andi_i16
func.func @test_func_vand_arith.andi_i16(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>)  -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vand_arith.andi_i32
func.func @test_func_brc_vand_arith.andi_i32(%arg0:tensor<8x1x8xi32>,%arg1:tensor<8x8x1xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.andi %[[brc1]], %[[brc2]] : tensor<8x8x8xi32>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<8x1x8xi32>, tensor<8x8x1xi32>) outs(%0:tensor<8x8x8xi32>) broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vand_arith.andi_i32
func.func @test_func_vand_arith.andi_i32(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>)  -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vand_arith.andi_i64
func.func @test_func_brc_vand_arith.andi_i64(%arg0:tensor<8x1x8xi64>,%arg1:tensor<8x8x1xi64>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[RET:.*]] = arith.andi %[[brc1]], %[[brc2]] : tensor<8x8x8xi64>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<8x1x8xi64>, tensor<8x8x1xi64>) outs(%0:tensor<8x8x8xi64>) broadcast = [0,1,2] -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vand_arith.andi_i64
func.func @test_func_vand_arith.andi_i64(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi64> {
    %0 = tensor.empty():tensor<4x64x32xi64>
    //     CHECK: %[[RET:.*]] = arith.andi {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vand ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi64>)  -> tensor<4x64x32xi64>
    return %0 : tensor<4x64x32xi64>
}

// CHECK-LABEL: func.func @test_func_brc_vor_arith.ori_i1
func.func @test_func_brc_vor_arith.ori_i1(%arg0:tensor<8x1x8xi1>,%arg1:tensor<8x8x1xi1>) -> tensor<8x8x8xi1> {
    %0 = tensor.empty():tensor<8x8x8xi1>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.ori %[[brc1]], %[[brc2]] : tensor<8x8x8xi1>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<8x1x8xi1>, tensor<8x8x1xi1>) outs(%0:tensor<8x8x8xi1>) broadcast = [0,1,2] -> tensor<8x8x8xi1>
    return %0 : tensor<8x8x8xi1>
}

// CHECK-LABEL: func.func @test_func_vor_arith.ori_i1
func.func @test_func_vor_arith.ori_i1(%arg0:tensor<4x64x32xi1>,%arg1:tensor<4x64x32xi1>) -> tensor<4x64x32xi1> {
    %0 = tensor.empty():tensor<4x64x32xi1>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<4x64x32xi1>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<4x64x32xi1>, tensor<4x64x32xi1>) outs(%0:tensor<4x64x32xi1>)  -> tensor<4x64x32xi1>
    return %0 : tensor<4x64x32xi1>
}

// CHECK-LABEL: func.func @test_func_brc_vor_arith.ori_i8
func.func @test_func_brc_vor_arith.ori_i8(%arg0:tensor<8x1x8xi8>,%arg1:tensor<8x8x1xi8>) -> tensor<8x8x8xi8> {
    %0 = tensor.empty():tensor<8x8x8xi8>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi8> -> tensor<8x8x8xi8>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi8> -> tensor<8x8x8xi8>
    //     CHECK: %[[RET:.*]] = arith.ori %[[brc1]], %[[brc2]] : tensor<8x8x8xi8>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<8x1x8xi8>, tensor<8x8x1xi8>) outs(%0:tensor<8x8x8xi8>) broadcast = [0,1,2] -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_vor_arith.ori_i8
func.func @test_func_vor_arith.ori_i8(%arg0:tensor<4x64x32xi8>,%arg1:tensor<4x64x32xi8>) -> tensor<4x64x32xi8> {
    %0 = tensor.empty():tensor<4x64x32xi8>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<4x64x32xi8>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<4x64x32xi8>, tensor<4x64x32xi8>) outs(%0:tensor<4x64x32xi8>)  -> tensor<4x64x32xi8>
    return %0 : tensor<4x64x32xi8>
}

// CHECK-LABEL: func.func @test_func_brc_vor_arith.ori_i16
func.func @test_func_brc_vor_arith.ori_i16(%arg0:tensor<8x1x8xi16>,%arg1:tensor<8x8x1xi16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.ori %[[brc1]], %[[brc2]] : tensor<8x8x8xi16>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<8x1x8xi16>, tensor<8x8x1xi16>) outs(%0:tensor<8x8x8xi16>) broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vor_arith.ori_i16
func.func @test_func_vor_arith.ori_i16(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>)  -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vor_arith.ori_i32
func.func @test_func_brc_vor_arith.ori_i32(%arg0:tensor<8x1x8xi32>,%arg1:tensor<8x8x1xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.ori %[[brc1]], %[[brc2]] : tensor<8x8x8xi32>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<8x1x8xi32>, tensor<8x8x1xi32>) outs(%0:tensor<8x8x8xi32>) broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vor_arith.ori_i32
func.func @test_func_vor_arith.ori_i32(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>)  -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vor_arith.ori_i64
func.func @test_func_brc_vor_arith.ori_i64(%arg0:tensor<8x1x8xi64>,%arg1:tensor<8x8x1xi64>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[RET:.*]] = arith.ori %[[brc1]], %[[brc2]] : tensor<8x8x8xi64>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<8x1x8xi64>, tensor<8x8x1xi64>) outs(%0:tensor<8x8x8xi64>) broadcast = [0,1,2] -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vor_arith.ori_i64
func.func @test_func_vor_arith.ori_i64(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi64> {
    %0 = tensor.empty():tensor<4x64x32xi64>
    //     CHECK: %[[RET:.*]] = arith.ori {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vor ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi64>)  -> tensor<4x64x32xi64>
    return %0 : tensor<4x64x32xi64>
}

// CHECK-LABEL: func.func @test_func_brc_vadd_arith.addi_i16
func.func @test_func_brc_vadd_arith.addi_i16(%arg0:tensor<8x1x8xi16>,%arg1:tensor<8x8x1xi16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.addi %[[brc1]], %[[brc2]] : tensor<8x8x8xi16>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<8x1x8xi16>, tensor<8x8x1xi16>) outs(%0:tensor<8x8x8xi16>) broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vadd_arith.addi_i16
func.func @test_func_vadd_arith.addi_i16(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.addi {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>)  -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vadd_arith.addi_i32
func.func @test_func_brc_vadd_arith.addi_i32(%arg0:tensor<8x1x8xi32>,%arg1:tensor<8x8x1xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.addi %[[brc1]], %[[brc2]] : tensor<8x8x8xi32>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<8x1x8xi32>, tensor<8x8x1xi32>) outs(%0:tensor<8x8x8xi32>) broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vadd_arith.addi_i32
func.func @test_func_vadd_arith.addi_i32(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.addi {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>)  -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vadd_arith.addf_f16
func.func @test_func_brc_vadd_arith.addf_f16(%arg0:tensor<8x1x8xf16>,%arg1:tensor<8x8x1xf16>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.addf %[[brc1]], %[[brc2]] : tensor<8x8x8xf16>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<8x1x8xf16>, tensor<8x8x1xf16>) outs(%0:tensor<8x8x8xf16>) broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_vadd_arith.addf_f16
func.func @test_func_vadd_arith.addf_f16(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xf16> {
    %0 = tensor.empty():tensor<4x64x32xf16>
    //     CHECK: %[[RET:.*]] = arith.addf {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xf16>)  -> tensor<4x64x32xf16>
    return %0 : tensor<4x64x32xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vadd_arith.addf_f32
func.func @test_func_brc_vadd_arith.addf_f32(%arg0:tensor<8x1x8xf32>,%arg1:tensor<8x8x1xf32>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.addf %[[brc1]], %[[brc2]] : tensor<8x8x8xf32>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<8x1x8xf32>, tensor<8x8x1xf32>) outs(%0:tensor<8x8x8xf32>) broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vadd_arith.addf_f32
func.func @test_func_vadd_arith.addf_f32(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xf32> {
    %0 = tensor.empty():tensor<4x64x32xf32>
    //     CHECK: %[[RET:.*]] = arith.addf {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xf32>)  -> tensor<4x64x32xf32>
    return %0 : tensor<4x64x32xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vadd_arith.addi_i64
func.func @test_func_brc_vadd_arith.addi_i64(%arg0:tensor<8x1x8xi64>,%arg1:tensor<8x8x1xi64>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[RET:.*]] = arith.addi %[[brc1]], %[[brc2]] : tensor<8x8x8xi64>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<8x1x8xi64>, tensor<8x8x1xi64>) outs(%0:tensor<8x8x8xi64>) broadcast = [0,1,2] -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vadd_arith.addi_i64
func.func @test_func_vadd_arith.addi_i64(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi64> {
    %0 = tensor.empty():tensor<4x64x32xi64>
    //     CHECK: %[[RET:.*]] = arith.addi {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vadd ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi64>)  -> tensor<4x64x32xi64>
    return %0 : tensor<4x64x32xi64>
}

// CHECK-LABEL: func.func @test_func_brc_vsub_arith.subi_i16
func.func @test_func_brc_vsub_arith.subi_i16(%arg0:tensor<8x1x8xi16>,%arg1:tensor<8x8x1xi16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.subi %[[brc1]], %[[brc2]] : tensor<8x8x8xi16>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<8x1x8xi16>, tensor<8x8x1xi16>) outs(%0:tensor<8x8x8xi16>) broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vsub_arith.subi_i16
func.func @test_func_vsub_arith.subi_i16(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.subi {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>)  -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vsub_arith.subi_i32
func.func @test_func_brc_vsub_arith.subi_i32(%arg0:tensor<8x1x8xi32>,%arg1:tensor<8x8x1xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.subi %[[brc1]], %[[brc2]] : tensor<8x8x8xi32>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<8x1x8xi32>, tensor<8x8x1xi32>) outs(%0:tensor<8x8x8xi32>) broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vsub_arith.subi_i32
func.func @test_func_vsub_arith.subi_i32(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.subi {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>)  -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vsub_arith.subf_f16
func.func @test_func_brc_vsub_arith.subf_f16(%arg0:tensor<8x1x8xf16>,%arg1:tensor<8x8x1xf16>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.subf %[[brc1]], %[[brc2]] : tensor<8x8x8xf16>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<8x1x8xf16>, tensor<8x8x1xf16>) outs(%0:tensor<8x8x8xf16>) broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_vsub_arith.subf_f16
func.func @test_func_vsub_arith.subf_f16(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xf16> {
    %0 = tensor.empty():tensor<4x64x32xf16>
    //     CHECK: %[[RET:.*]] = arith.subf {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xf16>)  -> tensor<4x64x32xf16>
    return %0 : tensor<4x64x32xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vsub_arith.subf_f32
func.func @test_func_brc_vsub_arith.subf_f32(%arg0:tensor<8x1x8xf32>,%arg1:tensor<8x8x1xf32>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.subf %[[brc1]], %[[brc2]] : tensor<8x8x8xf32>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<8x1x8xf32>, tensor<8x8x1xf32>) outs(%0:tensor<8x8x8xf32>) broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vsub_arith.subf_f32
func.func @test_func_vsub_arith.subf_f32(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xf32> {
    %0 = tensor.empty():tensor<4x64x32xf32>
    //     CHECK: %[[RET:.*]] = arith.subf {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xf32>)  -> tensor<4x64x32xf32>
    return %0 : tensor<4x64x32xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vsub_arith.subi_i64
func.func @test_func_brc_vsub_arith.subi_i64(%arg0:tensor<8x1x8xi64>,%arg1:tensor<8x8x1xi64>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[RET:.*]] = arith.subi %[[brc1]], %[[brc2]] : tensor<8x8x8xi64>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<8x1x8xi64>, tensor<8x8x1xi64>) outs(%0:tensor<8x8x8xi64>) broadcast = [0,1,2] -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vsub_arith.subi_i64
func.func @test_func_vsub_arith.subi_i64(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi64> {
    %0 = tensor.empty():tensor<4x64x32xi64>
    //     CHECK: %[[RET:.*]] = arith.subi {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vsub ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi64>)  -> tensor<4x64x32xi64>
    return %0 : tensor<4x64x32xi64>
}

// CHECK-LABEL: func.func @test_func_brc_vmul_arith.muli_i16
func.func @test_func_brc_vmul_arith.muli_i16(%arg0:tensor<8x1x8xi16>,%arg1:tensor<8x8x1xi16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.muli %[[brc1]], %[[brc2]] : tensor<8x8x8xi16>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<8x1x8xi16>, tensor<8x8x1xi16>) outs(%0:tensor<8x8x8xi16>) broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vmul_arith.muli_i16
func.func @test_func_vmul_arith.muli_i16(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.muli {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>)  -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vmul_arith.muli_i32
func.func @test_func_brc_vmul_arith.muli_i32(%arg0:tensor<8x1x8xi32>,%arg1:tensor<8x8x1xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.muli %[[brc1]], %[[brc2]] : tensor<8x8x8xi32>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<8x1x8xi32>, tensor<8x8x1xi32>) outs(%0:tensor<8x8x8xi32>) broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vmul_arith.muli_i32
func.func @test_func_vmul_arith.muli_i32(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.muli {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>)  -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vmul_arith.mulf_f16
func.func @test_func_brc_vmul_arith.mulf_f16(%arg0:tensor<8x1x8xf16>,%arg1:tensor<8x8x1xf16>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.mulf %[[brc1]], %[[brc2]] : tensor<8x8x8xf16>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<8x1x8xf16>, tensor<8x8x1xf16>) outs(%0:tensor<8x8x8xf16>) broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_vmul_arith.mulf_f16
func.func @test_func_vmul_arith.mulf_f16(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xf16> {
    %0 = tensor.empty():tensor<4x64x32xf16>
    //     CHECK: %[[RET:.*]] = arith.mulf {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xf16>)  -> tensor<4x64x32xf16>
    return %0 : tensor<4x64x32xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vmul_arith.mulf_f32
func.func @test_func_brc_vmul_arith.mulf_f32(%arg0:tensor<8x1x8xf32>,%arg1:tensor<8x8x1xf32>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.mulf %[[brc1]], %[[brc2]] : tensor<8x8x8xf32>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<8x1x8xf32>, tensor<8x8x1xf32>) outs(%0:tensor<8x8x8xf32>) broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vmul_arith.mulf_f32
func.func @test_func_vmul_arith.mulf_f32(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xf32> {
    %0 = tensor.empty():tensor<4x64x32xf32>
    //     CHECK: %[[RET:.*]] = arith.mulf {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xf32>)  -> tensor<4x64x32xf32>
    return %0 : tensor<4x64x32xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vmul_arith.muli_i64
func.func @test_func_brc_vmul_arith.muli_i64(%arg0:tensor<8x1x8xi64>,%arg1:tensor<8x8x1xi64>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[RET:.*]] = arith.muli %[[brc1]], %[[brc2]] : tensor<8x8x8xi64>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<8x1x8xi64>, tensor<8x8x1xi64>) outs(%0:tensor<8x8x8xi64>) broadcast = [0,1,2] -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vmul_arith.muli_i64
func.func @test_func_vmul_arith.muli_i64(%arg0:tensor<4x64x32xi64>,%arg1:tensor<4x64x32xi64>) -> tensor<4x64x32xi64> {
    %0 = tensor.empty():tensor<4x64x32xi64>
    //     CHECK: %[[RET:.*]] = arith.muli {{.*}}, {{.*}} : tensor<4x64x32xi64>
    hivm.hir.vmul ins(%arg0,%arg1 : tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi64>)  -> tensor<4x64x32xi64>
    return %0 : tensor<4x64x32xi64>
}

// CHECK-LABEL: func.func @test_func_vmodui_arith.remui_i16
func.func @test_func_vmodui_arith.remui_i16(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.remui {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vmodui ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>)  -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_vmodui_arith.remui_i32
func.func @test_func_vmodui_arith.remui_i32(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.remui {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vmodui ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>)  -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vdiv_arith.divf_f16_signed
func.func @test_func_brc_vdiv_arith.divf_f16_signed(%arg0:tensor<8x1x8xf16>,%arg1:tensor<8x8x1xf16>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.divf %[[brc1]], %[[brc2]] : tensor<8x8x8xf16>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<8x1x8xf16>, tensor<8x8x1xf16>) outs(%0:tensor<8x8x8xf16>) isSigned = true broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vdiv_arith.divf_f32_signed
func.func @test_func_brc_vdiv_arith.divf_f32_signed(%arg0:tensor<8x1x8xf32>,%arg1:tensor<8x8x1xf32>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.divf %[[brc1]], %[[brc2]] : tensor<8x8x8xf32>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<8x1x8xf32>, tensor<8x8x1xf32>) outs(%0:tensor<8x8x8xf32>) isSigned = true broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vdiv_arith.divsi_i16_signed
func.func @test_func_brc_vdiv_arith.divsi_i16_signed(%arg0:tensor<8x1x8xi16>,%arg1:tensor<8x8x1xi16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.divsi %[[brc1]], %[[brc2]] : tensor<8x8x8xi16>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<8x1x8xi16>, tensor<8x8x1xi16>) outs(%0:tensor<8x8x8xi16>) isSigned = true broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vdiv_arith.divsi_i32_signed
func.func @test_func_brc_vdiv_arith.divsi_i32_signed(%arg0:tensor<8x1x8xi32>,%arg1:tensor<8x8x1xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.divsi %[[brc1]], %[[brc2]] : tensor<8x8x8xi32>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<8x1x8xi32>, tensor<8x8x1xi32>) outs(%0:tensor<8x8x8xi32>) isSigned = true broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vdiv_arith.divf_f16_unsigned
func.func @test_func_brc_vdiv_arith.divf_f16_unsigned(%arg0:tensor<8x1x8xf16>,%arg1:tensor<8x8x1xf16>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.divf %[[brc1]], %[[brc2]] : tensor<8x8x8xf16>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<8x1x8xf16>, tensor<8x8x1xf16>) outs(%0:tensor<8x8x8xf16>) isSigned = false broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vdiv_arith.divf_f32_unsigned
func.func @test_func_brc_vdiv_arith.divf_f32_unsigned(%arg0:tensor<8x1x8xf32>,%arg1:tensor<8x8x1xf32>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.divf %[[brc1]], %[[brc2]] : tensor<8x8x8xf32>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<8x1x8xf32>, tensor<8x8x1xf32>) outs(%0:tensor<8x8x8xf32>) isSigned = false broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vdiv_arith.divui_i16_unsigned
func.func @test_func_brc_vdiv_arith.divui_i16_unsigned(%arg0:tensor<8x1x8xi16>,%arg1:tensor<8x8x1xi16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.divui %[[brc1]], %[[brc2]] : tensor<8x8x8xi16>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<8x1x8xi16>, tensor<8x8x1xi16>) outs(%0:tensor<8x8x8xi16>) isSigned = false broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vdiv_arith.divui_i32_unsigned
func.func @test_func_brc_vdiv_arith.divui_i32_unsigned(%arg0:tensor<8x1x8xi32>,%arg1:tensor<8x8x1xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.divui %[[brc1]], %[[brc2]] : tensor<8x8x8xi32>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<8x1x8xi32>, tensor<8x8x1xi32>) outs(%0:tensor<8x8x8xi32>) isSigned = false broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vdiv_arith.divf_f16_signed
func.func @test_func_vdiv_arith.divf_f16_signed(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xf16> {
    %0 = tensor.empty():tensor<4x64x32xf16>
    //     CHECK: %[[RET:.*]] = arith.divf {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xf16>) isSigned = true -> tensor<4x64x32xf16>
    return %0 : tensor<4x64x32xf16>
}

// CHECK-LABEL: func.func @test_func_vdiv_arith.divf_f32_signed
func.func @test_func_vdiv_arith.divf_f32_signed(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xf32> {
    %0 = tensor.empty():tensor<4x64x32xf32>
    //     CHECK: %[[RET:.*]] = arith.divf {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xf32>) isSigned = true -> tensor<4x64x32xf32>
    return %0 : tensor<4x64x32xf32>
}

// CHECK-LABEL: func.func @test_func_vdiv_arith.divsi_i16_signed
func.func @test_func_vdiv_arith.divsi_i16_signed(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.divsi {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>) isSigned = true -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_vdiv_arith.divsi_i32_signed
func.func @test_func_vdiv_arith.divsi_i32_signed(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.divsi {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>) isSigned = true -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_vdiv_arith.divf_f16_unsigned
func.func @test_func_vdiv_arith.divf_f16_unsigned(%arg0:tensor<4x64x32xf16>,%arg1:tensor<4x64x32xf16>) -> tensor<4x64x32xf16> {
    %0 = tensor.empty():tensor<4x64x32xf16>
    //     CHECK: %[[RET:.*]] = arith.divf {{.*}}, {{.*}} : tensor<4x64x32xf16>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xf16>) isSigned = false -> tensor<4x64x32xf16>
    return %0 : tensor<4x64x32xf16>
}

// CHECK-LABEL: func.func @test_func_vdiv_arith.divf_f32_unsigned
func.func @test_func_vdiv_arith.divf_f32_unsigned(%arg0:tensor<4x64x32xf32>,%arg1:tensor<4x64x32xf32>) -> tensor<4x64x32xf32> {
    %0 = tensor.empty():tensor<4x64x32xf32>
    //     CHECK: %[[RET:.*]] = arith.divf {{.*}}, {{.*}} : tensor<4x64x32xf32>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xf32>) isSigned = false -> tensor<4x64x32xf32>
    return %0 : tensor<4x64x32xf32>
}

// CHECK-LABEL: func.func @test_func_vdiv_arith.divui_i16_unsigned
func.func @test_func_vdiv_arith.divui_i16_unsigned(%arg0:tensor<4x64x32xi16>,%arg1:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.divui {{.*}}, {{.*}} : tensor<4x64x32xi16>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>) isSigned = false -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_vdiv_arith.divui_i32_unsigned
func.func @test_func_vdiv_arith.divui_i32_unsigned(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.divui {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vdiv ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>) isSigned = false -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_vmulext_arith.mulsi_extended_i32_low_with_return
func.func @test_func_vmulext_arith.mulsi_extended_i32_low_with_return(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    %1 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended {{.*}}, {{.*}} : tensor<4x64x32xi32>
    %2:2 = hivm.hir.vmulext                 ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>)                 outs(%0,%1:tensor<4x64x32xi32>,tensor<4x64x32xi32>)                  -> tensor<4x64x32xi32>,tensor<4x64x32xi32>
    return %2#0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_vmulext_arith.mulsi_extended_i32_high_with_return
func.func @test_func_vmulext_arith.mulsi_extended_i32_high_with_return(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    %1 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended {{.*}}, {{.*}} : tensor<4x64x32xi32>
    %2:2 = hivm.hir.vmulext                 ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>)                 outs(%0,%1:tensor<4x64x32xi32>,tensor<4x64x32xi32>)                  -> tensor<4x64x32xi32>,tensor<4x64x32xi32>
    return %2#1 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_vmulext_arith.mulsi_extended_i32_low
func.func @test_func_vmulext_arith.mulsi_extended_i32_low(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    %1 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vmulext                 ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>)                 outs(%0,%1:tensor<4x64x32xi32>,tensor<4x64x32xi32>)                  -> tensor<4x64x32xi32>,tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_vmulext_arith.mulsi_extended_i32_high
func.func @test_func_vmulext_arith.mulsi_extended_i32_high(%arg0:tensor<4x64x32xi32>,%arg1:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    %1 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[LOW:.*]], %[[HIGH:.*]] = arith.mulsi_extended {{.*}}, {{.*}} : tensor<4x64x32xi32>
    hivm.hir.vmulext                 ins(%arg0,%arg1 : tensor<4x64x32xi32>, tensor<4x64x32xi32>)                 outs(%0,%1:tensor<4x64x32xi32>,tensor<4x64x32xi32>)                  -> tensor<4x64x32xi32>,tensor<4x64x32xi32>
    return %1 : tensor<4x64x32xi32>
}