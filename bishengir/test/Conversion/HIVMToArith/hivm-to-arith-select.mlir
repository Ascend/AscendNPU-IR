// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_brc_vsel_arith.select_i16_i1
func.func @test_func_brc_vsel_arith.select_i16_i1(%arg0:tensor<8x1x8xi1>,%arg1:tensor<8x8x1xi16>,%arg2:tensor<1x8x8xi16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[brc3:.*]] = tt.broadcast %arg2  : tensor<1x8x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.select %[[brc1]], %[[brc2]], %[[brc3]] : tensor<8x8x8xi1>, tensor<8x8x8xi16>
    hivm.hir.vsel {broadcast=array<i64:0,1,2>} ins(%arg0,%arg1,%arg2 : tensor<8x1x8xi1>, tensor<8x8x1xi16>, tensor<1x8x8xi16>) outs(%0:tensor<8x8x8xi16>)  -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vsel_arith.select_f16_i1
func.func @test_func_brc_vsel_arith.select_f16_i1(%arg0:tensor<8x1x8xi1>,%arg1:tensor<8x8x1xf16>,%arg2:tensor<1x8x8xf16>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[brc3:.*]] = tt.broadcast %arg2  : tensor<1x8x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.select %[[brc1]], %[[brc2]], %[[brc3]] : tensor<8x8x8xi1>, tensor<8x8x8xf16>
    hivm.hir.vsel {broadcast=array<i64:0,1,2>} ins(%arg0,%arg1,%arg2 : tensor<8x1x8xi1>, tensor<8x8x1xf16>, tensor<1x8x8xf16>) outs(%0:tensor<8x8x8xf16>)  -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vsel_arith.select_i32_i1
func.func @test_func_brc_vsel_arith.select_i32_i1(%arg0:tensor<8x1x8xi1>,%arg1:tensor<8x8x1xi32>,%arg2:tensor<1x8x8xi32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[brc3:.*]] = tt.broadcast %arg2  : tensor<1x8x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.select %[[brc1]], %[[brc2]], %[[brc3]] : tensor<8x8x8xi1>, tensor<8x8x8xi32>
    hivm.hir.vsel {broadcast=array<i64:0,1,2>} ins(%arg0,%arg1,%arg2 : tensor<8x1x8xi1>, tensor<8x8x1xi32>, tensor<1x8x8xi32>) outs(%0:tensor<8x8x8xi32>)  -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vsel_arith.select_f32_i1
func.func @test_func_brc_vsel_arith.select_f32_i1(%arg0:tensor<8x1x8xi1>,%arg1:tensor<8x8x1xf32>,%arg2:tensor<1x8x8xf32>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[brc3:.*]] = tt.broadcast %arg2  : tensor<1x8x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.select %[[brc1]], %[[brc2]], %[[brc3]] : tensor<8x8x8xi1>, tensor<8x8x8xf32>
    hivm.hir.vsel {broadcast=array<i64:0,1,2>} ins(%arg0,%arg1,%arg2 : tensor<8x1x8xi1>, tensor<8x8x1xf32>, tensor<1x8x8xf32>) outs(%0:tensor<8x8x8xf32>)  -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vsel_arith.select_i64_i1
func.func @test_func_brc_vsel_arith.select_i64_i1(%arg0:tensor<8x1x8xi1>,%arg1:tensor<8x8x1xi64>,%arg2:tensor<1x8x8xi64>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[brc2:.*]] = tt.broadcast %arg1  : tensor<8x8x1xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[brc3:.*]] = tt.broadcast %arg2  : tensor<1x8x8xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[RET:.*]] = arith.select %[[brc1]], %[[brc2]], %[[brc3]] : tensor<8x8x8xi1>, tensor<8x8x8xi64>
    hivm.hir.vsel {broadcast=array<i64:0,1,2>} ins(%arg0,%arg1,%arg2 : tensor<8x1x8xi1>, tensor<8x8x1xi64>, tensor<1x8x8xi64>) outs(%0:tensor<8x8x8xi64>)  -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_i16_i1
func.func @test_func_vsel_arith.select_i16_i1(%arg0:tensor<4x64x32xi1>,%arg1:tensor<4x64x32xi16>,%arg2:tensor<4x64x32xi16>) -> tensor<4x64x32xi16> {
    %0 = tensor.empty():tensor<4x64x32xi16>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<4x64x32xi1>, tensor<4x64x32xi16>
    hivm.hir.vsel  ins(%arg0,%arg1,%arg2 : tensor<4x64x32xi1>, tensor<4x64x32xi16>, tensor<4x64x32xi16>) outs(%0:tensor<4x64x32xi16>)  -> tensor<4x64x32xi16>
    return %0 : tensor<4x64x32xi16>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_f16_i1
func.func @test_func_vsel_arith.select_f16_i1(%arg0:tensor<4x64x32xi1>,%arg1:tensor<4x64x32xf16>,%arg2:tensor<4x64x32xf16>) -> tensor<4x64x32xf16> {
    %0 = tensor.empty():tensor<4x64x32xf16>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<4x64x32xi1>, tensor<4x64x32xf16>
    hivm.hir.vsel  ins(%arg0,%arg1,%arg2 : tensor<4x64x32xi1>, tensor<4x64x32xf16>, tensor<4x64x32xf16>) outs(%0:tensor<4x64x32xf16>)  -> tensor<4x64x32xf16>
    return %0 : tensor<4x64x32xf16>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_i32_i1
func.func @test_func_vsel_arith.select_i32_i1(%arg0:tensor<4x64x32xi1>,%arg1:tensor<4x64x32xi32>,%arg2:tensor<4x64x32xi32>) -> tensor<4x64x32xi32> {
    %0 = tensor.empty():tensor<4x64x32xi32>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<4x64x32xi1>, tensor<4x64x32xi32>
    hivm.hir.vsel  ins(%arg0,%arg1,%arg2 : tensor<4x64x32xi1>, tensor<4x64x32xi32>, tensor<4x64x32xi32>) outs(%0:tensor<4x64x32xi32>)  -> tensor<4x64x32xi32>
    return %0 : tensor<4x64x32xi32>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_f32_i1
func.func @test_func_vsel_arith.select_f32_i1(%arg0:tensor<4x64x32xi1>,%arg1:tensor<4x64x32xf32>,%arg2:tensor<4x64x32xf32>) -> tensor<4x64x32xf32> {
    %0 = tensor.empty():tensor<4x64x32xf32>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<4x64x32xi1>, tensor<4x64x32xf32>
    hivm.hir.vsel  ins(%arg0,%arg1,%arg2 : tensor<4x64x32xi1>, tensor<4x64x32xf32>, tensor<4x64x32xf32>) outs(%0:tensor<4x64x32xf32>)  -> tensor<4x64x32xf32>
    return %0 : tensor<4x64x32xf32>
}

// CHECK-LABEL: func.func @test_func_vsel_arith.select_i64_i1
func.func @test_func_vsel_arith.select_i64_i1(%arg0:tensor<4x64x32xi1>,%arg1:tensor<4x64x32xi64>,%arg2:tensor<4x64x32xi64>) -> tensor<4x64x32xi64> {
    %0 = tensor.empty():tensor<4x64x32xi64>
    //     CHECK: %[[RET:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : tensor<4x64x32xi1>, tensor<4x64x32xi64>
    hivm.hir.vsel  ins(%arg0,%arg1,%arg2 : tensor<4x64x32xi1>, tensor<4x64x32xi64>, tensor<4x64x32xi64>) outs(%0:tensor<4x64x32xi64>)  -> tensor<4x64x32xi64>
    return %0 : tensor<4x64x32xi64>
}