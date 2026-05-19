// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_brc_vcast_i8_i16_cast_signed_extsi
func.func @test_func_brc_vcast_i8_i16_cast_signed_extsi(%arg0:tensor<8x1x8xi8>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi8> -> tensor<8x8x8xi8>
    //     CHECK: %[[RET:.*]] = arith.extsi %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi8> to tensor<8x8x8xi16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi8>) outs(%0:tensor<8x8x8xi16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_i8_i16_cast_signed_extsi
func.func @test_func_vcast_i8_i16_cast_signed_extsi(%arg0:tensor<4x32x64xi8>) -> tensor<4x32x64xi16> {
    %0 = tensor.empty():tensor<4x32x64xi16>
    //     CHECK: %[[RET:.*]] = arith.extsi {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi8> to tensor<4x32x64xi16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi8>) outs(%0:tensor<4x32x64xi16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi16>
    return %0 : tensor<4x32x64xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i8_i32_cast_signed_extsi
func.func @test_func_brc_vcast_i8_i32_cast_signed_extsi(%arg0:tensor<8x1x8xi8>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi8> -> tensor<8x8x8xi8>
    //     CHECK: %[[RET:.*]] = arith.extsi %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi8> to tensor<8x8x8xi32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi8>) outs(%0:tensor<8x8x8xi32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_i8_i32_cast_signed_extsi
func.func @test_func_vcast_i8_i32_cast_signed_extsi(%arg0:tensor<4x32x64xi8>) -> tensor<4x32x64xi32> {
    %0 = tensor.empty():tensor<4x32x64xi32>
    //     CHECK: %[[RET:.*]] = arith.extsi {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi8> to tensor<4x32x64xi32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi8>) outs(%0:tensor<4x32x64xi32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi32>
    return %0 : tensor<4x32x64xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i16_i32_cast_signed_extsi
func.func @test_func_brc_vcast_i16_i32_cast_signed_extsi(%arg0:tensor<8x1x8xi16>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.extsi %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi16> to tensor<8x8x8xi32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi16>) outs(%0:tensor<8x8x8xi32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_i16_i32_cast_signed_extsi
func.func @test_func_vcast_i16_i32_cast_signed_extsi(%arg0:tensor<4x32x64xi16>) -> tensor<4x32x64xi32> {
    %0 = tensor.empty():tensor<4x32x64xi32>
    //     CHECK: %[[RET:.*]] = arith.extsi {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi16> to tensor<4x32x64xi32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi16>) outs(%0:tensor<4x32x64xi32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi32>
    return %0 : tensor<4x32x64xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i32_i64_cast_signed_extsi
func.func @test_func_brc_vcast_i32_i64_cast_signed_extsi(%arg0:tensor<8x1x8xi32>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.extsi %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi32> to tensor<8x8x8xi64>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi32>) outs(%0:tensor<8x8x8xi64>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_i64_cast_signed_extsi
func.func @test_func_vcast_i32_i64_cast_signed_extsi(%arg0:tensor<4x32x64xi32>) -> tensor<4x32x64xi64> {
    %0 = tensor.empty():tensor<4x32x64xi64>
    //     CHECK: %[[RET:.*]] = arith.extsi {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi32> to tensor<4x32x64xi64>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi32>) outs(%0:tensor<4x32x64xi64>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi64>
    return %0 : tensor<4x32x64xi64>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_i8_cast_signed_extsi
func.func @test_func_brc_vcast_i1_i8_cast_signed_extsi(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xi8> {
    %0 = tensor.empty():tensor<8x8x8xi8>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.extui %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xi8>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xi8>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i8_cast_signed_extsi
func.func @test_func_vcast_i1_i8_cast_signed_extsi(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xi8> {
    %0 = tensor.empty():tensor<4x32x64xi8>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xi8>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xi8>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi8>
    return %0 : tensor<4x32x64xi8>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_i16_cast_signed_extsi
func.func @test_func_brc_vcast_i1_i16_cast_signed_extsi(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.extui %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xi16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xi16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i16_cast_signed_extsi
func.func @test_func_vcast_i1_i16_cast_signed_extsi(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xi16> {
    %0 = tensor.empty():tensor<4x32x64xi16>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xi16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xi16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi16>
    return %0 : tensor<4x32x64xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_i32_cast_signed_extsi
func.func @test_func_brc_vcast_i1_i32_cast_signed_extsi(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.extui %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xi32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xi32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i32_cast_signed_extsi
func.func @test_func_vcast_i1_i32_cast_signed_extsi(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xi32> {
    %0 = tensor.empty():tensor<4x32x64xi32>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xi32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xi32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi32>
    return %0 : tensor<4x32x64xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_i16_cast_signed_extui
func.func @test_func_brc_vcast_i1_i16_cast_signed_extui(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.extui %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xi16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xi16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i16_cast_signed_extui
func.func @test_func_vcast_i1_i16_cast_signed_extui(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xi16> {
    %0 = tensor.empty():tensor<4x32x64xi16>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xi16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xi16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi16>
    return %0 : tensor<4x32x64xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_i32_cast_signed_extui
func.func @test_func_brc_vcast_i1_i32_cast_signed_extui(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.extui %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xi32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xi32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i32_cast_signed_extui
func.func @test_func_vcast_i1_i32_cast_signed_extui(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xi32> {
    %0 = tensor.empty():tensor<4x32x64xi32>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xi32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xi32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xi32>
    return %0 : tensor<4x32x64xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i32_i64_cast_unsigned_extui
func.func @test_func_brc_vcast_i32_i64_cast_unsigned_extui(%arg0:tensor<8x1x8xi32>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.extui %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi32> to tensor<8x8x8xi64>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi32>) outs(%0:tensor<8x8x8xi64>) round_mode = <rint> cast = <cast_unsigned> broadcast = [0,1,2] -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_i64_cast_unsigned_extui
func.func @test_func_vcast_i32_i64_cast_unsigned_extui(%arg0:tensor<4x32x64xi32>) -> tensor<4x32x64xi64> {
    %0 = tensor.empty():tensor<4x32x64xi64>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi32> to tensor<4x32x64xi64>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi32>) outs(%0:tensor<4x32x64xi64>) round_mode = <rint> cast = <cast_unsigned> -> tensor<4x32x64xi64>
    return %0 : tensor<4x32x64xi64>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_f16_f32_cast_signed_extf
func.func @test_func_brc_vcast_f16_f32_cast_signed_extf(%arg0:tensor<8x1x8xf16>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.extf %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xf16> to tensor<8x8x8xf32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xf16>) outs(%0:tensor<8x8x8xf32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_f16_f32_cast_signed_extf
func.func @test_func_vcast_f16_f32_cast_signed_extf(%arg0:tensor<4x32x64xf16>) -> tensor<4x32x64xf32> {
    %0 = tensor.empty():tensor<4x32x64xf32>
    //     CHECK: %[[RET:.*]] = arith.extf {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xf16> to tensor<4x32x64xf32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xf16>) outs(%0:tensor<4x32x64xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf32>
    return %0 : tensor<4x32x64xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_bf16_f32_cast_signed_extf
func.func @test_func_brc_vcast_bf16_f32_cast_signed_extf(%arg0:tensor<8x1x8xbf16>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xbf16> -> tensor<8x8x8xbf16>
    //     CHECK: %[[RET:.*]] = arith.extf %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xbf16> to tensor<8x8x8xf32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xbf16>) outs(%0:tensor<8x8x8xf32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_bf16_f32_cast_signed_extf
func.func @test_func_vcast_bf16_f32_cast_signed_extf(%arg0:tensor<4x32x64xbf16>) -> tensor<4x32x64xf32> {
    %0 = tensor.empty():tensor<4x32x64xf32>
    //     CHECK: %[[RET:.*]] = arith.extf {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xbf16> to tensor<4x32x64xf32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xbf16>) outs(%0:tensor<4x32x64xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf32>
    return %0 : tensor<4x32x64xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_f16_i16_cast_signed_fptosi
func.func @test_func_brc_vcast_f16_i16_cast_signed_fptosi(%arg0:tensor<8x1x8xf16>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.fptosi %[[brc1]] {round_mode = #hivm.round_mode<trunc>} : tensor<8x8x8xf16> to tensor<8x8x8xi16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xf16>) outs(%0:tensor<8x8x8xi16>) round_mode = <trunc> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_f16_i16_cast_signed_fptosi
func.func @test_func_vcast_f16_i16_cast_signed_fptosi(%arg0:tensor<4x32x64xf16>) -> tensor<4x32x64xi16> {
    %0 = tensor.empty():tensor<4x32x64xi16>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} {round_mode = #hivm.round_mode<trunc>} : tensor<4x32x64xf16> to tensor<4x32x64xi16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xf16>) outs(%0:tensor<4x32x64xi16>) round_mode = <trunc> cast = <cast_signed> -> tensor<4x32x64xi16>
    return %0 : tensor<4x32x64xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_f16_i32_cast_signed_fptosi
func.func @test_func_brc_vcast_f16_i32_cast_signed_fptosi(%arg0:tensor<8x1x8xf16>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf16> -> tensor<8x8x8xf16>
    //     CHECK: %[[RET:.*]] = arith.fptosi %[[brc1]] {round_mode = #hivm.round_mode<trunc>} : tensor<8x8x8xf16> to tensor<8x8x8xi32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xf16>) outs(%0:tensor<8x8x8xi32>) round_mode = <trunc> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_f16_i32_cast_signed_fptosi
func.func @test_func_vcast_f16_i32_cast_signed_fptosi(%arg0:tensor<4x32x64xf16>) -> tensor<4x32x64xi32> {
    %0 = tensor.empty():tensor<4x32x64xi32>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} {round_mode = #hivm.round_mode<trunc>} : tensor<4x32x64xf16> to tensor<4x32x64xi32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xf16>) outs(%0:tensor<4x32x64xi32>) round_mode = <trunc> cast = <cast_signed> -> tensor<4x32x64xi32>
    return %0 : tensor<4x32x64xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_bf16_i32_cast_signed_fptosi
func.func @test_func_brc_vcast_bf16_i32_cast_signed_fptosi(%arg0:tensor<8x1x8xbf16>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xbf16> -> tensor<8x8x8xbf16>
    //     CHECK: %[[RET:.*]] = arith.fptosi %[[brc1]] {round_mode = #hivm.round_mode<trunc>} : tensor<8x8x8xbf16> to tensor<8x8x8xi32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xbf16>) outs(%0:tensor<8x8x8xi32>) round_mode = <trunc> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_bf16_i32_cast_signed_fptosi
func.func @test_func_vcast_bf16_i32_cast_signed_fptosi(%arg0:tensor<4x32x64xbf16>) -> tensor<4x32x64xi32> {
    %0 = tensor.empty():tensor<4x32x64xi32>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} {round_mode = #hivm.round_mode<trunc>} : tensor<4x32x64xbf16> to tensor<4x32x64xi32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xbf16>) outs(%0:tensor<4x32x64xi32>) round_mode = <trunc> cast = <cast_signed> -> tensor<4x32x64xi32>
    return %0 : tensor<4x32x64xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_f32_i32_cast_signed_fptosi
func.func @test_func_brc_vcast_f32_i32_cast_signed_fptosi(%arg0:tensor<8x1x8xf32>) -> tensor<8x8x8xi32> {
    %0 = tensor.empty():tensor<8x8x8xi32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.fptosi %[[brc1]] {round_mode = #hivm.round_mode<trunc>} : tensor<8x8x8xf32> to tensor<8x8x8xi32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xf32>) outs(%0:tensor<8x8x8xi32>) round_mode = <trunc> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_f32_i32_cast_signed_fptosi
func.func @test_func_vcast_f32_i32_cast_signed_fptosi(%arg0:tensor<4x32x64xf32>) -> tensor<4x32x64xi32> {
    %0 = tensor.empty():tensor<4x32x64xi32>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} {round_mode = #hivm.round_mode<trunc>} : tensor<4x32x64xf32> to tensor<4x32x64xi32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xf32>) outs(%0:tensor<4x32x64xi32>) round_mode = <trunc> cast = <cast_signed> -> tensor<4x32x64xi32>
    return %0 : tensor<4x32x64xi32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_f32_i64_cast_signed_fptosi
func.func @test_func_brc_vcast_f32_i64_cast_signed_fptosi(%arg0:tensor<8x1x8xf32>) -> tensor<8x8x8xi64> {
    %0 = tensor.empty():tensor<8x8x8xi64>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.fptosi %[[brc1]] {round_mode = #hivm.round_mode<trunc>} : tensor<8x8x8xf32> to tensor<8x8x8xi64>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xf32>) outs(%0:tensor<8x8x8xi64>) round_mode = <trunc> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_vcast_f32_i64_cast_signed_fptosi
func.func @test_func_vcast_f32_i64_cast_signed_fptosi(%arg0:tensor<4x32x64xf32>) -> tensor<4x32x64xi64> {
    %0 = tensor.empty():tensor<4x32x64xi64>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} {round_mode = #hivm.round_mode<trunc>} : tensor<4x32x64xf32> to tensor<4x32x64xi64>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xf32>) outs(%0:tensor<4x32x64xi64>) round_mode = <trunc> cast = <cast_signed> -> tensor<4x32x64xi64>
    return %0 : tensor<4x32x64xi64>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i8_f16_cast_signed_sitofp
func.func @test_func_brc_vcast_i8_f16_cast_signed_sitofp(%arg0:tensor<8x1x8xi8>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi8> -> tensor<8x8x8xi8>
    //     CHECK: %[[RET:.*]] = arith.sitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi8> to tensor<8x8x8xf16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi8>) outs(%0:tensor<8x8x8xf16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i8_f16_cast_signed_sitofp
func.func @test_func_vcast_i8_f16_cast_signed_sitofp(%arg0:tensor<4x32x64xi8>) -> tensor<4x32x64xf16> {
    %0 = tensor.empty():tensor<4x32x64xf16>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi8> to tensor<4x32x64xf16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi8>) outs(%0:tensor<4x32x64xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf16>
    return %0 : tensor<4x32x64xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i16_f16_cast_signed_sitofp
func.func @test_func_brc_vcast_i16_f16_cast_signed_sitofp(%arg0:tensor<8x1x8xi16>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.sitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi16> to tensor<8x8x8xf16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi16>) outs(%0:tensor<8x8x8xf16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i16_f16_cast_signed_sitofp
func.func @test_func_vcast_i16_f16_cast_signed_sitofp(%arg0:tensor<4x32x64xi16>) -> tensor<4x32x64xf16> {
    %0 = tensor.empty():tensor<4x32x64xf16>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi16> to tensor<4x32x64xf16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi16>) outs(%0:tensor<4x32x64xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf16>
    return %0 : tensor<4x32x64xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i16_f32_cast_signed_sitofp
func.func @test_func_brc_vcast_i16_f32_cast_signed_sitofp(%arg0:tensor<8x1x8xi16>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.sitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi16> to tensor<8x8x8xf32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi16>) outs(%0:tensor<8x8x8xf32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i16_f32_cast_signed_sitofp
func.func @test_func_vcast_i16_f32_cast_signed_sitofp(%arg0:tensor<4x32x64xi16>) -> tensor<4x32x64xf32> {
    %0 = tensor.empty():tensor<4x32x64xf32>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi16> to tensor<4x32x64xf32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi16>) outs(%0:tensor<4x32x64xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf32>
    return %0 : tensor<4x32x64xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i32_f32_cast_signed_sitofp
func.func @test_func_brc_vcast_i32_f32_cast_signed_sitofp(%arg0:tensor<8x1x8xi32>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.sitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi32> to tensor<8x8x8xf32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi32>) outs(%0:tensor<8x8x8xf32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_f32_cast_signed_sitofp
func.func @test_func_vcast_i32_f32_cast_signed_sitofp(%arg0:tensor<4x32x64xi32>) -> tensor<4x32x64xf32> {
    %0 = tensor.empty():tensor<4x32x64xf32>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi32> to tensor<4x32x64xf32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi32>) outs(%0:tensor<4x32x64xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf32>
    return %0 : tensor<4x32x64xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i64_f32_cast_signed_sitofp
func.func @test_func_brc_vcast_i64_f32_cast_signed_sitofp(%arg0:tensor<8x1x8xi64>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi64> -> tensor<8x8x8xi64>
    //     CHECK: %[[RET:.*]] = arith.sitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi64> to tensor<8x8x8xf32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi64>) outs(%0:tensor<8x8x8xf32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i64_f32_cast_signed_sitofp
func.func @test_func_vcast_i64_f32_cast_signed_sitofp(%arg0:tensor<4x32x64xi64>) -> tensor<4x32x64xf32> {
    %0 = tensor.empty():tensor<4x32x64xf32>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi64> to tensor<4x32x64xf32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi64>) outs(%0:tensor<4x32x64xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf32>
    return %0 : tensor<4x32x64xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_f16_cast_signed_sitofp
func.func @test_func_brc_vcast_i1_f16_cast_signed_sitofp(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.sitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xf16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xf16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_f16_cast_signed_sitofp
func.func @test_func_vcast_i1_f16_cast_signed_sitofp(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xf16> {
    %0 = tensor.empty():tensor<4x32x64xf16>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xf16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf16>
    return %0 : tensor<4x32x64xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_bf16_cast_signed_sitofp
func.func @test_func_brc_vcast_i1_bf16_cast_signed_sitofp(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xbf16> {
    %0 = tensor.empty():tensor<8x8x8xbf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.sitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xbf16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xbf16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xbf16>
    return %0 : tensor<8x8x8xbf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_bf16_cast_signed_sitofp
func.func @test_func_vcast_i1_bf16_cast_signed_sitofp(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xbf16> {
    %0 = tensor.empty():tensor<4x32x64xbf16>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xbf16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xbf16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xbf16>
    return %0 : tensor<4x32x64xbf16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_f32_cast_signed_sitofp
func.func @test_func_brc_vcast_i1_f32_cast_signed_sitofp(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.sitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xf32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xf32>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_f32_cast_signed_sitofp
func.func @test_func_vcast_i1_f32_cast_signed_sitofp(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xf32> {
    %0 = tensor.empty():tensor<4x32x64xf32>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xf32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf32>
    return %0 : tensor<4x32x64xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_f16_cast_unsigned_uitofp
func.func @test_func_brc_vcast_i1_f16_cast_unsigned_uitofp(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.uitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xf16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xf16>) round_mode = <rint> cast = <cast_unsigned> broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_f16_cast_unsigned_uitofp
func.func @test_func_vcast_i1_f16_cast_unsigned_uitofp(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xf16> {
    %0 = tensor.empty():tensor<4x32x64xf16>
    //     CHECK: %[[RET:.*]] = arith.uitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xf16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xf16>) round_mode = <rint> cast = <cast_unsigned> -> tensor<4x32x64xf16>
    return %0 : tensor<4x32x64xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_bf16_cast_unsigned_uitofp
func.func @test_func_brc_vcast_i1_bf16_cast_unsigned_uitofp(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xbf16> {
    %0 = tensor.empty():tensor<8x8x8xbf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.uitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xbf16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xbf16>) round_mode = <rint> cast = <cast_unsigned> broadcast = [0,1,2] -> tensor<8x8x8xbf16>
    return %0 : tensor<8x8x8xbf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_bf16_cast_unsigned_uitofp
func.func @test_func_vcast_i1_bf16_cast_unsigned_uitofp(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xbf16> {
    %0 = tensor.empty():tensor<4x32x64xbf16>
    //     CHECK: %[[RET:.*]] = arith.uitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xbf16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xbf16>) round_mode = <rint> cast = <cast_unsigned> -> tensor<4x32x64xbf16>
    return %0 : tensor<4x32x64xbf16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i1_f32_cast_unsigned_uitofp
func.func @test_func_brc_vcast_i1_f32_cast_unsigned_uitofp(%arg0:tensor<8x1x8xi1>) -> tensor<8x8x8xf32> {
    %0 = tensor.empty():tensor<8x8x8xf32>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi1> -> tensor<8x8x8xi1>
    //     CHECK: %[[RET:.*]] = arith.uitofp %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xi1> to tensor<8x8x8xf32>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi1>) outs(%0:tensor<8x8x8xf32>) round_mode = <rint> cast = <cast_unsigned> broadcast = [0,1,2] -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_f32_cast_unsigned_uitofp
func.func @test_func_vcast_i1_f32_cast_unsigned_uitofp(%arg0:tensor<4x32x64xi1>) -> tensor<4x32x64xf32> {
    %0 = tensor.empty():tensor<4x32x64xf32>
    //     CHECK: %[[RET:.*]] = arith.uitofp {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xi1> to tensor<4x32x64xf32>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi1>) outs(%0:tensor<4x32x64xf32>) round_mode = <rint> cast = <cast_unsigned> -> tensor<4x32x64xf32>
    return %0 : tensor<4x32x64xf32>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i16_i8_cast_signed_trunci
func.func @test_func_brc_vcast_i16_i8_cast_signed_trunci(%arg0:tensor<8x1x8xi16>) -> tensor<8x8x8xi8> {
    %0 = tensor.empty():tensor<8x8x8xi8>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi16> -> tensor<8x8x8xi16>
    //     CHECK: %[[RET:.*]] = arith.trunci %[[brc1]] {round_mode = #hivm.round_mode<truncwithoverflow>} : tensor<8x8x8xi16> to tensor<8x8x8xi8>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi16>) outs(%0:tensor<8x8x8xi8>) round_mode = <truncwithoverflow> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_vcast_i16_i8_cast_signed_trunci
func.func @test_func_vcast_i16_i8_cast_signed_trunci(%arg0:tensor<4x32x64xi16>) -> tensor<4x32x64xi8> {
    %0 = tensor.empty():tensor<4x32x64xi8>
    //     CHECK: %[[RET:.*]] = arith.trunci {{.*}} {round_mode = #hivm.round_mode<truncwithoverflow>} : tensor<4x32x64xi16> to tensor<4x32x64xi8>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi16>) outs(%0:tensor<4x32x64xi8>) round_mode = <truncwithoverflow> cast = <cast_signed> -> tensor<4x32x64xi8>
    return %0 : tensor<4x32x64xi8>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i32_i8_cast_signed_trunci
func.func @test_func_brc_vcast_i32_i8_cast_signed_trunci(%arg0:tensor<8x1x8xi32>) -> tensor<8x8x8xi8> {
    %0 = tensor.empty():tensor<8x8x8xi8>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.trunci %[[brc1]] {round_mode = #hivm.round_mode<truncwithoverflow>} : tensor<8x8x8xi32> to tensor<8x8x8xi8>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi32>) outs(%0:tensor<8x8x8xi8>) round_mode = <truncwithoverflow> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_i8_cast_signed_trunci
func.func @test_func_vcast_i32_i8_cast_signed_trunci(%arg0:tensor<4x32x64xi32>) -> tensor<4x32x64xi8> {
    %0 = tensor.empty():tensor<4x32x64xi8>
    //     CHECK: %[[RET:.*]] = arith.trunci {{.*}} {round_mode = #hivm.round_mode<truncwithoverflow>} : tensor<4x32x64xi32> to tensor<4x32x64xi8>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi32>) outs(%0:tensor<4x32x64xi8>) round_mode = <truncwithoverflow> cast = <cast_signed> -> tensor<4x32x64xi8>
    return %0 : tensor<4x32x64xi8>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_i32_i16_cast_signed_trunci
func.func @test_func_brc_vcast_i32_i16_cast_signed_trunci(%arg0:tensor<8x1x8xi32>) -> tensor<8x8x8xi16> {
    %0 = tensor.empty():tensor<8x8x8xi16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xi32> -> tensor<8x8x8xi32>
    //     CHECK: %[[RET:.*]] = arith.trunci %[[brc1]] {round_mode = #hivm.round_mode<truncwithoverflow>} : tensor<8x8x8xi32> to tensor<8x8x8xi16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xi32>) outs(%0:tensor<8x8x8xi16>) round_mode = <truncwithoverflow> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_i16_cast_signed_trunci
func.func @test_func_vcast_i32_i16_cast_signed_trunci(%arg0:tensor<4x32x64xi32>) -> tensor<4x32x64xi16> {
    %0 = tensor.empty():tensor<4x32x64xi16>
    //     CHECK: %[[RET:.*]] = arith.trunci {{.*}} {round_mode = #hivm.round_mode<truncwithoverflow>} : tensor<4x32x64xi32> to tensor<4x32x64xi16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xi32>) outs(%0:tensor<4x32x64xi16>) round_mode = <truncwithoverflow> cast = <cast_signed> -> tensor<4x32x64xi16>
    return %0 : tensor<4x32x64xi16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_f32_f16_cast_signed_truncf
func.func @test_func_brc_vcast_f32_f16_cast_signed_truncf(%arg0:tensor<8x1x8xf32>) -> tensor<8x8x8xf16> {
    %0 = tensor.empty():tensor<8x8x8xf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.truncf %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xf32> to tensor<8x8x8xf16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xf32>) outs(%0:tensor<8x8x8xf16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_f32_f16_cast_signed_truncf
func.func @test_func_vcast_f32_f16_cast_signed_truncf(%arg0:tensor<4x32x64xf32>) -> tensor<4x32x64xf16> {
    %0 = tensor.empty():tensor<4x32x64xf16>
    //     CHECK: %[[RET:.*]] = arith.truncf {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xf32> to tensor<4x32x64xf16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xf32>) outs(%0:tensor<4x32x64xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xf16>
    return %0 : tensor<4x32x64xf16>
}

// CHECK-LABEL: func.func @test_func_brc_vcast_f32_bf16_cast_signed_truncf
func.func @test_func_brc_vcast_f32_bf16_cast_signed_truncf(%arg0:tensor<8x1x8xf32>) -> tensor<8x8x8xbf16> {
    %0 = tensor.empty():tensor<8x8x8xbf16>
    //     CHECK: %[[brc1:.*]] = tt.broadcast %arg0  : tensor<8x1x8xf32> -> tensor<8x8x8xf32>
    //     CHECK: %[[RET:.*]] = arith.truncf %[[brc1]] {round_mode = #hivm.round_mode<rint>} : tensor<8x8x8xf32> to tensor<8x8x8xbf16>
    hivm.hir.vcast ins(%arg0 : tensor<8x1x8xf32>) outs(%0:tensor<8x8x8xbf16>) round_mode = <rint> cast = <cast_signed> broadcast = [0,1,2] -> tensor<8x8x8xbf16>
    return %0 : tensor<8x8x8xbf16>
}

// CHECK-LABEL: func.func @test_func_vcast_f32_bf16_cast_signed_truncf
func.func @test_func_vcast_f32_bf16_cast_signed_truncf(%arg0:tensor<4x32x64xf32>) -> tensor<4x32x64xbf16> {
    %0 = tensor.empty():tensor<4x32x64xbf16>
    //     CHECK: %[[RET:.*]] = arith.truncf {{.*}} {round_mode = #hivm.round_mode<rint>} : tensor<4x32x64xf32> to tensor<4x32x64xbf16>
    hivm.hir.vcast ins(%arg0 : tensor<4x32x64xf32>) outs(%0:tensor<4x32x64xbf16>) round_mode = <rint> cast = <cast_signed> -> tensor<4x32x64xbf16>
    return %0 : tensor<4x32x64xbf16>
}