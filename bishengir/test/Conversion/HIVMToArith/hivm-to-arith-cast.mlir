// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_vcast_i8_i16_cast_signed_extsi
func.func @test_func_vcast_i8_i16_cast_signed_extsi(%arg0:tensor<6x6xi8>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.extsi {{.*}} : tensor<6x6xi8> to tensor<6x6xi16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi8>) outs(%0:tensor<6x6xi16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi16>
    return %1 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_i8_i32_cast_signed_extsi
func.func @test_func_vcast_i8_i32_cast_signed_extsi(%arg0:tensor<6x6xi8>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.extsi {{.*}} : tensor<6x6xi8> to tensor<6x6xi32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi8>) outs(%0:tensor<6x6xi32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi32>
    return %1 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_i16_i32_cast_signed_extsi
func.func @test_func_vcast_i16_i32_cast_signed_extsi(%arg0:tensor<6x6xi16>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.extsi {{.*}} : tensor<6x6xi16> to tensor<6x6xi32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi16>) outs(%0:tensor<6x6xi32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi32>
    return %1 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_i64_cast_signed_extsi
func.func @test_func_vcast_i32_i64_cast_signed_extsi(%arg0:tensor<6x6xi32>) -> tensor<6x6xi64> {
    %0 = tensor.empty():tensor<6x6xi64>
    //     CHECK: %[[RET:.*]] = arith.extsi {{.*}} : tensor<6x6xi32> to tensor<6x6xi64>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi32>) outs(%0:tensor<6x6xi64>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi64>
    return %1 : tensor<6x6xi64>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i8_cast_signed_extsi
func.func @test_func_vcast_i1_i8_cast_signed_extsi(%arg0:tensor<6x6xi1>) -> tensor<6x6xi8> {
    %0 = tensor.empty():tensor<6x6xi8>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} : tensor<6x6xi1> to tensor<6x6xi8>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xi8>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi8>
    return %1 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i16_cast_signed_extsi
func.func @test_func_vcast_i1_i16_cast_signed_extsi(%arg0:tensor<6x6xi1>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} : tensor<6x6xi1> to tensor<6x6xi16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xi16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi16>
    return %1 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i32_cast_signed_extsi
func.func @test_func_vcast_i1_i32_cast_signed_extsi(%arg0:tensor<6x6xi1>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} : tensor<6x6xi1> to tensor<6x6xi32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xi32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi32>
    return %1 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i16_cast_signed_extui
func.func @test_func_vcast_i1_i16_cast_signed_extui(%arg0:tensor<6x6xi1>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} : tensor<6x6xi1> to tensor<6x6xi16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xi16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi16>
    return %1 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_i32_cast_signed_extui
func.func @test_func_vcast_i1_i32_cast_signed_extui(%arg0:tensor<6x6xi1>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.extui {{.*}} : tensor<6x6xi1> to tensor<6x6xi32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xi32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xi32>
    return %1 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_f16_f32_cast_signed_extf
func.func @test_func_vcast_f16_f32_cast_signed_extf(%arg0:tensor<6x6xf16>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.extf {{.*}} : tensor<6x6xf16> to tensor<6x6xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xf16>) outs(%0:tensor<6x6xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf32>
    return %1 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_bf16_f32_cast_signed_extf
func.func @test_func_vcast_bf16_f32_cast_signed_extf(%arg0:tensor<6x6xbf16>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.extf {{.*}} : tensor<6x6xbf16> to tensor<6x6xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xbf16>) outs(%0:tensor<6x6xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf32>
    return %1 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_f16_i16_cast_signed_fptosi
func.func @test_func_vcast_f16_i16_cast_signed_fptosi(%arg0:tensor<6x6xf16>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} : tensor<6x6xf16> to tensor<6x6xi16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xf16>) outs(%0:tensor<6x6xi16>) round_mode = <trunc> cast = <cast_signed> -> tensor<6x6xi16>
    return %1 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_f16_i32_cast_signed_fptosi
func.func @test_func_vcast_f16_i32_cast_signed_fptosi(%arg0:tensor<6x6xf16>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} : tensor<6x6xf16> to tensor<6x6xi32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xf16>) outs(%0:tensor<6x6xi32>) round_mode = <trunc> cast = <cast_signed> -> tensor<6x6xi32>
    return %1 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_bf16_i32_cast_signed_fptosi
func.func @test_func_vcast_bf16_i32_cast_signed_fptosi(%arg0:tensor<6x6xbf16>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} : tensor<6x6xbf16> to tensor<6x6xi32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xbf16>) outs(%0:tensor<6x6xi32>) round_mode = <trunc> cast = <cast_signed> -> tensor<6x6xi32>
    return %1 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_f32_i32_cast_signed_fptosi
func.func @test_func_vcast_f32_i32_cast_signed_fptosi(%arg0:tensor<6x6xf32>) -> tensor<6x6xi32> {
    %0 = tensor.empty():tensor<6x6xi32>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} : tensor<6x6xf32> to tensor<6x6xi32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xf32>) outs(%0:tensor<6x6xi32>) round_mode = <trunc> cast = <cast_signed> -> tensor<6x6xi32>
    return %1 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_vcast_f32_i64_cast_signed_fptosi
func.func @test_func_vcast_f32_i64_cast_signed_fptosi(%arg0:tensor<6x6xf32>) -> tensor<6x6xi64> {
    %0 = tensor.empty():tensor<6x6xi64>
    //     CHECK: %[[RET:.*]] = arith.fptosi {{.*}} : tensor<6x6xf32> to tensor<6x6xi64>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xf32>) outs(%0:tensor<6x6xi64>) round_mode = <trunc> cast = <cast_signed> -> tensor<6x6xi64>
    return %1 : tensor<6x6xi64>
}

// CHECK-LABEL: func.func @test_func_vcast_i8_f16_cast_signed_sitofp
func.func @test_func_vcast_i8_f16_cast_signed_sitofp(%arg0:tensor<6x6xi8>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} : tensor<6x6xi8> to tensor<6x6xf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi8>) outs(%0:tensor<6x6xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf16>
    return %1 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i16_f16_cast_signed_sitofp
func.func @test_func_vcast_i16_f16_cast_signed_sitofp(%arg0:tensor<6x6xi16>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} : tensor<6x6xi16> to tensor<6x6xf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi16>) outs(%0:tensor<6x6xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf16>
    return %1 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i16_f32_cast_signed_sitofp
func.func @test_func_vcast_i16_f32_cast_signed_sitofp(%arg0:tensor<6x6xi16>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} : tensor<6x6xi16> to tensor<6x6xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi16>) outs(%0:tensor<6x6xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf32>
    return %1 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_f32_cast_signed_sitofp
func.func @test_func_vcast_i32_f32_cast_signed_sitofp(%arg0:tensor<6x6xi32>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} : tensor<6x6xi32> to tensor<6x6xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi32>) outs(%0:tensor<6x6xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf32>
    return %1 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i64_f32_cast_signed_sitofp
func.func @test_func_vcast_i64_f32_cast_signed_sitofp(%arg0:tensor<6x6xi64>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} : tensor<6x6xi64> to tensor<6x6xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi64>) outs(%0:tensor<6x6xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf32>
    return %1 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_f16_cast_signed_sitofp
func.func @test_func_vcast_i1_f16_cast_signed_sitofp(%arg0:tensor<6x6xi1>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} : tensor<6x6xi1> to tensor<6x6xf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf16>
    return %1 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_bf16_cast_signed_sitofp
func.func @test_func_vcast_i1_bf16_cast_signed_sitofp(%arg0:tensor<6x6xi1>) -> tensor<6x6xbf16> {
    %0 = tensor.empty():tensor<6x6xbf16>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} : tensor<6x6xi1> to tensor<6x6xbf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xbf16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xbf16>
    return %1 : tensor<6x6xbf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_f32_cast_signed_sitofp
func.func @test_func_vcast_i1_f32_cast_signed_sitofp(%arg0:tensor<6x6xi1>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.sitofp {{.*}} : tensor<6x6xi1> to tensor<6x6xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf32>
    return %1 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_f16_cast_unsigned_uitofp
func.func @test_func_vcast_i1_f16_cast_unsigned_uitofp(%arg0:tensor<6x6xi1>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.uitofp {{.*}} : tensor<6x6xi1> to tensor<6x6xf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xf16>) round_mode = <rint> cast = <cast_unsigned> -> tensor<6x6xf16>
    return %1 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_bf16_cast_unsigned_uitofp
func.func @test_func_vcast_i1_bf16_cast_unsigned_uitofp(%arg0:tensor<6x6xi1>) -> tensor<6x6xbf16> {
    %0 = tensor.empty():tensor<6x6xbf16>
    //     CHECK: %[[RET:.*]] = arith.uitofp {{.*}} : tensor<6x6xi1> to tensor<6x6xbf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xbf16>) round_mode = <rint> cast = <cast_unsigned> -> tensor<6x6xbf16>
    return %1 : tensor<6x6xbf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i1_f32_cast_unsigned_uitofp
func.func @test_func_vcast_i1_f32_cast_unsigned_uitofp(%arg0:tensor<6x6xi1>) -> tensor<6x6xf32> {
    %0 = tensor.empty():tensor<6x6xf32>
    //     CHECK: %[[RET:.*]] = arith.uitofp {{.*}} : tensor<6x6xi1> to tensor<6x6xf32>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi1>) outs(%0:tensor<6x6xf32>) round_mode = <rint> cast = <cast_unsigned> -> tensor<6x6xf32>
    return %1 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_vcast_i8_f16_cast_unsigned_uitofp
func.func @test_func_vcast_i8_f16_cast_unsigned_uitofp(%arg0:tensor<6x6xi8>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.uitofp {{.*}} : tensor<6x6xi8> to tensor<6x6xf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi8>) outs(%0:tensor<6x6xf16>) round_mode = <rint> cast = <cast_unsigned> -> tensor<6x6xf16>
    return %1 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_i16_i8_cast_signed_trunci
func.func @test_func_vcast_i16_i8_cast_signed_trunci(%arg0:tensor<6x6xi16>) -> tensor<6x6xi8> {
    %0 = tensor.empty():tensor<6x6xi8>
    //     CHECK: %[[RET:.*]] = arith.trunci {{.*}} : tensor<6x6xi16> to tensor<6x6xi8>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi16>) outs(%0:tensor<6x6xi8>) round_mode = <truncwithoverflow> cast = <cast_signed> -> tensor<6x6xi8>
    return %1 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_i8_cast_signed_trunci
func.func @test_func_vcast_i32_i8_cast_signed_trunci(%arg0:tensor<6x6xi32>) -> tensor<6x6xi8> {
    %0 = tensor.empty():tensor<6x6xi8>
    //     CHECK: %[[RET:.*]] = arith.trunci {{.*}} : tensor<6x6xi32> to tensor<6x6xi8>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi32>) outs(%0:tensor<6x6xi8>) round_mode = <truncwithoverflow> cast = <cast_signed> -> tensor<6x6xi8>
    return %1 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_vcast_i32_i16_cast_signed_trunci
func.func @test_func_vcast_i32_i16_cast_signed_trunci(%arg0:tensor<6x6xi32>) -> tensor<6x6xi16> {
    %0 = tensor.empty():tensor<6x6xi16>
    //     CHECK: %[[RET:.*]] = arith.trunci {{.*}} : tensor<6x6xi32> to tensor<6x6xi16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xi32>) outs(%0:tensor<6x6xi16>) round_mode = <truncwithoverflow> cast = <cast_signed> -> tensor<6x6xi16>
    return %1 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_vcast_f32_f16_cast_signed_truncf
func.func @test_func_vcast_f32_f16_cast_signed_truncf(%arg0:tensor<6x6xf32>) -> tensor<6x6xf16> {
    %0 = tensor.empty():tensor<6x6xf16>
    //     CHECK: %[[RET:.*]] = arith.truncf {{.*}} : tensor<6x6xf32> to tensor<6x6xf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xf32>) outs(%0:tensor<6x6xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xf16>
    return %1 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_vcast_f32_bf16_cast_signed_truncf
func.func @test_func_vcast_f32_bf16_cast_signed_truncf(%arg0:tensor<6x6xf32>) -> tensor<6x6xbf16> {
    %0 = tensor.empty():tensor<6x6xbf16>
    //     CHECK: %[[RET:.*]] = arith.truncf {{.*}} : tensor<6x6xf32> to tensor<6x6xbf16>
    %1 = hivm.hir.vcast ins(%arg0 : tensor<6x6xf32>) outs(%0:tensor<6x6xbf16>) round_mode = <rint> cast = <cast_signed> -> tensor<6x6xbf16>
    return %1 : tensor<6x6xbf16>
}

