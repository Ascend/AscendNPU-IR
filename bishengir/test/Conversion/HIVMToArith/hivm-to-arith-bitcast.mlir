// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_bitcast_f8E4M3FN_f8E4M3FNUZ
func.func @test_func_bitcast_f8E4M3FN_f8E4M3FNUZ(%arg0:tensor<8x8x8xf8E4M3FN>) -> tensor<8x8x8xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E4M3FN> -> tensor<8x8x8xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E4M3FN>  -> tensor<8x8x8xf8E4M3FNUZ>
    return %0 : tensor<8x8x8xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E4M3FNUZ_f8E4M3FN
func.func @test_func_bitcast_f8E4M3FNUZ_f8E4M3FN(%arg0:tensor<8x8x8xf8E4M3FNUZ>) -> tensor<8x8x8xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E4M3FNUZ> -> tensor<8x8x8xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E4M3FNUZ>  -> tensor<8x8x8xf8E4M3FN>
    return %0 : tensor<8x8x8xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E4M3FN_f8E5M2
func.func @test_func_bitcast_f8E4M3FN_f8E5M2(%arg0:tensor<8x8x8xf8E4M3FN>) -> tensor<8x8x8xf8E5M2> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E4M3FN> -> tensor<8x8x8xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E4M3FN>  -> tensor<8x8x8xf8E5M2>
    return %0 : tensor<8x8x8xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E5M2_f8E4M3FN
func.func @test_func_bitcast_f8E5M2_f8E4M3FN(%arg0:tensor<8x8x8xf8E5M2>) -> tensor<8x8x8xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E5M2> -> tensor<8x8x8xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E5M2>  -> tensor<8x8x8xf8E4M3FN>
    return %0 : tensor<8x8x8xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E4M3FN_f8E5M2FNUZ
func.func @test_func_bitcast_f8E4M3FN_f8E5M2FNUZ(%arg0:tensor<8x8x8xf8E4M3FN>) -> tensor<8x8x8xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E4M3FN> -> tensor<8x8x8xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E4M3FN>  -> tensor<8x8x8xf8E5M2FNUZ>
    return %0 : tensor<8x8x8xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E5M2FNUZ_f8E4M3FN
func.func @test_func_bitcast_f8E5M2FNUZ_f8E4M3FN(%arg0:tensor<8x8x8xf8E5M2FNUZ>) -> tensor<8x8x8xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E5M2FNUZ> -> tensor<8x8x8xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E5M2FNUZ>  -> tensor<8x8x8xf8E4M3FN>
    return %0 : tensor<8x8x8xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E4M3FN_i8
func.func @test_func_bitcast_f8E4M3FN_i8(%arg0:tensor<8x8x8xf8E4M3FN>) -> tensor<8x8x8xi8> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E4M3FN> -> tensor<8x8x8xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E4M3FN>  -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_i8_f8E4M3FN
func.func @test_func_bitcast_i8_f8E4M3FN(%arg0:tensor<8x8x8xi8>) -> tensor<8x8x8xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xi8> -> tensor<8x8x8xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xi8>  -> tensor<8x8x8xf8E4M3FN>
    return %0 : tensor<8x8x8xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E4M3FNUZ_f8E5M2
func.func @test_func_bitcast_f8E4M3FNUZ_f8E5M2(%arg0:tensor<8x8x8xf8E4M3FNUZ>) -> tensor<8x8x8xf8E5M2> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E4M3FNUZ> -> tensor<8x8x8xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E4M3FNUZ>  -> tensor<8x8x8xf8E5M2>
    return %0 : tensor<8x8x8xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E5M2_f8E4M3FNUZ
func.func @test_func_bitcast_f8E5M2_f8E4M3FNUZ(%arg0:tensor<8x8x8xf8E5M2>) -> tensor<8x8x8xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E5M2> -> tensor<8x8x8xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E5M2>  -> tensor<8x8x8xf8E4M3FNUZ>
    return %0 : tensor<8x8x8xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E4M3FNUZ_f8E5M2FNUZ
func.func @test_func_bitcast_f8E4M3FNUZ_f8E5M2FNUZ(%arg0:tensor<8x8x8xf8E4M3FNUZ>) -> tensor<8x8x8xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E4M3FNUZ> -> tensor<8x8x8xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E4M3FNUZ>  -> tensor<8x8x8xf8E5M2FNUZ>
    return %0 : tensor<8x8x8xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E5M2FNUZ_f8E4M3FNUZ
func.func @test_func_bitcast_f8E5M2FNUZ_f8E4M3FNUZ(%arg0:tensor<8x8x8xf8E5M2FNUZ>) -> tensor<8x8x8xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E5M2FNUZ> -> tensor<8x8x8xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E5M2FNUZ>  -> tensor<8x8x8xf8E4M3FNUZ>
    return %0 : tensor<8x8x8xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E4M3FNUZ_i8
func.func @test_func_bitcast_f8E4M3FNUZ_i8(%arg0:tensor<8x8x8xf8E4M3FNUZ>) -> tensor<8x8x8xi8> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E4M3FNUZ> -> tensor<8x8x8xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E4M3FNUZ>  -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_i8_f8E4M3FNUZ
func.func @test_func_bitcast_i8_f8E4M3FNUZ(%arg0:tensor<8x8x8xi8>) -> tensor<8x8x8xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xi8> -> tensor<8x8x8xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xi8>  -> tensor<8x8x8xf8E4M3FNUZ>
    return %0 : tensor<8x8x8xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E5M2_f8E5M2FNUZ
func.func @test_func_bitcast_f8E5M2_f8E5M2FNUZ(%arg0:tensor<8x8x8xf8E5M2>) -> tensor<8x8x8xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E5M2> -> tensor<8x8x8xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E5M2>  -> tensor<8x8x8xf8E5M2FNUZ>
    return %0 : tensor<8x8x8xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E5M2FNUZ_f8E5M2
func.func @test_func_bitcast_f8E5M2FNUZ_f8E5M2(%arg0:tensor<8x8x8xf8E5M2FNUZ>) -> tensor<8x8x8xf8E5M2> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E5M2FNUZ> -> tensor<8x8x8xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E5M2FNUZ>  -> tensor<8x8x8xf8E5M2>
    return %0 : tensor<8x8x8xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E5M2_i8
func.func @test_func_bitcast_f8E5M2_i8(%arg0:tensor<8x8x8xf8E5M2>) -> tensor<8x8x8xi8> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E5M2> -> tensor<8x8x8xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E5M2>  -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_i8_f8E5M2
func.func @test_func_bitcast_i8_f8E5M2(%arg0:tensor<8x8x8xi8>) -> tensor<8x8x8xf8E5M2> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xi8> -> tensor<8x8x8xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xi8>  -> tensor<8x8x8xf8E5M2>
    return %0 : tensor<8x8x8xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_f8E5M2FNUZ_i8
func.func @test_func_bitcast_f8E5M2FNUZ_i8(%arg0:tensor<8x8x8xf8E5M2FNUZ>) -> tensor<8x8x8xi8> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf8E5M2FNUZ> -> tensor<8x8x8xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf8E5M2FNUZ>  -> tensor<8x8x8xi8>
    return %0 : tensor<8x8x8xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_i8_f8E5M2FNUZ
func.func @test_func_bitcast_i8_f8E5M2FNUZ(%arg0:tensor<8x8x8xi8>) -> tensor<8x8x8xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xi8> -> tensor<8x8x8xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xi8>  -> tensor<8x8x8xf8E5M2FNUZ>
    return %0 : tensor<8x8x8xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_f16_i16
func.func @test_func_bitcast_f16_i16(%arg0:tensor<8x8x8xf16>) -> tensor<8x8x8xi16> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf16> -> tensor<8x8x8xi16>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf16>  -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_bitcast_i16_f16
func.func @test_func_bitcast_i16_f16(%arg0:tensor<8x8x8xi16>) -> tensor<8x8x8xf16> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xi16> -> tensor<8x8x8xf16>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xi16>  -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_bitcast_f16_bf16
func.func @test_func_bitcast_f16_bf16(%arg0:tensor<8x8x8xf16>) -> tensor<8x8x8xbf16> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf16> -> tensor<8x8x8xbf16>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf16>  -> tensor<8x8x8xbf16>
    return %0 : tensor<8x8x8xbf16>
}

// CHECK-LABEL: func.func @test_func_bitcast_bf16_f16
func.func @test_func_bitcast_bf16_f16(%arg0:tensor<8x8x8xbf16>) -> tensor<8x8x8xf16> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xbf16> -> tensor<8x8x8xf16>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xbf16>  -> tensor<8x8x8xf16>
    return %0 : tensor<8x8x8xf16>
}

// CHECK-LABEL: func.func @test_func_bitcast_i16_bf16
func.func @test_func_bitcast_i16_bf16(%arg0:tensor<8x8x8xi16>) -> tensor<8x8x8xbf16> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xi16> -> tensor<8x8x8xbf16>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xi16>  -> tensor<8x8x8xbf16>
    return %0 : tensor<8x8x8xbf16>
}

// CHECK-LABEL: func.func @test_func_bitcast_bf16_i16
func.func @test_func_bitcast_bf16_i16(%arg0:tensor<8x8x8xbf16>) -> tensor<8x8x8xi16> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xbf16> -> tensor<8x8x8xi16>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xbf16>  -> tensor<8x8x8xi16>
    return %0 : tensor<8x8x8xi16>
}

// CHECK-LABEL: func.func @test_func_bitcast_f32_i32
func.func @test_func_bitcast_f32_i32(%arg0:tensor<8x8x8xf32>) -> tensor<8x8x8xi32> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf32> -> tensor<8x8x8xi32>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf32>  -> tensor<8x8x8xi32>
    return %0 : tensor<8x8x8xi32>
}

// CHECK-LABEL: func.func @test_func_bitcast_i32_f32
func.func @test_func_bitcast_i32_f32(%arg0:tensor<8x8x8xi32>) -> tensor<8x8x8xf32> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xi32> -> tensor<8x8x8xf32>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xi32>  -> tensor<8x8x8xf32>
    return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func.func @test_func_bitcast_f64_i64
func.func @test_func_bitcast_f64_i64(%arg0:tensor<8x8x8xf64>) -> tensor<8x8x8xi64> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xf64> -> tensor<8x8x8xi64>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xf64>  -> tensor<8x8x8xi64>
    return %0 : tensor<8x8x8xi64>
}

// CHECK-LABEL: func.func @test_func_bitcast_i64_f64
func.func @test_func_bitcast_i64_f64(%arg0:tensor<8x8x8xi64>) -> tensor<8x8x8xf64> {
    //     CHECK: %[[RET:.*]] = tt.bitcast {{.*}} : tensor<8x8x8xi64> -> tensor<8x8x8xf64>
    %0 = hivm.hir.bitcast %arg0 : tensor<8x8x8xi64>  -> tensor<8x8x8xf64>
    return %0 : tensor<8x8x8xf64>
}