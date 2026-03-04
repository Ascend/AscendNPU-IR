// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E4M3FN_
func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E4M3FN_(%arg0:tensor<6x6xf8E4M3>) -> tensor<6x6xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3> to tensor<6x6xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3>  -> tensor<6x6xf8E4M3FN>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FN>
    return %0 : tensor<6x6xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E4M3_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E4M3_(%arg0:tensor<6x6xf8E4M3FN>) -> tensor<6x6xf8E4M3> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FN> to tensor<6x6xf8E4M3>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FN>  -> tensor<6x6xf8E4M3>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3>
    return %0 : tensor<6x6xf8E4M3>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E4M3FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E4M3FNUZ_(%arg0:tensor<6x6xf8E4M3>) -> tensor<6x6xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3> to tensor<6x6xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3>  -> tensor<6x6xf8E4M3FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FNUZ>
    return %0 : tensor<6x6xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E4M3_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E4M3_(%arg0:tensor<6x6xf8E4M3FNUZ>) -> tensor<6x6xf8E4M3> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FNUZ> to tensor<6x6xf8E4M3>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FNUZ>  -> tensor<6x6xf8E4M3>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3>
    return %0 : tensor<6x6xf8E4M3>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E4M3B11FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E4M3B11FNUZ_(%arg0:tensor<6x6xf8E4M3>) -> tensor<6x6xf8E4M3B11FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3> to tensor<6x6xf8E4M3B11FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3>  -> tensor<6x6xf8E4M3B11FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3B11FNUZ>
    return %0 : tensor<6x6xf8E4M3B11FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E4M3_
func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E4M3_(%arg0:tensor<6x6xf8E4M3B11FNUZ>) -> tensor<6x6xf8E4M3> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3B11FNUZ> to tensor<6x6xf8E4M3>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3B11FNUZ>  -> tensor<6x6xf8E4M3>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3>
    return %0 : tensor<6x6xf8E4M3>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E5M2FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E5M2FNUZ_(%arg0:tensor<6x6xf8E4M3>) -> tensor<6x6xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3> to tensor<6x6xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3>  -> tensor<6x6xf8E5M2FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2FNUZ>
    return %0 : tensor<6x6xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E4M3_
func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E4M3_(%arg0:tensor<6x6xf8E5M2FNUZ>) -> tensor<6x6xf8E4M3> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2FNUZ> to tensor<6x6xf8E4M3>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2FNUZ>  -> tensor<6x6xf8E4M3>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3>
    return %0 : tensor<6x6xf8E4M3>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E5M2_
func.func @test_func_bitcast_arith.bitcast_f8E4M3_f8E5M2_(%arg0:tensor<6x6xf8E4M3>) -> tensor<6x6xf8E5M2> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3> to tensor<6x6xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3>  -> tensor<6x6xf8E5M2>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2>
    return %0 : tensor<6x6xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E4M3_
func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E4M3_(%arg0:tensor<6x6xf8E5M2>) -> tensor<6x6xf8E4M3> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2> to tensor<6x6xf8E4M3>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2>  -> tensor<6x6xf8E4M3>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3>
    return %0 : tensor<6x6xf8E4M3>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3_i8_
func.func @test_func_bitcast_arith.bitcast_f8E4M3_i8_(%arg0:tensor<6x6xf8E4M3>) -> tensor<6x6xi8> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3> to tensor<6x6xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3>  -> tensor<6x6xi8>
    //     CHECK: return %[[RET]] : tensor<6x6xi8>
    return %0 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i8_f8E4M3_
func.func @test_func_bitcast_arith.bitcast_i8_f8E4M3_(%arg0:tensor<6x6xi8>) -> tensor<6x6xf8E4M3> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi8> to tensor<6x6xf8E4M3>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi8>  -> tensor<6x6xf8E4M3>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3>
    return %0 : tensor<6x6xf8E4M3>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E4M3FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E4M3FNUZ_(%arg0:tensor<6x6xf8E4M3FN>) -> tensor<6x6xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FN> to tensor<6x6xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FN>  -> tensor<6x6xf8E4M3FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FNUZ>
    return %0 : tensor<6x6xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E4M3FN_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E4M3FN_(%arg0:tensor<6x6xf8E4M3FNUZ>) -> tensor<6x6xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FNUZ> to tensor<6x6xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FNUZ>  -> tensor<6x6xf8E4M3FN>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FN>
    return %0 : tensor<6x6xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E4M3B11FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E4M3B11FNUZ_(%arg0:tensor<6x6xf8E4M3FN>) -> tensor<6x6xf8E4M3B11FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FN> to tensor<6x6xf8E4M3B11FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FN>  -> tensor<6x6xf8E4M3B11FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3B11FNUZ>
    return %0 : tensor<6x6xf8E4M3B11FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E4M3FN_
func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E4M3FN_(%arg0:tensor<6x6xf8E4M3B11FNUZ>) -> tensor<6x6xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3B11FNUZ> to tensor<6x6xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3B11FNUZ>  -> tensor<6x6xf8E4M3FN>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FN>
    return %0 : tensor<6x6xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E5M2FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E5M2FNUZ_(%arg0:tensor<6x6xf8E4M3FN>) -> tensor<6x6xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FN> to tensor<6x6xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FN>  -> tensor<6x6xf8E5M2FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2FNUZ>
    return %0 : tensor<6x6xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E4M3FN_
func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E4M3FN_(%arg0:tensor<6x6xf8E5M2FNUZ>) -> tensor<6x6xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2FNUZ> to tensor<6x6xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2FNUZ>  -> tensor<6x6xf8E4M3FN>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FN>
    return %0 : tensor<6x6xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E5M2_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_f8E5M2_(%arg0:tensor<6x6xf8E4M3FN>) -> tensor<6x6xf8E5M2> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FN> to tensor<6x6xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FN>  -> tensor<6x6xf8E5M2>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2>
    return %0 : tensor<6x6xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E4M3FN_
func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E4M3FN_(%arg0:tensor<6x6xf8E5M2>) -> tensor<6x6xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2> to tensor<6x6xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2>  -> tensor<6x6xf8E4M3FN>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FN>
    return %0 : tensor<6x6xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_i8_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FN_i8_(%arg0:tensor<6x6xf8E4M3FN>) -> tensor<6x6xi8> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FN> to tensor<6x6xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FN>  -> tensor<6x6xi8>
    //     CHECK: return %[[RET]] : tensor<6x6xi8>
    return %0 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i8_f8E4M3FN_
func.func @test_func_bitcast_arith.bitcast_i8_f8E4M3FN_(%arg0:tensor<6x6xi8>) -> tensor<6x6xf8E4M3FN> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi8> to tensor<6x6xf8E4M3FN>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi8>  -> tensor<6x6xf8E4M3FN>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FN>
    return %0 : tensor<6x6xf8E4M3FN>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E4M3B11FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E4M3B11FNUZ_(%arg0:tensor<6x6xf8E4M3FNUZ>) -> tensor<6x6xf8E4M3B11FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FNUZ> to tensor<6x6xf8E4M3B11FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FNUZ>  -> tensor<6x6xf8E4M3B11FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3B11FNUZ>
    return %0 : tensor<6x6xf8E4M3B11FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E4M3FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E4M3FNUZ_(%arg0:tensor<6x6xf8E4M3B11FNUZ>) -> tensor<6x6xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3B11FNUZ> to tensor<6x6xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3B11FNUZ>  -> tensor<6x6xf8E4M3FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FNUZ>
    return %0 : tensor<6x6xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E5M2FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E5M2FNUZ_(%arg0:tensor<6x6xf8E4M3FNUZ>) -> tensor<6x6xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FNUZ> to tensor<6x6xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FNUZ>  -> tensor<6x6xf8E5M2FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2FNUZ>
    return %0 : tensor<6x6xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E4M3FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E4M3FNUZ_(%arg0:tensor<6x6xf8E5M2FNUZ>) -> tensor<6x6xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2FNUZ> to tensor<6x6xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2FNUZ>  -> tensor<6x6xf8E4M3FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FNUZ>
    return %0 : tensor<6x6xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E5M2_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_f8E5M2_(%arg0:tensor<6x6xf8E4M3FNUZ>) -> tensor<6x6xf8E5M2> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FNUZ> to tensor<6x6xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FNUZ>  -> tensor<6x6xf8E5M2>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2>
    return %0 : tensor<6x6xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E4M3FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E4M3FNUZ_(%arg0:tensor<6x6xf8E5M2>) -> tensor<6x6xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2> to tensor<6x6xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2>  -> tensor<6x6xf8E4M3FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FNUZ>
    return %0 : tensor<6x6xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_i8_
func.func @test_func_bitcast_arith.bitcast_f8E4M3FNUZ_i8_(%arg0:tensor<6x6xf8E4M3FNUZ>) -> tensor<6x6xi8> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3FNUZ> to tensor<6x6xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3FNUZ>  -> tensor<6x6xi8>
    //     CHECK: return %[[RET]] : tensor<6x6xi8>
    return %0 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i8_f8E4M3FNUZ_
func.func @test_func_bitcast_arith.bitcast_i8_f8E4M3FNUZ_(%arg0:tensor<6x6xi8>) -> tensor<6x6xf8E4M3FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi8> to tensor<6x6xf8E4M3FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi8>  -> tensor<6x6xf8E4M3FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3FNUZ>
    return %0 : tensor<6x6xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E5M2FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E5M2FNUZ_(%arg0:tensor<6x6xf8E4M3B11FNUZ>) -> tensor<6x6xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3B11FNUZ> to tensor<6x6xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3B11FNUZ>  -> tensor<6x6xf8E5M2FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2FNUZ>
    return %0 : tensor<6x6xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E4M3B11FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E4M3B11FNUZ_(%arg0:tensor<6x6xf8E5M2FNUZ>) -> tensor<6x6xf8E4M3B11FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2FNUZ> to tensor<6x6xf8E4M3B11FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2FNUZ>  -> tensor<6x6xf8E4M3B11FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3B11FNUZ>
    return %0 : tensor<6x6xf8E4M3B11FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E5M2_
func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_f8E5M2_(%arg0:tensor<6x6xf8E4M3B11FNUZ>) -> tensor<6x6xf8E5M2> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3B11FNUZ> to tensor<6x6xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3B11FNUZ>  -> tensor<6x6xf8E5M2>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2>
    return %0 : tensor<6x6xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E4M3B11FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E4M3B11FNUZ_(%arg0:tensor<6x6xf8E5M2>) -> tensor<6x6xf8E4M3B11FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2> to tensor<6x6xf8E4M3B11FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2>  -> tensor<6x6xf8E4M3B11FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3B11FNUZ>
    return %0 : tensor<6x6xf8E4M3B11FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_i8_
func.func @test_func_bitcast_arith.bitcast_f8E4M3B11FNUZ_i8_(%arg0:tensor<6x6xf8E4M3B11FNUZ>) -> tensor<6x6xi8> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E4M3B11FNUZ> to tensor<6x6xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E4M3B11FNUZ>  -> tensor<6x6xi8>
    //     CHECK: return %[[RET]] : tensor<6x6xi8>
    return %0 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i8_f8E4M3B11FNUZ_
func.func @test_func_bitcast_arith.bitcast_i8_f8E4M3B11FNUZ_(%arg0:tensor<6x6xi8>) -> tensor<6x6xf8E4M3B11FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi8> to tensor<6x6xf8E4M3B11FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi8>  -> tensor<6x6xf8E4M3B11FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E4M3B11FNUZ>
    return %0 : tensor<6x6xf8E4M3B11FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E5M2_
func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_f8E5M2_(%arg0:tensor<6x6xf8E5M2FNUZ>) -> tensor<6x6xf8E5M2> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2FNUZ> to tensor<6x6xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2FNUZ>  -> tensor<6x6xf8E5M2>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2>
    return %0 : tensor<6x6xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E5M2FNUZ_
func.func @test_func_bitcast_arith.bitcast_f8E5M2_f8E5M2FNUZ_(%arg0:tensor<6x6xf8E5M2>) -> tensor<6x6xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2> to tensor<6x6xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2>  -> tensor<6x6xf8E5M2FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2FNUZ>
    return %0 : tensor<6x6xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_i8_
func.func @test_func_bitcast_arith.bitcast_f8E5M2FNUZ_i8_(%arg0:tensor<6x6xf8E5M2FNUZ>) -> tensor<6x6xi8> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2FNUZ> to tensor<6x6xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2FNUZ>  -> tensor<6x6xi8>
    //     CHECK: return %[[RET]] : tensor<6x6xi8>
    return %0 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i8_f8E5M2FNUZ_
func.func @test_func_bitcast_arith.bitcast_i8_f8E5M2FNUZ_(%arg0:tensor<6x6xi8>) -> tensor<6x6xf8E5M2FNUZ> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi8> to tensor<6x6xf8E5M2FNUZ>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi8>  -> tensor<6x6xf8E5M2FNUZ>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2FNUZ>
    return %0 : tensor<6x6xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f8E5M2_i8_
func.func @test_func_bitcast_arith.bitcast_f8E5M2_i8_(%arg0:tensor<6x6xf8E5M2>) -> tensor<6x6xi8> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf8E5M2> to tensor<6x6xi8>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf8E5M2>  -> tensor<6x6xi8>
    //     CHECK: return %[[RET]] : tensor<6x6xi8>
    return %0 : tensor<6x6xi8>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i8_f8E5M2_
func.func @test_func_bitcast_arith.bitcast_i8_f8E5M2_(%arg0:tensor<6x6xi8>) -> tensor<6x6xf8E5M2> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi8> to tensor<6x6xf8E5M2>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi8>  -> tensor<6x6xf8E5M2>
    //     CHECK: return %[[RET]] : tensor<6x6xf8E5M2>
    return %0 : tensor<6x6xf8E5M2>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f16_i16_
func.func @test_func_bitcast_arith.bitcast_f16_i16_(%arg0:tensor<6x6xf16>) -> tensor<6x6xi16> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf16> to tensor<6x6xi16>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf16>  -> tensor<6x6xi16>
    //     CHECK: return %[[RET]] : tensor<6x6xi16>
    return %0 : tensor<6x6xi16>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i16_f16_
func.func @test_func_bitcast_arith.bitcast_i16_f16_(%arg0:tensor<6x6xi16>) -> tensor<6x6xf16> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi16> to tensor<6x6xf16>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi16>  -> tensor<6x6xf16>
    //     CHECK: return %[[RET]] : tensor<6x6xf16>
    return %0 : tensor<6x6xf16>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f32_i32_
func.func @test_func_bitcast_arith.bitcast_f32_i32_(%arg0:tensor<6x6xf32>) -> tensor<6x6xi32> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf32> to tensor<6x6xi32>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf32>  -> tensor<6x6xi32>
    //     CHECK: return %[[RET]] : tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i32_f32_
func.func @test_func_bitcast_arith.bitcast_i32_f32_(%arg0:tensor<6x6xi32>) -> tensor<6x6xf32> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi32> to tensor<6x6xf32>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi32>  -> tensor<6x6xf32>
    //     CHECK: return %[[RET]] : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_f64_i64_
func.func @test_func_bitcast_arith.bitcast_f64_i64_(%arg0:tensor<6x6xf64>) -> tensor<6x6xi64> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xf64> to tensor<6x6xi64>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xf64>  -> tensor<6x6xi64>
    //     CHECK: return %[[RET]] : tensor<6x6xi64>
    return %0 : tensor<6x6xi64>
}

// CHECK-LABEL: func.func @test_func_bitcast_arith.bitcast_i64_f64_
func.func @test_func_bitcast_arith.bitcast_i64_f64_(%arg0:tensor<6x6xi64>) -> tensor<6x6xf64> {
    //     CHECK: %[[RET:.*]] = arith.bitcast {{.*}} : tensor<6x6xi64> to tensor<6x6xf64>
    %0 = hivm.hir.bitcast %arg0 : tensor<6x6xi64>  -> tensor<6x6xf64>
    //     CHECK: return %[[RET]] : tensor<6x6xf64>
    return %0 : tensor<6x6xf64>
}

