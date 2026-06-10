// RUN: bishengir-opt -convert-hivm-to-tritongpu="allow-return-value=true" %s -split-input-file -verify-diagnostics | FileCheck %s

// Test vreduce sum operation on float32
// CHECK-LABEL: func.func @test_vreduce_sum_f32
func.func @test_vreduce_sum_f32(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %0 = tensor.empty():tensor<1x16xf32>
    // CHECK: tt.reduce
    // CHECK: arith.addf
    %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xf32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
    return %2 : tensor<16xf32>
}

// Test vreduce sum operation on integer
// CHECK-LABEL: func.func @test_vreduce_sum_i32
func.func @test_vreduce_sum_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.addi
    %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce prod operation on float32
// CHECK-LABEL: func.func @test_vreduce_prod_f32
func.func @test_vreduce_prod_f32(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %0 = tensor.empty():tensor<1x16xf32>
    // CHECK: tt.reduce
    // CHECK: arith.mulf
    %1 = hivm.hir.vreduce <prod> ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xf32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
    return %2 : tensor<16xf32>
}

// Test vreduce prod operation on integer
// CHECK-LABEL: func.func @test_vreduce_prod_i32
func.func @test_vreduce_prod_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.muli
    %1 = hivm.hir.vreduce <prod> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce max operation on float32
// CHECK-LABEL: func.func @test_vreduce_max_f32
func.func @test_vreduce_max_f32(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %0 = tensor.empty():tensor<1x16xf32>
    // CHECK: tt.reduce
    // CHECK: arith.maximumf
    %1 = hivm.hir.vreduce <max> ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xf32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
    return %2 : tensor<16xf32>
}

// Test vreduce max operation on signed integer
// CHECK-LABEL: func.func @test_vreduce_max_signed_i32
func.func @test_vreduce_max_signed_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.maxsi
    %1 = hivm.hir.vreduce <max> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce max operation on unsigned integer
// CHECK-LABEL: func.func @test_vreduce_max_unsigned_i32
func.func @test_vreduce_max_unsigned_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.maxui
    %1 = hivm.hir.vreduce <max> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = true reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce min operation on float32
// CHECK-LABEL: func.func @test_vreduce_min_f32
func.func @test_vreduce_min_f32(%arg0: tensor<16x16xf32>) -> tensor<16xf32> {
    %0 = tensor.empty():tensor<1x16xf32>
    // CHECK: tt.reduce
    // CHECK: arith.minimumf
    %1 = hivm.hir.vreduce <min> ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xf32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xf32> into tensor<16xf32>
    return %2 : tensor<16xf32>
}

// Test vreduce min operation on signed integer
// CHECK-LABEL: func.func @test_vreduce_min_signed_i32
func.func @test_vreduce_min_signed_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.minsi
    %1 = hivm.hir.vreduce <min> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce min operation on unsigned integer
// CHECK-LABEL: func.func @test_vreduce_min_unsigned_i32
func.func @test_vreduce_min_unsigned_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.minui
    %1 = hivm.hir.vreduce <min> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = true reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce andi operation
// CHECK-LABEL: func.func @test_vreduce_andi_i32
func.func @test_vreduce_andi_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.andi
    %1 = hivm.hir.vreduce <andi> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce ori operation
// CHECK-LABEL: func.func @test_vreduce_ori_i32
func.func @test_vreduce_ori_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.ori
    %1 = hivm.hir.vreduce <ori> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce xori operation
// CHECK-LABEL: func.func @test_vreduce_xori_i32
func.func @test_vreduce_xori_i32(%arg0: tensor<16x16xi32>) -> tensor<16xi32> {
    %0 = tensor.empty():tensor<1x16xi32>
    // CHECK: tt.reduce
    // CHECK: arith.xori
    %1 = hivm.hir.vreduce <xori> ins(%arg0 : tensor<16x16xi32>) outs(%0 : tensor<1x16xi32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi32>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi32> into tensor<16xi32>
    return %2 : tensor<16xi32>
}

// Test vreduce any operation
// CHECK-LABEL: func.func @test_vreduce_any_i1
func.func @test_vreduce_any_i1(%arg0: tensor<16x16xi1>) -> tensor<16xi1> {
    %0 = tensor.empty():tensor<1x16xi1>
    // CHECK: tt.reduce
    // CHECK: arith.ori
    %1 = hivm.hir.vreduce <any> ins(%arg0 : tensor<16x16xi1>) outs(%0 : tensor<1x16xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi1>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi1> into tensor<16xi1>
    return %2 : tensor<16xi1>
}

// Test vreduce all operation
// CHECK-LABEL: func.func @test_vreduce_all_i1
func.func @test_vreduce_all_i1(%arg0: tensor<16x16xi1>) -> tensor<16xi1> {
    %0 = tensor.empty():tensor<1x16xi1>
    // CHECK: tt.reduce
    // CHECK: arith.andi
    %1 = hivm.hir.vreduce <all> ins(%arg0 : tensor<16x16xi1>) outs(%0 : tensor<1x16xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xi1>
    %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<1x16xi1> into tensor<16xi1>
    return %2 : tensor<16xi1>
}

// Test vreduce with expand_dims - output shape has dim 1
// CHECK-LABEL: func.func @test_vreduce_sum_with_expand_f32
func.func @test_vreduce_sum_with_expand_f32(%arg0: tensor<16x16xf32>) -> tensor<1x16xf32> {
    %0 = tensor.empty():tensor<1x16xf32>
    // CHECK: tt.reduce
    // CHECK: tt.expand_dims
    %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<1x16xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1x16xf32>
    return %1 : tensor<1x16xf32>
}

// Test vreduce with expand_dims - output shape has dim 1
// CHECK-LABEL: func.func @test_multiDim_vreduce_sum_f32
func.func @test_multiDim_vreduce_sum_f32(%arg0: tensor<16x16xf32>) -> tensor<1x1xf32> {
    %0 = tensor.empty():tensor<1x1xf32>
    // CHECK: tt.reduce
    // CHECK: tt.expand_dims
    %1 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<16x16xf32>) outs(%0 : tensor<1x1xf32>) unsigned_src = false reduce_dims = [0, 1] -> tensor<1x1xf32>
    return %1 : tensor<1x1xf32>
}
