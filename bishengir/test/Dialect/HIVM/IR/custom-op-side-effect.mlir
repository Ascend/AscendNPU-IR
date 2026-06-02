// RUN: bishengir-opt --canonicalize %s -split-input-file | FileCheck %s

// Test custom op without no_side_effect attribute (default behavior with side effects)
// CHECK-LABEL: @custom_op_with_side_effect
// CHECK: hivm.hir.custom
// CHECK-NOT: no_side_effect
func.func @custom_op_with_side_effect(%arg0 : memref<?xf32>,
                                       %arg1 : tensor<3x3xi64>) {
  %c4_i64 = arith.constant 4 : i64
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  %0 = hivm.hir.custom
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>, symbol = "my_custom_op" }
       "my_custom_op"
       ins(%arg0, %arg1, %c4_i64, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}

// -----

// Test custom op with no_side_effect attribute (pure operation)
// CHECK-LABEL: @custom_op_no_side_effect
// CHECK-NOT: hivm.hir.custom
func.func @custom_op_no_side_effect(%arg0 : memref<?xf32>,
                                     %arg1 : tensor<3x3xi64>) {
  %c4_i64 = arith.constant 4 : i64
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  %0 = hivm.hir.custom
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_side_effect, symbol = "my_custom_op" }
       "my_custom_op"
       ins(%arg0, %arg1, %c4_i64, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}

// -----

// Test custom_macro op without no_side_effect attribute
// CHECK-LABEL: @custom_macro_with_side_effect
// CHECK: hivm.hir.custom_macro
// CHECK-NOT: no_side_effect
func.func @custom_macro_with_side_effect(%arg0 : memref<?xf32>,
                                          %arg1 : tensor<3x3xi64>) {
  %c4_i64 = arith.constant 4 : i64
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  %0 = hivm.hir.custom_macro
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>, symbol = "my_custom_op" }
       "my_custom_op"
       ins(%arg0, %arg1, %c4_i64, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}

// -----

// Test custom_macro op with no_side_effect attribute
// CHECK-LABEL: @custom_macro_no_side_effect
// CHECK-NOT: hivm.hir.custom_macro
func.func @custom_macro_no_side_effect(%arg0 : memref<?xf32>,
                                        %arg1 : tensor<3x3xi64>) {
  %c4_i64 = arith.constant 4 : i64
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  %0 = hivm.hir.custom_macro
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>, no_side_effect, symbol = "my_custom_op" }
       "my_custom_op"
       ins(%arg0, %arg1, %c4_i64, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}
