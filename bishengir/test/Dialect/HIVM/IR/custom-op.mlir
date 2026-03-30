// UNSUPPORTED: bishengir_published
// RUN: bishengir-compile --enable-hfusion-compile=true \
// RUN:                   --enable-triton-kernel-compile=true \
// RUN:                   --enable-lir-compile=false | FileCheck %s

// Check custom bishengir-compile compilation succeed
// CHECK-LABEL: custom_test
func.func @custom_test(%arg0 : memref<?xf32>,
                                 %arg1 : tensor<3x3xi64>) {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  // CHECK: call void @custom_todo
  %0 = hivm.hir.custom
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.vf_mode = #hivm.vf_mode<SIMD> }
       "my_custom_op"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}

// Check builtin without attributes bishengir-compile compilation succeed
// CHECK-LABEL: gather_out_to_ub_test
func.func @gather_out_to_ub_test(%arg0 : memref<?xf32>,
                                 %arg1 : tensor<3x3xi64>) {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  // CHECK: call void @gather_out_to_ub_2d_float_int64_t
  %0 = hivm.hir.custom
       "__builtin_gather_load"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}
