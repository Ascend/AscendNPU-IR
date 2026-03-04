// RUN: bishengir-compile --enable-hfusion-compile=true \
// RUN:                   --target=Ascend950PR_9589 \
// RUN:                   --enable-triton-kernel-compile=true \
// RUN:                   --enable-lir-compile=false \
// RUN:                   --disable-ffts %s | FileCheck %s

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

// Check custom_macro bishengir-compile compilation succeed
// CHECK-LABEL: macro_custom_test
func.func @macro_custom_test(%arg0 : memref<?xf32>,
                             %arg1 : tensor<3x3xi64>) {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  // CHECK: call void @custom_todo
  %0 = hivm.hir.custom_macro
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V> }
       "my_custom_op"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}

// Check builtin without attributes bishengir-compile compilation succeed
// CHECK-LABEL: macro_gather_out_to_ub_test
func.func @macro_gather_out_to_ub_test(%arg0 : memref<?xf32>,
                                       %arg1 : tensor<3x3xi64>) {
  %c4_i64 = arith.constant 4 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %empty = tensor.empty() : tensor<3x3xf32>
  // CHECK: call void @gather_out_to_ub_2d_float_int64_t
  %0 = hivm.hir.custom_macro
       "__builtin_gather_load"
       ins(%arg0, %arg1, %c4_i64, %c0_i32, %c2_i64, %c1_i64, %c2_i32, %c2_i32, %c0_i32, %c0_i32
           : memref<?xf32>, tensor<3x3xi64>, i64, i32, i64, i64, i32, i32, i32, i32)
       outs(%empty : tensor<3x3xf32>) -> tensor<3x3xf32>
  return
}

// Check builtin without attributes bishengir-compile compilation succeed
// CHECK-LABEL: indexselect_test
func.func @indexselect_test(%arg0: memref<?xf32>,
                            %arg1: tensor<16x400xi32>,
                            %arg2: tensor<16x400x32xf32>)
                            attributes {DirectlyUsedGMArgIdxList = [0]} {
  %c0_i32 = arith.constant 0 : i32
  %c32_i32 = arith.constant 32 : i32
  %c4000_i32 = arith.constant 4000 : i32
  %c9000_i64 = arith.constant 9000 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x4x32xf32>
  // CHECK: call void @index_select_3d_float_2d_int32_t
  %1 = hivm.hir.custom
         {extra_attr = "srcStrideLength=3", hivm.vf_mode = #hivm.vf_mode<SIMT>}
         "__builtin_index_select"
         ins(%arg0, %arg1, %c0_i32, %c9000_i64, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c32_i32, %c4000_i32, %c32_i32
         : memref<?xf32>, tensor<16x400xi32>, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32) 
         outs(%0 : tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
  return
}