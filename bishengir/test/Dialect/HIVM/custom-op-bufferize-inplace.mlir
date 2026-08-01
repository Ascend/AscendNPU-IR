// RUN: bishengir-opt %s -one-shot-bufferize -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @custom_no_inplace_attr
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: hivm.hir.custom
// CHECK-SAME: ins(%[[INS:[^)]+]] : memref<256xf32>)
// CHECK-SAME: outs(%[[OUT:[^)]+]] : memref<256xf32>)
// CHECK-NOT: outs(%[[INS]]
func.func @custom_no_inplace_attr(%arg0 : memref<256xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %in = bufferization.to_tensor %arg0 restrict writable : memref<256xf32>
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_no_inplace"}
      "user.no_inplace"
      ins(%in : tensor<256xf32>)
      outs(%in : tensor<256xf32>) -> tensor<256xf32>
  return
}

// -----

// CHECK-LABEL: func.func @custom_inplace_single_out
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.custom
// CHECK-SAME: ins(%[[INS:[^)]+]] : memref<256xf32> {inplace_outs = 0 : i32})
// CHECK-SAME: outs(%[[INS]] : memref<256xf32>)
func.func @custom_inplace_single_out(%arg0 : memref<256xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %in = bufferization.to_tensor %arg0 restrict writable : memref<256xf32>
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_inplace"}
      "user.inplace"
      ins(%in : tensor<256xf32> {inplace_outs = 0 : i32})
      outs(%in : tensor<256xf32>) -> tensor<256xf32>
  return
}

// -----

// CHECK-LABEL: func.func @custom_macro_inplace_single_out
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: hivm.hir.custom_macro
// CHECK-SAME: ins(%[[INS:[^)]+]] : memref<256xf32> {inplace_outs = 0 : i32})
// CHECK-SAME: outs(%[[INS]] : memref<256xf32>)
func.func @custom_macro_inplace_single_out(%arg0 : memref<256xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %in = bufferization.to_tensor %arg0 restrict writable : memref<256xf32>
  %0 = hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_macro_inplace"}
      "user.macro_inplace"
      ins(%in : tensor<256xf32> {inplace_outs = 0 : i32})
      outs(%in : tensor<256xf32>) -> tensor<256xf32>
  return
}

// -----

// CHECK-LABEL: func.func @custom_inplace_variadic
// CHECK-SAME: (%[[ARG0:.*]]: memref<128xf32>, %[[ARG1:.*]]: memref<128xf32>)
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: hivm.hir.custom
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : memref<128xf32>, memref<128xf32> {inplace_outs = 0 : i32})
// CHECK-SAME: outs(%[[ARG1]], %[[ALLOC]] : memref<128xf32>, memref<128xf32>)
func.func @custom_inplace_variadic(%arg0 : memref<128xf32>,
                                   %arg1 : memref<128xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %in0 = bufferization.to_tensor %arg0 restrict writable : memref<128xf32>
  %in1 = bufferization.to_tensor %arg1 restrict writable : memref<128xf32>
  %empty = tensor.empty() : tensor<128xf32>
  %0:2 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_variadic_inplace"}
      "user.variadic_inplace"
      ins(%in0, %in1 : tensor<128xf32>, tensor<128xf32> {inplace_outs = 0 : i32})
      outs(%in1, %empty : tensor<128xf32>, tensor<128xf32>)
      -> (tensor<128xf32>, tensor<128xf32>)
  return
}
