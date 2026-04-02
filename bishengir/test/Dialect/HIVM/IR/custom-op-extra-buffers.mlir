// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-alloc-extra-buffer))" -split-input-file %s | FileCheck %s

// -----

// CustomOp: allocate tmps from extra_buffers_types / extra_buffers_sizes.
func.func @test_custom_op_alloc_extra_buffers_single(%arg0 : memref<4xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<4xf32>
  // CHECK: memref.alloc() : memref<256xf32>
  // CHECK: hivm.hir.custom{{.*}}tmps({{.*}}memref<256xf32>)
  %0 = hivm.hir.custom
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         symbol = "foo_vec1",
         extra_buffers_types = [f32],
         extra_buffers_sizes = [256 : i64] }
       "my_custom_op"
       ins(%arg0 : memref<4xf32>)
       outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

func.func @test_custom_op_alloc_extra_buffers_multi(%arg0 : memref<2x2xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<2x2xf32>
  // CHECK-DAG: memref.alloc() : memref<512xf32>
  // CHECK-DAG: memref.alloc() : memref<128xf16>
  // CHECK: hivm.hir.custom{{.*}}tmps({{.*}}memref<512xf32>{{.*}}memref<128xf16>)
  %0 = hivm.hir.custom
       { hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         symbol = "foo_vec1",
         extra_buffers_types = [f32, f16],
         extra_buffers_sizes = [512 : i64, 128 : i64] }
       "my_custom_op"
       ins(%arg0 : memref<2x2xf32>)
       outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  return
}
