// RUN: bishengir-opt %s -one-shot-bufferize -canonicalize \
// RUN:   -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -one-shot-bufferize \
// RUN:   -hacc-append-device-spec=target=Ascend910B3 \
// RUN:   -convert-hivm-to-std -canonicalize -split-input-file \
// RUN:   | FileCheck %s --check-prefix=LOWER

// A no_side_effect CustomOp has no hidden effects, but it still writes its
// declared memref output. Canonicalization must not erase the bufferized op.

// CHECK-LABEL: func.func @keep_declared_output_write
// CHECK: hivm.hir.custom
// CHECK-SAME: no_side_effect
// CHECK-SAME: ins(%[[INPUT:[^)]+]] : memref<256xf32>)
// CHECK-SAME: outs(%[[OUTPUT:[^)]+]] : memref<256xf32>)
// LOWER-LABEL: func.func @keep_declared_output_write
// LOWER: call @pure_kernel
func.func @keep_declared_output_write(%input: memref<256xf32>,
                                     %output: memref<256xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       no_side_effect,
       symbol = "pure_kernel"}
      "user.pure"
      ins(%input : memref<256xf32>)
      outs(%output : memref<256xf32>)
  return
}

// -----

// Exercise the tensor-to-memref transition used by the compile pipeline. Once
// bufferized, the CustomOp has no SSA result keeping it alive; its declared
// output write must do so.

// CHECK-LABEL: func.func @keep_bufferized_output_write
// CHECK: hivm.hir.custom
// CHECK-SAME: no_side_effect
// CHECK-SAME: ins(%[[INPUT:[^)]+]] : memref<256xf32>)
// CHECK-SAME: outs(%[[OUTPUT:[^)]+]] : memref<256xf32>)
// LOWER-LABEL: func.func @keep_bufferized_output_write
// LOWER: call @pure_kernel
func.func @keep_bufferized_output_write(%input: memref<256xf32>,
                                        %output: memref<256xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %input_tensor = bufferization.to_tensor %input restrict
      : memref<256xf32>
  %output_tensor = bufferization.to_tensor %output restrict writable
      : memref<256xf32>
  %result = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       no_side_effect,
       symbol = "pure_kernel"}
      "user.pure"
      ins(%input_tensor : tensor<256xf32>)
      outs(%output_tensor : tensor<256xf32>) -> tensor<256xf32>
  return
}

// -----

// The tensor result of a no_side_effect CustomOp may feed an explicit Store.
// Bufferization must make the CustomOp write and the Store read the same
// buffer; retaining both operations alone is not sufficient.

// CHECK-LABEL: func.func @store_declared_output
// CHECK: hivm.hir.custom
// CHECK-SAME: no_side_effect
// CHECK-SAME: outs(%[[RESULT:.*]] : memref<256xf32, #hivm.address_space<ub>>)
// CHECK: hivm.hir.store
// CHECK-SAME: ins(%[[RESULT]] : memref<256xf32, #hivm.address_space<ub>>)
// CHECK-SAME: outs(%[[OUTPUT:.*]] : memref<256xf32, #hivm.address_space<gm>>)
// LOWER-LABEL: func.func @store_declared_output
// LOWER: call @pure_kernel
// LOWER: call @store_ubuf_to_gm_1d_float
func.func @store_declared_output(
    %input: memref<256xf32, #hivm.address_space<ub>>,
    %init: memref<256xf32, #hivm.address_space<ub>>,
    %output: memref<256xf32, #hivm.address_space<gm>>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %input_tensor = bufferization.to_tensor %input restrict
      : memref<256xf32, #hivm.address_space<ub>>
  %init_tensor = bufferization.to_tensor %init restrict writable
      : memref<256xf32, #hivm.address_space<ub>>
  %result = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       no_side_effect,
       symbol = "pure_kernel"}
      "user.pure"
      ins(%input_tensor : tensor<256xf32>)
      outs(%init_tensor : tensor<256xf32>) -> tensor<256xf32>
  hivm.hir.store ins(%result : tensor<256xf32>)
      outs(%output : memref<256xf32, #hivm.address_space<gm>>)
  return
}
