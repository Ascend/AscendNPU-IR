// RUN: bishengir-opt %s -hfusion-vectorize-ops -canonicalize -cse -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -hfusion-vectorize-ops="for-manual-scope=true" -canonicalize -cse -split-input-file | FileCheck %s --check-prefix=CHECK-MANUAL-SCOPE

// CHECK-LABEL: func.func @test_elemwise_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<64xf32>, %[[arg1:.*]]: tensor<64xf32>, %[[arg2:.*]]: tensor<64xf32>, %[[arg3:.*]]: tensor<64xf32>
// CHECK: %[[read0:.*]] = vector.transfer_read %[[arg0]]
// CHECK: %[[read1:.*]] = vector.transfer_read %[[arg1]]
// CHECK: %[[sub:.*]] = arith.subf %[[read0]], %[[read1]]
// CHECK: %[[exp:.*]] = math.exp %[[sub]]
// CHECK: %[[write0:.*]] = vector.transfer_write %[[exp]]
// CHECK: %[[read2:.*]] = vector.transfer_read %[[arg2]]
// CHECK: %[[mul:.*]] = arith.mulf %[[read2]]
// CHECK: %[[read3:.*]] = vector.transfer_read %[[arg3]]
// CHECK: %[[add:.*]] = arith.addf %[[mul]], %[[read3]]
// CHECK: %[[write1:.*]] = vector.transfer_write %[[add]]
// CHECK: return %[[write0]], %[[write1]]
func.func @test_elemwise_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) 
attributes {vector_mode = "simd"} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg2, %2 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %arg3 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  return %2, %4 : tensor<64xf32>, tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @test_callee_0(
// CHECK-SAME: hivm.vector_function
// CHECK-SAME: no_inline
module {
func.func @test_callee_0(%arg0: tensor<64xf32>) -> tensor<64xf32>
attributes {vector_mode = "simd"} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} 
                              ins(%arg0 : tensor<64xf32>) 
                              outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  return %1 : tensor<64xf32>
}

// CHECK-LABEL: func.func @test_caller_0(
// CHECK: call @test_callee_0
// CHECK-SAME: hivm.vector_function
// CHECK-SAME: no_inline
func.func @test_caller_0(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %0 = func.call @test_callee_0(%arg0) : (tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
}

// -----

// CHECK-LABEL: func.func @test_elemwise_reduce_0(
// CHECK: %[[shapecast:.*]] = vector.shape_cast {{.*}} vector<1x64xf32> to vector<4x1x16xf32>
// CHECK: %[[trunc:.*]] = arith.truncf %[[shapecast]] {{.*}} : vector<4x1x16xf32> to vector<4x1x16xf16>
// CHECK: vector.transfer_write %[[trunc]], {{.*}} vector<4x1x16xf16>, tensor<4x1x16xf16>
// CHECK: vector.multi_reduction {{.*}} [1] : vector<1x64xf32> to vector<1xf32>
// CHECK: vector.transfer_write {{.*}} : vector<1xf32>, tensor<1xf32>
func.func @test_elemwise_reduce_0(%arg0: tensor<8x64x16xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x128xf32>, %arg4: index, %arg5: tensor<3xi64>, %arg6: f32) -> (tensor<8x64x16xf16>, tensor<64xf32>) attributes {hacc.inline = "inline", noinline, outline = true, vector_mode = "simd"} {
  %c0_i32 = arith.constant 0 : i32
  %c10_i32 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  %0:2 = scf.for %arg7 = %c0_i32 to %c10_i32 step %c1_i32 iter_args(%arg8 = %arg0, %arg9 = %arg1) -> (tensor<8x64x16xf16>, tensor<64xf32>)  : i32 {
    %1 = arith.index_cast %arg7 : i32 to index
    %extracted_slice = tensor.extract_slice %arg2[%1] [1] [1] : tensor<64xf32> to tensor<1xf32>
    %extracted_slice_0 = tensor.extract_slice %arg3[%1, 0] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
    %extracted_slice_1 = tensor.extract_slice %arg3[%1, 64] [1, 64] [1, 1] : tensor<64x128xf32> to tensor<1x64xf32>
    %extracted = tensor.extract %extracted_slice[%arg4] : tensor<1xf32>
    %2 = tensor.empty() : tensor<1x64xf32>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%extracted_slice_0, %extracted : tensor<1x64xf32>, f32) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%extracted_slice_1, %extracted : tensor<1x64xf32>, f32) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %5 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%3 : tensor<1x64xf32>) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%4 : tensor<1x64xf32>) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %expanded = tensor.expand_shape %5 [[0], [1, 2, 3]] output_shape [1, 4, 1, 16] : tensor<1x64xf32> into tensor<1x4x1x16xf32>
    %collapsed = tensor.collapse_shape %expanded [[0, 1], [2], [3]] : tensor<1x4x1x16xf32> into tensor<4x1x16xf32>
    %7 = tensor.empty() : tensor<4x1x16xf16>
    %8 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<4x1x16xf32>) outs(%7 : tensor<4x1x16xf16>) -> tensor<4x1x16xf16>
    %inserted_slice = tensor.insert_slice %8 into %arg8[0, %1, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<8x64x16xf16>
    %expanded_2 = tensor.expand_shape %6 [[0], [1, 2, 3]] output_shape [1, 4, 1, 16] : tensor<1x64xf32> into tensor<1x4x1x16xf32>
    %collapsed_3 = tensor.collapse_shape %expanded_2 [[0, 1], [2], [3]] : tensor<1x4x1x16xf32> into tensor<4x1x16xf32>
    %9 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, round_mode = #hfusion.round_mode<rint>} ins(%collapsed_3 : tensor<4x1x16xf32>) outs(%7 : tensor<4x1x16xf16>) -> tensor<4x1x16xf16>
    %inserted_slice_4 = tensor.insert_slice %9 into %inserted_slice[4, %1, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<8x64x16xf16>
    %10 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %6 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %11 = tensor.empty() : tensor<1xf32>
    %reduced = linalg.reduce ins(%10 : tensor<1x64xf32>) outs(%11 : tensor<1xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %12 = arith.addf %in, %init : f32
        linalg.yield %12 : f32
      }
    %inserted_slice_5 = tensor.insert_slice %reduced into %arg9[%1] [1] [1] : tensor<1xf32> into tensor<64xf32>
    scf.yield %inserted_slice_4, %inserted_slice_5 : tensor<8x64x16xf16>, tensor<64xf32>
  }
  return %0#0, %0#1 : tensor<8x64x16xf16>, tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @insert_copy_for_return_func_arg_0
// CHECK-SAME: (%[[arg0:.*]]: tensor<64xf32>, %[[arg1:.*]]: tensor<64xf32>
// CHECK: %[[empty:.*]] = tensor.empty() {__copy_id__ = 1 : i64} : tensor<64xf32>
// CHECK-DAG: %[[src:.*]] = bufferization.to_memref %[[arg1]] : memref<64xf32>
// CHECK-DAG: %[[dst:.*]] = bufferization.to_memref %[[empty]] : memref<64xf32>
// CHECK: %[[read:.*]] = vector.transfer_read %[[src]]
// CHECK: vector.transfer_write %[[read]], %[[dst]]
// CHECK: %[[res:.*]] = bufferization.to_tensor %[[dst]] restrict writable : memref<64xf32>
// CHECK: return {{.*}}, %[[res]]
func.func @insert_copy_for_return_func_arg_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) 
attributes {noinline, outline = true, vector_mode = "simd"} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  return %2, %arg1 : tensor<64xf32>, tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @insert_copy_for_return_multi_func_args
// CHECK-SAME: (%[[arg0:.*]]: tensor<64xf32>, %[[arg1:.*]]: tensor<64xf32>
// CHECK-DAG: %[[empty0:.*]] = tensor.empty() {__copy_id__ = 0 : i64} : tensor<64xf32>
// CHECK-DAG: %[[empty1:.*]] = tensor.empty() {__copy_id__ = 1 : i64} : tensor<64xf32>
// CHECK-DAG: %[[src0:.*]] = bufferization.to_memref %[[arg0]] : memref<64xf32>
// CHECK-DAG: %[[dst0:.*]] = bufferization.to_memref %[[empty0]] : memref<64xf32>
// CHECK-DAG: %[[src1:.*]] = bufferization.to_memref %[[arg1]] : memref<64xf32>
// CHECK-DAG: %[[dst1:.*]] = bufferization.to_memref %[[empty1]] : memref<64xf32>
// CHECK-DAG: %[[res0:.*]] = bufferization.to_tensor %[[dst0]] restrict writable : memref<64xf32>
// CHECK-DAG: %[[res1:.*]] = bufferization.to_tensor %[[dst1]] restrict writable : memref<64xf32>
// CHECK: return %[[res0]], %[[res1]]
func.func @insert_copy_for_return_multi_func_args(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
attributes {noinline, outline = true, vector_mode = "simd"} {
  return %arg0, %arg1 : tensor<64xf32>, tensor<64xf32>
}

// -----

// CHECK-MANUAL-SCOPE-LABEL: func.func @test_return_arg_has_manual_scope_0
// CHECK-MANUAL-SCOPE: %[[res:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<64xf32>
// CHECK-MANUAL-SCOPE: return {{.*}}, %[[res]]
func.func @test_return_arg_has_manual_scope_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) 
attributes {outline = true, vector_mode = "simd"} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  return %2, %arg1 : tensor<64xf32>, tensor<64xf32>
}

// CHECK-MANUAL-SCOPE-LABEL: func.func @test_return_arg_no_manual_scope_0
// CHECK-MANUAL-SCOPE: (%[[arg0:.*]]: tensor<64xf32>, %[[arg1:.*]]: tensor<64xf32>
// CHECK-MANUAL-SCOPE: return {{.*}}, %[[arg1]]
func.func @test_return_arg_no_manual_scope_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%1 : tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  return %2, %arg1 : tensor<64xf32>, tensor<64xf32>
}