// RUN: bishengir-opt --normalize-tensor-ops -split-input-file %s | FileCheck %s
// RUN: bishengir-opt --normalize-tensor-ops="skip-aligned-slice=true" -split-input-file %s | FileCheck %s -check-prefix=CHECK-TRITON

func.func @fold_insert_pad_fill(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
  //CHECK-LABEL : @fold_insert_pad_fill
  //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
  //CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
  //CHECK-DAG: %[[padded:.*]] = tensor.pad %[[source]] low[2046] high[0] {
  //CHECK-DAG: ^bb0(%arg1: index):
  //CHECK-DAG: tensor.yield %[[cst_0]] : f32
  //CHECK-DAG: } : tensor<2047xf32> to tensor<4093xf32>
  //CHECK: return %[[padded]] : tensor<4093xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<6140xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<6140xf32>) -> tensor<6140xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %1[2046] [2047] [1] : tensor<2047xf32> into tensor<6140xf32>
    %padded = tensor.pad %inserted_slice low[0] high[-2047] {
    ^bb0(%arg1: index):
        tensor.yield %cst : f32
    } : tensor<6140xf32> to tensor<4093xf32>
    return %padded : tensor<4093xf32>
}

// -----

func.func @fold_insert_pad_cst(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
  //CHECK-LABEL : @fold_insert_pad_cst
  //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
  //CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
  //CHECK-DAG: %[[padded:.*]] = tensor.pad %[[source]] low[2046] high[0] {
  //CHECK-DAG: ^bb0(%arg1: index):
  //CHECK-DAG: tensor.yield %[[cst_0]] : f32
  //CHECK-DAG: } : tensor<2047xf32> to tensor<4093xf32>
  //CHECK: return %[[padded]] : tensor<4093xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_dense = arith.constant dense<0.000000e+00> : tensor<6140xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %cst_dense[2046] [2047] [1] : tensor<2047xf32> into tensor<6140xf32>
    %padded = tensor.pad %inserted_slice low[0] high[-2047] {
    ^bb0(%arg1: index):
        tensor.yield %cst : f32
    } : tensor<6140xf32> to tensor<4093xf32>
    return %padded : tensor<4093xf32>
}

// -----

func.func @fold_insert_pad_rank(%arg0 : tensor<1x1x2047xf32>) -> tensor<1x4093xf32> {
  //CHECK-LABEL : @fold_insert_pad_fill
  //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
  //CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
  //CHECK-DAG: %[[padded:.*]] = tensor.pad %[[source]] low[0, 2046] high[0, 0] {
  //CHECK-DAG: ^bb0(%arg1: index, %arg2: index):
  //CHECK-DAG:   tensor.yield %[[cst_0]] : f32
  //CHECK-DAG: } : tensor<1x2047xf32> to tensor<1x4093xf32>
  //CHECK: return %[[padded]] : tensor<1x4093xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x6140xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x6140xf32>) -> tensor<1x6140xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1],[2]] : tensor<1x1x2047xf32> into tensor<1x2047xf32>
    %inserted_slice = tensor.insert_slice %collapsed into %1[0, 2046] [1, 2047] [1, 1] : tensor<1x2047xf32> into tensor<1x6140xf32>
    %padded = tensor.pad %inserted_slice low[0, 0] high[0, -2047] {
    ^bb0(%arg1: index, %arg2: index):
        tensor.yield %cst : f32
    } : tensor<1x6140xf32> to tensor<1x4093xf32>
    
    return %padded : tensor<1x4093xf32>
}

// -----

func.func @fold_double_pad(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
  //CHECK-LABEL : @fold_double_pad
  //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
  //CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
  //CHECK-DAG: %[[padded:.*]] = tensor.pad %[[source]] low[2046] high[0] {
  //CHECK-DAG: ^bb0(%arg1: index):
  //CHECK-DAG: tensor.yield %[[cst_0]] : f32
  //CHECK-DAG: } : tensor<2047xf32> to tensor<4093xf32>
  //CHECK: return %[[padded]] : tensor<4093xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<6140xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<6140xf32>) -> tensor<6140xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
    %padded = tensor.pad %collapsed low[2046] high[2047] {
    ^bb0(%arg1: index):
        tensor.yield %cst : f32
    } : tensor<2047xf32> to tensor<6140xf32>
    %paddedToo = tensor.pad %padded low[0] high[-2047] {
    ^bb0(%arg1: index):
        tensor.yield %cst : f32
    } : tensor<6140xf32> to tensor<4093xf32>
    return %paddedToo : tensor<4093xf32>
}

// -----

// CHECK-LABEL: func.func @normalize_insert_slice_to_concat_0(
// CHECK-SAME: %[[arg0:.*]]: tensor<2x2xf32>, %[[arg1:.*]]: tensor<2xf32>
// CHECK: %[[expanded:.*]] = tensor.expand_shape
// CHECK: %[[extracted:.*]] = tensor.extract_slice %[[arg0]]{{\[}}1, 0] {{\[}}1, 2] {{\[}}1, 1]
// CHECK: tensor.concat dim(0) %[[expanded]], %[[extracted]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
func.func @normalize_insert_slice_to_concat_0(%arg0: tensor<2x2xf32>, %arg1: tensor<2xf32>) -> tensor<2x2xf32> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [1, 2] : tensor<2xf32> into tensor<1x2xf32>
  %inserted_slice = tensor.insert_slice %expanded into %arg0[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
  return %inserted_slice : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @normalize_insert_slice_to_concat_1(
// CHECK-SAME: %[[arg0:.*]]: tensor<2048x32x192xbf16>, %[[arg1:.*]]: tensor<2048x32x128xbf16>
// CHECK: %[[extracted:.*]] = tensor.extract_slice %[[arg0]]{{\[}}0, 0, 128] {{\[}}2048, 32, 64] {{\[}}1, 1, 1]
// CHECK: tensor.concat dim(2) %[[arg1]], %[[extracted]] : (tensor<2048x32x128xbf16>, tensor<2048x32x64xbf16>) -> tensor<2048x32x192xbf16>
func.func @normalize_insert_slice_to_concat_1(%arg0: tensor<2048x32x192xbf16>, %arg1: tensor<2048x32x128xbf16>) -> tensor<2048x32x192xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %inserted_slice = tensor.insert_slice %arg1 into %arg0[0, 0, 0] [2048, 32, 128] [1, 1, 1] : tensor<2048x32x128xbf16> into tensor<2048x32x192xbf16>
  return %inserted_slice : tensor<2048x32x192xbf16>
}

// -----

// CHECK-LABEL: func.func @normalize_insert_slice_to_concat_2
// CHECK-SAME: %[[arg0:.*]]: tensor<2048x32x192xbf16>, %[[arg1:.*]]: tensor<2048x32x64xbf16>
// CHECK: %[[extracted:.*]] = tensor.extract_slice %[[arg0]]{{\[}}0, 0, 0] {{\[}}2048, 32, 128] {{\[}}1, 1, 1]
// CHECK: tensor.concat dim(2) %[[extracted]], %[[arg1]]
func.func @normalize_insert_slice_to_concat_2(%arg0: tensor<2048x32x192xbf16>, %arg1: tensor<2048x32x64xbf16>) -> tensor<2048x32x192xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %inserted_slice = tensor.insert_slice %arg1 into %arg0[0, 0, 128] [2048, 32, 64] [1, 1, 1] : tensor<2048x32x64xbf16> into tensor<2048x32x192xbf16>
  return %inserted_slice : tensor<2048x32x192xbf16>
}

// -----

// CHECK-LABEL: func.func @not_normalize_insert_slice_to_concat_0
// CHECK-NOT: tensor.concat
func.func @not_normalize_insert_slice_to_concat_0(%arg0: tensor<8x4x64xf16>, %arg1: tensor<8x4x64xf16>) -> tensor<8x4x128xf16> {
  %0 = tensor.empty() : tensor<8x4x64xf16>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg0, %arg1 : tensor<8x4x64xf16>, tensor<8x4x64xf16>) outs(%0 : tensor<8x4x64xf16>) -> tensor<8x4x64xf16>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %1 : tensor<8x4x64xf16>, tensor<8x4x64xf16>) outs(%0 : tensor<8x4x64xf16>) -> tensor<8x4x64xf16>
  %3 = tensor.empty() : tensor<8x4x128xf16>
  %inserted_slice = tensor.insert_slice %1 into %3[0, 0, 0] [8, 4, 64] [1, 1, 2] : tensor<8x4x64xf16> into tensor<8x4x128xf16>
  %inserted_slice_0 = tensor.insert_slice %2 into %inserted_slice[0, 0, 1] [8, 4, 64] [1, 1, 2] : tensor<8x4x64xf16> into tensor<8x4x128xf16>
  return %inserted_slice_0 : tensor<8x4x128xf16>
}

// -----

// CHECK-LABEL: func.func @normalize_last_dim_concat_to_interleave_0
// CHECK: %[[input0:.*]] = hfusion.elemwise_unary
// CHECK: %[[input1:.*]] = hfusion.elemwise_unary
// CHECK: hfusion.interleave %[[input0]], %[[input1]]
func.func @normalize_last_dim_concat_to_interleave_0(%arg0: tensor<4096x1x32x1xbf16>, %arg1: tensor<4096x1x32x1xbf16>) -> tensor<4096x1x32x2xbf16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %0 = tensor.empty() : tensor<4096x1x32x1xbf16>
  %1 = tensor.empty() : tensor<4096x1x32x1xbf16>
  %2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%arg0 : tensor<4096x1x32x1xbf16>) 
                                                              outs(%0 : tensor<4096x1x32x1xbf16>) -> tensor<4096x1x32x1xbf16>
  %3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} ins(%arg1 : tensor<4096x1x32x1xbf16>) 
                                                              outs(%1 : tensor<4096x1x32x1xbf16>) -> tensor<4096x1x32x1xbf16>
  %concat = tensor.concat dim(3) %2, %3 : (tensor<4096x1x32x1xbf16>, tensor<4096x1x32x1xbf16>) -> tensor<4096x1x32x2xbf16>
  return %concat : tensor<4096x1x32x2xbf16>
}

// -----
func.func @fold_tensor_generate_into_fill(%arg0: tensor<?xf32>) -> tensor<?xf32>
// CHECK-LABEL: func.func @fold_tensor_generate_into_fill
// CHECK: %[[cst:.*]] = arith.constant 0.0
// CHECK: %[[dynsize:.*]] = tensor.dim
// CHECK: %[[empty:.*]] = tensor.empty(%[[dynsize]])
// CHECK: %[[fill:.*]] = linalg.fill ins(%[[cst]] : f32) outs(%[[empty]] : tensor<?xf32>) -> tensor<?xf32>
// CHECK: annotation.mark %[[fill]]
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dynsize = tensor.dim %arg0, %c0 : tensor<?xf32>
  %generated = tensor.generate %dynsize {
  ^bb0(%arg1: index):
    tensor.yield %cst : f32
  } : tensor<?xf32>
  annotation.mark %generated {buffer_size_in_byte = 98304 : i64} : tensor<?xf32>
  return %generated : tensor<?xf32>
}

// -----
// CHECK-LABEL: func.func @fold_static_negative_high_pad_0
// CHECK-NOT: tensor.pad
// CHECK: %[[slice:.*]] = tensor.extract_slice {{.*}}{{\[}}0, 0, 0] {{\[}}3072, 12, 12] {{\[}}1, 1, 1]
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[slice]]
func.func @fold_static_negative_high_pad_0(%arg0: tensor<4x768x13x13xf32>) -> tensor<4x768x12x12xf32> {
  %cst = arith.constant 1.70150435 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<3072x12x12xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<4x768x13x13xf32> into tensor<3072x13x13xf32>
  %padded = tensor.pad %collapsed low[0, 0, 0] high[0, -1, -1] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst_0 : f32
  } : tensor<3072x13x13xf32> to tensor<3072x12x12xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%padded, %cst : tensor<3072x12x12xf32>, f32) outs(%0 : tensor<3072x12x12xf32>) -> tensor<3072x12x12xf32>
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape [4, 768, 12, 12] : tensor<3072x12x12xf32> into tensor<4x768x12x12xf32>
  return %expanded : tensor<4x768x12x12xf32>
}

// -----
// CHECK-TRITON-LABEL: triton_avoid_normalize_aligned_insert_slice_0
// CHECK-TRITON-NOT：concat
func.func @triton_avoid_normalize_aligned_insert_slice_0(
    %arg0: tensor<1xf32>, %arg1: tensor<4xf32>, %arg2: tensor<33xf32>) -> (tensor<4xf32>, tensor<33xf32>) {
  %inserted_slice_0 = tensor.insert_slice %arg0 into %arg1[0] [1] [1] : tensor<1xf32> into tensor<4xf32>
  %inserted_slice_1 = tensor.insert_slice %arg0 into %arg2[32] [1] [1] : tensor<1xf32> into tensor<33xf32>
  return %inserted_slice_0, %inserted_slice_1 : tensor<4xf32>, tensor<33xf32>
}

// -----
// CHECK-TRITON-LABEL: triton_normalize_unaligned_insert_slice_0
// CHECK-TRITON：concat
func.func @triton_normalize_unaligned_insert_slice_0(
    %arg0: tensor<1xf32>, %arg1: tensor<4xf32>, %arg2: tensor<33xf32>) -> (tensor<4xf32>, tensor<33xf32>) {
  %inserted_slice_0 = tensor.insert_slice %arg0 into %arg1[1] [1] [1] : tensor<1xf32> into tensor<4xf32>
  %inserted_slice_1 = tensor.insert_slice %arg0 into %arg2[3] [1] [1] : tensor<1xf32> into tensor<33xf32>
  return %inserted_slice_0, %inserted_slice_1 : tensor<4xf32>, tensor<33xf32>
}
