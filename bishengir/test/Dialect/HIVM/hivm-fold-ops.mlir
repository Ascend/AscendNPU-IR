// RUN: bishengir-opt %s --canonicalize-ext -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_opt_brc_tensor
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x16xf32>
// CHECK-NOT: hivm.hir.brc
// CHECK-NEXT: return %[[ARG0]]
func.func @test_opt_brc_tensor(%arg0: tensor<1x16xf32>) -> (tensor<1x16xf32>) {
  %0 = tensor.empty() : tensor<1x16xf32>
  %1 = hivm.hir.vbrc ins(%arg0 : tensor<1x16xf32>)
                     outs(%0 : tensor<1x16xf32>)
                     broadcast_dims = [0] -> tensor<1x16xf32>
  return %1 : tensor<1x16xf32>
}

// -----

// CHECK-LABEL: func @test_opt_brc_memref
// CHECK-NOT: hivm.hir.brc
func.func @test_opt_brc_memref(%arg0: memref<1x16xf32>,
                               %arg1: memref<1x16xf32>) {
  hivm.hir.vbrc ins(%arg0 : memref<1x16xf32>)
                  outs(%arg1 : memref<1x16xf32>)
                  broadcast_dims = [0]
  return
}

// -----

// Check not to remove live transpose (memref)
// CHECK-LABEL: func @test_opt_transpose_memref_live
// CHECK: hivm.hir.vtranspose
func.func @test_opt_transpose_memref_live(%arg0: memref<1x16xf32>, %arg1: memref<16x1xf32>) {
  hivm.hir.vtranspose ins(%arg0 : memref<1x16xf32>) outs(%arg1 : memref<16x1xf32>) permutation = [1, 0]
  return
}

// -----

// Check not to remove live transpose (tensor)
// CHECK-LABEL: func @test_opt_transpose_tensor_live
// CHECK: hivm.hir.vtranspose
func.func @test_opt_transpose_tensor_live(%arg0: tensor<1x16xf32>) -> tensor<16x1xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %0 = hivm.hir.vtranspose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0] -> tensor<16x1xf32>
  return %0 : tensor<16x1xf32>
}

// -----

// CHECK-LABEL: func @test_opt_transpose_tensor_unused
// CHECK-NOT: hivm.hir.vtranspose
func.func @test_opt_transpose_tensor_unused(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %0 = hivm.hir.vtranspose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0] -> tensor<16x1xf32>
  return %arg0 : tensor<1x16xf32>
}

// -----

// CHECK-LABEL: func @test_opt_transpose_tensor_chain0
// CHECK: (%[[arg0:.*]]: tensor<1x16xf32>)
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: return %[[arg0]]
func.func @test_opt_transpose_tensor_chain0(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %empty_transpose = tensor.empty() : tensor<1x16xf32>
  %0 = hivm.hir.vtranspose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0] -> tensor<16x1xf32>
  %1 = hivm.hir.vtranspose ins(%0 : tensor<16x1xf32>) outs(%empty_transpose : tensor<1x16xf32>) permutation = [1, 0] -> tensor<1x16xf32>
  return %1 : tensor<1x16xf32>
}

// -----

// CHECK-LABEL: func @test_opt_transpose_tensor_chain1
// CHECK: (%[[arg0:.*]]: tensor<1x16xf32>)
// CHECK: %[[trans:.*]] = hivm.hir.vtranspose ins(%[[arg0]]
// CHECK: return %[[trans]]
func.func @test_opt_transpose_tensor_chain1(%arg0: tensor<1x16xf32>) -> tensor<16x1xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %empty_transpose = tensor.empty() : tensor<1x16xf32>
  %0 = hivm.hir.vtranspose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0] -> tensor<16x1xf32>
  %1 = hivm.hir.vtranspose ins(%0 : tensor<16x1xf32>) outs(%empty_transpose : tensor<1x16xf32>) permutation = [1, 0] -> tensor<1x16xf32>
  %2 = hivm.hir.vtranspose ins(%1 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0] -> tensor<16x1xf32>
  return %2 : tensor<16x1xf32>
}

// -----

// CHECK-LABEL: func @test_opt_transpose_tensor_chain2
// CHECK: (%[[arg0:.*]]: tensor<1x16xf32>)
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: return %[[arg0]]
func.func @test_opt_transpose_tensor_chain2(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
  %empty = tensor.empty() : tensor<16x1xf32>
  %empty_transpose = tensor.empty() : tensor<1x16xf32>
  %0 = hivm.hir.vtranspose ins(%arg0 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0] -> tensor<16x1xf32>
  %1 = hivm.hir.vtranspose ins(%0 : tensor<16x1xf32>) outs(%empty_transpose : tensor<1x16xf32>) permutation = [1, 0] -> tensor<1x16xf32>
  %2 = hivm.hir.vtranspose ins(%1 : tensor<1x16xf32>) outs(%empty : tensor<16x1xf32>) permutation = [1, 0] -> tensor<16x1xf32>
  %3 = hivm.hir.vtranspose ins(%2 : tensor<16x1xf32>) outs(%empty_transpose : tensor<1x16xf32>) permutation = [1, 0] -> tensor<1x16xf32>
  return %3 : tensor<1x16xf32>
}

// -----

// CHECK-LABEL: func @test_opt_transpose_tensor_chain3
// CHECK: (%[[arg0:.*]]: tensor<1x16x8xf32>)
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: return %[[arg0]]
func.func @test_opt_transpose_tensor_chain3(%arg0: tensor<1x16x8xf32>) -> tensor<1x16x8xf32> {
  %empty0 = tensor.empty() : tensor<16x1x8xf32>
  %empty1 = tensor.empty() : tensor<8x1x16xf32>
  %empty2 = tensor.empty() : tensor<1x8x16xf32>
  %empty3 = tensor.empty() : tensor<1x16x8xf32>
  %0 = hivm.hir.vtranspose ins(%arg0 : tensor<1x16x8xf32>) outs(%empty0 : tensor<16x1x8xf32>) permutation = [1, 0, 2] -> tensor<16x1x8xf32>
  %1 = hivm.hir.vtranspose ins(%0 : tensor<16x1x8xf32>) outs(%empty1 : tensor<8x1x16xf32>) permutation = [2, 1, 0] -> tensor<8x1x16xf32>
  %2 = hivm.hir.vtranspose ins(%1 : tensor<8x1x16xf32>) outs(%empty2 : tensor<1x8x16xf32>) permutation = [1, 0, 2] -> tensor<1x8x16xf32>
  %3 = hivm.hir.vtranspose ins(%2 : tensor<1x8x16xf32>) outs(%empty3 : tensor<1x16x8xf32>) permutation = [0, 2, 1] -> tensor<1x16x8xf32>
  return %3 : tensor<1x16x8xf32>
}

// -----

// CHECK-LABEL: func @test_hivm_cast
// CHECK-NOT: memref.cast
func.func @test_hivm_cast() {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %0 = memref.cast %alloc : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%0 : memref<?xf32, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func @test_hivm_reinterpret_cast
// CHECK: %[[REINTERPRET:.*]] = memref.reinterpret_cast
// CHECK-SAME: to memref<16x16xf32, strided<[1, 1]>, #hivm.address_space<ub>>
// CHECK: hivm.hir.vbrc
// CHECK-SAME: outs(%[[REINTERPRET]]
func.func @test_hivm_reinterpret_cast() {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [16, 16], strides: [%c1, %c1] : memref<128xf32, #hivm.address_space<ub>> to memref<16x16xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast : memref<16x16xf32, strided<[?, ?], offset: ?>, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func.func @test_load_pad_fold
func.func @test_load_pad_fold(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
    //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
    //CHECK-DAG: %[[cst_2046:.*]] = arith.constant 2046 : index
    //CHECK-DAG: %[[cst_0i:.*]] = arith.constant 0 : index
    //CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
    //CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4093xf32>
    //CHECK-DAG: %[[padload:.*]] = hivm.hir.load ins(%[[source]] : tensor<2047xf32>) outs(%[[empty]] : tensor<4093xf32>) pad_mode = <PadValue> pad_value = %[[cst_0]]  : f32 left_padding_num = %[[cst_2046]]  : index init_out_buffer = false right_padding_num = %[[cst_0i]] : index may_implicit_transpose_with_last_axis = false -> tensor<4093xf32>
    //CHECK: return %[[padload]] : tensor<4093xf32>
    %right = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2047xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
    %1 = hivm.hir.load ins(%collapsed : tensor<2047xf32>) outs(%0 : tensor<2047xf32>) -> tensor<2047xf32>
    %3 = tensor.empty() : tensor<4093xf32>
    %padded = hivm.hir.vpad ins(%1 : tensor<2047xf32>) outs(%3 : tensor<4093xf32>) low[2046] high[%right] pad_value %cst : f32 -> tensor<4093xf32>
    return %padded : tensor<4093xf32>
}

// -----

// CHECK-LABEL: func.func @test_load_pad_one_side
func.func @test_load_pad_one_side(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
    //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
    //CHECK-DAG: %[[cst_2046:.*]] = arith.constant 2046 : index
    //CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
    //CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4093xf32>
    //CHECK-DAG: %[[padload:.*]] = hivm.hir.load ins(%[[source]] : tensor<2047xf32>) outs(%[[empty]] : tensor<4093xf32>) pad_mode = <PadValue> pad_value = %[[cst_0]]  : f32 init_out_buffer = false right_padding_num = %[[cst_2046]]  : index may_implicit_transpose_with_last_axis = false -> tensor<4093xf32>
    //CHECK: return %[[padload]] : tensor<4093xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2047xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
    %1 = hivm.hir.load ins(%collapsed : tensor<2047xf32>) outs(%0 : tensor<2047xf32>) -> tensor<2047xf32>
    %3 = tensor.empty() : tensor<4093xf32>
    %padded = hivm.hir.vpad ins(%1 : tensor<2047xf32>) outs(%3 : tensor<4093xf32>) low[] high[2046] pad_value %cst : f32 -> tensor<4093xf32>
    return %padded : tensor<4093xf32>
}

// -----

//CHECK-LABEL: func.func @test_load_pad_left_dynamic_right_static
func.func @test_load_pad_left_dynamic_right_static(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
  //CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
  //CHECK-DAG: %[[cst_2046:.*]] = arith.constant 2046 : index
  //CHECK-DAG: %[[cst_0i:.*]] = arith.constant 0 : index
  //CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
  //CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4093xf32>
  //CHECK-DAG: %[[padload:.*]] = hivm.hir.load ins(%[[source]] : tensor<2047xf32>) outs(%[[empty]] : tensor<4093xf32>) pad_mode = <PadValue> pad_value = %[[cst_0]]  : f32 left_padding_num = %[[cst_0i]]  : index init_out_buffer = false right_padding_num = %[[cst_2046]]  : index may_implicit_transpose_with_last_axis = false -> tensor<4093xf32>
  //CHECK: return %[[padload]] : tensor<4093xf32>
  %left = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %src  = tensor.empty() : tensor<2047xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0,1,2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
  %load  = hivm.hir.load ins(%collapsed : tensor<2047xf32>) outs(%src : tensor<2047xf32>) -> tensor<2047xf32>
  %out   = tensor.empty() : tensor<4093xf32>
  %padded = hivm.hir.vpad ins(%load : tensor<2047xf32>) outs(%out : tensor<4093xf32>) low[%left] high[2046] pad_value %cst : f32 -> tensor<4093xf32>
  return %padded : tensor<4093xf32>
}

// -----

//CHECK-LABEL: func.func @test_load_pad_both_dynamic
func.func @test_load_pad_both_dynamic(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
//CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
//CHECK-DAG: %[[cst_1020:.*]] = arith.constant 1020 : index
//CHECK-DAG: %[[cst_1026:.*]] = arith.constant 1026 : index
//CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
//CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4093xf32>
//CHECK-DAG: %[[padload:.*]] = hivm.hir.load ins(%[[source]] : tensor<2047xf32>) outs(%[[empty]] : tensor<4093xf32>) pad_mode = <PadValue> pad_value = %[[cst_0]]  : f32 left_padding_num = %[[cst_1020]]  : index init_out_buffer = false right_padding_num = %[[cst_1026]]  : index may_implicit_transpose_with_last_axis = false -> tensor<4093xf32>
//CHECK: return %[[padload]] : tensor<4093xf32>
%left = arith.constant 1020 : index
%right = arith.constant 1026 : index  
%cst = arith.constant 0.000000e+00 : f32
%src = tensor.empty() : tensor<2047xf32>
%collapsed = tensor.collapse_shape %arg0 [[0,1,2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
%load = hivm.hir.load ins(%collapsed : tensor<2047xf32>) outs(%src : tensor<2047xf32>) -> tensor<2047xf32>
%out = tensor.empty() : tensor<4093xf32>
%padded = hivm.hir.vpad ins(%load : tensor<2047xf32>) outs(%out : tensor<4093xf32>) low[%left] high[%right] pad_value %cst : f32 -> tensor<4093xf32>
return %padded : tensor<4093xf32>
}

// -----

// CHECK-LABEL: func.func @test_load_pad_zero_cases
func.func @test_load_pad_zero_cases(%arg0 : tensor<1x1x2047xf32>) -> tensor<4093xf32> {
// CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[pad_i:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[source:.*]] = tensor.collapse_shape {{.*}}
// CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4093xf32>
// CHECK-DAG: %[[padload:.*]] = hivm.hir.load ins(%[[source]] : tensor<2047xf32>) outs(%[[empty]] : tensor<4093xf32>) pad_mode = <PadValue> pad_value = %[[cst_0]] : f32 left_padding_num = %[[pad_i]] : index init_out_buffer = false right_padding_num = %[[pad_i]] : index may_implicit_transpose_with_last_axis = false -> tensor<4093xf32>
// CHECK: return %[[padload]] : tensor<4093xf32>
%cst_f = arith.constant 0.000000e+00 : f32          
%right_dyn = arith.constant 0 : index             
%src = tensor.collapse_shape %arg0 [[0,1,2]] : tensor<1x1x2047xf32> into tensor<2047xf32>
%empty = tensor.empty() : tensor<2047xf32>
%load = hivm.hir.load ins(%src : tensor<2047xf32>) outs(%empty : tensor<2047xf32>) -> tensor<2047xf32>
%out = tensor.empty() : tensor<4093xf32>
%padded = hivm.hir.vpad ins(%load : tensor<2047xf32>) outs(%out : tensor<4093xf32>) low[0] high[%right_dyn] pad_value %cst_f : f32 -> tensor<4093xf32>
return %padded : tensor<4093xf32>
}

// -----

// CHECK-LABEL: func.func @test_load_already_has_pad
func.func @test_load_already_has_pad(%arg0 : tensor<1x10xf32>) -> tensor<20xf32> {
  // CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[cst_5:.*]] = arith.constant 5 : index
  // CHECK-DAG: %[[src:.*]] = tensor.collapse_shape {{.*}}
  // CHECK-DAG: %[[empty_b:.*]] = tensor.empty() : tensor<20xf32>
  // CHECK-DAG: %[[padload:.*]] = hivm.hir.load ins(%[[src]] : tensor<10xf32>) outs(%[[empty_b]] : tensor<20xf32>) pad_mode = <PadValue> pad_value = %[[cst_0]] : f32 left_padding_num = %[[cst_5]] : index right_padding_num = %[[cst_5]] : index -> tensor<20xf32>
  // CHECK: %[[vpad:.*]] = hivm.hir.vpad ins(%[[padload]] : tensor<20xf32>) outs(%[[out:.*]] : tensor<20xf32>) low[0] high[0] pad_value %[[cst_0]] : f32 -> tensor<20xf32>
  // CHECK-NOT: hivm.hir.load 
  // CHECK: return %[[vpad]] : tensor<20xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %lp  = arith.constant 5 : index
  %rp  = arith.constant 5 : index
  %src = tensor.collapse_shape %arg0 [[0,1]] : tensor<1x10xf32> into tensor<10xf32>
  %empty = tensor.empty() : tensor<20xf32>
  %load  = hivm.hir.load ins(%src : tensor<10xf32>) outs(%empty : tensor<20xf32>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %lp : index right_padding_num = %rp : index -> tensor<20xf32>
  %out = tensor.empty() : tensor<20xf32>
  %vpad = hivm.hir.vpad ins(%load : tensor<20xf32>) outs(%out : tensor<20xf32>) low[0] high[0] pad_value %cst : f32 -> tensor<20xf32>
  return %vpad : tensor<20xf32>
}

// -----

// CHECK-LABEL: func.func @test_load_pad_multi_dim
func.func @test_load_pad_multi_dim(%arg0 : tensor<1x2x5xf32>) -> tensor<2x5xf32> {
  // CHECK-DAG: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[src:.*]] = tensor.collapse_shape {{.*}}
  // CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<2x5xf32>
  // CHECK-DAG: %[[load:.*]] = hivm.hir.load ins(%[[src]] : tensor<2x5xf32>) outs(%[[empty]] : tensor<2x5xf32>) -> tensor<2x5xf32>
  // CHECK: %[[out:.*]] = tensor.empty() : tensor<2x5xf32>
  // CHECK: %[[vpad:.*]] = hivm.hir.vpad ins(%[[load]] : tensor<2x5xf32>) outs(%[[out]] : tensor<2x5xf32>) low[0] high[0] pad_value %cst : f32 -> tensor<2x5xf32>
  // CHECK-NOT: hivm.hir.load 
  // CHECK: return %[[vpad]] : tensor<2x5xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %src = tensor.collapse_shape %arg0 [[0,1],[2]] : tensor<1x2x5xf32> into tensor<2x5xf32>
  %empty = tensor.empty() : tensor<2x5xf32>
  %load = hivm.hir.load ins(%src : tensor<2x5xf32>) outs(%empty : tensor<2x5xf32>) -> tensor<2x5xf32>
  %out = tensor.empty() : tensor<2x5xf32>
  %vpad = hivm.hir.vpad ins(%load : tensor<2x5xf32>) outs(%out : tensor<2x5xf32>) low[0] high[0] pad_value %cst : f32 -> tensor<2x5xf32>
  return %vpad : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: func.func @test_tensor_reduce_with_index
func.func @test_tensor_reduce_with_index(%arg0 : tensor<3x1xf32>,
                                         %arg1 : tensor<3x1xf32>,
                                         %arg2 : tensor<3x1xi32>) -> tensor<3x1xf32> {
  // CHECK-SAME: %[[ARG0:.*]]: tensor<3x1xf32>, %[[ARG1:.*]]: tensor<3x1xf32>
  // CHECK: return %[[ARG0]]
  %0:2 = hivm.hir.vreduce <max_with_index_left> ins(%arg0 : tensor<3x1xf32>)
                                           outs(%arg1, %arg2 : tensor<3x1xf32>, tensor<3x1xi32>)
                                           reduce_dims = [1] -> tensor<3x1xf32>, tensor<3x1xi32>
  return %0#0 : tensor<3x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_tensor_reduce_with_returned_index
func.func @test_tensor_reduce_with_returned_index(%arg0 : tensor<256x1xf16>,
                                         %arg1 : tensor<256x1xf16>,
                                         %arg2 : tensor<256x1xi32>) -> (tensor<256x1xf16>, tensor<256x1xi32>) {
  // CHECK: (%[[arg0:.*]]: tensor<256x1xf16>, %[[arg1:.*]]: tensor<256x1xf16>, %[[arg2:.*]]: tensor<256x1xi32>)
  // CHECK: %[[const0:.*]] = arith.constant 0
  // CHECK: %[[copy:.*]] = hivm.hir.copy ins(%[[arg0]] : tensor<256x1xf16>) outs(%[[arg1]] : tensor<256x1xf16>)
  // CHECK: %[[vbrc:.*]] = hivm.hir.vbrc ins(%[[const0]] : i32) outs(%[[arg2]] : tensor<256x1xi32>)
  // CHECK: return %[[copy]], %[[vbrc]]
  %0,%1 = hivm.hir.vreduce <max_with_index_left> ins(%arg0 : tensor<256x1xf16>)
                                           outs(%arg1, %arg2 : tensor<256x1xf16>, tensor<256x1xi32>)
                                           reduce_dims = [1] -> tensor<256x1xf16>, tensor<256x1xi32>
  return %0,%1 : tensor<256x1xf16>, tensor<256x1xi32>
}

// -----

// CHECK-LABEL: func.func @test_memref_reduce_with_index
func.func @test_memref_reduce_with_index(%arg0 : memref<3x1xf32>,
                                         %arg1 : memref<3x1xf32>,
                                         %arg2 : memref<3x1xi32>) {
  // CHECK: (%[[arg0:.*]]: memref<3x1xf32>, %[[arg1:.*]]: memref<3x1xf32>, %[[arg2:.*]]: memref<3x1xi32>)
  // CHECK: %[[const0:.*]] = arith.constant 0
  // CHECK: hivm.hir.copy ins(%[[arg0]] : memref<3x1xf32>) outs(%[[arg1]] : memref<3x1xf32>)
  // CHECK: hivm.hir.vbrc ins(%[[const0]] : i32) outs(%[[arg2]] : memref<3x1xi32>)
  hivm.hir.vreduce <max_with_index_left> ins(%arg0 : memref<3x1xf32>)
                                    outs(%arg1, %arg2 : memref<3x1xf32>, memref<3x1xi32>)
                                    reduce_dims = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_memref_reduce_fallback
func.func @test_memref_reduce_fallback(
    %src : memref<4x1xf32>,   // 输入张量，形状 4×1，元素类型 f32
    %dst : memref<4x1xf32>) { // 目标张量，形状同上

  // CHECK: (%[[SRC:.*]]: memref<4x1xf32>, %[[DST:.*]]: memref<4x1xf32>)
  // CHECK: hivm.hir.copy ins(%[[SRC]] : memref<4x1xf32>) outs(%[[DST:.*]]: memref<4x1xf32>)
  // CHECK-NOT: hivm.hir.vbrc
  hivm.hir.vreduce <sum> ins(%src : memref<4x1xf32>)
      outs(%dst : memref<4x1xf32>)
      reduce_dims = [1]
  return
}

// -----

// CHECK-LABEL: func.func @test_memref_cumsum
func.func @test_memref_cumsum(%arg0 : memref<3x1xf32>,
                              %arg1 : memref<3x1xf32>) {
  // CHECK: hivm.hir.copy ins(%[[arg0:.*]] : memref<3x1xf32>) outs(%[[arg1:.*]] : memref<3x1xf32>)
  hivm.hir.vcumsum ins(%arg0 : memref<3x1xf32>)
                   outs(%arg1: memref<3x1xf32>)
                   cum_dims = [1] reverse = false
  return
}

// -----

// CHECK-LABEL: func.func @test_memref_cumprod
func.func @test_memref_cumprod(%arg0 : memref<1x3xf32>,
                              %arg1 : memref<1x3xf32>) {
  // CHECK: hivm.hir.copy ins(%[[arg0:.*]] : memref<1x3xf32>) outs(%[[arg1:.*]] : memref<1x3xf32>)
  hivm.hir.vcumsum ins(%arg0 : memref<1x3xf32>)
                   outs(%arg1: memref<1x3xf32>)
                   cum_dims = [0] reverse = false
  return
}

// -----

// CHECK-LABEL: func.func @test_memref_cumsum_one_dim_non_one
func.func @test_memref_cumsum_one_dim_non_one(%src : memref<3x1xf32>,
                                         %dst : memref<3x1xf32>) {
  // CHECK-NOT: hivm.hir.copy
  // CHECK: hivm.hir.vcumsum ins(%[[src:.*]] : memref<3x1xf32>) outs(%[[dst:.*]] : memref<3x1xf32>)
  hivm.hir.vcumsum ins(%src : memref<3x1xf32>)
                   outs(%dst : memref<3x1xf32>)
                   cum_dims = [0] reverse = false
  return
}

// -----

// CHECK-LABEL: func.func @test_memref_cumprod_one_dim_non_one
func.func @test_memref_cumprod_one_dim_non_one(%src : memref<2x5xf32>,
                                               %dst : memref<2x5xf32>) {
  // CHECK-NOT: hivm.hir.copy
  // CHECK: hivm.hir.vcumprod ins(%[[src:.*]] : memref<2x5xf32>) outs(%[[dst:.*]] : memref<2x5xf32>)
  hivm.hir.vcumprod ins(%src : memref<2x5xf32>)
                    outs(%dst : memref<2x5xf32>)
                    cum_dims = [0] reverse = false
  return
}

// -----

// CHECK-LABEL: func.func @test_tensor_cumsum_one_dim
func.func @test_tensor_cumsum_one_dim(
  %src4 : tensor<1x1xf32>,
  %dst4 : tensor<1x1xf32>) -> tensor<1x1xf32> {
  // CHECK-NOT: hivm.hir.vcumsum
  %0 = hivm.hir.vcumsum ins(%src4 : tensor<1x1xf32>)
                        outs(%dst4 : tensor<1x1xf32>)
                        cum_dims = [0] reverse = false -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_tensor_cumprod_one_dim
func.func @test_tensor_cumprod_one_dim(
  %src5 : tensor<1x1xf32>,
  %dst5 : tensor<1x1xf32>) -> tensor<1x1xf32> {
  // CHECK-NOT: hivm.hir.vcumprod
  %0 = hivm.hir.vcumprod ins(%src5 : tensor<1x1xf32>)
                         outs(%dst5 : tensor<1x1xf32>)
                         cum_dims = [0] reverse = false -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
} 


// CHECK-LABEL: func.func @test_tensor_cumsum_non_one_dim
func.func @test_tensor_cumsum_non_one_dim(
  // CHECK: hivm.hir.vcumsum ins(%[[arg0:.*]] : tensor<3x4xf32>) outs(%[[arg1:.*]] : tensor<3x4xf32>)
  %arg0 : tensor<3x4xf32>,
  %arg1 : tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = hivm.hir.vcumsum ins(%arg0 : tensor<3x4xf32>)
                        outs(%arg1 : tensor<3x4xf32>)
                        cum_dims = [0] reverse = false -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func.func @test_tensor_cumprod_non_one_dim
func.func @test_tensor_cumprod_non_one_dim(
  %arg0 : tensor<2x5xf32>,
  %arg1 : tensor<2x5xf32>) -> tensor<2x5xf32> {
// CHECK: hivm.hir.vcumprod ins(%[[arg0:.*]] : tensor<2x5xf32>) outs(%[[arg1:.*]] : tensor<2x5xf32>)
%0 = hivm.hir.vcumprod ins(%arg0 : tensor<2x5xf32>)
                       outs(%arg1 : tensor<2x5xf32>)
                       cum_dims = [0] reverse = false -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: func.func @test_redundant_reduce_sum_init
// CHECK: %[[EMPTY:.*]] = tensor.empty
// CHECK: hivm.hir.vreduce
// CHECK-SAME: %[[EMPTY]]
func.func @test_redundant_reduce_sum_init(%arg0: tensor<128x1xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x1xf32> {
  %cst_0 = arith.constant 0xFF800000 : f32
  %t0 = hivm.hir.vbrc ins(%cst_0: f32) outs(%arg0: tensor<128x1xf32>) -> tensor<128x1xf32>
  %t1 = hivm.hir.vreduce <max> ins(%arg1: tensor<128x128xf32>)
                         outs(%t0: tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
  return %t1: tensor<128x1xf32>
}

// CHECK-LABEL: func.func @test_unredundant_reduce_sum_init	 
// CHECK-NOT: tensor.empty	 
func.func @test_unredundant_reduce_sum_init(%arg0: tensor<128x1xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x1xf32> {	 
  %cst_0 = arith.constant 1.000000e+00 : f32	 
  %t0 = hivm.hir.vbrc ins(%cst_0: f32) outs(%arg0: tensor<128x1xf32>) -> tensor<128x1xf32>	 
  %t1 = hivm.hir.vreduce <max> ins(%arg1: tensor<128x128xf32>)	 
                        outs(%t0: tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>	 
  return %t1: tensor<128x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_redundant_init_for_reduce_with_index_init
// CHECK-NOT: hivm.hir.vbrc
// CHECK: hivm.hir.vreduce
func.func @test_redundant_init_for_reduce_with_index_init(%arg0: tensor<128x3xf32>, %arg1: tensor<128x1xf32>, %arg2: tensor<128x1xi32>) -> tensor<128x1xf32> {
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 2147483647 : i32
  %t0 = hivm.hir.vbrc ins(%cst_0: f32) outs(%arg1: tensor<128x1xf32>) -> tensor<128x1xf32>
  %t1 = hivm.hir.vbrc ins(%cst_1: i32) outs(%arg2: tensor<128x1xi32>) -> tensor<128x1xi32>
  %0:2 = hivm.hir.vreduce <max_with_index_left> ins(%arg0 : tensor<128x3xf32>)
                                           outs(%t0, %t1 : tensor<128x1xf32>, tensor<128x1xi32>)
                                           reduce_dims = [1] -> tensor<128x1xf32>, tensor<128x1xi32>
  return %0#0 : tensor<128x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_redundant_init_and_reduce_with_index
// CHECK-NOT: hivm.hir.vbrc
// CHECK-NOT: hivm.hir.vreduce
func.func @test_redundant_init_and_reduce_with_index(%arg0: tensor<128x1xf32>, %arg1: tensor<128x1xf32>, %arg2: tensor<128x1xi32>) -> tensor<128x1xf32> {
  %cst_0 = arith.constant 0x7F800000 : f32
  %cst_1 = arith.constant -1 : i32
  %t0 = hivm.hir.vbrc ins(%cst_0: f32) outs(%arg1: tensor<128x1xf32>) -> tensor<128x1xf32>
  %t1 = hivm.hir.vbrc ins(%cst_1: i32) outs(%arg2: tensor<128x1xi32>) -> tensor<128x1xi32>
  %0:2 = hivm.hir.vreduce <min_with_index_right> ins(%arg0 : tensor<128x1xf32>)
                                           outs(%t0, %t1 : tensor<128x1xf32>, tensor<128x1xi32>)
                                           reduce_dims = [1] -> tensor<128x1xf32>, tensor<128x1xi32>
  return %0#0 : tensor<128x1xf32>
}