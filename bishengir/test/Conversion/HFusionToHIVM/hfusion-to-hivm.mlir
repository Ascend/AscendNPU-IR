// RUN: bishengir-opt -convert-hfusion-to-hivm %s -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt -convert-to-hivm-pipeline %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_elemwise_unary_ops
func.func @test_elemwise_unary_ops(
  %src : memref<6x6xf32>, %dst : memref<6x6xf32>) {
  //     CHECK: hivm.hir.vexp
  linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vabs
  linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vln
  linalg.elemwise_unary {fun = #linalg.unary_fn<log>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_elemwise_binary_ops
func.func @test_elemwise_binary_ops(
  %src1 : memref<6x6xf32>, %src2 : memref<6x6xf32>, %dst : memref<6x6xf32>) {
  //     CHECK: hivm.hir.vadd
  linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vmul
  linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vsub
  linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} 
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vdiv
  linalg.elemwise_binary {fun = #linalg.binary_fn<div>} 
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vmax
  linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} 
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vmin
  linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} 
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vmax
  linalg.elemwise_binary {fun = #linalg.binary_fn<max_unsigned>} 
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vmin
  linalg.elemwise_binary {fun = #linalg.binary_fn<min_unsigned>} 
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_ops
func.func @test_hfusion_elemwise_unary_ops(
  %src : memref<6x6xf32>, %dst : memref<6x6xf32>,
  %srci : memref<6x6xi32>, %dsti : memref<6x6xi32>) {
  //     CHECK: hivm.hir.vrelu
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vsqrt
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vrsqrt
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vrec
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} 
    ins(%src : memref<6x6xf32>) 
    outs(%dst : memref<6x6xf32>)

  //     CHECK: hivm.hir.vnot
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} 
    ins(%srci : memref<6x6xi32>) 
    outs(%dsti : memref<6x6xi32>)

  //     CHECK: hivm.hir.vtanh
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>}
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vsin
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>}
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xf32>)
  //     CHECK: hivm.hir.vcos
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>}
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xf32>)

  //     CHECK: hivm.hir.vabs
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>}
    ins(%srci : memref<6x6xi32>)
    outs(%dsti : memref<6x6xi32>)

  //     CHECK: hivm.hir.verf
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>}
    ins(%src : memref<6x6xf32>)
    outs(%dst : memref<6x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_elemwise_ops_fix_operand
func.func @test_elemwise_ops_fix_operand(
  %src0 : f32, %src1 : f32, %src2 : tensor<6x6xf32>, %dst : tensor<6x6xf32>,
  %src3 : i16, %src4 : i16, %src5 : memref<6x6xi16>, %dst1 : memref<6x6xi16>) -> tensor<6x6xf32> {
  // CHECK: tensor.empty
  // CHECK: hivm.hir.vbrc
  // CHECK: hivm.hir.vmul
  %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
  ins(%src0, %src0 : f32, f32) 
  outs(%dst : tensor<6x6xf32>) -> tensor<6x6xf32>
  // CHECK: hivm.hir.vadd
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
  ins(%src1, %0 : f32, tensor<6x6xf32>) 
  outs(%dst : tensor<6x6xf32>) -> tensor<6x6xf32>
  // CHECK: memref.alloc
  // CHECK: hivm.hir.vbrc   
  // CHECK: hivm.hir.vand
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
    ins(%src4, %src5 : i16, memref<6x6xi16>) 
    outs(%dst1 : memref<6x6xi16>)
  // CHECK: memref.alloc
  // CHECK: hivm.hir.vbrc  
  // CHECK: memref.alloc
  // CHECK: hivm.hir.vbrc  
  // CHECK: hivm.hir.vor
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
    ins(%src3, %src4 : i16, i16)
    outs(%dst1 : memref<6x6xi16>)
  // CHECK: tensor.empty
  // CHECK: hivm.hir.vbrc   
  // CHECK: hivm.hir.vexp
  %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} 
    ins(%src1 : f32) 
    outs(%dst : tensor<6x6xf32>) -> tensor<6x6xf32>
  // CHECK: hivm.hir.vdiv
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} 
  ins(%1, %2 : tensor<6x6xf32>, tensor<6x6xf32>) 
  outs(%dst : tensor<6x6xf32>) -> tensor<6x6xf32>
  return %3 : tensor<6x6xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_absi
func.func @test_hfusion_absi(
  %src : memref<6x6xi32>, %dst : memref<6x6xi32>) {
  // CHECK: hivm.hir.vabs {{.*}} memref<6x6xi32>
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>}
    ins(%src : memref<6x6xi32>)
    outs(%dst : memref<6x6xi32>)
  return
}

// CHECK-LABEL: func.func @test_hfusion_not_op
func.func @test_hfusion_not_op(
  %src : memref<6x6xi1>, %dst : memref<6x6xi1>) {
  //     CHECK: hivm.hir.vnot
  hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} 
    ins(%src : memref<6x6xi1>)
    outs(%dst : memref<6x6xi1>)
  return
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_binary_ops
func.func @test_hfusion_elemwise_binary_ops(
  %src1 : memref<6x6xi16>, %src2 : memref<6x6xi16>, %dst : memref<6x6xi16>,
  %src3 : memref<6x6xf16>, %src4 : memref<6x6xf16>, %dst1 : memref<6x6xf16>) {
  //     CHECK: hivm.hir.vor
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} 
    ins(%src1, %src2 : memref<6x6xi16>, memref<6x6xi16>)
    outs(%dst : memref<6x6xi16>)
  //     CHECK: hivm.hir.vand
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} 
    ins(%src1, %src2 : memref<6x6xi16>, memref<6x6xi16>)
    outs(%dst : memref<6x6xi16>)
  //     CHECK: hivm.hir.vmax
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
    ins(%src3, %src4 : memref<6x6xf16>, memref<6x6xf16>)
    outs(%dst1 : memref<6x6xf16>)
  //     CHECK: hivm.hir.vmin
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
    ins(%src3, %src4 : memref<6x6xf16>, memref<6x6xf16>)
    outs(%dst1 : memref<6x6xf16>)
  // CHECK: hivm.hir.vmod
  hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} 
    ins(%src3, %src4 : memref<6x6xf16>, memref<6x6xf16>)
    outs(%dst1 : memref<6x6xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_extended_ops
func.func @test_hfusion_elemwise_extended_ops(
  %src1 :  tensor<6xi32>, %src2 :  tensor<6xi32>, %dst :  tensor<6xi32>) -> tensor<6xi32> {
  //     CHECK: hivm.hir.vmulext
  %low, %high = hfusion.mulext %src1, %src2 : tensor<6xi32>
  return %low : tensor<6xi32>
}
// -----

// CHECK-LABEL: func @normal_tensor_copy
func.func @normal_tensor_copy(%input : tensor<4x8xf32>, %output : tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: hivm.hir.copy
  %0 = linalg.copy ins(%input: tensor<4x8xf32>) outs(%output: tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0: tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @normal_memref_copy
func.func @normal_memref_copy() {
  %input = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>
  %output = memref.alloc() : memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>
  // CHECK: hivm.hir.copy
  linalg.copy ins(%input: memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<gm>>)
    outs(%output: memref<16x16xf16, strided<[16, 1]>, #hivm.address_space<ub>>)
  return
}

// -----

// CHECK-LABEL: func.func @test_matmul_ops
func.func @test_matmul_ops(
  %src1 : memref<4x8xf16>, %src2 : memref<8x16xf16>, %dst : memref<4x16xf16>) {
  //     CHECK: hivm.hir.matmul
  linalg.matmul ins(%src1, %src2 : memref<4x8xf16>, memref<8x16xf16>) outs(%dst : memref<4x16xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @test_matmul_transpose_a_ops
func.func @test_matmul_transpose_a_ops(
  %src1 : memref<8x4xf16>, %src2 : memref<8x16xf16>, %dst : memref<4x16xf16>) {
  //     CHECK: hivm.hir.matmul
  linalg.matmul_transpose_a ins(%src1, %src2 : memref<8x4xf16>, memref<8x16xf16>) outs(%dst : memref<4x16xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @test_matmul_transpose_b_ops
func.func @test_matmul_transpose_b_ops(
  %src1 : memref<4x8xf16>, %src2 : memref<16x8xf16>, %dst : memref<4x16xf16>) {
  //     CHECK: hivm.hir.matmul
  linalg.matmul_transpose_b ins(%src1, %src2 : memref<4x8xf16>, memref<16x8xf16>) outs(%dst : memref<4x16xf16>)
  return
}

//===----------------------------------------------------------------------===//
// Test HFusion to HIVM: linalg.broadcast to hivm.hir.vbrc. Memref.
//===----------------------------------------------------------------------===//

// -----
// CHECK: func.func @broadcast_memref(%[[arg0:.*]]: memref<8x32xf32>, %[[arg1:.*]]: memref<8x16x32xf32>) {
func.func @broadcast_memref(%input: memref<8x32xf32>, %init: memref<8x16x32xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %arg0 {{\[}}[0, 1], [2]] output_shape {{\[}}8, 1, 32] : memref<8x32xf32> into memref<8x1x32xf32>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<8x1x32xf32>) outs(%[[arg1]] : memref<8x16x32xf32>) broadcast_dims = [1]

  linalg.broadcast
      ins(%input:memref<8x32xf32>)
      outs(%init:memref<8x16x32xf32>)
      dimensions = [1]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_static_1d_to_2d_input_stride_dim0(%[[arg0:.*]]: memref<8xf32, strided<[1], offset: ?>>, %[[arg1:.*]]: memref<16x8xf32>) {
func.func @broadcast_memref_static_1d_to_2d_input_stride_dim0(%input: memref<8xf32, strided<[1], offset:?>>, %init: memref<16x8xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1]] output_shape {{\[}}1, 8] : memref<8xf32, strided<[1], offset: ?>> into memref<1x8xf32, strided<[8, 1], offset: ?>>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<1x8xf32, strided<[8, 1], offset: ?>>) outs(%[[arg1]] : memref<16x8xf32>) broadcast_dims = [0]

  linalg.broadcast
      ins(%input:memref<8xf32, strided<[1], offset:?>>)
      outs(%init:memref<16x8xf32>)
      dimensions = [0]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_static_2d_to_3d_input_stride_dim0(%[[arg0:.*]]: memref<8x12xf32, strided<[16, 1]>>, %[[arg1:.*]]: memref<2x8x12xf32>) {
func.func @broadcast_memref_static_2d_to_3d_input_stride_dim0(%input: memref<8x12xf32, strided<[16,1]>>, %init: memref<2x8x12xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1], [2]] output_shape {{\[}}1, 8, 12] : memref<8x12xf32, strided<[16, 1]>> into memref<1x8x12xf32, strided<[128, 16, 1]>>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<1x8x12xf32, strided<[128, 16, 1]>>) outs(%[[arg1]] : memref<2x8x12xf32>) broadcast_dims = [0]

  linalg.broadcast
      ins(%input:memref<8x12xf32, strided<[16, 1]>>)
      outs(%init:memref<2x8x12xf32>)
      dimensions = [0]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_dyn_with_input_stride_dim0(%[[arg0:.*]]: memref<16x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, %[[arg1:.*]]: memref<8x16x?xf32, #hivm.address_space<ub>>) {
func.func @broadcast_memref_dyn_with_input_stride_dim0(
    %input: memref<16x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>,
    %init: memref<8x16x?xf32, #hivm.address_space<ub>>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1], [2]] output_shape {{\[}}1, 16, %dim] : memref<16x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>> into memref<1x16x?xf32, strided<[?, ?, 1], offset: ?>, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<1x16x?xf32, strided<[?, ?, 1], offset: ?>, #hivm.address_space<ub>>) outs(%[[arg1]] : memref<8x16x?xf32, #hivm.address_space<ub>>) broadcast_dims = [0]

  linalg.broadcast
      ins(%input:memref<16x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>)
      outs(%init:memref<8x16x?xf32, #hivm.address_space<ub>>)
      dimensions = [0]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_2d_to_4d_input_stride(%[[arg0:.*]]: memref<8x16xf32, strided<[16, 1]>>, %[[arg1:.*]]: memref<8x10x16x32xf32>) {
func.func @broadcast_memref_2d_to_4d_input_stride(%input: memref<8x16xf32, strided<[16, 1]>>, %init: memref<8x10x16x32xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1], [2, 3]] output_shape {{\[}}8, 1, 16, 1] : memref<8x16xf32, strided<[16, 1]>> into memref<8x1x16x1xf32, strided<[16, 16, 1, 1]>>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<8x1x16x1xf32, strided<[16, 16, 1, 1]>>) outs(%[[arg1]] : memref<8x10x16x32xf32>) broadcast_dims = [1, 3]

  linalg.broadcast
      ins(%input:memref<8x16xf32, strided<[16, 1]>>)
      outs(%init:memref<8x10x16x32xf32>)
      dimensions = [1, 3]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_dyn_2d_to_4d_input_stride(%[[arg0:.*]]: memref<8x?xf32, strided<[?, 1]>>, %[[arg1:.*]]: memref<8x16x?x32xf32>) {
func.func @broadcast_memref_dyn_2d_to_4d_input_stride(%input: memref<8x?xf32, strided<[?, 1]>>, %init: memref<8x16x?x32xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1], [2, 3]] output_shape {{\[}}8, 1, %dim, 1] : memref<8x?xf32, strided<[?, 1]>> into memref<8x1x?x1xf32, strided<[?, ?, 1, 1]>>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<8x1x?x1xf32, strided<[?, ?, 1, 1]>>) outs(%[[arg1]] : memref<8x16x?x32xf32>) broadcast_dims = [1, 3]

  linalg.broadcast
      ins(%input:memref<8x?xf32, strided<[?, 1]>>)
      outs(%init:memref<8x16x?x32xf32>)
      dimensions = [1, 3]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_dyn_1d_to_6d(%[[arg0:.*]]: memref<8xf32>, %[[arg1:.*]]: memref<2x3x8x16x?x16xf32>) {
func.func @broadcast_memref_dyn_1d_to_6d(%input: memref<8xf32>, %init: memref<2x3x8x16x?x16xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1, 2, 3, 4, 5]] output_shape {{\[}}1, 1, 8, 1, 1, 1] : memref<8xf32> into memref<1x1x8x1x1x1xf32>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<1x1x8x1x1x1xf32>) outs(%[[arg1]] : memref<2x3x8x16x?x16xf32>) broadcast_dims = [0, 1, 3, 4, 5]

  linalg.broadcast
      ins(%input:memref<8xf32>)
      outs(%init:memref<2x3x8x16x?x16xf32>)
      dimensions = [0,1,3,4,5]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_dyn_2d_to_6d(%[[arg0:.*]]: memref<8x?xf32>, %[[arg1:.*]]: memref<2x3x8x16x?x16xf32>) {
func.func @broadcast_memref_dyn_2d_to_6d(%input: memref<8x?xf32>, %init: memref<2x3x8x16x?x16xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1, 2, 3], [4, 5]] output_shape [1, 1, 8, 1, %dim, 1] : memref<8x?xf32> into memref<1x1x8x1x?x1xf32>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<1x1x8x1x?x1xf32>) outs(%[[arg1]] : memref<2x3x8x16x?x16xf32>) broadcast_dims = [0, 1, 3, 5]

  linalg.broadcast
      ins(%input:memref<8x?xf32>)
      outs(%init:memref<2x3x8x16x?x16xf32>)
      dimensions = [0,1,3,5]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_dyn_3d_to_8d(%[[arg0:.*]]: memref<8x2x?xf32>, %[[arg1:.*]]: memref<2x3x4x8x2x16x?x32xf32>) {
func.func @broadcast_memref_dyn_3d_to_8d(%input: memref<8x2x?xf32>, %init: memref<2x3x4x8x2x16x?x32xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1, 2, 3], [4, 5], [6, 7]] output_shape [1, 1, 1, 8, 2, 1, %dim, 1] : memref<8x2x?xf32> into memref<1x1x1x8x2x1x?x1xf32>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<1x1x1x8x2x1x?x1xf32>) outs(%[[arg1]] : memref<2x3x4x8x2x16x?x32xf32>) broadcast_dims = [0, 1, 2, 5, 7]

  linalg.broadcast
      ins(%input:memref<8x2x?xf32>)
      outs(%init:memref<2x3x4x8x2x16x?x32xf32>)
      dimensions = [0,1,2,5,7]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_dyn_4d_to_8d(%[[arg0:.*]]: memref<4x2x16x?xf32>, %[[arg1:.*]]: memref<2x3x4x8x2x16x?x32xf32>) {
func.func @broadcast_memref_dyn_4d_to_8d(%input: memref<4x2x16x?xf32>, %init: memref<2x3x4x8x2x16x?x32xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1, 2, 3], [4], [5], [6, 7]] output_shape {{\[}}1, 1, 4, 1, 2, 16, %dim, 1] : memref<4x2x16x?xf32> into memref<1x1x4x1x2x16x?x1xf32>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<1x1x4x1x2x16x?x1xf32>) outs(%[[arg1]] : memref<2x3x4x8x2x16x?x32xf32>) broadcast_dims = [0, 1, 3, 7]

  linalg.broadcast
      ins(%input:memref<4x2x16x?xf32>)
      outs(%init:memref<2x3x4x8x2x16x?x32xf32>)
      dimensions = [0,1,3,7]
  func.return
}

// -----
// CHECK: func.func @broadcast_memref_dyn_7d_to_8d(%[[arg0:.*]]: memref<2x3x4x8x2x16x?xf32>, %[[arg1:.*]]: memref<2x3x4x8x2x16x?x32xf32>) {
func.func @broadcast_memref_dyn_7d_to_8d(%input: memref<2x3x4x8x2x16x?xf32>, %init: memref<2x3x4x8x2x16x?x32xf32>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0], [1], [2], [3], [4], [5], [6, 7]] output_shape {{\[}}2, 3, 4, 8, 2, 16, %dim, 1] : memref<2x3x4x8x2x16x?xf32> into memref<2x3x4x8x2x16x?x1xf32>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<2x3x4x8x2x16x?x1xf32>) outs(%[[arg1]] : memref<2x3x4x8x2x16x?x32xf32>) broadcast_dims = [7]

  linalg.broadcast
      ins(%input:memref<2x3x4x8x2x16x?xf32>)
      outs(%init:memref<2x3x4x8x2x16x?x32xf32>)
      dimensions = [7]
  func.return
}


// -----
// CHECK-LABEL: func.func @broadcast_memref_dyn_with_stride_scope(
// CHECK-SAME: %[[arg0:.*]]: memref<{{.*}}>, %[[arg1:.*]]: memref<{{.*}}>)
func.func @broadcast_memref_dyn_with_stride_scope(
              %input: memref<8x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>,
              %init: memref<8x16x?xf32, strided<[?, ?, 1]>, #hivm.address_space<ub>>) {
  // CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1], [2]] output_shape {{\[}}8, 1, %dim] : memref<8x?xf32, strided<[?, 1]>, #hivm.address_space<ub>> into memref<8x1x?xf32, strided<[?, ?, 1]>, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vbrc ins(%[[expand_shape]] : memref<8x1x?xf32, strided<[?, ?, 1]>, #hivm.address_space<ub>>) outs(%[[arg1]] : memref<8x16x?xf32, strided<[?, ?, 1]>, #hivm.address_space<ub>>) broadcast_dims = [1]

  linalg.broadcast
      ins(%input:memref<8x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>)
      outs(%init:memref<8x16x?xf32, strided<[?, ?, 1]>, #hivm.address_space<ub>>)
      dimensions = [1]
  func.return
}

//===----------------------------------------------------------------------===//
// Test HFusion to HIVM: linalg.broadcast to hivm.hir.vbrc. Tensor.
//===----------------------------------------------------------------------===//

// -----
// CHECK: func.func @broadcast_tensor(%[[arg0:.*]]: tensor<16x32xf32>, %[[arg1:.*]]: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
func.func @broadcast_tensor(%input: tensor<16x32xf32>, %init: tensor<8x16x32xf32>)
              -> tensor<8x16x32xf32> {
  // CHECK: %[[expanded:.*]] = tensor.expand_shape %[[arg0]] {{\[}}[0, 1], [2]] output_shape {{\[}}1, 16, 32] : tensor<16x32xf32> into tensor<1x16x32xf32>
  // CHECK: %[[T0:.*]] = hivm.hir.vbrc ins(%[[expanded]] : tensor<1x16x32xf32>) outs(%[[arg1]] : tensor<8x16x32xf32>) broadcast_dims = [0] -> tensor<8x16x32xf32>

  %bcast = linalg.broadcast
      ins(%input:tensor<16x32xf32>)
      outs(%init:tensor<8x16x32xf32>)
      dimensions = [0]
  func.return %bcast : tensor<8x16x32xf32>
}

// -----
// CHECK: func.func @broadcast_tensor_dyn(%[[arg0:.*]]: tensor<8x?xf32>, %[[arg1:.*]]: tensor<8x16x?xf32>) -> tensor<8x16x?xf32> {
func.func @broadcast_tensor_dyn(%input: tensor<8x?xf32>, %init: tensor<8x16x?xf32>)
              -> tensor<8x16x?xf32> {
  // CHECK: %[[expanded:.*]] = tensor.expand_shape %arg0 {{\[}}[0, 1], [2]] output_shape {{\[}}8, 1, %dim] : tensor<8x?xf32> into tensor<8x1x?xf32>
  // CHECK: %[[T0:.*]] = hivm.hir.vbrc ins(%[[expanded]] : tensor<8x1x?xf32>) outs(%[[arg1]] : tensor<8x16x?xf32>) broadcast_dims = [1] -> tensor<8x16x?xf32>

  %bcast = linalg.broadcast
      ins(%input:tensor<8x?xf32>)
      outs(%init:tensor<8x16x?xf32>)
      dimensions = [1]
  func.return %bcast : tensor<8x16x?xf32>
}

// -----
// CHECK: func.func @broadcast_tensor_dyn_2d_to_6d(%[[arg0:.*]]: tensor<8x?xf32>, %[[arg1:.*]]: tensor<2x3x8x16x?x16xf32>) -> tensor<2x3x8x16x?x16xf32> {
func.func @broadcast_tensor_dyn_2d_to_6d(%input: tensor<8x?xf32>, %init: tensor<2x3x8x16x?x16xf32>)
              -> tensor<2x3x8x16x?x16xf32> {
  // CHECK: %[[expanded:.*]] = tensor.expand_shape %[[arg0]] {{\[}}[0, 1, 2, 3], [4, 5]] output_shape {{\[}}1, 1, 8, 1, %dim, 1] : tensor<8x?xf32> into tensor<1x1x8x1x?x1xf32>
  // CHECK: %[[T0:.*]] = hivm.hir.vbrc ins(%[[expanded]] : tensor<1x1x8x1x?x1xf32>) outs(%[[arg1]] : tensor<2x3x8x16x?x16xf32>) broadcast_dims = [0, 1, 3, 5] -> tensor<2x3x8x16x?x16xf32>

  %bcast = linalg.broadcast
      ins(%input:tensor<8x?xf32>)
      outs(%init:tensor<2x3x8x16x?x16xf32>)
      dimensions = [0,1,3,5]
  func.return %bcast : tensor<2x3x8x16x?x16xf32>
}

// -----
// CHECK: func.func @fill_to_brc() -> tensor<256xf32> {
func.func @fill_to_brc() -> tensor<256xf32> {
// CHECK: %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[VAL_2:.*]] = hivm.hir.vbrc ins(%[[VAL_0]] : f32) outs(%[[VAL_1]] : tensor<256xf32>) -> tensor<256xf32>
// CHECK: %[[VAL_3:.*]] = memref.alloc() : memref<256xf32>
// CHECK: hivm.hir.vbrc ins(%[[VAL_0]] : f32) outs(%[[VAL_3]] : memref<256xf32>)
// CHECK: return %[[VAL_2]] : tensor<256xf32>

  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<256xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
  %alloc = memref.alloc() : memref<256xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<256xf32>)
  return %1 : tensor<256xf32>
}

// -----
// CHECK: func.func @transpose_2d() -> tensor<8x32xf32> {
func.func @transpose_2d() -> tensor<8x32xf32> {
// CHECK: %[[VAL_2:.*]] = arith.constant dense<[32, 8]> : tensor<2xi64>
// CHECK: %[[VAL_0:.*]] = memref.alloc() : memref<256xf32>
// CHECK: %[[VAL_1:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<256xf32>
// CHECK: %[[VAL_3:.*]] = tensor.reshape %[[VAL_1]](%[[VAL_2]]) : (tensor<256xf32>, tensor<2xi64>) -> tensor<32x8xf32>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<8x32xf32>
// CHECK: %[[VAL_5:.*]] = hivm.hir.vtranspose ins(%[[VAL_3]] : tensor<32x8xf32>) outs(%[[VAL_4]] : tensor<8x32xf32>) permutation = [1, 0] -> tensor<8x32xf32>
// CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<32x8xf32>
// CHECK: %[[VAL_7:.*]] = memref.alloc() : memref<8x32xf32>
// CHECK: hivm.hir.vtranspose ins(%[[VAL_6]] : memref<32x8xf32>) outs(%[[VAL_7]] : memref<8x32xf32>) permutation = [1, 0]
// CHECK: return %[[VAL_5]] : tensor<8x32xf32>

  %cst = arith.constant dense<[32, 8]> : tensor<2xi64>
  %alloc = memref.alloc() : memref<256xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
  %reshape = tensor.reshape %0(%cst) : (tensor<256xf32>, tensor<2xi64>) -> tensor<32x8xf32>
  %1 = tensor.empty() : tensor<8x32xf32>
  %transposed = linalg.transpose ins(%reshape : tensor<32x8xf32>) outs(%1 : tensor<8x32xf32>) permutation = [1, 0]

  %src = memref.alloc() : memref<32x8xf32>
  %dst = memref.alloc() : memref<8x32xf32>
  linalg.transpose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]
  return %transposed : tensor<8x32xf32>
}

// -----
// CHECK: func.func @transpose_2d() -> tensor<8x32xf32> {
func.func @transpose_2d() -> tensor<8x32xf32> {
// CHECK: %[[VAL_2:.*]] = arith.constant dense<[32, 8]> : tensor<2xi64>
// CHECK: %[[VAL_0:.*]] = memref.alloc() : memref<256xf32>
// CHECK: %[[VAL_1:.*]] = bufferization.to_tensor %[[VAL_0]] restrict writable : memref<256xf32>
// CHECK: %[[VAL_3:.*]] = tensor.reshape %[[VAL_1]](%[[VAL_2]]) : (tensor<256xf32>, tensor<2xi64>) -> tensor<32x8xf32>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<8x32xf32>
// CHECK: %[[VAL_5:.*]] = hivm.hir.vtranspose ins(%[[VAL_3]] : tensor<32x8xf32>) outs(%[[VAL_4]] : tensor<8x32xf32>) permutation = [1, 0] disable_align = true -> tensor<8x32xf32>
// CHECK: %[[VAL_6:.*]] = memref.alloc() : memref<32x8xf32>
// CHECK: %[[VAL_7:.*]] = memref.alloc() : memref<8x32xf32>
// CHECK: hivm.hir.vtranspose ins(%[[VAL_6]] : memref<32x8xf32>) outs(%[[VAL_7]] : memref<8x32xf32>) permutation = [1, 0] disable_align = true
// CHECK: return %[[VAL_5]] : tensor<8x32xf32>

  %cst = arith.constant dense<[32, 8]> : tensor<2xi64>
  %alloc = memref.alloc() : memref<256xf32>
  %0 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
  %reshape = tensor.reshape %0(%cst) : (tensor<256xf32>, tensor<2xi64>) -> tensor<32x8xf32>
  %1 = tensor.empty() : tensor<8x32xf32>
  %transposed = linalg.transpose ins(%reshape : tensor<32x8xf32>) outs(%1 : tensor<8x32xf32>) permutation = [1, 0]
  annotation.mark %transposed {transpose_without_align} : tensor<8x32xf32>

  %src = memref.alloc() : memref<32x8xf32>
  %dst = memref.alloc() : memref<8x32xf32>
  linalg.transpose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]
  annotation.mark %dst {transpose_without_align} : memref<8x32xf32>
  
  return %transposed : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_comparei_ops
func.func @test_hfusion_comparei_ops(
  %src1 : memref<6x6xi32>, %src2 : memref<6x6xi32>, %dst : memref<6x6xi1>) {
  //     CHECK: hivm.hir.vcmp
  hfusion.compare {compare_fn  = #hfusion.compare_fn<veq>}
    ins(%src1, %src2 : memref<6x6xi32>, memref<6x6xi32>)
    outs(%dst : memref<6x6xi1>)
  //     CHECK: hivm.hir.vcmp
  //     CHECK-SAME: compare_mode = <ne>
  hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %src2 : memref<6x6xi32>, memref<6x6xi32>)
    outs(%dst : memref<6x6xi1>)
  return
}

// -----

// CHECK-LABEL: func.func @test_hfusion_comparef_ops
func.func @test_hfusion_comparef_ops(
  %src1 : memref<6x6xf32>, %src2 : memref<6x6xf32>, %dst : memref<6x6xi1>) {
  //     CHECK: hivm.hir.vcmp
  hfusion.compare {compare_fn  = #hfusion.compare_fn<veq>}
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
    outs(%dst : memref<6x6xi1>)
  //     CHECK: hivm.hir.vcmp
  //     CHECK-SAME: compare_mode = <ne>
  hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
    outs(%dst : memref<6x6xi1>)
  //     CHECK: hivm.hir.vcmp
  //     CHECK-SAME: compare_mode = <lt>
  hfusion.compare {compare_fn  = #hfusion.compare_fn<vlt>}
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
    outs(%dst : memref<6x6xi1>)
  //     CHECK: hivm.hir.vcmp
  //     CHECK-SAME: compare_mode = <le>
  hfusion.compare {compare_fn  = #hfusion.compare_fn<vle>}
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
    outs(%dst : memref<6x6xi1>)
  //     CHECK: hivm.hir.vcmp
  //     CHECK-SAME: compare_mode = <gt>
  hfusion.compare {compare_fn  = #hfusion.compare_fn<vgt>}
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
    outs(%dst : memref<6x6xi1>)
  //     CHECK: hivm.hir.vcmp
  //     CHECK-SAME: compare_mode = <ge>
  hfusion.compare {compare_fn  = #hfusion.compare_fn<vge>}
    ins(%src1, %src2 : memref<6x6xf32>, memref<6x6xf32>)
    outs(%dst : memref<6x6xi1>)
  return
}

// -----

// CHECK-LABEL: func.func @test_hfusion_compare_scalar_ops
func.func @test_hfusion_compare_scalar_ops() -> tensor<4xi1> {
  // CHECK: %[[VAL_0:.*]] = arith.constant 4 : i32
  // CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4xi32>
  // CHECK: %[[VAL_2:.*]] = hivm.hir.vbrc ins(%[[VAL_0]] : i32) outs(%[[VAL_1]] : tensor<4xi32>) -> tensor<4xi32>
  // CHECK: hivm.hir.vcmp
  %src1 = arith.constant 4 : i32
  %src2 = arith.constant 0 : i32
  %dst = tensor.empty() : tensor<4xi1>
  %0 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} 
    ins(%src1, %src2 : i32, i32) 
    outs(%dst : tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_selecti_ops
func.func @test_hfusion_selecti_ops(
  %src1 : memref<6x6xi1>, %src2 : memref<6x6xi32>, %src3 : memref<6x6xi32>, %dst : memref<6x6xi32>) {
  //     CHECK: hivm.hir.vsel
  hfusion.select
    ins(%src1, %src2, %src3 : memref<6x6xi1>, memref<6x6xi32>, memref<6x6xi32>)
    outs(%dst : memref<6x6xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @test_hfusion_selectf_ops
func.func @test_hfusion_selectf_ops(
  %src1 : memref<6x6xi1>, %src2 : memref<6x6xf32>, %src3 : memref<6x6xf32>, %dst : memref<6x6xf32>) {
  //     CHECK: hivm.hir.vsel
  hfusion.select
    ins(%src1, %src2, %src3 : memref<6x6xi1>, memref<6x6xf32>, memref<6x6xf32>)
    outs(%dst : memref<6x6xf32>)
  return
}

// -----
// CHECK-LABEL: func.func @arange_1d
func.func @arange_1d() -> tensor<256xi32> {

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: hivm.hir.varange
// CHECK-SAME: strides
// CHECK-SAME: outs(%[[EMPTY]] : tensor<256xi32>) -> tensor<256xi32>

// CHECK: %[[DST:.*]] = memref.alloc() : memref<256xi32>
// CHECK: hivm.hir.varange
// CHECK-SAME: strides
// CHECK-SAME: outs(%[[DST]] : memref<256xi32>)
// CHECK: return

  %empty = tensor.empty() : tensor<256xi32>
  %c1 = arith.constant 1 : index
  %transposed = hfusion.arange strides[%c1] outs(%empty : tensor<256xi32>) -> tensor<256xi32>

  %dst = memref.alloc() : memref<256xi32>
  hfusion.arange strides[%c1] outs(%dst : memref<256xi32>)
  return %transposed : tensor<256xi32>
}

// -----
// CHECK-LABEL: func.func @arange_1d_1f
func.func @arange_1d_1f() -> tensor<256xi32> {
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<256xi32>
// CHECK: hivm.hir.varange
// CHECK-SAME: offset
// CHECK-SAME: strides
// CHECK-SAME: outs(%[[EMPTY]] : tensor<256xi32>) -> tensor<256xi32>

// CHECK: %[[DST:.*]] = memref.alloc() : memref<256xi32>
// CHECK: hivm.hir.varange
// CHECK-SAME: strides
// CHECK-SAME: outs(%[[DST]] : memref<256xi32>)
// CHECK: return

  %empty = tensor.empty() : tensor<256xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 5 : index
  %transposed = hfusion.arange offset[%c2] strides[%c1] outs(%empty : tensor<256xi32>) -> tensor<256xi32>

  %dst = memref.alloc() : memref<256xi32>
  hfusion.arange strides[%c1] outs(%dst : memref<256xi32>)
  return %transposed : tensor<256xi32>
}

// -----
// CHECK-LABEL: func.func @arange_2d
func.func @arange_2d() -> tensor<16x16xi32> {

// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<16x16xi32>
// CHECK: hivm.hir.varange
// CHECK-SAME: strides
// CHECK-SAME: outs(%[[EMPTY]] : tensor<16x16xi32>) -> tensor<16x16xi32>

  %empty = tensor.empty() : tensor<16x16xi32>
  %c1 = arith.constant 1 : index
  %transposed = hfusion.arange strides[%c1, %c1] outs(%empty : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %transposed : tensor<16x16xi32>
}

// -----
// CHECK-LABEL: func.func @test_cast_int8_t_to_bool_rintmode
func.func @test_cast_int8_t_to_bool_rintmode() {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<1024xi8>
  // CHECK: %[[DST:.*]] = memref.alloc() : memref<1024xi1>
  // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<1024xi8>) outs(%[[DST]] : memref<1024xi1>)
  // CHECK: return
  %arg0 = memref.alloc() : memref<1024xi8>
  %arg1 = memref.alloc() : memref<1024xi1>
  hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : memref<1024xi8>) outs(%arg1 : memref<1024xi1>)
  return
}

// -----
// CHECK-LABEL: func.func @extract_scalar_rhs_for_binary_shift_op
// CHECK: %[[extract0:.*]] = tensor.extract {{.*}} : tensor<1xi64>
// CHECK: hivm.hir.vshl ins({{.*}}, %[[extract0]] : tensor<1xi64>, i64)
// CHECK: %[[extract1:.*]] = tensor.extract {{.*}} : tensor<1x1xi64>
// CHECK: hivm.hir.vshl ins({{.*}}, %[[extract1]] : tensor<1x1xi64>, i64)
func.func @extract_scalar_rhs_for_binary_shift_op(%arg0 : tensor<1xi64>, %arg1 : tensor<1x1xi64>) -> (tensor<1xi64>, tensor<1x1xi64>) {
  %c2_i64 = arith.constant 2 : i64
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c2_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%arg0, %1 : tensor<1xi64>, tensor<1xi64>) 
                                                                outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %3 = tensor.empty() : tensor<1x1xi64>
  %4 = linalg.fill ins(%c2_i64 : i64) outs(%3 : tensor<1x1xi64>) -> tensor<1x1xi64>
  %5 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%arg1, %4 : tensor<1x1xi64>, tensor<1x1xi64>) 
                                                                outs(%3 : tensor<1x1xi64>) -> tensor<1x1xi64>
  return %2, %5 : tensor<1xi64>, tensor<1x1xi64>
}

// -----
// CHECK-LABEL: func.func @test_brc_scalar
func.func @test_brc_scalar() -> tensor<200x200xf32> {
// CHECK: %[[CST0:.*]] = arith.constant dense<2.885390e+00>
// CHECK: %[[ARG0:.*]] = tensor.empty() : tensor<200x200xf32>
// CHECK: %[[ARG1:.*]] = hivm.hir.vbrc ins(%{{.*}} : tensor<{{(1x1x)?f32}}>) outs(%[[ARG0]] : tensor<200x200xf32>) broadcast_dims = [0, 1] -> tensor<200x200xf32>
// return %[[ARG1]] : tensor<200x200xf32>
  %cst_0 = arith.constant dense<2.885390e+00> : tensor<f32>
  %0 = tensor.empty() : tensor<200x200xf32>
  %broadcasted = linalg.broadcast ins(%cst_0 : tensor<f32>) outs(%0 : tensor<200x200xf32>) dimensions = [0, 1]
  return %broadcasted : tensor<200x200xf32>
}

// -----
// CHECK-LABEL: func.func @test_barrier
func.func @test_barrier() {
  hfusion.barrier
  return
}
// CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]


// -----
// CHECK-LABEL: func.func @test_interleave
func.func @test_interleave() -> tensor<4x2x128xf16> {
  // CHECK: %[[ARG0:.*]] = tensor.empty() : tensor<4x2x64xf16>
  // CHECK: %[[ARG1:.*]] = tensor.empty() : tensor<4x2x64xf16>
  // CHECK: %[[ARG2:.*]] = tensor.empty() : tensor<4x2x128xf16>
  // CHECK: %[[ARG3:.*]] = hivm.hir.vinterleave ins(%[[ARG0]], %[[ARG1]] : tensor<4x2x64xf16>, tensor<4x2x64xf16>) outs(%[[ARG2]] : tensor<4x2x128xf16>) interleave_channel_nums = 2 -> tensor<4x2x128xf16>
  // CHECK: return %[[ARG3]] : tensor<4x2x128xf16>
  %0 = tensor.empty() : tensor<4x2x64xf16>
  %1 = tensor.empty() : tensor<4x2x64xf16>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
  return %2 : tensor<4x2x128xf16>
}

// -----
// CHECK-LABEL: func.func @test_flip
func.func @test_flip()-> tensor<4x8x8xf32> {
  // CHECK: %{{.*}} = hivm.hir.vflip ins(%{{.*}} : tensor<4x8x8xf32>) outs(%{{.*}} : tensor<4x8x8xf32>) flip_axis = 2 -> tensor<4x8x8xf32>
  %0 = tensor.empty() : tensor<4x8x8xf32>
  %1 = hfusion.flip %0 : tensor<4x8x8xf32> flip_axis = 2 -> tensor<4x8x8xf32>
  return %1 : tensor<4x8x8xf32>
}

// -----
// CHECK-LABEL: func.func @test_atomic_add
module {
  func.func @test_atomic_add(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %arg1: tensor<256xf32>) {
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.index_cast %c256_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %1 = bufferization.to_memref %arg1 : memref<256xf32, strided<[1], offset: ?>>
    // CHECK: hivm.hir.store ins(%[[UB_MEMREF:.*]] : memref<256xf32, strided<[1], offset: ?>>) outs(%[[GM_MEMREF:.*]] : memref<256xf32, strided<[1], offset: ?>>) atomic = <add>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<add>} ins(%1 : memref<256xf32, strided<[1], offset: ?>>) outs(%reinterpret_cast : memref<256xf32, strided<[1], offset: ?>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @test_powi_to_vpow(
// CHECK: hivm.hir.vpow
func.func @test_powi_to_vpow(%arg0: tensor<5x4xi32>, %arg1: tensor<5x4xi32>, %arg2: tensor<5x4xi32> ) -> tensor<5x4xi32> {
  %empty = tensor.empty() : tensor<5x4xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0, %arg1 : tensor<5x4xi32>, tensor<5x4xi32>) outs(%arg2: tensor<5x4xi32>) -> tensor<5x4xi32>
  return %1 : tensor<5x4xi32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_cast_ops(
// CHECK: hivm.hir.vcast[[_:.*]] round_mode = <round>
// CHECK: hivm.hir.vcast[[_:.*]] round_mode = <ceil>
// CHECK: hivm.hir.vcast[[_:.*]] round_mode = <floor>
// CHECK: hivm.hir.vcast[[_:.*]] round_mode = <trunc>
// CHECK: hivm.hir.vcast[[_:.*]] round_mode = <odd>
func.func @test_hfusion_cast_ops() {
  %arg0 = memref.alloc() : memref<1024xf16>
  %arg1 = memref.alloc() : memref<1024xi8>
  %arg2 = memref.alloc() : memref<1024xf32>
  %arg3 = memref.alloc() : memref<1024xf16>
  hfusion.cast {round_mode = #hfusion.round_mode<round>} ins(%arg0 : memref<1024xf16>) outs(%arg1 : memref<1024xi8>)
  hfusion.cast {round_mode = #hfusion.round_mode<ceil>} ins(%arg0 : memref<1024xf16>) outs(%arg1 : memref<1024xi8>)
  hfusion.cast {round_mode = #hfusion.round_mode<floor>} ins(%arg0 : memref<1024xf16>) outs(%arg1 : memref<1024xi8>)
  hfusion.cast {round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : memref<1024xf16>) outs(%arg1 : memref<1024xi8>)
  hfusion.cast {round_mode = #hfusion.round_mode<odd>} ins(%arg2 : memref<1024xf32>) outs(%arg3 : memref<1024xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @test_hfusion_cum_ops(
// CHECK: hivm.hir.vcumsum ins(%{{.*}} : tensor<1x2x3x4xi32>) outs(%{{.*}} : tensor<1x2x3x4xi32>) cum_dims = [0] reverse = false -> tensor<1x2x3x4xi32>
// CHECK: hivm.hir.vcumsum ins(%{{.*}} : tensor<1x2x3x4xi32>) outs(%{{.*}} : tensor<1x2x3x4xi32>) cum_dims = [1] reverse = false -> tensor<1x2x3x4xi32>
// CHECK: hivm.hir.vcumsum ins(%{{.*}} : tensor<1x2x3x4xi32>) outs(%{{.*}} : tensor<1x2x3x4xi32>) cum_dims = [2] reverse = false -> tensor<1x2x3x4xi32>
// CHECK: hivm.hir.vcumsum ins(%{{.*}} : tensor<1x2x3x4xi32>) outs(%{{.*}} : tensor<1x2x3x4xi32>) cum_dims = [3] reverse = false -> tensor<1x2x3x4xi32>
// CHECK: hivm.hir.vcumprod ins(%{{.*}} : tensor<1x2x3x4xi32>) outs(%{{.*}} : tensor<1x2x3x4xi32>) cum_dims = [0] reverse = false -> tensor<1x2x3x4xi32>
// CHECK: hivm.hir.vcumprod ins(%{{.*}} : tensor<1x2x3x4xi32>) outs(%{{.*}} : tensor<1x2x3x4xi32>) cum_dims = [1] reverse = false -> tensor<1x2x3x4xi32>
// CHECK: hivm.hir.vcumprod ins(%{{.*}} : tensor<1x2x3x4xi32>) outs(%{{.*}} : tensor<1x2x3x4xi32>) cum_dims = [2] reverse = false -> tensor<1x2x3x4xi32>
// CHECK: hivm.hir.vcumprod ins(%{{.*}} : tensor<1x2x3x4xi32>) outs(%{{.*}} : tensor<1x2x3x4xi32>) cum_dims = [3] reverse = false -> tensor<1x2x3x4xi32>
func.func @test_hfusion_cum_ops() -> tensor<1x2x3x4xi32> {
  %arg = tensor.empty() : tensor<1x2x3x4xi32>
  %0 = hfusion.cumsum %arg : tensor<1x2x3x4xi32> cum_dims = [0] reverse = false -> tensor<1x2x3x4xi32>
  %1 = hfusion.cumsum %0 : tensor<1x2x3x4xi32> cum_dims = [1] reverse = false -> tensor<1x2x3x4xi32>
  %2 = hfusion.cumsum %1 : tensor<1x2x3x4xi32> cum_dims = [2] reverse = false -> tensor<1x2x3x4xi32>
  %3 = hfusion.cumsum %2 : tensor<1x2x3x4xi32> cum_dims = [3] reverse = false -> tensor<1x2x3x4xi32>
  %4 = hfusion.cumprod %3 : tensor<1x2x3x4xi32> cum_dims = [0] reverse = false -> tensor<1x2x3x4xi32>
  %5 = hfusion.cumprod %4 : tensor<1x2x3x4xi32> cum_dims = [1] reverse = false -> tensor<1x2x3x4xi32>
  %6 = hfusion.cumprod %5 : tensor<1x2x3x4xi32> cum_dims = [2] reverse = false -> tensor<1x2x3x4xi32>
  %7 = hfusion.cumprod %6 : tensor<1x2x3x4xi32> cum_dims = [3] reverse = false -> tensor<1x2x3x4xi32>
  return %7 : tensor<1x2x3x4xi32>
}

// -----
// CHECK-LABEL: func.func @test_atomic_cas
module {
  func.func @test_atomic_cas(%arg0: memref<?xi16>) {
    
    %alloc = memref.alloc() : memref<256xi16>
    %alloc_2 = memref.alloc() : memref<256xi16>
    // CHECK: hivm.hir.atomic_cas
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
    hfusion.atomic_cas ins(%alloc_2, %alloc : memref<256xi16>, memref<256xi16>) outs(%reinterpret_cast_0 : memref<256xi16, strided<[1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @test_atomic_xchg
module {
  func.func @test_atomic_xchg(%arg0: memref<?xi16>) {
    %alloc = memref.alloc() : memref<256xi16>
    // CHECK: hivm.hir.atomic_xchg ins(%[[VAL_1:.*]] : memref<256xi16>) outs(%[[VAL_2:.*]] : memref<256xi16, strided<[1]>>)
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
    hfusion.atomic_xchg ins(%alloc: memref<256xi16>) outs(%reinterpret_cast_0 : memref<256xi16, strided<[1]>>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @test_masked_atomic_xchg
module {
  func.func @test_masked_atomic_xchg(%arg0: memref<?xi16>, %arg1: memref<256xi1>) {
    %alloc = memref.alloc() : memref<256xi16>
    // CHECK: hivm.hir.atomic_xchg ins(%[[VAL_2:.*]] : memref<256xi16>) outs(%[[VAL_3:.*]] : memref<256xi16, strided<[1]>>) mask(%[[VAL_1:.*]] : memref<256xi1>)
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1] : memref<?xi16> to memref<256xi16, strided<[1]>>
    hfusion.atomic_xchg ins(%alloc: memref<256xi16>) outs(%reinterpret_cast_0 : memref<256xi16, strided<[1]>>) mask(%arg1: memref<256xi1>)
    return
  }
}

// -----
func.func @multi_buffer_three_result() -> (i32, f32, f32) {
  %result1, %result2, %result3 = "test.three_result"() <{kind = 1 : i64}> : () -> (i32, f32, f32)

  // CHECK: annotation.mark %result1 {hivm.multi_buffer = 2 : i32} : i32
  // CHECK: annotation.mark %result2 {hivm.multi_buffer = 3 : i32} : f32
  // CHECK: annotation.mark %result3 {hivm.multi_buffer = 4 : i32} : f32

  annotation.mark %result1 {hfusion.multi_buffer = 2 : i32} : i32
  annotation.mark %result2 {hfusion.multi_buffer = 3 : i32} : f32
  annotation.mark %result3 {hfusion.multi_buffer = 4 : i32} : f32

  return %result1, %result2, %result3 : i32, f32, f32
}

// -----
// CHECK-LABEL: func.func @bind_sub_block_to_map_attr
func.func @bind_sub_block_to_map_attr(%arg0: tensor<64x64xf32> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg1: memref<32x64xf16>) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<14xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
  %cst = arith.constant 1.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32_i32 = arith.constant 32 : i32
  %0 = tensor.empty() : tensor<32x64xf32>
  scf.for %arg2 = %c0 to %c2 step %c1 {
    %1 = arith.index_cast %arg2 : index to i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %extracted_slice = tensor.extract_slice %arg0[%3, 0] [32, 64] [1, 1] : tensor<64x64xf32> to tensor<32x64xf32>
    %4 = hivm.hir.vadd ins(%extracted_slice, %cst : tensor<32x64xf32>, f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %5 = tensor.empty() : tensor<32x64xf16>
    %6 = hivm.hir.vcast ins(%4 : tensor<32x64xf32>) outs(%5 : tensor<32x64xf16>) -> tensor<32x64xf16>
    bufferization.materialize_in_destination %6 in writable %arg1 : (tensor<32x64xf16>, memref<32x64xf16>) -> ()
  } {hfusion.bind_sub_block}
  // CHECK: } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
  return
}

// -----
func.func @storage_align() -> (i32, f32, f32) {
  %result1, %result2, %result3 = "test.three_result"() <{kind = 1 : i64}> : () -> (i32, f32, f32)

  // CHECK: annotation.mark %result1 {hivm.stride_align_dims = [0], hivm.stride_align_value_in_byte = [32]} : i32
  // CHECK: annotation.mark %result2 {hivm.stride_align_dims = [0], hivm.stride_align_value_in_byte = [32]} : f32
  // CHECK: annotation.mark %result3 {hivm.stride_align_dims = [0], hivm.stride_align_value_in_byte = [32]} : f32

  annotation.mark %result1 {hfusion.stride_align_dims = [0], hfusion.stride_align_value_in_byte = [32]} : i32
  annotation.mark %result2 {hfusion.stride_align_dims = [0], hfusion.stride_align_value_in_byte = [32]} : f32
  annotation.mark %result3 {hfusion.stride_align_dims = [0], hfusion.stride_align_value_in_byte = [32]} : f32

  return %result1, %result2, %result3 : i32, f32, f32
}

// -----
// CHECK-LABEL: func.func @test_atomic_max_ui8
module {
  func.func @test_atomic_max_ui8(%arg0: memref<?xi8> {tt.divisibility = 16 : i32}, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1] : memref<?xi8> to memref<256xi8>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<?xi8> to memref<256xi8>
    // CHECK: hivm.hir.store ins(%{{.*}} : memref<256xi8>) outs(%{{.*}} : memref<256xi8>) atomic = <umax>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<umax>} ins(%reinterpret_cast : memref<256xi8>) outs(%reinterpret_cast_1 : memref<256xi8>)
    return
  }
}

// -----
// CHECK-LABEL: func.func @test_atomic_min_ui8
module {
  func.func @test_atomic_min_ui8(%arg0: memref<?xi8> {tt.divisibility = 16 : i32}, %arg1: memref<?xi8> {tt.divisibility = 16 : i32}) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [256], strides: [1] : memref<?xi8> to memref<256xi8>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [256], strides: [1] : memref<?xi8> to memref<256xi8>
    // CHECK: hivm.hir.store ins(%{{.*}} : memref<256xi8>) outs(%{{.*}} : memref<256xi8>) atomic = <umin>
    hfusion.store {atomic_kind = #hfusion.atomic_kind<umin>} ins(%reinterpret_cast : memref<256xi8>) outs(%reinterpret_cast_1 : memref<256xi8>)
    return
  }
}