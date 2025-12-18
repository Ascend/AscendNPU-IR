// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func.func @test_elementwise_unary_op_shape(%dim0 : index, %dim1: index) {
  %a = memref.alloc() : memref<2x16xf16>
  %b = memref.alloc(%dim0) : memref<?x16xf16>
  %c = memref.alloc(%dim1) : memref<2x?xf16>
  %d = memref.alloc(%dim0, %dim1) : memref<?x?xf16>
  bishengir_test.elementwise_unary ins(%a : memref<2x16xf16>) outs(%a : memref<2x16xf16>)
  bishengir_test.elementwise_unary ins(%b : memref<?x16xf16>) outs(%a : memref<2x16xf16>)
  bishengir_test.elementwise_unary ins(%c : memref<2x?xf16>) outs(%a : memref<2x16xf16>)
  bishengir_test.elementwise_unary ins(%d : memref<?x?xf16>) outs(%a : memref<2x16xf16>)
  return
}

// -----

func.func @test_tensor_elementwise_binary_op(
  %src1 : tensor<?x?xf16>, %src2 : tensor<?x?xf16>, %dst : tensor<?x?xf16>) {
  %ret =
  bishengir_test.elementwise_binary ins(%src1, %src2 : tensor<?x?xf16>, tensor<?x?xf16>)
                                    outs(%dst : tensor<?x?xf16>) -> tensor<?x?xf16>
  return
}

// -----

func.func @test_tensor_elementwise_binary_op_mismatch_rank(
   %src1 : tensor<?xf16>, %src2 : tensor<?x?xf16>, %dst : tensor<?x?xf16>) {
  // expected-error@+2 {{operands must have same rank}}
  %ret =
  bishengir_test.elementwise_binary ins(%src1, %src2 : tensor<?xf16>, tensor<?x?xf16>)
                                    outs(%dst : tensor<?x?xf16>) -> tensor<?x?xf16>
  return
}

// -----

func.func @test_tensor_elementwise_binary_op_incorrect_input_num(
   %src1 : tensor<?x?xf16>, %src2 : tensor<?x?xf16>, %dst : tensor<?x?xf16>) {
  // expected-error@+2 {{elementwise op expected 2 inputs, but found 1}}
  %ret =
  bishengir_test.elementwise_binary ins(%src1 : tensor<?x?xf16>)
                                    outs(%dst: tensor<?x?xf16>) -> tensor<?x?xf16>
  return
}

// -----

func.func @test_memref_elementwise_binary_op(
  %src1 : memref<?x?xf16>, %src2 : memref<?x?xf16>, %dst : memref<?x?xf16>) {
  bishengir_test.elementwise_binary ins(%src1, %src2 : memref<?x?xf16>, memref<?x?xf16>)
                                    outs(%dst : memref<?x?xf16>)
  return
}

// -----

func.func @test_elementwise_binary_op_shape(%dim0 : index, %dim1: index) {
  %a = memref.alloc() : memref<2x16xf16>
  %b = memref.alloc(%dim0) : memref<?x16xf16>
  %c = memref.alloc(%dim1) : memref<2x?xf16>
  %d = memref.alloc(%dim0, %dim1) : memref<?x?xf16>
  bishengir_test.elementwise_binary ins(%a, %b : memref<2x16xf16>, memref<?x16xf16>) outs(%a : memref<2x16xf16>)
  bishengir_test.elementwise_binary ins(%a, %c : memref<2x16xf16>, memref<2x?xf16>) outs(%a : memref<2x16xf16>)
  bishengir_test.elementwise_binary ins(%a, %d : memref<2x16xf16>, memref<?x?xf16>) outs(%a : memref<2x16xf16>)
  bishengir_test.elementwise_binary ins(%b, %c : memref<?x16xf16>, memref<2x?xf16>) outs(%a : memref<2x16xf16>)
  bishengir_test.elementwise_binary ins(%b, %d : memref<?x16xf16>, memref<?x?xf16>) outs(%a : memref<2x16xf16>)
  bishengir_test.elementwise_binary ins(%c, %d : memref<2x?xf16>, memref<?x?xf16>) outs(%a : memref<2x16xf16>)
  return
}

// -----

func.func @test_fully_static_memref_elementwise_binary_op(
  %src1 : memref<16x16xf16, strided<[16, 1], offset: 0>>,
  %src2 : memref<16x16xf16, strided<[16, 1], offset: 0>>,
  %dst : memref<16x16xf16, strided<[16, 1], offset: 0>>) {
  bishengir_test.elementwise_binary
    ins(%src1, %src2 : memref<16x16xf16, strided<[16, 1], offset: 0>>, memref<16x16xf16, strided<[16, 1], offset: 0>>)
    outs(%dst : memref<16x16xf16, strided<[16, 1], offset: 0>>)
  return
}

// -----

func.func @test_partial_dynamic_memref_elementwise_binary_op(
  %src1 : memref<16x16xf16, strided<[?, ?], offset: 0>>,
  %src2 : memref<16x16xf16, strided<[?, ?], offset: 0>>,
  %dst : memref<16x16xf16, strided<[?, ?], offset: 0>>) {
  bishengir_test.elementwise_binary
    ins(%src1, %src2 : memref<16x16xf16, strided<[?, ?], offset: 0>>, memref<16x16xf16, strided<[?, ?], offset: 0>>)
    outs(%dst : memref<16x16xf16, strided<[?, ?], offset: 0>>)
  return
}

// -----

func.func @test_vector_scalar_only_trait(
  %src1 : memref<?xf16>, %dst : memref<?xf16>, %cst : f16) {
  bishengir_test.vector_scalar_only ins(%src1, %cst : memref<?xf16>, f16)
                                  outs(%dst : memref<?xf16>)
  // operands not specified could have any type
  bishengir_test.vector_scalar_only ins(%src1, %cst, %src1 : memref<?xf16>, f16, memref<?xf16>)
                                    outs(%dst : memref<?xf16>)
  bishengir_test.vector_scalar_only ins(%src1, %cst, %cst : memref<?xf16>, f16, f16)
                                    outs(%dst : memref<?xf16>)
  return
}

// -----

func.func @test_broadcastable_on_the_fly_dynamic(
  %src1 : memref<1x?xf16>, %dst : memref<?x?xf16>, %cst : f16) {
  bishengir_test.broadcastableOTF ins(%src1, %src1, %cst : memref<1x?xf16>, memref<1x?xf16>, f16)
                                  outs(%dst : memref<?x?xf16>)
                                  broadcast = [0]
  return
}

// -----

func.func @test_broadcastable_on_the_fly_static(
  %src1 : memref<1x1x16xf16>, %dst : memref<32x16x16xf16>, %cst : f16) {
  bishengir_test.broadcastableOTF ins(%src1, %src1, %cst : memref<1x1x16xf16>, memref<1x1x16xf16>, f16)
                                  outs(%dst : memref<32x16x16xf16>)
                                  broadcast = [0, 1]
  return
}

// -----

func.func @test_broadcastable_on_the_fly_last_dim_broadcast(
  %src1 : memref<?x1xf16>, %dst : memref<?x?xf16>, %cst : f16) {
  // expected-error@+1 {{broadcast dim exceeds op's rank}}
  bishengir_test.broadcastableOTF ins(%src1, %src1, %cst : memref<?x1xf16>, memref<?x1xf16>, f16)
                                  outs(%dst : memref<?x?xf16>)
                                  broadcast = [2]
  return
}

// -----

func.func @test_broadcastable_on_the_fly_incorrect_dim(
  %src1 : memref<?x1xf16>, %dst : memref<?x?xf16>, %cst : f16) {
  // expected-error@+1 {{broadcast dim exceeds op's rank}}
  bishengir_test.broadcastableOTF ins(%src1, %src1, %cst : memref<?x1xf16>, memref<?x1xf16>, f16)
                                  outs(%dst : memref<?x?xf16>)
                                  broadcast = [1000]
  return
}

// -----

func.func @test_broadcastable_on_the_fly_broadcast_dim_incorrect(
  %src1 : memref<2x16xf16>, %dst : memref<16x16xf16>, %cst : f16) {
  // expected-error@+1 {{input operand's broadcast dim is not 1}}
  bishengir_test.broadcastableOTF ins(%src1, %src1, %cst : memref<2x16xf16>, memref<2x16xf16>, f16)
                                  outs(%dst : memref<16x16xf16>)
                                  broadcast = [0]
  return
}

// -----

func.func @test_broadcastable_on_the_fly_non_broadcast_dim_incorrect(
  %src1 : memref<1x16xf16>, %dst : memref<16x32xf16>, %cst : f16) {
  // expected-error@+1 {{input operand's non-broadcast dim does not match with output}}
  bishengir_test.broadcastableOTF ins(%src1, %src1, %cst : memref<1x16xf16>, memref<1x16xf16>, f16)
                                  outs(%dst : memref<16x32xf16>)
                                  broadcast = [0]
  return
}

// -----

func.func @test_transposable_on_the_fly(
  %src : memref<4x8x16xf16>, %src1 : memref<?x?x?xf16>,
  %dst : memref<4x8x16xf16>, %dst1 : memref<8x4x16xf16>, %cst : f16) {
  // no transpose
  bishengir_test.transposableOTF ins(%src, %cst : memref<4x8x16xf16>, f16)
                                 outs(%dst : memref<4x8x16xf16>)

  bishengir_test.transposableOTF ins(%src, %cst : memref<4x8x16xf16>, f16)
                                 outs(%dst : memref<4x8x16xf16>)
                                 transpose = [0, 1, 2]
  // non-last dim transpose
  bishengir_test.transposableOTF ins(%src, %cst : memref<4x8x16xf16>, f16)
                                 outs(%dst1 : memref<8x4x16xf16>)
                                 transpose = [1, 0, 2]

  bishengir_test.transposableOTF ins(%src1, %cst : memref<?x?x?xf16>, f16)
                                 outs(%dst1 : memref<8x4x16xf16>)
                                 transpose = [1, 0, 2]
  return
}

// -----

func.func @test_transposable_on_the_fly_corner_cases(%dim0 : index) {
  %a = memref.alloc() : memref<4x8x16xf16>
  %b = memref.alloc(%dim0) : memref<?x8x16xf16>
  %c = memref.alloc() : memref<8x4x16xf16>
  %d = memref.alloc(%dim0) : memref<8x?x16xf16>
  // inputs with dynamic dim
  bishengir_test.transposableOTF ins(%a, %b : memref<4x8x16xf16>, memref<?x8x16xf16>)
                                 outs(%c : memref<8x4x16xf16>)
                                 transpose = [1, 0, 2]
  // output with dynamic dim
  bishengir_test.transposableOTF ins(%a, %a : memref<4x8x16xf16>, memref<4x8x16xf16>)
                                 outs(%d : memref<8x?x16xf16>)
                                 transpose = [1, 0, 2]
  return
}

// -----

func.func @test_transposable_on_the_fly_last_dim_transpose(
  %src : memref<?x?xf16>, %dst : memref<?x?xf16>, %cst : f16) {
  // expected-error@+1 {{transpose dim is the last dimension}}
  bishengir_test.transposableOTF ins(%src, %cst : memref<?x?xf16>, f16)
                                 outs(%dst : memref<?x?xf16>)
                                 transpose = [1, 0]
  return
}

// -----

func.func @test_transposable_on_the_fly_incorrect_transpose(
  %src : memref<?x?xf16>, %dst : memref<?x?xf16>, %cst : f16) {
  // expected-error@+1 {{expects 'transpose' to be a permutation}}
  bishengir_test.transposableOTF ins(%src, %cst : memref<?x?xf16>, f16)
                                 outs(%dst : memref<?x?xf16>)
                                 transpose = [3, 0]
  return
}

// -----

func.func @test_transposable_on_the_fly_incorrect_transpose(
  %src : memref<2x8x16xf16>, %dst : memref<8x2x16xf16>, %cst : f16) {
  // expected-error@+1 {{failed to verify transpose behavior}}
  bishengir_test.transposableOTF ins(%src, %cst : memref<2x8x16xf16>, f16)
                                 outs(%dst : memref<8x2x16xf16>)
                                 transpose = [0, 1, 2]
  return
}

// -----

func.func @test_binary_vector_op_vv_vs(
  %src1 : memref<?x?xf16>, %src2 : memref<?x?xf16>, %dst : memref<?x?xf16>, %cst : f16) {
  hivm.hir.vadd ins(%src1, %src2 : memref<?x?xf16>, memref<?x?xf16>)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vmul ins(%src1, %src2 : memref<?x?xf16>, memref<?x?xf16>)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vsub ins(%src1, %src2 : memref<?x?xf16>, memref<?x?xf16>)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vdiv ins(%src1, %src2 : memref<?x?xf16>, memref<?x?xf16>)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vmax ins(%src1, %src2 : memref<?x?xf16>, memref<?x?xf16>)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vmin ins(%src1, %src2 : memref<?x?xf16>, memref<?x?xf16>)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vadd ins(%src1, %cst : memref<?x?xf16>, f16)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vmul ins(%src1, %cst : memref<?x?xf16>, f16)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vsub ins(%src1, %cst : memref<?x?xf16>, f16)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vmax ins(%src1, %cst : memref<?x?xf16>, f16)
                outs(%dst : memref<?x?xf16>)
  hivm.hir.vmin ins(%src1, %cst : memref<?x?xf16>, f16)
                outs(%dst : memref<?x?xf16>)
  return
}

// -----

func.func @test_binary_vector_op_vv_vs_int16(
  %src1 : memref<?x?xi16>, %src2 : memref<?x?xi16>, %dst : memref<?x?xi16>, %cst : i16) {
  hivm.hir.vor ins(%src1, %src2 : memref<?x?xi16>, memref<?x?xi16>)
               outs(%dst : memref<?x?xi16>)
  hivm.hir.vand ins(%src1, %src2 : memref<?x?xi16>, memref<?x?xi16>)
                outs(%dst : memref<?x?xi16>)
  hivm.hir.vshl ins(%src1, %cst : memref<?x?xi16>, i16)
                outs(%dst : memref<?x?xi16>)
  hivm.hir.vshr ins(%src1, %cst : memref<?x?xi16>, i16)
                outs(%dst : memref<?x?xi16>)
  hivm.hir.vshr ins(%src1, %cst : memref<?x?xi16>, i16)
                outs(%dst : memref<?x?xi16>)
                round: true
  hivm.hir.vshr ins(%src1, %cst : memref<?x?xi16>, i16)
                outs(%dst : memref<?x?xi16>)
                round: false
  return
}

// -----

func.func @test_binary_vector_op_only_vv_int16(
  %src1 : memref<?x?xi16>, %src2 : memref<?x?xi16>, %dst : memref<?x?xi16>, %cst : i16) {
  // expected-error@+1 {{failed to verify that operand at index 1 is vector-only}}
  hivm.hir.vor ins(%src1, %cst : memref<?x?xi16>, i16)
               outs(%dst : memref<?x?xi16>)
  return
}

// -----

func.func @test_binary_vector_op_only_vv_int16(
  %src1 : memref<?x?xi16>, %src2 : memref<?x?xi16>, %dst : memref<?x?xi16>, %cst : i16) {
  // expected-error@+1 {{failed to verify that operand at index 1 is vector-only}}
  hivm.hir.vand ins(%src1, %cst : memref<?x?xi16>, i16)
                outs(%dst : memref<?x?xi16>)
  return
}

// -----

func.func @test_vadd_vv_vs_with_broadcast(
  %src1 : memref<1x?xf16>, %src2 : memref<1x?xf16>, %dst : memref<?x?xf16>, %cst : f16) {
  hivm.hir.vadd ins(%src1, %src2 : memref<1x?xf16>, memref<1x?xf16>)
                outs(%dst : memref<?x?xf16>)
                broadcast = [0]

  hivm.hir.vadd ins(%src1, %cst : memref<1x?xf16>, f16)
                outs(%dst : memref<?x?xf16>)
                broadcast = [0]
  return
}

// -----

func.func @test_vadd_vv_mismatch_element_type(
  %src1 : memref<?x?xf16>, %src2 : memref<?x?xf16>, %dst : memref<?x?xf32>) {
  // expected-error@+1 {{requires the same element type for all operands}}
  hivm.hir.vadd ins(%src1, %src2 : memref<?x?xf16>, memref<?x?xf16>)
                outs(%dst : memref<?x?xf32>)
  return
}

// -----

func.func @test_vadd_vs_mismatch_element_type(
  %src1 : memref<?x?xf16>, %src2 : f32, %dst : memref<?x?xf16>) {
  // expected-error@+1 {{requires the same element type for all operands}}
  hivm.hir.vadd ins(%src1, %src2 : memref<?x?xf16>, f32)
                outs(%dst : memref<?x?xf16>)
  return
}

// -----

func.func @test_unary_vector_op(
  %src : memref<?x?xf16>, %dst : memref<?x?xf16>) {
  hivm.hir.vexp ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vabs ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vln ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vrelu ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vrsqrt ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vsqrt ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vrec ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vtanh ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vsin ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.vcos ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  hivm.hir.verf ins(%src : memref<?x?xf16>) outs(%dst : memref<?x?xf16>)
  return
}

// -----

func.func @test_unary_vector_op_int16_t(
  %src : memref<?x?xi16>, %dst : memref<?x?xi16>) {
  hivm.hir.vnot ins(%src : memref<?x?xi16>) outs(%dst : memref<?x?xi16>)
  return
}

// -----

func.func @test_vbrc_op_memref(%src1 : memref<1xi32>, %dst1 : memref<?xi32>,
                               %src2 : memref<1x1xi32>, %dst2 : memref<?x?xi32>,
                               %src3 : memref<?x1xi32>, %dst3 : memref<?x?xi32>) {
  hivm.hir.vbrc ins(%src1 : memref<1xi32>)
                outs(%dst1 : memref<?xi32>)
                broadcast_dims = [0]

  hivm.hir.vbrc ins(%src2 : memref<1x1xi32>)
                outs(%dst2 : memref<?x?xi32>)
                broadcast_dims = [0, 1]

  hivm.hir.vbrc ins(%src2 : memref<1x1xi32>)
                outs(%dst2 : memref<?x?xi32>)
                broadcast_dims = [1]

  hivm.hir.vbrc ins(%src3 : memref<?x1xi32>)
                outs(%dst3 : memref<?x?xi32>)
                broadcast_dims = [1]

  return
}

// -----

func.func @test_vbrc_op_tensor(%src1 : tensor<1xi32>, %dst1 : tensor<?xi32>,
                               %src2 : tensor<1x1xi32>, %dst2 : tensor<?x?xi32>) {
  %rst1 = hivm.hir.vbrc ins(%src1 : tensor<1xi32>)
                        outs(%dst1 : tensor<?xi32>)
                        broadcast_dims = [0] -> tensor<?xi32>

  %rst2 = hivm.hir.vbrc ins(%src2 : tensor<1x1xi32>)
                        outs(%dst2 : tensor<?x?xi32>)
                        broadcast_dims = [0, 1] -> tensor<?x?xi32>

  return
}

// -----

func.func @test_vbrc_op_scalar(%src : f32, %dst : memref<?xf32>,
                               %tdst : tensor<?xf32>) {
  hivm.hir.vbrc ins(%src : f32)
                outs(%dst : memref<?xf32>)

  hivm.hir.vbrc ins(%src : f32)
                outs(%dst : memref<?xf32>)
                broadcast_dims = []

  %rst = hivm.hir.vbrc ins(%src : f32)
                       outs(%tdst : tensor<?xf32>) -> tensor<?xf32>

  return
}

// -----

func.func @test_vbrc_op_custom_check_empty_brc_dims(%src1 : memref<1xi32>, %dst1 : memref<?xi32>) {
  // expected-error@+1 {{empty broadcast dims array}}
  hivm.hir.vbrc ins(%src1 : memref<1xi32>)
                outs(%dst1 : memref<?xi32>)
                broadcast_dims = []
  return
}

// -----

func.func @test_vbrc_op_custom_check_brc_dims_overflow(%src1 : memref<1xi32>, %dst1 : memref<?xi32>) {
  // expected-error@+1 {{too many indices}}
  hivm.hir.vbrc ins(%src1 : memref<1xi32>)
                outs(%dst1 : memref<?xi32>)
                broadcast_dims = [0, 1, 2]
  return
}

// -----

func.func @test_vbrc_op_custom_check_invalid_src_shape(%src1 : memref<2xi32>, %dst1 : memref<?xi32>) {
  // expected-error@+1 {{invalid source vector shape, 'SrcVecDim[0]' != 1}}
  hivm.hir.vbrc ins(%src1 : memref<2xi32>)
                outs(%dst1 : memref<?xi32>)
                broadcast_dims = [0]
  return
}

// -----

func.func @test_vbrc_op_custom_check_invalid_brc_dims_indices(%src1 : memref<1xi32>, %dst1 : memref<?xi32>) {
  // expected-error@+1 {{invalid index '1'}}
  hivm.hir.vbrc ins(%src1 : memref<1xi32>)
                outs(%dst1 : memref<?xi32>)
                broadcast_dims = [1]
  return
}

// -----

func.func @test_vbrc_op_custom_check_dims_for_scalar(%src : i32, %dst : memref<?xi32>) {
  // expected-error@+1 {{dims must be empty for scalar src}}
  hivm.hir.vbrc ins(%src : i32)
                outs(%dst : memref<?xi32>)
                broadcast_dims = [1]
  return
}

// -----

func.func @test_vbrc_op_check_temp_buffer_rank(%src1 : memref<1x10xi32>, %dst1 : memref<3x10xi32>) {
  %tmp_3 = memref.alloc() : memref<3x16xi32>

  // expected-error@+1 {{temp_buffer'rank should be one}}
  hivm.hir.vbrc ins(%src1 : memref<1x10xi32>)
                outs(%dst1 : memref<3x10xi32>)
                temp_buffer(%tmp_3 : memref<3x16xi32>)
                broadcast_dims = [1]
  return
}

// -----

func.func @test_vreduce_op_memref(%src1 : memref<?xi32>, %dst1 : memref<1xi32>,
                                  %src2 : memref<?x?xi32>, %dst2 : memref<?x1xi32>,
                                  %dst3 : memref<1x1xi32>, %dst_index : memref<1xi32>) {
  hivm.hir.vreduce <sum> ins(%src1 : memref<?xi32>)
                         outs(%dst1 : memref<1xi32>)
                         reduce_dims = [0]
  hivm.hir.vreduce <max> ins(%src1 : memref<?xi32>)
                         outs(%dst1 : memref<1xi32>)
                         reduce_dims = [0]
  hivm.hir.vreduce <min> ins(%src1 : memref<?xi32>)
                         outs(%dst1 : memref<1xi32>)
                         reduce_dims = [0]

  hivm.hir.vreduce <sum> ins(%src2 : memref<?x?xi32>)
                         outs(%dst2 : memref<?x1xi32>)
                         reduce_dims = [1]
  hivm.hir.vreduce <max> ins(%src2 : memref<?x?xi32>)
                         outs(%dst3 : memref<1x1xi32>)
                         reduce_dims = [0, 1]

  hivm.hir.vreduce <max_with_index_left> ins(%src1 : memref<?xi32>)
                         outs(%dst1, %dst_index : memref<1xi32>, memref<1xi32>)
                         reduce_dims = [0]
  hivm.hir.vreduce <min_with_index_left> ins(%src1 : memref<?xi32>)
                         outs(%dst1, %dst_index : memref<1xi32>, memref<1xi32>)
                         reduce_dims = [0]

  hivm.hir.vreduce <any> ins(%src1 : memref<?xi32>)
                         outs(%dst1 : memref<1xi32>)
                         reduce_dims = [0]

  hivm.hir.vreduce <all> ins(%src1 : memref<?xi32>)
                         outs(%dst1 : memref<1xi32>)
                         reduce_dims = [0]

  return
}

// -----

func.func @test_vreduce_op_tensor(%src2 : tensor<?x?xi32>, %dst2 : tensor<?x1xi32>,
                                  %dst3 : tensor<1x1xi32>, %dst_index : tensor<?x1xi32>) {
  %rst1 = hivm.hir.vreduce <sum> ins(%src2 : tensor<?x?xi32>)
                                 outs(%dst2 : tensor<?x1xi32>)
                                 reduce_dims = [1] -> tensor<?x1xi32>

  %rst2 = hivm.hir.vreduce <max> ins(%src2 : tensor<?x?xi32>)
                                 outs(%dst3 : tensor<1x1xi32>)
                                 reduce_dims = [0, 1] -> tensor<1x1xi32>

  %rst_value, %rst_index = hivm.hir.vreduce <max_with_index_left> ins(%src2 : tensor<?x?xi32>)
                                 outs(%dst2, %dst_index : tensor<?x1xi32>, tensor<?x1xi32>)
                                 reduce_dims = [1]
                                 -> tensor<?x1xi32>, tensor<?x1xi32>
  return
}

// -----

func.func @test_vreduce_op_custom_check_empty_dims(%src : memref<?xf32>,
                                                   %dst : memref<1xf32>) {
  // expected-error@+1 {{empty reduce dims}}
  hivm.hir.vreduce <max> ins(%src : memref<?xf32>)
                         outs(%dst : memref<1xf32>)
                         reduce_dims = []
  return
}

// -----

func.func @test_vreduce_op_custom_check_dims_overflow(%src : memref<?xf32>,
                                                      %dst : memref<1xf32>) {
  // expected-error@+1 {{too many indices}}
  hivm.hir.vreduce <max> ins(%src : memref<?xf32>)
                         outs(%dst : memref<1xf32>)
                         reduce_dims = [0, 1]
  return
}

// -----

func.func @test_vreduce_op_custom_check_invalid_dims_index(%src : memref<2x?x?xf32>,
                                                           %dst : memref<1x?x1xf32>) {
  // expected-error@+1 {{invalid index '3'}}
  hivm.hir.vreduce <max> ins(%src : memref<2x?x?xf32>)
                         outs(%dst : memref<1x?x1xf32>)
                         reduce_dims = [3]
  return
}

// -----

func.func @test_vreduce_op_custom_check_invalid_dst_shape(%src : memref<2x?x?xf32>,
                                                          %dst : memref<2x?x1xf32>) {
  // expected-error@+1 {{invalid dst vector shape, 'DstVecDim[0]' != 1}}
  hivm.hir.vreduce <max> ins(%src : memref<2x?x?xf32>)
                         outs(%dst : memref<2x?x1xf32>)
                         reduce_dims = [0, 2]
  return
}

// -----

func.func @test_vreduce_op_check_temp_buffer_rank(%src1 : memref<3x10xi32>, %dst1 : memref<1x10xi32>) {
  %tmp_3 = memref.alloc() : memref<3x16xi32>

  // expected-error@+1 {{temp_buffer'rank should be one}}
  hivm.hir.vreduce <max> ins(%src1 : memref<3x10xi32>)
                         outs(%dst1 : memref<1x10xi32>)
                         temp_buffer(%tmp_3 : memref<3x16xi32>)
                         reduce_dims = [0]
  return
}

// -----

func.func @test_vcast_tensor_op() {
  %f16 = tensor.empty() : tensor<2x16xf16>
  %f32 = tensor.empty() : tensor<2x16xf32>
  %s4 = tensor.empty() : tensor<2x16xi4>
  %s8 = tensor.empty() : tensor<2x16xi8>
  %s16 = tensor.empty() : tensor<2x16xi16>
  %s32 = tensor.empty() : tensor<2x16xi32>
  %s64 = tensor.empty() : tensor<2x16xi64>
  %bf16 = tensor.empty() : tensor<2x16xbf16>
  %u8 = tensor.empty() : tensor<2x16xui8>
  %res0 = hivm.hir.vcast ins(%bf16 : tensor<2x16xbf16>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf32>
  %res1 = hivm.hir.vcast ins(%bf16 : tensor<2x16xbf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xi32>
  %res2 = hivm.hir.vcast ins(%bf16 : tensor<2x16xbf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xi32>
  %res3 = hivm.hir.vcast ins(%bf16 : tensor<2x16xbf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xi32>
  %res4 = hivm.hir.vcast ins(%bf16 : tensor<2x16xbf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi32>
  %res5 = hivm.hir.vcast ins(%bf16 : tensor<2x16xbf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xi32>
  %res6 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf32>
  %res7 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xi16>
  %res8 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xi16>
  %res9 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xi16>
  %res10 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi16>
  %res11 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xi16>
  %res12 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xi32>
  %res13 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xi32>
  %res14 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xi32>
  %res15 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi32>
  %res16 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xi32>
  %res17 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s4 : tensor<2x16xi4>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi4>
  %res18 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s4 : tensor<2x16xi4>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xi4>
  %res19 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s4 : tensor<2x16xi4>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xi4>
  %res20 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s4 : tensor<2x16xi4>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xi4>
  %res21 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s4 : tensor<2x16xi4>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi4>
  %res22 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s4 : tensor<2x16xi4>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xi4>
  %res23 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s8 : tensor<2x16xi8>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi8>
  %res24 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s8 : tensor<2x16xi8>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xi8>
  %res25 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s8 : tensor<2x16xi8>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xi8>
  %res26 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s8 : tensor<2x16xi8>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xi8>
  %res27 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s8 : tensor<2x16xi8>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi8>
  %res28 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%s8 : tensor<2x16xi8>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xi8>
  %res29 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%u8 : tensor<2x16xui8>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xui8>
  %res30 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%u8 : tensor<2x16xui8>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xui8>
  %res31 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%u8 : tensor<2x16xui8>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xui8>
  %res32 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%u8 : tensor<2x16xui8>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xui8>
  %res33 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%u8 : tensor<2x16xui8>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xui8>
  %res34 = hivm.hir.vcast ins(%f16 : tensor<2x16xf16>) outs(%u8 : tensor<2x16xui8>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xui8>
  %res35 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%bf16 : tensor<2x16xbf16>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xbf16>
  %res36 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%bf16 : tensor<2x16xbf16>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xbf16>
  %res37 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%bf16 : tensor<2x16xbf16>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xbf16>
  %res38 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%bf16 : tensor<2x16xbf16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xbf16>
  %res39 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%bf16 : tensor<2x16xbf16>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xbf16>
  %res40 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf16>
  %res41 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xf16>
  %res42 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xf16>
  %res43 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xf16>
  %res44 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<odd> -> tensor<2x16xf16>
  %res45 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf16>
  %res46 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xf16>
  %res47 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xf32>
  %res48 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xf32>
  %res49 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xf32>
  %res50 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf32>
  %res51 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xf32>
  %res52 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xi16>
  %res53 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xi16>
  %res54 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xi16>
  %res55 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi16>
  %res56 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xi16>
  %res57 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xi32>
  %res58 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xi32>
  %res59 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xi32>
  %res60 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi32>
  %res61 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xi32>
  %res62 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s64 : tensor<2x16xi64>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xi64>
  %res63 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s64 : tensor<2x16xi64>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xi64>
  %res64 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s64 : tensor<2x16xi64>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xi64>
  %res65 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s64 : tensor<2x16xi64>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi64>
  %res66 = hivm.hir.vcast ins(%f32 : tensor<2x16xf32>) outs(%s64 : tensor<2x16xi64>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xi64>
  %res67 = hivm.hir.vcast ins(%s16 : tensor<2x16xi16>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf16>
  %res69 = hivm.hir.vcast ins(%s16 : tensor<2x16xi16>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xf16>
  %res70 = hivm.hir.vcast ins(%s16 : tensor<2x16xi16>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xf16>
  %res71 = hivm.hir.vcast ins(%s16 : tensor<2x16xi16>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xf16>
  %res72 = hivm.hir.vcast ins(%s16 : tensor<2x16xi16>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf16>
  %res73 = hivm.hir.vcast ins(%s16 : tensor<2x16xi16>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xf16>
  %res74 = hivm.hir.vcast ins(%s16 : tensor<2x16xi16>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf32>
  %res75 = hivm.hir.vcast ins(%s32 : tensor<2x16xi32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf32>
  %res76 = hivm.hir.vcast ins(%s32 : tensor<2x16xi32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xf32>
  %res77 = hivm.hir.vcast ins(%s32 : tensor<2x16xi32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xf32>
  %res78 = hivm.hir.vcast ins(%s32 : tensor<2x16xi32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xf32>
  %res79 = hivm.hir.vcast ins(%s32 : tensor<2x16xi32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf32>
  %res80 = hivm.hir.vcast ins(%s32 : tensor<2x16xi32>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xf32>
  %res81 = hivm.hir.vcast ins(%s32 : tensor<2x16xi32>) outs(%s16 : tensor<2x16xi16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi16>
  %res82 = hivm.hir.vcast ins(%s32 : tensor<2x16xi32>) outs(%s64 : tensor<2x16xi64>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi64>
  %res83 = hivm.hir.vcast ins(%s4 : tensor<2x16xi4>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf16>
  %res84 = hivm.hir.vcast ins(%s64 : tensor<2x16xi64>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<round> -> tensor<2x16xf32>
  %res85 = hivm.hir.vcast ins(%s64 : tensor<2x16xi64>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<ceil> -> tensor<2x16xf32>
  %res86 = hivm.hir.vcast ins(%s64 : tensor<2x16xi64>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<floor> -> tensor<2x16xf32>
  %res87 = hivm.hir.vcast ins(%s64 : tensor<2x16xi64>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf32>
  %res88 = hivm.hir.vcast ins(%s64 : tensor<2x16xi64>) outs(%f32 : tensor<2x16xf32>)
                   round_mode = #hivm.round_mode<trunc> -> tensor<2x16xf32>
  %res89 = hivm.hir.vcast ins(%s64 : tensor<2x16xi64>) outs(%s32 : tensor<2x16xi32>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xi32>
  %res90 = hivm.hir.vcast ins(%s8 : tensor<2x16xi8>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf16>
  %res91 = hivm.hir.vcast ins(%u8 : tensor<2x16xui8>) outs(%f16 : tensor<2x16xf16>)
                   round_mode = #hivm.round_mode<rint> -> tensor<2x16xf16>
  return
}

// -----

func.func @test_vcast_op() {
  %f16 = memref.alloc() : memref<2x16xf16>
  %f32 = memref.alloc() : memref<2x16xf32>
  %s4 = memref.alloc() : memref<2x16xi4>
  %s8 = memref.alloc() : memref<2x16xi8>
  %s16 = memref.alloc() : memref<2x16xi16>
  %s32 = memref.alloc() : memref<2x16xi32>
  %s64 = memref.alloc() : memref<2x16xi64>
  %bf16 = memref.alloc() : memref<2x16xbf16>
  %u8 = memref.alloc() : memref<2x16xui8>
  hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%bf16 : memref<2x16xbf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s4 : memref<2x16xi4>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%s8 : memref<2x16xi8>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f16 : memref<2x16xf16>) outs(%u8 : memref<2x16xui8>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%bf16 : memref<2x16xbf16>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<odd>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%f32 : memref<2x16xf32>) outs(%s64 : memref<2x16xi64>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%s16 : memref<2x16xi16>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%s16 : memref<2x16xi16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s32 : memref<2x16xi32>) outs(%s64 : memref<2x16xi64>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s4 : memref<2x16xi4>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<round>
  hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<ceil>
  hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<floor>
  hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%f32 : memref<2x16xf32>)
                 round_mode = #hivm.round_mode<trunc>
  hivm.hir.vcast ins(%s64 : memref<2x16xi64>) outs(%s32 : memref<2x16xi32>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%s8 : memref<2x16xi8>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<rint>
  hivm.hir.vcast ins(%u8 : memref<2x16xui8>) outs(%f16 : memref<2x16xf16>)
                 round_mode = #hivm.round_mode<rint>
  return
}

// -----

func.func @test_vcmp_op_memref() {
  %a_f16 = memref.alloc() : memref<2x16xf16>
  %b_f16 = memref.alloc() : memref<2x16xf16>
  %a_f32 = memref.alloc() : memref<2x16xf32>
  %b_f32 = memref.alloc() : memref<2x16xf32>
  %a_i32 = memref.alloc() : memref<2x16xi32>
  %b_i32 = memref.alloc() : memref<2x16xi32>
  %c_i1 = memref.alloc() : memref<2x16xi1>
  hivm.hir.vcmp ins(%a_f16, %b_f16: memref<2x16xf16>, memref<2x16xf16>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<eq>
  hivm.hir.vcmp ins(%a_f16, %b_f16: memref<2x16xf16>, memref<2x16xf16>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<ne>
  hivm.hir.vcmp ins(%a_f16, %b_f16: memref<2x16xf16>, memref<2x16xf16>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<lt>
  hivm.hir.vcmp ins(%a_f16, %b_f16: memref<2x16xf16>, memref<2x16xf16>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<gt>
  hivm.hir.vcmp ins(%a_f16, %b_f16: memref<2x16xf16>, memref<2x16xf16>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<ge>
  hivm.hir.vcmp ins(%a_f16, %b_f16: memref<2x16xf16>, memref<2x16xf16>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<le>

  hivm.hir.vcmp ins(%a_f32, %b_f32: memref<2x16xf32>, memref<2x16xf32>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<eq>
  hivm.hir.vcmp ins(%a_f32, %b_f32: memref<2x16xf32>, memref<2x16xf32>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<ne>
  hivm.hir.vcmp ins(%a_f32, %b_f32: memref<2x16xf32>, memref<2x16xf32>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<lt>
  hivm.hir.vcmp ins(%a_f32, %b_f32: memref<2x16xf32>, memref<2x16xf32>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<gt>
  hivm.hir.vcmp ins(%a_f32, %b_f32: memref<2x16xf32>, memref<2x16xf32>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<ge>
  hivm.hir.vcmp ins(%a_f32, %b_f32: memref<2x16xf32>, memref<2x16xf32>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<le>

  hivm.hir.vcmp ins(%a_i32, %b_i32: memref<2x16xi32>, memref<2x16xi32>)
                outs(%c_i1 : memref<2x16xi1>)
                compare_mode = #hivm.compare_mode<eq>
  return
}

// -----

func.func @test_vcmp_op_tensor(%a_f16 : tensor<2x16xf16>, %b_f16 : tensor<2x16xf16>,
                               %a_f32 : tensor<2x16xf32>, %b_f32 : tensor<2x16xf32>,
                               %a_i32 : tensor<2x16xi32>, %b_i32 : tensor<2x16xi32>,
                               %c_i1: tensor<2x16xi1>) {
  %res0 = hivm.hir.vcmp ins(%a_f16, %b_f16: tensor<2x16xf16>, tensor<2x16xf16>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<eq> -> tensor<2x16xi1>
  %res1 = hivm.hir.vcmp ins(%a_f16, %b_f16: tensor<2x16xf16>, tensor<2x16xf16>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<ne> -> tensor<2x16xi1>
  %res2 = hivm.hir.vcmp ins(%a_f16, %b_f16: tensor<2x16xf16>, tensor<2x16xf16>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<lt> -> tensor<2x16xi1>
  %res3 = hivm.hir.vcmp ins(%a_f16, %b_f16: tensor<2x16xf16>, tensor<2x16xf16>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<gt> -> tensor<2x16xi1>
  %res4 = hivm.hir.vcmp ins(%a_f16, %b_f16: tensor<2x16xf16>, tensor<2x16xf16>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<ge> -> tensor<2x16xi1>
  %res5 = hivm.hir.vcmp ins(%a_f16, %b_f16: tensor<2x16xf16>, tensor<2x16xf16>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<le> -> tensor<2x16xi1>

  %res6 = hivm.hir.vcmp ins(%a_f32, %b_f32: tensor<2x16xf32>, tensor<2x16xf32>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<eq> -> tensor<2x16xi1>
  %res7 = hivm.hir.vcmp ins(%a_f32, %b_f32: tensor<2x16xf32>, tensor<2x16xf32>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<ne> -> tensor<2x16xi1>
  %res8 = hivm.hir.vcmp ins(%a_f32, %b_f32: tensor<2x16xf32>, tensor<2x16xf32>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<lt> -> tensor<2x16xi1>
  %res9 = hivm.hir.vcmp ins(%a_f32, %b_f32: tensor<2x16xf32>, tensor<2x16xf32>)
                        outs(%c_i1: tensor<2x16xi1>)
                        compare_mode = #hivm.compare_mode<gt> -> tensor<2x16xi1>
  %res10 = hivm.hir.vcmp ins(%a_f32, %b_f32: tensor<2x16xf32>, tensor<2x16xf32>)
                         outs(%c_i1: tensor<2x16xi1>)
                         compare_mode = #hivm.compare_mode<ge> -> tensor<2x16xi1>
  %res11 = hivm.hir.vcmp ins(%a_f32, %b_f32: tensor<2x16xf32>, tensor<2x16xf32>)
                         outs(%c_i1: tensor<2x16xi1>)
                         compare_mode = #hivm.compare_mode<le> -> tensor<2x16xi1>

  %res12 = hivm.hir.vcmp ins(%a_i32, %b_i32: tensor<2x16xi32>, tensor<2x16xi32>)
                         outs(%c_i1: tensor<2x16xi1>)
                         compare_mode = #hivm.compare_mode<eq> -> tensor<2x16xi1>
  return
}

// -----

func.func @test_vsel_op_memref() {
  %a_f16 = memref.alloc() : memref<2x16xf16>
  %b_f16 = memref.alloc() : memref<2x16xf16>
  %a_f32 = memref.alloc() : memref<2x16xf32>
  %b_f32 = memref.alloc() : memref<2x16xf32>
  %c_i1 = memref.alloc() : memref<2x16xi1>
  %d_f16 = memref.alloc() : memref<2x16xf16>
  %d_f32 = memref.alloc() : memref<2x16xf32>
  hivm.hir.vsel ins(%c_i1, %a_f16, %b_f16: memref<2x16xi1>, memref<2x16xf16>, memref<2x16xf16>)
                outs(%d_f16 : memref<2x16xf16>)
  hivm.hir.vsel ins( %c_i1, %a_f32, %b_f32: memref<2x16xi1>, memref<2x16xf32>, memref<2x16xf32>)
                outs(%d_f32 : memref<2x16xf32>)
  return
}

// -----

func.func @test_vsel_op_tensor(%a_f16 : tensor<2x16xf16>, %b_f16 : tensor<2x16xf16>,
                               %a_f32 : tensor<2x16xf32>, %b_f32 : tensor<2x16xf32>,
                               %c_i1 : tensor<2x16xi1>,
                               %d_f16 : tensor<2x16xf16>,
                               %d_f32 : tensor<2x16xf32>) {
  %res0 = hivm.hir.vsel ins(%c_i1, %a_f16, %b_f16: tensor<2x16xi1>, tensor<2x16xf16>, tensor<2x16xf16>)
                        outs(%d_f16 : tensor<2x16xf16>) -> tensor<2x16xf16>
  %res1 = hivm.hir.vsel ins(%c_i1, %a_f32, %b_f32: tensor<2x16xi1>, tensor<2x16xf32>, tensor<2x16xf32>)
                        outs(%d_f32 : tensor<2x16xf32>) -> tensor<2x16xf32>
  return
}

// -----
func.func @test_vinterleave_op() {
  %a_f16 = memref.alloc() : memref<2x16xf16>
  %b_f16 = memref.alloc() : memref<2x16xf16>
  %c_f16 = memref.alloc() : memref<2x32xf16>
  hivm.hir.vinterleave ins(%a_f16, %b_f16 : memref<2x16xf16>, memref<2x16xf16>)
                outs(%c_f16 : memref<2x32xf16>)
                interleave_channel_nums = 2
  return
}

// -----
func.func @test_vdeinterleave_op() {
  %input_f16 = memref.alloc() : memref<2x32xf16>
  %output_even_f16 = memref.alloc() : memref<2x16xf16>
  %output_odd_f16 = memref.alloc() : memref<2x16xf16>
  hivm.hir.vdeinterleave ins(%input_f16 : memref<2x32xf16>)
    outs(%output_even_f16, %output_odd_f16 : memref<2x16xf16>, memref<2x16xf16>)
    index_mode = <ALL_CHANNELS>
  return
}

// -----
func.func @test_vflip_op_memref() {
  %src = memref.alloc() : memref<2x16xf16>
  %dst = memref.alloc() : memref<2x16xf16>
  hivm.hir.vflip ins(%src: memref<2x16xf16>)
                outs(%dst : memref<2x16xf16>)
                flip_axis = 1
  return
}

// -----
func.func @test_vpad_op_tensor(
  %src : tensor<2x16xf32>,
  %dst : tensor<?x16xf32>,
  %pad_value : f32,
  %first_dim_low : index,
  %first_dim_high : index
) {
  %result = hivm.hir.vpad ins(%src : tensor<2x16xf32>)
                          outs(%dst : tensor<?x16xf32>)
                          low[%first_dim_low, 0]
                          high[%first_dim_high, 0]
                          pad_value %pad_value : f32
                          -> tensor<?x16xf32>
  return
}

// -----
func.func @test_vpad_op_memref(
  %pad_value : f16
) {
  %src = memref.alloc() : memref<10xf16>
  %dst = memref.alloc() : memref<12xf16>
  hivm.hir.vpad ins(%src : memref<10xf16>)
                outs(%dst : memref<12xf16>)
                low[1]
                high[1]
                pad_value %pad_value : f16
  return
}

// -----
func.func @test_vconcat_op_tensor(%a_f16 : tensor<2x16xf16>, %b_f16 : tensor<2x16xf16>,
                               %c_f16 : tensor<2x32xf16>) {
  %res0 = hivm.hir.vconcat dim(1) ins(%a_f16, %b_f16: tensor<2x16xf16>, tensor<2x16xf16>)
                                  outs(%c_f16 : tensor<2x32xf16>) -> tensor<2x32xf16>
  return
}

// -----
func.func @test_vconcat_op_memref() {
  %a_f16 = memref.alloc() : memref<2x16xf16>
  %b_f16 = memref.alloc() : memref<2x16xf16>
  %c_f16 = memref.alloc() : memref<2x32xf16>
  hivm.hir.vconcat dim(1) ins(%a_f16, %b_f16: memref<2x16xf16>, memref<2x16xf16>)
                   outs(%c_f16 : memref<2x32xf16>)

  return
}

// -----
func.func @test_vpow_1d_op() {
  %0 = memref.alloc() : memref<32xi32>
  %1 = memref.alloc() : memref<32xi32>
  %2 = memref.alloc() : memref<32xi32>
  %3 = memref.alloc() : memref<136xi32>
  hivm.hir.vpow ins(%0, %1 : memref<32xi32>, memref<32xi32>)
                outs(%2 : memref<32xi32>)
                temp_buffer(%3 : memref<136xi32>)
  return
}

// -----

func.func @test_cumsum_memref(%src : memref<2x?x?xf32>, %dst : memref<2x?x?xf32>) {
  %s16 = memref.alloc() : memref<2x16xi16>
  %s32 = memref.alloc() : memref<2x16xi32>
  %s64 = memref.alloc() : memref<2x16xi64>
  %f16 = memref.alloc() : memref<2x16xf16>
  %f32 = memref.alloc() : memref<2x16xf32>
  hivm.hir.vcumsum ins(%s16 : memref<2x16xi16>) outs(%s16 : memref<2x16xi16>) cum_dims = [0] reverse = false
  hivm.hir.vcumsum ins(%s32 : memref<2x16xi32>) outs(%s32 : memref<2x16xi32>) cum_dims = [1] reverse = false
  hivm.hir.vcumsum ins(%s64 : memref<2x16xi64>) outs(%s64 : memref<2x16xi64>) cum_dims = [1] reverse = false
  hivm.hir.vcumsum ins(%f16 : memref<2x16xf16>) outs(%f16 : memref<2x16xf16>) cum_dims = [0] reverse = false
  hivm.hir.vcumsum ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>) cum_dims = [1] reverse = false
  hivm.hir.vcumsum ins(%src : memref<2x?x?xf32>) outs(%dst : memref<2x?x?xf32>) cum_dims = [2] reverse = false
  return
}

// -----

func.func @test_cumprod_memref(%src : memref<2x?x?xf32>, %dst : memref<2x?x?xf32>) {
  %s16 = memref.alloc() : memref<2x16xi16>
  %s32 = memref.alloc() : memref<2x16xi32>
  %s64 = memref.alloc() : memref<2x16xi64>
  %f16 = memref.alloc() : memref<2x16xf16>
  %f32 = memref.alloc() : memref<2x16xf32>
  hivm.hir.vcumprod ins(%s16 : memref<2x16xi16>) outs(%s16 : memref<2x16xi16>) cum_dims = [0] reverse = false
  hivm.hir.vcumprod ins(%s32 : memref<2x16xi32>) outs(%s32 : memref<2x16xi32>) cum_dims = [1] reverse = false
  hivm.hir.vcumprod ins(%s64 : memref<2x16xi64>) outs(%s64 : memref<2x16xi64>) cum_dims = [0] reverse = false
  hivm.hir.vcumprod ins(%f16 : memref<2x16xf16>) outs(%f16 : memref<2x16xf16>) cum_dims = [0] reverse = false
  hivm.hir.vcumprod ins(%f32 : memref<2x16xf32>) outs(%f32 : memref<2x16xf32>) cum_dims = [1] reverse = false
  hivm.hir.vcumprod ins(%src : memref<2x?x?xf32>) outs(%dst : memref<2x?x?xf32>) cum_dims = [2] reverse = false
  return
}

// -----

func.func @test_cumsum_tensor(%src : tensor<2x?x?xf32>, %dst : tensor<2x?x?xf32>) {
  %s16 = tensor.empty() : tensor<2x16xi16>
  %s32 = tensor.empty() : tensor<2x16xi32>
  %s64 = tensor.empty() : tensor<2x16xi64>
  %f16 = tensor.empty() : tensor<2x16xf16>
  %f32 = tensor.empty() : tensor<2x16xf32>
  %res1 = hivm.hir.vcumsum ins(%s16 : tensor<2x16xi16>) outs(%s16 : tensor<2x16xi16>) cum_dims = [0] reverse = false -> tensor<2x16xi16>
  %res2 = hivm.hir.vcumsum ins(%s32 : tensor<2x16xi32>) outs(%s32 : tensor<2x16xi32>) cum_dims = [1] reverse = false -> tensor<2x16xi32>
  %res3 = hivm.hir.vcumsum ins(%s64 : tensor<2x16xi64>) outs(%s64 : tensor<2x16xi64>) cum_dims = [0] reverse = false -> tensor<2x16xi64>
  %res4 = hivm.hir.vcumsum ins(%f16 : tensor<2x16xf16>) outs(%f16 : tensor<2x16xf16>) cum_dims = [0] reverse = false -> tensor<2x16xf16>
  %res5 = hivm.hir.vcumsum ins(%f32 : tensor<2x16xf32>) outs(%f32 : tensor<2x16xf32>) cum_dims = [1] reverse = false -> tensor<2x16xf32>
  %res6 = hivm.hir.vcumsum ins(%src : tensor<2x?x?xf32>) outs(%dst : tensor<2x?x?xf32>) cum_dims = [2] reverse = false -> tensor<2x?x?xf32>
  return
}

// -----

func.func @test_cumprod_tensor(%src : tensor<2x?x?xf32>, %dst : tensor<2x?x?xf32>) {
  %s16 = tensor.empty() : tensor<2x16xi16>
  %s32 = tensor.empty() : tensor<2x16xi32>
  %s64 = tensor.empty() : tensor<2x16xi64>
  %f16 = tensor.empty() : tensor<2x16xf16>
  %f32 = tensor.empty() : tensor<2x16xf32>
  %res1 = hivm.hir.vcumprod ins(%s16 : tensor<2x16xi16>) outs(%s16 : tensor<2x16xi16>) cum_dims = [0] reverse = false -> tensor<2x16xi16>
  %res2 = hivm.hir.vcumprod ins(%s32 : tensor<2x16xi32>) outs(%s32 : tensor<2x16xi32>) cum_dims = [1] reverse = false -> tensor<2x16xi32>
  %res3 = hivm.hir.vcumprod ins(%s64 : tensor<2x16xi64>) outs(%s64 : tensor<2x16xi64>) cum_dims = [1] reverse = false -> tensor<2x16xi64>
  %res4 = hivm.hir.vcumprod ins(%f16 : tensor<2x16xf16>) outs(%f16 : tensor<2x16xf16>) cum_dims = [0] reverse = false -> tensor<2x16xf16>
  %res5 = hivm.hir.vcumprod ins(%f32 : tensor<2x16xf32>) outs(%f32 : tensor<2x16xf32>) cum_dims = [1] reverse = false -> tensor<2x16xf32>
  %res6 = hivm.hir.vcumprod ins(%src : tensor<2x?x?xf32>) outs(%dst : tensor<2x?x?xf32>) cum_dims = [2] reverse = false -> tensor<2x?x?xf32>
  return
}

// -----


func.func @test_varange_op_with_offset(%offset: index, %stride0: index, %stride1: index) -> tensor<2x4xi32> {
  %dst = tensor.empty() : tensor<2x4xi32>
  %result = hivm.hir.varange offset[%offset] strides[%stride0, %stride1] outs(%dst : tensor<2x4xi32>) -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}

func.func @test_varange_op(%stride0: index, %stride1: index) -> tensor<2x4xi32> {
  %dst = tensor.empty() : tensor<2x4xi32>
  %result = hivm.hir.varange strides[%stride0, %stride1] outs(%dst : tensor<2x4xi32>) -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}