// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics


// invalid data layout attribute
func.func @invalid_data_layout() {
  "test.invalid_data_layout"() {
      // expected-error@+1 {{'transpose' is only valid if data layout is DOTA_ND or DOTB_ND}}
      nZ_transpose = #hivm.data_layout<nZ, transpose = true>
    } : () -> ()
  return
}

// -----

// invalid data layout attribute
func.func @invalid_data_layout() {
  "test.invalid_data_layout"() {
      // expected-error@+1 {{'transpose' must be set if data layout is DOTA_ND or DOTB_ND}}
      nZ_transpose = #hivm.data_layout<dotA_ND>
    } : () -> ()
  return
}

// -----

// CHECK-LABEL: test_invalid_convert_layout
#dot_a_layout = #hivm.data_layout<dotA_ND, transpose = false>
#nZ_layout = #hivm.data_layout<nZ>
func.func @test_invalid_convert_layout(%arg : memref<128x128xf16, strided<[?, ?], offset: ?>>) {
  %alloc = memref.alloc() : memref<128x128xf16>
  memref.copy %arg, %alloc : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<128x128xf16>
  // expected-error@+1 {{'hivm.hir.convert_layout' op requires the same element type for all operands and results}}
  %alloc_new_layout = hivm.hir.convert_layout %alloc {srcLayout = #dot_a_layout, dstLayout = #nZ_layout} : (memref<128x128xf16>) -> memref<8x8x16x16xf32>
  "some_use"(%alloc_new_layout) : (memref<8x8x16x16xf32>) -> ()
  return
}

// -----
// CHECK-LABEL: incorrect_copy_op_gm_gm
func.func @incorrect_copy_op_gm_gm() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  // expected-error@+1 {{'hivm.hir.copy' op Unsupported copy from gm to gm!}}
  hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
  return
}

// -----
// CHECK-LABEL: incorrect_copy_op_element_type
func.func @incorrect_copy_op_element_type() {
  %src = memref.alloc() : memref<16x16xf32, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // expected-error@+1 {{'hivm.hir.load' op element types of dst and src should be the same!}}
  hivm.hir.load ins(%src : memref<16x16xf32, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
  return
}


// -----
// CHECK-LABEL: incorrect_copy_op_padval_type
func.func @incorrect_copy_op_padval_type() {
  %val = arith.constant 10.0 : f32
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // expected-error@+1 {{'hivm.hir.load' op dtype of pad_value and element type of dst/src should be the same!}}
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                pad_mode = #hivm.padmode<PadValue>
                pad_value = %val : f32
  return
}

// -----
// CHECK-LABEL: incorrect_copy_op_tensor_memref_mixed
func.func @incorrect_copy_op_tensor_memref_mixed() {
	%src = tensor.empty() : tensor<16x16xf16>
	%dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // expected-error@+1 {{'hivm.hir.copy' op dst/src should be memref/memref or tensor/tensor, res should be tensor!}}
  hivm.hir.copy ins(%src : tensor<16x16xf16>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
	return
}

// -----
// CHECK-LABEL: incorrect_copy_op_pad_value
func.func @incorrect_copy_op_pad_value() {
	%src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
	%dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  // expected-error@+1 {{'hivm.hir.load' op if padmode is PadValue, pad_value is required!}}
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
                pad_mode = #hivm.padmode<PadValue>
	return
}

// -----
// CHECK-LABEL: incorrect_copy_op_shape
func.func @incorrect_copy_op_shape() {
	%src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
	%dst = memref.alloc() : memref<256xf16, #hivm.address_space<ub>>
  // expected-error@+1 {{'hivm.hir.load' op src and dst should have the same dimensions!}}
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<256xf16, #hivm.address_space<ub>>)
	return
}

// -----
// CHECK-LABEL: incorrect_matmul_descale_shape
func.func @incorrect_matmul_descale_shape(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                               %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                               %descale_pertensor_gm :  memref<16xf16, #hivm.address_space<gm>>,
                               %res_gm :memref<16x16xf16, #hivm.address_space<gm>> ) {
    // expected-error@+1 {{'hivm.hir.matmul' op The descaleMode is DescalePerTensor, the size of descale is equal to 1}}
    hivm.hir.matmul
         ins(%A_gm, %B_gm:
             memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
         outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
         descale = %descale_pertensor_gm :  memref<16xf16, #hivm.address_space<gm>>
         descale_mode = #hivm.descale_mode<DescalePerTensor>
  return
}


// -----
// CHECK-LABEL: incorrect_matmul_bias_shape
func.func @incorrect_matmul_bias_shape(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                               %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                               %bias_gm :  memref<32xf16, #hivm.address_space<gm>>,
                               %res_gm :memref<16x16xf16, #hivm.address_space<gm>> ) {
    // expected-error@+1 {{'hivm.hir.matmul' op The size of bias is equal to the col size of B}}
    hivm.hir.matmul
         ins(%A_gm, %B_gm:
             memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
         outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
         bias = %bias_gm :  memref<32xf16, #hivm.address_space<gm>>
  return
}


// -----
module {
// CHECK-LABEL: test_ptr0
  func.func @test_ptr0() {
    // expected-error@+1 {{'hivm.hir.pointer_cast' op addrs of PointerCastOp should not be empty!}}
    %1 = hivm.hir.pointer_cast() [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    return
  }
}

// -----
module {
// CHECK-LABEL: incorrect_tranpose_permutation
  func.func @incorrect_tranpose_permutation() {
    %src = memref.alloc() : memref<16x8x32x8xf32>
    %dst = memref.alloc() : memref<8x16x8x32xf32>
    // expected-error@+1 {{'hivm.hir.vtranspose' op Vtranspose only support two axes transpose}}
    hivm.hir.vtranspose ins(%src : memref<16x8x32x8xf32>) outs(%dst : memref<8x16x8x32xf32>) permutation = [1, 0, 3, 2]
    return
  }
}

// -----
module {
// CHECK-LABEL: incorrect_tranpose_permutation
  func.func @incorrect_load_op_left_pad_value_type() {
    %c0_i64 = arith.constant 0 : i64
    %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
    %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
    // expected-error@+1 {{'hivm.hir.load' op operand #2 must be index, but got 'i64'}}
    hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                  outs(%dst :  memref<16x16xf16, #hivm.address_space<ub>>)
                  left_padding_num = %c0_i64 : i64
    return
  }
}

// -----
module {
// CHECK-LABEL: alloc_workspace
  func.func @alloc_workspace(%dyn_size : index){
    // expected-error@+1 {{'memref_ext.alloc_workspace' op dimension operand count does not equal memref dynamic dimension count}}
    memref_ext.alloc_workspace(%dyn_size) : memref<100xi8>

    return
  }
}

// -----

func.func @transpose_broadcast_otf(%src0 : memref<4x8x16xf16>, %src1 : memref<4x1x16xf16>, %dst : memref<8x4x16xf16>) {
  // expected-error@+1 {{Broadcast OTF and Transpose OTF cannot be enabled at the same time}}
  hivm.hir.vadd ins(%src0, %src1 : memref<4x8x16xf16>, memref<4x1x16xf16>) outs(%dst : memref<8x4x16xf16>)
    broadcast = [1]
    transpose = [1, 0, 2]
  return
}

// -----
// CHECK-LABEL: invalid_create_sync_block_lock
func.func @invalid_create_sync_block_lock() {
  // expected-error@+1 {{'create_sync_block_lock' op should only support static shape}}
  hivm.hir.create_sync_block_lock : memref<?xi64>
  return
}

// -----
// CHECK-LABEL: hivm_load_type
func.func @hivm_load_type(%src: tensor<16x16xf16>, %dst: tensor<16x16xf32>) {
  // expected-error@+1 {{'hivm.hir.load' op element types of dst and src should be the same!}}
  %res = hivm.hir.load ins(%src : tensor<16x16xf16>)
                outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
  return
}

// -----
// CHECK-LABEL: hivm_load_dim
func.func @hivm_load_dim(%src: tensor<16x16xf16>, %dst: tensor<16x32xf16>) {
  // expected-error@+1 {{'hivm.hir.load' op if pad_mode is not set, src and dst shape should be the same!}}
  %res = hivm.hir.load ins(%src : tensor<16x16xf16>)
                outs(%dst : tensor<16x32xf16>) -> tensor<16x32xf16>
  return
}

// -----
// CHECK-LABEL: hivm_load_dims
func.func @hivm_load_dims(%src: tensor<16x16x2xf16>, %dst: tensor<16x16xf16>) {
  // expected-error@+1 {{'hivm.hir.load' op src and dst should have the same dimensions!}}
  %res = hivm.hir.load ins(%src : tensor<16x16x2xf16>)
                outs(%dst : tensor<16x16xf16>) -> tensor<16x16xf16>
  return
}

// -----
// CHECK-LABEL: hivm_load_rank
func.func @hivm_load_dim(%src: tensor<16x16xf16>, %dst: memref<16x16xf16, #hivm.address_space<gm>>) {
  // expected-error@+1 {{'hivm.hir.load' op dst/src should be memref/memref or tensor/tensor, res should be tensor!}}
  hivm.hir.load ins(%src : tensor<16x16xf16>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
  return
}

// -----
// CHECK-LABEL: hivm_memref_load_gm_to_gm_fail
func.func @hivm_memref_load_gm_to_gm_fail() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  // expected-error@+1 {{'hivm.hir.load' op only support src == gm and dst != gm currently!}}
  hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
  return
}

// -----
// CHECK-LABEL: hivm_store_type
func.func @hivm_store_type(%src: tensor<16x16xf16>, %dst: tensor<16x16xf32>) {
  // expected-error@+1 {{'hivm.hir.store' op element types of dst and src should be the same!}}
  %res = hivm.hir.store ins(%src : tensor<16x16xf16>)
                outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
  return
}

// -----
// CHECK-LABEL: hivm_store_dims
func.func @hivm_store_dims(%src: tensor<16x16x2xf16>, %dst: tensor<16x16xf16>) {
  // expected-error@+1 {{'hivm.hir.store' op src and dst should have the same dimensions!}}
  %res = hivm.hir.store ins(%src : tensor<16x16x2xf16>)
                outs(%dst : tensor<16x16xf16>) -> tensor<16x16xf16>
  return
}

// -----
// CHECK-LABEL: hivm_memref_store_gm_to_gm_fail
func.func @hivm_memref_store_gm_to_gm_fail() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  // expected-error@+1 {{'hivm.hir.store' op only support copy gm to ub or copy ub to gm or copy ub to ub currently!}}
  hivm.hir.store ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
  return
}

// -----
// CHECK-LABEL: hivm_copy_type
func.func @hivm_copy_type(%src: tensor<16x16xf16>, %dst: tensor<16x16xf32>) {
  // expected-error@+1 {{'hivm.hir.copy' op element types of dst and src should be the same!}}
  %res = hivm.hir.copy ins(%src : tensor<16x16xf16>)
                outs(%dst : tensor<16x16xf32>) -> tensor<16x16xf32>
  return
}

// -----
// CHECK-LABEL: hivm_copy_dims
func.func @hivm_copy_dims(%src: tensor<16x16x2xf16>, %dst: tensor<16x16xf16>) {
  // expected-error@+1 {{'hivm.hir.copy' op src and dst should have the same dimensions!}}
  %res = hivm.hir.copy ins(%src : tensor<16x16x2xf16>)
                outs(%dst : tensor<16x16xf16>) -> tensor<16x16xf16>
  return
}

// -----
// CHECK-LABEL: test_mix_group_matmul_invalid_a
func.func @test_mix_group_matmul_invalid_a(%weight_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                                 %tokens_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                                 %tokens_per_expert_gm :  memref<16xi64, #hivm.address_space<gm>>,
                                 %tiling_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                 %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                 %res_gm :memref<16x16xf16, #hivm.address_space<gm>>,
                                 %post_vector_func_ins0_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_ins1_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_ins2_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_outs_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %workspace_ins_gm: memref<1024xi64, #hivm.address_space<gm>>) {
    // expected-error@+1 {{'hivm.hir.mix_group_matmul' op matrix A must be 3D}}
    hivm.hir.mix_group_matmul
       ins(%weight_gm, %tokens_gm, %tokens_per_expert_gm:
           memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>,
           memref<16xi64, #hivm.address_space<gm>>)
       post_vector_func_ins(%post_vector_func_ins0_gm,
                            %post_vector_func_ins1_gm,
                            %post_vector_func_ins2_gm :
                            memref<16xi64, #hivm.address_space<gm>>,
                            memref<16xi64, #hivm.address_space<gm>>,
                            memref<16xi64, #hivm.address_space<gm>>)
       post_vector_func_outs(%post_vector_func_outs_gm : memref<16xi64, #hivm.address_space<gm>>)
       workspace_ins(%workspace_ins_gm : memref<1024xi64, #hivm.address_space<gm>>)
       outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       tiling_params = %tiling_params_gm : memref<16xi64, #hivm.address_space<gm>>
       comm_params = %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>
    return
}

// -----
// CHECK-LABEL: test_mix_group_matmul_invalid_b
func.func @test_mix_group_matmul_invalid_b(%weight_gm : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %tokens_gm :  memref<16xf16, #hivm.address_space<gm>>,
                                 %tokens_per_expert_gm :  memref<16xi64, #hivm.address_space<gm>>,
                                 %tiling_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                 %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                 %res_gm :memref<16x16xf16, #hivm.address_space<gm>>,
                                 %post_vector_func_ins0_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_ins1_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_ins2_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_outs_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %workspace_ins_gm: memref<1024xi64, #hivm.address_space<gm>>) {
    // expected-error@+1 {{'hivm.hir.mix_group_matmul' op matrix B must be 2D}}
    hivm.hir.mix_group_matmul
       ins(%weight_gm, %tokens_gm, %tokens_per_expert_gm:
           memref<16x16x16xf16, #hivm.address_space<gm>>, memref<16xf16, #hivm.address_space<gm>>,
           memref<16xi64, #hivm.address_space<gm>>)
       post_vector_func_ins(%post_vector_func_ins0_gm,
                            %post_vector_func_ins1_gm,
                            %post_vector_func_ins2_gm :
                            memref<16xi64, #hivm.address_space<gm>>,
                            memref<16xi64, #hivm.address_space<gm>>,
                            memref<16xi64, #hivm.address_space<gm>>)
       post_vector_func_outs(%post_vector_func_outs_gm : memref<16xi64, #hivm.address_space<gm>>)
       workspace_ins(%workspace_ins_gm : memref<1024xi64, #hivm.address_space<gm>>)
       outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       tiling_params = %tiling_params_gm : memref<16xi64, #hivm.address_space<gm>>
       comm_params = %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>
    return
}

// -----
// CHECK-LABEL: test_mix_group_matmul_invalid_token
func.func @test_mix_group_matmul_invalid_token(%weight_gm : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %tokens_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                                 %tokens_per_expert_gm :  memref<16x16xi64, #hivm.address_space<gm>>,
                                 %tiling_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                 %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                 %res_gm :memref<16x16xf16, #hivm.address_space<gm>>,
                                 %post_vector_func_ins0_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_ins1_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_ins2_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_outs_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %workspace_ins_gm: memref<1024xi64, #hivm.address_space<gm>>) {
    // expected-error@+1 {{'hivm.hir.mix_group_matmul' op matrix TokensPerExpert must be 1D}}
    hivm.hir.mix_group_matmul
       ins(%weight_gm, %tokens_gm, %tokens_per_expert_gm:
           memref<16x16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>,
           memref<16x16xi64, #hivm.address_space<gm>>)
       post_vector_func_ins(%post_vector_func_ins0_gm,
                            %post_vector_func_ins1_gm,
                            %post_vector_func_ins2_gm :
                            memref<16xi64, #hivm.address_space<gm>>,
                            memref<16xi64, #hivm.address_space<gm>>,
                            memref<16xi64, #hivm.address_space<gm>>)
       post_vector_func_outs(%post_vector_func_outs_gm : memref<16xi64, #hivm.address_space<gm>>)
       workspace_ins(%workspace_ins_gm : memref<1024xi64, #hivm.address_space<gm>>)
       outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
            tiling_params = %tiling_params_gm : memref<16xi64, #hivm.address_space<gm>>
            comm_params = %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>
    return
}

// -----
// CHECK-LABEL: mix_matmul_invalid_input
func.func @mix_matmul_invalid_input(
                              %arg1: memref<?xf16, #hivm.address_space<gm>>,
                              %arg2: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg3: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg4: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg5: memref<?x?xf16, #hivm.address_space<gm>>) {
  // expected-error@+1 {{'hivm.hir.mix_matmul' op matrix A and matrix B must be 2D}}
  hivm.hir.mix_matmul
    ins(%arg1, %arg2 : memref<?xf16, #hivm.address_space<gm>>, memref<?x?xf16, #hivm.address_space<gm>>)
        post_vector_func_ins(%arg3 : memref<?x?xf16, #hivm.address_space<gm>>)
        workspace_ins(%arg5 : memref<?x?xf16, #hivm.address_space<gm>>)
    outs(%arg4 : memref<?x?xf16, #hivm.address_space<gm>>)
  return
}

// -----
// CHECK-LABEL: test_matmul_invalid_input
func.func @test_matmul_invalid_input(
                              %arg0: memref<4x16x16xf16>,
                              %arg1: memref<16x16xf16>,
                              %arg2: memref<16x16xf32>) {
  // expected-error@+1 {{'hivm.hir.matmul' op matrix A and matrix B must be 2D}}
  hivm.hir.matmul ins(%arg0, %arg1 : memref<4x16x16xf16>, memref<16x16xf16>) outs(%arg2 : memref<16x16xf32>)
  return
}

// -----
// CHECK-LABEL test_matmul_valid
func.func @test_matmul_valid(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                       %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                       %descale_pertensor_gm :  memref<1xf16, #hivm.address_space<gm>>,
                       %descale_perchannel_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                       %res_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
  // expected-error@+1 {{'hivm.hir.matmul' op The descaleMode is defined, descale params must be defined!}}
  hivm.hir.matmul
    ins(%A_gm, %B_gm:
        memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
    outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
         descale_mode = #hivm.descale_mode<DescalePerTensor>

  hivm.hir.matmul
    ins(%A_gm, %B_gm:
        memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
    outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
         descale = %descale_perchannel_gm :  memref<16x16xf16, #hivm.address_space<gm>>

  return
}

// -----
// CHECK-LABEL test_matmul_valid_descale
func.func @test_matmul_valid_descale(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                       %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                       %descale_perchannel_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                       %res_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
  // expected-error@+1 {{'hivm.hir.matmul' op descale must must be 1D}}
  hivm.hir.matmul
  ins(%A_gm, %B_gm:
      memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
  outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       descale = %descale_perchannel_gm :  memref<16x16xf16, #hivm.address_space<gm>>
       descale_mode = #hivm.descale_mode<DescalePerChannel>

  return
}

// -----
// CHECK-LABEL test_matmul_valid_bias
func.func @test_matmul_valid_bias(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                       %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                       %bias_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                       %res_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
  // expected-error@+1 {{'hivm.hir.matmul' op bias must must be 1D}}
  hivm.hir.matmul
  ins(%A_gm, %B_gm:
      memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
  outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       bias = %bias_gm : memref<16x16xf16, #hivm.address_space<gm>>

  return
}

// -----
// CHECK-LABEL test_matmul_valid_descale_dim
func.func @test_matmul_valid_descale_dim(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                       %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                       %descale_perchannel_gm :  memref<3xf16, #hivm.address_space<gm>>,
                       %res_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
  // expected-error@+1 {{'hivm.hir.matmul' op The descaleMode is DescalePerChannel, the size of descale is equal to the col size of B}}
  hivm.hir.matmul
  ins(%A_gm, %B_gm:
      memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
  outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       descale = %descale_perchannel_gm :  memref<3xf16, #hivm.address_space<gm>>
       descale_mode = #hivm.descale_mode<DescalePerChannel>

  return
}