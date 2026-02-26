// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: bishengir-opt -allow-unregistered-dialect -mlir-print-op-generic %s -split-input-file | bishengir-opt -allow-unregistered-dialect | FileCheck %s

// -----
// CHECK-LABEL: get_block_idx
func.func @test_get_block_idx() {
  %0 = hivm.hir.get_block_idx -> i64
  return
}

// -----
// CHECK-LABEL: test_mmadL1
func.func @test_mmadL1() {
  %ma = memref.alloc() : memref<256x128xf16>
  %mb = memref.alloc() : memref<128x256xf16>
  %mc = memref.alloc() : memref<256x256xf32>
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %init = arith.constant 1 : i1
  hivm.hir.mmadL1 ins(%ma, %mb, %init, %c256, %c128, %c256 :
                        memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                  outs(%mc : memref<256x256xf32>)

  %ma_t = memref.alloc() : memref<128x256xf16>
  hivm.hir.mmadL1 {a_transpose}
                 ins(%ma_t, %mb, %init, %c256, %c128, %c256 :
                       memref<128x256xf16>, memref<128x256xf16>, i1, index, index, index)
                 outs(%mc : memref<256x256xf32>)

  %mb_t = memref.alloc() : memref<256x128xf16>
  hivm.hir.mmadL1 {b_transpose}
                 ins(%ma, %mb_t, %init, %c256, %c128, %c256 :
                       memref<256x128xf16>, memref<256x128xf16>, i1, index, index, index)
                 outs(%mc : memref<256x256xf32>)

  hivm.hir.mmadL1 {a_transpose, b_transpose}
                 ins(%ma_t, %mb_t, %init, %c256, %c128, %c256 :
                       memref<128x256xf16>, memref<256x128xf16>, i1, index, index, index)
                 outs(%mc : memref<256x256xf32>)
  return
}

// -----
// CHECK-LABEL: test_mmadL1_with_k_init
func.func @test_mmadL1_with_k_init() {
  %mc = memref.alloc() : memref<256x256xf32>
  %start = arith.constant 0 : index
  %end = arith.constant 1024 : index
  %step = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  scf.for %arg0 = %start to %end step %step {
    // Data loaded to L1
    %ma = memref.alloc() : memref<256x128xf16>
    %mb = memref.alloc() : memref<128x256xf16>
    // L0C data is cleared for the first iteration
    %init_condition = arith.cmpi eq, %arg0, %start : index
    hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                          memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                    outs(%mc : memref<256x256xf32>)
  }

  return
}

// -----
// CHECK-LABEL: test_mmadL1_with_k_init_tensor
func.func @test_mmadL1_with_k_init_tensor() {
  %mc = tensor.empty() : tensor<256x256xf32>
  %start = arith.constant 0 : index
  %end = arith.constant 1024 : index
  %step = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %scf_ret = scf.for %arg0 = %start to %end step %step iter_args(%mC_iter = %mc) -> (tensor<256x256xf32>) {
    // Data loaded to L1
    %ma = tensor.empty() : tensor<256x128xf16>
    %mb = tensor.empty() : tensor<128x256xf16>
    // L0C data is cleared for the first iteration
    %init_condition = arith.cmpi eq, %arg0, %start : index
    %res = hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                                 tensor<256x128xf16>, tensor<128x256xf16>, i1, index, index, index)
                           outs(%mC_iter : tensor<256x256xf32>) -> tensor<256x256xf32>
    scf.yield %res : tensor<256x256xf32>
  }

  return
}

// -----
// CHECK-LABEL: test_batchMmadL1
func.func @test_batchMmadL1() {
  %ma = memref.alloc() : memref<2x256x128xf16>
  %mb = memref.alloc() : memref<2x128x256xf16>
  %mc = memref.alloc() : memref<2x256x256xf32>
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %init = arith.constant 1 : i1
  hivm.hir.batchMmadL1 ins(%ma, %mb, %init, %c256, %c128, %c256 :
                        memref<2x256x128xf16>, memref<2x128x256xf16>, i1, index, index, index)
                  outs(%mc : memref<2x256x256xf32>)
  return
}

// -----
// CHECK-LABEL: test_convert_layout
// CHECK: hivm.hir.convert_layout
#dot_a_layout = #hivm.data_layout<dotA_ND, transpose = false>
#nZ_layout = #hivm.data_layout<nZ>
func.func @test_convert_layout(%arg : memref<128x128xf16, strided<[?, ?], offset: ?>>) {
  %alloc = memref.alloc() : memref<128x128xf16>
  memref.copy %arg, %alloc : memref<128x128xf16, strided<[?, ?], offset: ?>> to memref<128x128xf16>
  %alloc_new_layout = hivm.hir.convert_layout %alloc output_shape [8, 8, 16, 16] {srcLayout = #dot_a_layout, dstLayout = #nZ_layout} : (memref<128x128xf16>) -> memref<8x8x16x16xf16>
  "some_use"(%alloc_new_layout) : (memref<8x8x16x16xf16>) -> ()
  return
}

// -----
// CHECK-LABEL: test_matmul_basic
func.func @test_matmul_basic(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                             %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                             %res_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
  hivm.hir.matmul
     ins(%A_gm, %B_gm:
         memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
     outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
     descale_mode = #hivm.descale_mode<DescaleNull>
  return
}

// -----
// CHECK-LABEL: test_matmul_descale_perchannel
func.func @test_matmul_descale_perchannel(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                               %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                               %bias_gm : memref<16xf16, #hivm.address_space<gm>>,
                               %descale_perchannel_gm :  memref<16xf16, #hivm.address_space<gm>>,
                               %res_gm :memref<16x16xf16, #hivm.address_space<gm>> ) {
  hivm.hir.matmul
       ins(%A_gm, %B_gm:
           memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
       outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
       descale = %descale_perchannel_gm :  memref<16xf16, #hivm.address_space<gm>>
       descale_mode = #hivm.descale_mode<DescalePerChannel>
  return
}



// -----
// CHECK-LABEL: test_matmul_descale_pertensor
func.func @test_matmul_descale_pertensor(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                               %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                               %bias_gm : memref<16xf16, #hivm.address_space<gm>>,
                               %descale_pertensor_gm :  memref<1xf16, #hivm.address_space<gm>>,
                               %res_gm :memref<16x16xf16, #hivm.address_space<gm>> ) {
    hivm.hir.matmul
         ins(%A_gm, %B_gm:
             memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
         outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
         bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
         descale = %descale_pertensor_gm :  memref<1xf16, #hivm.address_space<gm>>
         descale_mode = #hivm.descale_mode<DescalePerTensor>
  return
}

// -----
// CHECK-LABEL: test_mix_matmul_workspace_add_0_300_mix_aiv
module  {
  func.func @test_mix_matmul_workspace_add_0_300_mix_aiv(%arg0: i64 {hfusion.ffts_base_address},
                                                     %arg1: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                                     %arg2: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                                     %arg3: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                                     %arg4: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                                     %arg5: memref<1024x1024xf16, #hivm.address_space<gm>>)
                                                     attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                                                                 hacc.tiling_func = "matmul_add_0_tiling_func",
                                                                 hacc.block_dim = 20 : i64,
                                                                 hfusion.fusion_kind = #hfusion.fusion_kind<MIX_CV>,
                                                                 hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c256_i64 = arith.constant 256 : i64
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.mix_matmul {post_vector_func = "bishengir_gen_vector_epilogue_func"}
      ins(%arg1, %arg2 :
          memref<1024x1024xf16, #hivm.address_space<gm>>, memref<1024x1024xf16, #hivm.address_space<gm>>)
      post_vector_func_ins(%arg3 : memref<1024x1024xf16, #hivm.address_space<gm>>)
      workspace_ins(%arg4 : memref<1024x1024xf16, #hivm.address_space<gm>>)
      outs(%arg5 : memref<1024x1024xf16, #hivm.address_space<gm>>)
      block_sizes(%c128_i64, %c256_i64, %c256_i64 : i64, i64, i64)
      process_sizes(%c128_i64, %c256_i64, %c64_i64 : i64, i64, i64)
      swizzle_offset = %c1_i64 : i64
      swizzle_direction = %c0_i64 : i64
      epilogue_p_tiles = %c4_i64 : i64
    return
  }
}

// -----
module {
  func.func @test_mix_matmul(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                         %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                         %bias_gm : memref<16xf16, #hivm.address_space<gm>>,
                         %tiling_params_gm : memref<16xf16, #hivm.address_space<gm>>,
                         %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                         %res_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
      hivm.hir.mix_matmul
         ins(%A_gm, %B_gm:
             memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
         outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
         tiling_params = %tiling_params_gm : memref<16xf16, #hivm.address_space<gm>>
         comm_params = %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>
      return
  }
}


// -----
module {
  func.func @test_mix_group_matmul(%A_gm : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                   %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                                   %tokens_per_expert_gm :  memref<16xi64, #hivm.address_space<gm>>,
                                   %bias_gm : memref<16xf16, #hivm.address_space<gm>>,
                                   %tiling_params_gm : memref<16xf16, #hivm.address_space<gm>>,
                                   %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                   %res_gm :memref<16x16xf16, #hivm.address_space<gm>>,
                                   %post_vector_func_ins: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                   %post_vector_func_outs: memref<1024x1024xf16, #hivm.address_space<gm>>) {
      hivm.hir.mix_group_matmul
         ins(%A_gm, %B_gm, %tokens_per_expert_gm:
             memref<16x16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>,
             memref<16xi64, #hivm.address_space<gm>>)
         post_vector_func_ins(%post_vector_func_ins : memref<1024x1024xf16, #hivm.address_space<gm>>)
         post_vector_func_outs(%post_vector_func_outs : memref<1024x1024xf16, #hivm.address_space<gm>>)
         outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
         tiling_params = %tiling_params_gm : memref<16xf16, #hivm.address_space<gm>>
         comm_params = %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>
      return
  }
}

// -----
// CHECK-LABEL: @test_set_ffts_base_addr
func.func @test_set_ffts_base_addr() {
  %ffts_base_addr = arith.constant 0 : i64
  hivm.hir.set_ffts_base_addr %ffts_base_addr
  return
}

// -----
//  CHECK-LABEL: @test_sync_block
func.func @test_sync_block() {
  %ffts_base_addr = arith.constant 0 : i64
  hivm.hir.sync_block[#hivm.sync_block_mode<ALL>, 1 : i16]
            ffts_base_addr = %ffts_base_addr
            tcube_pipe=#hivm.pipe<PIPE_FIX>
            tvector_pipe=#hivm.pipe<PIPE_MTE3>
  hivm.hir.sync_block[#hivm.sync_block_mode<ALL_CUBE>, 1 : i16]
            ffts_base_addr = %ffts_base_addr
            tcube_pipe=#hivm.pipe<PIPE_FIX>
  hivm.hir.sync_block[#hivm.sync_block_mode<ALL_VECTOR>, 1 : i16]
            ffts_base_addr = %ffts_base_addr
            tvector_pipe=#hivm.pipe<PIPE_MTE3>
  return
}

// -----
module {
  func.func @test_ptr() {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<16x16x16xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) : memref<16x16x16xf16, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c0_i64, %c16_i64, %c32_i64) : memref<16x16x16xf16, #hivm.address_space<ub>>
    return
  }
}

// -----
module {
  // CHECK-LABEL func.func @print_test
  // CHECK: (%[[ARG1:.*]]: tensor<32xf32>, %[[ARG2:.*]]: memref<32xi8>)
  // CHECK: hivm.hir.init_debug
  // CHECK: hivm.hir.debug {debugtype = "print", hex = true, prefix = "Val: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %[[ARG1]] : tensor<32xf32>
  // CHECK: hivm.hir.debug {debugtype = "print", hex = false, prefix = "", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %[[ARG2]] : memref<32xi8>
  // CHECK: hivm.hir.finish_debug
  func.func @print_test(%arg1 : tensor<32xf32>, %arg2 : memref<32xi8>) {
    hivm.hir.init_debug
    hivm.hir.debug {debugtype = "print", hex = true, prefix = "Val: ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg1 : tensor<32xf32>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = "", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg2 : memref<32xi8>
    hivm.hir.finish_debug
    func.return
  }
}

// -----
// CHECK-LABEL: test_convert_layout_tensor
// CHECK: hivm.hir.convert_layout
#dot_a_layout = #hivm.data_layout<dotA_ND, transpose = false>
#nZ_layout = #hivm.data_layout<nZ>
func.func @test_convert_layout_tensor(%arg : tensor<128x128xf16>) -> tensor<8x8x16x16xf16> {
  %alloc_new_layout = hivm.hir.convert_layout %arg output_shape [8, 8, 16, 16] {srcLayout = #dot_a_layout, dstLayout = #nZ_layout}
                      : (tensor<128x128xf16>) -> tensor<8x8x16x16xf16>
  return %alloc_new_layout : tensor<8x8x16x16xf16>
}
