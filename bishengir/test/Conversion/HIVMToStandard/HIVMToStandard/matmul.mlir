// RUN: bishengir-opt %s -convert-hivm-to-std -split-input-file | FileCheck %s

// -----

module {
  func.func @test_matmul(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                         %B_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                         %bias_gm : memref<16xf16, #hivm.address_space<gm>>,
                         %tiling_params_gm : memref<16xf16, #hivm.address_space<gm>>,
                         %descale_perchannel_gm :  memref<16xf16, #hivm.address_space<gm>>,
                         %descale_pertensor_gm :  memref<1xf16, #hivm.address_space<gm>>,
                         %res_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
    // CHECK: call @matmul_Xbias_Xdescale_XtransposeA_XtransposeB_TAhalf_TBhalf_TChalf({{.*}}, {{.*}}, {{.*}})
    hivm.hir.matmul
       ins(%A_gm, %B_gm:
           memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
       outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       descale_mode = #hivm.descale_mode<DescaleNull>

    // CHECK: call @matmul_bias_TBIAShalf_Xdescale_XtransposeA_XtransposeB_TAhalf_TBhalf_TChalf({{.*}}, {{.*}}, {{.*}}, {{.*}})
    hivm.hir.matmul
       ins(%A_gm, %B_gm:
           memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
       outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
       descale_mode = #hivm.descale_mode<DescaleNull>

    // CHECK: call @matmul_bias_TBIAShalf_descalePerChannel_TDESCALEhalf_XtransposeA_XtransposeB_TAhalf_TBhalf_TChalf({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
    hivm.hir.matmul
       ins(%A_gm, %B_gm:
           memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
       outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
       descale = %descale_perchannel_gm :  memref<16xf16, #hivm.address_space<gm>>
       descale_mode = #hivm.descale_mode<DescalePerChannel>

    // CHECK: call @matmul_bias_TBIAShalf_descalePerChannel_TDESCALEhalf_transposeA_transposeB_TAhalf_TBhalf_TChalf({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
    hivm.hir.matmul
       ins(%A_gm, %B_gm:
           memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
       outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
       bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
       descale = %descale_perchannel_gm :  memref<16xf16, #hivm.address_space<gm>>
       a_transpose b_transpose
       descale_mode = #hivm.descale_mode<DescalePerChannel>

      // CHECK: call @matmul_bias_TBIAShalf_descalePerTensor_TDESCALEhalf_transposeA_transposeB_TAhalf_TBhalf_TChalf({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
    hivm.hir.matmul
      ins(%A_gm, %B_gm:
          memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
      outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
      bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
      descale = %descale_pertensor_gm :  memref<1xf16, #hivm.address_space<gm>>
      a_transpose b_transpose
      descale_mode = #hivm.descale_mode<DescalePerTensor>
    return
  }
}

// -----

func.func @test_mix_group_matmul(%weight_gm : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %tokens_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                                 %tokens_per_expert_gm :  memref<16xi64, #hivm.address_space<gm>>,
                                 %bias_gm : memref<16xf16, #hivm.address_space<gm>>,
                                 %tiling_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                 %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>,
                                 %res_gm :memref<16x16xf16, #hivm.address_space<gm>>,
                                 %post_vector_func_ins0_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_ins1_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_ins2_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %post_vector_func_outs_gm: memref<16xi64, #hivm.address_space<gm>>,
                                 %workspace_ins_gm: memref<1024xi64, #hivm.address_space<gm>>)
                                 attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    // CHECK: mix_group_matmul_Xbias_Xdescale_XtransposeA_XtransposeB_TAhalf_TBhalf_TChalf_TMint64_t_TIint64_t_TOint64_t_TGint64_t_TTint64_t_TWint64_t_TTint64_t_TCint64_t
    hivm.hir.mix_group_matmul
       ins(%weight_gm, %tokens_gm, %tokens_per_expert_gm:
           memref<16x16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>,
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

module  {
  func.func @mix_matmul_add_0_300_mix_aiv(%arg0: i64 {hfusion.ffts_base_address},
                                      %arg1: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                      %arg2: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                      %arg3: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                      %arg4: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                      %arg5: memref<1024x1024xf16, #hivm.address_space<gm>>,
                                      %arg6 : memref<1024x1024xf16, #hivm.address_space<gm>>,
                                      %arg7 : memref<1024x1024xf16, #hivm.address_space<gm>>)
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
    // CHECK: call @mix_matmul_Xbias_Xdescale_XtransposeA_XtransposeB_TAhalf_TBhalf_TChalf_TVhalf_TWhalf_128_256_256_128_256_64_1_0_4_mix_aiv({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
    hivm.hir.mix_matmul
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

    // CHECK: call  @mix_matmul_Xbias_Xdescale_XtransposeA_XtransposeB_TAhalf_TBhalf_TChalf_TVhalf_TWhalf_TThalf_mix_aiv({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
    hivm.hir.mix_matmul
        ins(%arg1, %arg2 :
            memref<1024x1024xf16, #hivm.address_space<gm>>, memref<1024x1024xf16, #hivm.address_space<gm>>)
        post_vector_func_ins(%arg3 : memref<1024x1024xf16, #hivm.address_space<gm>>)
        workspace_ins(%arg4 : memref<1024x1024xf16, #hivm.address_space<gm>>)
        outs(%arg5 : memref<1024x1024xf16, #hivm.address_space<gm>>)
        tiling_params = %arg6 : memref<1024x1024xf16, #hivm.address_space<gm>>

    // CHECK: call  @mix_matmul_Xbias_Xdescale_XtransposeA_XtransposeB_TAhalf_TBhalf_TChalf_TVhalf_TWhalf_TThalf_TChalf_mix_aiv({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}})
    hivm.hir.mix_matmul
        ins(%arg1, %arg2 :
            memref<1024x1024xf16, #hivm.address_space<gm>>, memref<1024x1024xf16, #hivm.address_space<gm>>)
        post_vector_func_ins(%arg3 : memref<1024x1024xf16, #hivm.address_space<gm>>)
        workspace_ins(%arg4 : memref<1024x1024xf16, #hivm.address_space<gm>>)
        outs(%arg5 : memref<1024x1024xf16, #hivm.address_space<gm>>)
        tiling_params = %arg6 : memref<1024x1024xf16, #hivm.address_space<gm>>
        comm_params = %arg7 : memref<1024x1024xf16, #hivm.address_space<gm>>

    return
  }
}

// -----

// CHECK: func.func private @mix_matmul_
// CHECK-SAME: _mix_aic
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>

// CHECK: func.func private @mix_matmul_
// CHECK-SAME: _mix_aiv
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>

func.func @mix_matmul_add_mix_aic(%arg0: i64,
                              %arg1: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg2: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg3: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg4: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg5: memref<?x?xf16, #hivm.address_space<gm>>)
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
  %c4_i64 = arith.constant 4 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c256_i64 = arith.constant 256 : i64
  %c128_i64 = arith.constant 128 : i64
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.mix_matmul
    ins(%arg1, %arg2 : memref<?x?xf16, #hivm.address_space<gm>>, memref<?x?xf16, #hivm.address_space<gm>>)
    post_vector_func_ins(%arg3 : memref<?x?xf16, #hivm.address_space<gm>>)
    workspace_ins(%arg5 : memref<?x?xf16, #hivm.address_space<gm>>)
    outs(%arg4 : memref<?x?xf16, #hivm.address_space<gm>>)
    block_sizes(%c128_i64, %c256_i64, %c256_i64 : i64, i64, i64)
    process_sizes(%c128_i64, %c256_i64, %c64_i64 : i64, i64, i64)
    swizzle_offset = %c1_i64 : i64
    swizzle_direction = %c0_i64 : i64
    epilogue_p_tiles = %c4_i64 : i64
  return
}

func.func @mix_matmul_add_mix_aiv(%arg0: i64,
                              %arg1: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg2: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg3: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg4: memref<?x?xf16, #hivm.address_space<gm>>,
                              %arg5: memref<?x?xf16, #hivm.address_space<gm>>)
attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c4_i64 = arith.constant 4 : i64
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c256_i64 = arith.constant 256 : i64
  %c128_i64 = arith.constant 128 : i64
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.mix_matmul
    ins(%arg1, %arg2 : memref<?x?xf16, #hivm.address_space<gm>>, memref<?x?xf16, #hivm.address_space<gm>>)
    post_vector_func_ins(%arg3 : memref<?x?xf16, #hivm.address_space<gm>>)
    workspace_ins(%arg5 : memref<?x?xf16, #hivm.address_space<gm>>)
    outs(%arg4 : memref<?x?xf16, #hivm.address_space<gm>>)
    block_sizes(%c128_i64, %c256_i64, %c256_i64 : i64, i64, i64)
    process_sizes(%c128_i64, %c256_i64, %c64_i64 : i64, i64, i64)
    swizzle_offset = %c1_i64 : i64
    swizzle_direction = %c0_i64 : i64
    epilogue_p_tiles = %c4_i64 : i64
  return
}

// -----

// CHECK-LABEL: @triton_dot_hf32
func.func @triton_dot_hf32(%cast: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %cast_1: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %2: memref<1x39xf32, #hivm.address_space<cbuf>>, %cast_3: memref<?x?x?x?xf32, #hivm.address_space<cc>>) {
  %true = arith.constant true
  %c39 = arith.constant 39 : index
  %c35 = arith.constant 35 : index
  %c13 = arith.constant 13 : index
  hivm.hir.mmadL1 {enable_HF32} ins(%cast, %cast_1, %true, %c13, %c35, %c39, %2 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index, memref<1x39xf32, #hivm.address_space<cbuf>>) outs(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  // CHECK: call @mma_tile_with_float_bias_float_to_float_hf32
  return
}

// -----

// CHECK-LABEL: @triton_dot_ta
func.func @triton_dot_ta(%cast: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %cast_1: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %2: memref<1x39xf32, #hivm.address_space<cbuf>>, %cast_3: memref<?x?x?x?xf32, #hivm.address_space<cc>>) {
  %true = arith.constant true
  %c39 = arith.constant 39 : index
  %c35 = arith.constant 35 : index
  %c13 = arith.constant 13 : index
  hivm.hir.mmadL1 {a_transpose} ins(%cast, %cast_1, %true, %c13, %c35, %c39, %2 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index, memref<1x39xf32, #hivm.address_space<cbuf>>) outs(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  // CHECK: call @mma_tile_with_float_bias_float_to_float_ta
  return
}

// -----

// CHECK-LABEL: @triton_dot_tb
func.func @triton_dot_tb(%cast: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %cast_1: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %2: memref<1x39xf32, #hivm.address_space<cbuf>>, %cast_3: memref<?x?x?x?xf32, #hivm.address_space<cc>>) {
  %true = arith.constant true
  %c39 = arith.constant 39 : index
  %c35 = arith.constant 35 : index
  %c13 = arith.constant 13 : index
  hivm.hir.mmadL1 {b_transpose} ins(%cast, %cast_1, %true, %c13, %c35, %c39, %2 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index, memref<1x39xf32, #hivm.address_space<cbuf>>) outs(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  // CHECK: call @mma_tile_with_float_bias_float_to_float_tb
  return
}

// -----

// CHECK-LABEL: @triton_dot_ta_tb
func.func @triton_dot_ta_tb(%cast: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %cast_1: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %2: memref<1x39xf32, #hivm.address_space<cbuf>>, %cast_3: memref<?x?x?x?xf32, #hivm.address_space<cc>>) {
  %true = arith.constant true
  %c39 = arith.constant 39 : index
  %c35 = arith.constant 35 : index
  %c13 = arith.constant 13 : index
  hivm.hir.mmadL1 {a_transpose, b_transpose} ins(%cast, %cast_1, %true, %c13, %c35, %c39, %2 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index, memref<1x39xf32, #hivm.address_space<cbuf>>) outs(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  // CHECK: call @mma_tile_with_float_bias_float_to_float_ta_tb
  return
}

// -----

// CHECK-LABEL: @triton_dot_ta_hf32
func.func @triton_dot_ta_hf32(%cast: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %cast_1: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %2: memref<1x39xf32, #hivm.address_space<cbuf>>, %cast_3: memref<?x?x?x?xf32, #hivm.address_space<cc>>) {
  %true = arith.constant true
  %c39 = arith.constant 39 : index
  %c35 = arith.constant 35 : index
  %c13 = arith.constant 13 : index
  hivm.hir.mmadL1 {a_transpose, enable_HF32} ins(%cast, %cast_1, %true, %c13, %c35, %c39, %2 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index, memref<1x39xf32, #hivm.address_space<cbuf>>) outs(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  // CHECK: call @mma_tile_with_float_bias_float_to_float_ta_hf32
  return
}

// -----

// CHECK-LABEL: @triton_dot_tb_hf32
func.func @triton_dot_tb_hf32(%cast: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %cast_1: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %2: memref<1x39xf32, #hivm.address_space<cbuf>>, %cast_3: memref<?x?x?x?xf32, #hivm.address_space<cc>>) {
  %true = arith.constant true
  %c39 = arith.constant 39 : index
  %c35 = arith.constant 35 : index
  %c13 = arith.constant 13 : index
  hivm.hir.mmadL1 {b_transpose, enable_HF32} ins(%cast, %cast_1, %true, %c13, %c35, %c39, %2 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index, memref<1x39xf32, #hivm.address_space<cbuf>>) outs(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  // CHECK: call @mma_tile_with_float_bias_float_to_float_tb_hf32
  return
}

// -----

// CHECK-LABEL: @triton_dot_ta_tb_hf32
func.func @triton_dot_ta_tb_hf32(%cast: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %cast_1: memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, %2: memref<1x39xf32, #hivm.address_space<cbuf>>, %cast_3: memref<?x?x?x?xf32, #hivm.address_space<cc>>) {
  %true = arith.constant true
  %c39 = arith.constant 39 : index
  %c35 = arith.constant 35 : index
  %c13 = arith.constant 13 : index
  hivm.hir.mmadL1 {a_transpose, b_transpose, enable_HF32} ins(%cast, %cast_1, %true, %c13, %c35, %c39, %2 : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, memref<?x?x?x?xf32, #hivm.address_space<cbuf>>, i1, index, index, index, memref<1x39xf32, #hivm.address_space<cbuf>>) outs(%cast_3 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
  // CHECK: call @mma_tile_with_float_bias_float_to_float_ta_tb_hf32
  return
}
