// RUN: bishengir-opt %s -hivm-split-mix-kernel -split-input-file -verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: add(
  func.func private @add(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>, 
                         %arg2: tensor<64x64xf16> {hacc.arg_type = #hacc.arg_type<output>}) -> tensor<64x64xf16> 
  attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}

  // CHECK-LABEL: mul_add_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: mul_add_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  // CHECK: annotation.mark
  func.func @mul_add(%arg0: tensor<64x64xf16>,
                      %arg1: tensor<64x64xf16>,
                      %arg2: tensor<64x64xf16>,
                      %arg3: tensor<64x64xf16>,
                      %arg4: tensor<64x64xf16>) -> tensor<64x64xf16>
                      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %0 = func.call @add(%arg0, %arg1, %arg2) : (tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
    %1 = hivm.hir.matmul ins(%0, %arg3 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg4: tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }
}

// -----

module {
  // CHECK-LABEL: mul_add_with_collapse_shape_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK: %[[T0:.*]] = hivm.hir.matmul
  // CHECK: annotation.mark %[[T0:.*]]
  // CHECK-LABEL: mul_add_with_collapse_shape_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  func.func @mul_add_with_collapse_shape(%arg0: tensor<64x64xf16>,
                      %arg1: tensor<64x64xf16>,
                      %arg2: tensor<64x64xf16>,
                      %arg3: tensor<4096xf16>,
                      %arg4: tensor<4096xf16>) -> tensor<4096xf16>
                      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %1 = hivm.hir.matmul ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg2: tensor<64x64xf16>) -> tensor<64x64xf16>
    %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<64x64xf16> into tensor<4096xf16>
    %2 = hivm.hir.vadd ins(%collapsed, %arg3 : tensor<4096xf16>, tensor<4096xf16>) outs(%arg4 : tensor<4096xf16>) -> tensor<4096xf16>
    return %2 : tensor<4096xf16>
  }
}

// -----

module {
  // CHECK-LABEL: mixed_matmul_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: mixed_matmul_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  func.func @mixed_matmul(%arg0: tensor<64x64xf16>,
                      %arg1: tensor<64x64xf16>,
                      %arg2: tensor<64x64xf16>,
                      %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
                      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %1 = hivm.hir.mix_matmul ins(%arg0, %arg2 : tensor<64x64xf16>, tensor<64x64xf16>)
                         post_vector_func_ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
                         outs(%arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }
}

// -----

module {
  // CHECK-LABEL: func.func private @mixed_matmul
  // CHECK-SAME: attributes
  // CHECK-SAME: hacc.function_kind = #hacc.function_kind<DEVICE>
  // CHECK-SAME: hacc.mix_entry
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>

  // CHECK-LABEL: mixed_matmul_mix_aic({{.*}} hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: mixed_matmul_mix_aiv({{.*}} hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  func.func @mixed_matmul(%arg0: tensor<64x64xf16>,
                          %arg1: tensor<64x64xf16>,
                          %arg2: tensor<64x64xf16>,
                          %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %1 = hivm.hir.mix_matmul ins(%arg0, %arg2 : tensor<64x64xf16>, tensor<64x64xf16>)
                         post_vector_func_ins(%arg0, %arg1 : tensor<64x64xf16>, tensor<64x64xf16>)
                         outs(%arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }

  func.func @host_caller(%arg0: tensor<64x64xf16>,
                         %arg1: tensor<64x64xf16>,
                         %arg2: tensor<64x64xf16>,
                         %arg3: tensor<64x64xf16>) -> tensor<64x64xf16>
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    %1 = func.call @mixed_matmul(%arg0, %arg1, %arg2, %arg3) : (tensor<64x64xf16>,tensor<64x64xf16>,tensor<64x64xf16>,tensor<64x64xf16>) -> tensor<64x64xf16>
    return %1 : tensor<64x64xf16>
  }
}

// -----

module {
  // expected-error@below {{Currently, MIX kernels can only be called by host functions!}}
  func.func private @mix_function() -> tensor<16xf16>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>}

  func.func @device_caller() -> tensor<16xf16>
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %1 = func.call @mix_function() : () -> tensor<16xf16>
    return %1 : tensor<16xf16>
  }
}


// -----

// CHECK-LABEL: @test_callee_arg_with_inconsistent_order_mix_aic({{.*}}: i64, {{.*}}: tensor<128x256xf32>, 
// CHECK-SAME: {{.*}}: tensor<256xf32>, {{.*}}: tensor<768x256xf32>, %[[arg4:.*]]: tensor<128xf32>, %[[arg5:.*]]: tensor<128x1xf32>, 
// CHECK-SAME: {{.*}}: tensor<128x768xf32>, {{.*}}: tensor<128x256xf32>)
// CHECK: return %[[arg4]], %[[arg5]], {{.*}} : tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>
module {
  func.func @callee_arg_with_inconsistent_order(
    %arg0: tensor<128xf32> {hacc.arg_type = #hacc.arg_type<output>}, 
    %arg1: tensor<128x1xf32> {hacc.arg_type = #hacc.arg_type<output>}, 
    %arg2: tensor<128x256xf32>, 
    %arg3: tensor<256xf32>, 
    %arg4: tensor<128x256xf32> {hacc.arg_type = #hacc.arg_type<output>}) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>) attributes {hacc.always_inline, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_func = "", hacc.block_dim = 1 : i64, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
      return %arg0, %arg1, %arg4 : tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>
  }
  func.func @test_callee_arg_with_inconsistent_order(
    %arg0: i64, %arg1: tensor<128x256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<768x256xf32>, %arg4: tensor<128xf32>, 
    %arg5: tensor<128x1xf32>, %arg6: tensor<128x768xf32>, %arg7: tensor<128x256xf32>) -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
      %0:3 = call @callee_arg_with_inconsistent_order(%arg4, %arg5, %arg1, %arg2, %arg7) : 
        (tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>, tensor<256xf32>, tensor<128x256xf32>) 
        -> (tensor<128xf32>, tensor<128x1xf32>, tensor<128x256xf32>)
      %1 = hivm.hir.mix_matmul ins(%0#2, %arg3 : tensor<128x256xf32>, tensor<768x256xf32>) outs(%arg6 : tensor<128x768xf32>) b_transpose -> tensor<128x768xf32>
        return %0#0, %0#1, %1 : tensor<128xf32>, tensor<128x1xf32>, tensor<128x768xf32>
  }
}

// -----

// CHECK-LABEL: func.func @test_split_and_cleanup_mix_aic
// CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
// CHECK: %[[OUTER_RES:.*]]:3 = scf.for %{{.*}} iter_args(%[[ACC_CUBE:.*]] = %{{.*}}, %[[ACC_VEC:.*]] = %{{.*}}, %[[ACC_RED:.*]] = %{{.*}})
// CHECK:   hivm.hir.mmadL1 {{.*}} outs(%[[ACC_CUBE]] : tensor<64x256xf32>)
// CHECK:   %[[RES_INNER1:.*]] = scf.for %{{.*}} iter_args(%[[ITER_VEC:.*]] = %[[ACC_VEC]])
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: tensor.insert_slice
// CHECK:     scf.yield %[[ITER_VEC]] : tensor<64x256xf32>
// CHECK:   %[[RES_INNER2:.*]] = scf.for %{{.*}} iter_args(%[[ITER_RED:.*]] = %[[ACC_RED]])
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: tensor.extract_slice
// CHECK:     %[[NEW_EMPTY:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:     scf.yield %[[NEW_EMPTY]] : tensor<64xf32>
// CHECK:   scf.yield %{{.*}}, %[[RES_INNER1]], %[[RES_INNER2]]

// CHECK-LABEL: func.func @test_split_and_cleanup_mix_aiv
// CHECK-SAME: attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
// CHECK: %[[OUTER_RES_V:.*]]:3 = scf.for %{{.*}} iter_args(%[[ACC_CUBE_V:.*]] = %{{.*}}, %[[ACC_VEC_V:.*]] = %{{.*}}, %[[ACC_RED_V:.*]] = %{{.*}})
// CHECK-NOT: hivm.hir.mmadL1
// CHECK:   %[[RES_INNER1_V:.*]] = scf.for %{{.*}} iter_args(%[[ITER_VEC_V:.*]] = %[[ACC_VEC_V]])
// CHECK:     %[[EXT1:.*]] = tensor.extract_slice %[[ITER_VEC_V]]
// CHECK:     %[[VADD1:.*]] = hivm.hir.vadd ins(%[[EXT1]], %[[EXT1]]
// CHECK:     %[[INS1:.*]] = tensor.insert_slice %[[VADD1]] into %[[ITER_VEC_V]]
// CHECK:     scf.yield %[[INS1]]
// CHECK:   %[[RES_INNER2_V:.*]] = scf.for %{{.*}} iter_args(%[[ITER_RED_V:.*]] = %[[ACC_RED_V]])
// CHECK:     %[[VADD2:.*]] = hivm.hir.vadd
// CHECK:     %[[EXT2:.*]] = tensor.extract_slice %[[VADD2]]
// CHECK:     scf.yield %[[EXT2]]
// CHECK:   scf.yield %[[ACC_CUBE_V]], %[[RES_INNER1_V]], %[[RES_INNER2_V]]
func.func @test_split_and_cleanup(%A: tensor<64x128xf16>, 
                                  %B: tensor<256x128xf16>, 
                                  %C_init: tensor<64x256xf32>,
                                  %V_init: tensor<64x256xf32>,
                                  %V_reduce_init: tensor<64xf32>) -> (tensor<64x256xf32>, tensor<64x256xf32>, tensor<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %true = arith.constant true
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index

  %res_outer:3 = scf.for %outer = %c0 to %c2 step %c1 iter_args(%acc_cube = %C_init, %acc_vec = %V_init, %acc_reduce = %V_reduce_init) -> (tensor<64x256xf32>, tensor<64x256xf32>, tensor<64xf32>) {
    %mmad_res = hivm.hir.mmadL1 {b_transpose} ins(%A, %B, %true, %c64, %c128, %c256 : tensor<64x128xf16>, tensor<256x128xf16>, i1, index, index, index) outs(%acc_cube : tensor<64x256xf32>) -> tensor<64x256xf32>

    %res_inner1 = scf.for %inner1 = %c0 to %c2 step %c1 iter_args(%iter_vec = %acc_vec) -> (tensor<64x256xf32>) {
      %extracted1 = tensor.extract_slice %iter_vec[%inner1, 0] [32, 256] [1, 1] : tensor<64x256xf32> to tensor<32x256xf32>
      %empty_for_vadd = tensor.empty() : tensor<32x256xf32>
      %vadd_res = hivm.hir.vadd ins(%extracted1, %extracted1 : tensor<32x256xf32>, tensor<32x256xf32>) outs(%empty_for_vadd : tensor<32x256xf32>) -> tensor<32x256xf32>
      %inserted = tensor.insert_slice %vadd_res into %iter_vec[%inner1, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<64x256xf32>
      scf.yield %inserted : tensor<64x256xf32>
    }

    %res_inner2 = scf.for %inner2 = %c0 to %c2 step %c1 iter_args(%iter_red = %acc_reduce) -> (tensor<64xf32>) {
      %empty_for_reduce = tensor.empty() : tensor<1x64xf32>
      %dummy_extracted = tensor.extract_slice %res_inner1[%inner2, 0] [1, 64] [1, 1] : tensor<64x256xf32> to tensor<1x64xf32>
      %vreduce_res = hivm.hir.vadd ins(%dummy_extracted, %dummy_extracted : tensor<1x64xf32>, tensor<1x64xf32>) outs(%empty_for_reduce : tensor<1x64xf32>) -> tensor<1x64xf32>
      %extracted3 = tensor.extract_slice %vreduce_res[0, 0] [1, 64] [1, 1] : tensor<1x64xf32> to tensor<64xf32>
      scf.yield %extracted3 : tensor<64xf32>
    }

    scf.yield %mmad_res, %res_inner1, %res_inner2 : tensor<64x256xf32>, tensor<64x256xf32>, tensor<64xf32>
  }

  return %res_outer#0, %res_outer#1, %res_outer#2 : tensor<64x256xf32>, tensor<64x256xf32>, tensor<64xf32>
}
