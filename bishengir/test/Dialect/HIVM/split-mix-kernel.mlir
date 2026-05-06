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

// -----

// CHECK-LABEL: func.func @test_scope_tensor_mix_aic(
// CHECK-SAME:    %[[EXT_TENSOR:.*]]: tensor<128x128xf32>, %[[EXT_VAL:.*]]: i32)
// CHECK-SAME:    hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-DAG:     %[[STUB_VEC_INT:.*]] = arith.constant 0 : i32
// CHECK:         %[[CUBE_RES:.*]]:3 = scope.scope : () -> (tensor<128x128xf32>, tensor<128x128xf32>, i32) {
// CHECK:           hivm.hir.mmadL1
// CHECK:           scope.return
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NOT:     scope.scope {{.*}} {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK-NOT:     hivm.hir.vadd
// CHECK-DAG:     %[[STUB_VEC_TENSOR:.*]] = tensor.empty() : tensor<128x128xf32>
// CHECK:         return %[[CUBE_RES]]#0, %[[CUBE_RES]]#1, %[[CUBE_RES]]#2, %[[STUB_VEC_TENSOR]], %[[EXT_TENSOR]], %[[STUB_VEC_INT]]

// CHECK-LABEL: func.func @test_scope_tensor_mix_aiv(
// CHECK-SAME:    %[[EXT_TENSOR:.*]]: tensor<128x128xf32>, %[[EXT_VAL:.*]]: i32)
// CHECK-SAME:    hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-NOT:     scope.scope {{.*}} {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
// CHECK-NOT:     hivm.hir.mmadL1
// CHECK-DAG:     %[[STUB_CUBE_TENSOR:.*]] = tensor.empty() : tensor<128x128xf32>
// CHECK-DAG:     %[[STUB_CUBE_INT:.*]] = arith.constant 0 : i32
// CHECK:         %[[VEC_RES:.*]]:3 = scope.scope : () -> (tensor<128x128xf32>, tensor<128x128xf32>, i32) {
// CHECK:           hivm.hir.vadd
// CHECK:           scope.return
// CHECK:         } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// CHECK:         return %[[STUB_CUBE_TENSOR]], %[[EXT_TENSOR]], %[[STUB_CUBE_INT]], %[[VEC_RES]]#0, %[[VEC_RES]]#1, %[[VEC_RES]]#2
func.func @test_scope_tensor(%ext_tensor: tensor<128x128xf32>, %ext_val: i32) -> (tensor<128x128xf32>, tensor<128x128xf32>, i32, tensor<128x128xf32>, tensor<128x128xf32>, i32) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %cube_res:3 = scope.scope : () -> (tensor<128x128xf32>, tensor<128x128xf32>, i32) {
    %a = tensor.empty() : tensor<128x128xbf16>
    %b = tensor.empty() : tensor<128x128xbf16>
    %c = tensor.empty() : tensor<128x128xf32>
    %mmad = hivm.hir.mmadL1 {b_transpose} ins(%a, %b, %true, %c128, %c128, %c128 : tensor<128x128xbf16>, tensor<128x128xbf16>, i1, index, index, index) outs(%c : tensor<128x128xf32>) -> tensor<128x128xf32>
    %local_int_c = arith.constant 42 : i32
    scope.return %mmad, %ext_tensor, %local_int_c : tensor<128x128xf32>, tensor<128x128xf32>, i32
  } {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
  %vec_res:3 = scope.scope : () -> (tensor<128x128xf32>, tensor<128x128xf32>, i32) {
    %v1 = tensor.empty() : tensor<128x128xf32>
    %v2 = tensor.empty() : tensor<128x128xf32>
    %vadd = hivm.hir.vadd ins(%v1, %v2 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%v1 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %local_int_v = arith.constant 100 : i32
    scope.return %vadd, %ext_tensor, %local_int_v : tensor<128x128xf32>, tensor<128x128xf32>, i32
  } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
  return %cube_res#0, %cube_res#1, %cube_res#2, %vec_res#0, %vec_res#1, %vec_res#2 
    : tensor<128x128xf32>, tensor<128x128xf32>, i32, tensor<128x128xf32>, tensor<128x128xf32>, i32
}

// -----

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-LABEL: func.func @test_true_marks_mix_aic({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix}
  // CHECK-LABEL: func.func @test_true_marks_mix_aiv({{.*}} attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix}
  // CHECK:         %[[VADD_PRE:.*]] = hivm.hir.vadd
  // CHECK-NOT:     annotation.mark
  // CHECK:         scf.for {{.*}} iter_args(%[[ARG_V:.*]] = %[[VADD_PRE]]
  // CHECK:           %[[VADD_INNER:.*]] = hivm.hir.vadd
  // CHECK-NOT:       annotation.mark
  // CHECK:           %[[STORE:.*]] = hivm.hir.store ins(%[[VADD_INNER]] : tensor<128x128xf32>)
  // CHECK-NEXT:      annotation.mark %[[STORE]] : tensor<128x128xf32>
  // CHECK-NOT:       annotation.mark
  func.func @test_true_marks(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) 
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %cst = arith.constant 1.0 : f32
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<128x128xf32>
    %vadd_pre = hivm.hir.vadd ins(%arg0, %cst : tensor<128x128xf32>, f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %loop_res:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc_v = %vadd_pre, %acc_c = %empty) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
      %vadd = hivm.hir.vadd ins(%acc_v, %cst : tensor<128x128xf32>, f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
      %store = hivm.hir.store ins(%vadd : tensor<128x128xf32>) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
      %load = hivm.hir.load ins(%store : tensor<128x128xf32>) outs(%empty : tensor<128x128xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf32>
      %mmad_inner = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%load, %arg1, %true, %c128, %c128, %c128 : tensor<128x128xf32>, tensor<128x128xf32>, i1, index, index, index) outs(%acc_c : tensor<128x128xf32>) -> tensor<128x128xf32>
      %vadd_inner = hivm.hir.vadd ins(%acc_v, %cst : tensor<128x128xf32>, f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
      scf.yield %vadd_inner, %mmad_inner : tensor<128x128xf32>, tensor<128x128xf32>
    }
    %mmad1 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%loop_res#1, %arg1, %true, %c128, %c128, %c128 : tensor<128x128xf32>, tensor<128x128xf32>, i1, index, index, index) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mmad2 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%loop_res#1, %arg1, %true, %c128, %c128, %c128 : tensor<128x128xf32>, tensor<128x128xf32>, i1, index, index, index) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mmad3 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%loop_res#1, %arg1, %true, %c128, %c128, %c128 : tensor<128x128xf32>, tensor<128x128xf32>, i1, index, index, index) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mmad4 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%loop_res#1, %arg1, %true, %c128, %c128, %c128 : tensor<128x128xf32>, tensor<128x128xf32>, i1, index, index, index) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %fixpipe = hivm.hir.fixpipe {enable_nz2nd} ins(%mmad4 : tensor<128x128xf32>) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    return
  }
}

// -----

module attributes {hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-LABEL: func.func @triton_dot_mix_aic
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
  // CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<2x1001x1xi32>) {
  // CHECK:   %[[MMAD:.*]] = hivm.hir.mmadL1
  // CHECK:   %[[FIXPIPE:.*]] = hivm.hir.fixpipe {{.*}} ins(%[[MMAD]] : tensor<1001x1xi32>)
  // CHECK-NEXT: annotation.mark %[[FIXPIPE]] : tensor<1001x1xi32>
  // CHECK:   %[[INSERT:.*]] = tensor.insert_slice %[[FIXPIPE]]
  // CHECK:   scf.yield %[[INSERT]] : tensor<2x1001x1xi32>
  // CHECK: }
  // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
  // CHECK-NOT: hivm.hir.vadd
  // CHECK-LABEL: func.func @triton_dot_mix_aiv
  // CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
  // CHECK: %[[FOR_RES:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %{{.*}}) -> (tensor<2x1001x1xi32>) {
  // CHECK-NEXT: scf.yield %[[ARG]] : tensor<2x1001x1xi32>
  // CHECK-NEXT: }
  // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
  // CHECK: %[[LOAD:.*]] = hivm.hir.load ins(%[[FOR_RES]] : tensor<2x1001x1xi32>)
  // CHECK: %[[VADD:.*]] = hivm.hir.vadd ins(%[[LOAD]], %{{.*}} : tensor<2x1001x1xi32>, tensor<2x1001x1xi32>)
  // CHECK: hivm.hir.store ins(%[[VADD]] : tensor<2x1001x1xi32>)
  func.func @triton_dot(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c1001 = arith.constant 1001 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [2, 1001, 1], strides: [1001, 1, 1] : memref<?xi8> to memref<2x1001x1xi8, strided<[1001, 1, 1]>>
    %alloc = memref.alloc() : memref<2x1001x1xi8>
    hivm.hir.load ins(%reinterpret_cast : memref<2x1001x1xi8, strided<[1001, 1, 1]>>) outs(%alloc : memref<2x1001x1xi8>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2x1001x1xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [2, 1, 1], strides: [1, 1, 1] : memref<?xi8> to memref<2x1x1xi8, strided<[1, 1, 1]>>
    %alloc_1 = memref.alloc() : memref<2x1x1xi8>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<2x1x1xi8, strided<[1, 1, 1]>>) outs(%alloc_1 : memref<2x1x1xi8>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<2x1x1xi8>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [2, 1001, 1], strides: [1001, 1, 1] : memref<?xi32> to memref<2x1001x1xi32, strided<[1001, 1, 1]>>
    %alloc_3 = memref.alloc() : memref<2x1001x1xi32>
    hivm.hir.load ins(%reinterpret_cast_2 : memref<2x1001x1xi32, strided<[1001, 1, 1]>>) outs(%alloc_3 : memref<2x1001x1xi32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %4 = bufferization.to_tensor %alloc_3 restrict writable : memref<2x1001x1xi32>
    %5 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<2x1001x1xi32>
    %6 = bufferization.to_tensor %5 restrict writable : memref<2x1001x1xi32>
    %7 = scf.for %arg10 = %c0 to %c2 step %c1 iter_args(%arg11 = %6) -> (tensor<2x1001x1xi32>) {
      %extracted_slice = tensor.extract_slice %2[%arg10, 0, 0] [1, 1001, 1] [1, 1, 1] : tensor<2x1001x1xi8> to tensor<1001x1xi8>
      %extracted_slice_5 = tensor.extract_slice %3[%arg10, 0, 0] [1, 1, 1] [1, 1, 1] : tensor<2x1x1xi8> to tensor<1x1xi8>
      %11 = tensor.empty() : tensor<1001x1xi32>
      %12 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%extracted_slice, %extracted_slice_5, %true, %c1001, %c1, %c1 : tensor<1001x1xi8>, tensor<1x1xi8>, i1, index, index, index) outs(%11 : tensor<1001x1xi32>) -> tensor<1001x1xi32>
      %extracted_slice_6 = tensor.extract_slice %arg11[%arg10, 0, 0] [1, 1001, 1] [1, 1, 1] : tensor<2x1001x1xi32> to tensor<1001x1xi32>
      %13 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%12 : tensor<1001x1xi32>) outs(%extracted_slice_6 : tensor<1001x1xi32>) -> tensor<1001x1xi32>
      %inserted_slice = tensor.insert_slice %13 into %arg11[%arg10, 0, 0] [1, 1001, 1] [1, 1, 1] {elide_after_bufferize} : tensor<1001x1xi32> into tensor<2x1001x1xi32>
      scf.yield %inserted_slice : tensor<2x1001x1xi32>
    }
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %8 = tensor.empty() : tensor<2x1001x1xi32>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %9 = hivm.hir.load ins(%7 : tensor<2x1001x1xi32>) outs(%8 : tensor<2x1001x1xi32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<2x1001x1xi32>
    %10 = hivm.hir.vadd ins(%9, %4 : tensor<2x1001x1xi32>, tensor<2x1001x1xi32>) outs(%8 : tensor<2x1001x1xi32>) -> tensor<2x1001x1xi32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2, 1001, 1], strides: [1001, 1, 1] : memref<?xi32> to memref<2x1001x1xi32, strided<[1001, 1, 1]>>
    hivm.hir.store ins(%10 : tensor<2x1001x1xi32>) outs(%reinterpret_cast_4 : memref<2x1001x1xi32, strided<[1001, 1, 1]>>)
    return
  }
}
