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