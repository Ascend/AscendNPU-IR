// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @custom_inplace_index_out_of_range(%arg0: tensor<16xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op inplace_outs must be a valid outs operand index}}
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_bad_index"}
      "user.bad_index"
      ins(%arg0 : tensor<16xf32> {inplace_outs = 1 : i32})
      outs(%arg0 : tensor<16xf32>) -> tensor<16xf32>
  return
}

// -----

func.func @custom_inplace_duplicate_input(%arg0: tensor<16xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op inplace_outs cannot map multiple ins operands to the same outs operand}}
  %0:2 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_duplicate"}
      "user.duplicate"
      ins(%arg0, %arg0 : tensor<16xf32> {inplace_outs = 0 : i32},
                         tensor<16xf32> {inplace_outs = 0 : i32})
      outs(%arg0, %arg0 : tensor<16xf32>, tensor<16xf32>)
      -> (tensor<16xf32>, tensor<16xf32>)
  return
}

// -----

func.func @custom_inplace_type_conversion(%arg0: tensor<16xf16>,
                                          %arg1: tensor<16xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op inplace_outs requires mapped ins and outs operands to have identical ranked tensor or memref types}}
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_type_convert"}
      "user.type_convert"
      ins(%arg0 : tensor<16xf16> {inplace_outs = 0 : i32})
      outs(%arg1 : tensor<16xf32>) -> tensor<16xf32>
  return
}

// -----

func.func @custom_inplace_different_value(%arg0: tensor<16xf32>,
                                          %arg1: tensor<16xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom' op inplace_outs requires each mapped ins operand to be the same SSA value as its mapped outs operand}}
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_different_value"}
      "user.different_value"
      ins(%arg0 : tensor<16xf32> {inplace_outs = 0 : i32})
      outs(%arg1 : tensor<16xf32>) -> tensor<16xf32>
  return
}

// -----

func.func @custom_macro_inplace_type_conversion(%arg0: tensor<16xf16>,
                                                %arg1: tensor<16xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // expected-error @+1 {{'hivm.hir.custom_macro' op inplace_outs requires mapped ins and outs operands to have identical ranked tensor or memref types}}
  %0 = hivm.hir.custom_macro
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
       hivm.pipe_out = #hivm.pipe<PIPE_V>,
       hivm.vf_mode = #hivm.vf_mode<SIMD>,
       symbol = "k_macro_type_convert"}
      "user.macro_type_convert"
      ins(%arg0 : tensor<16xf16> {inplace_outs = 0 : i32})
      outs(%arg1 : tensor<16xf32>) -> tensor<16xf32>
  return
}
