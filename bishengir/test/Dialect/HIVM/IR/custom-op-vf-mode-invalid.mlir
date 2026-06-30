// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

// -----

// Reg-based (950PR) vector CustomOp must specify vf_mode.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @custom_missing_vf_mode(%arg0: memref<1xf32>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %empty = tensor.empty() : tensor<1xf32>
    // expected-error @+1 {{Missing vf mode information}}
    %0 = hivm.hir.custom
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe = #hivm.pipe<PIPE_V>,
         symbol = "k_no_vf"}
        "user.no_vf"
        ins(%arg0 : memref<1xf32>)
        outs(%empty : tensor<1xf32>) -> tensor<1xf32>
    return
  }
}

// -----

// Reg-based (950PR) vector CustomMacroOp must specify vf_mode.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @custom_macro_missing_vf_mode(%arg0: memref<4xf32>)
      attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // expected-error @+1 {{Missing vf mode information}}
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
         hivm.pipe_out = #hivm.pipe<PIPE_V>,
         symbol = "k_macro_no_vf"}
        "user.macro_no_vf"
        ins(%arg0 : memref<4xf32>)
        outs(%arg0 : memref<4xf32>)
    return
  }
}

// -----

// Mem-based default (no hacc.target): vf_mode is optional for vector CustomOp.
func.func @custom_vf_mode_optional_on_mem_based(%arg0: memref<1xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %empty = tensor.empty() : tensor<1xf32>
  %0 = hivm.hir.custom
      {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
       hivm.pipe = #hivm.pipe<PIPE_V>,
       symbol = "k_optional_vf"}
      "user.optional_vf"
      ins(%arg0 : memref<1xf32>)
      outs(%empty : tensor<1xf32>) -> tensor<1xf32>
  return
}
