// REQUIRES: enable-lir-compile

// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s

// CHECK-LABEL: @single_cube
func.func @single_cube(%arg0: tensor<2x64xf16>, %arg1: tensor<96x64xf16>, %arg2: tensor<2x96xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> tensor<2x96xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SINGLE_CUBE>} {
  %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<2x64xf16>, tensor<96x64xf16>) outs(%arg2 : tensor<2x96xf16>) -> tensor<2x96xf16>
  return %0 : tensor<2x96xf16>
}
