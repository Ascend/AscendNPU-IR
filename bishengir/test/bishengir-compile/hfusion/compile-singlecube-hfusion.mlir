// RUN: bishengir-compile -enable-lir-compile=false -enable-hfusion-compile=true -block-dim=20 %s | FileCheck %s

// CHECK-LABEL: @single_cube
func.func @single_cube(%arg0: tensor<?x4096xf16>, %arg1: tensor<6144x4096xf16>, %arg2: tensor<?x6144xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> tensor<?x6144xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SINGLE_CUBE>} {
  %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<?x4096xf16>, tensor<6144x4096xf16>) outs(%arg2 : tensor<?x6144xf16>) -> tensor<?x6144xf16>
  return %0 : tensor<?x6144xf16>
}
