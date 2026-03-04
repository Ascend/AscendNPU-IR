// RUN: bishengir-opt --hfusion-auto-schedule -split-input-file %s | FileCheck %s

module {
  // CHECK: testA_multi_LAST_AXIS_PBR_0_tiling_function
  // CHECK: tensor<7x4096xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}
  // CHECK: tensor<7xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}
  // CHECK: tensor<7xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  // CHECK: tensor<7x1xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}
  // CHECK: testA_multi_LAST_AXIS_PBR_0_0
  // CHECK: tensor<7x4096xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}
  // CHECK: tensor<7xf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}
  // CHECK: tensor<7xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}
  // CHECK: tensor<7x1xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}
  // CHECK: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>}
  // CHECK: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
  // CHECK: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
  // CHECK: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
  // CHECK: testA
  // CHECK: {{.*}}: tensor<7x4096xf16>,
  // CHECK: {{.*}}: tensor<7xf16>,
  // CHECK: {{.*}}: tensor<7xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>},
  // CHECK: {{.*}}: tensor<7x1xf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>},
  // CHECK: {{.*}}: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
  // CHECK: {{.*}}: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
  // CHECK: {{.*}}: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
  func.func @testA_multi_LAST_AXIS_PBR_0(%arg0: tensor<7x4096xf16>, %arg1: tensor<7xf16>) -> (tensor<7xf16>, tensor<7x1xf16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %0 = tensor.empty() : tensor<7xf16>
    %1 = tensor.empty() : tensor<7x4096xf32>
    %2 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<7x4096xf16>) outs(%1 : tensor<7x4096xf32>) -> tensor<7x4096xf32>
    %3 = tensor.empty() : tensor<7xf32>
    %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<7xf16>) outs(%3 : tensor<7xf32>) -> tensor<7xf32>
    %reduced = linalg.reduce { arith.addf } ins(%2 : tensor<7x4096xf32>) outs(%4 : tensor<7xf32>) dimensions = [1] 
    %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<7xf32>) outs(%0 : tensor<7xf16>) -> tensor<7xf16>
    %6 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%5 : tensor<7xf16>) outs(%0 : tensor<7xf16>) -> tensor<7xf16>
    %expanded = tensor.expand_shape %6 [[0, 1]] output_shape [7, 1] : tensor<7xf16> into tensor<7x1xf16>
    return %5, %expanded : tensor<7xf16>, tensor<7x1xf16>
  }
  func.func @testA(%arg0: tensor<7x4096xf16>, %arg1: tensor<7xf16>) -> (tensor<7xf16>, tensor<7x1xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_VV>} {
    %0:2 = call @testA_multi_LAST_AXIS_PBR_0(%arg0, %arg1) : (tensor<7x4096xf16>, tensor<7xf16>) -> (tensor<7xf16>, tensor<7x1xf16>)
    return %0#0, %0#1 : tensor<7xf16>, tensor<7x1xf16>
  }
}