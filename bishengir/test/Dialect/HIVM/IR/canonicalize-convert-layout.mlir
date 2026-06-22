// RUN: bishengir-opt %s --cse --canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @eliminate_redundant_conversions(
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: return
#ND_layout = #hivm.data_layout<ND>
#nZ_layout = #hivm.data_layout<nZ>
func.func @eliminate_redundant_conversions(%arg : tensor<8x8x16x16xf16>) -> tensor<8x8x16x16xf16> {
  %converted_layout = hivm.hir.convert_layout %arg output_shape [128, 128] {srcLayout = #nZ_layout, dstLayout = #ND_layout}
                        : (tensor<8x8x16x16xf16>) -> tensor<128x128xf16>
  %converted_layout_2 = hivm.hir.convert_layout %converted_layout output_shape [8, 8, 16, 16] {srcLayout = #ND_layout, dstLayout = #nZ_layout}
                        : (tensor<128x128xf16>) -> tensor<8x8x16x16xf16>
  return %converted_layout_2 : tensor<8x8x16x16xf16>
}
// -----

// CHECK-LABEL: @eliminate_redundant_conversions(
// CHECK-NOT: hivm.hir.convert_layout
// CHECK: return
#ND_layout = #hivm.data_layout<ND>
#nZ_layout = #hivm.data_layout<nZ>
func.func @eliminate_redundant_conversions(%arg : tensor<8x8x16x16xf16>) -> tensor<8x8x16x16xf16> {
  %converted_layout = hivm.hir.convert_layout %arg output_shape [128, 128]  {srcLayout = #nZ_layout, dstLayout = #ND_layout}
                        : (tensor<8x8x16x16xf16>) -> tensor<128x128xf16>
  %converted_layout_2 = hivm.hir.convert_layout %converted_layout output_shape [8, 8, 16, 16] {srcLayout = #ND_layout, dstLayout = #nZ_layout}
                        : (tensor<128x128xf16>) -> tensor<8x8x16x16xf16>
  return %converted_layout_2 : tensor<8x8x16x16xf16>
}
