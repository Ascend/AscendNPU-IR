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
// -----

// The intermediate result has multiple uses: the inverse conversion is folded
// back to the original source while the first conversion is kept for the
// remaining use.
// CHECK-LABEL: @eliminate_inverse_conversion_with_multiple_uses(
// CHECK-SAME: %[[ARG:.*]]: tensor<8x8x16x16xf16>
// CHECK: %[[B:.+]] = hivm.hir.convert_layout %[[ARG]] output_shape [128, 128] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>}
// CHECK: %[[C:.+]] = hivm.hir.convert_layout %[[B]] output_shape [64, 256] {dstLayout = #hivm.data_layout<zN>, srcLayout = #hivm.data_layout<ND>}
// CHECK: return %[[ARG]], %[[C]]
#ND_layout = #hivm.data_layout<ND>
#nZ_layout = #hivm.data_layout<nZ>
#zN_layout = #hivm.data_layout<zN>
func.func @eliminate_inverse_conversion_with_multiple_uses(%arg : tensor<8x8x16x16xf16>) -> (tensor<8x8x16x16xf16>, tensor<64x256xf16>) {
  %converted_layout = hivm.hir.convert_layout %arg output_shape [128, 128] {srcLayout = #nZ_layout, dstLayout = #ND_layout}
                        : (tensor<8x8x16x16xf16>) -> tensor<128x128xf16>
  %converted_layout_2 = hivm.hir.convert_layout %converted_layout output_shape [8, 8, 16, 16] {srcLayout = #ND_layout, dstLayout = #nZ_layout}
                        : (tensor<128x128xf16>) -> tensor<8x8x16x16xf16>
  %converted_layout_3 = hivm.hir.convert_layout %converted_layout output_shape [64, 256] {srcLayout = #ND_layout, dstLayout = #zN_layout}
                        : (tensor<128x128xf16>) -> tensor<64x256xf16>
  return %converted_layout_2, %converted_layout_3 : tensor<8x8x16x16xf16>, tensor<64x256xf16>
}
// -----

// Opposite direction: the ND value is the original source and the roundtrip
// goes through nZ.
// CHECK-LABEL: @eliminate_inverse_conversion_opposite_direction(
// CHECK-SAME: %[[ARG:.*]]: tensor<128x128xf16>
// CHECK: %[[B:.+]] = hivm.hir.convert_layout %[[ARG]] output_shape [8, 8, 16, 16] {dstLayout = #hivm.data_layout<nZ>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[C:.+]] = hivm.hir.convert_layout %[[B]] output_shape [64, 256] {dstLayout = #hivm.data_layout<zN>, srcLayout = #hivm.data_layout<nZ>}
// CHECK: return %[[ARG]], %[[C]]
#ND_layout = #hivm.data_layout<ND>
#nZ_layout = #hivm.data_layout<nZ>
#zN_layout = #hivm.data_layout<zN>
func.func @eliminate_inverse_conversion_opposite_direction(%arg : tensor<128x128xf16>) -> (tensor<128x128xf16>, tensor<64x256xf16>) {
  %converted_layout = hivm.hir.convert_layout %arg output_shape [8, 8, 16, 16] {srcLayout = #ND_layout, dstLayout = #nZ_layout}
                        : (tensor<128x128xf16>) -> tensor<8x8x16x16xf16>
  %converted_layout_2 = hivm.hir.convert_layout %converted_layout output_shape [128, 128] {srcLayout = #nZ_layout, dstLayout = #ND_layout}
                        : (tensor<8x8x16x16xf16>) -> tensor<128x128xf16>
  %converted_layout_3 = hivm.hir.convert_layout %converted_layout output_shape [64, 256] {srcLayout = #nZ_layout, dstLayout = #zN_layout}
                        : (tensor<8x8x16x16xf16>) -> tensor<64x256xf16>
  return %converted_layout_2, %converted_layout_3 : tensor<128x128xf16>, tensor<64x256xf16>
}
// -----

// Fractal layouts carry a fractalSizes attribute; matching fractal sizes make
// the roundtrip an inverse pair that folds.
// CHECK-LABEL: @eliminate_inverse_conversion_fractal(
// CHECK-SAME: %[[ARG:.*]]: tensor<16x16xf16>
// CHECK: %[[F:.+]] = hivm.hir.convert_layout %[[ARG]] output_shape [1, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[Z:.+]] = hivm.hir.convert_layout %[[F]] output_shape [16, 16] {dstLayout = #hivm.data_layout<zN>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
// CHECK: return %[[ARG]], %[[Z]]
func.func @eliminate_inverse_conversion_fractal(%arg : tensor<16x16xf16>) -> (tensor<16x16xf16>, tensor<16x16xf16>) {
  %fractal = hivm.hir.convert_layout %arg output_shape [1, 1, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
                        : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  %roundtrip = hivm.hir.convert_layout %fractal output_shape [16, 16] {srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, dstLayout = #hivm.data_layout<ND>}
                        : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>
  %other = hivm.hir.convert_layout %fractal output_shape [16, 16] {srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, dstLayout = #hivm.data_layout<zN>}
                        : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>
  return %roundtrip, %other : tensor<16x16xf16>, tensor<16x16xf16>
}
// -----

// Different fractal sizes are not inverse layouts: nothing is folded.
// CHECK-LABEL: @no_fold_different_fractal_sizes(
// CHECK: %[[F:.+]] = hivm.hir.convert_layout %arg0 output_shape [1, 1, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>}
// CHECK: %[[R:.+]] = hivm.hir.convert_layout %[[F]] output_shape [16, 16] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>}
// CHECK: return %[[R]]
func.func @no_fold_different_fractal_sizes(%arg : tensor<16x16xf16>) -> tensor<16x16xf16> {
  %fractal = hivm.hir.convert_layout %arg output_shape [1, 1, 16, 16] {srcLayout = #hivm.data_layout<ND>, dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
                        : (tensor<16x16xf16>) -> tensor<1x1x16x16xf16>
  %roundtrip = hivm.hir.convert_layout %fractal output_shape [16, 16] {srcLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>, dstLayout = #hivm.data_layout<ND>}
                        : (tensor<1x1x16x16xf16>) -> tensor<16x16xf16>
  return %roundtrip : tensor<16x16xf16>
}
