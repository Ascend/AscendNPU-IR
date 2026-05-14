// RUN: bishengir-opt %s --hivm-convert-layout-to-transpose --split-input-file | FileCheck %s

// CHECK: func.func @nd_to_fractal_aligned_f16(%[[VAL_0:.*]]: tensor<128x144xf16>)
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2, 3]] output_shape [8, 16, 9, 16] : tensor<128x144xf16> into tensor<8x16x9x16xf16>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<9x8x16x16xf16>
// CHECK: %[[VAL_3:.*]] = hivm.hir.vtranspose ins(%[[VAL_1]] : tensor<8x16x9x16xf16>) outs(%[[VAL_2]] : tensor<9x8x16x16xf16>) permutation = [2, 0, 1, 3] -> tensor<9x8x16x16xf16>
func.func @nd_to_fractal_aligned_f16(%arg0: tensor<128x144xf16>) -> tensor<9x8x16x16xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [9, 8, 16, 16]
       {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
        srcLayout = #hivm.data_layout<ND, transpose = false>}
       : (tensor<128x144xf16>) -> tensor<9x8x16x16xf16>
  return %0 : tensor<9x8x16x16xf16>
}

// -----

// CHECK: func.func @fractal_to_nd_aligned_f16(%[[VAL_0:.*]]: tensor<9x8x16x16xf16>)
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<8x16x9x16xf16>
// CHECK: %[[VAL_2:.*]] = hivm.hir.vtranspose ins(%[[VAL_0]] : tensor<9x8x16x16xf16>) outs(%[[VAL_1]] : tensor<8x16x9x16xf16>) permutation = [1, 2, 0, 3] -> tensor<8x16x9x16xf16>
// CHECK: %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0, 1], [2, 3]] : tensor<8x16x9x16xf16> into tensor<128x144xf16>
func.func @fractal_to_nd_aligned_f16(%arg0: tensor<9x8x16x16xf16>) -> tensor<128x144xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [128, 144]
       {dstLayout = #hivm.data_layout<ND, transpose = false>,
        srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
       : (tensor<9x8x16x16xf16>) -> tensor<128x144xf16>
  return %0 : tensor<128x144xf16>
}

// -----

// CHECK: func.func @nd_to_fractal_aligned_f32(%[[VAL_0:.*]]: tensor<32x256xf32>) -> tensor<16x2x16x16xf32> {
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2, 3]] output_shape [2, 16, 16, 16] : tensor<32x256xf32> into tensor<2x16x16x16xf32>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<16x2x16x16xf32>
// CHECK: %[[VAL_3:.*]] = hivm.hir.vtranspose ins(%[[VAL_1]] : tensor<2x16x16x16xf32>) outs(%[[VAL_2]] : tensor<16x2x16x16xf32>) permutation = [2, 0, 1, 3] -> tensor<16x2x16x16xf32>
// CHECK: return %[[VAL_3]] : tensor<16x2x16x16xf32>
func.func @nd_to_fractal_aligned_f32(%arg0: tensor<32x256xf32>) -> tensor<16x2x16x16xf32> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [16, 2, 16, 16]
       {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
        srcLayout = #hivm.data_layout<ND, transpose = true>}
       : (tensor<32x256xf32>) -> tensor<16x2x16x16xf32>
  return %0 : tensor<16x2x16x16xf32>
}

// -----

// CHECK: func.func @fractal_to_nd_aligned_f32(%[[VAL_0:.*]]: tensor<16x2x16x16xf16>) -> tensor<32x256xf16> {
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<2x16x16x16xf16>
// CHECK: %[[VAL_2:.*]] = hivm.hir.vtranspose ins(%[[VAL_0]] : tensor<16x2x16x16xf16>) outs(%[[VAL_1]] : tensor<2x16x16x16xf16>) permutation = [1, 2, 0, 3] -> tensor<2x16x16x16xf16>
// CHECK: %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0, 1], [2, 3]] : tensor<2x16x16x16xf16> into tensor<32x256xf16>
// CHECK: return %[[VAL_3]] : tensor<32x256xf16>
func.func @fractal_to_nd_aligned_f32(%arg0: tensor<16x2x16x16xf16>) -> tensor<32x256xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [32, 256]
       {dstLayout = #hivm.data_layout<ND, transpose = true>,
        srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
       : (tensor<16x2x16x16xf16>) -> tensor<32x256xf16>
  return %0 : tensor<32x256xf16>
}

// -----

// ND(128x144) -> Fractal[16,8] => [144/8, 128/16, 16, 8] = [18, 8, 16, 8]
// CHECK: func.func @nd_to_fractal_aligned_16x8_f16(%[[VAL_0:.*]]: tensor<128x144xf16>) -> tensor<18x8x16x8xf16> {
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2, 3]] output_shape [8, 16, 18, 8] : tensor<128x144xf16> into tensor<8x16x18x8xf16>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<18x8x16x8xf16>
// CHECK: %[[VAL_3:.*]] = hivm.hir.vtranspose ins(%[[VAL_1]] : tensor<8x16x18x8xf16>) outs(%[[VAL_2]] : tensor<18x8x16x8xf16>) permutation = [2, 0, 1, 3] -> tensor<18x8x16x8xf16>
// CHECK: return %[[VAL_3]] : tensor<18x8x16x8xf16>
func.func @nd_to_fractal_aligned_16x8_f16(%arg0: tensor<128x144xf16>) -> tensor<18x8x16x8xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [18, 8, 16, 8]
       {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>,
        srcLayout = #hivm.data_layout<ND, transpose = false>}
       : (tensor<128x144xf16>) -> tensor<18x8x16x8xf16>
  return %0 : tensor<18x8x16x8xf16>
}

// -----

// Fractal[16,8] -> ND(128x144)
// CHECK: func.func @fractal_to_nd_aligned_16x8_f16(%[[VAL_0:.*]]: tensor<18x8x16x8xf16>) -> tensor<128x144xf16> {
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<8x16x18x8xf16>
// CHECK: %[[VAL_2:.*]] = hivm.hir.vtranspose ins(%[[VAL_0]] : tensor<18x8x16x8xf16>) outs(%[[VAL_1]] : tensor<8x16x18x8xf16>) permutation = [1, 2, 0, 3] -> tensor<8x16x18x8xf16>
// CHECK: %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0, 1], [2, 3]] : tensor<8x16x18x8xf16> into tensor<128x144xf16>
// CHECK: return %[[VAL_3]] : tensor<128x144xf16>
func.func @fractal_to_nd_aligned_16x8_f16(%arg0: tensor<18x8x16x8xf16>) -> tensor<128x144xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [128, 144]
       {dstLayout = #hivm.data_layout<ND, transpose = false>,
        srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>}
       : (tensor<18x8x16x8xf16>) -> tensor<128x144xf16>
  return %0 : tensor<128x144xf16>
}

// -----

// ND(32x256) -> Fractal[16,8] => [256/8, 32/16, 16, 8] = [32, 2, 16, 8]
// CHECK: func.func @nd_to_fractal_aligned_16x8_f32(%[[VAL_0:.*]]: tensor<32x256xf32>) -> tensor<32x2x16x8xf32> {
// CHECK: %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2, 3]] output_shape [2, 16, 32, 8] : tensor<32x256xf32> into tensor<2x16x32x8xf32>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<32x2x16x8xf32>
// CHECK: %[[VAL_3:.*]] = hivm.hir.vtranspose ins(%[[VAL_1]] : tensor<2x16x32x8xf32>) outs(%[[VAL_2]] : tensor<32x2x16x8xf32>) permutation = [2, 0, 1, 3] -> tensor<32x2x16x8xf32>
// CHECK: return %[[VAL_3]] : tensor<32x2x16x8xf32>
func.func @nd_to_fractal_aligned_16x8_f32(%arg0: tensor<32x256xf32>) -> tensor<32x2x16x8xf32> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [32, 2, 16, 8]
       {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>,
        srcLayout = #hivm.data_layout<ND, transpose = false>}
       : (tensor<32x256xf32>) -> tensor<32x2x16x8xf32>
  return %0 : tensor<32x2x16x8xf32>
}

// -----

// Fractal[16,8] -> ND(32x256)
// CHECK: func.func @fractal_to_nd_aligned_16x8_f32(%[[VAL_0:.*]]: tensor<32x2x16x8xf32>) -> tensor<32x256xf32> {
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<2x16x32x8xf32>
// CHECK: %[[VAL_2:.*]] = hivm.hir.vtranspose ins(%[[VAL_0]] : tensor<32x2x16x8xf32>) outs(%[[VAL_1]] : tensor<2x16x32x8xf32>) permutation = [1, 2, 0, 3] -> tensor<2x16x32x8xf32>
// CHECK: %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0, 1], [2, 3]] : tensor<2x16x32x8xf32> into tensor<32x256xf32>
// CHECK: return %[[VAL_3]] : tensor<32x256xf32>
func.func @fractal_to_nd_aligned_16x8_f32(%arg0: tensor<32x2x16x8xf32>) -> tensor<32x256xf32> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [32, 256]
       {dstLayout = #hivm.data_layout<ND, transpose = false>,
        srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>}
       : (tensor<32x2x16x8xf32>) -> tensor<32x256xf32>
  return %0 : tensor<32x256xf32>
}


// CHECK-LABEL: func.func @nd_to_fractal_unaligned_130x145(
// CHECK: arith.constant 0.000000e+00 : f16
// CHECK: tensor.empty() : tensor<144x160xf16>
// CHECK: hivm.hir.vbrc
// CHECK: tensor.insert_slice
// CHECK: tensor.expand_shape
// CHECK: hivm.hir.vtranspose
func.func @nd_to_fractal_unaligned_130x145(%arg0: tensor<130x145xf16>) -> tensor<10x9x16x16xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [10, 9, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND, transpose = false>}
      : (tensor<130x145xf16>) -> tensor<10x9x16x16xf16>
  return %0 : tensor<10x9x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @nd_to_fractal_unaligned_17x33(
// CHECK: tensor.empty() : tensor<32x48xf16>
// CHECK: hivm.hir.vbrc
// CHECK: tensor.insert_slice
// CHECK: tensor.expand_shape
// CHECK: hivm.hir.vtranspose
func.func @nd_to_fractal_unaligned_17x33(%arg0: tensor<17x33xf16>) -> tensor<3x2x16x16xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [3, 2, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND, transpose = false>}
      : (tensor<17x33xf16>) -> tensor<3x2x16x16xf16>
  return %0 : tensor<3x2x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @nd_to_fractal_unaligned_1x1(
// CHECK: tensor.empty() : tensor<16x16xf16>
// CHECK: hivm.hir.vbrc
// CHECK: tensor.insert_slice
// CHECK: tensor.expand_shape
// CHECK: hivm.hir.vtranspose
func.func @nd_to_fractal_unaligned_1x1(%arg0: tensor<1x1xf16>) -> tensor<1x1x16x16xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [1, 1, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND, transpose = false>}
      : (tensor<1x1xf16>) -> tensor<1x1x16x16xf16>
  return %0 : tensor<1x1x16x16xf16>
}

// -----
// =========================
// Fractal -> ND (UNALIGNED) static
// =========================

// CHECK-LABEL: func.func @fractal_to_nd_unaligned_130x145(
// CHECK: hivm.hir.vtranspose
// CHECK: tensor.collapse_shape
// CHECK: tensor.extract_slice
func.func @fractal_to_nd_unaligned_130x145(%arg0: tensor<10x9x16x16xf16>) -> tensor<130x145xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [130, 145]
      {dstLayout = #hivm.data_layout<ND, transpose = false>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<10x9x16x16xf16>) -> tensor<130x145xf16>
  return %0 : tensor<130x145xf16>
}

// -----

// CHECK-LABEL: func.func @fractal_to_nd_unaligned_17x33(
// CHECK: hivm.hir.vtranspose
// CHECK: tensor.collapse_shape
// CHECK: tensor.extract_slice
func.func @fractal_to_nd_unaligned_17x33(%arg0: tensor<3x2x16x16xf16>) -> tensor<17x33xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [17, 33]
      {dstLayout = #hivm.data_layout<ND, transpose = false>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<3x2x16x16xf16>) -> tensor<17x33xf16>
  return %0 : tensor<17x33xf16>
}

// -----
// =========================
// ND -> Fractal (UNALIGNED) dynamic
// =========================

// CHECK-LABEL: func.func @nd_to_fractal_unaligned_dynamic_qq(
// CHECK: tensor.empty
// CHECK: hivm.hir.vbrc
// CHECK: tensor.insert_slice
// CHECK: tensor.expand_shape
// CHECK: hivm.hir.vtranspose
func.func @nd_to_fractal_unaligned_dynamic_qq(%arg0: tensor<?x?xf16>) -> tensor<?x?x16x16xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %n = tensor.dim %arg0, %c1 : tensor<?x?xf16>
  %mt = affine.apply affine_map<(d0) -> ((d0 + 15) floordiv 16)>(%m)
  %nt = affine.apply affine_map<(d0) -> ((d0 + 15) floordiv 16)>(%n)

  %0 = hivm.hir.convert_layout %arg0 output_shape [%nt, %mt, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND, transpose = false>}
      : (tensor<?x?xf16>) -> tensor<?x?x16x16xf16>
  return %0 : tensor<?x?x16x16xf16>
}

// -----

// M dynamic, N static unaligned (=145).
// CHECK-LABEL: func.func @nd_to_fractal_unaligned_dynamic_m_static_n145(
// CHECK: hivm.hir.vbrc
// CHECK: tensor.insert_slice
// CHECK: tensor.expand_shape
// CHECK: hivm.hir.vtranspose
func.func @nd_to_fractal_unaligned_dynamic_m_static_n145(%arg0: tensor<?x145xf16>) -> tensor<10x?x16x16xf16> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x145xf16>
  %mt = affine.apply affine_map<(d0) -> ((d0 + 15) floordiv 16)>(%m)

  %0 = hivm.hir.convert_layout %arg0 output_shape [10, %mt, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND, transpose = false>}
      : (tensor<?x145xf16>) -> tensor<10x?x16x16xf16>
  return %0 : tensor<10x?x16x16xf16>
}

// -----

// M static unaligned (=130), N dynamic.
// CHECK-LABEL: func.func @nd_to_fractal_unaligned_static_m130_dynamic_n(
// CHECK: hivm.hir.vbrc
// CHECK: tensor.insert_slice
// CHECK: tensor.expand_shape
// CHECK: hivm.hir.vtranspose
func.func @nd_to_fractal_unaligned_static_m130_dynamic_n(%arg0: tensor<130x?xf16>) -> tensor<?x9x16x16xf16> {
  %c1 = arith.constant 1 : index
  %n = tensor.dim %arg0, %c1 : tensor<130x?xf16>
  %nt = affine.apply affine_map<(d0) -> ((d0 + 15) floordiv 16)>(%n)

  %0 = hivm.hir.convert_layout %arg0 output_shape [%nt, 9, 16, 16]
      {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>,
       srcLayout = #hivm.data_layout<ND, transpose = false>}
      : (tensor<130x?xf16>) -> tensor<?x9x16x16xf16>
  return %0 : tensor<?x9x16x16xf16>
}

// -----
// =========================
// Fractal -> ND (UNALIGNED) dynamic
// =========================

// CHECK-LABEL: func.func @fractal_to_nd_unaligned_dynamic_qq(
// CHECK: hivm.hir.vtranspose
// CHECK: tensor.collapse_shape
// CHECK: tensor.extract_slice
func.func @fractal_to_nd_unaligned_dynamic_qq(%arg0: tensor<?x?x16x16xf16>, %m: index, %n: index) -> tensor<?x?xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [%m, %n]
      {dstLayout = #hivm.data_layout<ND, transpose = false>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<?x?x16x16xf16>) -> tensor<?x?xf16>
  return %0 : tensor<?x?xf16>
}

// -----

// Dynamic N only, M fixed unaligned (=130).
// CHECK-LABEL: func.func @fractal_to_nd_unaligned_static_m130_dynamic_n(
// CHECK: hivm.hir.vtranspose
// CHECK: tensor.collapse_shape
// CHECK: tensor.extract_slice
func.func @fractal_to_nd_unaligned_static_m130_dynamic_n(%arg0: tensor<?x9x16x16xf16>, %n: index) -> tensor<130x?xf16> {
  %0 = hivm.hir.convert_layout %arg0 output_shape [130, %n]
      {dstLayout = #hivm.data_layout<ND, transpose = false>,
       srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>}
      : (tensor<?x9x16x16xf16>) -> tensor<130x?xf16>
  return %0 : tensor<130x?xf16>
}
