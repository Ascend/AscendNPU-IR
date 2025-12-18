// REQUIRES: asserts
// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file -debug-only="hfusion-auto-schedule" -verify-diagnostics 2>&1 | FileCheck %s -check-prefix=CHECK-ALIGN

// CHECK-ALIGN: dimension to align: 1
// CHECK-ALIGN: type before alignment: tensor<24x32x12xf32>
// CHECK-ALIGN: dim: 2 stride is aligned to: 8
module {
  func.func @reduce(%arg0: tensor<24x32x12xf32>, %arg1: tensor<24x32xf32>) -> (tensor<24x32xf32>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %reduced = linalg.reduce ins(%arg0 : tensor<24x32x12xf32>) outs(%arg1 : tensor<24x32xf32>) dimensions = [2] 
      (%in: f32, %init: f32) {
        %0 = arith.addf %in, %init : f32
        linalg.yield %0: f32
      }
    return %reduced : tensor<24x32xf32>
  }
}

// -----

// CHECK-ALIGN: dimension to align: 2
// CHECK-ALIGN: type before alignment: tensor<24x32x16x36xf32>
// CHECK-ALIGN: dim: 3 stride is aligned to: 8
module {
  func.func @brodacast(%arg0: tensor<24x32xf32>, %arg1: tensor<24x32x16x36xf32>) -> (tensor<24x32x16x36xf32>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %broadcasted = linalg.broadcast ins(%arg0 : tensor<24x32xf32>) outs(%arg1 : tensor<24x32x16x36xf32>) dimensions = [2, 3] 
    return %broadcasted : tensor<24x32x16x36xf32>
  }
}

// -----

// CHECK-ALIGN: dimension to align: 0
// CHECK-ALIGN: type before alignment: tensor<12x36xf32>
// CHECK-ALIGN: dim: 1 stride is aligned to: 8
module {
  func.func @brodacast(%arg0: tensor<36xf32>, %arg1: tensor<12x36xf32>) -> (tensor<12x36xf32>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %broadcasted = linalg.broadcast ins(%arg0 : tensor<36xf32>) outs(%arg1 : tensor<12x36xf32>) dimensions = [0] 
    return %broadcasted : tensor<12x36xf32>
  }
}

// -----

// CHECK-ALIGN: type before alignment: tensor<32768x169xi1>
// CHECK-ALIGN: dim: 1 stride is aligned to: 256
module {
  func.func @reduce_with_i1(%arg0: tensor<128x256x13x13xi8>, %arg1: tensor<128x256x13x13xf16>) -> (tensor<128x256x13x13xf16>, tensor<128x256xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<32768x169xf16>
    %1 = linalg.fill ins(%cst_0 : f16) outs(%0 : tensor<32768x169xf16>) -> tensor<32768x169xf16>
    %2 = tensor.empty() : tensor<32768x169xi1>
    %3 = tensor.empty() : tensor<32768x169xf32>
    %4 = tensor.empty() : tensor<32768xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<32768xf32>) -> tensor<32768xf32>
    %6 = tensor.empty() : tensor<32768xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<128x256x13x13xi8> into tensor<32768x169xi8>
    %7 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%collapsed : tensor<32768x169xi8>) outs(%0 : tensor<32768x169xf16>) -> tensor<32768x169xf16>
    %collapsed_1 = tensor.collapse_shape %arg1 [[0, 1], [2, 3]] : tensor<128x256x13x13xf16> into tensor<32768x169xf16>
    %8 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%7, %cst_0 : tensor<32768x169xf16>, f16) outs(%2 : tensor<32768x169xi1>) -> tensor<32768x169xi1>
    %9 = hfusion.select ins(%8, %1, %collapsed_1 : tensor<32768x169xi1>, tensor<32768x169xf16>, tensor<32768x169xf16>) outs(%0 : tensor<32768x169xf16>) -> tensor<32768x169xf16>
    %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%9 : tensor<32768x169xf16>) outs(%3 : tensor<32768x169xf32>) -> tensor<32768x169xf32>
    %expanded = tensor.expand_shape %9 [[0, 1], [2, 3]] output_shape [128, 256, 13, 13] : tensor<32768x169xf16> into tensor<128x256x13x13xf16>
    %reduced = linalg.reduce ins(%10 : tensor<32768x169xf32>) outs(%5 : tensor<32768xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %12 = arith.addf %in, %init : f32
        linalg.yield %12 : f32
      }
    %11 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%reduced : tensor<32768xf32>) outs(%6 : tensor<32768xf16>) -> tensor<32768xf16>
    %expanded_2 = tensor.expand_shape %11 [[0, 1]] output_shape [128, 256] : tensor<32768xf16> into tensor<128x256xf16>
    return %expanded, %expanded_2 : tensor<128x256x13x13xf16>, tensor<128x256xf16>
  }
}

// -----

// CHECK-ALIGN: type before alignment: tensor<80000x1xf32>
// CHECK-ALIGN: dim: 1 stride is aligned to: 8
module {
  func.func @mlir_fused_add_mul_71(%arg0: tensor<1x1x80000x3xf32>) -> tensor<1x1x80000x1xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %cst = arith.constant 1.000000e+02 : f32
    %cst_0 = arith.constant -5.000000e+01 : f32
    %0 = tensor.empty() : tensor<80000x1xf32>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3]] : tensor<1x1x80000x3xf32> into tensor<80000x3xf32>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0] [80000, 1] [1, 1] : tensor<80000x3xf32> to tensor<80000x1xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%extracted_slice, %cst : tensor<80000x1xf32>, f32) outs(%0 : tensor<80000x1xf32>) -> tensor<80000x1xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %cst_0 : tensor<80000x1xf32>, f32) outs(%0 : tensor<80000x1xf32>) -> tensor<80000x1xf32>
    %expanded = tensor.expand_shape %2 [[0, 1, 2], [3]] output_shape [1, 1, 80000, 1] : tensor<80000x1xf32> into tensor<1x1x80000x1xf32>
    return %expanded : tensor<1x1x80000x1xf32>
  }
}

// -----

// CHECK-ALIGN-NOT: stride is aligned to
// CHECK-ALIGN-DAG: dim: 0 size is aligned to: 16
// CHECK-ALIGN-DAG: dim: 1 size is aligned to: 16
module {
  func.func @transpose(%b32:tensor<17x23xf32>) -> tensor<23x17xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<23x17xf32>
    %t32 = linalg.transpose ins(%b32:tensor<17x23xf32>) outs(%0:tensor<23x17xf32>) permutation = [1, 0]
    return %t32: tensor<23x17xf32>
  }
}

// -----

// CHECK-ALIGN-NOT: stride is aligned to
// CHECK-ALIGN-DAG: dim: 0 size is aligned to: 16
// CHECK-ALIGN-DAG: dim: 1 size is aligned to: 16
module {
  func.func @transpose(%b16:tensor<17x23xf16>) -> tensor<23x17xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<23x17xf16>
    %t16 = linalg.transpose ins(%b16:tensor<17x23xf16>) outs(%0:tensor<23x17xf16>) permutation = [1, 0]
    return %t16: tensor<23x17xf16>
  }
}

// -----

// CHECK-ALIGN-NOT: stride is aligned to
// CHECK-ALIGN-DAG: dim: 0 size is aligned to: 32
// CHECK-ALIGN-DAG: dim: 1 size is aligned to: 32
module {
  func.func @transpose(%b8:tensor<17x23xi8>) -> tensor<23x17xi8> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<23x17xi8>
    %t8 = linalg.transpose ins(%b8:tensor<17x23xi8>) outs(%0:tensor<23x17xi8>) permutation = [1, 0]
    return %t8: tensor<23x17xi8>
  }
}

// -----

// CHECK-ALIGN-DAG: dim: 2 stride is aligned to: 8
module {
  func.func @transpose_non_last_axis(%arg0: tensor<9x4488x75xf32>) -> tensor<4488x9x75xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
    %0 = tensor.empty() : tensor<4488x9x75xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<9x4488x75xf32>) outs(%0 : tensor<4488x9x75xf32>) permutation = [1, 0, 2] 
    return %transposed : tensor<4488x9x75xf32>
  }
}


// -----

// CHECK-ALIGN: permuteDims: 0, 3
// CHECK-ALIGN-NOT: permuteDims: 0, 1
// CHECK-ALIGN-DAG: dim: 0 size is aligned to: 16
// CHECK-ALIGN-DAG: dim: 3 size is aligned to: 16
module {
  func.func @transpose_with_one_size_axis_0(%arg0: tensor<64x32x32x1xf32>) -> tensor<64x32x32x1xf32> 
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %0 = tensor.empty() : tensor<1x32x32x64xf32>
    %1 = tensor.empty() : tensor<1x32x32x64xf32>
    %2 = tensor.empty() : tensor<64x32x32x1xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<64x32x32x1xf32>) outs(%0 : tensor<1x32x32x64xf32>) permutation = [3, 1, 2, 0] 
    %mul = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%transposed, %transposed : tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) outs(%1 : tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    %transposed_0 = linalg.transpose ins(%mul : tensor<1x32x32x64xf32>) outs(%2 : tensor<64x32x32x1xf32>) permutation = [3, 1, 2, 0] 
    return %transposed_0 : tensor<64x32x32x1xf32>
  }
}

// -----

// CHECK-ALIGN: type before alignment: tensor<106000x10xf16>
// CHECK-ALIGN: dim: 1 stride is aligned to: 16
func.func @test_concat_with_last_dim_unaligned(%arg0: tensor<1x96000x10xf16>, %arg1: tensor<1x10000x10xf16>) -> tensor<1x106000x10xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>} {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x96000x10xf16> into tensor<96000x10xf16>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2]] : tensor<1x10000x10xf16> into tensor<10000x10xf16>
  %concat = tensor.concat dim(0) %collapsed, %collapsed_0 : (tensor<96000x10xf16>, tensor<10000x10xf16>) -> tensor<106000x10xf16>
  %expanded = tensor.expand_shape %concat [[0, 1], [2]] output_shape [1, 106000, 10] : tensor<106000x10xf16> into tensor<1x106000x10xf16>
  return %expanded : tensor<1x106000x10xf16>
}

// -----

// CHECK-ALIGN: dim: 2 size is aligned to: 32
// CHECK-ALIGN: dim: 3 size is aligned to: 32
func.func @cast_i32_to_i8_rank2(%arg0: tensor<2x57x5x3xi32>) -> tensor<2x57x5x3xi8>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}{
  %0 = tensor.empty() : tensor<2x57x5x3xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} 
        ins(%arg0 : tensor<2x57x5x3xi32>) 
        outs(%0 : tensor<2x57x5x3xi8>) -> tensor<2x57x5x3xi8>
  return %1 : tensor<2x57x5x3xi8>
}

// -----

// CHECK-ALIGN: dim: 0 size is aligned to: 1024
func.func @cast_i32_to_i8_rank1_small_shape(%arg0: tensor<3xi32>) -> tensor<3xi8>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}{
  %0 = tensor.empty() : tensor<3xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} 
        ins(%arg0 : tensor<3xi32>) 
        outs(%0 : tensor<3xi8>) -> tensor<3xi8>
  return %1 : tensor<3xi8>
}

// -----

// CHECK-ALIGN: dim: 0 size is aligned to: 1024
func.func @cast_i32_to_i8_rank1_big_shape(%arg0: tensor<1111xi32>) -> tensor<1111xi8>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}{
  %0 = tensor.empty() : tensor<1111xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} 
        ins(%arg0 : tensor<1111xi32>) 
        outs(%0 : tensor<1111xi8>) -> tensor<1111xi8>
  return %1 : tensor<1111xi8>
}

// -----

// CHECK-ALIGN: dim: 2 size is aligned to: 32
// CHECK-ALIGN: dim: 3 size is aligned to: 32
func.func @cast_i16_to_i8_rank2(%arg0: tensor<2x57x5x3xi16>) -> tensor<2x57x5x3xi8>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}{
  %0 = tensor.empty() : tensor<2x57x5x3xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} 
        ins(%arg0 : tensor<2x57x5x3xi16>) 
        outs(%0 : tensor<2x57x5x3xi8>) -> tensor<2x57x5x3xi8>
  return %1 : tensor<2x57x5x3xi8>
}

// -----

// CHECK-ALIGN: dim: 0 size is aligned to: 1024
func.func @cast_i16_to_i8_rank1_small_shape(%arg0: tensor<3xi16>) -> tensor<3xi8>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}{
  %0 = tensor.empty() : tensor<3xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} 
        ins(%arg0 : tensor<3xi16>) 
        outs(%0 : tensor<3xi8>) -> tensor<3xi8>
  return %1 : tensor<3xi8>
}

// -----

// CHECK-ALIGN: dim: 0 size is aligned to: 1024
func.func @cast_i16_to_i8_rank1_big_shape(%arg0: tensor<1111xi16>) -> tensor<1111xi8>
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<PURE_ELEMWISE>}{
  %0 = tensor.empty() : tensor<1111xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} 
        ins(%arg0 : tensor<1111xi16>) 
        outs(%0 : tensor<1111xi8>) -> tensor<1111xi8>
  return %1 : tensor<1111xi8>
}

// -----

// CHECK-ALIGN: dim: 1 size is aligned to: 16
func.func @concat_last_dim_align_tile_size(%arg0: tensor<2x1024xi16>, %arg1: tensor<2x2048xi16>) -> tensor<2x3072x5x1024xi16> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %0 = tensor.empty() : tensor<2x3072x5x1024xi16>
  %concat = tensor.concat dim(1) %arg0, %arg1 : (tensor<2x1024xi16>, tensor<2x2048xi16>) -> tensor<2x3072xi16>
  %broadcasted = linalg.broadcast ins(%concat : tensor<2x3072xi16>) outs(%0 : tensor<2x3072x5x1024xi16>) dimensions = [2, 3] 
  return %broadcasted : tensor<2x3072x5x1024xi16>
}

// -----

// CHECK-ALIGN: dim: 0 size is aligned to: 32
// CHECK-ALIGN: dim: 1 size is aligned to: 32
func.func @test_transpose_size_align_with_smallest_element_in_kernel(%arg0: tensor<1200x2xi8>) -> tensor<2x1200xi8> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
  %0 = tensor.empty() : tensor<1200x2xf16>
  %1 = tensor.empty() : tensor<2x1200xf16>
  %2 = tensor.empty() : tensor<2x1200xi8>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<1200x2xi8>) outs(%0 : tensor<1200x2xf16>) -> tensor<1200x2xf16>
  %transposed = linalg.transpose ins(%3 : tensor<1200x2xf16>) outs(%1 : tensor<2x1200xf16>) permutation = [1, 0] 
  %4 = hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%transposed : tensor<2x1200xf16>) outs(%2 : tensor<2x1200xi8>) -> tensor<2x1200xi8>
  return %4 : tensor<2x1200xi8>
}

// -----

// CHECK-ALIGN: dim: 2 size is aligned to: 32
// CHECK-ALIGN: dim: 4 size is aligned to: 32
func.func @test_size_align_with_interchange(%arg0: tensor<3x4x5x6x7xi8>) -> tensor<5x4x7x6x3xi8> 
attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PBR>} {
  %1 = tensor.empty() : tensor<7x4x5x6x3xi8>
  %transposed = linalg.transpose ins(%arg0 : tensor<3x4x5x6x7xi8>) outs(%1 : tensor<7x4x5x6x3xi8>) permutation = [4, 1, 2, 3, 0] 
  %2 = tensor.empty() : tensor<5x4x7x6x3xi8>
  %transposed_0 = linalg.transpose ins(%transposed : tensor<7x4x5x6x3xi8>) outs(%2 : tensor<5x4x7x6x3xi8>) permutation = [2, 1, 0, 3, 4] 
  return %transposed_0 : tensor<5x4x7x6x3xi8>
}