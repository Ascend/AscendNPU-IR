// RUN: bishengir-opt -hivm-decompose-op %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// Decompose Conv3d is memref-only, init-rank2-only, and no-bias-only.

// CHECK-LABEL: func.func @decompose_conv3d_rank2_memref_g1_p0
// CHECK-NOT: hivm.hir.Conv3dL1
// CHECK: memref.subview %{{.*}}[0, 0] [64, 96]
// CHECK: %[[INPUT_2D:.*]] = memref.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3], [4], [5]] : memref<1x6x2x10x13x16xf16
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = 0 : i32}
// CHECK-SAME: ins(%[[INPUT_2D]]
// CHECK-SAME: outs(%{{.*}} : memref<64x96xf16
// CHECK: scf.for
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = 0 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<64x96xf16
// CHECK: memref.subview %{{.*}}[0, 96] [64, 96]
// CHECK: memref.subview %{{.*}}[1, 0, 0, 0, 0, 0] [1, 6, 2, 10, 13, 16]
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = 0 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<64x96xf16
// CHECK: return
func.func @decompose_conv3d_rank2_memref_g1_p0(
    %input: memref<2x8x2x10x13x16xf16>,
    %weight: memref<3x2x4x5x11x16xf16>,
    %init: memref<64x192xf16>) {
  %true = arith.constant true
  hivm.hir.Conv3dL1 {groups = 1 : i32, padding = 0 : i32}
      ins(%input, %weight, %true : memref<2x8x2x10x13x16xf16>, memref<3x2x4x5x11x16xf16>, i1)
      outs(%init : memref<64x192xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @decompose_conv3d_rank2_memref_g1_p1_depth_padded
// CHECK-NOT: hivm.hir.Conv3dL1
// CHECK: memref.subview %{{.*}}[0, 0] [112, 128]
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = 1 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<112x128xf16
// CHECK: scf.for
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = 1 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<112x128xf16
// CHECK: memref.subview %{{.*}}[0, 128] [112, 128]
// CHECK: memref.subview %{{.*}}[1, 0, 0, 0, 0, 0] [1, 8, 2, 10, 13, 16]
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = 1 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<112x128xf16
// CHECK: return
func.func @decompose_conv3d_rank2_memref_g1_p1_depth_padded(
    %input: memref<2x10x2x10x13x16xf16>,
    %weight: memref<3x2x4x5x11x16xf16>,
    %init: memref<112x256xf16>) {
  %true = arith.constant true
  hivm.hir.Conv3dL1 {conv3dDepthPadded, groups = 1 : i32, padding = 1 : i32}
      ins(%input, %weight, %true : memref<2x10x2x10x13x16xf16>, memref<3x2x4x5x11x16xf16>, i1)
      outs(%init : memref<112x256xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @decompose_conv3d_rank2_memref_dhw_padding
// CHECK-NOT: hivm.hir.Conv3dL1
// CHECK: memref.subview %{{.*}}[0, 0] [176, 128]
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = [2, 3]}
// CHECK-SAME: outs(%{{.*}} : memref<176x128xf16
// CHECK: scf.for
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = [2, 3]}
// CHECK-SAME: outs(%{{.*}} : memref<176x128xf16
// CHECK: memref.subview %{{.*}}[0, 128] [176, 128]
// CHECK: hivm.hir.Conv2dL1 {groups = 1 : i32, padding = [2, 3]}
// CHECK-SAME: outs(%{{.*}} : memref<176x128xf16
// CHECK: return
func.func @decompose_conv3d_rank2_memref_dhw_padding(
    %input: memref<2x10x2x10x13x16xf16>,
    %weight: memref<3x2x4x5x11x16xf16>,
    %init: memref<176x256xf16>) {
  %true = arith.constant true
  hivm.hir.Conv3dL1 {conv3dDepthPadded, groups = 1 : i32, padding = [1, 2, 3]}
      ins(%input, %weight, %true : memref<2x10x2x10x13x16xf16>, memref<3x2x4x5x11x16xf16>, i1)
      outs(%init : memref<176x256xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @decompose_conv3d_rank2_memref_g2_p0
// CHECK-NOT: hivm.hir.Conv3dL1
// CHECK: memref.subview %{{.*}}[0, 0] [64, 192]
// CHECK: hivm.hir.Conv2dL1 {groups = 2 : i32, padding = 0 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<64x192xf32
// CHECK: scf.for
// CHECK: hivm.hir.Conv2dL1 {groups = 2 : i32, padding = 0 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<64x192xf32
// CHECK: memref.subview %{{.*}}[0, 192] [64, 192]
// CHECK: memref.subview %{{.*}}[1, 0, 0, 0, 0, 0] [1, 6, 4, 10, 13, 8]
// CHECK: hivm.hir.Conv2dL1 {groups = 2 : i32, padding = 0 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<64x192xf32
// CHECK: return
func.func @decompose_conv3d_rank2_memref_g2_p0(
    %input: memref<2x8x4x10x13x8xf32>,
    %weight: memref<3x2x4x5x12x8xf32>,
    %init: memref<64x384xf32>) {
  %true = arith.constant true
  hivm.hir.Conv3dL1 {groups = 2 : i32, padding = 0 : i32}
      ins(%input, %weight, %true : memref<2x8x4x10x13x8xf32>, memref<3x2x4x5x12x8xf32>, i1)
      outs(%init : memref<64x384xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @decompose_conv3d_rank2_memref_g2_p1_depth_padded
// CHECK-NOT: hivm.hir.Conv3dL1
// CHECK: memref.subview %{{.*}}[0, 0] [112, 256]
// CHECK: hivm.hir.Conv2dL1 {groups = 2 : i32, padding = 1 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<112x256xf32
// CHECK: scf.for
// CHECK: hivm.hir.Conv2dL1 {groups = 2 : i32, padding = 1 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<112x256xf32
// CHECK: memref.subview %{{.*}}[0, 256] [112, 256]
// CHECK: memref.subview %{{.*}}[1, 0, 0, 0, 0, 0] [1, 8, 4, 10, 13, 8]
// CHECK: hivm.hir.Conv2dL1 {groups = 2 : i32, padding = 1 : i32}
// CHECK-SAME: outs(%{{.*}} : memref<112x256xf32
// CHECK: return
func.func @decompose_conv3d_rank2_memref_g2_p1_depth_padded(
    %input: memref<2x10x4x10x13x8xf32>,
    %weight: memref<3x2x4x5x12x8xf32>,
    %init: memref<112x512xf32>) {
  %true = arith.constant true
  hivm.hir.Conv3dL1 {conv3dDepthPadded, groups = 2 : i32, padding = 1 : i32}
      ins(%input, %weight, %true : memref<2x10x4x10x13x8xf32>, memref<3x2x4x5x12x8xf32>, i1)
      outs(%init : memref<112x512xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @keep_conv3d_rank2_memref_bias_unchanged
// CHECK: hivm.hir.Conv3dL1 {groups = 1 : i32, padding = 0 : i32}
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<2x8x2x10x13x16xf16>, memref<3x2x4x5x11x16xf16>, i1, memref<11xf16>)
// CHECK: return
func.func @keep_conv3d_rank2_memref_bias_unchanged(
    %input: memref<2x8x2x10x13x16xf16>,
    %weight: memref<3x2x4x5x11x16xf16>,
    %bias: memref<11xf16>,
    %init: memref<64x192xf16>) {
  %true = arith.constant true
  hivm.hir.Conv3dL1 {groups = 1 : i32, padding = 0 : i32}
      ins(%input, %weight, %true, %bias : memref<2x8x2x10x13x16xf16>, memref<3x2x4x5x11x16xf16>, i1, memref<11xf16>)
      outs(%init : memref<64x192xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @keep_conv3d_rank5_memref_unchanged
// CHECK: hivm.hir.Conv3dL1 {groups = 1 : i32, padding = 0 : i32}
// CHECK: return
func.func @keep_conv3d_rank5_memref_unchanged(
    %input: memref<2x8x2x10x13x16xf16>,
    %weight: memref<3x2x4x5x11x16xf16>,
    %bias: memref<11xf16>,
    %init: memref<2x6x11x7x9xf16>) {
  %true = arith.constant true
  hivm.hir.Conv3dL1 {groups = 1 : i32, padding = 0 : i32}
      ins(%input, %weight, %true, %bias : memref<2x8x2x10x13x16xf16>, memref<3x2x4x5x11x16xf16>, i1, memref<11xf16>)
      outs(%init : memref<2x6x11x7x9xf16>)
  return
}

// -----

// CHECK-LABEL: func.func @keep_tensor_conv3d_unchanged
// CHECK: %[[C:.*]] = hivm.hir.Conv3dL1 {groups = 1 : i32, padding = 0 : i32}
// CHECK: return %[[C]] : tensor<2x6x11x7x9xf16>
func.func @keep_tensor_conv3d_unchanged(
    %input: tensor<2x8x2x10x13x16xf16>,
    %weight: tensor<3x2x4x5x11x16xf16>,
    %bias: tensor<11xf16>,
    %init: tensor<2x6x11x7x9xf16>) -> tensor<2x6x11x7x9xf16> {
  %true = arith.constant true
  %0 = hivm.hir.Conv3dL1 {groups = 1 : i32, padding = 0 : i32}
      ins(%input, %weight, %true, %bias : tensor<2x8x2x10x13x16xf16>, tensor<3x2x4x5x11x16xf16>, i1, tensor<11xf16>)
      outs(%init : tensor<2x6x11x7x9xf16>) -> tensor<2x6x11x7x9xf16>
  return %0 : tensor<2x6x11x7x9xf16>
}
