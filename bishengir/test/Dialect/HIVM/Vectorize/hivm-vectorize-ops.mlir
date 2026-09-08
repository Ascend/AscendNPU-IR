// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops --cse --canonicalize | FileCheck %s

// Vector sizes follow the VL packing policy (256 bytes):
//   capacity = VL / sizeof(elem)  (f32 -> 64, f16 -> 128)
// Shapes must already fit that capacity (as after tiling into a VF).

// Exact f32 tile: 4x16 = 64 lanes -> vector<4x16xf32>
// CHECK-LABEL: func.func @vadd_4x16(
// CHECK-NOT: hivm.hir.vadd
// CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[PAD]] {{.*}} : tensor<4x16xf32>, vector<4x16xf32>
// CHECK: %[[B:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[PAD]] {{.*}} : tensor<4x16xf32>, vector<4x16xf32>
// CHECK: %[[R:.*]] = arith.addf %[[A]], %[[B]] : vector<4x16xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<4x16xf32>, tensor<4x16xf32>
func.func @vadd_4x16(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>) -> tensor<4x16xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<4x16xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%0 : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}

// -----

// Leading dim expanded to fill remaining VL capacity: 2x16 -> vector<4x16xf32> with mask.
// CHECK-LABEL: func.func @vadd_2x16_expand_leading(
// CHECK-NOT: hivm.hir.vadd
// CHECK: %[[MASK:.*]] = vector.constant_mask [2, 16] : vector<4x16xi1>
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}}{{.*}}, %[[MASK]] {{.*}} : tensor<2x16xf32>, vector<4x16xf32>
// CHECK: %[[B:.*]] = vector.transfer_read %{{.*}}{{.*}}, %[[MASK]] {{.*}} : tensor<2x16xf32>, vector<4x16xf32>
// CHECK: %[[R:.*]] = arith.addf %[[A]], %[[B]] : vector<4x16xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}}{{.*}}, %[[MASK]] {{.*}} : vector<4x16xf32>, tensor<2x16xf32>
func.func @vadd_2x16_expand_leading(%arg0: tensor<2x16xf32>, %arg1: tensor<2x16xf32>) -> tensor<2x16xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<2x16xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<2x16xf32>, tensor<2x16xf32>) outs(%0 : tensor<2x16xf32>) -> tensor<2x16xf32>
  return %1 : tensor<2x16xf32>
}

// -----

// Unit leading dim + full VL trailing: 1x64 -> vector<1x64xf32>
// CHECK-LABEL: func.func @vadd_1x64(
// CHECK-NOT: hivm.hir.vadd
// CHECK: vector.transfer_read %{{.*}} : tensor<1x64xf32>, vector<1x64xf32>
// CHECK: vector.transfer_read %{{.*}} : tensor<1x64xf32>, vector<1x64xf32>
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<1x64xf32>
// CHECK: vector.transfer_write %{{.*}} : vector<1x64xf32>, tensor<1x64xf32>
func.func @vadd_1x64(%arg0: tensor<1x64xf32>, %arg1: tensor<1x64xf32>) -> tensor<1x64xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<1x64xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) -> tensor<1x64xf32>
  return %1 : tensor<1x64xf32>
}

// -----

// 1-D full VL: 64xf32 -> vector<64xf32>
// CHECK-LABEL: func.func @vadd_64(
// CHECK-NOT: hivm.hir.vadd
// CHECK: vector.transfer_read %{{.*}} : tensor<64xf32>, vector<64xf32>
// CHECK: vector.transfer_read %{{.*}} : tensor<64xf32>, vector<64xf32>
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: vector.transfer_write %{{.*}} : vector<64xf32>, tensor<64xf32>
func.func @vadd_64(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<64xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @vsub_2x32(
// CHECK-NOT: hivm.hir.vsub
// CHECK: arith.subf %{{.*}}, %{{.*}} : vector<2x32xf32>
// CHECK: vector.transfer_write %{{.*}} : vector<2x32xf32>, tensor<2x32xf32>
func.func @vsub_2x32(%arg0: tensor<2x32xf32>, %arg1: tensor<2x32xf32>) -> tensor<2x32xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<2x32xf32>
  %1 = hivm.hir.vsub ins(%arg0, %arg1 : tensor<2x32xf32>, tensor<2x32xf32>) outs(%0 : tensor<2x32xf32>) -> tensor<2x32xf32>
  return %1 : tensor<2x32xf32>
}

// -----

// f16 capacity is 128 lanes: 4x32 = 128 -> vector<4x32xf16>
// CHECK-LABEL: func.func @vmul_f16_4x32(
// CHECK-NOT: hivm.hir.vmul
// CHECK-DAG: %[[PAD:.*]] = arith.constant 1.000000e+00 : f16
// CHECK: vector.transfer_read %{{.*}}, %[[PAD]] {{.*}} : tensor<4x32xf16>, vector<4x32xf16>
// CHECK: arith.mulf %{{.*}}, %{{.*}} : vector<4x32xf16>
// CHECK: vector.transfer_write %{{.*}} : vector<4x32xf16>, tensor<4x32xf16>
func.func @vmul_f16_4x32(%arg0: tensor<4x32xf16>, %arg1: tensor<4x32xf16>) -> tensor<4x32xf16>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<4x32xf16>
  %1 = hivm.hir.vmul ins(%arg0, %arg1 : tensor<4x32xf16>, tensor<4x32xf16>) outs(%0 : tensor<4x32xf16>) -> tensor<4x32xf16>
  return %1 : tensor<4x32xf16>
}

// -----

// CHECK-LABEL: func.func @vdiv_1x64(
// CHECK-NOT: hivm.hir.vdiv
// CHECK-DAG: %[[PAD:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: vector.transfer_read %{{.*}}, %[[PAD]] {{.*}} : tensor<1x64xf32>, vector<1x64xf32>
// CHECK: arith.divf %{{.*}}, %{{.*}} : vector<1x64xf32>
func.func @vdiv_1x64(%arg0: tensor<1x64xf32>, %arg1: tensor<1x64xf32>) -> tensor<1x64xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<1x64xf32>
  %1 = hivm.hir.vdiv ins(%arg0, %arg1 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%0 : tensor<1x64xf32>) -> tensor<1x64xf32>
  return %1 : tensor<1x64xf32>
}

// -----

// CHECK-LABEL: func.func @vmax_8x8(
// CHECK-NOT: hivm.hir.vmax
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : vector<8x8xf32>
func.func @vmax_8x8(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vmax ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vmin_8x8(
// CHECK-NOT: hivm.hir.vmin
// CHECK: arith.minimumf %{{.*}}, %{{.*}} : vector<8x8xf32>
func.func @vmin_8x8(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vmin ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vabs_8x8(
// CHECK-NOT: hivm.hir.vabs
// CHECK: math.absf %{{.*}} : vector<8x8xf32>
// CHECK: vector.transfer_write %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vabs_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vabs ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// Last-axis reduction on a VL-fitting tile: 4x16 -> multi_reduction
// CHECK-LABEL: func.func @vreduce_sum_4x16(
// CHECK-NOT: hivm.hir.vreduce
// CHECK: %[[SRC:.*]] = vector.transfer_read %{{.*}} : tensor<4x16xf32>, vector<4x16xf32>
// CHECK: %[[INIT:.*]] = vector.transfer_read %{{.*}} : tensor<4x1xf32>, vector<4x1xf32>
// CHECK: %[[ACC:.*]] = vector.shape_cast %[[INIT]] : vector<4x1xf32> to vector<4xf32>
// CHECK: %[[RED:.*]] = vector.multi_reduction <add>, %[[SRC]], %[[ACC]] [1] : vector<4x16xf32> to vector<4xf32>
// CHECK: %[[OUT:.*]] = vector.shape_cast %[[RED]] : vector<4xf32> to vector<4x1xf32>
// CHECK: vector.transfer_write %[[OUT]], %{{.*}} : vector<4x1xf32>, tensor<4x1xf32>
func.func @vreduce_sum_4x16(%arg0: tensor<4x16xf32>, %init: tensor<4x1xf32>) -> tensor<4x1xf32>
    attributes {hivm.vector_function} {
  %0 = hivm.hir.vreduce <sum> ins(%arg0 : tensor<4x16xf32>) outs(%init : tensor<4x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: func.func @vreduce_max_2x32(
// CHECK-NOT: hivm.hir.vreduce
// CHECK: vector.multi_reduction <maximumf>, %{{.*}}, %{{.*}} [1] : vector<2x32xf32> to vector<2xf32>
func.func @vreduce_max_2x32(%arg0: tensor<2x32xf32>, %init: tensor<2x1xf32>) -> tensor<2x1xf32>
    attributes {hivm.vector_function} {
  %0 = hivm.hir.vreduce <max> ins(%arg0 : tensor<2x32xf32>) outs(%init : tensor<2x1xf32>) unsigned_src = false reduce_dims = [1] -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// -----

// CHECK-LABEL: func.func @vexp_8x8(
// CHECK-NOT: hivm.hir.vexp
// CHECK: math.exp %{{.*}} : vector<8x8xf32>
func.func @vexp_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vexp ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vln_8x8(
// CHECK-NOT: hivm.hir.vln
// CHECK: math.log %{{.*}} : vector<8x8xf32>
func.func @vln_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vln ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vsqrt_8x8(
// CHECK-NOT: hivm.hir.vsqrt
// CHECK: math.sqrt %{{.*}} : vector<8x8xf32>
func.func @vsqrt_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vsqrt ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vrsqrt_8x8(
// CHECK-NOT: hivm.hir.vrsqrt
// CHECK: math.rsqrt %{{.*}} : vector<8x8xf32>
func.func @vrsqrt_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vrsqrt ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vrec_8x8(
// CHECK-NOT: hivm.hir.vrec
// CHECK-DAG: %[[ONE:.*]] = arith.constant dense<1.000000e+00> : vector<8x8xf32>
// CHECK: arith.divf %[[ONE]], %{{.*}} : vector<8x8xf32>
func.func @vrec_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vrec ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vrelu_8x8(
// CHECK-NOT: hivm.hir.vrelu
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : vector<8x8xf32>
// CHECK: arith.maximumf %{{.*}}, %[[ZERO]] : vector<8x8xf32>
func.func @vrelu_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vrelu ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vnot_8x8(
// CHECK-NOT: hivm.hir.vnot
// CHECK-DAG: %[[ONES:.*]] = arith.constant dense<-1> : vector<8x8xi32>
// CHECK: arith.xori %{{.*}}, %[[ONES]] : vector<8x8xi32>
func.func @vnot_8x8(%arg0: tensor<8x8xi32>) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vnot ins(%arg0 : tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @vand_8x8(
// CHECK-NOT: hivm.hir.vand
// CHECK: arith.andi %{{.*}}, %{{.*}} : vector<8x8xi32>
func.func @vand_8x8(%arg0: tensor<8x8xi32>, %arg1: tensor<8x8xi32>) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vand ins(%arg0, %arg1 : tensor<8x8xi32>, tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @vor_8x8(
// CHECK-NOT: hivm.hir.vor
// CHECK: arith.ori %{{.*}}, %{{.*}} : vector<8x8xi32>
func.func @vor_8x8(%arg0: tensor<8x8xi32>, %arg1: tensor<8x8xi32>) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vor ins(%arg0, %arg1 : tensor<8x8xi32>, tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @vxor_8x8(
// CHECK-NOT: hivm.hir.vxor
// CHECK: arith.xori %{{.*}}, %{{.*}} : vector<8x8xi32>
func.func @vxor_8x8(%arg0: tensor<8x8xi32>, %arg1: tensor<8x8xi32>) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vxor ins(%arg0, %arg1 : tensor<8x8xi32>, tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// Single dynamic dim: assign the full f32 VL (64) and mask to the runtime extent.
// CHECK-LABEL: func.func @vadd_dynamic(
// CHECK-NOT: hivm.hir.vadd
// CHECK: vector.transfer_read %{{.*}} : tensor<?xf32>, vector<64xf32>
// CHECK: arith.addf %{{.*}}, %{{.*}} : vector<64xf32>
// CHECK: vector.transfer_write %{{.*}} : vector<64xf32>, tensor<?xf32>
func.func @vadd_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32>
    attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// Non-VF functions must be left unchanged.
// CHECK-LABEL: func.func @skip_non_vf(
// CHECK: hivm.hir.vadd
// CHECK-NOT: vector.transfer_read
func.func @skip_non_vf(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %0 = tensor.empty() : tensor<4x16xf32>
  %1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<4x16xf32>) outs(%0 : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}

// -----

// Ops that opted out through NotVectorizableTrait are skipped gracefully:
// the op stays, the pass does not fail.
// CHECK-LABEL: func.func @skip_opted_out_vlog2(
// CHECK: hivm.hir.vlog2
// CHECK-NOT: vector.transfer_read
func.func @skip_opted_out_vlog2(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vlog2 ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// A scalar-source vbrc vectorizes through a splat: vector.broadcast takes the
// scalar directly to the target vector shape.
// CHECK-LABEL: func.func @vbrc_scalar_1x64(
// CHECK-NOT: hivm.hir.vbrc
// CHECK: %[[SPLAT:.*]] = vector.broadcast %{{.*}} : f32 to vector<1x64xf32>
// CHECK: vector.transfer_write %[[SPLAT]], %{{.*}} : vector<1x64xf32>, tensor<1x64xf32>
func.func @vbrc_scalar_1x64(%arg0: f32) -> tensor<1x64xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<1x64xf32>
  %1 = hivm.hir.vbrc ins(%arg0 : f32) outs(%0 : tensor<1x64xf32>) -> tensor<1x64xf32>
  return %1 : tensor<1x64xf32>
}

// -----

// A scalar source whose target exceeds one register still vectorizes: the
// splat covers the one-register vector and the masked write spans the rest.
// CHECK-LABEL: func.func @vbrc_scalar_8x8(
// CHECK-NOT: hivm.hir.vbrc
// CHECK: %[[SPLAT:.*]] = vector.broadcast %{{.*}} : f32 to vector<8x8xf32>
// CHECK: vector.transfer_write %[[SPLAT]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vbrc_scalar_8x8(%arg0: f32) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vbrc ins(%arg0 : f32) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// A transpose reads its source through the permuted map: the vector lands in
// the destination orientation with no vector.transpose in between.
// CHECK-LABEL: func.func @vtranspose_64x1(
// CHECK-NOT: hivm.hir.vtranspose
// CHECK: %[[READ:.*]] = vector.transfer_read %{{.*}} : tensor<64x1xf32>, vector<1x64xf32>
// CHECK: vector.transfer_write %[[READ]], %{{.*}} : vector<1x64xf32>, tensor<1x64xf32>
func.func @vtranspose_64x1(%arg0: tensor<64x1xf32>) -> tensor<1x64xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<1x64xf32>
  %1 = hivm.hir.vtranspose ins(%arg0 : tensor<64x1xf32>) outs(%0 : tensor<1x64xf32>) permutation = [1, 0] -> tensor<1x64xf32>
  return %1 : tensor<1x64xf32>
}

// -----

// A shaped-source vbrc whose target fits one register vectorizes: the source
// reads at its own unit shape and vector.broadcast stretches it.
// CHECK-LABEL: func.func @vbrc_shaped_1x64(
// CHECK-NOT: hivm.hir.vbrc
// CHECK: %[[SRC:.*]] = vector.transfer_read %{{.*}} : tensor<1x1xf32>, vector<1x1xf32>
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[SRC]] : vector<1x1xf32> to vector<1x64xf32>
// CHECK: vector.transfer_write %[[BCAST]], %{{.*}} : vector<1x64xf32>, tensor<1x64xf32>
func.func @vbrc_shaped_1x64(%arg0: tensor<1x1xf32>) -> tensor<1x64xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<1x64xf32>
  %1 = hivm.hir.vbrc ins(%arg0 : tensor<1x1xf32>) outs(%0 : tensor<1x64xf32>) broadcast_dims = [1] -> tensor<1x64xf32>
  return %1 : tensor<1x64xf32>
}

// Elementwise math unary ops: one math op between the transfers, matching
// what AutoVectorizeV2 emits for the equivalent linalg math regions.

// CHECK-LABEL: func.func @vtanh_8x8(
// CHECK-NOT: hivm.hir.vtanh
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = math.tanh %[[A]] : vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vtanh_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vtanh ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vsin_8x8(
// CHECK-NOT: hivm.hir.vsin
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = math.sin %[[A]] : vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vsin_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vsin ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vcos_8x8(
// CHECK-NOT: hivm.hir.vcos
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = math.cos %[[A]] : vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vcos_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vcos ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @verf_8x8(
// CHECK-NOT: hivm.hir.verf
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = math.erf %[[A]] : vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @verf_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.verf ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// vcast: round_mode/cast attributes map to the same arith cast ops that V2
// emits for the equivalent convertScalarToDtype regions.

// Float narrowing with rint semantics: arith.truncf.
// CHECK-LABEL: func.func @vcast_f32_to_f16_8x8(
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = arith.truncf %[[A]] : vector<8x8xf32> to vector<8x8xf16>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf16>, tensor<8x8xf16>
func.func @vcast_f32_to_f16_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xf16>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf16>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf16>) round_mode = <rint> cast = <cast_signed> -> tensor<8x8xf16>
  return %1 : tensor<8x8xf16>
}

// -----

// Float widening with rint semantics: arith.extf.
// CHECK-LABEL: func.func @vcast_f16_to_f32_8x8(
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf16>, vector<8x8xf16>
// CHECK: %[[R:.*]] = arith.extf %[[A]] : vector<8x8xf16> to vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vcast_f16_to_f32_8x8(%arg0: tensor<8x8xf16>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8x8xf16>) outs(%0 : tensor<8x8xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// Float to signed integer with trunc semantics: arith.fptosi.
// CHECK-LABEL: func.func @vcast_f32_to_i32_8x8(
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = arith.fptosi %[[A]] : vector<8x8xf32> to vector<8x8xi32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xi32>, tensor<8x8xi32>
func.func @vcast_f32_to_i32_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xi32>) round_mode = <trunc> cast = <cast_signed> -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// Float to narrow integer with trunc semantics: arith.fptosi.
// CHECK-LABEL: func.func @vcast_f16_to_i8_8x16(
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x16xf16>, vector<8x16xf16>
// CHECK: %[[R:.*]] = arith.fptosi %[[A]] : vector<8x16xf16> to vector<8x16xi8>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x16xi8>, tensor<8x16xi8>
func.func @vcast_f16_to_i8_8x16(%arg0: tensor<8x16xf16>) -> tensor<8x16xi8>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x16xi8>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8x16xf16>) outs(%0 : tensor<8x16xi8>) round_mode = <trunc> cast = <cast_signed> -> tensor<8x16xi8>
  return %1 : tensor<8x16xi8>
}

// -----

// Signed integer to float with rint semantics: arith.sitofp.
// CHECK-LABEL: func.func @vcast_i32_to_f32_8x8(
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xi32>, vector<8x8xi32>
// CHECK: %[[R:.*]] = arith.sitofp %[[A]] : vector<8x8xi32> to vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vcast_i32_to_f32_8x8(%arg0: tensor<8x8xi32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8x8xi32>) outs(%0 : tensor<8x8xf32>) round_mode = <rint> cast = <cast_signed> -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// Integer widening with rint semantics: arith.extsi.
// CHECK-LABEL: func.func @vcast_i8_to_i16_8x16(
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x16xi8>, vector<8x16xi8>
// CHECK: %[[R:.*]] = arith.extsi %[[A]] : vector<8x16xi8> to vector<8x16xi16>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x16xi16>, tensor<8x16xi16>
func.func @vcast_i8_to_i16_8x16(%arg0: tensor<8x16xi8>) -> tensor<8x16xi16>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x16xi16>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8x16xi8>) outs(%0 : tensor<8x16xi16>) round_mode = <rint> cast = <cast_signed> -> tensor<8x16xi16>
  return %1 : tensor<8x16xi16>
}

// -----

// Same-width bit reinterpretation: arith.bitcast.
// CHECK-LABEL: func.func @vcast_bitcast_f32_to_i32_8x8(
// CHECK-NOT: hivm.hir.vcast
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = arith.bitcast %[[A]] : vector<8x8xf32> to vector<8x8xi32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xi32>, tensor<8x8xi32>
func.func @vcast_bitcast_f32_to_i32_8x8(%arg0: tensor<8x8xf32>) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vcast ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xi32>) round_mode = <rint> cast = <bitcast> -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// vcmp: predicates follow HFusion's buildCompareFn, which is what V2's
// linalg regions carry.

// Float less-than to an i1 result: arith.cmpf olt.
// CHECK-LABEL: func.func @vcmp_f32_lt_8x8(
// CHECK-NOT: hivm.hir.vcmp
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[B:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = arith.cmpf olt, %[[A]], %[[B]] : vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xi1>, tensor<8x8xi1>
func.func @vcmp_f32_lt_8x8(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xi1>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi1>
  %1 = hivm.hir.vcmp ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xi1>) compare_mode = <lt> -> tensor<8x8xi1>
  return %1 : tensor<8x8xi1>
}

// -----

// Float not-equal lowers to `une`, mirroring HFusion's vne (V2's region).
// CHECK-LABEL: func.func @vcmp_f32_ne_8x8(
// CHECK-NOT: hivm.hir.vcmp
// CHECK: arith.cmpf une, %{{.*}}, %{{.*}} : vector<8x8xf32>
// CHECK: vector.transfer_write %{{.*}} : vector<8x8xi1>, tensor<8x8xi1>
func.func @vcmp_f32_ne_8x8(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xi1>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi1>
  %1 = hivm.hir.vcmp ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xi1>) compare_mode = <ne> -> tensor<8x8xi1>
  return %1 : tensor<8x8xi1>
}

// -----

// Signed integer less-than: arith.cmpi slt.
// CHECK-LABEL: func.func @vcmp_i32_slt_8x8(
// CHECK-NOT: hivm.hir.vcmp
// CHECK: arith.cmpi slt, %{{.*}}, %{{.*}} : vector<8x8xi32>
// CHECK: vector.transfer_write %{{.*}} : vector<8x8xi1>, tensor<8x8xi1>
func.func @vcmp_i32_slt_8x8(%arg0: tensor<8x8xi32>, %arg1: tensor<8x8xi32>) -> tensor<8x8xi1>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi1>
  %1 = hivm.hir.vcmp ins(%arg0, %arg1 : tensor<8x8xi32>, tensor<8x8xi32>) outs(%0 : tensor<8x8xi1>) compare_mode = <lt> -> tensor<8x8xi1>
  return %1 : tensor<8x8xi1>
}

// -----

// Unsigned integer less-than: arith.cmpi ult (is_signed : false on signless i32).
// CHECK-LABEL: func.func @vcmp_i32_ult_8x8(
// CHECK-NOT: hivm.hir.vcmp
// CHECK: arith.cmpi ult, %{{.*}}, %{{.*}} : vector<8x8xi32>
// CHECK: vector.transfer_write %{{.*}} : vector<8x8xi1>, tensor<8x8xi1>
func.func @vcmp_i32_ult_8x8(%arg0: tensor<8x8xi32>, %arg1: tensor<8x8xi32>) -> tensor<8x8xi1>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi1>
  %1 = hivm.hir.vcmp ins(%arg0, %arg1 : tensor<8x8xi32>, tensor<8x8xi32>) outs(%0 : tensor<8x8xi1>) compare_mode = <lt> is_signed : false -> tensor<8x8xi1>
  return %1 : tensor<8x8xi1>
}

// -----

// A byte-wide comparison result widens back with arith.extui, as the scalar
// lowering does.
// CHECK-LABEL: func.func @vcmp_f32_lt_i8dst_8x8(
// CHECK-NOT: hivm.hir.vcmp
// CHECK: %[[C:.*]] = arith.cmpf olt, %{{.*}}, %{{.*}} : vector<8x8xf32>
// CHECK: %[[R:.*]] = arith.extui %[[C]] : vector<8x8xi1> to vector<8x8xi8>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xi8>, tensor<8x8xi8>
func.func @vcmp_f32_lt_i8dst_8x8(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xi8>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi8>
  %1 = hivm.hir.vcmp ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xi8>) compare_mode = <lt> -> tensor<8x8xi8>
  return %1 : tensor<8x8xi8>
}

// -----

// An i1 condition drives arith.select directly.
// CHECK-LABEL: func.func @vsel_i1_cond_8x8(
// CHECK-NOT: hivm.hir.vsel
// CHECK: %[[C:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xi1>, vector<8x8xi1>
// CHECK: %[[X:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[Y:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = arith.select %[[C]], %[[X]], %[[Y]] : vector<8x8xi1>, vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vsel_i1_cond_8x8(%cond: tensor<8x8xi1>, %arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vsel ins(%cond, %arg0, %arg1 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// An i8 condition converts to i1 through a zero comparison before select.
// CHECK-LABEL: func.func @vsel_i8_cond_8x8(
// CHECK-NOT: hivm.hir.vsel
// CHECK: %[[C8:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xi8>, vector<8x8xi8>
// CHECK: %[[C:.*]] = arith.cmpi ne, %[[C8]], %{{.*}} : vector<8x8xi8>
// CHECK: arith.select %[[C]], %{{.*}}, %{{.*}} : vector<8x8xi1>, vector<8x8xf32>
func.func @vsel_i8_cond_8x8(%cond: tensor<8x8xi8>, %arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vsel ins(%cond, %arg0, %arg1 : tensor<8x8xi8>, tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// Shifts: the scalar shift amount splats with vector.broadcast, the same
// broadcast V2 applies to scalar region inputs.

// CHECK-LABEL: func.func @vshl_scalar_8x8(
// CHECK-NOT: hivm.hir.vshl
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xi32>, vector<8x8xi32>
// CHECK: %[[S:.*]] = vector.broadcast %{{.*}} : i32 to vector<8x8xi32>
// CHECK: %[[R:.*]] = arith.shli %[[A]], %[[S]] : vector<8x8xi32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xi32>, tensor<8x8xi32>
func.func @vshl_scalar_8x8(%arg0: tensor<8x8xi32>, %shift: i32) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vshl ins(%arg0, %shift : tensor<8x8xi32>, i32) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// Signed shift right: arith.shrsi.
// CHECK-LABEL: func.func @vshr_scalar_signed_8x8(
// CHECK-NOT: hivm.hir.vshr
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xi32>, vector<8x8xi32>
// CHECK: %[[S:.*]] = vector.broadcast %{{.*}} : i32 to vector<8x8xi32>
// CHECK: %[[R:.*]] = arith.shrsi %[[A]], %[[S]] : vector<8x8xi32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xi32>, tensor<8x8xi32>
func.func @vshr_scalar_signed_8x8(%arg0: tensor<8x8xi32>, %shift: i32) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vshr ins(%arg0, %shift : tensor<8x8xi32>, i32) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// Unsigned shift right: arith.shrui (is_signed : false on signless i32).
// CHECK-LABEL: func.func @vshr_scalar_unsigned_8x8(
// CHECK-NOT: hivm.hir.vshr
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xi32>, vector<8x8xi32>
// CHECK: %[[S:.*]] = vector.broadcast %{{.*}} : i32 to vector<8x8xi32>
// CHECK: %[[R:.*]] = arith.shrui %[[A]], %[[S]] : vector<8x8xi32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xi32>, tensor<8x8xi32>
func.func @vshr_scalar_unsigned_8x8(%arg0: tensor<8x8xi32>, %shift: i32) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vshr ins(%arg0, %shift : tensor<8x8xi32>, i32) outs(%0 : tensor<8x8xi32>) is_signed : false -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// CHECK-LABEL: func.func @vpow_f32_8x8(
// CHECK-NOT: hivm.hir.vpow
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[B:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK: %[[R:.*]] = math.powf %[[A]], %[[B]] : vector<8x8xf32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xf32>, tensor<8x8xf32>
func.func @vpow_f32_8x8(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func.func @vpow_i32_8x8(
// CHECK-NOT: hivm.hir.vpow
// CHECK: %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xi32>, vector<8x8xi32>
// CHECK: %[[B:.*]] = vector.transfer_read %{{.*}} : tensor<8x8xi32>, vector<8x8xi32>
// CHECK: %[[R:.*]] = math.ipowi %[[A]], %[[B]] : vector<8x8xi32>
// CHECK: vector.transfer_write %[[R]], %{{.*}} : vector<8x8xi32>, tensor<8x8xi32>
func.func @vpow_i32_8x8(%arg0: tensor<8x8xi32>, %arg1: tensor<8x8xi32>) -> tensor<8x8xi32>
    attributes {hivm.vector_function} {
  %0 = tensor.empty() : tensor<8x8xi32>
  %1 = hivm.hir.vpow ins(%arg0, %arg1 : tensor<8x8xi32>, tensor<8x8xi32>) outs(%0 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}
