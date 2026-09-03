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
