// RUN: bishengir-opt %s --split-input-file --hfusion-inline-brc \
// RUN:   --hfusion-pre-vectorization-fusion --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hfusion-inline-brc \
// RUN:   --hivm-vectorize-ops --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Diff: broadcast source rank mismatch
// Linalg represents an inserted leading axis by changing rank. HIVM keeps the
// destination rank and uses a unit source dimension.

// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_leading(
// CHECK-HFUSION: %[[BROADCAST:.*]] = vector.transfer_read {{.*}} : tensor<8xf32>, vector<8x8xf32>
// CHECK-HFUSION: vector.transfer_write %[[BROADCAST]]
func.func @hfusion_broadcast_leading(%input: tensor<8xf32>)
    -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = linalg.broadcast ins(%input : tensor<8xf32>)
      outs(%empty : tensor<8x8xf32>) dimensions = [0]
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_broadcast_leading(
// CHECK-HIVM: %[[SOURCE:.*]] = vector.transfer_read {{.*}} : tensor<1x8xf32>, vector<1x8xf32>
// CHECK-HIVM: %[[BROADCAST:.*]] = vector.broadcast %[[SOURCE]] : vector<1x8xf32> to vector<8x8xf32>
// CHECK-HIVM: vector.transfer_write %[[BROADCAST]]
func.func @hivm_broadcast_leading(%input: tensor<1x8xf32>)
    -> tensor<8x8xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vbrc ins(%input : tensor<1x8xf32>)
      outs(%empty : tensor<8x8xf32>) broadcast_dims = [0]
      -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// -----

// Diff: broadcast source rank mismatch
// Trailing-axis broadcast.

// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_trailing(
// CHECK-HFUSION: %[[BROADCAST:.*]] = vector.transfer_read {{.*}} : tensor<8xf32>, vector<8x8xf32>
// CHECK-HFUSION: vector.transfer_write %[[BROADCAST]]
func.func @hfusion_broadcast_trailing(%input: tensor<8xf32>)
    -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = linalg.broadcast ins(%input : tensor<8xf32>)
      outs(%empty : tensor<8x8xf32>) dimensions = [1]
  return %result : tensor<8x8xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_broadcast_trailing(
// CHECK-HIVM: %[[SOURCE:.*]] = vector.transfer_read {{.*}} : tensor<8x1xf32>, vector<8x1xf32>
// CHECK-HIVM: %[[BROADCAST:.*]] = vector.broadcast %[[SOURCE]] : vector<8x1xf32> to vector<8x8xf32>
// CHECK-HIVM: vector.transfer_write %[[BROADCAST]]
func.func @hivm_broadcast_trailing(%input: tensor<8x1xf32>)
    -> tensor<8x8xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vbrc ins(%input : tensor<8x1xf32>)
      outs(%empty : tensor<8x8xf32>) broadcast_dims = [1]
      -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// -----

// Diff: broadcast source rank mismatch
// Multiple inserted axes use the same output lane mapping.

// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_multi(
// CHECK-HFUSION: %[[BROADCAST:.*]] = vector.transfer_read {{.*}} : tensor<8xf32>, vector<2x8x4xf32>
// CHECK-HFUSION: vector.transfer_write %[[BROADCAST]]
func.func @hfusion_broadcast_multi(%input: tensor<8xf32>)
    -> tensor<2x8x4xf32> {
  %empty = tensor.empty() : tensor<2x8x4xf32>
  %result = linalg.broadcast ins(%input : tensor<8xf32>)
      outs(%empty : tensor<2x8x4xf32>) dimensions = [0, 2]
  return %result : tensor<2x8x4xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_broadcast_multi(
// CHECK-HIVM: %[[SOURCE:.*]] = vector.transfer_read {{.*}} : tensor<1x8x1xf32>, vector<1x8x1xf32>
// CHECK-HIVM: %[[BROADCAST:.*]] = vector.broadcast %[[SOURCE]] : vector<1x8x1xf32> to vector<2x8x4xf32>
// CHECK-HIVM: vector.transfer_write %[[BROADCAST]]
func.func @hivm_broadcast_multi(%input: tensor<1x8x1xf32>)
    -> tensor<2x8x4xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<2x8x4xf32>
  %result = hivm.hir.vbrc ins(%input : tensor<1x8x1xf32>)
      outs(%empty : tensor<2x8x4xf32>) broadcast_dims = [0, 2]
      -> tensor<2x8x4xf32>
  return %result : tensor<2x8x4xf32>
}

// -----

// Diff: broadcast scalar representation
// The common preprocessing pass inlines the Linalg broadcast into the add.
// Linalg retains its rank-0 tensor operand while HIVM starts from the
// equivalent scalar operand.

// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_inline(
// CHECK-HFUSION-DAG: %[[SPLAT:.*]] = vector.transfer_read {{.*}} : tensor<f32>, vector<64xf32>
// CHECK-HFUSION-DAG: %[[INPUT:.*]] = vector.transfer_read {{.*}} : tensor<64xf32>, vector<64xf32>
// CHECK-HFUSION: %[[RESULT:.*]] = arith.addf %[[INPUT]], %[[SPLAT]] : vector<64xf32>
// CHECK-HFUSION: vector.transfer_write %[[RESULT]]
// CHECK-HIVM-LABEL: func.func @hivm_broadcast_inline(
// CHECK-HIVM-DAG: %[[SPLAT:.*]] = vector.broadcast {{.*}} : f32 to vector<64xf32>
// CHECK-HIVM-DAG: %[[INPUT:.*]] = vector.transfer_read {{.*}} : tensor<64xf32>, vector<64xf32>
// CHECK-HIVM: %[[RESULT:.*]] = arith.addf %[[INPUT]], %[[SPLAT]] : vector<64xf32>
// CHECK-HIVM: vector.transfer_write %[[RESULT]]
func.func @hfusion_broadcast_inline(%input: tensor<64xf32>,
    %scalar: tensor<f32>) -> tensor<64xf32> {
  %empty = tensor.empty() : tensor<64xf32>
  %broadcast = linalg.broadcast ins(%scalar : tensor<f32>)
      outs(%empty : tensor<64xf32>) dimensions = [0]
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
      ins(%input, %broadcast : tensor<64xf32>, tensor<64xf32>)
      outs(%empty : tensor<64xf32>) -> tensor<64xf32>
  return %result : tensor<64xf32>
}

func.func @hivm_broadcast_inline(%input: tensor<64xf32>, %scalar: f32)
    -> tensor<64xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<64xf32>
  %result = hivm.hir.vadd ins(%input, %scalar : tensor<64xf32>, f32)
      outs(%empty : tensor<64xf32>) -> tensor<64xf32>
  return %result : tensor<64xf32>
}
