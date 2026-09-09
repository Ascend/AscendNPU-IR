// RUN: bishengir-opt %s --split-input-file --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Broadcast then reduce the broadcast axes. The producer vector must feed the
// reduction directly after transfer forwarding.

// Diff: broadcast source rank mismatch
// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_reduce(
// CHECK-HFUSION: %[[INPUT:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[INIT:.*]] = vector.transfer_read
// CHECK-HFUSION-NOT: vector.transfer_write
// CHECK-HFUSION-NOT: vector.transfer_read
// CHECK-HFUSION: %[[REDUCE:.*]] = vector.multi_reduction <add>, %[[INPUT]], %[[INIT]] [0, 2]
// CHECK-HFUSION-SAME: vector<2x8x4xf32> to vector<8xf32>
// CHECK-HFUSION: vector.transfer_write %[[REDUCE]]
func.func @hfusion_broadcast_reduce(%input: tensor<8xf32>,
    %init: tensor<8xf32>) -> tensor<8xf32> {
  %expanded = tensor.empty() : tensor<2x8x4xf32>
  %broadcast = linalg.broadcast ins(%input : tensor<8xf32>)
      outs(%expanded : tensor<2x8x4xf32>) dimensions = [0, 2]
  %result = linalg.reduce {arith.addf}
      ins(%broadcast : tensor<2x8x4xf32>)
      outs(%init : tensor<8xf32>) dimensions = [0, 2]
  return %result : tensor<8xf32>
}

// Diff: reduce shape mismatch
// CHECK-HIVM-LABEL: func.func @hivm_broadcast_reduce(
// CHECK-HIVM: %[[INPUT:.*]] = vector.transfer_read
// CHECK-HIVM: %[[BROADCAST:.*]] = vector.broadcast %[[INPUT]]
// CHECK-HIVM: %[[INIT:.*]] = vector.transfer_read
// CHECK-HIVM: %[[INIT_FLAT:.*]] = vector.shape_cast %[[INIT]]
// CHECK-HIVM-NOT: vector.transfer_write
// CHECK-HIVM-NOT: vector.transfer_read
// CHECK-HIVM: %[[REDUCE:.*]] = vector.multi_reduction <add>, %[[BROADCAST]], %[[INIT_FLAT]] [0, 2]
// CHECK-HIVM-SAME: vector<2x8x4xf32> to vector<8xf32>
// CHECK-HIVM: %[[SHAPED:.*]] = vector.shape_cast %[[REDUCE]]
// CHECK-HIVM: vector.transfer_write %[[SHAPED]]
func.func @hivm_broadcast_reduce(%input: tensor<1x8x1xf32>,
    %init: tensor<1x8x1xf32>) -> tensor<1x8x1xf32>
    attributes {hivm.vector_function} {
  %expanded = tensor.empty() : tensor<2x8x4xf32>
  %broadcast = hivm.hir.vbrc ins(%input : tensor<1x8x1xf32>)
      outs(%expanded : tensor<2x8x4xf32>) broadcast_dims = [0, 2]
      -> tensor<2x8x4xf32>
  %result = hivm.hir.vreduce <sum>
      ins(%broadcast : tensor<2x8x4xf32>)
      outs(%init : tensor<1x8x1xf32>) reduce_dims = [0, 2]
      -> tensor<1x8x1xf32>
  return %result : tensor<1x8x1xf32>
}

// -----

// Reduce then broadcast the retained axis. This covers the opposite
// producer-consumer direction and an output with two broadcast axes.

// Diff: reduce shape mismatch
// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_broadcast(
// CHECK-HFUSION: %[[INPUT:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[INIT:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[REDUCE:.*]] = vector.multi_reduction <maximumf>, %[[INPUT]], %[[INIT]] [0, 2]
// CHECK-HFUSION-NOT: vector.transfer_read
// CHECK-HFUSION: %[[BROADCAST:.*]] = vector.broadcast %[[REDUCE]]
// CHECK-HFUSION: %[[TRANSPOSE:.*]] = vector.transpose %[[BROADCAST]], [0, 2, 1]
// CHECK-HFUSION: vector.transfer_write %[[TRANSPOSE]]
func.func @hfusion_reduce_broadcast(%input: tensor<2x8x4xf32>,
    %init: tensor<8xf32>) -> tensor<2x8x4xf32> {
  %reduced = linalg.reduce {arith.maximumf}
      ins(%input : tensor<2x8x4xf32>)
      outs(%init : tensor<8xf32>) dimensions = [0, 2]
  %empty = tensor.empty() : tensor<2x8x4xf32>
  %result = linalg.broadcast ins(%reduced : tensor<8xf32>)
      outs(%empty : tensor<2x8x4xf32>) dimensions = [0, 2]
  return %result : tensor<2x8x4xf32>
}

// Diff: broadcast source rank mismatch
// CHECK-HIVM-LABEL: func.func @hivm_reduce_broadcast(
// CHECK-HIVM: %[[INPUT:.*]] = vector.transfer_read
// CHECK-HIVM: %[[INIT:.*]] = vector.transfer_read
// CHECK-HIVM: %[[INIT_FLAT:.*]] = vector.shape_cast %[[INIT]]
// CHECK-HIVM: %[[REDUCE:.*]] = vector.multi_reduction <maximumf>, %[[INPUT]], %[[INIT_FLAT]] [0, 2]
// CHECK-HIVM: %[[SHAPED:.*]] = vector.shape_cast %[[REDUCE]]
// CHECK-HIVM-NOT: vector.transfer_read
// CHECK-HIVM: %[[BROADCAST:.*]] = vector.broadcast %[[SHAPED]]
// CHECK-HIVM: vector.transfer_write %[[BROADCAST]]
func.func @hivm_reduce_broadcast(%input: tensor<2x8x4xf32>,
    %init: tensor<1x8x1xf32>) -> tensor<2x8x4xf32>
    attributes {hivm.vector_function} {
  %reduced = hivm.hir.vreduce <max>
      ins(%input : tensor<2x8x4xf32>)
      outs(%init : tensor<1x8x1xf32>) reduce_dims = [0, 2]
      -> tensor<1x8x1xf32>
  %empty = tensor.empty() : tensor<2x8x4xf32>
  %result = hivm.hir.vbrc ins(%reduced : tensor<1x8x1xf32>)
      outs(%empty : tensor<2x8x4xf32>) broadcast_dims = [0, 2]
      -> tensor<2x8x4xf32>
  return %result : tensor<2x8x4xf32>
}
