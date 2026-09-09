// RUN: bishengir-opt %s --split-input-file --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Elementwise: a 3-D tile whose trailing dimension needs a mask.

// CHECK-HFUSION-LABEL: func.func @hfusion_elemwise_tail(
// CHECK-HFUSION: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HFUSION: %[[READ:.*]] = vector.transfer_read {{.*}}, %[[MASK]]
// CHECK-HFUSION: %[[EXP:.*]] = math.exp %[[READ]] : vector<1x1x64xf32>
// CHECK-HFUSION: vector.transfer_write %[[EXP]], {{.*}}, %[[MASK]]
func.func @hfusion_elemwise_tail(%input: tensor<1x1x17xf32>)
    -> tensor<1x1x17xf32> {
  %empty = tensor.empty() : tensor<1x1x17xf32>
  %result = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
      ins(%input : tensor<1x1x17xf32>) outs(%empty : tensor<1x1x17xf32>)
      -> tensor<1x1x17xf32>
  return %result : tensor<1x1x17xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_elemwise_tail(
// CHECK-HIVM: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HIVM: %[[READ:.*]] = vector.transfer_read {{.*}}, %[[MASK]]
// CHECK-HIVM-SAME: : tensor<1x1x17xf32>, vector<1x1x64xf32>
// CHECK-HIVM: %[[EXP:.*]] = math.exp %[[READ]] : vector<1x1x64xf32>
// CHECK-HIVM: vector.transfer_write %[[EXP]], {{.*}}, %[[MASK]]
func.func @hivm_elemwise_tail(%input: tensor<1x1x17xf32>)
    -> tensor<1x1x17xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<1x1x17xf32>
  %result = hivm.hir.vexp ins(%input : tensor<1x1x17xf32>)
      outs(%empty : tensor<1x1x17xf32>) -> tensor<1x1x17xf32>
  return %result : tensor<1x1x17xf32>
}

// -----

// Elementwise: all dimensions are unit dimensions.

// CHECK-HFUSION-LABEL: func.func @hfusion_elemwise_unit(
// CHECK-HFUSION: vector.constant_mask [1, 1] : vector<1x64xi1>
// CHECK-HFUSION: math.exp {{.*}} : vector<1x64xf32>
// CHECK-HFUSION: vector.transfer_write {{.*}} : vector<1x64xf32>, tensor<1x1xf32>
func.func @hfusion_elemwise_unit(%input: tensor<1x1xf32>)
    -> tensor<1x1xf32> {
  %empty = tensor.empty() : tensor<1x1xf32>
  %result = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
      ins(%input : tensor<1x1xf32>) outs(%empty : tensor<1x1xf32>)
      -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_elemwise_unit(
// CHECK-HIVM: vector.constant_mask [1, 1] : vector<1x64xi1>
// CHECK-HIVM: math.exp {{.*}} : vector<1x64xf32>
// CHECK-HIVM: vector.transfer_write {{.*}} : vector<1x64xf32>, tensor<1x1xf32>
func.func @hivm_elemwise_unit(%input: tensor<1x1xf32>)
    -> tensor<1x1xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<1x1xf32>
  %result = hivm.hir.vexp ins(%input : tensor<1x1xf32>)
      outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

// -----

// Diff: broadcast source rank mismatch
// Linalg inserts an axis; HIVM represents it as a same-rank unit axis.
// Broadcast: expand the partial trailing dimension of a 3-D tile.

// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_tail(
// CHECK-HFUSION: vector.transfer_read {{.*}} : tensor<1x1xf32>, vector<1x1x64xf32>
// CHECK-HFUSION: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HFUSION: vector.transfer_write {{.*}}, {{.*}}, %[[MASK]]
func.func @hfusion_broadcast_tail(%input: tensor<1x1xf32>)
    -> tensor<1x1x17xf32> {
  %empty = tensor.empty() : tensor<1x1x17xf32>
  %result = linalg.broadcast ins(%input : tensor<1x1xf32>)
      outs(%empty : tensor<1x1x17xf32>) dimensions = [2]
  return %result : tensor<1x1x17xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_broadcast_tail(
// CHECK-HIVM: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HIVM: vector.broadcast {{.*}} : vector<1x1x1xf32> to vector<1x1x64xf32>
// CHECK-HIVM: vector.transfer_write {{.*}}, {{.*}}, %[[MASK]]
func.func @hivm_broadcast_tail(%input: tensor<1x1x1xf32>)
    -> tensor<1x1x17xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<1x1x17xf32>
  %result = hivm.hir.vbrc ins(%input : tensor<1x1x1xf32>)
      outs(%empty : tensor<1x1x17xf32>) broadcast_dims = [2]
      -> tensor<1x1x17xf32>
  return %result : tensor<1x1x17xf32>
}

// -----

// Diff: broadcast scalar representation
// Linalg uses a rank-0 tensor; HIVM accepts the scalar directly.
// Broadcast: both destination dimensions are unit dimensions.

// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_unit(
// CHECK-HFUSION: vector.transfer_read {{.*}} : tensor<f32>, vector<1x64xf32>
// CHECK-HFUSION: vector.constant_mask [1, 1] : vector<1x64xi1>
// CHECK-HFUSION: vector.transfer_write {{.*}} : vector<1x64xf32>, tensor<1x1xf32>
func.func @hfusion_broadcast_unit(%input: tensor<f32>)
    -> tensor<1x1xf32> {
  %empty = tensor.empty() : tensor<1x1xf32>
  %result = linalg.broadcast ins(%input : tensor<f32>)
      outs(%empty : tensor<1x1xf32>) dimensions = [0, 1]
  return %result : tensor<1x1xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_broadcast_unit(
// CHECK-HIVM: vector.constant_mask [1, 1] : vector<1x64xi1>
// CHECK-HIVM: vector.broadcast {{.*}} : f32 to vector<1x64xf32>
// CHECK-HIVM: vector.transfer_write {{.*}} : vector<1x64xf32>, tensor<1x1xf32>
func.func @hivm_broadcast_unit(%input: f32)
    -> tensor<1x1xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<1x1xf32>
  %result = hivm.hir.vbrc ins(%input : f32)
      outs(%empty : tensor<1x1xf32>)
      -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

// -----

// Diff: reduce shape mismatch
// Linalg drops the reduced axis; HIVM retains it as a unit dimension.
// Reduction: reduce a masked trailing dimension.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_tail(
// CHECK-HFUSION: vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HFUSION: vector.multi_reduction <add>, {{.*}} [2]
// CHECK-HFUSION-SAME: vector<1x1x64xf32> to vector<1x1xf32>
// CHECK-HFUSION: vector.transfer_write {{.*}} : vector<1x1xf32>, tensor<1x1xf32>
func.func @hfusion_reduce_tail(%input: tensor<1x1x17xf32>,
    %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %result = linalg.reduce ins(%input : tensor<1x1x17xf32>)
      outs(%init : tensor<1x1xf32>) dimensions = [2]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  return %result : tensor<1x1xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_reduce_tail(
// CHECK-HIVM: vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HIVM: vector.multi_reduction <add>, {{.*}} [2]
// CHECK-HIVM-SAME: vector<1x1x64xf32> to vector<1x1xf32>
// CHECK-HIVM: vector.shape_cast {{.*}} : vector<1x1xf32> to vector<1x1x1xf32>
// CHECK-HIVM: vector.transfer_write {{.*}} : vector<1x1x1xf32>, tensor<1x1x1xf32>
func.func @hivm_reduce_tail(%input: tensor<1x1x17xf32>,
    %init: tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    attributes {hivm.vector_function} {
  %result = hivm.hir.vreduce <sum> ins(%input : tensor<1x1x17xf32>)
      outs(%init : tensor<1x1x1xf32>) unsigned_src = false reduce_dims = [2]
      -> tensor<1x1x1xf32>
  return %result : tensor<1x1x1xf32>
}

// -----

// Diff: reduce shape mismatch
// Linalg drops the reduced axis; HIVM retains it as a unit dimension.
// Reduction: both input dimensions are unit dimensions.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_unit(
// CHECK-HFUSION: vector.multi_reduction <add>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<1x64xf32> to vector<1xf32>
func.func @hfusion_reduce_unit(%input: tensor<1x1xf32>,
    %init: tensor<1xf32>) -> tensor<1xf32> {
  %result = linalg.reduce ins(%input : tensor<1x1xf32>)
      outs(%init : tensor<1xf32>) dimensions = [1]
      (%in: f32, %acc: f32) {
        %sum = arith.addf %in, %acc : f32
        linalg.yield %sum : f32
      }
  return %result : tensor<1xf32>
}

// CHECK-HIVM-LABEL: func.func @hivm_reduce_unit(
// CHECK-HIVM: vector.multi_reduction <add>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<1x64xf32> to vector<1xf32>
// CHECK-HIVM: vector.shape_cast {{.*}} : vector<1xf32> to vector<1x1xf32>
func.func @hivm_reduce_unit(%input: tensor<1x1xf32>,
    %init: tensor<1x1xf32>) -> tensor<1x1xf32>
    attributes {hivm.vector_function} {
  %result = hivm.hir.vreduce <sum> ins(%input : tensor<1x1xf32>)
      outs(%init : tensor<1x1xf32>) unsigned_src = false reduce_dims = [1]
      -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}
