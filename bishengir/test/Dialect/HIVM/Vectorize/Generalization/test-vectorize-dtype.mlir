// RUN: bishengir-opt %s --split-input-file \
// RUN:   --hfusion-pre-vectorization-fusion --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// i1 elementwise operation.

// CHECK-HFUSION-LABEL: func.func @hfusion_xor_i1(
// CHECK-HFUSION: %[[LHS:.*]] = vector.transfer_read {{.*}} : tensor<16x16xi1>, vector<16x16xi1>
// CHECK-HFUSION: %[[RHS:.*]] = vector.transfer_read {{.*}} : tensor<16x16xi1>, vector<16x16xi1>
// CHECK-HFUSION: %[[RESULT:.*]] = arith.xori %[[LHS]], %[[RHS]] : vector<16x16xi1>
// CHECK-HFUSION: vector.transfer_write %[[RESULT]]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @hfusion_xor_i1(%lhs: tensor<16x16xi1>, %rhs: tensor<16x16xi1>)
    -> tensor<16x16xi1> {
  %empty = tensor.empty() : tensor<16x16xi1>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>}
      ins(%lhs, %rhs : tensor<16x16xi1>, tensor<16x16xi1>)
      outs(%empty : tensor<16x16xi1>) -> tensor<16x16xi1>
  return %result : tensor<16x16xi1>
}

// CHECK-HIVM-LABEL: func.func @hivm_xor_i1(
// CHECK-HIVM: %[[LHS:.*]] = vector.transfer_read {{.*}} : tensor<16x16xi1>, vector<16x16xi1>
// CHECK-HIVM: %[[RHS:.*]] = vector.transfer_read {{.*}} : tensor<16x16xi1>, vector<16x16xi1>
// CHECK-HIVM: %[[RESULT:.*]] = arith.xori %[[LHS]], %[[RHS]] : vector<16x16xi1>
// CHECK-HIVM: vector.transfer_write %[[RESULT]]
func.func @hivm_xor_i1(%lhs: tensor<16x16xi1>, %rhs: tensor<16x16xi1>)
    -> tensor<16x16xi1> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<16x16xi1>
  %result = hivm.hir.vxor
      ins(%lhs, %rhs : tensor<16x16xi1>, tensor<16x16xi1>)
      outs(%empty : tensor<16x16xi1>) -> tensor<16x16xi1>
  return %result : tensor<16x16xi1>
}
}

// -----

// Diff: broadcast scalar representation
// Linalg uses a rank-0 tensor; HIVM accepts the scalar directly.
// i1 scalar broadcast.

// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_i1(
// CHECK-HFUSION: vector.transfer_read {{.*}} : tensor<i1>, vector<16x16xi1>
// CHECK-HFUSION: vector.transfer_write {{.*}} : vector<16x16xi1>, tensor<16x16xi1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @hfusion_broadcast_i1(%input: tensor<i1>) -> tensor<16x16xi1> {
  %empty = tensor.empty() : tensor<16x16xi1>
  %result = linalg.broadcast ins(%input : tensor<i1>)
      outs(%empty : tensor<16x16xi1>) dimensions = [0, 1]
  return %result : tensor<16x16xi1>
}

// CHECK-HIVM-LABEL: func.func @hivm_broadcast_i1(
// CHECK-HIVM: vector.broadcast {{.*}} : i1 to vector<16x16xi1>
// CHECK-HIVM: vector.transfer_write {{.*}} : vector<16x16xi1>, tensor<16x16xi1>
func.func @hivm_broadcast_i1(%input: i1) -> tensor<16x16xi1>
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<16x16xi1>
  %result = hivm.hir.vbrc ins(%input : i1)
      outs(%empty : tensor<16x16xi1>) -> tensor<16x16xi1>
  return %result : tensor<16x16xi1>
}
}

// -----

// Diff: reduce shape mismatch
// Linalg drops the reduced axis; HIVM retains it as a unit dimension.
// i1 reduction. Sum is intentionally used because it is supported by both
// vectorizers and exposes the i1 combining operation directly.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_i1(
// CHECK-HFUSION: vector.multi_reduction <add>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<16x16xi1> to vector<16xi1>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @hfusion_reduce_i1(%input: tensor<16x16xi1>,
    %init: tensor<16xi1>) -> tensor<16xi1> {
  %result = linalg.reduce ins(%input : tensor<16x16xi1>)
      outs(%init : tensor<16xi1>) dimensions = [1]
      (%in: i1, %acc: i1) {
        %sum = arith.addi %in, %acc : i1
        linalg.yield %sum : i1
      }
  return %result : tensor<16xi1>
}

// CHECK-HIVM-LABEL: func.func @hivm_reduce_i1(
// CHECK-HIVM: vector.multi_reduction <add>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<16x16xi1> to vector<16xi1>
func.func @hivm_reduce_i1(%input: tensor<16x16xi1>,
    %init: tensor<16x1xi1>) -> tensor<16x1xi1>
    attributes {hivm.vector_function} {
  %result = hivm.hir.vreduce <sum> ins(%input : tensor<16x16xi1>)
      outs(%init : tensor<16x1xi1>) unsigned_src = false reduce_dims = [1]
      -> tensor<16x1xi1>
  return %result : tensor<16x1xi1>
}
}

// -----

// f8E4M3FN elementwise operation.

// CHECK-HFUSION-LABEL: func.func @hfusion_max_e4m3(
// CHECK-HFUSION: arith.maximumf {{.*}} : vector<16x16xf8E4M3FN>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @hfusion_max_e4m3(%lhs: tensor<16x16xf8E4M3FN>,
    %rhs: tensor<16x16xf8E4M3FN>) -> tensor<16x16xf8E4M3FN> {
  %empty = tensor.empty() : tensor<16x16xf8E4M3FN>
  %result = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
      ins(%lhs, %rhs : tensor<16x16xf8E4M3FN>, tensor<16x16xf8E4M3FN>)
      outs(%empty : tensor<16x16xf8E4M3FN>) -> tensor<16x16xf8E4M3FN>
  return %result : tensor<16x16xf8E4M3FN>
}

// CHECK-HIVM-LABEL: func.func @hivm_max_e4m3(
// CHECK-HIVM: arith.maximumf {{.*}} : vector<16x16xf8E4M3FN>
func.func @hivm_max_e4m3(%lhs: tensor<16x16xf8E4M3FN>,
    %rhs: tensor<16x16xf8E4M3FN>) -> tensor<16x16xf8E4M3FN>
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<16x16xf8E4M3FN>
  %result = hivm.hir.vmax
      ins(%lhs, %rhs : tensor<16x16xf8E4M3FN>, tensor<16x16xf8E4M3FN>)
      outs(%empty : tensor<16x16xf8E4M3FN>) -> tensor<16x16xf8E4M3FN>
  return %result : tensor<16x16xf8E4M3FN>
}
}

// -----

// Diff: broadcast scalar representation
// Linalg uses a rank-0 tensor; HIVM accepts the scalar directly.
// f8E5M2 scalar broadcast.

// CHECK-HFUSION-LABEL: func.func @hfusion_broadcast_e5m2(
// CHECK-HFUSION: vector.transfer_read {{.*}} : tensor<f8E5M2>, vector<16x16xf8E5M2>
// CHECK-HFUSION: vector.transfer_write {{.*}} : vector<16x16xf8E5M2>, tensor<16x16xf8E5M2>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @hfusion_broadcast_e5m2(%input: tensor<f8E5M2>)
    -> tensor<16x16xf8E5M2> {
  %empty = tensor.empty() : tensor<16x16xf8E5M2>
  %result = linalg.broadcast ins(%input : tensor<f8E5M2>)
      outs(%empty : tensor<16x16xf8E5M2>) dimensions = [0, 1]
  return %result : tensor<16x16xf8E5M2>
}

// CHECK-HIVM-LABEL: func.func @hivm_broadcast_e5m2(
// CHECK-HIVM: vector.broadcast {{.*}} : f8E5M2 to vector<16x16xf8E5M2>
// CHECK-HIVM: vector.transfer_write {{.*}} : vector<16x16xf8E5M2>, tensor<16x16xf8E5M2>
func.func @hivm_broadcast_e5m2(%input: f8E5M2) -> tensor<16x16xf8E5M2>
    attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<16x16xf8E5M2>
  %result = hivm.hir.vbrc ins(%input : f8E5M2)
      outs(%empty : tensor<16x16xf8E5M2>) -> tensor<16x16xf8E5M2>
  return %result : tensor<16x16xf8E5M2>
}
}

// -----

// Diff: reduce shape mismatch
// Linalg drops the reduced axis; HIVM retains it as a unit dimension.
// f16 reduction.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_f16(
// CHECK-HFUSION: vector.multi_reduction <maximumf>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<8x16xf16> to vector<8xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @hfusion_reduce_f16(%input: tensor<8x16xf16>,
    %init: tensor<8xf16>) -> tensor<8xf16> {
  %result = linalg.reduce ins(%input : tensor<8x16xf16>)
      outs(%init : tensor<8xf16>) dimensions = [1]
      (%in: f16, %acc: f16) {
        %max = arith.maximumf %in, %acc : f16
        linalg.yield %max : f16
      }
  return %result : tensor<8xf16>
}

// CHECK-HIVM-LABEL: func.func @hivm_reduce_f16(
// CHECK-HIVM: vector.multi_reduction <maximumf>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<8x16xf16> to vector<8xf16>
func.func @hivm_reduce_f16(%input: tensor<8x16xf16>,
    %init: tensor<8x1xf16>) -> tensor<8x1xf16>
    attributes {hivm.vector_function} {
  %result = hivm.hir.vreduce <max>
      ins(%input : tensor<8x16xf16>)
      outs(%init : tensor<8x1xf16>) unsigned_src = false
      reduce_dims = [1] -> tensor<8x1xf16>
  return %result : tensor<8x1xf16>
}
}

// -----

// Mixed-width i1-to-f16 cast. Both paths must use the wider element type to
// choose the common 8x16 tile.

// CHECK-HFUSION-LABEL: func.func @hfusion_cast_i1(
// CHECK-HFUSION: arith.uitofp {{.*}} : vector<8x16xi1> to vector<8x16xf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @hfusion_cast_i1(%input: tensor<8x16xi1>)
    -> tensor<8x16xf16> {
  %empty = tensor.empty() : tensor<8x16xf16>
  %result = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>,
      round_mode = #hfusion.round_mode<rint>}
      ins(%input : tensor<8x16xi1>) outs(%empty : tensor<8x16xf16>)
      -> tensor<8x16xf16>
  return %result : tensor<8x16xf16>
}

// CHECK-HIVM-LABEL: func.func @hivm_cast_i1(
// CHECK-HIVM: arith.uitofp {{.*}} : vector<8x16xi1> to vector<8x16xf16>
func.func @hivm_cast_i1(%input: tensor<8x16xi1>)
    -> tensor<8x16xf16> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x16xf16>
  %result = hivm.hir.vcast ins(%input : tensor<8x16xi1>)
      outs(%empty : tensor<8x16xf16>) round_mode = <rint>
      cast = <cast_unsigned> -> tensor<8x16xf16>
  return %result : tensor<8x16xf16>
}
}
