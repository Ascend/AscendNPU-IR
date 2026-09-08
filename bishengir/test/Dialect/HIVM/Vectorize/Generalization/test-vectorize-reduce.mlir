// RUN: bishengir-opt %s --split-input-file --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Floating product, maximum, and minimum.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_float(
// CHECK-HFUSION: vector.multi_reduction <mul>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<8x8xf32> to vector<8xf32>
// CHECK-HFUSION: vector.multi_reduction <maximumf>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<8x8xf32> to vector<8xf32>
// CHECK-HFUSION: vector.multi_reduction <minimumf>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<8x8xf32> to vector<8xf32>
func.func @hfusion_reduce_float(%input: tensor<8x8xf32>,
    %prod_init: tensor<8xf32>, %max_init: tensor<8xf32>,
    %min_init: tensor<8xf32>)
    -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
  %prod = linalg.reduce {arith.mulf} ins(%input : tensor<8x8xf32>)
      outs(%prod_init : tensor<8xf32>) dimensions = [1]
  %max = linalg.reduce {arith.maximumf} ins(%input : tensor<8x8xf32>)
      outs(%max_init : tensor<8xf32>) dimensions = [1]
  %min = linalg.reduce {arith.minimumf} ins(%input : tensor<8x8xf32>)
      outs(%min_init : tensor<8xf32>) dimensions = [1]
  return %prod, %max, %min : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// Diff: reduce shape mismatch
// HIVM retains the reduced axis as a unit dimension.
// CHECK-HIVM-LABEL: func.func @hivm_reduce_float(
// CHECK-HIVM: vector.multi_reduction <mul>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<8x8xf32> to vector<8xf32>
// CHECK-HIVM: vector.multi_reduction <maximumf>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<8x8xf32> to vector<8xf32>
// CHECK-HIVM: vector.multi_reduction <minimumf>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<8x8xf32> to vector<8xf32>
func.func @hivm_reduce_float(%input: tensor<8x8xf32>,
    %prod_init: tensor<8x1xf32>, %max_init: tensor<8x1xf32>,
    %min_init: tensor<8x1xf32>)
    -> (tensor<8x1xf32>, tensor<8x1xf32>, tensor<8x1xf32>)
    attributes {hivm.vector_function} {
  %prod = hivm.hir.vreduce <prod> ins(%input : tensor<8x8xf32>)
      outs(%prod_init : tensor<8x1xf32>) reduce_dims = [1]
      -> tensor<8x1xf32>
  %max = hivm.hir.vreduce <max> ins(%input : tensor<8x8xf32>)
      outs(%max_init : tensor<8x1xf32>) reduce_dims = [1]
      -> tensor<8x1xf32>
  %min = hivm.hir.vreduce <min> ins(%input : tensor<8x8xf32>)
      outs(%min_init : tensor<8x1xf32>) reduce_dims = [1]
      -> tensor<8x1xf32>
  return %prod, %max, %min : tensor<8x1xf32>, tensor<8x1xf32>,
      tensor<8x1xf32>
}

// -----

// Signedness is part of the reduction semantic contract.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_int(
// CHECK-HFUSION: vector.multi_reduction <maxsi>, {{.*}} [1]
// CHECK-HFUSION: vector.multi_reduction <maxui>, {{.*}} [1]
// CHECK-HFUSION: vector.multi_reduction <minsi>, {{.*}} [1]
// CHECK-HFUSION: vector.multi_reduction <minui>, {{.*}} [1]
func.func @hfusion_reduce_int(%input: tensor<8x8xi32>,
    %smax_init: tensor<8xi32>, %umax_init: tensor<8xi32>,
    %smin_init: tensor<8xi32>, %umin_init: tensor<8xi32>)
    -> (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>, tensor<8xi32>) {
  %smax = linalg.reduce {arith.maxsi} ins(%input : tensor<8x8xi32>)
      outs(%smax_init : tensor<8xi32>) dimensions = [1]
  %umax = linalg.reduce {arith.maxui} ins(%input : tensor<8x8xi32>)
      outs(%umax_init : tensor<8xi32>) dimensions = [1]
  %smin = linalg.reduce {arith.minsi} ins(%input : tensor<8x8xi32>)
      outs(%smin_init : tensor<8xi32>) dimensions = [1]
  %umin = linalg.reduce {arith.minui} ins(%input : tensor<8x8xi32>)
      outs(%umin_init : tensor<8xi32>) dimensions = [1]
  return %smax, %umax, %smin, %umin : tensor<8xi32>, tensor<8xi32>,
      tensor<8xi32>, tensor<8xi32>
}

// Diff: reduce shape mismatch
// CHECK-HIVM-LABEL: func.func @hivm_reduce_int(
// CHECK-HIVM: vector.multi_reduction <maxsi>, {{.*}} [1]
// CHECK-HIVM: vector.multi_reduction <maxui>, {{.*}} [1]
// CHECK-HIVM: vector.multi_reduction <minsi>, {{.*}} [1]
// CHECK-HIVM: vector.multi_reduction <minui>, {{.*}} [1]
func.func @hivm_reduce_int(%input: tensor<8x8xi32>,
    %smax_init: tensor<8x1xi32>, %umax_init: tensor<8x1xi32>,
    %smin_init: tensor<8x1xi32>, %umin_init: tensor<8x1xi32>)
    -> (tensor<8x1xi32>, tensor<8x1xi32>, tensor<8x1xi32>,
        tensor<8x1xi32>) attributes {hivm.vector_function} {
  %smax = hivm.hir.vreduce <max> ins(%input : tensor<8x8xi32>)
      outs(%smax_init : tensor<8x1xi32>) unsigned_src = false
      reduce_dims = [1] -> tensor<8x1xi32>
  %umax = hivm.hir.vreduce <max> ins(%input : tensor<8x8xi32>)
      outs(%umax_init : tensor<8x1xi32>) unsigned_src = true
      reduce_dims = [1] -> tensor<8x1xi32>
  %smin = hivm.hir.vreduce <min> ins(%input : tensor<8x8xi32>)
      outs(%smin_init : tensor<8x1xi32>) unsigned_src = false
      reduce_dims = [1] -> tensor<8x1xi32>
  %umin = hivm.hir.vreduce <min> ins(%input : tensor<8x8xi32>)
      outs(%umin_init : tensor<8x1xi32>) unsigned_src = true
      reduce_dims = [1] -> tensor<8x1xi32>
  return %smax, %umax, %smin, %umin : tensor<8x1xi32>, tensor<8x1xi32>,
      tensor<8x1xi32>, tensor<8x1xi32>
}

// -----

// Multiple reduction axes and a caller-provided non-zero accumulator.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_multi(
// CHECK-HFUSION: %[[INIT:.*]] = vector.transfer_read {{.*}} : tensor<2xf32>, vector<2xf32>
// CHECK-HFUSION: vector.multi_reduction <add>, {{.*}}, %[[INIT]] [1, 2]
// CHECK-HFUSION-SAME: vector<2x4x8xf32> to vector<2xf32>
func.func @hfusion_reduce_multi(%input: tensor<2x4x8xf32>,
    %init: tensor<2xf32>) -> tensor<2xf32> {
  %result = linalg.reduce {arith.addf} ins(%input : tensor<2x4x8xf32>)
      outs(%init : tensor<2xf32>) dimensions = [1, 2]
  return %result : tensor<2xf32>
}

// Diff: reduce shape mismatch
// CHECK-HIVM-LABEL: func.func @hivm_reduce_multi(
// CHECK-HIVM: vector.shape_cast {{.*}} : vector<2x1x1xf32> to vector<2xf32>
// CHECK-HIVM: vector.multi_reduction <add>, {{.*}} [1, 2]
// CHECK-HIVM-SAME: vector<2x4x8xf32> to vector<2xf32>
func.func @hivm_reduce_multi(%input: tensor<2x4x8xf32>,
    %init: tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
    attributes {hivm.vector_function} {
  %result = hivm.hir.vreduce <sum> ins(%input : tensor<2x4x8xf32>)
      outs(%init : tensor<2x1x1xf32>) reduce_dims = [1, 2]
      -> tensor<2x1x1xf32>
  return %result : tensor<2x1x1xf32>
}

// -----

// Logical and bitwise reductions are supported by AutoVectorizeV2's Linalg
// vectorization and by the HFusion-to-HIVM reduction mapping.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_logic(
// CHECK-HFUSION: vector.multi_reduction <or>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<16x16xi1> to vector<16xi1>
// CHECK-HFUSION: vector.multi_reduction <and>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<16x16xi1> to vector<16xi1>
// CHECK-HFUSION: vector.multi_reduction <xor>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<8x8xi32> to vector<8xi32>
// CHECK-HFUSION: vector.multi_reduction <or>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<8x8xi32> to vector<8xi32>
// CHECK-HFUSION: vector.multi_reduction <and>, {{.*}} [1]
// CHECK-HFUSION-SAME: vector<8x8xi32> to vector<8xi32>
func.func @hfusion_reduce_logic(%bool_input: tensor<16x16xi1>,
    %any_init: tensor<16xi1>, %all_init: tensor<16xi1>,
    %int_input: tensor<8x8xi32>, %xor_init: tensor<8xi32>,
    %or_init: tensor<8xi32>, %and_init: tensor<8xi32>)
    -> (tensor<16xi1>, tensor<16xi1>, tensor<8xi32>, tensor<8xi32>,
        tensor<8xi32>) {
  %any = linalg.reduce {arith.ori} ins(%bool_input : tensor<16x16xi1>)
      outs(%any_init : tensor<16xi1>) dimensions = [1]
  %all = linalg.reduce {arith.andi} ins(%bool_input : tensor<16x16xi1>)
      outs(%all_init : tensor<16xi1>) dimensions = [1]
  %xor = linalg.reduce {arith.xori} ins(%int_input : tensor<8x8xi32>)
      outs(%xor_init : tensor<8xi32>) dimensions = [1]
  %or = linalg.reduce {arith.ori} ins(%int_input : tensor<8x8xi32>)
      outs(%or_init : tensor<8xi32>) dimensions = [1]
  %and = linalg.reduce {arith.andi} ins(%int_input : tensor<8x8xi32>)
      outs(%and_init : tensor<8xi32>) dimensions = [1]
  return %any, %all, %xor, %or, %and : tensor<16xi1>, tensor<16xi1>,
      tensor<8xi32>, tensor<8xi32>, tensor<8xi32>
}

// Diff: reduce shape mismatch
// CHECK-HIVM-LABEL: func.func @hivm_reduce_logic(
// CHECK-HIVM: vector.multi_reduction <or>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<16x16xi1> to vector<16xi1>
// CHECK-HIVM: vector.multi_reduction <and>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<16x16xi1> to vector<16xi1>
// CHECK-HIVM: vector.multi_reduction <xor>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<8x8xi32> to vector<8xi32>
// CHECK-HIVM: vector.multi_reduction <or>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<8x8xi32> to vector<8xi32>
// CHECK-HIVM: vector.multi_reduction <and>, {{.*}} [1]
// CHECK-HIVM-SAME: vector<8x8xi32> to vector<8xi32>
func.func @hivm_reduce_logic(%bool_input: tensor<16x16xi1>,
    %any_init: tensor<16x1xi1>, %all_init: tensor<16x1xi1>,
    %int_input: tensor<8x8xi32>, %xor_init: tensor<8x1xi32>,
    %or_init: tensor<8x1xi32>, %and_init: tensor<8x1xi32>)
    -> (tensor<16x1xi1>, tensor<16x1xi1>, tensor<8x1xi32>,
        tensor<8x1xi32>, tensor<8x1xi32>) attributes {hivm.vector_function} {
  %any = hivm.hir.vreduce <any> ins(%bool_input : tensor<16x16xi1>)
      outs(%any_init : tensor<16x1xi1>) reduce_dims = [1]
      -> tensor<16x1xi1>
  %all = hivm.hir.vreduce <all> ins(%bool_input : tensor<16x16xi1>)
      outs(%all_init : tensor<16x1xi1>) reduce_dims = [1]
      -> tensor<16x1xi1>
  %xor = hivm.hir.vreduce <xori> ins(%int_input : tensor<8x8xi32>)
      outs(%xor_init : tensor<8x1xi32>) reduce_dims = [1]
      -> tensor<8x1xi32>
  %or = hivm.hir.vreduce <ori> ins(%int_input : tensor<8x8xi32>)
      outs(%or_init : tensor<8x1xi32>) reduce_dims = [1]
      -> tensor<8x1xi32>
  %and = hivm.hir.vreduce <andi> ins(%int_input : tensor<8x8xi32>)
      outs(%and_init : tensor<8x1xi32>) reduce_dims = [1]
      -> tensor<8x1xi32>
  return %any, %all, %xor, %or, %and : tensor<16x1xi1>, tensor<16x1xi1>,
      tensor<8x1xi32>, tensor<8x1xi32>, tensor<8x1xi32>
}

// -----

// Diff: reduction NaN semantics mismatch
// A5 Linalg vectorization distinguishes maxnum/minnum. HIVM currently has
// only NaN-propagating max/min reduction kinds.

// CHECK-HFUSION-LABEL: func.func @hfusion_reduce_num(
// CHECK-HFUSION: vector.multi_reduction <maxnumf>, {{.*}} [1]
// CHECK-HFUSION: vector.multi_reduction <minnumf>, {{.*}} [1]
func.func @hfusion_reduce_num(%input: tensor<8x8xf32>,
    %max_init: tensor<8xf32>, %min_init: tensor<8xf32>)
    -> (tensor<8xf32>, tensor<8xf32>) {
  %max = linalg.reduce {arith.maxnumf} ins(%input : tensor<8x8xf32>)
      outs(%max_init : tensor<8xf32>) dimensions = [1]
  %min = linalg.reduce {arith.minnumf} ins(%input : tensor<8x8xf32>)
      outs(%min_init : tensor<8xf32>) dimensions = [1]
  return %max, %min : tensor<8xf32>, tensor<8xf32>
}

// Diff: reduce shape mismatch
// CHECK-HIVM-LABEL: func.func @hivm_reduce_num(
// CHECK-HIVM: vector.multi_reduction <maximumf>, {{.*}} [1]
// CHECK-HIVM: vector.multi_reduction <minimumf>, {{.*}} [1]
func.func @hivm_reduce_num(%input: tensor<8x8xf32>,
    %max_init: tensor<8x1xf32>, %min_init: tensor<8x1xf32>)
    -> (tensor<8x1xf32>, tensor<8x1xf32>)
    attributes {hivm.vector_function} {
  %max = hivm.hir.vreduce <max> ins(%input : tensor<8x8xf32>)
      outs(%max_init : tensor<8x1xf32>) reduce_dims = [1]
      -> tensor<8x1xf32>
  %min = hivm.hir.vreduce <min> ins(%input : tensor<8x8xf32>)
      outs(%min_init : tensor<8x1xf32>) reduce_dims = [1]
      -> tensor<8x1xf32>
  return %max, %min : tensor<8x1xf32>, tensor<8x1xf32>
}
