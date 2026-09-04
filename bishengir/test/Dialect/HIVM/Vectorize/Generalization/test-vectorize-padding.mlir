// RUN: bishengir-opt %s --split-input-file --hfusion-vectorize-ops \
// RUN:   --lower-vector-mask --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HFUSION
// RUN: bishengir-opt %s --split-input-file --hivm-vectorize-ops \
// RUN:   --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-HIVM

// Diff: reduction tail representation
// Linalg masks the reduction itself and can reuse a zero-padded source.
// HIVM instead pads each source with the exact combining identity.

// CHECK-HFUSION-LABEL: func.func @hfusion_float(
// CHECK-HFUSION-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-HFUSION: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HFUSION: %[[SRC:.*]] = vector.transfer_read {{.*}}, %[[ZERO]], %[[MASK]]
// CHECK-HFUSION: %[[SUM_INIT:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[SUM:.*]] = vector.mask %[[MASK]] { vector.multi_reduction <add>, %[[SRC]], %[[SUM_INIT]] [2]
// CHECK-HFUSION: vector.transfer_write %[[SUM]]
// CHECK-HFUSION: %[[PROD_INIT:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[PROD:.*]] = vector.mask %[[MASK]] { vector.multi_reduction <mul>, %[[SRC]], %[[PROD_INIT]] [2]
// CHECK-HFUSION: vector.transfer_write %[[PROD]]
// CHECK-HFUSION: %[[MAX_INIT:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[MAX:.*]] = vector.mask %[[MASK]] { vector.multi_reduction <maximumf>, %[[SRC]], %[[MAX_INIT]] [2]
// CHECK-HFUSION: vector.transfer_write %[[MAX]]
// CHECK-HFUSION: %[[MIN_INIT:.*]] = vector.transfer_read
// CHECK-HFUSION: %[[MIN:.*]] = vector.mask %[[MASK]] { vector.multi_reduction <minimumf>, %[[SRC]], %[[MIN_INIT]] [2]
// CHECK-HFUSION: vector.transfer_write %[[MIN]]
func.func @hfusion_float(%input: tensor<1x1x17xf32>,
    %sum_init: tensor<1x1xf32>, %prod_init: tensor<1x1xf32>,
    %max_init: tensor<1x1xf32>, %min_init: tensor<1x1xf32>)
    -> (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
        tensor<1x1xf32>) {
  %sum = linalg.reduce {arith.addf} ins(%input : tensor<1x1x17xf32>)
      outs(%sum_init : tensor<1x1xf32>) dimensions = [2]
  %prod = linalg.reduce {arith.mulf} ins(%input : tensor<1x1x17xf32>)
      outs(%prod_init : tensor<1x1xf32>) dimensions = [2]
  %max = linalg.reduce {arith.maximumf} ins(%input : tensor<1x1x17xf32>)
      outs(%max_init : tensor<1x1xf32>) dimensions = [2]
  %min = linalg.reduce {arith.minimumf} ins(%input : tensor<1x1x17xf32>)
      outs(%min_init : tensor<1x1xf32>) dimensions = [2]
  return %sum, %prod, %max, %min : tensor<1x1xf32>, tensor<1x1xf32>,
      tensor<1x1xf32>, tensor<1x1xf32>
}

// Diff: reduce shape mismatch
// CHECK-HIVM-LABEL: func.func @hivm_float(
// CHECK-HIVM-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-HIVM-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-HIVM-DAG: %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK-HIVM-DAG: %[[PINF:.*]] = arith.constant 0x7F800000 : f32
// CHECK-HIVM: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HIVM: %[[SUM_SRC:.*]] = vector.transfer_read {{.*}}, %[[ZERO]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <add>, %[[SUM_SRC]], {{.*}} [2]
// CHECK-HIVM: %[[PROD_SRC:.*]] = vector.transfer_read {{.*}}, %[[ONE]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <mul>, %[[PROD_SRC]], {{.*}} [2]
// CHECK-HIVM: %[[MAX_SRC:.*]] = vector.transfer_read {{.*}}, %[[NINF]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <maximumf>, %[[MAX_SRC]], {{.*}} [2]
// CHECK-HIVM: %[[MIN_SRC:.*]] = vector.transfer_read {{.*}}, %[[PINF]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <minimumf>, %[[MIN_SRC]], {{.*}} [2]
func.func @hivm_float(%input: tensor<1x1x17xf32>,
    %sum_init: tensor<1x1x1xf32>, %prod_init: tensor<1x1x1xf32>,
    %max_init: tensor<1x1x1xf32>, %min_init: tensor<1x1x1xf32>)
    -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>,
        tensor<1x1x1xf32>) attributes {hivm.vector_function} {
  %sum = hivm.hir.vreduce <sum> ins(%input : tensor<1x1x17xf32>)
      outs(%sum_init : tensor<1x1x1xf32>) reduce_dims = [2]
      -> tensor<1x1x1xf32>
  %prod = hivm.hir.vreduce <prod> ins(%input : tensor<1x1x17xf32>)
      outs(%prod_init : tensor<1x1x1xf32>) reduce_dims = [2]
      -> tensor<1x1x1xf32>
  %max = hivm.hir.vreduce <max> ins(%input : tensor<1x1x17xf32>)
      outs(%max_init : tensor<1x1x1xf32>) reduce_dims = [2]
      -> tensor<1x1x1xf32>
  %min = hivm.hir.vreduce <min> ins(%input : tensor<1x1x17xf32>)
      outs(%min_init : tensor<1x1x1xf32>) reduce_dims = [2]
      -> tensor<1x1x1xf32>
  return %sum, %prod, %max, %min : tensor<1x1x1xf32>, tensor<1x1x1xf32>,
      tensor<1x1x1xf32>, tensor<1x1x1xf32>
}

// -----

// Integer extrema and bitwise reductions exercise every distinct HIVM
// identity: signed limits, unsigned limits, zero, and all ones. Linalg uses
// the equivalent masked-reduction representation.

// CHECK-HFUSION-LABEL: func.func @hfusion_integer(
// CHECK-HFUSION-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-HFUSION: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HFUSION: %[[SRC:.*]] = vector.transfer_read {{.*}}, %[[ZERO]], %[[MASK]]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <maxsi>, %[[SRC]], {{.*}} [2]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <minsi>, %[[SRC]], {{.*}} [2]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <maxui>, %[[SRC]], {{.*}} [2]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <minui>, %[[SRC]], {{.*}} [2]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <xor>, %[[SRC]], {{.*}} [2]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <or>, %[[SRC]], {{.*}} [2]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <and>, %[[SRC]], {{.*}} [2]
func.func @hfusion_integer(%input: tensor<1x1x17xi32>,
    %smax_init: tensor<1x1xi32>, %smin_init: tensor<1x1xi32>,
    %umax_init: tensor<1x1xi32>, %umin_init: tensor<1x1xi32>,
    %xor_init: tensor<1x1xi32>, %or_init: tensor<1x1xi32>,
    %and_init: tensor<1x1xi32>)
    -> (tensor<1x1xi32>, tensor<1x1xi32>, tensor<1x1xi32>,
        tensor<1x1xi32>, tensor<1x1xi32>, tensor<1x1xi32>,
        tensor<1x1xi32>) {
  %smax = linalg.reduce {arith.maxsi} ins(%input : tensor<1x1x17xi32>)
      outs(%smax_init : tensor<1x1xi32>) dimensions = [2]
  %smin = linalg.reduce {arith.minsi} ins(%input : tensor<1x1x17xi32>)
      outs(%smin_init : tensor<1x1xi32>) dimensions = [2]
  %umax = linalg.reduce {arith.maxui} ins(%input : tensor<1x1x17xi32>)
      outs(%umax_init : tensor<1x1xi32>) dimensions = [2]
  %umin = linalg.reduce {arith.minui} ins(%input : tensor<1x1x17xi32>)
      outs(%umin_init : tensor<1x1xi32>) dimensions = [2]
  %xor = linalg.reduce {arith.xori} ins(%input : tensor<1x1x17xi32>)
      outs(%xor_init : tensor<1x1xi32>) dimensions = [2]
  %or = linalg.reduce {arith.ori} ins(%input : tensor<1x1x17xi32>)
      outs(%or_init : tensor<1x1xi32>) dimensions = [2]
  %and = linalg.reduce {arith.andi} ins(%input : tensor<1x1x17xi32>)
      outs(%and_init : tensor<1x1xi32>) dimensions = [2]
  return %smax, %smin, %umax, %umin, %xor, %or, %and : tensor<1x1xi32>,
      tensor<1x1xi32>, tensor<1x1xi32>, tensor<1x1xi32>, tensor<1x1xi32>,
      tensor<1x1xi32>, tensor<1x1xi32>
}

// Diff: reduce shape mismatch
// CHECK-HIVM-LABEL: func.func @hivm_integer(
// CHECK-HIVM-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-HIVM-DAG: %[[ONES:.*]] = arith.constant -1 : i32
// CHECK-HIVM-DAG: %[[SMIN:.*]] = arith.constant -2147483648 : i32
// CHECK-HIVM-DAG: %[[SMAX:.*]] = arith.constant 2147483647 : i32
// CHECK-HIVM: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x64xi1>
// CHECK-HIVM: %[[SMAX_SRC:.*]] = vector.transfer_read {{.*}}, %[[SMIN]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <maxsi>, %[[SMAX_SRC]], {{.*}} [2]
// CHECK-HIVM: %[[SMIN_SRC:.*]] = vector.transfer_read {{.*}}, %[[SMAX]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <minsi>, %[[SMIN_SRC]], {{.*}} [2]
// CHECK-HIVM: %[[UMAX_SRC:.*]] = vector.transfer_read {{.*}}, %[[ZERO]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <maxui>, %[[UMAX_SRC]], {{.*}} [2]
// CHECK-HIVM: %[[UMIN_SRC:.*]] = vector.transfer_read {{.*}}, %[[ONES]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <minui>, %[[UMIN_SRC]], {{.*}} [2]
// CHECK-HIVM: vector.multi_reduction <xor>, %[[UMAX_SRC]], {{.*}} [2]
// CHECK-HIVM: vector.multi_reduction <or>, %[[UMAX_SRC]], {{.*}} [2]
// CHECK-HIVM: vector.multi_reduction <and>, %[[UMIN_SRC]], {{.*}} [2]
func.func @hivm_integer(%input: tensor<1x1x17xi32>,
    %smax_init: tensor<1x1x1xi32>, %smin_init: tensor<1x1x1xi32>,
    %umax_init: tensor<1x1x1xi32>, %umin_init: tensor<1x1x1xi32>,
    %xor_init: tensor<1x1x1xi32>, %or_init: tensor<1x1x1xi32>,
    %and_init: tensor<1x1x1xi32>)
    -> (tensor<1x1x1xi32>, tensor<1x1x1xi32>, tensor<1x1x1xi32>,
        tensor<1x1x1xi32>, tensor<1x1x1xi32>, tensor<1x1x1xi32>,
        tensor<1x1x1xi32>) attributes {hivm.vector_function} {
  %smax = hivm.hir.vreduce <max> ins(%input : tensor<1x1x17xi32>)
      outs(%smax_init : tensor<1x1x1xi32>) reduce_dims = [2]
      -> tensor<1x1x1xi32>
  %smin = hivm.hir.vreduce <min> ins(%input : tensor<1x1x17xi32>)
      outs(%smin_init : tensor<1x1x1xi32>) reduce_dims = [2]
      -> tensor<1x1x1xi32>
  %umax = hivm.hir.vreduce <max> ins(%input : tensor<1x1x17xi32>)
      outs(%umax_init : tensor<1x1x1xi32>) unsigned_src = true
      reduce_dims = [2] -> tensor<1x1x1xi32>
  %umin = hivm.hir.vreduce <min> ins(%input : tensor<1x1x17xi32>)
      outs(%umin_init : tensor<1x1x1xi32>) unsigned_src = true
      reduce_dims = [2] -> tensor<1x1x1xi32>
  %xor = hivm.hir.vreduce <xori> ins(%input : tensor<1x1x17xi32>)
      outs(%xor_init : tensor<1x1x1xi32>) reduce_dims = [2]
      -> tensor<1x1x1xi32>
  %or = hivm.hir.vreduce <ori> ins(%input : tensor<1x1x17xi32>)
      outs(%or_init : tensor<1x1x1xi32>) reduce_dims = [2]
      -> tensor<1x1x1xi32>
  %and = hivm.hir.vreduce <andi> ins(%input : tensor<1x1x17xi32>)
      outs(%and_init : tensor<1x1x1xi32>) reduce_dims = [2]
      -> tensor<1x1x1xi32>
  return %smax, %smin, %umax, %umin, %xor, %or, %and : tensor<1x1x1xi32>,
      tensor<1x1x1xi32>, tensor<1x1x1xi32>, tensor<1x1x1xi32>,
      tensor<1x1x1xi32>, tensor<1x1x1xi32>, tensor<1x1x1xi32>
}

// -----

// Boolean any/all use the same OR/AND identities at i1 width.

// CHECK-HFUSION-LABEL: func.func @hfusion_boolean(
// CHECK-HFUSION-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-HFUSION-DAG: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x256xi1>
// CHECK-HFUSION: %[[SRC:.*]] = vector.transfer_read {{.*}}, %[[FALSE]], %[[MASK]]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <or>, %[[SRC]], {{.*}} [2]
// CHECK-HFUSION: vector.mask %[[MASK]] { vector.multi_reduction <and>, %[[SRC]], {{.*}} [2]
func.func @hfusion_boolean(%input: tensor<1x1x17xi1>,
    %any_init: tensor<1x1xi1>, %all_init: tensor<1x1xi1>)
    -> (tensor<1x1xi1>, tensor<1x1xi1>) {
  %any = linalg.reduce {arith.ori} ins(%input : tensor<1x1x17xi1>)
      outs(%any_init : tensor<1x1xi1>) dimensions = [2]
  %all = linalg.reduce {arith.andi} ins(%input : tensor<1x1x17xi1>)
      outs(%all_init : tensor<1x1xi1>) dimensions = [2]
  return %any, %all : tensor<1x1xi1>, tensor<1x1xi1>
}

// Diff: reduce shape mismatch
// CHECK-HIVM-LABEL: func.func @hivm_boolean(
// CHECK-HIVM-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-HIVM-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-HIVM-DAG: %[[MASK:.*]] = vector.constant_mask [1, 1, 17] : vector<1x1x256xi1>
// CHECK-HIVM: %[[ANY_SRC:.*]] = vector.transfer_read {{.*}}, %[[FALSE]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <or>, %[[ANY_SRC]], {{.*}} [2]
// CHECK-HIVM: %[[ALL_SRC:.*]] = vector.transfer_read {{.*}}, %[[TRUE]], %[[MASK]]
// CHECK-HIVM: vector.multi_reduction <and>, %[[ALL_SRC]], {{.*}} [2]
func.func @hivm_boolean(%input: tensor<1x1x17xi1>,
    %any_init: tensor<1x1x1xi1>, %all_init: tensor<1x1x1xi1>)
    -> (tensor<1x1x1xi1>, tensor<1x1x1xi1>)
    attributes {hivm.vector_function} {
  %any = hivm.hir.vreduce <any> ins(%input : tensor<1x1x17xi1>)
      outs(%any_init : tensor<1x1x1xi1>) reduce_dims = [2]
      -> tensor<1x1x1xi1>
  %all = hivm.hir.vreduce <all> ins(%input : tensor<1x1x17xi1>)
      outs(%all_init : tensor<1x1x1xi1>) reduce_dims = [2]
      -> tensor<1x1x1xi1>
  return %any, %all : tensor<1x1x1xi1>, tensor<1x1x1xi1>
}
