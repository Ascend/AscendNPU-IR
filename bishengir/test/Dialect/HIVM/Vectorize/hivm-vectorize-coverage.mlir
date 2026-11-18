// RUN: bishengir-opt %s --hivm-vectorize --verify-each -split-input-file | FileCheck %s --implicit-check-not=hivm.hir.vreduce --implicit-check-not=hivm.hir.varange --implicit-check-not=hivm.hir.vrec --implicit-check-not=hivm.hir.vmin --implicit-check-not=hivm.hir.vmax --implicit-check-not=linalg.

module {
  func.func @reduction(%arg: tensor<8xf32>, %init: tensor<1xf32>)
      -> tensor<1xf32> {
    %result = hivm.hir.vreduce<sum> ins(%arg : tensor<8xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %result : tensor<1xf32>
  }
}

// CHECK-LABEL: func.func private @reduction_fused_0
// CHECK: vector.reduction <add>
// CHECK: func.func @reduction(
// CHECK: call @reduction_fused_0

// -----

module {
  func.func @destination_chain(%arg: tensor<8xi32>, %init: tensor<8xi32>)
      -> tensor<8xi32> {
    %one = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %first = hivm.hir.vadd ins(%arg, %one : tensor<8xi32>, i32)
        outs(%init : tensor<8xi32>) -> tensor<8xi32>
    %second = hivm.hir.vmin ins(%first, %one : tensor<8xi32>, i32)
        outs(%first : tensor<8xi32>) -> tensor<8xi32>
    %third = hivm.hir.vmax ins(%second, %zero : tensor<8xi32>, i32)
        outs(%second : tensor<8xi32>) -> tensor<8xi32>
    return %third : tensor<8xi32>
  }
}
// CHECK-LABEL: func.func private @destination_chain_fused_0
// CHECK: arith.addi
// CHECK: arith.minsi
// CHECK: arith.maxsi
// CHECK-LABEL: func.func @destination_chain(
// CHECK: call @destination_chain_fused_0

// -----

module {
  func.func @indexed(%arg: tensor<67xf32>) -> (tensor<1xf32>, tensor<1xi32>) {
    %values = tensor.empty() : tensor<1xf32>
    %indices = tensor.empty() : tensor<1xi32>
    %result:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index_left>
        ins(%arg : tensor<67xf32>)
        outs(%values, %indices : tensor<1xf32>, tensor<1xi32>)
        reduce_dims = [0] -> tensor<1xf32>, tensor<1xi32>
    return %result#0, %result#1 : tensor<1xf32>, tensor<1xi32>
  }
}
// CHECK-LABEL: func.func private @indexed_fused_0
// CHECK: arith.maximumf
// CHECK: arith.cmpf oeq
// CHECK: arith.cmpf uno
// CHECK: arith.select
// CHECK: arith.minsi
// CHECK-LABEL: func.func @indexed(
// CHECK: call @indexed_fused_0

// -----

module {
  func.func @range_reciprocal(%offset: index, %s0: index, %s1: index,
                              %init: tensor<3x67xf32>) -> tensor<3x67xf32> {
    %range = hivm.hir.varange offset[%offset] strides[%s0, %s1]
        outs(%init : tensor<3x67xf32>) -> tensor<3x67xf32>
    %result = hivm.hir.vrec ins(%range : tensor<3x67xf32>)
        outs(%init : tensor<3x67xf32>) -> tensor<3x67xf32>
    return %result : tensor<3x67xf32>
  }
}
// CHECK-LABEL: func.func private @range_reciprocal_fused_0
// CHECK: arith.sitofp
// CHECK: arith.divf
// CHECK-LABEL: func.func @range_reciprocal(
// CHECK: call @range_reciprocal_fused_0

// -----

module {
  func.func @elementwise_ops(%arg: tensor<8xf32>, %init: tensor<8xf32>)
      -> tensor<8xf32> {
    %0 = hivm.hir.vabs ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %1 = hivm.hir.vrelu ins(%0 : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
// CHECK-LABEL: func.func private @elementwise_ops_fused_0
// CHECK: math.absf
// CHECK: arith.maximumf
// CHECK-LABEL: func.func @elementwise_ops(
// CHECK: call @elementwise_ops_fused_0

// -----

module {
  func.func @float_bitwise(%arg: tensor<8xf32>, %rhs: tensor<8xf32>,
                           %init: tensor<8xf32>) -> tensor<8xf32> {
    %0 = hivm.hir.vor ins(%arg, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %1 = hivm.hir.vand ins(%0, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
// CHECK-LABEL: func.func private @float_bitwise_fused_0
// CHECK: arith.bitcast
// CHECK: arith.ori
// CHECK: arith.andi
// CHECK-LABEL: func.func @float_bitwise(
// CHECK: call @float_bitwise_fused_0

// -----

module {
  func.func @shifts(%arg: tensor<8xi32>, %rhs: tensor<8xi32>,
                    %init: tensor<8xi32>) -> tensor<8xi32> {
    %0 = hivm.hir.vshl ins(%arg, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%init : tensor<8xi32>) -> tensor<8xi32>
    %1 = hivm.hir.vshr ins(%0, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%init : tensor<8xi32>) round : false -> tensor<8xi32>
    return %1 : tensor<8xi32>
  }
}
// CHECK-LABEL: func.func private @shifts_fused_0
// CHECK: arith.shli
// CHECK: arith.shrsi
// CHECK-LABEL: func.func @shifts(
// CHECK: call @shifts_fused_0

// -----

module {
  func.func @product(%arg: tensor<8xf32>, %init: tensor<1xf32>)
      -> tensor<1xf32> {
    %0 = hivm.hir.vreduce<prod> ins(%arg : tensor<8xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
// CHECK-LABEL: func.func private @product_fused_0
// CHECK-NOT: vector.reduction <mul>
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK-LABEL: func.func @product(
// CHECK: call @product_fused_0

// -----

module {
  func.func @and_reduction(%arg: tensor<8xi1>, %init: tensor<1xi1>)
      -> tensor<1xi1> {
    %0 = hivm.hir.vreduce<andi> ins(%arg : tensor<8xi1>)
        outs(%init : tensor<1xi1>) reduce_dims = [0] -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}
// CHECK-LABEL: func.func private @and_reduction_fused_0
// CHECK-NOT: vector.reduction <and>
// CHECK: scf.for
// CHECK: arith.andi
// CHECK-LABEL: func.func @and_reduction(
// CHECK: call @and_reduction_fused_0
