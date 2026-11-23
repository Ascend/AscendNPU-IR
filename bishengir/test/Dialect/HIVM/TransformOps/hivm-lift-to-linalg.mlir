// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload" -split-input-file 2>&1 | FileCheck %s
// RUN: bishengir-opt %s -transform-interpreter="debug-payload-root-tag=payload disable-expensive-checks" -split-input-file 2>&1 | FileCheck %s


// CHECK-LABEL: func.func @unary
// CHECK-DAG: math.exp2
// CHECK-DAG: math.log2
// CHECK-DAG: math.log10
// CHECK-DAG: math.log1p
// CHECK-DAG: math.expm1
// CHECK-DAG: math.absf
// CHECK-DAG: math.log2
// CHECK-DAG: math.floor
// CHECK-DAG: arith.maximumf
// CHECK-DAG: math.tanh
// CHECK-DAG: math.sin
// CHECK-DAG: math.cos
// CHECK-DAG: math.atan
// CHECK-DAG: math.tan
// CHECK-DAG: math.erf
// CHECK-NOT: hivm.hir.v
module attributes {transform.target_tag = "payload"} {
  func.func @unary(%arg: tensor<8xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
    %0 = hivm.hir.vexp2 ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %1 = hivm.hir.vlog2 ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %2 = hivm.hir.vlog10 ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %3 = hivm.hir.vlog1p ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %4 = hivm.hir.vexpm1 ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %5 = hivm.hir.vilogb ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %6 = hivm.hir.vtanh ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %7 = hivm.hir.vsin ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %8 = hivm.hir.vcos ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %9 = hivm.hir.vatan ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %10 = hivm.hir.vtan ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %11 = hivm.hir.verf ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %12 = hivm.hir.vabs ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    %13 = hivm.hir.vrelu ins(%12 : tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %13 : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: #{{.*}} = affine_map<(d0) -> ()>
// CHECK-LABEL: func.func @binary
// CHECK-DAG: arith.addf
// CHECK-DAG: arith.mulf
// CHECK-DAG: arith.addi
// CHECK-DAG: arith.muli
// CHECK-DAG: arith.remf
// CHECK-DAG: arith.remsi
// CHECK-DAG: arith.remui
// CHECK-DAG: arith.bitcast
// CHECK-DAG: arith.ori
// CHECK-DAG: arith.andi
// CHECK-DAG: ins(%{{.*}}, %{{.*}} : tensor<8xf32>, f32)
// CHECK-NOT: hivm.hir.v
module attributes {transform.target_tag = "payload"} {
  func.func @binary(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>, %scalar: f32,
                    %ilhs: tensor<8xi32>, %irhs: tensor<8xi32>,
                    %finit: tensor<8xf32>, %iinit: tensor<8xi32>)
      -> tensor<8xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%finit : tensor<8xf32>) -> tensor<8xf32>
    %1 = hivm.hir.vmul ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%finit : tensor<8xf32>) -> tensor<8xf32>
    %2 = hivm.hir.vadd ins(%ilhs, %irhs : tensor<8xi32>, tensor<8xi32>)
        outs(%iinit : tensor<8xi32>) -> tensor<8xi32>
    %3 = hivm.hir.vmul ins(%ilhs, %irhs : tensor<8xi32>, tensor<8xi32>)
        outs(%iinit : tensor<8xi32>) -> tensor<8xi32>
    %4 = hivm.hir.vmod ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%finit : tensor<8xf32>) -> tensor<8xf32>
    %5 = hivm.hir.vmod ins(%ilhs, %irhs : tensor<8xi32>, tensor<8xi32>)
        outs(%iinit : tensor<8xi32>) -> tensor<8xi32>
    %6 = hivm.hir.vmodui ins(%ilhs, %irhs : tensor<8xi32>, tensor<8xi32>)
        outs(%iinit : tensor<8xi32>) -> tensor<8xi32>
    %7 = hivm.hir.vldexp ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%finit : tensor<8xf32>) -> tensor<8xf32>
    %8 = hivm.hir.vadd ins(%lhs, %scalar : tensor<8xf32>, f32)
        outs(%finit : tensor<8xf32>) -> tensor<8xf32>
    %9 = hivm.hir.vor ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%finit : tensor<8xf32>) -> tensor<8xf32>
    %10 = hivm.hir.vand ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%finit : tensor<8xf32>) -> tensor<8xf32>
    return %10 : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: #{{.*}} = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-LABEL: func.func @broadcast
// CHECK: linalg.generic
// CHECK-NOT: hivm.hir.v
module attributes {transform.target_tag = "payload"} {
  func.func @broadcast(%lhs: tensor<8x1xf32>, %rhs: tensor<8x8xf32>,
                       %init: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<8x1xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) broadcast = [1] -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK: #{{.*}} = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-LABEL: func.func @transpose
// CHECK: linalg.generic
// CHECK-NOT: hivm.hir.v
module attributes {transform.target_tag = "payload"} {
  func.func @transpose(%lhs: tensor<4x8xf32>, %rhs: tensor<4x8xf32>,
                       %init: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = hivm.hir.vadd ins(%lhs, %rhs : tensor<4x8xf32>, tensor<4x8xf32>)
        outs(%init : tensor<8x4xf32>) transpose = [1, 0] -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func @concat
// CHECK: hivm.hir.vconcat
// CHECK-NOT: linalg.generic
module attributes {transform.target_tag = "payload"} {
  func.func @concat(%lhs: tensor<4x2xf32>, %rhs: tensor<4x2xf32>,
                    %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = hivm.hir.vconcat dim(1) ins(%lhs, %rhs : tensor<4x2xf32>, tensor<4x2xf32>)
        outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func @mulext
// CHECK: %[[SI_LOW:.*]], %[[SI_HIGH:.*]] = arith.mulsi_extended
// CHECK: linalg.yield %[[SI_LOW]], %[[SI_HIGH]] : i32, i32
// CHECK: %[[UI_LOW:.*]], %[[UI_HIGH:.*]] = arith.mului_extended
// CHECK: linalg.yield %[[UI_LOW]], %[[UI_HIGH]] : i32, i32
// CHECK-NOT: hivm.hir.v
module attributes {transform.target_tag = "payload"} {
  func.func @mulext(%lhs: tensor<8xi32>, %rhs: tensor<8xi32>,
                    %low: tensor<8xi32>, %high: tensor<8xi32>)
      -> (tensor<8xi32>, tensor<8xi32>) {
    %0:2 = hivm.hir.vmulext
        ins(%lhs, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%low, %high : tensor<8xi32>, tensor<8xi32>)
        -> tensor<8xi32>, tensor<8xi32>
    %1:2 = hivm.hir.vmulextui
        ins(%lhs, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%low, %high : tensor<8xi32>, tensor<8xi32>)
        -> tensor<8xi32>, tensor<8xi32>
    return %0#0, %1#0 : tensor<8xi32>, tensor<8xi32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func @predicates
// CHECK: arith.constant
// CHECK: arith.cmpf oeq
// CHECK: arith.cmpf uno
// CHECK-NOT: hivm.hir.v
module attributes {transform.target_tag = "payload"} {
  func.func @predicates(%arg: tensor<8xf32>, %init: tensor<8xi1>)
      -> tensor<8xi1> {
    %0 = hivm.hir.visinf ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xi1>) -> tensor<8xi1>
    %1 = hivm.hir.visnan ins(%arg : tensor<8xf32>)
        outs(%init : tensor<8xi1>) -> tensor<8xi1>
    return %1 : tensor<8xi1>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func private @vf(
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: linalg.generic
// CHECK-LABEL: func.func @caller(
// CHECK: %{{.*}} = call @vf({{.*}}) : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
// CHECK-NOT: hivm.hir.v
module attributes {transform.target_tag = "payload"} {
  func.func private @vf(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                        %init: tensor<8xf32>) -> tensor<8xf32>
      attributes {hivm.vector_function, no_inline} {
    %0 = hivm.hir.vmul ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
        outs(%init : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @caller(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                    %init: tensor<8xf32>) -> tensor<8xf32> {
    %0 = call @vf(%lhs, %rhs, %init)
        : (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func @empty(
// CHECK: return %arg0 : tensor<8xf32>
// CHECK-NOT: linalg.generic
module attributes {transform.target_tag = "payload"} {
  func.func @empty(%arg: tensor<8xf32>) -> tensor<8xf32> {
    return %arg : tensor<8xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match attributes {not_a_vector_function} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func @casts(
// CHECK: arith.truncf %{{.*}} {enable_saturate = true, round_mode = #hfusion.round_mode<rint>} : f32 to f16
// CHECK: arith.extui %{{.*}} {round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<ui2ui>} : i8 to i16
// CHECK-NOT: hivm.hir.vcast
// CHECK-NOT: hfusion.cast
module attributes {
  hacc.target = #hacc.target<"Ascend960DT_968B7">,
  transform.target_tag = "payload"
} {
  func.func @casts(%input: tensor<8xf32>, %half: tensor<8xf16>,
                   %integer: tensor<8xi8>, %wide: tensor<8xi16>)
      -> (tensor<8xf16>, tensor<8xi16>) {
    %0 = hivm.hir.vcast {enable_saturate = true, enable_overflow = true}
        ins(%input : tensor<8xf32>) outs(%half : tensor<8xf16>)
        -> tensor<8xf16>
    %1 = hivm.hir.vcast {hivm.unsigned_mode = #hivm.unsigned_mode<ui2ui>}
        ins(%integer : tensor<8xi8>) outs(%wide : tensor<8xi16>)
        round_mode = <rint> cast = <cast_unsigned> -> tensor<8xi16>
    return %0, %1 : tensor<8xf16>, tensor<8xi16>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----


// CHECK-LABEL: func.func @shifts
// CHECK-DAG: arith.shli
// CHECK-DAG: arith.shrsi
// CHECK-NOT: hivm.hir.v
module attributes {transform.target_tag = "payload"} {
  func.func @shifts(%lhs: tensor<8xi32>, %rhs: tensor<8xi32>,
                    %init: tensor<8xi32>) -> tensor<8xi32> {
    %0 = hivm.hir.vshl ins(%lhs, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%init : tensor<8xi32>) -> tensor<8xi32>
    %1 = hivm.hir.vshr ins(%0, %rhs : tensor<8xi32>, tensor<8xi32>)
        outs(%init : tensor<8xi32>) round : false -> tensor<8xi32>
    return %1 : tensor<8xi32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op
  ) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %lifted = transform.hivm.lift_to_linalg %funcs
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
