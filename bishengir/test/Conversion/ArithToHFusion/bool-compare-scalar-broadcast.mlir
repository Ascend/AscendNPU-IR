// RUN: bishengir-opt %s -convert-arith-to-hfusion -hfusion-normalize-ops=use-regbase=true | FileCheck %s
module attributes {hacc.target = #hacc.target<"Ascend950">} {
  // CHECK-LABEL: func.func @bool_ge_scalar
  // CHECK: arith.extui {{.*}} : i1 to i8
  // CHECK-NOT: arith.extsi {{.*}} : i1 to i8
  // CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<vge>}
  func.func @bool_ge_scalar(%lhs: tensor<8xi1>, %rhs: i1) -> tensor<8xi1> {
    %empty = tensor.empty() : tensor<8xi1>
    %broadcast = linalg.fill ins(%rhs : i1) outs(%empty : tensor<8xi1>) -> tensor<8xi1>
    %result = arith.cmpi uge, %lhs, %broadcast : tensor<8xi1>
    return %result : tensor<8xi1>
  }
}
