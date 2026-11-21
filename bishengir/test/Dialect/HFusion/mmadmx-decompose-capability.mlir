// RUN: bishengir-opt --hfusion-decompose %s -split-input-file | FileCheck %s

// C310, K multiple of 64, unformatted f8E5M2 pair: the catalogue says Native,
// so decompose leaves the op in place.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @native_c310_keeps_matmul_mx
  // CHECK: hfusion.matmul_mx
  // CHECK-NOT: hfusion.elemwise_binary
  func.func @native_c310_keeps_matmul_mx(
      %a: tensor<16x64xf8E5M2>, %b: tensor<64x16xf8E5M2>,
      %sa: tensor<1x1xi8>, %sb: tensor<1x1xi8>, %acc: tensor<16x16xf32>)
      -> tensor<16x16xf32> {
    %0 = hfusion.matmul_mx
        ins(%a, %b, %sa, %sb : tensor<16x64xf8E5M2>, tensor<64x16xf8E5M2>,
            tensor<1x1xi8>, tensor<1x1xi8>)
        outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}

// -----

// C220 has no MMADMX rows. K=32 takes the software-emulation path.
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  // CHECK-LABEL: func.func @c220_k32_emulates_matmul_mx
  // CHECK-NOT: hfusion.matmul_mx
  // CHECK: hfusion.elemwise_binary
  func.func @c220_k32_emulates_matmul_mx(
      %a: tensor<16x32xf8E5M2>, %b: tensor<32x16xf8E5M2>,
      %sa: tensor<16x1xi8>, %sb: tensor<16x1xi8>, %acc: tensor<16x16xf32>)
      -> tensor<16x16xf32> {
    %0 = hfusion.matmul_mx
        ins(%a, %b, %sa, %sb : tensor<16x32xf8E5M2>, tensor<32x16xf8E5M2>,
            tensor<16x1xi8>, tensor<16x1xi8>)
        outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %0 : tensor<16x16xf32>
  }
}
