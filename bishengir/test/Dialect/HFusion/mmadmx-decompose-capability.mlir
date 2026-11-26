// RUN: bishengir-opt --hfusion-decompose %s -split-input-file -verify-diagnostics | FileCheck %s

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

// On C310, 32 packed FP4 bytes meet the logical K=64 alignment requirement.
// The same storage K for FP8 needs software emulation.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @c310_packed_fp4_logical_k64
  // CHECK-NEXT: %[[MX:.*]] = hfusion.matmul_mx {lhsFormat = 3 : i32, rhsFormat = 3 : i32}
  // CHECK-SAME: : tensor<16x32xi8>, tensor<32x32xi8>, tensor<16x2xi8>, tensor<32x2xi8>)
  // CHECK-NEXT: return %[[MX]] : tensor<16x32xf32>
  func.func @c310_packed_fp4_logical_k64(
      %a: tensor<16x32xi8>, %b: tensor<32x32xi8>,
      %sa: tensor<16x2xi8>, %sb: tensor<32x2xi8>, %acc: tensor<16x32xf32>)
      -> tensor<16x32xf32> {
    %0 = hfusion.matmul_mx {lhsFormat = 3 : i32, rhsFormat = 3 : i32}
        ins(%a, %b, %sa, %sb : tensor<16x32xi8>, tensor<32x32xi8>,
            tensor<16x2xi8>, tensor<32x2xi8>)
        outs(%acc : tensor<16x32xf32>) -> tensor<16x32xf32>
    return %0 : tensor<16x32xf32>
  }

  // CHECK-LABEL: func.func @c310_fp8_k32
  // CHECK-SAME: %[[ACC:[^ ]+]]: tensor<16x32xf32>
  // CHECK-NOT: hfusion.matmul_mx
  // CHECK: %[[MM:.*]] = linalg.matmul
  // CHECK-SAME: : tensor<16x32xf16>, tensor<32x32xf16>)
  // CHECK-SAME: outs(%[[ACC]] : tensor<16x32xf32>)
  // CHECK-NEXT: return %[[MM]] : tensor<16x32xf32>
  func.func @c310_fp8_k32(
      %a: tensor<16x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>,
      %sa: tensor<16x1xi8>, %sb: tensor<32x1xi8>, %acc: tensor<16x32xf32>)
      -> tensor<16x32xf32> {
    %0 = hfusion.matmul_mx
        ins(%a, %b, %sa, %sb : tensor<16x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>,
            tensor<16x1xi8>, tensor<32x1xi8>)
        outs(%acc : tensor<16x32xf32>) -> tensor<16x32xf32>
    return %0 : tensor<16x32xf32>
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

// -----

func.func @missing_rhs_format(%ab: tensor<32x32xi8>, %scale: tensor<32x2xi8>,
                              %acc: tensor<32x32xf32>) {
  // expected-error@+1 {{lhsFormat and rhsFormat must be specified together}}
  %0 = hfusion.matmul_mx {lhsFormat = 3 : i32}
      ins(%ab, %ab, %scale, %scale : tensor<32x32xi8>, tensor<32x32xi8>,
          tensor<32x2xi8>, tensor<32x2xi8>)
      outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  return
}

// -----

func.func @mixed_fp4_fp8(%a: tensor<32x32xi8>, %b: tensor<32x32xf8E4M3FN>,
                        %sa: tensor<32x2xi8>, %sb: tensor<32x1xi8>,
                        %acc: tensor<32x32xf32>) {
  // expected-error@+1 {{mixed FP4 and non-FP4 formats are unsupported}}
  %0 = hfusion.matmul_mx {lhsFormat = 3 : i32, rhsFormat = 2 : i32}
      ins(%a, %b, %sa, %sb : tensor<32x32xi8>, tensor<32x32xf8E4M3FN>,
          tensor<32x2xi8>, tensor<32x1xi8>)
      outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  return
}

// -----

func.func @missing_integer_formats(%ab: tensor<32x32xi8>, %scale: tensor<32x2xi8>,
                                   %acc: tensor<32x32xf32>) {
  // expected-error@+1 {{format attributes are required for 8-bit integer inputs}}
  %0 = hfusion.matmul_mx
      ins(%ab, %ab, %scale, %scale : tensor<32x32xi8>, tensor<32x32xi8>,
          tensor<32x2xi8>, tensor<32x2xi8>)
      outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  return
}
