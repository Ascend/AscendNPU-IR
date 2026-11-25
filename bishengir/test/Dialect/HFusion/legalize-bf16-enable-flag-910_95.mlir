// RUN: bishengir-opt -hfusion-legalize-bf16 %s -split-input-file | FileCheck %s --check-prefix=DISABLED
// RUN: bishengir-opt -hfusion-legalize-bf16=enable-bf16 %s -split-input-file | FileCheck %s --check-prefix=ENABLED
//
// Without the enable-bf16 option the legalization keeps rewriting the
// elementwise BF16 arithmetic to F32 (default behavior). With the option
// enabled the ops keep their BF16 element types so the native Ascend 950
// vector instructions are used.

// -----

// DISABLED-LABEL: func.func @test_elemwise_max_bf16
// DISABLED: hfusion.cast
// DISABLED: hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
// DISABLED-SAME: tensor<8xf32>, tensor<8xf32>
// DISABLED: hfusion.cast
// ENABLED-LABEL: func.func @test_elemwise_max_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
// ENABLED-SAME: tensor<8xbf16>, tensor<8xbf16>
// ENABLED-SAME: tensor<8xbf16>) -> tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_elemwise_max_bf16(
      %lhs: tensor<8xbf16>, %rhs: tensor<8xbf16>,
      %init: tensor<8xbf16>) -> tensor<8xbf16> {
    %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>}
        ins(%lhs, %rhs : tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xbf16>) -> tensor<8xbf16>
    return %result : tensor<8xbf16>
  }
}

// -----

// DISABLED-LABEL: func.func @test_elemwise_min_bf16
// DISABLED: hfusion.cast
// DISABLED: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
// DISABLED-SAME: tensor<8xf32>, tensor<8xf32>
// DISABLED: hfusion.cast
// ENABLED-LABEL: func.func @test_elemwise_min_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
// ENABLED-SAME: tensor<8xbf16>, tensor<8xbf16>
// ENABLED-SAME: tensor<8xbf16>) -> tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_elemwise_min_bf16(
      %lhs: tensor<8xbf16>, %rhs: tensor<8xbf16>,
      %init: tensor<8xbf16>) -> tensor<8xbf16> {
    %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>}
        ins(%lhs, %rhs : tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xbf16>) -> tensor<8xbf16>
    return %result : tensor<8xbf16>
  }
}

// -----

// The linalg min/max variants share the same native BF16 support; the signed
// variant is exercised here as the representative of the linalg fun family.
// DISABLED-LABEL: func.func @test_linalg_max_signed_bf16
// DISABLED: hfusion.cast
// DISABLED: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
// DISABLED-SAME: tensor<8xf32>, tensor<8xf32>
// DISABLED: hfusion.cast
// ENABLED-LABEL: func.func @test_linalg_max_signed_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
// ENABLED-SAME: tensor<8xbf16>, tensor<8xbf16>
// ENABLED-SAME: tensor<8xbf16>) -> tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_linalg_max_signed_bf16(
      %lhs: tensor<8xbf16>, %rhs: tensor<8xbf16>,
      %init: tensor<8xbf16>) -> tensor<8xbf16> {
    %result = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
        ins(%lhs, %rhs : tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xbf16>) -> tensor<8xbf16>
    return %result : tensor<8xbf16>
  }
}

// -----

// DISABLED-LABEL: func.func @test_compare_bf16
// DISABLED: hfusion.cast
// DISABLED: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// DISABLED-SAME: tensor<8xf32>, tensor<8xf32>
// ENABLED-LABEL: func.func @test_compare_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// ENABLED-SAME: tensor<8xbf16>, tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_compare_bf16(
      %lhs: tensor<8xbf16>, %rhs: tensor<8xbf16>,
      %init: tensor<8xi1>) -> tensor<8xi1> {
    %result = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
        ins(%lhs, %rhs : tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xi1>) -> tensor<8xi1>
    return %result : tensor<8xi1>
  }
}

// -----

// Vector selects keep BF16; rank-0 and singleton selects stay on the F32
// scalar path regardless of the option (see select_rank_zero /
// select_singleton in legalize-bf16-910_95.mlir).
// DISABLED-LABEL: func.func @test_select_bf16
// DISABLED: hfusion.cast
// DISABLED: hfusion.select
// DISABLED-SAME: tensor<8xi1>, tensor<8xf32>, tensor<8xf32>
// DISABLED: hfusion.cast
// ENABLED-LABEL: func.func @test_select_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: hfusion.select
// ENABLED-SAME: tensor<8xi1>, tensor<8xbf16>, tensor<8xbf16>
// ENABLED-SAME: tensor<8xbf16>) -> tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_select_bf16(
      %cond: tensor<8xi1>, %lhs: tensor<8xbf16>, %rhs: tensor<8xbf16>,
      %init: tensor<8xbf16>) -> tensor<8xbf16> {
    %result = hfusion.select
        ins(%cond, %lhs, %rhs : tensor<8xi1>, tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xbf16>) -> tensor<8xbf16>
    return %result : tensor<8xbf16>
  }
}

// -----

// DISABLED-LABEL: func.func @test_elemwise_add_bf16
// DISABLED: hfusion.cast
// DISABLED: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// DISABLED-SAME: tensor<8xf32>, tensor<8xf32>
// DISABLED: hfusion.cast
// ENABLED-LABEL: func.func @test_elemwise_add_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// ENABLED-SAME: tensor<8xbf16>, tensor<8xbf16>
// ENABLED-SAME: tensor<8xbf16>) -> tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_elemwise_add_bf16(
      %lhs: tensor<8xbf16>, %rhs: tensor<8xbf16>,
      %init: tensor<8xbf16>) -> tensor<8xbf16> {
    %result = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
        ins(%lhs, %rhs : tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xbf16>) -> tensor<8xbf16>
    return %result : tensor<8xbf16>
  }
}

// -----

// DISABLED-LABEL: func.func @test_elemwise_sub_bf16
// DISABLED: hfusion.cast
// DISABLED: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// DISABLED-SAME: tensor<8xf32>, tensor<8xf32>
// DISABLED: hfusion.cast
// ENABLED-LABEL: func.func @test_elemwise_sub_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// ENABLED-SAME: tensor<8xbf16>, tensor<8xbf16>
// ENABLED-SAME: tensor<8xbf16>) -> tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_elemwise_sub_bf16(
      %lhs: tensor<8xbf16>, %rhs: tensor<8xbf16>,
      %init: tensor<8xbf16>) -> tensor<8xbf16> {
    %result = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
        ins(%lhs, %rhs : tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xbf16>) -> tensor<8xbf16>
    return %result : tensor<8xbf16>
  }
}

// -----

// DISABLED-LABEL: func.func @test_elemwise_mul_bf16
// DISABLED: hfusion.cast
// DISABLED: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// DISABLED-SAME: tensor<8xf32>, tensor<8xf32>
// DISABLED: hfusion.cast
// ENABLED-LABEL: func.func @test_elemwise_mul_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
// ENABLED-SAME: tensor<8xbf16>, tensor<8xbf16>
// ENABLED-SAME: tensor<8xbf16>) -> tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_elemwise_mul_bf16(
      %lhs: tensor<8xbf16>, %rhs: tensor<8xbf16>,
      %init: tensor<8xbf16>) -> tensor<8xbf16> {
    %result = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
        ins(%lhs, %rhs : tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xbf16>) -> tensor<8xbf16>
    return %result : tensor<8xbf16>
  }
}

// -----

// Ascend 950 lowers fused multiply-add to the native vmula instruction for
// BF16 operands; the ternary op must keep its BF16 element type.
// DISABLED-LABEL: func.func @test_elemwise_ternary_fma_bf16
// DISABLED: hfusion.cast
// DISABLED: hfusion.elemwise_ternary {fun = #hfusion.ternary_fn<fma>}
// DISABLED-SAME: tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
// DISABLED: hfusion.cast
// ENABLED-LABEL: func.func @test_elemwise_ternary_fma_bf16
// ENABLED-NOT: hfusion.cast
// ENABLED: hfusion.elemwise_ternary {fun = #hfusion.ternary_fn<fma>}
// ENABLED-SAME: tensor<8xbf16>, tensor<8xbf16>, tensor<8xbf16>
// ENABLED-SAME: tensor<8xbf16>) -> tensor<8xbf16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_957b">} {
  func.func @test_elemwise_ternary_fma_bf16(
      %x: tensor<8xbf16>, %y: tensor<8xbf16>, %z: tensor<8xbf16>,
      %init: tensor<8xbf16>) -> tensor<8xbf16> {
    %result = hfusion.elemwise_ternary {fun = #hfusion.ternary_fn<fma>}
        ins(%x, %y, %z : tensor<8xbf16>, tensor<8xbf16>, tensor<8xbf16>)
        outs(%init : tensor<8xbf16>) -> tensor<8xbf16>
    return %result : tensor<8xbf16>
  }
}
