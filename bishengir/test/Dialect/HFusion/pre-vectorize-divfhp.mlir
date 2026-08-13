// RUN: bishengir-opt %s -hfusion-pre-vectorization-fusion | FileCheck %s --check-prefix=DIRECT
// RUN: bishengir-opt %s --hfusion-normalize-ops="use-regbase=true" \
// RUN:   -hfusion-pre-vectorization-fusion | FileCheck %s --check-prefix=MOD

// Verify that generalizing an HFusion high-precision division preserves its
// precision semantics in the scalar body of the resulting linalg.generic.

// DIRECT-LABEL: func.func @preserve_divfhp
// DIRECT: linalg.generic
// DIRECT-NOT: arith.divf
// DIRECT: mathExt.divfhp
// DIRECT-NOT: arith.divf
// DIRECT: return
func.func @preserve_divfhp(%lhs: tensor<512xf32>, %rhs: tensor<512xf32>)
    -> tensor<512xf32> {
  %empty = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>}
      ins(%lhs, %rhs : tensor<512xf32>, tensor<512xf32>)
      outs(%empty : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}

// Mixed input types are converted to the output element type before applying
// the binary function. Verify that divfhp accepts this case when that common
// type is floating point.

// DIRECT-LABEL: func.func @preserve_mixed_divfhp_to_f16
// DIRECT: arith.sitofp {{.*}} : i32 to f16
// DIRECT: mathExt.divfhp {{.*}} : f16
func.func @preserve_mixed_divfhp_to_f16(
    %lhs: tensor<512xf16>, %rhs: tensor<512xi32>) -> tensor<512xf16> {
  %empty = tensor.empty() : tensor<512xf16>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<divfhp>}
      ins(%lhs, %rhs : tensor<512xf16>, tensor<512xi32>)
      outs(%empty : tensor<512xf16>) -> tensor<512xf16>
  return %result : tensor<512xf16>
}

// Verify the complete mod normalization path that exposed the precision bug:
// mod -> divfhp/trunc/mul/sub -> linalg.generic with mathExt.divfhp.

// MOD-LABEL: func.func @preserve_normalized_mod_divfhp
// MOD: linalg.generic
// MOD-NOT: arith.divf
// MOD: mathExt.divfhp
// MOD: math.round
// MOD-NOT: arith.divf
// MOD: return
func.func @preserve_normalized_mod_divfhp(
    %lhs: tensor<512xf32>, %rhs: tensor<512xf32>) -> tensor<512xf32> {
  %empty = tensor.empty() : tensor<512xf32>
  %result = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%lhs, %rhs : tensor<512xf32>, tensor<512xf32>)
      outs(%empty : tensor<512xf32>) -> tensor<512xf32>
  return %result : tensor<512xf32>
}
