// RUN: split-file %s %t
// RUN: bishengir-opt %t/valid.mlir --split-input-file \
// RUN:   --hivm-vectorize-ops --canonicalize --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-VALID
// RUN: bishengir-opt %t/invalid.mlir --split-input-file --verify-diagnostics

//--- valid.mlir

// Regbase enables its software FP8 cast set and FP8 vector operations.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
// CHECK-VALID-LABEL: func.func @regbase_cast_to_fp8(
// CHECK-VALID: arith.truncf {{.*}} : vector<8x8xf32> to vector<8x8xf8E5M2>
func.func @regbase_cast_to_fp8(%input: tensor<8x8xf32>)
    -> tensor<8x8xf8E5M2> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf8E5M2>
  %result = hivm.hir.vcast ins(%input : tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf8E5M2>) round_mode = <rint>
      cast = <cast_signed> -> tensor<8x8xf8E5M2>
  return %result : tensor<8x8xf8E5M2>
}

// CHECK-VALID-LABEL: func.func @regbase_cast_from_fp8(
// CHECK-VALID: arith.extf {{.*}} : vector<8x8xf8E5M2> to vector<8x8xf32>
func.func @regbase_cast_from_fp8(%input: tensor<8x8xf8E5M2>)
    -> tensor<8x8xf32> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = hivm.hir.vcast ins(%input : tensor<8x8xf8E5M2>)
      outs(%empty : tensor<8x8xf32>) round_mode = <rint>
      cast = <cast_signed> -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-VALID-LABEL: func.func @regbase_broadcast_fp8(
// CHECK-VALID: vector.broadcast {{.*}} : f8E5M2 to vector<16x16xf8E5M2>
func.func @regbase_broadcast_fp8(%input: f8E5M2)
    -> tensor<16x16xf8E5M2> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<16x16xf8E5M2>
  %result = hivm.hir.vbrc ins(%input : f8E5M2)
      outs(%empty : tensor<16x16xf8E5M2>) -> tensor<16x16xf8E5M2>
  return %result : tensor<16x16xf8E5M2>
}
}

// -----

// Diff: target capability mismatch
// With no hacc.target, VBrcOp skips its FP8 hardware check. This is accepted
// even though other target helpers treat a missing target as legacy 910B.
module {
// CHECK-VALID-LABEL: func.func @no_target_broadcast_fp8(
// CHECK-VALID: vector.broadcast {{.*}} : f8E5M2 to vector<16x16xf8E5M2>
func.func @no_target_broadcast_fp8(%input: f8E5M2)
    -> tensor<16x16xf8E5M2> attributes {hivm.vector_function} {
  %empty = tensor.empty() : tensor<16x16xf8E5M2>
  %result = hivm.hir.vbrc ins(%input : f8E5M2)
      outs(%empty : tensor<16x16xf8E5M2>) -> tensor<16x16xf8E5M2>
  return %result : tensor<16x16xf8E5M2>
}
}

//--- invalid.mlir

// Diff: target capability mismatch
// Membase does not include float-to-FP8 in its software cast set.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @membase_cast_to_fp8(%input: tensor<8x8xf32>)
    -> tensor<8x8xf8E5M2> {
  %empty = tensor.empty() : tensor<8x8xf8E5M2>
  // expected-error@+1 {{currently don't support cast float_to_float8_e5m2_t_rintmode}}
  %result = hivm.hir.vcast ins(%input : tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf8E5M2>) round_mode = <rint>
      cast = <cast_signed> -> tensor<8x8xf8E5M2>
  return %result : tensor<8x8xf8E5M2>
}
}

// -----

// Diff: target capability mismatch
// Membase also excludes FP8-to-f32 from its software cast set.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @membase_cast_from_fp8_to_f32(%input: tensor<8x8xf8E5M2>)
    -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  // expected-error@+1 {{currently don't support cast float8_e5m2_t_to_float_rintmode}}
  %result = hivm.hir.vcast ins(%input : tensor<8x8xf8E5M2>)
      outs(%empty : tensor<8x8xf32>) round_mode = <rint>
      cast = <cast_signed> -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}
}

// -----

// Diff: target capability mismatch
// Explicit membase targets reject FP8 broadcast.
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @membase_broadcast_fp8(%input: f8E5M2)
    -> tensor<16x16xf8E5M2> {
  %empty = tensor.empty() : tensor<16x16xf8E5M2>
  // expected-error@+1 {{Current hardware doesn't support fp8 type}}
  %result = hivm.hir.vbrc ins(%input : f8E5M2)
      outs(%empty : tensor<16x16xf8E5M2>) -> tensor<16x16xf8E5M2>
  return %result : tensor<16x16xf8E5M2>
}
}

// -----

// Diff: target capability mismatch
// The cast is absent from both target-specific software sets and the common
// hardware set.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @regbase_cast_from_fp8(%input: tensor<8x8xf8E5M2>)
    -> tensor<8x8xf16> {
  %empty = tensor.empty() : tensor<8x8xf16>
  // expected-error@+1 {{currently don't support cast float8_e5m2_t_to_half_rintmode}}
  %result = hivm.hir.vcast ins(%input : tensor<8x8xf8E5M2>)
      outs(%empty : tensor<8x8xf16>) round_mode = <rint>
      cast = <cast_signed> -> tensor<8x8xf16>
  return %result : tensor<8x8xf16>
}
}

// -----

// Diff: target capability mismatch
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
func.func @membase_cast_from_fp8(%input: tensor<8x8xf8E5M2>)
    -> tensor<8x8xf16> {
  %empty = tensor.empty() : tensor<8x8xf16>
  // expected-error@+1 {{currently don't support cast float8_e5m2_t_to_half_rintmode}}
  %result = hivm.hir.vcast ins(%input : tensor<8x8xf8E5M2>)
      outs(%empty : tensor<8x8xf16>) round_mode = <rint>
      cast = <cast_signed> -> tensor<8x8xf16>
  return %result : tensor<8x8xf16>
}
}

// -----

// Diff: target capability mismatch
// VReduceOp's ODS constraint is target-independent, so regbase cannot enable
// FP8 reduction.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @regbase_reduce_fp8(%input: tensor<16x16xf8E4M3FN>,
    %init: tensor<16x1xf8E4M3FN>) -> tensor<16x1xf8E4M3FN> {
  // expected-error@+1 {{operand at idx 0 and 1 should have element type}}
  %result = hivm.hir.vreduce <max>
      ins(%input : tensor<16x16xf8E4M3FN>)
      outs(%init : tensor<16x1xf8E4M3FN>) reduce_dims = [1]
      -> tensor<16x1xf8E4M3FN>
  return %result : tensor<16x1xf8E4M3FN>
}
}

// -----

// Diff: target capability mismatch
// BF16 is rejected by the same target-independent ODS constraint.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
func.func @regbase_reduce_bf16(%input: tensor<8x16xbf16>,
    %init: tensor<8x1xbf16>) -> tensor<8x1xbf16> {
  // expected-error@+1 {{operand at idx 0 and 1 should have element type}}
  %result = hivm.hir.vreduce <max> ins(%input : tensor<8x16xbf16>)
      outs(%init : tensor<8x1xbf16>) reduce_dims = [1]
      -> tensor<8x1xbf16>
  return %result : tensor<8x1xbf16>
}
}
