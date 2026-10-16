// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file -verify-diagnostics | FileCheck %s
//
// Strength reduction of integer div/div_unsigned with power-of-two constant divisors into shifts, on Ascend950 only.
// scope: element types i8/i16/i32 (i64 excluded, see below);
//   i64 excluded; Ascend310B4 and no-target cases must not match.
//   divui(x, 2^k) ->  x >>u k
//       div(x, 2^k)  -> (x + corr) >>s k, corr = (x >>s (w-1)) & (2^k-1)

//===== positive (A5) =====

// -----

// divui(x, 8) -> x >>u 3
// CHECK-LABEL: func.func @test_divui_pow2_i32
// CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>} ins({{.*}}, %[[c3]] : tensor<16xi32>, i32)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_divui_pow2_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// div(x, 8) -> (x + corr) >>s 3, corr = (x >>s 31) & 7
// CHECK-LABEL: func.func @test_div_pow2_i32
// CHECK-DAG:   %[[c31:.*]] = arith.constant 31 : i32
// CHECK-DAG:   %[[c7:.*]] = arith.constant 7 : i32
// CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : i32
// CHECK:       %[[shr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %[[c31]] : tensor<16xi32>, i32)
// CHECK:       %[[corr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[shr]], %[[c7]] : tensor<16xi32>, i32)
// CHECK:       %[[sum:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %[[corr]] : tensor<16xi32>, tensor<16xi32>)
// CHECK:       %[[res:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[sum]], %[[c3]] : tensor<16xi32>, i32)
// CHECK:       return %[[res]] : tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_pow2_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// divui(x, 2^31) i32 unsigned upper bound (y = 2^(w-1), bit pattern -2147483648) -> x >>u 31
// CHECK-LABEL: func.func @test_divui_pow2_u31_i32
// CHECK-DAG:   %[[c31:.*]] = arith.constant 31 : i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrui>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_divui_pow2_u31_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant -2147483648 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// div(x, 4) i16 -> corr = (x >>s 15) & 3
// CHECK-LABEL: func.func @test_div_pow2_i16
// CHECK-DAG:   %[[c15:.*]] = arith.constant 15 : i16
// CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : i16
// CHECK-DAG:   %[[c2:.*]] = arith.constant 2 : i16
// CHECK:       %[[shr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %[[c15]] : tensor<16xi16>, i16)
// CHECK:       %[[corr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[shr]], %[[c3]] : tensor<16xi16>, i16)
// CHECK:       %[[sum:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %[[corr]] : tensor<16xi16>, tensor<16xi16>)
// CHECK:       %[[res:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[sum]], %[[c2]] : tensor<16xi16>, i16)
// CHECK:       return %[[res]] : tensor<16xi16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_pow2_i16(%arg0: tensor<16xi16>) -> tensor<16xi16> {
      %cst = arith.constant 4 : i16
      %0 = tensor.empty() : tensor<16xi16>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi16>, i16) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
      return %1 : tensor<16xi16>
  }
}

// -----

// linalg.fill constant source: div(x, fill(8))
// CHECK-LABEL: func.func @test_div_pow2_fill_i32
// CHECK-DAG:   %[[c31:.*]] = arith.constant 31 : i32
// CHECK-DAG:   %[[c7:.*]] = arith.constant 7 : i32
// CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : i32
// CHECK:       %[[shr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %[[c31]] : tensor<16xi32>, i32)
// CHECK:       %[[corr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[shr]], %[[c7]] : tensor<16xi32>, i32)
// CHECK:       %[[sum:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %[[corr]] : tensor<16xi32>, tensor<16xi32>)
// CHECK:       %[[res:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[sum]], %[[c3]] : tensor<16xi32>, i32)
// CHECK:       return %[[res]] : tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_pow2_fill_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %fill = linalg.fill ins(%cst : i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      %1 = tensor.empty() : tensor<16xi32>
      %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %fill : tensor<16xi32>, tensor<16xi32>) outs(%1 : tensor<16xi32>) -> tensor<16xi32>
      return %2 : tensor<16xi32>
  }
}

// -----

// k=0: div(x, 1) -> x (identity, no leftover ops)
// CHECK-LABEL: func.func @test_div_k0_i32
// CHECK-NOT:   binary_fn<div>
// CHECK-NOT:   binary_fn<mul>
// CHECK:       return %arg0 : tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_k0_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 1 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// k=0: divui(x, 1) -> x (identity)
// CHECK-LABEL: func.func @test_divui_k0_i32
// CHECK-NOT:   binary_fn<div>
// CHECK-NOT:   binary_fn<mul>
// CHECK:       return %arg0 : tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_divui_k0_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 1 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// i8: products are widened to i16 by ShiftI8ToI16; the CHECK only pins op presence
// CHECK-LABEL: func.func @test_div_pow2_i8
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>}
// CHECK-NOT:   binary_fn<div>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_pow2_i8(%arg0: tensor<16xi8>) -> tensor<16xi8> {
      %cst = arith.constant 8 : i8
      %0 = tensor.empty() : tensor<16xi8>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi8>, i8) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
      return %1 : tensor<16xi8>
  }
}

//===== negative (A5): must not be rewritten =====

// -----

// non-pow2 divisor 7: no shifts (general constant division is out of scope)
// CHECK-LABEL: func.func @test_div_nonpow2_i32
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_nonpow2_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 7 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// negative divisor -8: no shifts
// CHECK-LABEL: func.func @test_div_negdiv_i32
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_negdiv_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant -8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// zero divisor: no shifts
// CHECK-LABEL: func.func @test_div_zero_i32
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_zero_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 0 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// dynamic divisor: no shifts
// CHECK-LABEL: func.func @test_div_dyn_i32
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_dyn_i32(%arg0: tensor<16xi32>, %arg1: i32) -> tensor<16xi32> {
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// signed y = 2^(w-1) (INT_MIN bit pattern -2147483648): no shifts (counterpart of the divui unsigned upper bound)
// CHECK-LABEL: func.func @test_div_intmin_i32
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_intmin_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant -2147483648 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

//===== i64 excluded =====

// -----

// i64 does not match (width whitelist; products lower to scalar loops/libcalls)
// CHECK-LABEL: func.func @test_div_i64_outofscope
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_i64_outofscope(%arg0: tensor<16xi64>) -> tensor<16xi64> {
      %cst = arith.constant 8 : i64
      %0 = tensor.empty() : tensor<16xi64>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi64>, i64) outs(%0 : tensor<16xi64>) -> tensor<16xi64>
      return %1 : tensor<16xi64>
  }
}

//===== gating (no target attribute) =====

// -----

// no target attribute: the gate does not pass, no shifts (the f32 reciprocal path has none either)
// CHECK-LABEL: func.func @test_div_noattr_gating
// CHECK-NOT:   binary_fn<shr
// CHECK:       return
func.func @test_div_noattr_gating(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %cst = arith.constant 8 : i32
    %0 = tensor.empty() : tensor<16xi32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
    return %1 : tensor<16xi32>
}

// -----

// Ascend310B4 is regbased but not 950: the patterns are loaded and rejected by the archIsAscend950 gate
// CHECK-LABEL: func.func @test_div_a3_keep
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_div_a3_keep(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

//===== scalar dividend =====
//
// The two input type variables of elemwise_binary are independent, so a
// scalar dividend against a splat tensor divisor is legal IR. This degenerate foldable form must not be matched:
// otherwise building an empty dst from a non-shaped operand would crash (createEmptyOp ->
// k=0 would replace the op with a scalar, mismatching the tensor result;

// -----

// scalar dividend + k=3: must not be rewritten
// CHECK-LABEL: func.func @test_div_scalar_lhs_k3
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_scalar_lhs_k3(%arg0: i32) -> tensor<16xi32> {
      %cst = arith.constant dense<8> : tensor<16xi32>
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : i32, tensor<16xi32>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// scalar dividend + k=0: must not be identity-replaced (result is a tensor, %arg0 is a scalar)
// CHECK-LABEL: func.func @test_div_scalar_lhs_k0
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_scalar_lhs_k0(%arg0: i32) -> tensor<16xi32> {
      %cst = arith.constant dense<1> : tensor<16xi32>
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : i32, tensor<16xi32>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// scalar dividend + divui k=3: must not be rewritten (unsigned side)
// CHECK-LABEL: func.func @test_divui_scalar_lhs_k3
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_divui_scalar_lhs_k3(%arg0: i32) -> tensor<16xi32> {
      %cst = arith.constant dense<8> : tensor<16xi32>
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %cst : i32, tensor<16xi32>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

//===== type boundaries =====

// -----

// i32 dividend with an i64 divisor: must not be rewritten (wrong-width shift).
// CHECK-LABEL: func.func @test_div_rhs_width_mismatch
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_rhs_width_mismatch(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i64
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi32>, i64) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// i4 is not covered by the hardware normalization rules: must not be rewritten.
// CHECK-LABEL: func.func @test_div_unsupported_i4
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_unsupported_i4(%arg0: tensor<16xi4>) -> tensor<16xi4> {
      %cst = arith.constant 2 : i4
      %0 = tensor.empty() : tensor<16xi4>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi4>, i4) outs(%0 : tensor<16xi4>) -> tensor<16xi4>
      return %1 : tensor<16xi4>
  }
}

// -----

// CHECK-LABEL: func.func @test_div_narrow_rhs_width_mismatch
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<shr
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_narrow_rhs_width_mismatch(%arg0: tensor<16xi16>) -> tensor<16xi16> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi16>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %cst : tensor<16xi16>, i32) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
      return %1 : tensor<16xi16>
  }
}

// -----

// CHECK-LABEL: func.func @test_div_unsigned_rhs_large_shift
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
// CHECK-NOT:   binary_fn<shr
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_unsigned_rhs_large_shift(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 1099511627776 : i64
      %0 = tensor.empty() : tensor<16xi32>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %cst : tensor<16xi32>, i64) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_div_unsigned_unsupported_i4
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>}
// CHECK-NOT:   binary_fn<shr
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_unsigned_unsupported_i4(%arg0: tensor<16xi4>) -> tensor<16xi4> {
      %cst = arith.constant 2 : i4
      %0 = tensor.empty() : tensor<16xi4>
      %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %cst : tensor<16xi4>, i4) outs(%0 : tensor<16xi4>) -> tensor<16xi4>
      return %1 : tensor<16xi4>
  }
}

// -----

// Narrowing result (i32 dividend, i16 result): the original op casts the
// inputs to the result element type first (i32 65535 truncates to i16 -1),
// which the correction at the input width cannot model. Must not match.
// CHECK-LABEL: func.func @test_div_narrow_result_i32_i16
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK-NOT:   binary_fn<vand>
// CHECK-NOT:   binary_fn<shrsi>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_narrow_result_i32_i16(%x: tensor<16xi32>) -> tensor<16xi16> {
      %eight = arith.constant 8 : i32
      %out = tensor.empty() : tensor<16xi16>
      %r = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
        ins(%x, %eight : tensor<16xi32>, i32)
        outs(%out : tensor<16xi16>) -> tensor<16xi16>
      return %r : tensor<16xi16>
  }
}

// -----

// cast_unsigned promotion (i16 dividend promoted to 65535 by the op's cast
// attribute before dividing): the signed correction cannot model this.
// Must not match.
// CHECK-LABEL: func.func @test_div_cast_unsigned_promotion
// CHECK:       binary_fn<div>
// CHECK-NOT:   binary_fn<vand>
// CHECK-NOT:   binary_fn<shrsi>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_cast_unsigned_promotion(%x: tensor<16xi16>) -> tensor<16xi32> {
      %eight = arith.constant 8 : i16
      %out = tensor.empty() : tensor<16xi32>
      %r = linalg.elemwise_binary
        {fun = #linalg.binary_fn<div>, cast = #linalg.type_fn<cast_unsigned>}
        ins(%x, %eight : tensor<16xi16>, i16)
        outs(%out : tensor<16xi32>) -> tensor<16xi32>
      return %r : tensor<16xi32>
  }
}

// cast_unsigned must block the rewrite even when input and result widths
// match: the source operation's cast mode is part of its semantics.
// CHECK-LABEL: func.func @test_div_cast_unsigned_same_type
// CHECK:       binary_fn<div>
// CHECK-NOT:   binary_fn<shrsi>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_div_cast_unsigned_same_type(%x: tensor<16xi16>) -> tensor<16xi16> {
      %eight = arith.constant 8 : i16
      %out = tensor.empty() : tensor<16xi16>
      %r = linalg.elemwise_binary
        {fun = #linalg.binary_fn<div>, cast = #linalg.type_fn<cast_unsigned>}
        ins(%x, %eight : tensor<16xi16>, i16)
        outs(%out : tensor<16xi16>) -> tensor<16xi16>
      return %r : tensor<16xi16>
  }
}
