// RUN: bishengir-opt --hfusion-normalize-ops="use-regbase=true" %s -split-input-file -verify-diagnostics | FileCheck %s
//
// Strength reduction of integer mod/modui with power-of-two constant divisors into bitwise ops, on Ascend950 only.
// scope: element types i8/i16/i32 (i64 excluded, see below);
//   i64 excluded; Ascend310B4 and no-target cases must not match.
//   mod(x, 2^k)  ->  ((x + corr) & (2^k-1)) - corr
//       modui(x, 2^k) -> x & (2^k-1)

//===== positive (A5) =====

// -----

// modui(x, 8) -> x & 7
// CHECK-LABEL: func.func @test_modui_pow2_i32
// CHECK-DAG:   %[[c7:.*]] = arith.constant 7 : i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins({{.*}}, %[[c7]] : tensor<16xi32>, i32)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_modui_pow2_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// splat constant source: modui(x, dense<16>) -> x & 15
// CHECK-LABEL: func.func @test_modui_pow2_splat_i32
// CHECK-DAG:   %[[c15:.*]] = arith.constant 15 : i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins({{.*}}, %[[c15]] : tensor<16xi32>, i32)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_modui_pow2_splat_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant dense<16> : tensor<16xi32>
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi32>, tensor<16xi32>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// linalg.fill constant source: mod(x, fill(8)) -> ((x+corr)&7)-corr
// CHECK-LABEL: func.func @test_mod_pow2_fill_i32
// CHECK-DAG:   %[[c31:.*]] = arith.constant 31 : i32
// CHECK-DAG:   %[[c7:.*]] = arith.constant 7 : i32
// CHECK:       %[[shr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %[[c31]] : tensor<16xi32>, i32)
// CHECK:       %[[corr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[shr]], %[[c7]] : tensor<16xi32>, i32)
// CHECK:       %[[sum:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %[[corr]] : tensor<16xi32>, tensor<16xi32>)
// CHECK:       %[[rem:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[sum]], %[[c7]] : tensor<16xi32>, i32)
// CHECK:       %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[rem]], %[[corr]] : tensor<16xi32>, tensor<16xi32>)
// CHECK:       return %[[res]] : tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_pow2_fill_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %fill = linalg.fill ins(%cst : i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      %1 = tensor.empty() : tensor<16xi32>
      %2 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %fill : tensor<16xi32>, tensor<16xi32>) outs(%1 : tensor<16xi32>) -> tensor<16xi32>
      return %2 : tensor<16xi32>
  }
}

// -----

// mod(x, 8) -> ((x + corr) & 7) - corr, corr = (x >>s 31) & 7
// CHECK-LABEL: func.func @test_mod_pow2_i32
// CHECK-DAG:   %[[c31:.*]] = arith.constant 31 : i32
// CHECK-DAG:   %[[c7:.*]] = arith.constant 7 : i32
// CHECK:       %[[shr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %[[c31]] : tensor<16xi32>, i32)
// CHECK:       %[[corr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[shr]], %[[c7]] : tensor<16xi32>, i32)
// CHECK:       %[[sum:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %[[corr]] : tensor<16xi32>, tensor<16xi32>)
// CHECK:       %[[rem:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[sum]], %[[c7]] : tensor<16xi32>, i32)
// CHECK:       %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[rem]], %[[corr]] : tensor<16xi32>, tensor<16xi32>)
// CHECK:       return %[[res]] : tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_pow2_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// mod(x, 4) i16 -> corr = (x >>s 15) & 3
// CHECK-LABEL: func.func @test_mod_pow2_i16
// CHECK-DAG:   %[[c15:.*]] = arith.constant 15 : i16
// CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : i16
// CHECK:       %[[shr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %[[c15]] : tensor<16xi16>, i16)
// CHECK:       %[[corr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[shr]], %[[c3]] : tensor<16xi16>, i16)
// CHECK:       %[[sum:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %[[corr]] : tensor<16xi16>, tensor<16xi16>)
// CHECK:       %[[rem:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[sum]], %[[c3]] : tensor<16xi16>, i16)
// CHECK:       %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[rem]], %[[corr]] : tensor<16xi16>, tensor<16xi16>)
// CHECK:       return %[[res]] : tensor<16xi16>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_pow2_i16(%arg0: tensor<16xi16>) -> tensor<16xi16> {
      %cst = arith.constant 4 : i16
      %0 = tensor.empty() : tensor<16xi16>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi16>, i16) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
      return %1 : tensor<16xi16>
  }
}

// -----

// modui(x, 2^31) i32 unsigned upper bound (y = 2^(w-1), bit pattern -2147483648) -> x & 0x7FFFFFFF
// CHECK-LABEL: func.func @test_modui_pow2_u31_i32
// CHECK-DAG:   %[[cmask:.*]] = arith.constant 2147483647 : i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins({{.*}}, %[[cmask]] : tensor<16xi32>, i32)
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_modui_pow2_u31_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant -2147483648 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// k=0: modui(x, 1) -> 0 (modui must be gone)
// CHECK-LABEL: func.func @test_modui_k0_i32
// CHECK-NOT:   binary_fn<modui>
// CHECK:       return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_modui_k0_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 1 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// k=0: mod(x, 1) -> 0 (mod must be gone)
// CHECK-LABEL: func.func @test_mod_k0_i32
// CHECK-NOT:   binary_fn<mod>
// CHECK:       return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_k0_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 1 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// i8: the shrsi product is widened to i16 by ShiftI8ToI16 in the same greedy set
// (two casts in + i16 shift + one cast out); the CHECK only pins op presence
// and mod gone, without pinning constant widths (robust to the widening detail).
// CHECK-LABEL: func.func @test_mod_pow2_i8
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>}
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK:       linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}
// CHECK-NOT:   binary_fn<mod>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_pow2_i8(%arg0: tensor<16xi8>) -> tensor<16xi8> {
      %cst = arith.constant 4 : i8
      %0 = tensor.empty() : tensor<16xi8>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi8>, i8) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
      return %1 : tensor<16xi8>
  }
}

//===== negative (A5): must not be rewritten =====

// -----

// non-pow2 divisor 7: must not be rewritten
// CHECK-LABEL: func.func @test_mod_nonpow2_i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_nonpow2_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 7 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// zero divisor: must not be rewritten
// CHECK-LABEL: func.func @test_mod_zero_i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_zero_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 0 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// negative divisor -8: must not be rewritten
// CHECK-LABEL: func.func @test_mod_negdiv_i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_negdiv_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant -8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// dynamic divisor: must not be rewritten
// CHECK-LABEL: func.func @test_mod_dyn_i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_dyn_i32(%arg0: tensor<16xi32>, %arg1: i32) -> tensor<16xi32> {
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %arg1 : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// signed y = 2^(w-1) (INT_MIN bit pattern -2147483648): must not be rewritten (counterpart of the modui unsigned upper bound)
// CHECK-LABEL: func.func @test_mod_intmin_i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_intmin_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant -2147483648 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// modui non-pow2 divisor: must not be rewritten
// CHECK-LABEL: func.func @test_modui_nonpow2_i32
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_modui_nonpow2_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 7 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

//===== i64 excluded =====

// -----

// i64 does not match (width whitelist; i64 shift/and products lower to scalar loops/libcalls)
// CHECK-LABEL: func.func @test_mod_i64_outofscope
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_i64_outofscope(%arg0: tensor<16xi64>) -> tensor<16xi64> {
      %cst = arith.constant 1024 : i64
      %0 = tensor.empty() : tensor<16xi64>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi64>, i64) outs(%0 : tensor<16xi64>) -> tensor<16xi64>
      return %1 : tensor<16xi64>
  }
}

//===== gating (no target attribute) =====

// -----

// no target attribute: the archIsAscend950 gate does not pass, must not be rewritten
// CHECK-LABEL: func.func @test_modui_noattr_gating
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
func.func @test_modui_noattr_gating(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %cst = arith.constant 8 : i32
    %0 = tensor.empty() : tensor<16xi32>
    %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
    return %1 : tensor<16xi32>
}

// -----

// Ascend310B4 is regbased but not 950: the patterns are loaded and rejected by the archIsAscend950 gate
// CHECK-LABEL: func.func @test_mod_a3_keep
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
module attributes {hacc.target = #hacc.target<"Ascend310B4">} {
  func.func @test_mod_a3_keep(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i32) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

//===== scalar dividend =====
//
// The two input type variables of elemwise_binary are independent, so a
// scalar dividend against a splat tensor divisor is legal IR. This degenerate foldable form must not be matched:
// otherwise building an empty dst from a non-shaped operand would crash (createEmptyOp ->
// k=0 would have no shape to build the zero tensor from.

// -----

// scalar dividend + mod k=3: must not be rewritten
// CHECK-LABEL: func.func @test_mod_scalar_lhs_k3
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_scalar_lhs_k3(%arg0: i32) -> tensor<16xi32> {
      %cst = arith.constant dense<8> : tensor<16xi32>
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : i32, tensor<16xi32>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// scalar dividend + mod k=0: must not be replaced with a zero tensor
// CHECK-LABEL: func.func @test_mod_scalar_lhs_k0
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_scalar_lhs_k0(%arg0: i32) -> tensor<16xi32> {
      %cst = arith.constant dense<1> : tensor<16xi32>
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : i32, tensor<16xi32>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// scalar dividend + modui k=3: must not be rewritten (unsigned side)
// CHECK-LABEL: func.func @test_modui_scalar_lhs_k3
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_modui_scalar_lhs_k3(%arg0: i32) -> tensor<16xi32> {
      %cst = arith.constant dense<8> : tensor<16xi32>
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : i32, tensor<16xi32>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

//===== type boundaries =====

// -----

// i32 dividend with an i64 divisor: must not be rewritten (wrong-width shift/mask).
// CHECK-LABEL: func.func @test_mod_rhs_width_mismatch
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_rhs_width_mismatch(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 8 : i64
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi32>, i64) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// i4 is not covered by the hardware normalization rules: must not be rewritten.
// CHECK-LABEL: func.func @test_mod_unsupported_i4
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_unsupported_i4(%arg0: tensor<16xi4>) -> tensor<16xi4> {
      %cst = arith.constant 2 : i4
      %0 = tensor.empty() : tensor<16xi4>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi4>, i4) outs(%0 : tensor<16xi4>) -> tensor<16xi4>
      return %1 : tensor<16xi4>
  }
}

// -----

// CHECK-LABEL: func.func @test_mod_narrow_rhs_width_mismatch
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
// CHECK-NOT:   binary_fn<shr
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_narrow_rhs_width_mismatch(%arg0: tensor<16xi16>) -> tensor<16xi16> {
      %cst = arith.constant 8 : i32
      %0 = tensor.empty() : tensor<16xi16>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %cst : tensor<16xi16>, i32) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
      return %1 : tensor<16xi16>
  }
}

// -----

// CHECK-LABEL: func.func @test_mod_unsigned_rhs_large_shift
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
// CHECK-NOT:   binary_fn<shr
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_unsigned_rhs_large_shift(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant 1099511627776 : i64
      %0 = tensor.empty() : tensor<16xi32>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi32>, i64) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
      return %1 : tensor<16xi32>
  }
}

// -----

// CHECK-LABEL: func.func @test_mod_unsigned_unsupported_i4
// CHECK:       hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>}
// CHECK-NOT:   binary_fn<shr
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_unsigned_unsupported_i4(%arg0: tensor<16xi4>) -> tensor<16xi4> {
      %cst = arith.constant 2 : i4
      %0 = tensor.empty() : tensor<16xi4>
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<16xi4>, i4) outs(%0 : tensor<16xi4>) -> tensor<16xi4>
      return %1 : tensor<16xi4>
  }
}

// -----

// broadcast constant source: mod(x, broadcast(8)) -> ((x+corr)&7)-corr
// CHECK-LABEL: func.func @test_mod_pow2_broadcast_i32
// CHECK-DAG:   %[[c31:.*]] = arith.constant 31 : i32
// CHECK-DAG:   %[[c7:.*]] = arith.constant 7 : i32
// CHECK:       %[[shr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%arg0, %[[c31]] : tensor<16xi32>, i32)
// CHECK:       %[[corr:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[shr]], %[[c7]] : tensor<16xi32>, i32)
// CHECK:       %[[sum:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %[[corr]] : tensor<16xi32>, tensor<16xi32>)
// CHECK:       %[[rem:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[sum]], %[[c7]] : tensor<16xi32>, i32)
// CHECK:       %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[rem]], %[[corr]] : tensor<16xi32>, tensor<16xi32>)
// CHECK:       return %[[res]] : tensor<16xi32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_pow2_broadcast_i32(%arg0: tensor<16xi32>) -> tensor<16xi32> {
      %cst = arith.constant dense<8> : tensor<i32>
      %0 = tensor.empty() : tensor<16xi32>
      %bcast = linalg.broadcast ins(%cst : tensor<i32>) outs(%0 : tensor<16xi32>) dimensions = [0]
      %1 = tensor.empty() : tensor<16xi32>
      %2 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %bcast : tensor<16xi32>, tensor<16xi32>) outs(%1 : tensor<16xi32>) -> tensor<16xi32>
      return %2 : tensor<16xi32>
  }
}

// -----

// dynamic shape k=0: modui(x, 1) -> fill(0)
// result shape derived from the DPS init (%arg1), no static shape needed.
// CHECK-LABEL: func.func @test_modui_k0_dynshape
// CHECK-NOT:   binary_fn<modui>
// CHECK:       linalg.fill
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_modui_k0_dynshape(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
      %cst = arith.constant 1 : i32
      %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<modui>} ins(%arg0, %cst : tensor<?xi32>, i32) outs(%arg1 : tensor<?xi32>) -> tensor<?xi32>
      return %1 : tensor<?xi32>
  }
}

// -----

// Narrowing result (i32 dividend, i16 result): the original op casts the
// inputs to the result element type first (i32 65535 truncates to i16 -1,
// so x % 8 would be -1). Must not match.
// CHECK-LABEL: func.func @test_mod_narrow_result_i32_i16
// CHECK:       binary_fn<mod>
// CHECK-NOT:   binary_fn<vand>
// CHECK-NOT:   binary_fn<shrsi>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_narrow_result_i32_i16(%x: tensor<16xi32>) -> tensor<16xi16> {
      %eight = arith.constant 8 : i32
      %out = tensor.empty() : tensor<16xi16>
      %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
        ins(%x, %eight : tensor<16xi32>, i32)
        outs(%out : tensor<16xi16>) -> tensor<16xi16>
      return %r : tensor<16xi16>
  }
}

// cast_unsigned must block the rewrite even when input and result widths
// match: the source operation's cast mode is part of its semantics.
// CHECK-LABEL: func.func @test_mod_cast_unsigned_same_type
// CHECK:       binary_fn<modui>
// CHECK-NOT:   binary_fn<vand>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @test_mod_cast_unsigned_same_type(%x: tensor<16xi16>) -> tensor<16xi16> {
      %eight = arith.constant 8 : i16
      %out = tensor.empty() : tensor<16xi16>
      %r = hfusion.elemwise_binary
        {fun = #hfusion.binary_fn<modui>, cast = #hfusion.type_fn<cast_unsigned>}
        ins(%x, %eight : tensor<16xi16>, i16)
        outs(%out : tensor<16xi16>) -> tensor<16xi16>
      return %r : tensor<16xi16>
  }
}
