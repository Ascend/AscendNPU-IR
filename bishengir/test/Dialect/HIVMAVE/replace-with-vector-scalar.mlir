// RUN: bishengir-opt -ave-replace-with-vector-scalar %s -split-input-file | FileCheck %s

// Test that a vector-vector op with a broadcast scalar operand is folded into
// its vector-scalar variant for the supported element types.

// -----

// CHECK-LABEL: @fold_vmax_f16
func.func @fold_vmax_f16(%vec : vector<64xf16>, %mask : vector<64xi1>) -> vector<64xf16> {
  %cst = arith.constant 2.000000e+00 : f16
  %bcast = ave.hir.scalar_broadcast %cst : f16 -> vector<64xf16>
  %res = ave.hir.vmax %vec, %bcast, %mask : vector<64xf16>, vector<64xi1>
  // CHECK-NOT: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmaxs %{{.+}}, %{{.+}}, %{{.+}} : vector<64xf16>, f16, vector<64xi1>
  func.return %res : vector<64xf16>
}

// -----

// CHECK-LABEL: @fold_vmin_f16
func.func @fold_vmin_f16(%vec : vector<64xf16>, %mask : vector<64xi1>) -> vector<64xf16> {
  %cst = arith.constant 2.000000e+00 : f16
  %bcast = ave.hir.scalar_broadcast %cst : f16 -> vector<64xf16>
  %res = ave.hir.vmin %vec, %bcast, %mask : vector<64xf16>, vector<64xi1>
  // CHECK-NOT: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmins %{{.+}}, %{{.+}}, %{{.+}} : vector<64xf16>, f16, vector<64xi1>
  func.return %res : vector<64xf16>
}

// -----

// A masked BF16 broadcast with the consumer's mask also folds to scalar form.
// CHECK-LABEL: @fold_vmax_bf16_masked_broadcast
func.func @fold_vmax_bf16_masked_broadcast(%vec : vector<128xbf16>, %scalar : bf16, %mask : vector<128xi1>) -> vector<128xbf16> {
  %bcast = ave.hir.broadcast %scalar, %mask : bf16, vector<128xi1> -> vector<128xbf16>
  %res = ave.hir.vmax %vec, %bcast, %mask : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.broadcast
  // CHECK: ave.hir.vmaxs %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, bf16, vector<128xi1>
  func.return %res : vector<128xbf16>
}

// -----

// CHECK-LABEL: @fold_vmin_bf16_masked_broadcast
func.func @fold_vmin_bf16_masked_broadcast(%vec : vector<128xbf16>, %scalar : bf16, %mask : vector<128xi1>) -> vector<128xbf16> {
  %bcast = ave.hir.broadcast %scalar, %mask : bf16, vector<128xi1> -> vector<128xbf16>
  %res = ave.hir.vmin %vec, %bcast, %mask : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.broadcast
  // CHECK: ave.hir.vmins %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, bf16, vector<128xi1>
  func.return %res : vector<128xbf16>
}

// -----

// A non-broadcast BF16 RHS must remain a vector-vector operation.
// CHECK-LABEL: @no_fold_vmax_bf16_non_broadcast
func.func @no_fold_vmax_bf16_non_broadcast(%vec : vector<128xbf16>, %rhs : vector<128xbf16>, %mask : vector<128xi1>) -> vector<128xbf16> {
  %res = ave.hir.vmax %vec, %rhs, %mask : vector<128xbf16>, vector<128xi1>
  // CHECK: ave.hir.vmax %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.vmaxs
  func.return %res : vector<128xbf16>
}

// -----

// CHECK-LABEL: @no_fold_vmin_bf16_non_broadcast
func.func @no_fold_vmin_bf16_non_broadcast(%vec : vector<128xbf16>, %rhs : vector<128xbf16>, %mask : vector<128xi1>) -> vector<128xbf16> {
  %res = ave.hir.vmin %vec, %rhs, %mask : vector<128xbf16>, vector<128xi1>
  // CHECK: ave.hir.vmin %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.vmins
  func.return %res : vector<128xbf16>
}

// -----

// BF16 max/min vector-scalar variants are supported.
// CHECK-LABEL: @fold_vmax_bf16
func.func @fold_vmax_bf16(%vec : vector<64xbf16>, %mask : vector<64xi1>) -> vector<64xbf16> {
  %cst = arith.constant 2.000000e+00 : bf16
  %bcast = ave.hir.scalar_broadcast %cst : bf16 -> vector<64xbf16>
  %res = ave.hir.vmax %vec, %bcast, %mask : vector<64xbf16>, vector<64xi1>
  // CHECK-NOT: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmaxs %{{.+}}, %{{.+}}, %{{.+}} : vector<64xbf16>, bf16, vector<64xi1>
  func.return %res : vector<64xbf16>
}

// -----

// CHECK-LABEL: @fold_vmin_bf16
func.func @fold_vmin_bf16(%vec : vector<64xbf16>, %mask : vector<64xi1>) -> vector<64xbf16> {
  %cst = arith.constant 2.000000e+00 : bf16
  %bcast = ave.hir.scalar_broadcast %cst : bf16 -> vector<64xbf16>
  %res = ave.hir.vmin %vec, %bcast, %mask : vector<64xbf16>, vector<64xi1>
  // CHECK-NOT: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmins %{{.+}}, %{{.+}}, %{{.+}} : vector<64xbf16>, bf16, vector<64xi1>
  func.return %res : vector<64xbf16>
}

// -----

// CHECK-LABEL: @fold_vmul_f16
func.func @fold_vmul_f16(%vec : vector<64xf16>, %mask : vector<64xi1>) -> vector<64xf16> {
  %cst = arith.constant 2.000000e+00 : f16
  %bcast = ave.hir.scalar_broadcast %cst : f16 -> vector<64xf16>
  %res = ave.hir.vmul %vec, %bcast, %mask : vector<64xf16>, vector<64xi1>
  // CHECK-NOT: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmuls %{{.+}}, %{{.+}}, %{{.+}} : vector<64xf16>, f16, vector<64xi1>
  func.return %res : vector<64xf16>
}

// -----

// CHECK-LABEL: @no_fold_vmul_bf16
func.func @no_fold_vmul_bf16(%vec : vector<64xbf16>, %mask : vector<64xi1>) -> vector<64xbf16> {
  %cst = arith.constant 2.000000e+00 : bf16
  %bcast = ave.hir.scalar_broadcast %cst : bf16 -> vector<64xbf16>
  %res = ave.hir.vmul %vec, %bcast, %mask : vector<64xbf16>, vector<64xi1>
  // CHECK: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmul %{{.+}}, %{{.+}}, %{{.+}} : vector<64xbf16>, vector<64xi1>
  // CHECK-NOT: ave.hir.vmuls
  func.return %res : vector<64xbf16>
}

// -----

// BF16 add vector-scalar variants are supported.
// CHECK-LABEL: @fold_vadd_bf16
func.func @fold_vadd_bf16(%vec: vector<128xbf16>, %scalar: bf16, %mask: vector<128xi1>) -> vector<128xbf16> {
  %bcast = ave.hir.scalar_broadcast %scalar : bf16 -> vector<128xbf16>
  %res = ave.hir.vadd %vec, %bcast, %mask : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vadds %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, bf16, vector<128xi1>
  return %res : vector<128xbf16>
}

// -----

// A masked BF16 broadcast with the consumer's mask also folds to scalar form.
// CHECK-LABEL: @fold_vadd_bf16_masked_broadcast
func.func @fold_vadd_bf16_masked_broadcast(%vec: vector<128xbf16>, %scalar: bf16, %mask: vector<128xi1>) -> vector<128xbf16> {
  %bcast = ave.hir.broadcast %scalar, %mask : bf16, vector<128xi1> -> vector<128xbf16>
  %res = ave.hir.vadd %vec, %bcast, %mask : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.broadcast
  // CHECK: ave.hir.vadds %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, bf16, vector<128xi1>
  return %res : vector<128xbf16>
}

// -----

// Different producer/consumer masks do not establish scalar values on all
// consumer lanes, so the BF16 rewrite must conservatively keep the broadcast.
// CHECK-LABEL: @no_fold_vadd_bf16_different_masks
func.func @no_fold_vadd_bf16_different_masks(%vec: vector<128xbf16>, %scalar: bf16, %bm: vector<128xi1>, %cm: vector<128xi1>) -> vector<128xbf16> {
  %b = ave.hir.broadcast %scalar, %bm : bf16, vector<128xi1> -> vector<128xbf16>
  %r = ave.hir.vadd %vec, %b, %cm : vector<128xbf16>, vector<128xi1>
  // CHECK: ave.hir.broadcast
  // CHECK: ave.hir.vadd %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.vadds
  return %r : vector<128xbf16>
}

// -----

// Different producer/consumer masks do not establish scalar values on all
// consumer lanes, so the BF16 rewrite must conservatively keep the broadcast.
// CHECK-LABEL: @no_fold_vmax_bf16_different_masks
func.func @no_fold_vmax_bf16_different_masks(%vec: vector<128xbf16>, %scalar: bf16, %bm: vector<128xi1>, %cm: vector<128xi1>) -> vector<128xbf16> {
  %b = ave.hir.broadcast %scalar, %bm : bf16, vector<128xi1> -> vector<128xbf16>
  %r = ave.hir.vmax %vec, %b, %cm : vector<128xbf16>, vector<128xi1>
  // CHECK: ave.hir.broadcast
  // CHECK: ave.hir.vmax %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.vmaxs
  return %r : vector<128xbf16>
}

// -----

// Different producer/consumer masks do not establish scalar values on all
// consumer lanes, so the BF16 rewrite must conservatively keep the broadcast.
// CHECK-LABEL: @no_fold_vmin_bf16_different_masks
func.func @no_fold_vmin_bf16_different_masks(%vec: vector<128xbf16>, %scalar: bf16, %bm: vector<128xi1>, %cm: vector<128xi1>) -> vector<128xbf16> {
  %b = ave.hir.broadcast %scalar, %bm : bf16, vector<128xi1> -> vector<128xbf16>
  %r = ave.hir.vmin %vec, %b, %cm : vector<128xbf16>, vector<128xi1>
  // CHECK: ave.hir.broadcast
  // CHECK: ave.hir.vmin %{{.+}}, %{{.+}}, %{{.+}} : vector<128xbf16>, vector<128xi1>
  // CHECK-NOT: ave.hir.vmins
  return %r : vector<128xbf16>
}

// -----

// Folding one user must not erase a broadcast still used by another operation.
// CHECK-LABEL: @shared_broadcast_bf16
func.func @shared_broadcast_bf16(%vec: vector<128xbf16>, %scalar: bf16, %mask: vector<128xi1>) -> (vector<128xbf16>, vector<128xbf16>) {
  %b = ave.hir.scalar_broadcast %scalar : bf16 -> vector<128xbf16>
  %a = ave.hir.vmax %vec, %b, %mask : vector<128xbf16>, vector<128xi1>
  %m = ave.hir.vmul %vec, %b, %mask : vector<128xbf16>, vector<128xi1>
  // CHECK: %[[B:.*]] = ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmaxs
  // CHECK: ave.hir.vmul %{{.*}}, %[[B]],
  return %a, %m : vector<128xbf16>, vector<128xbf16>
}
