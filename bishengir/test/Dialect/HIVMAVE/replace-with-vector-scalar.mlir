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

// BF16 is not yet accepted by the vector-scalar instructions, so a BF16
// broadcast operand must not be folded: the vector-vector op is kept as is.
// CHECK-LABEL: @no_fold_vmax_bf16
func.func @no_fold_vmax_bf16(%vec : vector<64xbf16>, %mask : vector<64xi1>) -> vector<64xbf16> {
  %cst = arith.constant 2.000000e+00 : bf16
  %bcast = ave.hir.scalar_broadcast %cst : bf16 -> vector<64xbf16>
  %res = ave.hir.vmax %vec, %bcast, %mask : vector<64xbf16>, vector<64xi1>
  // CHECK: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmax %{{.+}}, %{{.+}}, %{{.+}} : vector<64xbf16>, vector<64xi1>
  // CHECK-NOT: ave.hir.vmaxs
  func.return %res : vector<64xbf16>
}

// -----

// CHECK-LABEL: @no_fold_vmin_bf16
func.func @no_fold_vmin_bf16(%vec : vector<64xbf16>, %mask : vector<64xi1>) -> vector<64xbf16> {
  %cst = arith.constant 2.000000e+00 : bf16
  %bcast = ave.hir.scalar_broadcast %cst : bf16 -> vector<64xbf16>
  %res = ave.hir.vmin %vec, %bcast, %mask : vector<64xbf16>, vector<64xi1>
  // CHECK: ave.hir.scalar_broadcast
  // CHECK: ave.hir.vmin %{{.+}}, %{{.+}}, %{{.+}} : vector<64xbf16>, vector<64xi1>
  // CHECK-NOT: ave.hir.vmins
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
