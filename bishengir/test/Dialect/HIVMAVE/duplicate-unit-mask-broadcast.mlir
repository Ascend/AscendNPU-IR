// RUN: bishengir-opt -duplicate-unit-mask-broadcast %s | FileCheck %s

// Test that ave.hir.broadcast with vector<1xi1> mask and multiple users
// gets duplicated so each result has at most 1 user.

// CHECK-LABEL: func.func @duplicate_v1x1_broadcast
func.func @duplicate_v1x1_broadcast(%arg0: memref<2xi1, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant 1.000000e+00 : f16
  %0 = ave.hir.pge <ALL> : vector<1xi1>
  %bcast = ave.hir.broadcast %cst_0, %0 : f16, vector<1xi1> -> vector<1xf16>
  %1 = ave.hir.pge <ALL> : vector<1xi1>
  %bcast2 = ave.hir.broadcast %cst_1, %1 : f16, vector<1xi1> -> vector<1xf16>

  // After duplication (0.0 canonicalized to %cst, 1.0 to %cst_0):
  // Each broadcast duplicate gets its own mask (pge) clone.
  // CHECK: ave.hir.broadcast %cst, %{{.*}} : f16, vector<1xi1> -> vector<1xf16>
  // CHECK-NEXT: ave.hir.pge
  // CHECK-NEXT: ave.hir.broadcast %cst_0, %{{.*}} : f16, vector<1xi1> -> vector<1xf16>
  // CHECK-NEXT: ave.hir.pge
  // CHECK-NEXT: ave.hir.broadcast %cst, %{{.*}} : f16, vector<1xi1> -> vector<1xf16>
  // CHECK-NEXT: builtin.unrealized_conversion_cast
  // CHECK-NEXT: ave.hir.pge
  // CHECK-NEXT: ave.hir.broadcast %cst, %{{.*}} : f16, vector<1xi1> -> vector<1xf16>
  // CHECK-NEXT: builtin.unrealized_conversion_cast
  // CHECK-NEXT: builtin.unrealized_conversion_cast
  // CHECK-NEXT: builtin.unrealized_conversion_cast
  // CHECK-NEXT: return

  %v1 = builtin.unrealized_conversion_cast %bcast : vector<1xf16> to vector<64xf16>
  %v2 = builtin.unrealized_conversion_cast %bcast : vector<1xf16> to vector<64xf16>
  %v3 = builtin.unrealized_conversion_cast %bcast : vector<1xf16> to vector<64xf16>
  %v4 = builtin.unrealized_conversion_cast %bcast2 : vector<1xf16> to vector<64xf16>

  return
}

// -----

// Test that broadcasts with non-vector<1xi1> mask are NOT duplicated.

// CHECK-LABEL: func.func @no_duplicate_v64x1_broadcast
func.func @no_duplicate_v64x1_broadcast() attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  %bcast = ave.hir.broadcast %cst, %0 : f16, vector<64xi1> -> vector<64xf16>

  // CHECK: ave.hir.broadcast %cst, %{{.*}} : f16, vector<64xi1> -> vector<64xf16>
  // CHECK: ave.hir.vadd
  // CHECK: ave.hir.vadd
  // CHECK-NOT: ave.hir.broadcast

  %1 = ave.hir.pge <ALL> : vector<64xi1>
  %r1 = ave.hir.vadd %bcast, %bcast, %1 : vector<64xf16>, vector<64xi1>
  %2 = ave.hir.pge <ALL> : vector<64xi1>
  %r2 = ave.hir.vadd %bcast, %bcast, %2 : vector<64xf16>, vector<64xi1>

  return
}
