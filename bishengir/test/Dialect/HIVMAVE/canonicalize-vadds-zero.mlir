// RUN: bishengir-opt -canonicalize %s | FileCheck %s

// All-true masks are foldable: inactive lanes do not exist, so replacing the
// zero-scalar vadds with the vector is value-preserving.

// CHECK-LABEL: @fold_vadds_f32_zero_all
func.func @fold_vadds_f32_zero_all(%vec : vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %cst = arith.constant 0.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK-NOT: ave.hir.vadds
  // CHECK-NOT: arith.constant
  // CHECK: return
  func.return %res : vector<64xf32>
}

// CHECK-LABEL: @fold_vadds_i32_zero_vl64
func.func @fold_vadds_i32_zero_vl64(%vec : vector<64xi32>) -> vector<64xi32> {
  %mask = ave.hir.pge <VL64> : vector<64xi1>
  %cst = arith.constant 0 : i32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xi32>, i32, vector<64xi1>
  // CHECK-NOT: ave.hir.vadds
  // CHECK-NOT: arith.constant
  // CHECK: return
  func.return %res : vector<64xi32>
}

// CHECK-LABEL: @fold_vadds_f32_zero_dense_all
func.func @fold_vadds_f32_zero_dense_all(%vec : vector<64xf32>) -> vector<64xf32> {
  %mask = arith.constant dense<true> : vector<64xi1>
  %cst = arith.constant 0.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK-NOT: ave.hir.vadds
  // CHECK-NOT: arith.constant
  // CHECK: return
  func.return %res : vector<64xf32>
}

// Partial/dynamic masks must not be folded: ZEROING still has to clear the
// inactive lanes.

// CHECK-LABEL: @no_fold_vadds_f32_zero_partial
func.func @no_fold_vadds_f32_zero_partial(%vec : vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <VL32> : vector<64xi1>
  %cst = arith.constant 0.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK: ave.hir.vadds
  func.return %res : vector<64xf32>
}

// CHECK-LABEL: @no_fold_vadds_i32_zero_dynamic
func.func @no_fold_vadds_i32_zero_dynamic(%vec : vector<64xi32>, %mask : vector<64xi1>) -> vector<64xi32> {
  %cst = arith.constant 0 : i32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xi32>, i32, vector<64xi1>
  // CHECK: ave.hir.vadds
  func.return %res : vector<64xi32>
}

// CHECK-LABEL: @no_fold_vadds_f32_zero_allf
func.func @no_fold_vadds_f32_zero_allf(%vec : vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALLF> : vector<64xi1>
  %cst = arith.constant 0.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK: ave.hir.vadds
  func.return %res : vector<64xf32>
}

// CHECK-LABEL: @no_fold_vadds_f32_zero_dense_false
func.func @no_fold_vadds_f32_zero_dense_false(%vec : vector<64xf32>) -> vector<64xf32> {
  %mask = arith.constant dense<false> : vector<64xi1>
  %cst = arith.constant 0.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK: ave.hir.vadds
  func.return %res : vector<64xf32>
}

// H may lack the bit-width attr; it must not reach getNumfromPgePattern and
// must not be folded.

// CHECK-LABEL: @no_fold_vadds_f32_zero_h
func.func @no_fold_vadds_f32_zero_h(%vec : vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <H> : vector<64xi1>
  %cst = arith.constant 0.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK: ave.hir.vadds
  func.return %res : vector<64xf32>
}

// Non-zero scalars remain unmodified even with an all-true mask.

// CHECK-LABEL: @no_fold_vadds_f32_nonzero_all
func.func @no_fold_vadds_f32_nonzero_all(%vec : vector<64xf32>) -> vector<64xf32> {
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  %cst = arith.constant 1.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK: ave.hir.vadds
  func.return %res : vector<64xf32>
}
