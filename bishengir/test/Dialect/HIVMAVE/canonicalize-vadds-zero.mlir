// RUN: bishengir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: @fold_vadds_f32_zero
func.func @fold_vadds_f32_zero(%vec : vector<64xf32>, %mask : vector<64xi1>) -> vector<64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK-NOT: ave.hir.vadds
  // CHECK-NOT: arith.constant
  // CHECK: return
  func.return %res : vector<64xf32>
}

// CHECK-LABEL: @fold_vadds_i32_zero
func.func @fold_vadds_i32_zero(%vec : vector<64xi32>, %mask : vector<64xi1>) -> vector<64xi32> {
  %cst = arith.constant 0 : i32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xi32>, i32, vector<64xi1>
  // CHECK-NOT: ave.hir.vadds
  // CHECK: return
  func.return %res : vector<64xi32>
}

// CHECK-LABEL: @no_fold_vadds_f32_nonzero
func.func @no_fold_vadds_f32_nonzero(%vec : vector<64xf32>, %mask : vector<64xi1>) -> vector<64xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %res = ave.hir.vadds %vec, %cst, %mask : vector<64xf32>, f32, vector<64xi1>
  // CHECK: ave.hir.vadds
  func.return %res : vector<64xf32>
}
