// RUN: bishengir-opt -ave-plt-to-pge %s | FileCheck %s

// CHECK-LABEL: func.func @plt_const_zero_to_allf
// CHECK: ave.hir.pge <ALLF> : vector<64xi1>
// CHECK-NOT: ave.hir.plt
func.func @plt_const_zero_to_allf() -> vector<64xi1> {
  %c0 = arith.constant 0 : index
  %mask, %new_shape = ave.hir.plt %c0 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// CHECK-LABEL: func.func @plt_const_one_to_vl1
// CHECK: ave.hir.pge <VL1> : vector<64xi1>
func.func @plt_const_one_to_vl1() -> vector<64xi1> {
  %c1 = arith.constant 1 : index
  %mask, %new_shape = ave.hir.plt %c1 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// CHECK-LABEL: func.func @plt_const_four_to_vl4
// CHECK: ave.hir.pge <VL4> : vector<64xi1>
func.func @plt_const_four_to_vl4() -> vector<64xi1> {
  %c4 = arith.constant 4 : index
  %mask, %new_shape = ave.hir.plt %c4 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// CHECK-LABEL: func.func @plt_const_eight_to_vl8
// CHECK: ave.hir.pge <VL8> : vector<64xi1>
func.func @plt_const_eight_to_vl8() -> vector<64xi1> {
  %c8 = arith.constant 8 : index
  %mask, %new_shape = ave.hir.plt %c8 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// CHECK-LABEL: func.func @plt_const_sixteen_to_vl16
// CHECK: ave.hir.pge <VL16> : vector<64xi1>
func.func @plt_const_sixteen_to_vl16() -> vector<64xi1> {
  %c16 = arith.constant 16 : index
  %mask, %new_shape = ave.hir.plt %c16 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// CHECK-LABEL: func.func @plt_const_thirtytwo_to_vl32
// CHECK: ave.hir.pge <VL32> : vector<64xi1>
func.func @plt_const_thirtytwo_to_vl32() -> vector<64xi1> {
  %c32 = arith.constant 32 : index
  %mask, %new_shape = ave.hir.plt %c32 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// true_shape == numElements => ALL
// CHECK-LABEL: func.func @plt_const_full_to_all
// CHECK: ave.hir.pge <ALL> : vector<64xi1>
func.func @plt_const_full_to_all() -> vector<64xi1> {
  %c64 = arith.constant 64 : index
  %mask, %new_shape = ave.hir.plt %c64 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// true_shape = 5 does not match any PgePattern => no conversion
// CHECK-LABEL: func.func @plt_const_no_match
// CHECK: ave.hir.plt
// CHECK-NOT: ave.hir.pge
func.func @plt_const_no_match() -> vector<64xi1> {
  %c5 = arith.constant 5 : index
  %mask, %new_shape = ave.hir.plt %c5 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// non-constant true_shape => no conversion
// CHECK-LABEL: func.func @plt_dynamic_no_convert
// CHECK: ave.hir.plt
// CHECK-NOT: ave.hir.pge
func.func @plt_dynamic_no_convert(%arg0 : index) -> vector<64xi1> {
  %mask, %new_shape = ave.hir.plt %arg0 : vector<64xi1>, index
  return %mask : vector<64xi1>
}

// -----

// new_true_shape has a user => no conversion
// CHECK-LABEL: func.func @plt_new_shape_used_no_convert
// CHECK: ave.hir.plt
// CHECK-NOT: ave.hir.pge
func.func @plt_new_shape_used_no_convert() -> (vector<64xi1>, index) {
  %c8 = arith.constant 8 : index
  %mask, %new_shape = ave.hir.plt %c8 : vector<64xi1>, index
  return %mask, %new_shape : vector<64xi1>, index
}
