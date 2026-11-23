// RUN: bishengir-opt -split-input-file -convert-non-power-two-tensors %s | FileCheck %s

// All 33 logical lanes pass, but the expanded range contains 33..63 too.
// Those synthetic lanes must not turn a successful assertion into a failure.
// CHECK-LABEL: tt.func @assertLogicalRange
// CHECK: %[[TRUE:.*]] = arith.constant dense<true> : tensor<64xi1>
// CHECK: %[[PADDED:.*]] = arith.select %{{.*}}, %{{.*}}, %[[TRUE]] : tensor<64xi1>
// CHECK: tt.assert %[[PADDED]], "logical lanes pass" : tensor<64xi1>
tt.func @assertLogicalRange() {
  %r = tt.make_range {end = 33 : i32, start = 0 : i32} : tensor<33xi32>
  %n = arith.constant dense<33> : tensor<33xi32>
  %cond = arith.cmpi slt, %r, %n : tensor<33xi32>
  tt.assert %cond, "logical lanes pass" : tensor<33xi1>
  tt.return
}

// -----

// False logical lanes must still fail; only synthetic lanes become true.
// CHECK-LABEL: tt.func @assertFalseLogicalLanes
// CHECK: %[[FALSE:.*]] = arith.constant dense<false> : tensor<128xi1>
// CHECK: %[[TRUE:.*]] = arith.constant dense<true> : tensor<128xi1>
// CHECK: %[[PADDED:.*]] = arith.select %{{.*}}, %[[FALSE]], %[[TRUE]] : tensor<128xi1>
// CHECK: tt.assert %[[PADDED]], "" : tensor<128xi1>
tt.func @assertFalseLogicalLanes() {
  %cond = arith.constant dense<false> : tensor<96xi1>
  tt.assert %cond, "" : tensor<96xi1>
  tt.return
}

// -----

// One predicate has conflicting padding requirements: true for assertion,
// false when used as a memory mask. Neither consumer may change logical lanes.
// CHECK-LABEL: tt.func @assertAndStoreMask
// CHECK: %[[PRED:.*]] = arith.cmpi
// CHECK-DAG: %[[FALSE:.*]] = arith.constant dense<false> : tensor<64xi1>
// CHECK-DAG: %[[STORE_MASK:.*]] = arith.select %{{.*}}, %[[PRED]], %[[FALSE]] : tensor<64xi1>
// CHECK-DAG: %[[TRUE:.*]] = arith.constant dense<true> : tensor<64xi1>
// CHECK-DAG: %[[ASSERT_COND:.*]] = arith.select %{{.*}}, %[[PRED]], %[[TRUE]] : tensor<64xi1>
// CHECK: tt.assert %[[ASSERT_COND]], "shared predicate" : tensor<64xi1>
// CHECK: tt.store %{{.*}}, %{{.*}}, %[[STORE_MASK]]
tt.func @assertAndStoreMask(%ptr: !tt.ptr<i32>) {
  %r = tt.make_range {end = 33 : i32, start = 0 : i32} : tensor<33xi32>
  %n = arith.constant dense<32> : tensor<33xi32>
  %cond = arith.cmpi slt, %r, %n : tensor<33xi32>
  %p = tt.splat %ptr : !tt.ptr<i32> -> tensor<33x!tt.ptr<i32>>
  %offsets = tt.addptr %p, %r : tensor<33x!tt.ptr<i32>>, tensor<33xi32>
  tt.assert %cond, "shared predicate" : tensor<33xi1>
  tt.store %offsets, %r, %cond : tensor<33x!tt.ptr<i32>>
  tt.return
}

// -----

// Power-of-two tensors and scalar assertions need no padding.
// CHECK-LABEL: tt.func @assertPowerTwoAndScalar
// CHECK-NOT: arith.select
// CHECK: tt.assert %{{.*}}, "power two" : tensor<16xi1>
// CHECK-NOT: arith.select
// CHECK: tt.assert %{{.*}}, "scalar" : i1
// CHECK-NOT: arith.select
// CHECK: tt.return
tt.func @assertPowerTwoAndScalar(%scalar: i1) {
  %cond = arith.constant dense<false> : tensor<16xi1>
  tt.assert %cond, "power two" : tensor<16xi1>
  tt.assert %scalar, "scalar" : i1
  tt.return
}
