// RUN: bishengir-opt %s -split-input-file -convert-hivmave-to-ave-intrin | FileCheck %s

// Full VL f32 VCI feeds a masked op without a widen UCC.
// CHECK-LABEL: test_vci_f32_consumer_no_ucc
func.func @test_vci_f32_consumer_no_ucc() {
  %c0 = arith.constant 0.0 : f32
  %vci = ave.hir.vci %c0, <INCREASE> : f32, vector<64xf32>
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vci"({{.*}}, {{.*}}) : (f32, i32) -> vector<64xf32>
  // CHECK-NOT: builtin.unrealized_conversion_cast {{.*}} : vector<32xf32> to vector<64xf32>
  %sum = ave.hir.vadd %vci, %vci, %mask : vector<64xf32>, vector<64xi1>
  "test.test"(%sum) : (vector<64 x f32>) -> ()
  return
}

// -----

// Narrow logical result type narrows the full VL VCI output.
// CHECK-LABEL: test_vci_f32_narrow_result_type
func.func @test_vci_f32_narrow_result_type() {
  %c0 = arith.constant 0.0 : f32
  // CHECK: [[VCI:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"({{.*}}, {{.*}}) : (f32, i32) -> vector<64xf32>
  // CHECK: builtin.unrealized_conversion_cast [[VCI]] : vector<64xf32> to vector<32xf32>
  // CHECK-NOT: builtin.unrealized_conversion_cast {{.*}} : vector<32xf32> to vector<64xf32>
  %0 = ave.hir.vci %c0, <INCREASE> : f32, vector<32xf32>
  "test.test"(%0) : (vector<32 x f32>) -> ()
  return
}
