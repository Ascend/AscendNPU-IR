// RUN: bishengir-opt %s -split-input-file -convert-hivmave-to-ave-intrin | FileCheck %s

// Full VL f16 VCI feeds a masked op without a widen UCC.
// CHECK-LABEL: test_vci_f16_consumer_no_ucc
func.func @test_vci_f16_consumer_no_ucc() {
  %c0 = arith.constant 0.0 : f16
  %vci = ave.hir.vci %c0, <INCREASE> : f16, vector<128xf16>
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vci"({{.*}}, {{.*}}) : (f16, i32) -> vector<128xf16>
  // CHECK-NOT: builtin.unrealized_conversion_cast {{.*}} : vector<64xf16> to vector<128xf16>
  %sum = ave.hir.vadd %vci, %vci, %mask : vector<128xf16>, vector<128xi1>
  "test.test"(%sum) : (vector<128 x f16>) -> ()
  return
}

// -----

// Narrow logical result type narrows the full VL VCI output.
// CHECK-LABEL: test_vci_f16_narrow_result_type
func.func @test_vci_f16_narrow_result_type() {
  %c0 = arith.constant 0.0 : f16
  // CHECK: [[VCI:%.*]] = "hivm_regbaseintrins.intr.hivm.vci"({{.*}}, {{.*}}) : (f16, i32) -> vector<128xf16>
  // CHECK: builtin.unrealized_conversion_cast [[VCI]] : vector<128xf16> to vector<64xf16>
  // CHECK-NOT: builtin.unrealized_conversion_cast {{.*}} : vector<64xf16> to vector<128xf16>
  %0 = ave.hir.vci %c0, <INCREASE> : f16, vector<64xf16>
  "test.test"(%0) : (vector<64 x f16>) -> ()
  return
}
