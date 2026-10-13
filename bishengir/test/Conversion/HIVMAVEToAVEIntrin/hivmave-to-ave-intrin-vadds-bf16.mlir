// RUN: bishengir-opt %s -split-input-file -convert-hivmave-to-ave-intrin | FileCheck %s

// CHECK-LABEL: test_vadds_bf16
func.func @test_vadds_bf16(%vec : vector<128xbf16>) {
  %cst = arith.constant 1.500000e+00 : bf16
  %mask = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vadds.s.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
  %0 = ave.hir.vadds %vec, %cst, %mask : vector<128xbf16>, bf16, vector<128xi1>
  "test.test"(%0) : (vector<128xbf16>) -> ()
  return
}

// -----

// CHECK-LABEL: test_vadds_bf16_narrow
func.func @test_vadds_bf16_narrow(%vec : vector<64xbf16>) {
  %cst = arith.constant 1.500000e+00 : bf16
  %mask = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: builtin.unrealized_conversion_cast {{.*}} : vector<64xbf16> to vector<128xbf16>
  // CHECK: "hivm_regbaseintrins.intr.hivm.vadds.s.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xbf16>, bf16, vector<256xi1>) -> vector<128xbf16>
  // CHECK: builtin.unrealized_conversion_cast {{.*}} : vector<128xbf16> to vector<64xbf16>
  %0 = ave.hir.vadds %vec, %cst, %mask : vector<64xbf16>, bf16, vector<64xi1>
  "test.test"(%0) : (vector<64xbf16>) -> ()
  return
}
