// RUN: bishengir-opt %s -convert-vector-to-hivmave -convert-hivmave-to-ave-intrin | FileCheck %s
 
// CHECK-LABEL: @test_vbr_bf16
func.func @test_vbr_bf16(%arg0 : bf16) -> vector<128xbf16> {
  // CHECK: %{{.*}} = "hivm_regbaseintrins.intr.hivm.vbr"(%{{.*}}) : (bf16) -> vector<128xbf16>
  %0 = vector.broadcast %arg0 : bf16 to vector<128xbf16>
  return %0 : vector<128xbf16>
}