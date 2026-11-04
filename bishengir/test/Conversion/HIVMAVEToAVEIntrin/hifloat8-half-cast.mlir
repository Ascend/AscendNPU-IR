// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9589 --normalize-vector --convert-arith-to-hivmave --convert-hivmave-to-ave-intrin | FileCheck %s
// The old HIVM vectorize() interface was removed upstream. Exercise the
// retained FP32 bridge directly through the current Arith -> AVE lowering.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
// CHECK-LABEL: func.func @decode_half
// CHECK: intr.hivm.vcvtff.hif82f32.x
// CHECK: intr.hivm.vcvtff.f322f16
func.func @decode_half(%x: vector<64xf8HIF>) -> vector<64xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %f = arith.extf %x : vector<64xf8HIF> to vector<64xf32>
  %y = arith.truncf %f : vector<64xf32> to vector<64xf16>
  return %y : vector<64xf16>
}
// CHECK-LABEL: func.func @encode_half_hybrid
// CHECK: intr.hivm.vcvtff.f162f32
// CHECK: %[[RND:.*]] = llvm.mlir.constant(6 : i32)
// CHECK: intr.hivm.vcvtff.f322hif8.x
// CHECK-SAME: %[[RND]]
func.func @encode_half_hybrid(%x: vector<64xf16>) -> vector<64xf8HIF> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %f = arith.extf %x : vector<64xf16> to vector<64xf32>
  %y = arith.truncf %f {round_mode = #hfusion.round_mode<hybrid>} : vector<64xf32> to vector<64xf8HIF>
  return %y : vector<64xf8HIF>
}
}
