// RUN: bishengir-opt %s -hacc-append-device-spec -convert-arith-to-hivmave -convert-hivmave-to-ave-intrin | FileCheck %s
//
// End-to-end V121 (dav-920r1 / Ascend960) type conversion: high-level
// arith.truncf / arith.extf on fp8/f16/bf16/f32 vectors are lowered through
// ArithToHIVMAVE -> HIVMAVEToAVEIntrin all the way to the regbase Vcvtff*
// intrinsics, exercising the optional `mode` operand plumbing (default
// MERGING) and the hardware operand counts (6 for f16/bf16 -> fp8, 3 for
// fp8 -> f16/bf16, 5 for f32 -> bf16/f16).

module attributes {hacc.target = #hacc.target<"Ascend960">} {

// CHECK-LABEL: func @call_trunc_f16_to_f8e4m3
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162f8e4m3.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32, i32) -> vector<256xf8E4M3FN>
func.func @call_trunc_f16_to_f8e4m3(%arg0: vector<128xf16>) -> vector<128xf8E4M3FN> attributes {hivm.vector_function} {
  %0 = arith.truncf %arg0 : vector<128xf16> to vector<128xf8E4M3FN>
  return %0 : vector<128xf8E4M3FN>
}

// CHECK-LABEL: func @call_trunc_bf16_to_f8e5m2
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f8e5m2.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xbf16>, vector<256xi1>, i32, i32, i32, i32) -> vector<256xf8E5M2>
func.func @call_trunc_bf16_to_f8e5m2(%arg0: vector<128xbf16>) -> vector<128xf8E5M2> attributes {hivm.vector_function} {
  %0 = arith.truncf %arg0 : vector<128xbf16> to vector<128xf8E5M2>
  return %0 : vector<128xf8E5M2>
}

// CHECK-LABEL: func @call_ext_f8e4m3_to_f16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<128xf16>
func.func @call_ext_f8e4m3_to_f16(%arg0: vector<128xf8E4M3FN>) -> vector<128xf16> attributes {hivm.vector_function} {
  %0 = arith.extf %arg0 : vector<128xf8E4M3FN> to vector<128xf16>
  return %0 : vector<128xf16>
}

// CHECK-LABEL: func @call_ext_f8e5m2_to_bf16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22bf16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xf8E5M2>, vector<256xi1>, i32) -> vector<128xbf16>
func.func @call_ext_f8e5m2_to_bf16(%arg0: vector<128xf8E5M2>) -> vector<128xbf16> attributes {hivm.vector_function} {
  %0 = arith.extf %arg0 : vector<128xf8E5M2> to vector<128xbf16>
  return %0 : vector<128xbf16>
}

// CHECK-LABEL: func @call_trunc_f32_to_bf16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
func.func @call_trunc_f32_to_bf16(%arg0: vector<64xf32>) -> vector<64xbf16> attributes {hivm.vector_function} {
  %0 = arith.truncf %arg0 : vector<64xf32> to vector<64xbf16>
  return %0 : vector<64xbf16>
}

// CHECK-LABEL: func @call_trunc_f32_to_f16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xf16>
func.func @call_trunc_f32_to_f16(%arg0: vector<64xf32>) -> vector<64xf16> attributes {hivm.vector_function} {
  %0 = arith.truncf %arg0 : vector<64xf32> to vector<64xf16>
  return %0 : vector<64xf16>
}

// CHECK-LABEL: func @call_ext_f16_to_f32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
func.func @call_ext_f16_to_f32(%arg0: vector<64xf16>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = arith.extf %arg0 : vector<64xf16> to vector<64xf32>
  return %0 : vector<64xf32>
}

// CHECK-LABEL: func @call_trunc_f32_to_f8e4m3
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
func.func @call_trunc_f32_to_f8e4m3(%arg0: vector<64xf32>) -> vector<64xf8E4M3FN> attributes {hivm.vector_function} {
  %0 = arith.truncf %arg0 : vector<64xf32> to vector<64xf8E4M3FN>
  return %0 : vector<64xf8E4M3FN>
}

// A chained case proving the end-to-end data flow: f16 -> fp8 -> f16 must
// produce the narrowing Vcvtff followed by the widening Vcvtff.

// CHECK-LABEL: func @call_chain_f16_to_f8e4m3_to_f16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162f8e4m3.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32, i32) -> vector<256xf8E4M3FN>
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<128xf16>
func.func @call_chain_f16_to_f8e4m3_to_f16(%arg0: vector<128xf16>) -> vector<128xf16> attributes {hivm.vector_function} {
  %0 = arith.truncf %arg0 : vector<128xf16> to vector<128xf8E4M3FN>
  %1 = arith.extf %0 : vector<128xf8E4M3FN> to vector<128xf16>
  return %1 : vector<128xf16>
}

}
