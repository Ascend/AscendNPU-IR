// RUN: not bishengir-opt %s -convert-hivmave-to-ave-intrin 2>&1 | FileCheck %s
//
// V121 (dav-920r1 / Ascend960) exclusive conversions must not be lowered on
// other targets even when the IR bypasses Normalize; they have to be legalized
// through f32 instead of selecting instructions the target cannot encode.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<AIV>} {

// CHECK: error: type conversion between fp8/e8m0/e6m2 and 16-bit float is only supported on Ascend960 (dav-920r1) and must be legalized through f32 before the AVE layer
func.func @test_vtruncf_f162f8e4m3(%src: vector<128xf16>, %mask: vector<128xi1>) -> vector<256xf8E4M3FN> {
  %0 = ave.hir.vtruncf %src, <rint>, false, <part_even>, %mask : vector<128xf16>, vector<256xf8E4M3FN>, vector<128xi1>
  return %0 : vector<256xf8E4M3FN>
}

// CHECK: error: type conversion between fp8/e8m0/e6m2 and 16-bit float is only supported on Ascend960 (dav-920r1) and must be legalized through f32 before the AVE layer
func.func @test_vextf_e8m0_bf16(%src: vector<256xui8>, %mask: vector<256xi1>) -> vector<128xbf16> {
  %0 = ave.hir.vextf.e8m0 %src, <part_even>, %mask : vector<256xui8>, vector<128xbf16>, vector<256xi1>
  return %0 : vector<128xbf16>
}

}
