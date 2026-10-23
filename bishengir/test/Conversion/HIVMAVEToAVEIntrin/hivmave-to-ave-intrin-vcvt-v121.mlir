// RUN: bishengir-opt %s -hacc-append-device-spec -split-input-file -convert-hivmave-to-ave-intrin | FileCheck %s
//
// V121 (dav-920r1 / Ascend960) type-conversion lowering: f16/bf16 <-> fp8 and
// the e8m0/e6m2/rcpe6m2 dedicated AVE ops -> regbase Vcvtff* intrinsics.
//
// module_core_type forces the HIVM dialect to load so that hivm::RoundModeAttr
// (the <rint>/<odd>/... custom attribute) can be parsed.

module attributes {hacc.target = #hacc.target<"Ascend960">, hivm.module_core_type = #hivm.module_core_type<AIV>} {

// CHECK-LABEL: func @test_vtruncf_f162f8e4m3
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162f8e4m3.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32, i32) -> vector<256xf8E4M3FN>
func.func @test_vtruncf_f162f8e4m3(%src: vector<128xf16>, %mask: vector<128xi1>) -> vector<256xf8E4M3FN> {
  %0 = ave.hir.vtruncf %src, <rint>, false, <part_even>, %mask : vector<128xf16>, vector<256xf8E4M3FN>, vector<128xi1>
  return %0 : vector<256xf8E4M3FN>
}

// CHECK-LABEL: func @test_vtruncf_bf162f8e5m2
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f8e5m2.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xbf16>, vector<256xi1>, i32, i32, i32, i32) -> vector<256xf8E5M2>
func.func @test_vtruncf_bf162f8e5m2(%src: vector<128xbf16>, %mask: vector<128xi1>) -> vector<256xf8E5M2> {
  %0 = ave.hir.vtruncf %src, <rint>, false, <part_even>, %mask : vector<128xbf16>, vector<256xf8E5M2>, vector<128xi1>
  return %0 : vector<256xf8E5M2>
}

// CHECK-LABEL: func @test_vextf_f8e4m32f16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<128xf16>
func.func @test_vextf_f8e4m32f16(%src: vector<256xf8E4M3FN>, %mask: vector<256xi1>) -> vector<128xf16> {
  %0 = ave.hir.vextf %src, <part_even>, %mask : vector<256xf8E4M3FN>, vector<128xf16>, vector<256xi1>
  return %0 : vector<128xf16>
}

// CHECK-LABEL: func @test_vextf_f8e5m22bf16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22bf16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xf8E5M2>, vector<256xi1>, i32) -> vector<128xbf16>
func.func @test_vextf_f8e5m22bf16(%src: vector<256xf8E5M2>, %mask: vector<256xi1>) -> vector<128xbf16> {
  %0 = ave.hir.vextf %src, <part_even>, %mask : vector<256xf8E5M2>, vector<128xbf16>, vector<256xi1>
  return %0 : vector<128xbf16>
}

// CHECK-LABEL: func @test_vtruncf_e8m0_bf162e8m0
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.bf162e8m0.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xbf16>, vector<256xi1>, i32, i32, i32, i32) -> vector<256xui8>
func.func @test_vtruncf_e8m0_bf162e8m0(%src: vector<128xbf16>, %mask: vector<128xi1>) -> vector<256xui8> {
  %0 = ave.hir.vtruncf.e8m0 %src, <rint>, false, <part_even>, %mask : vector<128xbf16>, vector<256xui8>, vector<128xi1>
  return %0 : vector<256xui8>
}

// CHECK-LABEL: func @test_vextf_e8m0_e8m02bf16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.e8m02bf16.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<256xui8>, vector<256xi1>, i32, i32) -> vector<128xbf16>
func.func @test_vextf_e8m0_e8m02bf16(%src: vector<256xui8>, %mask: vector<256xi1>) -> vector<128xbf16> {
  %0 = ave.hir.vextf.e8m0 %src, <part_even>, %mask : vector<256xui8>, vector<128xbf16>, vector<256xi1>
  return %0 : vector<128xbf16>
}

// CHECK-LABEL: func @test_vtruncf_e6m2_bf162e6m2
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.bf162e6m2.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xbf16>, vector<256xi1>, i32, i32, i32) -> vector<256xui8>
func.func @test_vtruncf_e6m2_bf162e6m2(%src: vector<128xbf16>, %mask: vector<128xi1>) -> vector<256xui8> {
  %0 = ave.hir.vtruncf.e6m2 %src, <rint>, <part_even>, %mask : vector<128xbf16>, vector<256xui8>, vector<128xi1>
  return %0 : vector<256xui8>
}

// CHECK-LABEL: func @test_vextf_e6m2_e6m22bf16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.e6m22bf16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xui8>, vector<256xi1>, i32) -> vector<128xbf16>
func.func @test_vextf_e6m2_e6m22bf16(%src: vector<256xui8>, %mask: vector<256xi1>) -> vector<128xbf16> {
  %0 = ave.hir.vextf.e6m2 %src, <part_even>, %mask : vector<256xui8>, vector<128xbf16>, vector<256xi1>
  return %0 : vector<128xbf16>
}

// CHECK-LABEL: func @test_vextf_rcpe6m2_rcpe6m22bf16
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.rcpe6m22bf16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xui8>, vector<256xi1>, i32) -> vector<128xbf16>
func.func @test_vextf_rcpe6m2_rcpe6m22bf16(%src: vector<256xui8>, %mask: vector<256xi1>) -> vector<128xbf16> {
  %0 = ave.hir.vextf.rcpe6m2 %src, <part_even>, %mask : vector<256xui8>, vector<128xbf16>, vector<256xi1>
  return %0 : vector<128xbf16>
}

// A generic u8 integer cast must NEVER be mislabeled as a V121 float
// conversion: it has to go through vcvtii, not vcvtff.

// CHECK-LABEL: func @test_u8_cast_guard
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtii.u82u16.x"
// CHECK-NOT: hivm_regbaseintrins.intr.hivm.vcvtff
func.func @test_u8_cast_guard(%src: vector<128xi8>, %mask: vector<128xi1>) -> vector<128xi16> {
  %0 = ave.hir.vextui %src, %mask {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<128xi16>, vector<128xi1>
  return %0 : vector<128xi16>
}

// CHECK-LABEL: func @test_vtruncf_f322bf16_odd_encoding
// 4.1.4.4: f32->bf16 with rnd=odd must lower round constant to 5 (ODD),
// not the historical RINT default.
// CHECK: %{{.*}} = llvm.mlir.constant(5 : i32) : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
func.func @test_vtruncf_f322bf16_odd_encoding(%src: vector<64xf32>, %mask: vector<64xi1>) -> vector<128xbf16> {
  %0 = ave.hir.vtruncf %src, <odd>, false, <part_even>, %mask : vector<64xf32>, vector<128xbf16>, vector<64xi1>
  return %0 : vector<128xbf16>
}

// CHECK-LABEL: func @test_vtruncf_f162f8e4m3_mode_encoding
// mode=zeroing must encode the regbase mode operand as 2 (ZEROING).
// CHECK: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162f8e4m3.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32, i32) -> vector<256xf8E4M3FN>
func.func @test_vtruncf_f162f8e4m3_mode_encoding(%src: vector<128xf16>, %mask: vector<128xi1>) -> vector<256xf8E4M3FN> {
  %0 = ave.hir.vtruncf %src, <rint>, false, <part_even>, %mask {mode = #ave.vcvt_mode_type<zeroing>} : vector<128xf16>, vector<256xf8E4M3FN>, vector<128xi1>
  return %0 : vector<256xf8E4M3FN>
}

// CHECK-LABEL: func @test_vtruncf_f162f8e4m3_part_sat_rnd_encoding
// rnd=odd(5), sat=true(1), part_odd(1), mode default MERGING(1).
// CHECK: %{{.*}} = llvm.mlir.constant(5 : i32) : i32
// CHECK: %{{.*}} = llvm.mlir.constant(1 : i32) : i32
// CHECK: %{{.*}} = llvm.mlir.constant(1 : i32) : i32
// CHECK: %{{.*}} = llvm.mlir.constant(1 : i32) : i32
// CHECK: "hivm_regbaseintrins.intr.hivm.vcvtff.f162f8e4m3.x"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32, i32) -> vector<256xf8E4M3FN>
func.func @test_vtruncf_f162f8e4m3_part_sat_rnd_encoding(%src: vector<128xf16>, %mask: vector<128xi1>) -> vector<256xf8E4M3FN> {
  %0 = ave.hir.vtruncf %src, <odd>, true, <part_odd>, %mask : vector<128xf16>, vector<256xf8E4M3FN>, vector<128xi1>
  return %0 : vector<256xf8E4M3FN>
}

}
