// RUN: bishengir-opt %s  -convert-hivmave-to-ave-intrin -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: @test_extf_f16
func.func @test_extf_f16(%arg0: vector<64xf16>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xf16>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}
 
// -----
 
// CHECK-LABEL: @test_extf_bf16
func.func @test_extf_bf16(%arg0: vector<64xbf16>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}
 
// -----
 
// CHECK-LABEL: @test_truncf_f16
func.func @test_truncf_f16(%arg0: vector<64xf32>) -> vector<64xf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x
  %1 = ave.hir.vtruncf %arg0, <ceil>, true, <part_even>, %0 : vector<64xf32>, vector<64xf16>, vector<64xi1>
  return %1 : vector<64xf16>
}
 
// -----
 
// CHECK-LABEL: @test_truncf_bf16
func.func @test_truncf_bf16(%arg0: vector<64xf32>) -> vector<64xbf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x
  %1 = ave.hir.vtruncf %arg0, <round>, true, <part_even>, %0 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
  return %1 : vector<64xbf16>
}
 
// -----
 
// CHECK-LABEL: @test_extui_i82i32
func.func @test_extui_i82i32(%arg0: vector<64xi8>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.u82u32.x
  %1 = ave.hir.vextui %arg0, %0 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi8>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}
 
// -----
 
// CHECK-LABEL: @test_extui_i162i32
func.func @test_extui_i162i32(%arg0: vector<64xi16>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.u162u32.x
  %1 = ave.hir.vextui %arg0, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}
 
// -----
 
// CHECK-LABEL: @test_extui_i82i16
func.func @test_extui_i82i16(%arg0: vector<128xi8>) -> vector<128xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.u82u16.x
  %1 = ave.hir.vextui %arg0, %0 {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<128xi16>, vector<128xi1>
  return %1 : vector<128xi16>
}
 
// -----
 
// CHECK-LABEL: @test_extsi_i82i32
func.func @test_extsi_i82i32(%arg0: vector<64xi8>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.s82s32.x
  %1 = ave.hir.vextsi %arg0, %0 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi8>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}
 
// -----
 
// CHECK-LABEL: @test_extsi_i162i32
func.func @test_extsi_i162i32(%arg0: vector<64xi16>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.s162s32.x
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}
 
// -----
 
// CHECK-LABEL: @test_extsi_i82i16
func.func @test_extsi_i82i16(%arg0: vector<128xi8>) -> vector<128xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.s82s16.x
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<128xi16>, vector<128xi1>
  return %1 : vector<128xi16>
}
 
// -----
 
// CHECK-LABEL: @test_extsi_i322i64
func.func @test_extsi_i322i64(%arg0: vector<32xi32>) -> vector<32xi64> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.s322s64.x
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>} : vector<32xi32>, vector<32xi64>, vector<64xi1>
  return %1 : vector<32xi64>
}
 
// -----
 
// CHECK-LABEL: @test_trunci_i642i32
func.func @test_trunci_i642i32(%arg0: vector<32xi64>, %mask: vector<256xi1>) -> vector<32xi32> attributes {hivm.vector_function} {
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.s642s32.x
  %1 = ave.hir.vtrunci %arg0, true, %mask {part = #ave.vcvt_part_type<part_even>} : vector<32xi64>, vector<32xi32>, vector<256xi1>
  return %1 : vector<32xi32>
}
 
// -----
 
// CHECK-LABEL: @test_trunci_i32i16
func.func @test_trunci_i32i16(%arg0: vector<64xi32>) -> vector<64xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.s322s16.x
  %1 = ave.hir.vtrunci %arg0, true, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xi32>, vector<64xi16>, vector<64xi1>
  return %1 : vector<64xi16>
}
 
// -----
 
// CHECK-LABEL: @test_trunci_i162i8
func.func @test_trunci_i162i8(%arg0: vector<128xi16>) -> vector<128xi8> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.s162u8.x
  %1 = ave.hir.vtrunci %arg0, true, %0 {part = #ave.vcvt_part_type<part_even>} : vector<128xi16>, vector<128xi8>, vector<128xi1>
  return %1 : vector<128xi8>
}
 
// -----
 
// CHECK-LABEL: @test_trunci_i322i8
func.func @test_trunci_i322i8(%arg0: vector<64xi32>) -> vector<64xi8> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtii.s322u8.x
  %1 = ave.hir.vtrunci %arg0, true, %0 {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi32>, vector<64xi8>, vector<64xi1>
  return %1 : vector<64xi8>
}
 
// -----
 
// CHECK-LABEL: @test_fp2si_f322i64
func.func @test_fp2si_f322i64(%arg0: vector<32xf32>) -> vector<32xi64> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f322s64.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<32xf32>, vector<64xi1>, vector<32xi64>
  return %1 : vector<32xi64>
}
 
// -----
 
// CHECK-LABEL: @test_fp2si_f322i32
func.func @test_fp2si_f322i32(%arg0: vector<64xf32>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f322s32.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true} : vector<64xf32>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}
 
// -----
 
// CHECK-LABEL: @test_fp2si_f322i32
func.func @test_fp2si_f322i32(%arg0: vector<64xf32>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f322s32.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true} : vector<64xf32>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}
 
// -----
 
// CHECK-LABEL: @test_fp2si_f322i16
func.func @test_fp2si_f322i16(%arg0: vector<64xf32>) -> vector<64xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f322s16.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xf32>, vector<64xi1>, vector<64xi16>
  return %1 : vector<64xi16>
}
 
// -----
 
// CHECK-LABEL: @test_fp2si_f162i32
func.func @test_fp2si_f162i32(%arg0: vector<64xf16>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f162s32.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xf16>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}
 
// -----
 
// CHECK-LABEL: @test_fp2si_f162i16
func.func @test_fp2si_f162i16(%arg0: vector<128xf16>) -> vector<128xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f162s16.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true} : vector<128xf16>, vector<128xi1>, vector<128xi16>
  return %1 : vector<128xi16>
}
 
// -----
 
// CHECK-LABEL: @test_fp2si_f162i8
func.func @test_fp2si_f162i8(%arg0: vector<128xf16>) -> vector<128xi8> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f162s8.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<128xf16>, vector<64xi1>, vector<128xi8>
  return %1 : vector<128xi8>
}
 
// -----
 
// CHECK-LABEL: @test_fp2si_bf162i32
func.func @test_fp2si_bf162i32(%arg0: vector<64xbf16>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.bf162s32.x
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xbf16>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}
 
// -----
 
// CHECK-LABEL: @test_fp2ui_f162i8
func.func @test_fp2ui_f162i8(%arg0: vector<128xf16>) -> vector<128xi8> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtfi.f162u8.x
  %1 = ave.hir.vfptoui %arg0, <round>, true, <part_even>, %0 : vector<128xf16>, vector<128xi1>, vector<128xi8>
  return %1 : vector<128xi8>
}
 
// -----
 
// CHECK-LABEL: @test_si2fp_i82f16
func.func @test_si2fp_i82f16(%arg0: vector<128xi8>) -> vector<128xf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtif.s82f16.x
  %1 = ave.hir.vsitofp %arg0, %0 {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}
 
// -----
 
// CHECK-LABEL: @test_si2fp_i162f16
func.func @test_si2fp_i162f16(%arg0: vector<128xi16>) -> vector<128xf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtif.s162f16.x
  %1 = ave.hir.vsitofp %arg0, %0 {rnd = #hivm.round_mode<round>} : vector<128xi16>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}
 
// -----
 
// CHECK-LABEL: @test_si2fp_i162f32
func.func @test_si2fp_i162f32(%arg0: vector<64xi16>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtif.s162f32.x
  %1 = ave.hir.vsitofp %arg0, %0 {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<64xi1>, vector<64xf32>
  return %1 : vector<64xf32>
}
 
// -----
 
// CHECK-LABEL: @test_si2fp_i322f32
func.func @test_si2fp_i322f32(%arg0: vector<64xi32>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtif.s322f32.x
  %1 = ave.hir.vsitofp %arg0, %0 {rnd = #hivm.round_mode<round>} : vector<64xi32>, vector<64xi1>, vector<64xf32>
  return %1 : vector<64xf32>
}
 
// -----
 
// CHECK-LABEL: @test_si2fp_i642f32
func.func @test_si2fp_i642f32(%arg0: vector<32xi64>, %mask: vector<256xi1>) -> vector<32xf32> attributes {hivm.vector_function} {
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtif.s642f32.x
  %1 = ave.hir.vsitofp %arg0, %mask {part = #ave.vcvt_part_type<part_even>, rnd = #hivm.round_mode<round>} : vector<32xi64>, vector<256xi1>, vector<32xf32>
  return %1 : vector<32xf32>
}
 
// -----
 
// CHECK-LABEL: @test_ui2fp_i82f16
func.func @test_ui2fp_i82f16(%arg0: vector<128xi8>) -> vector<128xf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.vcvtif.u82f16.x
  %1 = ave.hir.vuitofp %arg0, <part_even>, %0 : vector<128xi8>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}