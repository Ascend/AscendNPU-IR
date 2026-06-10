// RUN: bishengir-opt %s -ave-normalize-ops -convert-hivmave-to-ave-intrin -canonicalize -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: @test_extf_f8E5M2
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xf8E5M2> to vector<256xf8E5M2>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e5m22f32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xf8E5M2>, vector<256xi1>, i32) -> vector<64xf32>
  // CHECK: return {{.*}} : vector<64xf32>

func.func @test_extf_f8E5M2(%arg0: vector<64xf8E5M2>) -> vector<64xf32> attributes {hivm.vector_function} {

  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xf8E5M2>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}


// -----
// CHECK-LABEL: @test_truncf_f8E5M2
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e5m2.x"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E5M2>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<256xf8E5M2> to vector<64xf8E5M2>
  // CHECK: return {{.*}} : vector<64xf8E5M2>

func.func @test_truncf_f8E5M2(%arg0: vector<64xf32>) -> vector<64xf8E5M2> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vtruncf %arg0, <ceil>, true, <part_even>, %0 : vector<64xf32>, vector<64xf8E5M2>, vector<64xi1>
  return %1 : vector<64xf8E5M2>
}


// -----
// CHECK-LABEL: @test_extf_f8E4M3
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xf8E4M3FN> to vector<256xf8E4M3FN>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f8e4m32f32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xf8E4M3FN>, vector<256xi1>, i32) -> vector<64xf32>
  // CHECK: return {{.*}} : vector<64xf32>

func.func @test_extf_f8E4M3(%arg0: vector<64xf8E4M3FN>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xf8E4M3FN>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}


// -----
// CHECK-LABEL: @test_truncf_f8E4M3
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f8e4m3.x"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<256xf8E4M3FN>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<256xf8E4M3FN> to vector<64xf8E4M3FN>
  // CHECK: return {{.*}} : vector<64xf8E4M3FN>

func.func @test_truncf_f8E4M3(%arg0: vector<64xf32>) -> vector<64xf8E4M3FN> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vtruncf %arg0, <ceil>, true, <part_even>, %0 : vector<64xf32>, vector<64xf8E4M3FN>, vector<64xi1>
  return %1 : vector<64xf8E4M3FN>
}


// -----
// CHECK-LABEL: @test_extf_f16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xf16> to vector<128xf16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
  // CHECK: return {{.*}} : vector<64xf32>

func.func @test_extf_f16(%arg0: vector<64xf16>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xf16>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}


// -----
// CHECK-LABEL: @test_extf_bf16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xbf16> to vector<128xbf16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.bf162f32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xbf16>, vector<256xi1>, i32) -> vector<64xf32>
  // CHECK: return {{.*}} : vector<64xf32>

func.func @test_extf_bf16(%arg0: vector<64xbf16>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vextf %arg0, <part_even>, %0 : vector<64xbf16>, vector<64xf32>, vector<64xi1>
  return %1 : vector<64xf32>
}


// -----
// CHECK-LABEL: @test_truncf_f16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xf16>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<128xf16> to vector<64xf16>
  // CHECK: return {{.*}} : vector<64xf16>

func.func @test_truncf_f16(%arg0: vector<64xf32>) -> vector<64xf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vtruncf %arg0, <ceil>, true, <part_even>, %0 : vector<64xf32>, vector<64xf16>, vector<64xi1>
  return %1 : vector<64xf16>
}


// -----
// CHECK-LABEL: @test_truncf_bf16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtff.f322bf16.x"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xbf16>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<128xbf16> to vector<64xbf16>
  // CHECK: return {{.*}} : vector<64xbf16>

func.func @test_truncf_bf16(%arg0: vector<64xf32>) -> vector<64xbf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vtruncf %arg0, <round>, true, <part_even>, %0 : vector<64xf32>, vector<64xbf16>, vector<64xi1>
  return %1 : vector<64xbf16>
}


// -----
// CHECK-LABEL: @test_extui_i82i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xi8> to vector<256xi8>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.u82u32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
  // CHECK: return {{.*}} : vector<64xi32>

func.func @test_extui_i82i32(%arg0: vector<64xi8>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vextui %arg0, %0 {pp = #ave.vcvt_pp_type<pp0>, element_alignment_bit_width = -1 : i32} : vector<64xi8>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}


// -----
// CHECK-LABEL: @test_extui_i162i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xi16> to vector<128xi16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.u162u32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xi16>, vector<256xi1>, i32) -> vector<64xi32>
  // CHECK: return {{.*}} : vector<64xi32>

func.func @test_extui_i162i32(%arg0: vector<64xi16>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vextui %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xi16>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}


// -----
// CHECK-LABEL: @test_extui_i82i16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.u82u16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xi16>
  // CHECK: return {{.*}} : vector<128xi16>

func.func @test_extui_i82i16(%arg0: vector<128xi8>) -> vector<128xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vextui %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<128xi8>, vector<128xi16>, vector<128xi1>
  return %1 : vector<128xi16>
}


// -----
// CHECK-LABEL: @test_extsi_i82i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xi8> to vector<256xi8>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s82s32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<64xi32>
  // CHECK: return {{.*}} : vector<64xi32>

func.func @test_extsi_i82i32(%arg0: vector<64xi8>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vextsi %arg0, %0 {pp = #ave.vcvt_pp_type<pp0>, element_alignment_bit_width = -1 : i32} : vector<64xi8>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}


// -----
// CHECK-LABEL: @test_extsi_i162i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xi16> to vector<128xi16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s162s32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xi16>, vector<256xi1>, i32) -> vector<64xi32>
  // CHECK: return {{.*}} : vector<64xi32>

func.func @test_extsi_i162i32(%arg0: vector<64xi16>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xi16>, vector<64xi32>, vector<64xi1>
  return %1 : vector<64xi32>
}


// -----
// CHECK-LABEL: @test_extsi_i82i16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s82s16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xi16>
  // CHECK: return {{.*}} : vector<128xi16>

func.func @test_extsi_i82i16(%arg0: vector<128xi8>) -> vector<128xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<128xi8>, vector<128xi16>, vector<128xi1>
  return %1 : vector<128xi16>
}


// -----
// CHECK-LABEL: @test_extsi_i322i64
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<32xi32> to vector<64xi32>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s322s64.x"({{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32) -> vector<32xi64>
  // CHECK: return {{.*}} : vector<32xi64>

func.func @test_extsi_i322i64(%arg0: vector<32xi32>) -> vector<32xi64> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<32xi1>
  %1 = ave.hir.vextsi %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<32xi32>, vector<32xi64>, vector<32xi1>
  return %1 : vector<32xi64>
}


// -----
// CHECK-LABEL: @test_trunci_i642i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s642s32.x"(%arg0, %arg1, {{.*}}, {{.*}}) : (vector<32xi64>, vector<256xi1>, i32, i32) -> vector<64xi32>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<64xi32> to vector<32xi32>
  // CHECK: return {{.*}} : vector<32xi32>

func.func @test_trunci_i642i32(%arg0: vector<32xi64>, %mask: vector<256xi1>) -> vector<32xi32> attributes {hivm.vector_function} {
  %1 = ave.hir.vtrunci %arg0, true, %mask {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<32xi64>, vector<32xi32>, vector<256xi1>
  return %1 : vector<32xi32>
}


// -----
// CHECK-LABEL: @test_trunci_i32i16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s322s16.x"(%arg0, {{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<128xi16> to vector<64xi16>
  // CHECK: return {{.*}} : vector<64xi16>

func.func @test_trunci_i32i16(%arg0: vector<64xi32>) -> vector<64xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vtrunci %arg0, true, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xi32>, vector<64xi16>, vector<64xi1>
  return %1 : vector<64xi16>
}


// -----
// CHECK-LABEL: @test_trunci_i162i8
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s162u8.x"(%arg0, {{.*}}, {{.*}}, {{.*}}) : (vector<128xi16>, vector<256xi1>, i32, i32) -> vector<256xi8>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<256xi8> to vector<128xi8>
  // CHECK: return {{.*}} : vector<128xi8>

func.func @test_trunci_i162i8(%arg0: vector<128xi16>) -> vector<128xi8> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vtrunci %arg0, true, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<128xi16>, vector<128xi8>, vector<128xi1>
  return %1 : vector<128xi8>
}


// -----
// CHECK-LABEL: @test_trunci_i322i8
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s322u8.x"(%arg0, {{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<256xi8>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<256xi8> to vector<64xi8>
  // CHECK: return {{.*}} : vector<64xi8>

func.func @test_trunci_i322i8(%arg0: vector<64xi32>) -> vector<64xi8> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vtrunci %arg0, true, %0 {pp = #ave.vcvt_pp_type<pp0>, element_alignment_bit_width = -1 : i32} : vector<64xi32>, vector<64xi8>, vector<64xi1>
  return %1 : vector<64xi8>
}


// -----
// CHECK-LABEL: @test_fp2si_f322i64
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<32xf32> to vector<64xf32>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f322s64.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<32xi64>
  // CHECK: return {{.*}} : vector<32xi64>

func.func @test_fp2si_f322i64(%arg0: vector<32xf32>) -> vector<32xi64> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<32xi1>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true, element_alignment_bit_width = -1 : i32} : vector<32xf32>, vector<32xi1>, vector<32xi64>
  return %1 : vector<32xi64>
}


// -----
// CHECK-LABEL: @test_fp2si_f322i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f322s32.x"(%arg0, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32) -> vector<64xi32>
  // CHECK: return {{.*}} : vector<64xi32>

func.func @test_fp2si_f322i32(%arg0: vector<64xf32>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true, element_alignment_bit_width = -1 : i32} : vector<64xf32>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}


// -----
// CHECK-LABEL: @test_fp2si_f322i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f322s32.x"(%arg0, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32) -> vector<64xi32>
  // CHECK: return {{.*}} : vector<64xi32>

func.func @test_fp2si_f322i32(%arg0: vector<64xf32>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true, element_alignment_bit_width = -1 : i32} : vector<64xf32>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}


// -----
// CHECK-LABEL: @test_fp2si_f322i16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f322s16.x"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xi16>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<128xi16> to vector<64xi16>
  // CHECK: return {{.*}} : vector<64xi16>

func.func @test_fp2si_f322i16(%arg0: vector<64xf32>) -> vector<64xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true, element_alignment_bit_width = -1 : i32} : vector<64xf32>, vector<64xi1>, vector<64xi16>
  return %1 : vector<64xi16>
}


// -----
// CHECK-LABEL: @test_fp2si_f162i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xf16> to vector<128xf16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s32.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32) -> vector<64xi32>
  // CHECK: return {{.*}} : vector<64xi32>

func.func @test_fp2si_f162i32(%arg0: vector<64xf16>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xf16>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}


// -----
// CHECK-LABEL: @test_fp2si_f162i16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s16.x"(%arg0, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32) -> vector<128xi16>
  // CHECK: return {{.*}} : vector<128xi16>

func.func @test_fp2si_f162i16(%arg0: vector<128xf16>) -> vector<128xi16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {sat = true, element_alignment_bit_width = -1 : i32} : vector<128xf16>, vector<128xi1>, vector<128xi16>
  return %1 : vector<128xi16>
}


// -----
// CHECK-LABEL: @test_fp2si_f162i8
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s8.x"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32) -> vector<256xi8>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<256xi8> to vector<128xi8>
  // CHECK: return {{.*}} : vector<128xi8>

func.func @test_fp2si_f162i8(%arg0: vector<128xf16>) -> vector<128xi8> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true, element_alignment_bit_width = -1 : i32} : vector<128xf16>, vector<128xi1>, vector<128xi8>
  return %1 : vector<128xi8>
}


// -----
// CHECK-LABEL: @test_fp2si_bf162i32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xbf16> to vector<128xbf16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.bf162s32.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xbf16>, vector<256xi1>, i32, i32, i32) -> vector<64xi32>
  // CHECK: return {{.*}} : vector<64xi32>

func.func @test_fp2si_bf162i32(%arg0: vector<64xbf16>) -> vector<64xi32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vfptosi %arg0, <round>, %0 {part = #ave.vcvt_part_type<part_even>, sat = true, element_alignment_bit_width = -1 : i32} : vector<64xbf16>, vector<64xi1>, vector<64xi32>
  return %1 : vector<64xi32>
}


// -----
// CHECK-LABEL: @test_fp2ui_f162i8
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162u8.x"(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32, i32) -> vector<256xi8>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<256xi8> to vector<128xi8>
  // CHECK: return {{.*}} : vector<128xi8>

func.func @test_fp2ui_f162i8(%arg0: vector<128xf16>) -> vector<128xi8> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vfptoui %arg0, <round>, true, <part_even>, %0 : vector<128xf16>, vector<128xi1>, vector<128xi8>
  return %1 : vector<128xi8>
}


// -----
// CHECK-LABEL: @test_fp2ui_f162u16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xf16> to vector<128xf16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s32.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32) -> vector<64xi32>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s322u16.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<128xi16> to vector<64xi16>
  // CHECK: return {{.*}} : vector<64xi16>

func.func @test_fp2ui_f162u16(%arg0: vector<64xf16>) -> vector<64xi16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vfptoui %arg0, <round>, true, <part_even>, %0 : vector<64xf16>, vector<64xi1>, vector<64xi16>
  return %1 : vector<64xi16>
}


// -----
// CHECK-LABEL: @test_fp2ui_f162u16_2
  // CHECK: {{.*}} = llvm.mlir.undef : !llvm.struct<(vector<64xi16>, vector<64xi16>)>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<128xf16> to !llvm.struct<(vector<64xf16>, vector<64xf16>)>
  // CHECK: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm.struct<(vector<64xf16>, vector<64xf16>)>
  // CHECK: {{.*}} = llvm.extractvalue {{.*}}[1] : !llvm.struct<(vector<64xf16>, vector<64xf16>)>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<64xf16> to vector<128xf16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s32.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32) -> vector<64xi32>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s322u16.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<128xi16> to vector<64xi16>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<64xf16> to vector<128xf16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s32.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<128xf16>, vector<256xi1>, i32, i32) -> vector<64xi32>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtii.s322u16.x"({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32, i32) -> vector<128xi16>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<128xi16> to vector<64xi16>
  // CHECK: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm.struct<(vector<64xi16>, vector<64xi16>)>
  // CHECK: {{.*}} = llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm.struct<(vector<64xi16>, vector<64xi16>)>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : !llvm.struct<(vector<64xi16>, vector<64xi16>)> to vector<128xi16>
  // CHECK: return {{.*}} : vector<128xi16>

func.func @test_fp2ui_f162u16_2(%arg0: vector<128xf16>) -> vector<128xi16> attributes {hivm.vector_function, element_alignment_bit_width = -1 : i32} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vfptoui %arg0, <round>, true, <part_even>, %0 : vector<128xf16>, vector<128xi1>, vector<128xi16>
  return %1 : vector<128xi16>
}


// -----
// CHECK-LABEL: @test_si2fp_i82f16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtif.s82f16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xf16>
  // CHECK: return {{.*}} : vector<128xf16>

func.func @test_si2fp_i82f16(%arg0: vector<128xi8>) -> vector<128xf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vsitofp %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<128xi8>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}


// -----
// CHECK-LABEL: @test_si2fp_i162f16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtif.s162f16.x"(%arg0, {{.*}}, {{.*}}) : (vector<128xi16>, vector<256xi1>, i32) -> vector<128xf16>
  // CHECK: return {{.*}} : vector<128xf16>

func.func @test_si2fp_i162f16(%arg0: vector<128xi16>) -> vector<128xf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vsitofp %arg0, %0 {rnd = #hivm.round_mode<round>, element_alignment_bit_width = -1 : i32} : vector<128xi16>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}


// -----
// CHECK-LABEL: @test_si2fp_i162f32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<64xi16> to vector<128xi16>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtif.s162f32.x"({{.*}}, {{.*}}, {{.*}}) : (vector<128xi16>, vector<256xi1>, i32) -> vector<64xf32>
  // CHECK: return {{.*}} : vector<64xf32>

func.func @test_si2fp_i162f32(%arg0: vector<64xi16>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vsitofp %arg0, %0 {part = #ave.vcvt_part_type<part_even>, element_alignment_bit_width = -1 : i32} : vector<64xi16>, vector<64xi1>, vector<64xf32>
  return %1 : vector<64xf32>
}


// -----
// CHECK-LABEL: @test_si2fp_i322f32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b32"({{.*}}, {{.*}}) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtif.s322f32.x"(%arg0, {{.*}}, {{.*}}) : (vector<64xi32>, vector<256xi1>, i32) -> vector<64xf32>
  // CHECK: return {{.*}} : vector<64xf32>

func.func @test_si2fp_i322f32(%arg0: vector<64xi32>) -> vector<64xf32> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}: vector<64xi1>
  %1 = ave.hir.vsitofp %arg0, %0 {rnd = #hivm.round_mode<round>, element_alignment_bit_width = -1 : i32} : vector<64xi32>, vector<64xi1>, vector<64xf32>
  return %1 : vector<64xf32>
}


// -----
// CHECK-LABEL: @test_si2fp_i642f32
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtif.s642f32.x"(%arg0, %arg1, {{.*}}, {{.*}}) : (vector<32xi64>, vector<256xi1>, i32, i32) -> vector<64xf32>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast {{.*}} : vector<64xf32> to vector<32xf32>
  // CHECK: return {{.*}} : vector<32xf32>

func.func @test_si2fp_i642f32(%arg0: vector<32xi64>, %mask: vector<256xi1>) -> vector<32xf32> attributes {hivm.vector_function} {
  %1 = ave.hir.vsitofp %arg0, %mask {part = #ave.vcvt_part_type<part_even>, rnd = #hivm.round_mode<round>, element_alignment_bit_width = -1 : i32} : vector<32xi64>, vector<256xi1>, vector<32xf32>
  return %1 : vector<32xf32>
}


// -----
// CHECK-LABEL: @test_ui2fp_i82f16
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.pge.b16"({{.*}}, {{.*}}) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
  // CHECK: {{.*}} = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<256xi8>
  // CHECK: {{.*}} = "hivm_regbaseintrins.intr.hivm.vcvtif.u82f16.x"({{.*}}, {{.*}}, {{.*}}) : (vector<256xi8>, vector<256xi1>, i32) -> vector<128xf16>
  // CHECK: return {{.*}} : vector<128xf16>

func.func @test_ui2fp_i82f16(%arg0: vector<128xi8>) -> vector<128xf16> attributes {hivm.vector_function} {
  %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>}: vector<128xi1>
  %1 = ave.hir.vuitofp %arg0, <part_even>, %0 : vector<128xi8>, vector<128xi1>, vector<128xf16>
  return %1 : vector<128xf16>
}