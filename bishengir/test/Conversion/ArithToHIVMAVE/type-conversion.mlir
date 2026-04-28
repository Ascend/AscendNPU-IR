// RUN: bishengir-opt %s -convert-arith-to-hivmave -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// XFAIL: *

// CHECK-LABEL: @test_extf_f16
func.func @test_extf_f16(%vector: vector<64xf16>) -> vector<64xf32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extf %{{.*}}, <part_even>, %{{.*}} : vector<64xf16>, vector<64xf32>, vector<256xi1>
  %res = arith.extf %vector : vector<64xf16> to vector<64xf32> 
  return %res: vector<64xf32> 
}

// -----

// CHECK-LABEL: @test_extf_bf16
func.func @test_extf_bf16(%vector: vector<64xbf16>) -> vector<64xf32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extf %{{.*}}, <part_even>, %{{.*}} : vector<64xbf16>, vector<64xf32>, vector<256xi1>
  %res = arith.extf %vector : vector<64xbf16> to vector<64xf32> 
  return %res: vector<64xf32> 
}

// -----

// CHECK-LABEL: @test_truncf_f16
func.func @test_truncf_f16(%vector: vector<64xf32>) -> vector<64xf16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.truncf %{{.*}}, <ceil>, true, <part_even>, %{{.*}} : vector<64xf32>, vector<64xf16>, vector<256xi1>
  %res = arith.truncf %vector {round_mode = #hfusion.round_mode<ceil>} : vector<64xf32> to vector<64xf16> 
  return %res: vector<64xf16> 
}

// -----

// CHECK-LABEL: @test_truncf_bf16
func.func @test_truncf_bf16(%vector: vector<64xf32>) -> vector<64xbf16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.truncf %{{.*}}, <round>, true, <part_even>, %{{.*}} : vector<64xf32>, vector<64xbf16>, vector<256xi1>
  %res = arith.truncf %vector : vector<64xf32> to vector<64xbf16> 
  return %res: vector<64xbf16> 
}

// -----

// CHECK-LABEL: @test_extui_i82i32
func.func @test_extui_i82i32(%vector: vector<64xi8>) -> vector<64xi32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extui %{{.*}}, %{{.*}} {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi8>, vector<64xi32>, vector<256xi1>
  %res = arith.extui %vector : vector<64xi8> to vector<64xi32> 
  return %res: vector<64xi32> 
}

// -----

// CHECK-LABEL: @test_extui_i162i32
func.func @test_extui_i162i32(%vector: vector<64xi16>) -> vector<64xi32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extui %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<64xi32>, vector<256xi1>
  %res = arith.extui %vector : vector<64xi16> to vector<64xi32> 
  return %res: vector<64xi32> 
}

// -----

// CHECK-LABEL: @test_extui_i82i16
func.func @test_extui_i82i16(%vector: vector<128xi8>) -> vector<128xi16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extui %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<128xi16>, vector<256xi1>
  %res = arith.extui %vector : vector<128xi8> to vector<128xi16> 
  return %res: vector<128xi16> 
}

// -----

// CHECK-LABEL: @test_extsi_i82i32
func.func @test_extsi_i82i32(%vector: vector<64xi8>) -> vector<64xi32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extsi %{{.*}}, %{{.*}} {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi8>, vector<64xi32>, vector<256xi1>
  %res = arith.extsi %vector : vector<64xi8> to vector<64xi32> 
  return %res: vector<64xi32> 
}

// -----

// CHECK-LABEL: @test_extsi_i162i32
func.func @test_extsi_i162i32(%vector: vector<64xi16>) -> vector<64xi32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extsi %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<64xi32>, vector<256xi1>
  %res = arith.extsi %vector : vector<64xi16> to vector<64xi32> 
  return %res: vector<64xi32> 
}

// -----

// CHECK-LABEL: @test_extsi_i82i16
func.func @test_extsi_i82i16(%vector: vector<128xi8>) -> vector<128xi16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extsi %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<128xi16>, vector<256xi1>
  %res = arith.extsi %vector : vector<128xi8> to vector<128xi16> 
  return %res: vector<128xi16> 
}

// -----

// CHECK-LABEL: @test_extsi_i322i64
func.func @test_extsi_i322i64(%vector: vector<32xi32>) -> vector<32xi64> attributes {hivm.vector_function} {
  // CHECK: ave.hir.extsi %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<32xi32>, vector<32xi64>, vector<256xi1>
  %res = arith.extsi %vector : vector<32xi32> to vector<32xi64> 
  return %res: vector<32xi64> 
}

// -----

// CHECK-LABEL: @test_trunci_i642i32
func.func @test_trunci_i642i32(%vector: vector<32xi64>) -> vector<32xi32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.trunci %{{.*}}, true, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<32xi64>, vector<32xi32>, vector<256xi1>
  %res = arith.trunci %vector : vector<32xi64> to vector<32xi32> 
  return %res: vector<32xi32> 
}

// -----

// CHECK-LABEL: @test_trunci_i32i16
func.func @test_trunci_i32i16(%vector: vector<64xi32>) -> vector<64xi16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.trunci %{{.*}}, true, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<64xi32>, vector<64xi16>, vector<256xi1>
  %res = arith.trunci %vector : vector<64xi32> to vector<64xi16> 
  return %res: vector<64xi16> 
}

// -----

// CHECK-LABEL: @test_trunci_i162i8
func.func @test_trunci_i162i8(%vector: vector<128xi16>) -> vector<128xi8> attributes {hivm.vector_function} {
  // CHECK: ave.hir.trunci %{{.*}}, true, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<128xi16>, vector<128xi8>, vector<256xi1>
  %res = arith.trunci %vector : vector<128xi16> to vector<128xi8> 
  return %res: vector<128xi8> 
}

// -----

// CHECK-LABEL: @test_trunci_i322i8
func.func @test_trunci_i322i8(%vector: vector<64xi32>) -> vector<64xi8> attributes {hivm.vector_function} {
  // CHECK: ave.hir.trunci %{{.*}}, true, %{{.*}} {pp = #ave.vcvt_pp_type<pp0>} : vector<64xi32>, vector<64xi8>, vector<256xi1>
  %res = arith.trunci %vector : vector<64xi32> to vector<64xi8> 
  return %res: vector<64xi8> 
}

// -----

// CHECK-LABEL: @test_fp2si_f322i64
func.func @test_fp2si_f322i64(%vector: vector<32xf32>) -> vector<32xi64> attributes {hivm.vector_function} {
  // CHECK: ave.hir.fptosi %{{.*}}, <round>, %{{.*}} {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<32xf32>, vector<256xi1>, vector<32xi64>
  %res = arith.fptosi %vector : vector<32xf32> to vector<32xi64> 
  return %res: vector<32xi64> 
}

// -----

// CHECK-LABEL: @test_fp2si_f322i32
func.func @test_fp2si_f322i32(%vector: vector<64xf32>) -> vector<64xi32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.fptosi %{{.*}}, <round>, %{{.*}} {sat = true} : vector<64xf32>, vector<256xi1>, vector<64xi32>
  %res = arith.fptosi %vector : vector<64xf32> to vector<64xi32> 
  return %res: vector<64xi32> 
}

// -----

// CHECK-LABEL: @test_fp2si_f322i16
func.func @test_fp2si_f322i16(%vector: vector<64xf32>) -> vector<64xi16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.fptosi %{{.*}}, <round>, %{{.*}} {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xf32>, vector<256xi1>, vector<64xi16>
  %res = arith.fptosi %vector : vector<64xf32> to vector<64xi16> 
  return %res: vector<64xi16> 
}

// -----

// CHECK-LABEL: @test_fp2si_f162i32
func.func @test_fp2si_f162i32(%vector: vector<64xf16>) -> vector<64xi32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.fptosi %{{.*}}, <round>, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<64xf16>, vector<256xi1>, vector<64xi32>
  %res = arith.fptosi %vector : vector<64xf16> to vector<64xi32> 
  return %res: vector<64xi32> 
}

// -----

// CHECK-LABEL: @test_fp2si_f162i16
func.func @test_fp2si_f162i16(%vector: vector<128xf16>) -> vector<128xi16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.fptosi %{{.*}}, <round>, %{{.*}} {sat = true} : vector<128xf16>, vector<256xi1>, vector<128xi16>
  %res = arith.fptosi %vector : vector<128xf16> to vector<128xi16> 
  return %res: vector<128xi16> 
}

// -----

// CHECK-LABEL: @test_fp2si_f162i8
func.func @test_fp2si_f162i8(%vector: vector<128xf16>) -> vector<128xi8> attributes {hivm.vector_function} {
  // CHECK: ave.hir.fptosi %{{.*}}, <round>, %{{.*}} {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<128xf16>, vector<256xi1>, vector<128xi8>
  %res = arith.fptosi %vector : vector<128xf16> to vector<128xi8> 
  return %res: vector<128xi8> 
}

// -----

// CHECK-LABEL: @test_fp2si_bf162i32
func.func @test_fp2si_bf162i32(%vector: vector<64xbf16>) -> vector<64xi32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.fptosi %{{.*}}, <round>, %{{.*}} {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xbf16>, vector<256xi1>, vector<64xi32>
  %res = arith.fptosi %vector : vector<64xbf16> to vector<64xi32> 
  return %res: vector<64xi32> 
}

// -----

// CHECK-LABEL: @test_fp2ui_f162i8
func.func @test_fp2ui_f162i8(%vector: vector<128xf16>) -> vector<128xi8> attributes {hivm.vector_function} {
  // CHECK: ave.hir.fptoui %{{.*}}, <round>, true, <part_even>, %{{.*}} : vector<128xf16>, vector<256xi1>, vector<128xi8>
  %res = arith.fptoui %vector : vector<128xf16> to vector<128xi8> 
  return %res: vector<128xi8> 
}

// -----

// CHECK-LABEL: @test_si2fp_i82f16
func.func @test_si2fp_i82f16(%vector: vector<128xi8>) -> vector<128xf16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.sitofp %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<128xi8>, vector<256xi1>, vector<128xf16>
  %res = arith.sitofp %vector : vector<128xi8> to vector<128xf16> 
  return %res: vector<128xf16> 
}

// -----

// CHECK-LABEL: @test_si2fp_i162f16
func.func @test_si2fp_i162f16(%vector: vector<128xi16>) -> vector<128xf16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.sitofp %{{.*}}, %{{.*}} {rnd = #hivm.round_mode<round>} : vector<128xi16>, vector<256xi1>, vector<128xf16>
  %res = arith.sitofp %vector : vector<128xi16> to vector<128xf16> 
  return %res: vector<128xf16> 
}

// -----

// CHECK-LABEL: @test_si2fp_i162f32
func.func @test_si2fp_i162f32(%vector: vector<64xi16>) -> vector<64xf32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.sitofp %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>} : vector<64xi16>, vector<256xi1>, vector<64xf32>
  %res = arith.sitofp %vector : vector<64xi16> to vector<64xf32> 
  return %res: vector<64xf32> 
}

// -----

// CHECK-LABEL: @test_si2fp_i322f32
func.func @test_si2fp_i322f32(%vector: vector<64xi32>) -> vector<64xf32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.sitofp %{{.*}}, %{{.*}} {rnd = #hivm.round_mode<round>} : vector<64xi32>, vector<256xi1>, vector<64xf32>
  %res = arith.sitofp %vector : vector<64xi32> to vector<64xf32> 
  return %res: vector<64xf32> 
}

// -----

// CHECK-LABEL: @test_si2fp_i642f32
func.func @test_si2fp_i642f32(%vector: vector<32xi64>) -> vector<32xf32> attributes {hivm.vector_function} {
  // CHECK: ave.hir.sitofp %{{.*}}, %{{.*}} {part = #ave.vcvt_part_type<part_even>, rnd = #hivm.round_mode<round>} : vector<32xi64>, vector<256xi1>, vector<32xf32>
  %res = arith.sitofp %vector : vector<32xi64> to vector<32xf32> 
  return %res: vector<32xf32> 
}

// -----

// CHECK-LABEL: @test_ui2fp_i82f16
func.func @test_ui2fp_i82f16(%vector: vector<128xi8>) -> vector<128xf16> attributes {hivm.vector_function} {
  // CHECK: ave.hir.uitofp %{{.*}}, <part_even>, %{{.*}} : vector<128xi8>, vector<256xi1>, vector<128xf16>
  %res = arith.uitofp %vector : vector<128xi8> to vector<128xf16> 
  return %res: vector<128xf16> 
}

// -----

// CHECK-LABEL: @test_ui2fp_i642f32
func.func @test_ui2fp_i642f32(%arg0: memref<540xi64, #hivm.address_space<ub>>, %arg1: memref<540xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c64 = arith.constant 64 : index
  %c540 = arith.constant 540 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c540 step %c64 {
    %0 = affine.min affine_map<(d0) -> (-d0 + 540, 64)>(%arg2)
    %subview = memref.subview %arg0[%arg2] [%0] [1] : memref<540xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [%0] [1] : memref<540xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %0 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
    %1 = ave.hir.vload <NORM> %subview[%c0] : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
    // CHECK: ave.hir.vuitofp %1, <part_even>, %res {.*} : vector<64xi64>, vector<64xi1>, vector<64xf32>
    %2 = arith.uitofp %1 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<64xi64> to vector<64xf32>
    annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<64xf32>
    ave.hir.masked_store <NORM_B32> %subview_0[%c0], %res, %2 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  return
}

// -----
 
// CHECK-LABEL: @test_trunci_i642u32_sat
// CHECK: ave.hir.vtrunci %{{.*}}, true, %{{.*}} {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<si2ui>} : vector<64xi64>, vector<64xi32>, vector<64xi1>
func.func @test_trunci_i642u32_sat(%arg0: memref<125xi64, #hivm.address_space<ub>>, %arg1: memref<125xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c64 = arith.constant 64 : index
  %c125 = arith.constant 125 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c125 step %c64 {
    %0 = affine.min affine_map<(d0) -> (-d0 + 125, 64)>(%arg2)
    %subview = memref.subview %arg0[%arg2] [%0] [1] : memref<125xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [%0] [1] : memref<125xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %0 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
    %res_1 = ave.hir.vload <NORM> %subview[%c0] : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
    annotation.mark %res_1 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
    %1 = arith.trunci %res_1 {enable_saturate = true, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsgined_mode<si2ui>} : vector<64xi64> to vector<64xi32>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
    ave.hir.masked_store <NORM_B32> %subview_0[%c0], %res, %1 {hivm.is_continuous} : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  }
  return
}
 
// -----
 
// CHECK-LABEL: @test_trunci_u642u16_sat
// CHECK: ave.hir.vtrunci %{{.*}}, true, %{{.*}} {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2ui>} : vector<64xi64>, vector<64xi32>, vector<64xi1>
// CHECK: ave.hir.vtrunci %0, true, %{{.*}} {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2ui>} : vector<64xi32>, vector<64xi16>, vector<64xi1>
func.func @test_trunci_u642u16_sat(%arg0: memref<13xi64, #hivm.address_space<ub>>, %arg1: memref<13xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c13 = arith.constant 13 : index
  %res, %new_true_shape = ave.hir.plt %c13 {mask_op_idx = 0 : i32} : vector<64xi1>, index
  annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
  %res_0 = ave.hir.vload <NORM> %arg0[%c0] : memref<13xi64, #hivm.address_space<ub>> into vector<64xi64>
  annotation.mark %res_0 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
  %0 = arith.trunci %res_0 {enable_saturate = true, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsgined_mode<ui2ui>} : vector<64xi64> to vector<64xi16>
  annotation.mark %0 {reached_mask_ops_idx = 0 : i32} : vector<64xi16>
  ave.hir.masked_store <NORM_B16> %arg1[%c0], %res, %0 : memref<13xi16, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi16>
  return
}
 
// -----
 
// CHECK-LABEL: @test_trunci_u642i8_sat
// CHECK: ave.hir.vtrunci %res_1, true, %res {part = #ave.vcvt_part_type<part_even>, uni = #hivm.unsigned_mode<ui2ui>} : vector<64xi64>, vector<64xi32>, vector<64xi1>
// CHECK: ave.hir.vtrunci %1, true, %res {pp = #ave.vcvt_pp_type<pp0>, uni = #hivm.unsigned_mode<ui2ui>} : vector<64xi32>, vector<64xi8>, vector<64xi1>
// CHECK: ave.hir.vuitofp %2, <part_even>, %res : vector<64xi8>, vector<64xi1>, vector<64xf16>
// CHECK: ave.hir.vfptosi %3, <trunc>, %res {part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xf16>, vector<64xi1>, vector<64xi8>
func.func @test_trunci_u642i8_sat(%arg0: memref<300xi64, #hivm.address_space<ub>>, %arg1: memref<300xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c64 = arith.constant 64 : index
  %c300 = arith.constant 300 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c300 step %c64 {
    %0 = affine.min affine_map<(d0) -> (-d0 + 300, 64)>(%arg2)
    %subview = memref.subview %arg0[%arg2] [%0] [1] : memref<300xi64, #hivm.address_space<ub>> to memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [%0] [1] : memref<300xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %0 {mask_op_idx = 0 : i32} : vector<64xi1>, index
    annotation.mark %res {mask_op_idx = 0 : i32} : vector<64xi1>
    %res_1 = ave.hir.vload <NORM> %subview[%c0] : memref<?xi64, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi64>
    annotation.mark %res_1 {reached_mask_ops_idx = 0 : i32} : vector<64xi64>
    %1 = arith.trunci %res_1 {enable_saturate = true, round_mode = #hfusion.round_mode<trunc>, unsigned_mode = #hfusion.unsgined_mode<ui2si>} : vector<64xi64> to vector<64xi8>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<64xi8>
    ave.hir.masked_store <NORM_B8> %subview_0[%c0], %res, %1 {hivm.is_continuous} : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi8>
  }
  return
}