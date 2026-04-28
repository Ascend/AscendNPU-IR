// RUN: bishengir-opt %s -convert-arith-to-hivmave -split-input-file -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: @elemwise_abs_test_8xf16
func.func @elemwise_abs_test_8xf16(%arg0: memref<8xf16, #hivm.address_space<ub>>, %arg1: memref<8xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [8] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<8xf16, #hivm.address_space<ub>>, vector<128xf16>
  // CHECK: %[[PGE:.*]] = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK: %{{.*}} = ave.hir.vabs %{{.*}}, %[[PGE]] : vector<128xf16>, vector<128xi1>
  %2 = math.absf %1 : vector<128xf16>
  vector.transfer_write %2, %arg1[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<8xf16, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: @elemwise_abs_test_15xf32
func.func @elemwise_abs_test_15xf32(%arg0: memref<15xf32, #hivm.address_space<ub>>, %arg1: memref<15xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [15] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<15xf32, #hivm.address_space<ub>>, vector<64xf32>
  // CHECK: %[[PGE:.*]] = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK: %{{.*}} = ave.hir.vabs %{{.*}}, %[[PGE]] : vector<64xf32>, vector<64xi1>
  %2 = math.absf %1 : vector<64xf32>
  vector.transfer_write %2, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<15xf32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: @elemwise_abs_test_15x33xf32
func.func @elemwise_abs_test_15x33xf32(%arg0: memref<495xf32, #hivm.address_space<ub>>, %arg1: memref<495xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c495 = arith.constant 495 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c495 step %c64 {
    %0 = affine.min affine_map<(d0) -> (-d0 + 495, 64)>(%arg2)
    %subview = memref.subview %arg0[%arg2] [%0] [1] : memref<495xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [%0] [1] : memref<495xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %1 = vector.create_mask %0 : vector<64xi1>
    %2 = vector.transfer_read %subview[%c0], %cst, %1 {in_bounds = [true]} : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
    // CHECK: %[[PGE:.*]] = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK: %{{.*}} = ave.hir.vabs %{{.*}}, %[[PGE]] : vector<64xf32>, vector<64xi1>
    %3 = math.absf %2 : vector<64xf32>
    vector.transfer_write %3, %subview_0[%c0], %1 {in_bounds = [true]} : vector<64xf32>, memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @hivm_regbase_cmp_test
func.func @hivm_regbase_cmp_test(%arg0: vector<64xf32>, %arg1: vector<64xf32>,%arg2: vector<64xi32>, %arg3: vector<64xi32>) -> (vector<64xi1>, vector<64xi1>){
  // CHECK: ave.hir.vcmp <GT>
  %0 = arith.cmpf ogt, %arg0, %arg1 : vector<64xf32>
  // CHECK: ave.hir.vcmp <EQ>
  %1 = arith.cmpi eq, %arg2, %arg3 : vector<64xi32>
  return %0, %1: vector<64xi1>,vector<64xi1>
}

// -----

// CHECK-LABEL: func.func @hivm_regbase_select_test
func.func @hivm_regbase_select_test(%arg0: vector<64xf32>, %arg1: vector<64xf32>,%arg2: vector<64xi1>) -> (vector<64xf32>){
  // CHECK: ave.hir.vsel
  %0 = arith.select %arg2, %arg0, %arg1 : vector<64xi1>, vector<64xf32>
  return %0: vector<64xf32>
}

// -----

// CHECK-LABEL: func.func @elemwise_absi_i8_absi_16x32
func.func @elemwise_absi_i8_absi_16x32(%arg0: memref<512xi8, #hivm.address_space<ub>>, %arg1: memref<512xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c512 step %c256 {
    %subview = memref.subview %arg0[%arg2] [256] [1] : memref<512xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [256] [1] : memref<512xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
    // CHECK: ave.hir.pge
    // CHECK: ave.hir.vabs
    %1 = math.absi %0 : vector<256xi8>
    vector.transfer_write %1, %subview_0[%c0] {in_bounds = [true]} : vector<256xi8>, memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_bitwise_i8_vand_16
func.func @elemwise_bitwise_i8_vand_16(%arg0: memref<16xi8, #hivm.address_space<ub>>, %arg1: memref<16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  // CHECK: ave.hir.vand
  %3 = arith.andi %1, %2 : vector<256xi8>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<16xi8, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_bitwise_i8_vor_16
func.func @elemwise_bitwise_i8_vor_16(%arg0: memref<16xi8, #hivm.address_space<ub>>, %arg1: memref<16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  // CHECK: ave.hir.vor
  %3 = arith.ori %1, %2 : vector<256xi8>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<16xi8, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_bitwise_i8_vxor_32x1000
func.func @elemwise_bitwise_i8_vxor_32x1000(%arg0: memref<32000xi8, #hivm.address_space<ub>>, %arg1: memref<32000xi8, #hivm.address_space<ub>>, %arg2: memref<32000xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c256 = arith.constant 256 : index
  %c32000 = arith.constant 32000 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c32000 step %c256 {
    %subview = memref.subview %arg0[%arg3] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
    %1 = vector.transfer_read %subview_0[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
    // CHECK: ave.hir.pge 
    // CHECK: ave.hir.vxor
    %2 = arith.xori %0, %1 : vector<256xi8>
    vector.transfer_write %2, %subview_1[%c0] {in_bounds = [true]} : vector<256xi8>, memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_vnot_i8_32x1000
func.func @elemwise_vnot_i8_32x1000(%arg0: memref<32000xi8, #hivm.address_space<ub>>, %arg1: memref<32000xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c32000 = arith.constant 32000 : index
  %c256 = arith.constant 256 : index
  %c0_i8 = arith.constant 0 : i8
  %c255_i16 = arith.constant 255 : i16
  %0 = ave.hir.pge <ALL> : vector<256xi1>
  %1 = ave.hir.broadcast %c255_i16, %0 : i16, vector<256xi1> -> vector<256xi8>
  scf.for %arg2 = %c0 to %c32000 step %c256 {
    %subview = memref.subview %arg0[%arg2] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %2 = vector.transfer_read %subview[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
    // CHECK: ave.hir.pge
    // CHECK: ave.hir.vnot
    %3 = arith.xori %2, %1 : vector<256xi8>
    vector.transfer_write %3, %subview_0[%c0] {in_bounds = [true]} : vector<256xi8>, memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_sub_f16_sub_16
func.func @elemwise_sub_f16_sub_16(%arg0: memref<16xf16, #hivm.address_space<ub>>, %arg1: memref<16xf16, #hivm.address_space<ub>>, %arg2: memref<16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<128xf16>
  %2 = vector.transfer_read %arg1[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<128xf16>
  // CHECK: ave.hir.vsub
  %3 = arith.subf %1, %2 : vector<128xf16>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<16xf16, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_ceilfloor_f16_ceil_16
func.func @elemwise_ceilfloor_f16_ceil_16(%arg0: memref<16xf16, #hivm.address_space<ub>>, %arg1: memref<16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<64xi1>
  %1 = vector.constant_mask [32] : vector<128xi1>
  %2 = builtin.unrealized_conversion_cast %1 : vector<128xi1> to vector<64xi1>
  %3 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<64xf16>
  //  CHECK: ave.hir.vextf
  //  CHECK: ave.hir.pge
  //  CHECK: ave.hir.vtrc 
  //  CHECK: ave.hir.pge 
  //  CHECK: ave.hir.vtruncf
  %4 = arith.extf %3 : vector<64xf16> to vector<64xf32>
  %5 = math.round %4 {round_mode = #hfusion.round_mode<ceil>} : vector<64xf32>
  %6 = arith.truncf %5 {round_mode = #hfusion.round_mode<ceil>} : vector<64xf32> to vector<64xf16>
  vector.transfer_write %6, %arg1[%c0], %2 {in_bounds = [true]} : vector<64xf16>, memref<16xf16, #hivm.address_space<ub>>
  return
}

// -----


// CHECK-LABEL: func.func @elemwise_div_signed_f16_div_16
func.func @elemwise_div_signed_f16_div_16(%arg0: memref<16xf16, #hivm.address_space<ub>>, %arg1: memref<16xf16, #hivm.address_space<ub>>, %arg2: memref<16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<128xf16>
  %2 = vector.transfer_read %arg1[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<128xf16>
  // CHECK: ave.hir.vdiv
  %3 = arith.divf %1, %2 : vector<128xf16>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<16xf16, #hivm.address_space<ub>>
  return
}
// -----


// CHECK-LABEL: func.func @elemwise_exp_f16_exp_16
func.func @elemwise_exp_f16_exp_16(%arg0: memref<16xf16, #hivm.address_space<ub>>, %arg1: memref<16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<128xf16>
  // CHECK: ave.hir.vexp
  %2 = math.exp %1 : vector<128xf16>
  vector.transfer_write %2, %arg1[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<16xf16, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_hfusion_unary_explog_f16_exp2_16
func.func @elemwise_hfusion_unary_explog_f16_exp2_16(%arg0: memref<16xf16, #hivm.address_space<ub>>, %arg1: memref<16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 0.693147182 : f32
  %0 = ave.hir.pge <ALL> : vector<256xi1>
  %1 = ave.hir.broadcast %cst_0, %0 : f32, vector<256xi1> -> vector<64xf32>
  %2 = vector.constant_mask [16] : vector<64xi1>
  %3 = vector.constant_mask [32] : vector<128xi1>
  %4 = builtin.unrealized_conversion_cast %3 : vector<128xi1> to vector<64xi1>
  %5 = vector.transfer_read %arg0[%c0], %cst, %2 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<64xf16>
  %6 = arith.extf %5 : vector<64xf16> to vector<64xf32>
  %7 = arith.mulf %6, %1 : vector<64xf32>
  // CHECK: ave.hir.vexp
  %8 = math.exp %7 : vector<64xf32>
  %9 = arith.truncf %8 {round_mode = #hfusion.round_mode<round>} : vector<64xf32> to vector<64xf16>
  vector.transfer_write %9, %arg1[%c0], %4 {in_bounds = [true]} : vector<64xf16>, memref<16xf16, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_log_f16_log_1024
func.func @elemwise_log_f16_log_1024(%arg0: memref<1024xf16, #hivm.address_space<ub>>, %arg1: memref<1024xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c1024 step %c64 {
    %subview = memref.subview %arg0[%arg2] [64] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [64] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf16>
    %1 = arith.extf %0 : vector<64xf16> to vector<64xf32>
    // CHECK: ave.hir.vln
    %2 = math.log %1 : vector<64xf32>
    %3 = arith.truncf %2 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xf16>
    vector.transfer_write %3, %subview_0[%c0] {in_bounds = [true]} : vector<64xf16>, memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_maxminf_f16_maxf_16x32
func.func @elemwise_maxminf_f16_maxf_16x32(%arg0: memref<512xf16, #hivm.address_space<ub>>, %arg1: memref<512xf16, #hivm.address_space<ub>>, %arg2: memref<512xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c512 step %c128 {
    %subview = memref.subview %arg0[%arg3] [128] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [128] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [128] [1] : memref<512xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xf16>
    %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xf16>
    // CHECK: ave.hir.pge
    // CHECK: ave.hir.vmax
    %2 = arith.maximumf %0, %1 : vector<128xf16>
    vector.transfer_write %2, %subview_1[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_maxminf_f16_minf_1024
func.func @elemwise_maxminf_f16_minf_1024(%arg0: memref<1024xf16, #hivm.address_space<ub>>, %arg1: memref<1024xf16, #hivm.address_space<ub>>, %arg2: memref<1024xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c1024 step %c128 {
    %subview = memref.subview %arg0[%arg3] [128] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [128] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [128] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xf16>
    %1 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xf16>
    // CHECK: ave.hir.pge
    // CHECK: ave.hir.vmin
    %2 = arith.minimumf %0, %1 : vector<128xf16>
    vector.transfer_write %2, %subview_1[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----


// CHECK-LABEL: func.func @elemwise_maxmini_i8_max_signed_16
func.func @elemwise_maxmini_i8_max_signed_16(%arg0: memref<16xi8, #hivm.address_space<ub>>, %arg1: memref<16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  // CHECK: ave.hir.vsmax
  %3 = arith.maxsi %1, %2 : vector<256xi8>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<16xi8, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_maxmini_i8_min_signed_16
func.func @elemwise_maxmini_i8_min_signed_16(%arg0: memref<16xi8, #hivm.address_space<ub>>, %arg1: memref<16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  // CHECK: ave.hir.vsmin
  %3 = arith.minsi %1, %2 : vector<256xi8>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<16xi8, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_maxmini_i8_max_unsigned_16
func.func @elemwise_maxmini_i8_max_unsigned_16(%arg0: memref<16xi8, #hivm.address_space<ub>>, %arg1: memref<16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  // CHECK: ave.hir.vumax
  %3 = arith.maxui %1, %2 : vector<256xi8>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<16xi8, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_maxmini_i8_min_unsigned_16
func.func @elemwise_maxmini_i8_min_unsigned_16(%arg0: memref<16xi8, #hivm.address_space<ub>>, %arg1: memref<16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index                                                                                      %0 = vector.constant_mask [16] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<256xi8>
  // CHECK: ave.hir.vumin
  %3 = arith.minui %1, %2 : vector<256xi8>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<16xi8, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_negf_f16_negf_16
func.func @elemwise_negf_f16_negf_16(%arg0: memref<16xf16, #hivm.address_space<ub>>, %arg1: memref<16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<-1.000000e+00> : vector<128xf16>
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst_0, %0 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<128xf16>
  //  CHECK: ave.hir.vmul
  %2 = arith.mulf %1, %cst : vector<128xf16>
  vector.transfer_write %2, %arg1[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<16xf16, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_recsqrt_f16_rec_16
func.func @elemwise_recsqrt_f16_rec_16(%arg0: memref<16xf16, #hivm.address_space<ub>>, %arg1: memref<16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = ave.hir.pge <ALL> : vector<256xi1>
  %1 = ave.hir.broadcast %cst_0, %0 : f32, vector<256xi1> -> vector<64xf32>
  %2 = vector.constant_mask [16] : vector<64xi1>
  %3 = vector.constant_mask [32] : vector<128xi1>
  %4 = builtin.unrealized_conversion_cast %3 : vector<128xi1> to vector<64xi1>
  %5 = vector.transfer_read %arg0[%c0], %cst, %2 {in_bounds = [true]} : memref<16xf16, #hivm.address_space<ub>>, vector<64xf16>
  // CHECK: ave.hir.vextf 
  // CHECK: ave.hir.vdiv 
  // CHECK: ave.hir.pge
  // CHECK: ave.hir.vtruncf 
  %6 = arith.extf %5 : vector<64xf16> to vector<64xf32>
  %7 = arith.divf %1, %6 : vector<64xf32>
  %8 = arith.truncf %7 {round_mode = #hfusion.round_mode<rint>} : vector<64xf32> to vector<64xf16>
  vector.transfer_write %8, %arg1[%c0], %4 {in_bounds = [true]} : vector<64xf16>, memref<16xf16, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_shli_i8_32x1000
func.func @elemwise_shli_i8_32x1000(%arg0: memref<32000xi8, #hivm.address_space<ub>>, %arg1: memref<32000xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<2> : vector<256xi8>
  %c0_i8 = arith.constant 0 : i8
  %c256 = arith.constant 256 : index
  %c32000 = arith.constant 32000 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c32000 step %c256 {
    %subview = memref.subview %arg0[%arg2] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
    // CHECK: ave.hir.pge 
    // CHECK: ave.hir.vshl
    %1 = arith.shli %0, %cst : vector<256xi8>
    vector.transfer_write %1, %subview_0[%c0] {in_bounds = [true]} : vector<256xi8>, memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_shrsi_i8_32x1000
func.func @elemwise_shrsi_i8_32x1000(%arg0: memref<32000xi8, #hivm.address_space<ub>>, %arg1: memref<32000xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<2> : vector<256xi8>
  %c0_i8 = arith.constant 0 : i8
  %c256 = arith.constant 256 : index
  %c32000 = arith.constant 32000 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c32000 step %c256 {
    %subview = memref.subview %arg0[%arg2] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [256] [1] : memref<32000xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
    // CHECK: ave.hir.pge 
    // CHECK: ave.hir.vshr
    %1 = arith.shrsi %0, %cst : vector<256xi8>
    vector.transfer_write %1, %subview_0[%c0] {in_bounds = [true]} : vector<256xi8>, memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----


// CHECK-LABEL: func.func @elemwise_shrui_i32_32x1000
func.func @elemwise_shrui_i32_32x1000(%arg0: index, %arg1: memref<?xi32, #hivm.address_space<ub>>, %arg2: memref<?xi32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = ave.hir.pge <ALL> : vector<256xi1>
  %1 = ave.hir.broadcast %c2_i32, %0 : i32, vector<256xi1> -> vector<64xi32>
  scf.for %arg3 = %c0 to %arg0 step %c64 {
    %2 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg3)[%arg0]
    %subview = memref.subview %arg1[%arg3] [%2] [1] : memref<?xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg2[%arg3] [%2] [1] : memref<?xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %3 = vector.create_mask %2 : vector<64xi1>
    %4 = vector.transfer_read %subview[%c0], %c0_i32, %3 {in_bounds = [true]} : memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi32>
    // CHECK: ave.hir.pge 
    // CHECK: ave.hir.vshr   
    %5 = arith.shrui %4, %1 : vector<64xi32>
    vector.transfer_write %5, %subview_0[%c0], %3 {in_bounds = [true]} : vector<64xi32>, memref<?xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @elemwise_sub_i8_sub_16x32
func.func @elemwise_sub_i8_sub_16x32(%arg0: memref<512xi8, #hivm.address_space<ub>>, %arg1: memref<512xi8, #hivm.address_space<ub>>, %arg2: memref<512xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c512 step %c256 {
    %subview = memref.subview %arg0[%arg3] [256] [1] : memref<512xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [256] [1] : memref<512xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [256] [1] : memref<512xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
    %1 = vector.transfer_read %subview_0[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
    // CHECK: ave.hir.pge 
    // CHECK: ave.hir.vsub   
    %2 = arith.subi %0, %1 : vector<256xi8>
    vector.transfer_write %2, %subview_1[%c0] {in_bounds = [true]} : vector<256xi8>, memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_dense_constant_unit_dim
func.func @test_dense_constant_unit_dim(%arg0: memref<2x8xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<0xFF800000> : vector<1x64xf32>
  // CHECK: ave.hir.broadcast
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [1, 8] : vector<1x64xi1>
  scf.for %arg1 = %c0 to %c2 step %c1 {
    %subview = memref.subview %arg0[%arg1, 0] [1, 8] [1, 1] : memref<2x8xf32, #hivm.address_space<ub>> to memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    vector.transfer_write %cst, %subview[%c0, %c0], %0 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x8xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_preg_logic_ops
func.func @test_preg_logic_ops(%arg0: memref<64xi8, #hivm.address_space<ub>>, %arg1: memref<64xi8, #hivm.address_space<ub>>, %arg2: memref<64xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<1> : vector<256xi8>
  %cst_0 = arith.constant dense<0> : vector<256xi8>
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [64] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<64xi8, #hivm.address_space<ub>>, vector<256xi8>
  annotation.mark %1 {reached_out_ops_idx = [0 : i32]} : vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<64xi8, #hivm.address_space<ub>>, vector<256xi8>
  annotation.mark %2 {reached_out_ops_idx = [0 : i32]} : vector<256xi8>
  %3 = arith.cmpi ne, %2, %cst_0 : vector<256xi8>
  annotation.mark %3 {reached_out_ops_idx = [0 : i32]} : vector<256xi1>
  %4 = arith.cmpi ne, %1, %cst_0 : vector<256xi8>
  annotation.mark %4 {reached_out_ops_idx = [0 : i32]} : vector<256xi1>
  // CHECK: ave.hir.preg.or 
  %5 = arith.ori %4, %3 : vector<256xi1>
  // CHECK: ave.hir.preg.and 
  %6 = arith.andi %4, %3 : vector<256xi1>
  // CHECK: ave.hir.preg.xor 
  %7 = arith.xori %4, %3: vector<256xi1>
  annotation.mark %5 {reached_out_ops_idx = [0 : i32]} : vector<256xi1>
  annotation.mark %6 {reached_out_ops_idx = [0 : i32]} : vector<256xi1>
  annotation.mark %7 {reached_out_ops_idx = [0 : i32]} : vector<256xi1>
  %8 = arith.select %5, %cst, %cst_0 : vector<256xi1>, vector<256xi8>
  annotation.mark %8 {reached_out_ops_idx = [0 : i32]} : vector<256xi8>
  vector.transfer_write %8, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<64xi8, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @test_dense_constant_vci
func.func @test_dense_constant_vci(%arg0: memref<1024xi32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : vector<64xi32>
  // CHECK: ave.hir.vci
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  scf.for %arg1 = %c0 to %c1024 step %c64 {
    %subview = memref.subview %arg0[%arg1] [64] [1] : memref<1024xi32, #hivm.address_space<ub>> to memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = arith.index_cast %arg1 : index to i32
    %1 = vector.broadcast %0 : i32 to vector<64xi32>
    %2 = arith.addi %1, %cst : vector<64xi32>
    vector.transfer_write %2, %subview[%c0] {in_bounds = [true]} : vector<64xi32>, memref<64xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_vci
func.func @test_vci(%arg0: memref<512x16xf16, #hivm.address_space<ub>>, %arg1: memref<16x512xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  // CHECK: %[[CONST0:.*]] = arith.constant 0 : i16 
  // CHECK-NEXT: %[[VCI:.*]] = ave.hir.vci %[[CONST0]], <INCREASE> : i16, vector<128xi16>
  // CHECK-NEXT: %[[PGE:.*]] = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK-NEXT: %[[CONST16:.*]] = arith.constant 16 : i16
  // CHECK-NEXT: %[[VMULS:.*]] = ave.hir.vmuls %[[VCI]], %[[CONST16]], %[[PGE]] : vector<128xi16>, i16, vector<128xi1>
  %cst = arith.constant dense<"0x0000100020003000400050006000700080009000A000B000C000D000E000F0000001100120013001400150016001700180019001A001B001C001D001E001F0010002100220023002400250026002700280029002A002B002C002D002E002F0020003100320033003400350036003700380039003A003B003C003D003E003F0030004100420043004400450046004700480049004A004B004C004D004E004F0040005100520053005400550056005700580059005A005B005C005D005E005F0050006100620063006400650066006700680069006A006B006C006D006E006F0060007100720073007400750076007700780079007A007B007C007D007E007F007"> : vector<128xi16>
  return
}

// -----

// CHECK-LABEL: func.func @test_ule_unsigned_i8
func.func @test_ule_unsigned_i8(%arg0: memref<3978xi8, #hivm.address_space<ub>>, %arg1: memref<3978xi8, #hivm.address_space<ub>>, %arg2: memref<3978xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<1> : vector<256xi8>
  %cst_0 = arith.constant dense<0> : vector<256xi8>
  %c256 = arith.constant 256 : index
  %c3978 = arith.constant 3978 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c3978 step %c256 {
    %0 = affine.min affine_map<(d0) -> (-d0 + 3978, 256)>(%arg3)
    %subview = memref.subview %arg0[%arg3] [%0] [1] : memref<3978xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg1[%arg3] [%0] [1] : memref<3978xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg2[%arg3] [%0] [1] : memref<3978xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res, %new_true_shape = ave.hir.plt %0 {mask_op_idx = 0 : i32} : vector<256xi1>, index
    annotation.mark %res {mask_op_idx = 0 : i32} : vector<256xi1>
    %1 = ave.hir.vload <NORM> %subview[%c0] : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>
    annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<256xi8>
    %2 = ave.hir.vload <NORM> %subview_1[%c0] : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>
    annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<256xi8>

    // check : ave.hir.vcmp <ULE>
    %3 = arith.cmpi ule, %1, %2 : vector<256xi8>
    annotation.mark %3 {reached_mask_ops_idx = 0 : i32} : vector<256xi1>
    %4 = arith.select %3, %cst, %cst_0 : vector<256xi1>, vector<256xi8>
    annotation.mark %4 {reached_mask_ops_idx = 0 : i32} : vector<256xi8>
    ave.hir.masked_store <NORM_B8> %subview_2[%c0], %res, %4 : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  }
  return
}
