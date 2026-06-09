// RUN: bishengir-opt %s -convert-vector-to-hivmave -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

func.func @elemwise_abs_test__1xf32(%arg0: memref<1xf32, #hivm.address_space<ub>>, %arg1: memref<1xf32, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <VL1> : vector<64xi1>
  %0 = vector.constant_mask [1] : vector<64xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<1xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = ave.hir.vabs %2, %1 : vector<64xf32>, vector<256xi1>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<1xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__2xf32(%arg0: memref<2xf32, #hivm.address_space<ub>>, %arg1: memref<2xf32, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <VL2> : vector<64xi1>
  %0 = vector.constant_mask [2] : vector<64xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<2xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = ave.hir.vabs %2, %1 : vector<64xf32>, vector<256xi1>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<2xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__3xf32(%arg0: memref<3xf32, #hivm.address_space<ub>>, %arg1: memref<3xf32, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <VL3> : vector<64xi1>
  %0 = vector.constant_mask [3] : vector<64xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<3xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = ave.hir.vabs %2, %1 : vector<64xf32>, vector<256xi1>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<3xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__4xf32(%arg0: memref<4xf32, #hivm.address_space<ub>>, %arg1: memref<4xf32, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK:  %0 = ave.hir.pge <VL4> : vector<64xi1>
  %0 = vector.constant_mask [4] : vector<64xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<4xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = ave.hir.vabs %2, %1 : vector<64xf32>, vector<256xi1>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<4xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__7xf32(%arg0: memref<7xf32, #hivm.address_space<ub>>, %arg1: memref<7xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %c7 = arith.constant 7 : index
  // CHECK-NEXT: %res, %new_true_shape = ave.hir.plt %c7 : vector<64xi1>, index
  %0 = vector.constant_mask [7] : vector<64xi1>
  // CHECK: %res_0 = ave.hir.vload <NORM> %arg0[%c0] : memref<7xf32, #hivm.address_space<ub>> into vector<64xf32>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<7xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %3 = ave.hir.vabs %1, %2 : vector<64xf32>, vector<256xi1>
  // CHECK: ave.hir.masked_store <NORM_B32> %arg1[%c0], %res, %1 : memref<7xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<7xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__8xf32(%arg0: memref<8xf32, #hivm.address_space<ub>>, %arg1: memref<8xf32, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <VL8> : vector<64xi1>
  %0 = vector.constant_mask [8] : vector<64xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<8xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = ave.hir.vabs %2, %1 : vector<64xf32>, vector<256xi1>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<8xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__15xf32(%arg0: memref<15xf32, #hivm.address_space<ub>>, %arg1: memref<15xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %c15 = arith.constant 15 : index
  // CHECK-NEXT: %res, %new_true_shape = ave.hir.plt %c15 : vector<64xi1>, index
  %0 = vector.constant_mask [15] : vector<64xi1>
  // CHECK: %res_0 = ave.hir.vload <NORM> %arg0[%c0] : memref<15xf32, #hivm.address_space<ub>> into vector<64xf32>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<15xf32, #hivm.address_space<ub>>, vector<64xf32>
  // CHECK: %0 = builtin.unrealized_conversion_cast %res : vector<64xi1> to vector<256xi1>
  %2 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %3 = ave.hir.vabs %1, %2 : vector<64xf32>, vector<256xi1>
  // CHECK: ave.hir.masked_store <NORM_B32> %arg1[%c0], %res, %1 : memref<15xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<15xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__16xf32(%arg0: memref<16xf32, #hivm.address_space<ub>>, %arg1: memref<16xf32, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <VL16> : vector<64xi1>
  %0 = vector.constant_mask [16] : vector<64xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<16xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = ave.hir.vabs %2, %1 : vector<64xf32>, vector<256xi1>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<16xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__32xf32(%arg0: memref<32xf32, #hivm.address_space<ub>>, %arg1: memref<32xf32, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <VL32> : vector<64xi1>
  %0 = vector.constant_mask [32] : vector<64xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<32xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = ave.hir.vabs %2, %1 : vector<64xf32>, vector<256xi1>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<32xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__64xf32(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <ALL> : vector<64xi1>
  %0 = vector.constant_mask [64] : vector<64xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<64xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
  %3 = ave.hir.vabs %2, %1 : vector<64xf32>, vector<256xi1>
  // CHECK: %3 = ave.hir.pge <ALL> : vector<64xi1>
  // CHECK-NEXT: ave.hir.masked_store <NORM_B32> %arg1[%c0], %3, %2 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  vector.transfer_write %3, %arg1[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__7xf16(%arg0: memref<7xf16, #hivm.address_space<ub>>, %arg1: memref<7xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  // CHECK: %c7 = arith.constant 7 : index
  // CHECK-NEXT: %res, %new_true_shape = ave.hir.plt %c7 : vector<128xi1>, index
  %0 = vector.constant_mask [7] : vector<128xi1>
  // CHECK: %res_0 = ave.hir.vload <NORM> %arg0[%c0] : memref<7xf16, #hivm.address_space<ub>> into vector<128xf16>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<7xf16, #hivm.address_space<ub>>, vector<128xf16>
  %2 = builtin.unrealized_conversion_cast %0 : vector<128xi1> to vector<256xi1>
  %3 = ave.hir.vabs %1, %2 : vector<128xf16>, vector<256xi1>
  // CHECK: ave.hir.masked_store <NORM_B16> %arg1[%c0], %res, %1 : memref<7xf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<7xf16, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__15xf16(%arg0: memref<15xf16, #hivm.address_space<ub>>, %arg1: memref<15xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  // CHECK: %c15 = arith.constant 15 : index
  // CHECK-NEXT: %res, %new_true_shape = ave.hir.plt %c15 : vector<128xi1>, index
  %0 = vector.constant_mask [15] : vector<128xi1>
  // CHECK: %res_0 = ave.hir.vload <NORM> %arg0[%c0] : memref<15xf16, #hivm.address_space<ub>> into vector<128xf16>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<15xf16, #hivm.address_space<ub>>, vector<128xf16>
  %2 = builtin.unrealized_conversion_cast %0 : vector<128xi1> to vector<256xi1>
  %3 = ave.hir.vabs %1, %2 : vector<128xf16>, vector<256xi1>
  // CHECK: ave.hir.masked_store <NORM_B16> %arg1[%c0], %res, %1 : memref<15xf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<15xf16, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__64xf16(%arg0: memref<64xf16, #hivm.address_space<ub>>, %arg1: memref<64xf16, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <VL64> : vector<128xi1>
  %0 = vector.constant_mask [64] : vector<128xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<128xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<64xf16, #hivm.address_space<ub>>, vector<128xf16>
  %3 = ave.hir.vabs %2, %1 : vector<128xf16>, vector<256xi1>
  vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<64xf16, #hivm.address_space<ub>>
  return
}

func.func @elemwise_abs_test__128xf16(%arg0: memref<128xf16, #hivm.address_space<ub>>, %arg1: memref<128xf16, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <ALL> : vector<128xi1>
  %0 = vector.constant_mask [128] : vector<128xi1>
  %1 = builtin.unrealized_conversion_cast %0 : vector<128xi1> to vector<256xi1>
  %2 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<128xf16, #hivm.address_space<ub>>, vector<128xf16>
  %3 = ave.hir.vabs %2, %1 : vector<128xf16>, vector<256xi1>
  // CHECK: %3 = ave.hir.pge <ALL> : vector<128xi1>
  // CHECK-NEXT: ave.hir.masked_store <NORM_B16> %arg1[%c0], %3, %2 : memref<128xf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
  vector.transfer_write %3, %arg1[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, #hivm.address_space<ub>>
  return
}

func.func @elemwise_add_test__7xf16(%arg0: memref<7xf16, #hivm.address_space<ub>>, %arg1: memref<7xf16, #hivm.address_space<ub>>, %arg2: memref<7xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  // CHECK: %c7 = arith.constant 7 : index
  // CHECK-NEXT: %res, %new_true_shape = ave.hir.plt %c7 : vector<128xi1>, index
  %0 = vector.constant_mask [7] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst, %0 {in_bounds = [true]} : memref<7xf16, #hivm.address_space<ub>>, vector<128xf16>
  %2 = vector.transfer_read %arg1[%c0], %cst, %0 {in_bounds = [true]} : memref<7xf16, #hivm.address_space<ub>>, vector<128xf16>
  %3 = builtin.unrealized_conversion_cast %0 : vector<128xi1> to vector<256xi1>
  %4 = ave.hir.vadd %1, %2, %3 : vector<128xf16>, vector<256xi1>
  // CHECK: ave.hir.masked_store <NORM_B16> %arg2[%c0], %res, %1 : memref<7xf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
  vector.transfer_write %4, %arg2[%c0], %0 {in_bounds = [true]} : vector<128xf16>, memref<7xf16, #hivm.address_space<ub>>
  return
}


func.func @elemwise_add_test__128xi8(%arg0: memref<128xi8, #hivm.address_space<ub>>, %arg1: memref<128xi8, #hivm.address_space<ub>>, %arg2: memref<128xi8, #hivm.address_space<ub>>) attributes {hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>, hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  // CHECK: %0 = ave.hir.pge <VL128> : vector<256xi1>
  %0 = vector.constant_mask [128] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<128xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<128xi8, #hivm.address_space<ub>>, vector<256xi8>
  %3 = ave.hir.vadd %1, %2, %0 : vector<256xi8>, vector<256xi1>
  // CHECK: ave.hir.masked_store <NORM_B8> %arg2[%c0], %0, %1 : memref<128xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  vector.transfer_write %3, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<128xi8, #hivm.address_space<ub>>
  return
}


func.func @elemwise_add_test__129xi8(%arg0: memref<129xi8, #hivm.address_space<ub>>, %arg1: memref<129xi8, #hivm.address_space<ub>>, %arg2: memref<129xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  // CHECK: %[[CONST129:.+]] = arith.constant 129 : index
  // CHECK-NEXT: %res, %new_true_shape = ave.hir.plt %[[CONST129]] : vector<256xi1>, index
  %0 = vector.constant_mask [129] : vector<256xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<129xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = vector.transfer_read %arg1[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<129xi8, #hivm.address_space<ub>>, vector<256xi8>
  %3 = builtin.unrealized_conversion_cast %0 : vector<256xi1> to vector<256xi1>
  %4 = ave.hir.vadd %1, %2, %3 : vector<256xi8>, vector<256xi1>
  // CHECK: ave.hir.masked_store <NORM_B8> %arg2[%c0], %res, %{{.*}} : memref<129xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  vector.transfer_write %4, %arg2[%c0], %0 {in_bounds = [true]} : vector<256xi8>, memref<129xi8, #hivm.address_space<ub>>
  return
}

func.func @elemwise_add_test__256xi8(%arg0: memref<256xi8, #hivm.address_space<ub>>, %arg1: memref<256xi8, #hivm.address_space<ub>>, %arg2: memref<256xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, #hivm.address_space<ub>>, vector<256xi8>
  %1 = vector.transfer_read %arg1[%c0], %c0_i8 {in_bounds = [true]} : memref<256xi8, #hivm.address_space<ub>>, vector<256xi8>
  %2 = ave.hir.pge <ALL> : vector<256xi1>
  %3 = ave.hir.vadd %0, %1, %2 : vector<256xi8>, vector<256xi1>
  // CHECK: %2 = ave.hir.pge <ALL> : vector<256xi1>
  // CHECK-NEXT: ave.hir.masked_store <NORM_B8> %arg2[%c0], %2, %1 : memref<256xi8, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  vector.transfer_write %3, %arg2[%c0] {in_bounds = [true]} : vector<256xi8>, memref<256xi8, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @test_vector_brc_reduction_res_lowest_element
func.func @test_vector_brc_reduction_res_lowest_element(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x16xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64x16xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [1, 16] : vector<1x64xi1>
  scf.for %arg4 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg0[%arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg4, 0] [1, 16] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %1 = vector.transfer_read %subview_0[%c0, %c0], %cst, %0 {in_bounds = [true, true]} : memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %2 = vector.transfer_read %subview_1[%c0], %cst {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
    %3 = builtin.unrealized_conversion_cast %0 : vector<1x64xi1> to vector<64xi1>
    %4 = builtin.unrealized_conversion_cast %1 : vector<1x64xf32> to vector<64xf32>
    %5 = builtin.unrealized_conversion_cast %2 : vector<1xf32> to vector<64xf32>
    %6 = ave.hir.reduction <max>, %4, %3 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    %8 = ave.hir.vmax %5, %6, %7 : vector<64xf32>, vector<64xi1>
    %9 = builtin.unrealized_conversion_cast %8 : vector<64xf32> to vector<1xf32>
    %subview_2 = memref.subview %arg3[%arg4, 0] [1, 16] [1, 1] : memref<64x16xf32, #hivm.address_space<ub>> to memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %10 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    // CHECK: ave.hir.vector_broadcast
    %11 = vector.broadcast %9 : vector<1xf32> to vector<1x64xf32>
    %12 = vector.shape_cast %10 : vector<1x64xf32> to vector<64xf32>
    %13 = vector.shape_cast %11 : vector<1x64xf32> to vector<64xf32>
    %14 = ave.hir.pge <ALL> : vector<64xi1>
    %15 = ave.hir.vmax %12, %13, %14 : vector<64xf32>, vector<64xi1>
    %16 = vector.shape_cast %15 : vector<64xf32> to vector<1x64xf32>
    vector.transfer_write %16, %subview_2[%c0, %c0], %0 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @test_vector_gather
func.func @test_vector_gather(%arg0: memref<512x16xf16, #hivm.address_space<ub>>, %arg1: memref<16x512xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<0.000000e+00> : vector<128xf16>
  %cst_0 = arith.constant dense<"0x0000100020003000400050006000700080009000A000B000C000D000E000F0000001100120013001400150016001700180019001A001B001C001D001E001F0010002100220023002400250026002700280029002A002B002C002D002E002F0020003100320033003400350036003700380039003A003B003C003D003E003F0030004100420043004400450046004700480049004A004B004C004D004E004F0040005100520053005400550056005700580059005A005B005C005D005E005F0050006100620063006400650066006700680069006A006B006C006D006E006F0060007100720073007400750076007700780079007A007B007C007D007E007F007"> : vector<128xi16>
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  scf.for %arg2 = %c0 to %c16 step %c1 {
    scf.for %arg3 = %c0 to %c512 step %c128 {
      %subview = memref.subview %arg0[%arg3, %arg2] [128, 1] [1, 1] : memref<512x16xf16, #hivm.address_space<ub>> to memref<128x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg1[%arg2, %arg3] [1, 128] [1, 1] : memref<16x512xf16, #hivm.address_space<ub>> to memref<1x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.constant_mask [128] : vector<128xi1>
      %1 = vector.gather %subview[%c0, %c0] [%cst_0], %0, %cst : memref<128x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1>, vector<128xf16> into vector<128xf16>
      //CHECK: %[[PGE:.*]] = ave.hir.pge <ALL> : vector<128xi1>
      //CHECK-NEXT: %[[GATHER:.*]] = ave.hir.vgather %subview[%c0, %c0] [%cst_0], %[[PGE]] : memref<128x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1> into vector<128xf16>
      %subview_2 = memref.subview %subview_1[0, 0] [1, 128] [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to memref<128xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      vector.transfer_write %1, %subview_2[%c0] {in_bounds = [true]} : vector<128xf16>, memref<128xf16, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      //CHECK: %[[PGE1:.*]] = ave.hir.pge <ALL> : vector<128xi1>
      //CHECK-NEXT: ave.hir.masked_store <NORM_B16> %subview_2[%c0], %[[PGE1]], %[[GATHER]] : memref<128xf16, #map, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    }
  }
  return
}

// CHECK-LABEL: func.func @test_vsstb
func.func @test_vsstb(%arg0: memref<64x4x16xf16, #hivm.address_space<ub>>, %arg1: memref<4x64x16xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c64 step %c1 {
    %subview = memref.subview %arg1[0, %arg2, 0] [4, 1, 16] [1, 1, 1] : memref<4x64x16xf16, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1024, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg2, 0, 0] [1, 4, 16] [1, 1, 1] : memref<64x4x16xf16, #hivm.address_space<ub>> to memref<4x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<4x16xf16>
    %subview_1 = memref.subview %subview[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[1024, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + d1 + s0)>, #hivm.address_space<ub>>
    //CHECK: %c1024 = arith.constant 1024 : index
    //CHECK: ave.hir.store_with_stride %subview_1[%c0, %c0], %c1024, %[[V1:.*]], %[[V2:.*]] :
    vector.transfer_write %0, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<4x16xf16>, memref<4x16xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + d1 + s0)>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @test_vsstb_f8E4M3FN
func.func @test_vsstb_f8E4M3FN(%arg0: memref<128x4x32xf8E4M3FN, #hivm.address_space<ub>>, %arg1: memref<4x128x32xf8E4M3FN, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f8E4M3FN
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c128 step %c1 {
    %subview = memref.subview %arg1[0, %arg2, 0] [4, 1, 32] [1, 1, 1] : memref<4x128x32xf8E4M3FN, #hivm.address_space<ub>> to memref<4x1x32xf8E4M3FN, strided<[4096, 32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg2, 0, 0] [1, 4, 32] [1, 1, 1] : memref<128x4x32xf8E4M3FN, #hivm.address_space<ub>> to memref<4x32xf8E4M3FN, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [false, true]} : memref<4x32xf8E4M3FN, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<8x32xf8E4M3FN>
    %1 = vector.constant_mask [128] : vector<256xi1>
    %2 = vector.shape_cast %1 : vector<256xi1> to vector<8x32xi1>
    annotation.mark %2 {mask_op_idx = -1 : i32} : vector<8x32xi1>
    %subview_1 = memref.subview %subview[0, 0, 0] [4, 1, 32] [1, 1, 1] : memref<4x1x32xf8E4M3FN, strided<[4096, 32, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x32xf8E4M3FN, affine_map<(d0, d1)[s0] -> (d0 * 4096 + d1 + s0)>, #hivm.address_space<ub>>
    //CHECK: %c4096 = arith.constant 4096 : index
    //CHECK: ave.hir.store_with_stride %subview_1[%c0, %c0], %c4096, %[[V1:.*]], %[[V2:.*]] :
    vector.transfer_write %0, %subview_1[%c0, %c0], %2 {in_bounds = [true, true]} : vector<8x32xf8E4M3FN>, memref<4x32xf8E4M3FN, affine_map<(d0, d1)[s0] -> (d0 * 4096 + d1 + s0)>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @test_vsstb_f8E5M2
func.func @test_vsstb_f8E5M2(%arg0: memref<128x4x32xf8E5M2, #hivm.address_space<ub>>, %arg1: memref<4x128x32xf8E5M2, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f8E5M2
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c128 step %c1 {
    %subview = memref.subview %arg1[0, %arg2, 0] [4, 1, 32] [1, 1, 1] : memref<4x128x32xf8E5M2, #hivm.address_space<ub>> to memref<4x1x32xf8E5M2, strided<[4096, 32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg2, 0, 0] [1, 4, 32] [1, 1, 1] : memref<128x4x32xf8E5M2, #hivm.address_space<ub>> to memref<4x32xf8E5M2, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [false, true]} : memref<4x32xf8E5M2, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<8x32xf8E5M2>
    %1 = vector.constant_mask [128] : vector<256xi1>
    %2 = vector.shape_cast %1 : vector<256xi1> to vector<8x32xi1>
    annotation.mark %2 {mask_op_idx = -1 : i32} : vector<8x32xi1>
    %subview_1 = memref.subview %subview[0, 0, 0] [4, 1, 32] [1, 1, 1] : memref<4x1x32xf8E5M2, strided<[4096, 32, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x32xf8E5M2, affine_map<(d0, d1)[s0] -> (d0 * 4096 + d1 + s0)>, #hivm.address_space<ub>>
    //CHECK: %c4096 = arith.constant 4096 : index
    //CHECK: ave.hir.store_with_stride %subview_1[%c0, %c0], %c4096, %[[V1:.*]], %[[V2:.*]] :
    vector.transfer_write %0, %subview_1[%c0, %c0], %2 {in_bounds = [true, true]} : vector<8x32xf8E5M2>, memref<4x32xf8E5M2, affine_map<(d0, d1)[s0] -> (d0 * 4096 + d1 + s0)>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @test_vsstb_i32
func.func @test_vsstb_i32(%arg0: memref<128x4x8xi32, #hivm.address_space<ub>>, %arg1: memref<4x128x8xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c128 step %c1 {
    %subview = memref.subview %arg1[0, %arg2, 0] [4, 1, 8] [1, 1, 1] : memref<4x128x8xi32, #hivm.address_space<ub>> to memref<4x1x8xi32, strided<[1024, 8, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg2, 0, 0] [1, 4, 8] [1, 1, 1] : memref<128x4x8xi32, #hivm.address_space<ub>> to memref<4x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [false, true]} : memref<4x8xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>, vector<8x8xi32>
    %1 = vector.constant_mask [32] : vector<64xi1>
    %2 = vector.shape_cast %1 : vector<64xi1> to vector<8x8xi1>
    annotation.mark %2 {mask_op_idx = -1 : i32} : vector<8x8xi1>
    %subview_1 = memref.subview %subview[0, 0, 0] [4, 1, 8] [1, 1, 1] : memref<4x1x8xi32, strided<[1024, 8, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x8xi32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + d1 + s0)>, #hivm.address_space<ub>>
    //CHECK: %c1024 = arith.constant 1024 : index
    //CHECK: ave.hir.store_with_stride %subview_1[%c0, %c0], %c1024, %[[V1:.*]], %[[V2:.*]] :
    vector.transfer_write %0, %subview_1[%c0, %c0], %2 {in_bounds = [true, true]} : vector<8x8xi32>, memref<4x8xi32, affine_map<(d0, d1)[s0] -> (d0 * 1024 + d1 + s0)>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @test_vsstb_i16
func.func @test_vsstb_i16(%arg0: memref<128x4x16xi16, #hivm.address_space<ub>>, %arg1: memref<4x128x16xi16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0 : i16
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c128 step %c1 {
    %subview = memref.subview %arg1[0, %arg2, 0] [4, 1, 16] [1, 1, 1] : memref<4x128x16xi16, #hivm.address_space<ub>> to memref<4x1x16xi16, strided<[2048, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg2, 0, 0] [1, 4, 16] [1, 1, 1] : memref<128x4x16xi16, #hivm.address_space<ub>> to memref<4x16xi16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [false, true]} : memref<4x16xi16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<8x16xi16>
    %1 = vector.constant_mask [64] : vector<128xi1>
    %2 = vector.shape_cast %1 : vector<128xi1> to vector<8x16xi1>
    annotation.mark %2 {mask_op_idx = -1 : i32} : vector<8x16xi1>
    %subview_1 = memref.subview %subview[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xi16, strided<[2048, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xi16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + d1 + s0)>, #hivm.address_space<ub>>
    //CHECK: %c2048 = arith.constant 2048 : index
    //CHECK: ave.hir.store_with_stride %subview_1[%c0, %c0], %c2048, %[[V1:.*]], %[[V2:.*]] :
    vector.transfer_write %0, %subview_1[%c0, %c0], %2 {in_bounds = [true, true]} : vector<8x16xi16>, memref<4x16xi16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + d1 + s0)>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @test_vsstb_i8
func.func @test_vsstb_i8(%arg0: memref<128x4x32xi8, #hivm.address_space<ub>>, %arg1: memref<4x128x32xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0 : i8
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c128 step %c1 {
    %subview = memref.subview %arg1[0, %arg2, 0] [4, 1, 32] [1, 1, 1] : memref<4x128x32xi8, #hivm.address_space<ub>> to memref<4x1x32xi8, strided<[4096, 32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg0[%arg2, 0, 0] [1, 4, 32] [1, 1, 1] : memref<128x4x32xi8, #hivm.address_space<ub>> to memref<4x32xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [false, true]} : memref<4x32xi8, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>, vector<8x32xi8>
    %1 = vector.constant_mask [128] : vector<256xi1>
    %2 = vector.shape_cast %1 : vector<256xi1> to vector<8x32xi1>
    annotation.mark %2 {mask_op_idx = -1 : i32} : vector<8x32xi1>
    %subview_1 = memref.subview %subview[0, 0, 0] [4, 1, 32] [1, 1, 1] : memref<4x1x32xi8, strided<[4096, 32, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 4096 + d1 + s0)>, #hivm.address_space<ub>>
    //CHECK: %c4096 = arith.constant 4096 : index
    //CHECK: ave.hir.store_with_stride %subview_1[%c0, %c0], %c4096, %[[V1:.*]], %[[V2:.*]] :
    vector.transfer_write %0, %subview_1[%c0, %c0], %2 {in_bounds = [true, true]} : vector<8x32xi8>, memref<4x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 4096 + d1 + s0)>, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL: func.func @tensor_transfer_is_left_legal
// CHECK: vector.transfer_read {{.*}} tensor<1x64xf32>, vector<1x64xf32>
// CHECK: arith.addf
// CHECK: vector.transfer_write {{.*}} tensor<1x64xf32>
// CHECK-NOT: ave.hir.vload
func.func @tensor_transfer_is_left_legal(%arg0: tensor<1x64xf32>)
    -> tensor<1x64xf32> attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %zero = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]}
      : tensor<1x64xf32>, vector<1x64xf32>
  %1 = arith.addf %0, %zero : vector<1x64xf32>
  %2 = vector.transfer_write %1, %arg0[%c0, %c0] {in_bounds = [true, true]}
      : vector<1x64xf32>, tensor<1x64xf32>
  return %2 : tensor<1x64xf32>
}

// CHECK-LABEL: func.func @multi_dim_create_mask_uses_last_dim_bound
// CHECK: %[[MASK:.*]], %{{.*}} = ave.hir.plt %{{.*}} : vector<64xi1>, index
// CHECK: builtin.unrealized_conversion_cast %[[MASK]] : vector<64xi1> to vector<1x64xi1>
// CHECK-NOT: vector.create_mask
func.func @multi_dim_create_mask_uses_last_dim_bound(
    %arg0: memref<1x64xf32, #hivm.address_space<ub>>,
    %arg1: memref<1x64xf32, #hivm.address_space<ub>>, %bound: index)
    attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mask = vector.create_mask %c1, %bound : vector<1x64xi1>
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst, %mask
      {in_bounds = [true, true]} : memref<1x64xf32, #hivm.address_space<ub>>,
      vector<1x64xf32>
  vector.transfer_write %0, %arg1[%c0, %c0], %mask
      {in_bounds = [true, true]} : vector<1x64xf32>,
      memref<1x64xf32, #hivm.address_space<ub>>
  return
}
