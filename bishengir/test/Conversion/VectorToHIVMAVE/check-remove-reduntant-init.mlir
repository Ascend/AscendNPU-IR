// RUN: bishengir-opt %s -convert-vector-to-hivmave -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// CHECK-LABEL: func.func @test_remove_reduntant_init_off
func.func @test_remove_reduntant_init_off(%arg0: memref<3xi16, #hivm.address_space<ub>>, %arg1: memref<3x1835xi8, #hivm.address_space<ub>>, %arg2: memref<3xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c0_i16 = arith.constant 0 : i16
  %c0_i8 = arith.constant 0 : i8
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c1835 = arith.constant 1835 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  scf.for %arg3 = %c0 to %c3 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<3xi16, #hivm.address_space<ub>> to memref<1xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %subview[0] [1] [1] : memref<1xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
    %3 = vector.transfer_read %subview_0[], %c0_i16 : memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<i16>
    %4 = vector.shape_cast %3 : vector<i16> to vector<1xi16>
    %5 = scf.for %arg4 = %c0 to %c1835 step %c128 iter_args(%arg5 = %4) -> (vector<1xi16>) {
      %7 = affine.min affine_map<(d0) -> (-d0 + 1835, 128)>(%arg4)
      %subview_1 = memref.subview %arg1[%arg3, %arg4] [1, %7] [1, 1] : memref<3x1835xi8, #hivm.address_space<ub>> to memref<1x?xi8, strided<[1835, 1], offset: ?>, #hivm.address_space<ub>>
      %8 = vector.create_mask %7 : vector<128xi1>
      %subview_2 = memref.subview %subview_1[0, 0] [1, %7] [1, 1] : memref<1x?xi8, strided<[1835, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %9 = vector.transfer_read %subview_2[%c0], %c0_i8, %8 {in_bounds = [true]} : memref<?xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi8>
      %10 = arith.extsi %9 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<128xi8> to vector<128xi16>
      %11 = builtin.unrealized_conversion_cast %arg5 : vector<1xi16> to i16
      // CHECK锛歿{.*}} = ave.hir.reduction <max>
      // CHECK: {{.*}} = arith.maxsi
      %12 = vector.mask %8 { vector.reduction <maxsi>, %10, %11 : vector<128xi16> into i16 } : vector<128xi1> -> i16
      %13 = builtin.unrealized_conversion_cast %12 : i16 to vector<1xi16>
      scf.yield %13 : vector<1xi16>
    }
    %6 = vector.shape_cast %5 : vector<1xi16> to vector<i16>
    vector.transfer_write %6, %subview_0[] : vector<i16>, memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
  }
  %0 = vector.constant_mask [3] : vector<128xi1>
  annotation.mark %0 {mask_op_idx = 1 : i32} : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i16, %0 {in_bounds = [true]} : memref<3xi16, #hivm.address_space<ub>>, vector<128xi16>
  annotation.mark %1 {reached_mask_ops_idx = 1 : i32} : vector<128xi16>
  %2 = arith.trunci %1 {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>} : vector<128xi16> to vector<128xi8>
  annotation.mark %2 {reached_mask_ops_idx = 1 : i32} : vector<128xi8>
  vector.transfer_write %2, %arg2[%c0], %0 {in_bounds = [true]} : vector<128xi8>, memref<3xi8, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @test_remove_reduntant_init_on
// CHECK: {{.*}} = arith.maxsi
// CHECK锛歿{.*}} = ave.hir.reduction <max>
func.func @test_remove_reduntant_init_on(%arg0: memref<3xi16, #hivm.address_space<ub>>, %arg1: memref<1x128xi16, #hivm.address_space<ub>>, %arg2: memref<3x512xi8, #hivm.address_space<ub>>, %arg3: memref<3xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<-32768> : vector<128xi16>
  %c0_i8 = arith.constant 0 : i8
  %cst_0 = arith.constant dense<-32768> : vector<1x128xi16>
  %c0_i16 = arith.constant 0 : i16
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  scf.for %arg4 = %c0 to %c3 step %c1 {
    %subview = memref.subview %arg0[%arg4] [1] [1] : memref<3xi16, #hivm.address_space<ub>> to memref<1xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %3:2 = scf.for %arg5 = %c0 to %c512 step %c128 iter_args(%arg6 = %cst_0, %arg7 = %cst) -> (vector<1x128xi16>, vector<128xi16>) {
      %subview_2 = memref.subview %arg2[%arg4, %arg5] [1, 128] [1, 1] : memref<3x512xi8, #hivm.address_space<ub>> to memref<1x128xi8, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview_2[0, 0] [1, 128] [1, 1] : memref<1x128xi8, strided<[512, 1], offset: ?>, #hivm.address_space<ub>> to memref<128xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
      %10 = vector.transfer_read %subview_3[%c0], %c0_i8 {in_bounds = [true]} : memref<128xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi8>
      %11 = arith.extsi %10 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<128xi8> to vector<128xi16>
      %12 = arith.maxsi %11, %arg7 {reductionOp} : vector<128xi16>
      %13 = vector.shape_cast %12 : vector<128xi16> to vector<1x128xi16>
      scf.yield %13, %12 : vector<1x128xi16>, vector<128xi16>
    } {reductionLoop}
    %subview_1 = memref.subview %subview[0] [1] [1] : memref<1xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
    %4 = vector.transfer_read %subview_1[], %c0_i16 : memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<i16>
    %5 = vector.shape_cast %4 : vector<i16> to vector<1xi16>
    %6 = builtin.unrealized_conversion_cast %5 : vector<1xi16> to i16
    %7 = vector.reduction <maxsi>, %3#1, %6 : vector<128xi16> into i16
    %8 = builtin.unrealized_conversion_cast %7 : i16 to vector<1xi16>
    %9 = vector.shape_cast %8 : vector<1xi16> to vector<i16>
    vector.transfer_write %9, %subview_1[] : vector<i16>, memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
  }
  %0 = vector.constant_mask [3] : vector<128xi1>
  annotation.mark %0 {mask_op_idx = 0 : i32} : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i16, %0 {in_bounds = [true]} : memref<3xi16, #hivm.address_space<ub>>, vector<128xi16>
  annotation.mark %1 {reached_mask_ops_idx = 0 : i32} : vector<128xi16>
  %2 = arith.trunci %1 {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>} : vector<128xi16> to vector<128xi8>
  annotation.mark %2 {reached_mask_ops_idx = 0 : i32} : vector<128xi8>
  vector.transfer_write %2, %arg3[%c0], %0 {in_bounds = [true]} : vector<128xi8>, memref<3xi8, #hivm.address_space<ub>>
  return
}