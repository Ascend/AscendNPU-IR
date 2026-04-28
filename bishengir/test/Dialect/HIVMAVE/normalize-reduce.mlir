// RUN: bishengir-opt -convert-vector-to-hivmave %s | FileCheck %s


// CHECK-LABEL: func.func @reduction_maxui
func.func @reduction_maxui(%arg0: memref<16xi16, #hivm.address_space<ub>>, %arg1: memref<16x16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0> : vector<128xi16>
  %c0_i8 = arith.constant 0 : i8
  %c0_i16 = arith.constant 0 : i16
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<16xi16, #hivm.address_space<ub>> to memref<1xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3, 0] [1, 16] [1, 1] : memref<16x16xi8, #hivm.address_space<ub>> to memref<1x16xi8, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %3 = vector.constant_mask [16] : vector<128xi1>
    %subview_1 = memref.subview %subview_0[0, 0] [1, 16] [1, 1] : memref<1x16xi8, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %4 = vector.transfer_read %subview_1[%c0], %c0_i8, %3 {in_bounds = [true]} : memref<16xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi8>
    %5 = arith.extui %4 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<128xi8> to vector<128xi16>
    %6 = arith.select %3, %5, %cst : vector<128xi1>, vector<128xi16>
    %subview_2 = memref.subview %subview[0] [1] [1] : memref<1xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
    %7 = vector.transfer_read %subview_2[], %c0_i16 : memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<i16>
    %8 = vector.shape_cast %7 : vector<i16> to vector<1xi16>
    %9 = builtin.unrealized_conversion_cast %8 : vector<1xi16> to i16
    %10 = vector.reduction <maxui>, %6, %9 {withoutInitMergeOp} : vector<128xi16> into i16
    %11 = builtin.unrealized_conversion_cast %10 : i16 to vector<1xi16>
    %12 = vector.shape_cast %11 : vector<1xi16> to vector<i16>
    vector.transfer_write %12, %subview_2[] : vector<i16>, memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
  }
  %0 = vector.constant_mask [16] : vector<128xi1>
  annotation.mark %0 {mask_op_idx = 1 : i32} : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i16, %0 {in_bounds = [true]} : memref<16xi16, #hivm.address_space<ub>>, vector<128xi16>
  annotation.mark %1 {reached_mask_ops_idx = 1 : i32} : vector<128xi16>
  %2 = arith.trunci %1 {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>} : vector<128xi16> to vector<128xi8>
  annotation.mark %2 {reached_mask_ops_idx = 1 : i32} : vector<128xi8>
  vector.transfer_write %2, %arg2[%c0], %0 {in_bounds = [true]} : vector<128xi8>, memref<16xi8, #hivm.address_space<ub>>
  return
}

// CHECK-NOT: vector.reduction
// CHECK: ave.hir.reduction <umax>


// CHECK-LABEL: func.func @reduction_maxi
func.func @reduction_maxi(%arg0: memref<16xi16, #hivm.address_space<ub>>, %arg1: memref<16x16xi8, #hivm.address_space<ub>>, %arg2: memref<16xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<-32768> : vector<128xi16>
  %c0_i8 = arith.constant 0 : i8
  %c0_i16 = arith.constant 0 : i16
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg0[%arg3] [1] [1] : memref<16xi16, #hivm.address_space<ub>> to memref<1xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3, 0] [1, 16] [1, 1] : memref<16x16xi8, #hivm.address_space<ub>> to memref<1x16xi8, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %3 = vector.constant_mask [16] : vector<128xi1>
    %subview_1 = memref.subview %subview_0[0, 0] [1, 16] [1, 1] : memref<1x16xi8, strided<[16, 1], offset: ?>, #hivm.address_space<ub>> to memref<16xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %4 = vector.transfer_read %subview_1[%c0], %c0_i8, %3 {in_bounds = [true]} : memref<16xi8, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<128xi8>
    %5 = arith.extsi %4 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<128xi8> to vector<128xi16>
    %6 = arith.select %3, %5, %cst : vector<128xi1>, vector<128xi16>
    %subview_2 = memref.subview %subview[0] [1] [1] : memref<1xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
    %7 = vector.transfer_read %subview_2[], %c0_i16 : memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<i16>
    %8 = vector.shape_cast %7 : vector<i16> to vector<1xi16>
    %9 = builtin.unrealized_conversion_cast %8 : vector<1xi16> to i16
    %10 = vector.reduction <maxsi>, %6, %9 {withoutInitMergeOp} : vector<128xi16> into i16
    %11 = builtin.unrealized_conversion_cast %10 : i16 to vector<1xi16>
    %12 = vector.shape_cast %11 : vector<1xi16> to vector<i16>
    vector.transfer_write %12, %subview_2[] : vector<i16>, memref<i16, strided<[], offset: ?>, #hivm.address_space<ub>>
  }
  %0 = vector.constant_mask [16] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i16, %0 {in_bounds = [true]} : memref<16xi16, #hivm.address_space<ub>>, vector<128xi16>
  %2 = arith.trunci %1 {enable_saturate = false, round_mode = #hfusion.round_mode<trunc>} : vector<128xi16> to vector<128xi8>
  vector.transfer_write %2, %arg2[%c0], %0 {in_bounds = [true]} : vector<128xi8>, memref<16xi8, #hivm.address_space<ub>>
  return
}

// CHECK-NOT: vector.reduction
// CHECK: ave.hir.reduction <max>
