// RUN: bishengir-opt --arith-vector-mask-analyze -split-input-file %s | FileCheck %s

// in this case, %1 should not be markd to reach mask
// CHECK-LABEL: triton_where_2d_outlined_merged_vf_0
// CHECK: %[[VAL1:.*]] = arith.addi
// CHECK-NOT: annotation.mark %[[VAL1]]
func.func @triton_where_2d_outlined_merged_vf_0(%arg0: memref<13xi32, #hivm.address_space<ub>>, %arg1: memref<13xi32, #hivm.address_space<ub>>, %arg2: memref<19xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : vector<64xi32>
  %cst_0 = arith.constant dense<0> : vector<64xi32>
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [13] : vector<64xi1>
  vector.transfer_write %cst_0, %arg0[%c0], %0 {in_bounds = [true]} : vector<64xi32>, memref<13xi32, #hivm.address_space<ub>>
  %1 = arith.addi %cst_0, %cst : vector<64xi32>
  vector.transfer_write %1, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xi32>, memref<13xi32, #hivm.address_space<ub>>
  %2 = vector.constant_mask [19] : vector<64xi1>
  vector.transfer_write %1, %arg2[%c0], %2 {in_bounds = [true]} : vector<64xi32>, memref<19xi32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: triton_test_reduction_mask
// CHECK: %[[VAL:[0-9]+]] = vector.reduction
// CHECK-NOT: annotation.mark %[[VAL]]
func.func @triton_test_reduction_mask(%arg0: memref<52xf32, #hivm.address_space<ub>>, %arg1: memref<52x10xf32, #hivm.address_space<ub>>, %arg2: memref<52xf32, #hivm.address_space<ub>>, %arg3: memref<52xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  %cst_0 = arith.constant dense<1.000000e+01> : vector<64xf32>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c52 = arith.constant 52 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [1] : vector<64xi1>
  scf.for %arg4 = %c0 to %c52 step %c1 {
    %subview = memref.subview %arg0[%arg4] [1] [1] : memref<52xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg1[%arg4, 0] [1, 10] [1, 1] : memref<52x10xf32, #hivm.address_space<ub>> to memref<1x10xf32, strided<[10, 1], offset: ?>, #hivm.address_space<ub>>
    %1 = vector.constant_mask [10] : vector<64xi1>
    %subview_3 = memref.subview %subview_2[0, 0] [1, 10] [1, 1] : memref<1x10xf32, strided<[10, 1], offset: ?>, #hivm.address_space<ub>> to memref<10xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %2 = vector.transfer_read %subview_3[%c0], %cst_1, %1 {in_bounds = [true]} : memref<10xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xf32>
    %3 = arith.select %1, %2, %cst : vector<64xi1>, vector<64xf32>
    %4 = arith.addf %3, %cst {reductionOp} : vector<64xf32>
    %subview_4 = memref.subview %subview[0] [1] [1] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>
    %5 = vector.transfer_read %subview_4[], %cst_1 : memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>, vector<f32>
    %6 = vector.shape_cast %5 : vector<f32> to vector<1xf32>
    %7 = builtin.unrealized_conversion_cast %6 : vector<1xf32> to f32
    %8 = vector.reduction <add>, %4, %7 {withoutInitMergeOp} : vector<64xf32> into f32
    %9 = builtin.unrealized_conversion_cast %8 : f32 to vector<1xf32>
    %subview_5 = memref.subview %arg3[%arg4] [1] [1] : memref<52xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %10 = vector.broadcast %9 : vector<1xf32> to vector<64xf32>
    %11 = arith.divf %10, %cst_0 : vector<64xf32>
    vector.transfer_write %11, %subview_5[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}