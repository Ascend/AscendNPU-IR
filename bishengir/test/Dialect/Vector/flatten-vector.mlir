// RUN: bishengir-opt %s -normalize-vector -cse -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @flatten_with_continuous_transpose_0(
// CHECK: vector.gather {{.*}} vector<64xf16>
// CHECK: vector.transfer_write {{.*}} vector<64xf16>
func.func @flatten_with_continuous_transpose_0(%arg0: memref<16x4x4x16xf16, #hivm.address_space<ub>>, %arg1: memref<4x16x4x16xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c4 step %c1 {
    scf.for %arg3 = %c0 to %c16 step %c1 {
      %subview = memref.subview %arg0[%arg3, %arg2, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<16x4x4x16xf16, #hivm.address_space<ub>> to memref<1x1x4x16xf16, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<4x16x4x16xf16, #hivm.address_space<ub>> to memref<1x1x4x16xf16, strided<[1024, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>} : memref<1x1x4x16xf16, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x4x16xf16>
      vector.transfer_write %0, %subview_0[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x16xf16>, memref<1x1x4x16xf16, strided<[1024, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// -----

// make sure the pattern is not conflict with drop unit dims pattern
// CHECK-LABEL: func.func @flatten_with_continuous_transpose_1(
// CHECK: vector.gather
func.func @flatten_with_continuous_transpose_1(%arg0: memref<16x16x4x4xf16, strided<[1024, 64, 16, 1]>, #hivm.address_space<ub>>, %arg1: memref<16x4x4x16xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [1, 16, 1, 1] : vector<1x128x1x1xi1>
  %1 = vector.constant_mask [1, 1, 1, 16] : vector<1x1x1x128xi1>
  scf.for %arg2 = %c0 to %c16 step %c1 {
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c4 step %c1 {
        %subview = memref.subview %arg0[%arg2, 0, %arg4, %arg3] [1, 16, 1, 1] [1, 1, 1, 1] : memref<16x16x4x4xf16, strided<[1024, 64, 16, 1]>, #hivm.address_space<ub>> to memref<1x16x1x1xf16, strided<[1024, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_0 = memref.subview %arg1[%arg2, %arg3, %arg4, 0] [1, 1, 1, 16] [1, 1, 1, 1] : memref<16x4x4x16xf16, #hivm.address_space<ub>> to memref<1x1x1x16xf16, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %2 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %cst, %0 {in_bounds = [true, true, true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : memref<1x16x1x1xf16, strided<[1024, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x1x128xf16>
        vector.transfer_write %2, %subview_0[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : vector<1x1x1x128xf16>, memref<1x1x1x16xf16, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @flatten_with_non_continuous_transpose_0(
// CHECK: vector.transfer_read {{.*}} vector<4x16xf16>
// CHECK: vector.transfer_write {{.*}} vector<4x16xf16>
func.func @flatten_with_non_continuous_transpose_0(%arg0: memref<4x16x4x16xf16, #hivm.address_space<ub>>, %arg1: memref<4x16x4x16xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c16 step %c1 {
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %subview = memref.subview %arg0[%arg3, %arg2, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<4x16x4x16xf16, #hivm.address_space<ub>> to memref<1x1x4x16xf16, strided<[1024, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[0, %arg2, %arg3, 0] [4, 1, 1, 16] [1, 1, 1, 1] : memref<4x16x4x16xf16, #hivm.address_space<ub>> to memref<4x1x1x16xf16, strided<[1024, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} : memref<1x1x4x16xf16, strided<[1024, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<4x1x1x16xf16>
      vector.transfer_write %0, %subview_0[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<4x1x1x16xf16>, memref<4x1x1x16xf16, strided<[1024, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// -----

// make sure this pattern does not affect drop unit dims patterns.
// CHECK-LABEL: func.func @transpose_2d_outlined_vf_0_ci(
// CHECK: vector.gather
func.func @transpose_2d_outlined_vf_0_ci(%arg0: memref<16x16xf16, #hivm.address_space<ub>>, %arg1: memref<16x16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16, 1] : vector<128x1xi1>
  %1 = vector.constant_mask [1, 16] : vector<1x128xi1>
  scf.for %arg2 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg0[0, %arg2] [16, 1] [1, 1] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2, 0] [1, 16] [1, 1] : memref<16x16xf16, #hivm.address_space<ub>> to memref<1x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %2 = vector.transfer_read %subview[%c0, %c0], %cst, %0 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<16x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x128xf16>
    vector.transfer_write %2, %subview_0[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x128xf16>, memref<1x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @flatten_two_dims_0
// transfer_read will not be flattened
// CHECK: vector.transfer_read {{.*}} vector<4x16xf32>
// CHECK: vector.transfer_read {{.*}} vector<4x16xf32>
// CHECK: arith.addf {{.*}} : vector<64xf32>
// CHECK: arith.mulf {{.*}} : vector<64xf32>
// CHECK: vector.transfer_write {{.*}} : vector<64xf32>
func.func @flatten_two_dims_0(%arg0: memref<384x16xf32, #hivm.address_space<ub>>, %arg1: memref<384x16xf32, #hivm.address_space<ub>>, %arg2: memref<384x16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<0.353553385> : vector<4x16xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c384 = arith.constant 384 : index
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg1[0, 0] [4, 16] [1, 1] : memref<384x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>
  scf.for %arg3 = %c0 to %c384 step %c4 {
    %subview_1 = memref.subview %arg0[%arg3, 0] [4, 16] [1, 1] : memref<384x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg2[%arg3, 0] [4, 16] [1, 1] : memref<384x16xf32, #hivm.address_space<ub>> to memref<4x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview_1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<4x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<4x16xf32>
    %1 = vector.transfer_read %subview[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<4x16xf32, strided<[16, 1]>, #hivm.address_space<ub>>, vector<4x16xf32>
    %2 = arith.addf %0, %1 : vector<4x16xf32>
    %3 = arith.mulf %2, %cst : vector<4x16xf32>
    vector.transfer_write %3, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x16xf32>, memref<4x16xf32, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @flatten_two_dims_1
// CHECK: arith.bitcast {{.*}} : vector<64xf32> to vector<64xi32>
// check: arith.andi {{.*}} : vector<64xi32>
func.func @flatten_two_dims_1(%arg0: memref<4x16xf32, #hivm.address_space<ub>>, %arg1: memref<4x16xi32, #hivm.address_space<ub>>, %arg2: memref<4x16xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x16xf32, #hivm.address_space<ub>>, vector<4x16xf32>
  %1 = arith.bitcast %0 : vector<4x16xf32> to vector<4x16xi32>
  %2 = vector.transfer_read %arg1[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4x16xi32, #hivm.address_space<ub>>, vector<4x16xi32>
  %3 = arith.andi %1, %2 : vector<4x16xi32>
  vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x16xi32>, memref<4x16xi32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @flatten_two_dims_2
// CHECK: memref.collapse_shape {{.*}} {{\[}}{{\[}}0, 1]]
// CHECK: vector.transfer_write {{.*}} vector<64xi32>, memref<64xi32, #hivm.address_space<ub>>
func.func @flatten_two_dims_2(%arg0: memref<4x16xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<2147483647> : vector<4x16xi32>
  %c0 = arith.constant 0 : index
  vector.transfer_write %cst, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<4x16xi32>, memref<4x16xi32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @flatten_four_dims_0(
// CHECK: vector.transfer_read {{.*}} vector<4x16xf32>
// CHECK: arith.truncf {{.*}} : vector<64xf32> to vector<64xf16>
// CHECK: vector.transfer_write {{.*}} vector<64xf16>
func.func @flatten_four_dims_0(%arg0: memref<16x4x4x16xf32, #hivm.address_space<ub>>, %arg1: memref<16x4x4x16xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c16 step %c1 {
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %subview = memref.subview %arg0[%arg2, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<16x4x4x16xf32, #hivm.address_space<ub>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<16x4x4x16xf16, #hivm.address_space<ub>> to memref<1x1x4x16xf16, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x4x16xf32>
      %1 = arith.truncf %0 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x1x4x16xf32> to vector<1x1x4x16xf16>
      vector.transfer_write %1, %subview_0[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x16xf16>, memref<1x1x4x16xf16, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @flatten_arith_and_math_0(
// CHECK: arith.mulf {{.*}} vector<64xf16>
// CHECK: arith.extf {{.*}} vector<64xf16> to vector<64xf32>
// CHECK: arith.subf {{.*}} vector<64xf32>
// CHECK: math.exp {{.*}} vector<64xf32>
func.func @flatten_arith_and_math_0(%arg0: memref<16x4x4x16xf16, #hivm.address_space<ub>>, %arg1: memref<4x16xf32, #hivm.address_space<ub>>, %arg2: memref<16x4x4x16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<5.000000e-01> : vector<1x1x4x16xf16>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c16 step %c1 {
    scf.for %arg4 = %c0 to %c4 step %c1 {
      %subview = memref.subview %arg0[%arg3, %arg4, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<16x4x4x16xf16, #hivm.address_space<ub>> to memref<1x1x4x16xf16, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 4, 16] [1, 1, 1, 1] : memref<16x4x4x16xf32, #hivm.address_space<ub>> to memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %cst_1 {in_bounds = [true, true, true, true]} : memref<1x1x4x16xf16, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x4x16xf16>
      %1 = vector.transfer_read %arg1[%c0, %c0], %cst_0 {in_bounds = [true, true, true, true], permutation_map = affine_map<(d0, d1) -> (0, 0, d0, d1)>} : memref<4x16xf32, #hivm.address_space<ub>>, vector<1x1x4x16xf32>
      %2 = arith.mulf %0, %cst : vector<1x1x4x16xf16>
      %3 = arith.extf %2 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x1x4x16xf16> to vector<1x1x4x16xf32>
      %4 = arith.subf %3, %1 : vector<1x1x4x16xf32>
      %5 = math.exp %4 : vector<1x1x4x16xf32>
      vector.transfer_write %5, %subview_2[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x16xf32>, memref<1x1x4x16xf32, strided<[256, 64, 16, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @flatten_cast_with_vsstb_0(
// CHECK: %[[truncf:.*]] = arith.truncf {{.*}} vector<64xf32>
// CHECK: %[[cast:.*]] = vector.shape_cast %[[truncf]] : vector<64xf16> to vector<4x16xf16>
// CHECK: vector.transfer_write %[[cast]], {{.*}} : vector<4x16xf16>
func.func @flatten_cast_with_vsstb_0(%arg0: memref<64x8x16xf32, #hivm.address_space<ub>>, %arg1: memref<64x8x16xf16, #hivm.address_space<ub>>, %arg2: memref<8x64x16xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c8 step %c4 {
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg4, %arg3, 0] [1, 4, 16] [1, 1, 1] : memref<64x8x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x16xf32, strided<[128, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x4x16xf32>
      %1 = arith.truncf %0 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<1x4x16xf32> to vector<1x4x16xf16>
      %subview_0 = memref.subview %arg2[%arg3, %arg4, 0] [4, 1, 16] [1, 1, 1] : memref<8x64x16xf16, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1024, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %2 = vector.transpose %1, [1, 0, 2] : vector<1x4x16xf16> to vector<4x1x16xf16>
      vector.transfer_write %2, %subview_0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, memref<4x1x16xf16, strided<[1024, 16, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// -----


// CHECK-LABEL: func.func @test_preserve_attrbutes_0
// CHECK: arith.truncf {{.*}} {enable_saturate = false, round_mode = #hfusion.round_mode<rint>}
func.func @test_preserve_attrbutes_0(%arg0: memref<2x32xf32, #hivm.address_space<ub>>, %arg1: memref<2x32xbf16, #hivm.address_space<ub>>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x32xf32, #hivm.address_space<ub>>, vector<2x32xf32>
  %1 = arith.truncf %0 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<2x32xf32> to vector<2x32xbf16>
  vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<2x32xbf16>, memref<2x32xbf16, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @test_flatten_ternary_op_0
// CHECK: arith.cmpf {{.*}} : vector<64xf32>
// CHECK: arith.select {{.*}} : vector<64xi1>, vector<64xf32>
func.func @test_flatten_ternary_op_0(%arg0: memref<1x16x16xf32, #hivm.address_space<ub>>, %arg1: memref<1x16x16xf32, #hivm.address_space<ub>>, %arg2: memref<1x16x16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant dense<1.000000e+00> : vector<1x4x16xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x4x16xf32>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c16 step %c4 {
    %subview = memref.subview %arg0[0, %arg3, 0] [1, 4, 16] [1, 1, 1] : memref<1x16x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_2 = memref.subview %arg1[0, %arg3, 0] [1, 4, 16] [1, 1, 1] : memref<1x16x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_3 = memref.subview %arg2[0, %arg3, 0] [1, 4, 16] [1, 1, 1] : memref<1x16x16xf32, #hivm.address_space<ub>> to memref<1x4x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0, %c0, %c0], %cst_1 {in_bounds = [true, true, true]} : memref<1x4x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x4x16xf32>
    %1 = vector.transfer_read %subview_2[%c0, %c0, %c0], %cst_1 {in_bounds = [true, true, true]} : memref<1x4x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x4x16xf32>
    %2 = arith.cmpf ogt, %0, %1 : vector<1x4x16xf32>
    %3 = arith.select %2, %cst, %cst_0 : vector<1x4x16xi1>, vector<1x4x16xf32>
    vector.transfer_write %3, %subview_3[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x16xf32>, memref<1x4x16xf32, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}
