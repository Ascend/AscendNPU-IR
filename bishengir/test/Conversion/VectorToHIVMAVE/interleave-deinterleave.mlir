// RUN: bishengir-opt %s --convert-vector-to-hivmave | FileCheck %s

// -----
// CHECK-LABEL: triton_interleave_outlined_vf_0
// CHECK: %[[RES1:.*]], %[[RES2:.*]] = ave.hir.vintlv %[[SRC1:.*]], %[[SRC2:.*]] : vector<64xf32>, vector<64xf32>
#map = affine_map<()[s0] -> (s0 floordiv 2)>
module attributes {hacc.target = #hacc.target<"Ascend910_9579">} {
  func.func @triton_interleave_outlined_vf_0(%arg0: memref<1024xf32, #hivm.address_space<ub>>, %arg1: memref<1024xf32, #hivm.address_space<ub>>, %arg2: memref<2048xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [32] : vector<64xi1>
    scf.for %arg3 = %c0 to %c2048 step %c64 {
      %1 = affine.apply #map()[%arg3]
      %subview = memref.subview %arg0[%1] [32] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%1] [32] [1] : memref<1024xf32, #hivm.address_space<ub>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %2 = vector.transfer_read %subview[%c0], %cst, %0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
      %3 = vector.transfer_read %subview_0[%c0], %cst, %0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
      %4 = vector.interleave %2, %3 : vector<64xf32> -> vector<128xf32>
      %5 = builtin.unrealized_conversion_cast %4 : vector<128xf32> to vector<64xf32>
      vector.transfer_write %5, %arg2[%arg3] : vector<64xf32>, memref<2048xf32, #hivm.address_space<ub>>
    }
    return
  }
}

// -----
// CHECK-LABEL: triton_split_26_16_2_outlined_vf_0
// CHECK: %[[RES1:.*]], %[[RES2:.*]] = ave.hir.vdintlv %[[VAL9_0:.*]], %[[VAL9_1:.*]] : vector<256xi8>, vector<256xi8>
#map1 = affine_map<(d0) -> (-d0 + 416, 256)>
#map2 = affine_map<()[s0] -> (s0 * 2)>
#map3 = affine_map<()[s0] -> (s0 * 2 - 256)>
module attributes {hacc.target = #hacc.target<"Ascend910_9579">} {
  func.func @triton_split_26_16_2_outlined_vf_0(%arg0: memref<832xi8, #hivm.address_space<ub>>, %arg1: memref<416xi8, #hivm.address_space<ub>>, %arg2: memref<416xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0_i8 = arith.constant 0 : i8
    %c256 = arith.constant 256 : index
    %c416 = arith.constant 416 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c416 step %c256 {
      %0 = affine.min #map1(%arg3)
      %1 = affine.apply #map2()[%0]
      %2 = affine.apply #map2()[%arg3]
      %subview = memref.subview %arg0[%2] [%1] [1] : memref<832xi8, #hivm.address_space<ub>> to memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %3 = arith.minsi %1, %c256 : index
      %4 = vector.create_mask %3 : vector<256xi1>
      %5 = vector.transfer_read %subview[%c0], %c0_i8, %4 : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
      %6 = affine.apply #map3()[%0]
      %7 = arith.maxsi %6, %c0 : index
      %8 = vector.create_mask %7 : vector<256xi1>
      %9 = arith.cmpi sgt, %7, %c0 : index
      %10 = arith.select %9, %c256, %c0 : index
      %11 = vector.transfer_read %subview[%10], %c0_i8, %8 : memref<?xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi8>
      %12 = builtin.unrealized_conversion_cast %5, %11 : vector<256xi8>, vector<256xi8> to vector<512xi8>
      %res1, %res2 = vector.deinterleave %12 : vector<512xi8> -> vector<256xi8>
      %13 = vector.create_mask %0 : vector<256xi1>
      annotation.mark %13 {mask_op_idx = 2 : i32} : vector<256xi1>
      vector.transfer_write %res1, %arg1[%arg3], %13 : vector<256xi8>, memref<416xi8, #hivm.address_space<ub>>
      vector.transfer_write %res2, %arg2[%arg3], %13 : vector<256xi8>, memref<416xi8, #hivm.address_space<ub>>
    }
    return
  }
}
