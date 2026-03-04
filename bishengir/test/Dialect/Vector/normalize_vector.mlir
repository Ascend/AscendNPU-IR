// RUN: bishengir-opt %s -normalize-vector -vector-transfer-lowering -cse -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_maskstore_one_leading_dim
// CHECK: %[[CONST0:.*]] = arith.constant dense<0.000000e+00> : vector<64xf32>
// CHECK: %[[MASK0:.*]] = vector.create_mask %[[VALUE:.*]] : vector<64xi1>
// CHECK: vector.maskedstore %[[SUBVIEW0:.*]][%[[CONST1:.*]]], %[[MASK0]], %[[CONST0]]

#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
func.func @test_maskstore_one_leading_dim(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<1x?xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg5 = %c0 to %arg0 step %c64 {
    %0 = affine.min #map(%arg5)[%arg0]
    %subview = memref.subview %arg4[0, %arg5] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
    vector.transfer_write %cst, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    }
  return
}

// -----

// CHECK-LABEL: func.func @test_maskload_eltwise_maskstore_one_leading_dim
// CHECK: %[[CONST0:.*]] = arith.constant dense<0.000000e+00> : vector<64xf32>
// CHECK: %[[CONST1:.*]] = arith.constant 0 : index
// CHECK: %[[MASK0:.*]] = vector.create_mask %[[VALUE:.*]] : vector<64xi1>
// CHECK: %[[MASKLOAD0:.*]] = vector.maskedload %[[SUBVIEW0:.*]][%[[CONST1]]], %[[MASK0]], %[[CONST0]]
// CHECK: %[[MUL:.*]] = arith.mulf %[[MASKLOAD0]], %[[MASKLOAD0]] : vector<64xf32>
// CHECK: %[[MASKLOAD1:.*]] = vector.maskedload %[[SUBVIEW1:.*]][%[[CONST1]]], %[[MASK0]], %[[CONST0]]
// CHECK: %[[ADD:.*]] = arith.addf %[[MASKLOAD1]], %[[MUL]] : vector<64xf32>
// CHECK: vector.maskedstore %[[SUBVIEW2:.*]][%[[CONST1]]], %[[MASK0]], %[[ADD]]

#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
func.func @test_maskload_eltwise_maskstore_one_leading_dim(%arg0: index, %arg1: memref<1x?xf32, #hivm.address_space<ub>>, %arg2: memref<1x?xf32, #hivm.address_space<ub>>, %arg3: index, %arg4: index, %arg5: index, %arg6: memref<1x?xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg7 = %c0 to %arg0 step %c64 {
    %0 = affine.min #map1(%arg7)[%arg0]
    %subview = memref.subview %arg6[0, %arg7] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[0, %arg7] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
    %2 = vector.transfer_read %subview_0[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %3 = arith.mulf %2, %2 : vector<1x64xf32>
    %4 = vector.transfer_read %subview[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %5 = arith.addf %4, %3 : vector<1x64xf32>
    vector.transfer_write %5, %subview[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_reduction_one_leading_dim
// CHECK: %[[CONST0:.*]] = arith.constant dense<0.000000e+00> : vector<64xf32>
// CHECK: %[[CONST1:.*]] = arith.constant 0 : index
// CHECK: %[[MASK0:.*]] = vector.create_mask %[[VALUE:.*]] : vector<64xi1>
// CHECK: %[[MASKLOAD0:.*]] = vector.maskedload %[[SUBVIEW0:.*]][%[[CONST1]]], %[[MASK0]], %[[CONST0]]
// CHECK: %[[REDUCTION:.*]] = vector.mask %[[MASK0]] { vector.reduction <add>, %[[MASKLOAD0]], %{{.*}} : vector<64xf32> into f32 } : vector<64xi1> -> f32
// CHECK: %[[UCC:.*]] = builtin.unrealized_conversion_cast %[[REDUCTION]] : f32 to vector<1xf32>
// CHECK: %[[SC:.*]] = vector.shape_cast %[[UCC]] : vector<1xf32> to vector<f32>
// CHECK: %[[EXTRACT:.*]] = vector.extractelement %[[SC]][]
// CHECK: memref.store %[[EXTRACT]], %{{.*}}[]

#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
func.func @test_reduction_one_leading_dim(%arg0: index, %arg1: memref<1x?xf32, #hivm.address_space<ub>>, %arg2: index, %arg3: index, %arg4: index, %arg5: memref<1xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg6 = %c0 to %arg0 step %c64 {
    %0 = affine.min #map2(%arg6)[%arg0]
    %subview = memref.subview %arg1[0, %arg6] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    %1 = vector.create_mask %c1, %0 : vector<1x64xi1>
    %2 = vector.transfer_read %subview[%c0, %c0], %cst, %1 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %3 = vector.transfer_read %arg5[%c0], %cst {in_bounds = [true]} : memref<1xf32, #hivm.address_space<ub>>, vector<1xf32>
    %4 = vector.mask %1 { vector.multi_reduction <add>, %2, %3 [1] : vector<1x64xf32> to vector<1xf32> } : vector<1x64xi1> -> vector<1xf32>
    vector.transfer_write %4, %arg5[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #hivm.address_space<ub>>
  }
  return
}

// -----
// now we only lower 1D transfers
// CHECK-LABEL: func.func @test_broadcast
// CHECK: %[[CONST1:.*]] = arith.constant 0 : index
// CHECK: %[[MASKEDLOAD:.*]] = vector.maskedload %{{.*}}[%[[CONST1]]], %{{.*}}, %{{.*}}
// CHECK: %[[MUL:.*]] = arith.mulf %{{.*}}, %[[MASKEDLOAD]] : vector<64xf32>
// CHECK: %[[RES:.*]] = arith.mulf %{{.*}}, %[[MUL]] : vector<64xf32>
// CHECK: vector.maskedstore %{{.*}}[%[[CONST1]]], %{{.*}}, %[[RES]]

#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map4 = affine_map<(d0) -> (0, d0)>
#map5 = affine_map<(d0) -> (d0, 0)>
func.func @test_broadcast(%arg0: index, %arg1: memref<?xf32, #hivm.address_space<ub>>, %arg2: memref<1x?xf32, #hivm.address_space<ub>>, %arg3: memref<1xf32, #hivm.address_space<ub>>, %arg4: memref<1x?xf32, #hivm.address_space<ub>>, %arg5: index, %arg6: index, %arg7: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg8 = %c0 to %arg0 step %c64 {
    %0 = affine.min #map3(%arg8)[%arg0]
    %1 = vector.create_mask %0 : vector<64xi1>
    %subview = memref.subview %arg2[0, %arg8] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg8] [%0] [1] : memref<?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg4[0, %arg8] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    %2 = vector.create_mask %c1, %0 : vector<1x64xi1>
    %3 = vector.transfer_read %subview_0[%c0], %cst, %1 {in_bounds = [true, true], permutation_map = #map4} : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %4 = vector.transfer_read %arg3[%c0], %cst {in_bounds = [true, true], permutation_map = #map5} : memref<1xf32, #hivm.address_space<ub>>, vector<1x64xf32>
    %5 = vector.transfer_read %subview_1[%c0, %c0], %cst, %2 {in_bounds = [true, true]} : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
    %6 = arith.mulf %4, %5 : vector<1x64xf32>
    %7 = arith.mulf %3, %6 : vector<1x64xf32>
    vector.transfer_write %7, %subview[%c0, %c0], %2 {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_3d_multi_reduction
func.func @test_3d_multi_reduction(%arg0: memref<2x8x1024xf32, #hivm.address_space<ub>>, %arg1: memref<2x8xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  scf.for %arg2 = %c0 to %c2 step %c1 {
    scf.for %arg3 = %c0 to %c8 step %c1 {
      %subview = memref.subview %arg1[%arg2, %arg3] [1, 1] [1, 1] : memref<2x8xf32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1xf32>
      %1 = builtin.unrealized_conversion_cast %0 : vector<1x1xf32> to vector<64xf32>
      %2 = scf.for %arg4 = %c0 to %c1024 step %c64 iter_args(%arg5 = %1) -> (vector<64xf32>) {
        %4 = builtin.unrealized_conversion_cast %arg5 : vector<64xf32> to vector<1x1xf32>
        %subview_0 = memref.subview %arg0[%arg2, %arg3, %arg4] [1, 1, 64] [1, 1, 1] : memref<2x8x1024xf32, #hivm.address_space<ub>> to memref<1x1x64xf32, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[8192, 1024, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x1x64xf32>
        // CHECK: %[[REDUCTION:.*]] = vector.reduction <maxnumf>, %{{.*}}, %{{.*}} : vector<64xf32> into f32
        %6 = vector.multi_reduction <maxnumf>, %5, %4 [2] : vector<1x1x64xf32> to vector<1x1xf32>
        %7 = builtin.unrealized_conversion_cast %6 : vector<1x1xf32> to vector<64xf32>
        scf.yield %7 : vector<64xf32>
      }
      %3 = builtin.unrealized_conversion_cast %2 : vector<64xf32> to vector<1x1xf32>
      vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<1x1xf32>, memref<1x1xf32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_1d_multi_reduction
func.func @test_1d_multi_reduction(%arg0: memref<256xf32, #hivm.address_space<ub>>, %arg1: memref<256xf32, #hivm.address_space<ub>>, %arg2: memref<f32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 0.000000e+00 : f32
  %c64 = arith.constant 64 : index
  %c256 = arith.constant 256 : index
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg2[], %cst : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %1 = builtin.unrealized_conversion_cast %0 : vector<f32> to vector<64xf32>
  %2 = scf.for %arg3 = %c0 to %c256 step %c64 iter_args(%arg4 = %1) -> (vector<64xf32>) {
    %4 = builtin.unrealized_conversion_cast %arg4 : vector<64xf32> to vector<f32>
    %subview = memref.subview %arg0[%arg3] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %5 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
    %6 = vector.transfer_read %subview_0[%c0], %cst {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
    // CHECK-NO: vector.extractelement
    %7 = vector.extractelement %4[] : vector<f32>
    %8 = arith.addf %6, %5 : vector<64xf32>
    // CHECK: %[[REDUCTION:.*]] = vector.reduction <add>, %{{.*}}, %{{.*}} : vector<64xf32> into f32
    %9 = vector.multi_reduction <add>, %8, %7 [0] : vector<64xf32> to f32
    vector.transfer_write %8, %subview_0[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    // CHECK-NO: vector.broadcast
    %10 = vector.broadcast %9 : f32 to vector<f32>
    %11 = builtin.unrealized_conversion_cast %10 : vector<f32> to vector<64xf32>
    scf.yield %11 : vector<64xf32>
  }
  %3 = builtin.unrealized_conversion_cast %2 : vector<64xf32> to vector<f32>
  vector.transfer_write %3, %arg2[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: func.func @test_xori_reduction
func.func @test_xori_reduction(%arg0: memref<?x128xi32, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<?xi32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %arg1 step %c1 {
    %subview = memref.subview %arg2[%arg3] [1] [1] : memref<?xi32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0], %c0_i32 {in_bounds = [true]} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xi32>
    %1 = builtin.unrealized_conversion_cast %0 : vector<1xi32> to vector<64xi32>
    %2 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %1) -> (vector<64xi32>) {
      %4 = builtin.unrealized_conversion_cast %arg5 : vector<64xi32> to vector<1xi32>
      %subview_0 = memref.subview %arg0[%arg3, %arg4] [1, 64] [1, 1] : memref<?x128xi32, #hivm.address_space<ub>> to memref<1x64xi32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %5 = vector.transfer_read %subview_0[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<1x64xi32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xi32>
      // CHECK: %[[REDUCTION:.*]] = vector.reduction <xor>, %{{.*}}, %{{.*}} : vector<64xi32> into i32
      %6 = vector.multi_reduction <xor>, %5, %4 [1] : vector<1x64xi32> to vector<1xi32>
      %7 = builtin.unrealized_conversion_cast %6 : vector<1xi32> to vector<64xi32>
      scf.yield %7 : vector<64xi32>
    }
    %3 = builtin.unrealized_conversion_cast %2 : vector<64xi32> to vector<1xi32>
    vector.transfer_write %3, %subview[%c0] {in_bounds = [true]} : vector<1xi32>, memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----
// Normalize vector.transfer_read with permutation_map into vector.gather 

// This test is to normalize vector.transfer_read with a mask
// CHECK-LABEL: func.func @transpose_2d_outlined_vf_0
func.func @transpose_2d_outlined_vf_0(%arg0: memref<16x16xf16, #hivm.address_space<ub>>, %arg1: memref<16x16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  // CHECK: %[[INDEX:.*]] = arith.constant dense<"0x0000100020003000400050006000700080009000A000B000C000D000E000F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16, 1] : vector<128x1xi1>
  %1 = vector.constant_mask [1, 16] : vector<1x128xi1>
  scf.for %arg2 = %c0 to %c16 step %c1 {
    %subview = memref.subview %arg0[0, %arg2] [16, 1] [1, 1] : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2, 0] [1, 16] [1, 1] : memref<16x16xf16, #hivm.address_space<ub>> to memref<1x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
    // CHECK-NO: vector.transfer_read
    // CHECK: %[[MASK:.*]] = vector.constant_mask [16] : vector<128xi1>
    // CHECK-NEXT: %[[GATHER:.*]] = vector.gather %subview[%c0, %c0] [%[[INDEX]]], %[[MASK]], %cst : memref<16x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1>, vector<128xf16> into vector<128xf16>
    %2 = vector.transfer_read %subview[%c0, %c0], %cst, %0 {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<16x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x128xf16>
    vector.transfer_write %2, %subview_0[%c0, %c0], %1 {in_bounds = [true, true]} : vector<1x128xf16>, memref<1x16xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// This test is to normalize vector.transfer_read without a mask
// CHECK-LABEL: func.func @transpose_2d_outlined_vf_1
func.func @transpose_2d_outlined_vf_1(%arg0: memref<512x16xf16, #hivm.address_space<ub>>, %arg1: memref<16x512xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  // CHECK: %[[INDEX:.*]] = arith.constant dense<"0x0000100020003000400050006000700080009000A000B000C000D000E000F0000001100120013001400150016001700180019001A001B001C001D001E001F0010002100220023002400250026002700280029002A002B002C002D002E002F0020003100320033003400350036003700380039003A003B003C003D003E003F0030004100420043004400450046004700480049004A004B004C004D004E004F0040005100520053005400550056005700580059005A005B005C005D005E005F0050006100620063006400650066006700680069006A006B006C006D006E006F0060007100720073007400750076007700780079007A007B007C007D007E007F007"> : vector<128xi16>
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c512 = arith.constant 512 : index
  scf.for %arg2 = %c0 to %c16 step %c1 {
    scf.for %arg3 = %c0 to %c512 step %c128 {
      %subview = memref.subview %arg0[%arg3, %arg2] [128, 1] [1, 1] : memref<512x16xf16, #hivm.address_space<ub>> to memref<128x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3] [1, 128] [1, 1] : memref<16x512xf16, #hivm.address_space<ub>> to memref<1x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
      // CHECK-NO: vector.transfer_read
      // CHECK: %[[MASK:.*]] = vector.constant_mask [128] : vector<128xi1>
      // CHECK-NEXT: %[[GATHER:.*]] = vector.gather %subview[%c0, %c0] [%[[INDEX]]], %[[MASK]], %cst : memref<128x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<128xi16>, vector<128xi1>, vector<128xf16> into vector<128xf16>
     %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<128x1xf16, strided<[16, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x128xf16>
      vector.transfer_write %0, %subview_0[%c0, %c0] {in_bounds = [true, true]} : vector<1x128xf16>, memref<1x128xf16, strided<[512, 1], offset: ?>, #hivm.address_space<ub>>
    }
  }
  return
}

// CHECK-LABEL: func.func @triton_max_1d_dim0
func.func @triton_max_1d_dim0(%arg0: memref<2xi32, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [2] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i32, %0 {in_bounds = [true]} : memref<2xi32, #hivm.address_space<ub>>, vector<64xi32>
  %2 = vector.transfer_read %arg1[], %c0_i32 : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %3 = vector.extractelement %2[] : vector<i32>
  // CHECK: %[[MASK:.*]] = vector.mask %0 { vector.reduction <maxsi>, %{{.*}}, %{{.*}} : vector<64xi32> into i32 } : vector<64xi1> -> i32
  %4 = vector.mask %0 { vector.multi_reduction <maxsi>, %1, %3 [0] : vector<64xi32> to i32 } : vector<64xi1> -> i32
  %5 = vector.broadcast %4 : i32 to vector<i32>
  vector.transfer_write %5, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// CHECK-LABEL: func.func @triton_xor_sum_2d_dim0_dim1_outlined_vf_0
func.func @triton_xor_sum_2d_dim0_dim1_outlined_vf_0(%arg0: memref<2x3xi32, strided<[8, 1]>, #hivm.address_space<ub>>, %arg1: memref<i32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [1, 3] : vector<1x64xi1>
  %1 = vector.transfer_read %arg1[], %c0_i32 : memref<i32, #hivm.address_space<ub>>, vector<i32>
  %2 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %1) -> (vector<i32>) {
    %subview = memref.subview %arg0[%arg2, 0] [1, 3] [1, 1] : memref<2x3xi32, strided<[8, 1]>, #hivm.address_space<ub>> to memref<1x3xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>
    %3 = vector.transfer_read %subview[%c0, %c0], %c0_i32, %0 {in_bounds = [true, true]} : memref<1x3xi32, strided<[8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xi32>
    %4 = vector.extractelement %arg3[] : vector<i32>
    // CHECK: %[[MASK:.*]] = vector.mask %{{.*}} { vector.reduction <xor>, %{{.*}}, %{{.*}} : vector<64xi32> into i32 } : vector<64xi1> -> i32
    %5 = vector.mask %0 { vector.multi_reduction <xor>, %3, %4 [0, 1] : vector<1x64xi32> to i32 } : vector<1x64xi1> -> i32
    %6 = vector.broadcast %5 : i32 to vector<i32>
    scf.yield %6 : vector<i32>
  }
  vector.transfer_write %2, %arg1[] : vector<i32>, memref<i32, #hivm.address_space<ub>>
  return
}

// -----

// CHECK-LABEL: unit_dim_multi_reduction_to_reduction
// CHECK: %[[CST:.*]] = arith.constant dense<0xFF800000> : vector<64xf32>
// CHECK: %[[RES:.*]] = arith.select %[[_:.*]] %[[CST]] : vector<64xi1>, vector<64xf32>
// CHECK: vector.reduction <maximumf>, %[[RES]]
func.func @unit_dim_multi_reduction_to_reduction(%arg0: memref<2xf32, #hivm.address_space<ub>>, %arg1: memref<f32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<0xFF800000> : vector<64xf32>
  %cst_0 = arith.constant dense<0> : vector<64xi32>
  %cst_1 = arith.constant dense<1> : vector<64xi32>
  %cst_2 = arith.constant dense<-2139095040> : vector<64xi32>
  %cst_3 = arith.constant dense<2147483647> : vector<64xi32>
  %cst_4 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [2] : vector<64xi1>
  %1 = vector.transfer_read %arg0[%c0], %cst_4, %0 {in_bounds = [true]} : memref<2xf32, #hivm.address_space<ub>>, vector<64xf32>
  %2 = vector.transfer_read %arg1[], %cst_4 : memref<f32, #hivm.address_space<ub>>, vector<f32>
  %3 = vector.extractelement %2[] : vector<f32>
  %4 = arith.bitcast %1 : vector<64xf32> to vector<64xi32>
  %5 = arith.andi %4, %cst_3 : vector<64xi32>
  %6 = arith.addi %5, %cst_2 : vector<64xi32>
  %7 = arith.minsi %6, %cst_1 : vector<64xi32>
  %8 = arith.maxsi %7, %cst_0 : vector<64xi32>
  %9 = arith.cmpi ne, %8, %cst_0 : vector<64xi32>
  %10 = arith.select %9, %cst, %1 : vector<64xi1>, vector<64xf32>
  %11 = vector.mask %0 { vector.multi_reduction <maximumf>, %10, %3 [0] : vector<64xf32> to f32 } : vector<64xi1> -> f32
  %12 = vector.broadcast %11 : f32 to vector<f32>
  vector.transfer_write %12, %arg1[] : vector<f32>, memref<f32, #hivm.address_space<ub>>
  return
}

// -----

func.func @triton_min_1d_outlined_vf_0(%arg0: memref<16xi8, #hivm.address_space<ub>>, %arg1: memref<i16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  // CHECK: %[[CST:.*]] = arith.constant dense<32767> : vector<128xi16>
  // CHECK: %[[_:.*]] = arith.select %[[_:.*]], %[[_:.*]], %[[CST]] : vector<128xi1>, vector<128xi16>
  %c0_i16 = arith.constant 0 : i16
  %c0_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [16] : vector<128xi1>
  %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<16xi8, #hivm.address_space<ub>>, vector<128xi8>
  %2 = vector.transfer_read %arg1[], %c0_i16 : memref<i16, #hivm.address_space<ub>>, vector<i16>
  %3 = vector.extractelement %2[] : vector<i16>
  %4 = arith.extsi %1 {enable_saturate = false, round_mode = #hfusion.round_mode<rint>} : vector<128xi8> to vector<128xi16>
  %5 = vector.mask %0 { vector.multi_reduction <minsi>, %4, %3 [0] : vector<128xi16> to i16 } : vector<128xi1> -> i16
  %6 = vector.broadcast %5 : i16 to vector<i16>
  vector.transfer_write %6, %arg1[] : vector<i16>, memref<i16, #hivm.address_space<ub>>
  return
}

// -----

// test normalize vector when constant_mask for non-last axis contains non-unit masks

// CHECK-LABEL: func.func @test_drop_non_scalable_mask_unit_dims(
// CHECK: %[[mask:.*]] = vector.constant_mask [32] : vector<128xi1>
// CHECK: %[[cast:.*]] = vector.shape_cast %[[mask]] : vector<128xi1> to vector<8x16xi1>
// CHECK: vector.transfer_write {{.*}} %[[cast]]
#map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
func.func @test_drop_non_scalable_mask_unit_dims(%arg0: memref<2x2x16xf16, #hivm.address_space<ub>>, %arg1: memref<2x2x16xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %cst = arith.constant 0.000000e+00 : f16
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [1, 2, 16] : vector<1x8x16xi1>
  %1 = vector.constant_mask [2, 1, 16] : vector<8x1x16xi1>
  scf.for %arg2 = %c0 to %c2 step %c1 {
    %subview = memref.subview %arg0[%arg2, 0, 0] [1, 2, 16] [1, 1, 1] : memref<2x2x16xf16, #hivm.address_space<ub>> to memref<1x2x16xf16, strided<[32, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[0, %arg2, 0] [2, 1, 16] [1, 1, 1] : memref<2x2x16xf16, strided<[272, 16, 1]>, #hivm.address_space<ub>> to memref<2x1x16xf16, strided<[272, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %2 = vector.transfer_read %subview[%c0, %c0, %c0], %cst, %0 {in_bounds = [true, true, true], permutation_map = #map} : memref<1x2x16xf16, strided<[32, 16, 1], offset: ?>, #hivm.address_space<ub>>, vector<8x1x16xf16>
    vector.transfer_write %2, %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : vector<8x1x16xf16>, memref<2x1x16xf16, strided<[272, 16, 1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_tag_multi_reduction_init_from_func_arg
// CHECK: vector.reduction {{.*}} {withoutInitMergeOp}
func.func @test_tag_multi_reduction_init_from_func_arg(%arg0: f32, %arg1: tensor<1x64xf32>) -> vector<1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
  %1 = tensor.empty() : tensor<10xf32>
  %2 = vector.broadcast %arg0 : f32 to vector<1xf32>
  %3 = vector.multi_reduction <add>, %0, %2 [1] : vector<1x64xf32> to vector<1xf32>
  return %3 : vector<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_tag_multi_reduction_init_from_arith_constant
// CHECK: vector.reduction {{.*}} {withoutInitMergeOp}
func.func @test_tag_multi_reduction_init_from_arith_constant(%arg0: f32, %arg1: tensor<1x64xf32>) -> vector<1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
  %1 = tensor.empty() : tensor<10xf32>
  %2 = vector.broadcast %cst : f32 to vector<1xf32>
  %3 = vector.multi_reduction <add>, %0, %2 [1] : vector<1x64xf32> to vector<1xf32>
  return %3 : vector<1xf32>
}