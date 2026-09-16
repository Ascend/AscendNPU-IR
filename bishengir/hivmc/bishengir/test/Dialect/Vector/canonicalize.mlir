// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-vector-lowering -cse -canonicalize -split-input-file | FileCheck %s

// REQUIRES: issue2438
// -----
// CHECK-LABEL: fold_extract_broadcast
//  CHECK-SAME:   %[[A:.*]]: f32
//       CHECK:   return %[[A]] : f32
func.func @fold_extract_broadcast(%a : f32) -> f32 {
  %b = vector.broadcast %a : f32 to vector<f32>
  %r = vector.extractelement %b[] : vector<f32>
  return %r : f32
}

// -----
// CHECK-LABEL: func @insert_single_element_to_broadcast
//       CHECK-NOT:   vector.insertelement
#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
func.func @insert_single_element_to_broadcast(%arg0: index, %arg1: memref<1x?xf32, #hivm.address_space<ub>>, %arg2: index, %arg3: index, %arg4: index, %arg5: memref<1xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.for %arg6 = %c0 to %arg0 step %c64 {
      %0 = affine.min #map(%arg6)[%arg0]
      %subview = memref.subview %arg1[0, %arg6] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
      %1 = vector.create_mask %0 : vector<64xi1>
      %subview_1 = memref.subview %subview[0, 0] [1, %0] [1, 1] : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, #map1, #hivm.address_space<ub>>
      %2 = vector.maskedload %subview_1[%c0], %1, %cst_0 : memref<?xf32, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
      %subview_2 = memref.subview %arg5[0] [1] [1] : memref<1xf32, #hivm.address_space<ub>> to memref<f32, #hivm.address_space<ub>>
      %3 = memref.load %subview_2[] : memref<f32, #hivm.address_space<ub>>
      %4 = vector.mask %1 { vector.reduction <add>, %2, %3 : vector<64xf32> into f32 } : vector<64xi1> -> f32
      %5 = vector.insertelement %4, %cst[%c0 : index] : vector<1xf32>
      %6 = vector.shape_cast %5 : vector<1xf32> to vector<f32>
      %7 = vector.extractelement %6[] : vector<f32>
      memref.store %7, %subview_2[] : memref<f32, #hivm.address_space<ub>>
    }
    return
  }
