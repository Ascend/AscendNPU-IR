// RUN: bishengir-opt %s -normalize-vector -cse | FileCheck %s

#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
func.func @test_legalize_vector_storage_reduction(%arg0: index, %arg1: memref<1x?xf32, #hivm.address_space<ub>>, %arg2: index, %arg3: index, %arg4: index, %arg5: memref<1xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  scf.for %arg6 = %c0 to %arg0 step %c64 {
    %0 = affine.min #map(%arg6)[%arg0]
    %subview = memref.subview %arg1[0, %arg6] [1, %0] [1, 1] : memref<1x?xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    %1 = vector.create_mask %0 : vector<64xi1>
    %subview_0 = memref.subview %subview[0, 0] [1, %0] [1, 1] : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, #map1, #hivm.address_space<ub>>
    %2 = vector.maskedload %subview_0[%c0], %1, %cst : memref<?xf32, #map1, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
    %subview_1 = memref.subview %arg5[0] [1] [1] : memref<1xf32, #hivm.address_space<ub>> to memref<f32, #hivm.address_space<ub>>
    %3 = memref.load %subview_1[] : memref<f32, #hivm.address_space<ub>>
    %4 = vector.mask %1 { vector.reduction <add>, %2, %3 : vector<64xf32> into f32 } : vector<64xi1> -> f32
    memref.store %4, %subview_1[] : memref<f32, #hivm.address_space<ub>>
  }
  return
}

// CHECK-LABEL:       func.func @test_legalize_vector_storage_reduction
// CHECK:             %[[REDUCTION:.+]] = vector.mask {{.*}} { vector.reduction <add>, %[[VECTOR:.+]], %[[ACC:.+]] : vector<64xf32> into f32 } : vector<64xi1> -> f32
// CHECK:             memref.store %[[REDUCTION]]