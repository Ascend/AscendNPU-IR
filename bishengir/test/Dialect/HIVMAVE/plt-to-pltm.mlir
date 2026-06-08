// RUN: bishengir-opt -ave-plt-to-pltm %s -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

func.func @softmax_f32_8_8192_outlined_vf_6(%arg0: index, %arg1: memref<1x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>, %arg2: f32, %arg3: f32, %arg4: memref<1x?xf32, strided<[?, 1]>, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  // CHECK: for %[[IV:.*]] = %{{.*}} to %[[UB:.*]] step %{{.*}} {
  scf.for %arg5 = %c0 to %arg0 step %c64 {
    %0 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg5)[%arg0]
    %subview = memref.subview %arg1[0, %arg5] [1, %0] [1, 1] : memref<1x?xf32, strided<[?, 1]>, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg4[0, %arg5] [1, %0] [1, 1] : memref<1x?xf32, strided<[?, 1]>, #hivm.address_space<ub>> to memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>>
    // CHECK: pltm %[[IV]], %[[UB]]
    %res, %new_true_shape = ave.hir.plt %0 : vector<64xi1>, index
    %subview_1 = memref.subview %subview[0, 0] [1, %0] [1, 1] : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    %1 = ave.hir.vload <NORM> %subview_1[%c0] : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>> into vector<64xf32>
    %2 = ave.hir.scalar_broadcast %arg2 : f32 -> vector<64xf32>
    %5 = ave.hir.scalar_broadcast %arg3 : f32 -> vector<64xf32>
    %6 = ave.hir.vdiv %2, %5, %res : vector<64xf32>, vector<64xi1>
    %subview_2 = memref.subview %subview_0[0, 0] [1, %0] [1, 1] : memref<1x?xf32, strided<[?, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>
    ave.hir.masked_store <NORM_B32> %subview_2[%c0], %res, %6 : memref<?xf32, affine_map<(d0)[s0] -> (d0 + s0)>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
  }
  return
}

// -----

  // CHECK-LABEL: func.func @plt_no_conversion_when_min_operand_is_not_loop_iv
func.func @plt_no_conversion_when_min_operand_is_not_loop_iv() -> vector<64xi1> {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c704 = arith.constant 704 : index
  %all = ave.hir.pge <ALL> : vector<64xi1>
  %mask = scf.for %arg0 = %c0 to %c64 step %c64 iter_args(%iter = %all) -> (vector<64xi1>) {
    %0 = arith.addi %arg0, %c704 : index
    %1 = affine.min affine_map<(d0) -> (-d0 + 1500, 64)>(%0)
    // CHECK-NOT: ave.hir.pltm
    // CHECK: ave.hir.plt
    // CHECK-NOT: ave.hir.pltm
    %res, %new_true_shape = ave.hir.plt %1 : vector<64xi1>, index
    scf.yield %res : vector<64xi1>
  }
  return %mask : vector<64xi1>
}
