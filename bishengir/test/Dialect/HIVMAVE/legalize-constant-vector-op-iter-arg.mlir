// RUN: bishengir-opt %s -normalize-vector -scf-for-loop-canonicalization -cse -canonicalize -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

#map = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
func.func @rms_norm_f32_split_n_outlined_vf_1(%arg0: index, %arg1: memref<?x128xf32, #hivm.address_space<ub>>, %arg2: memref<?xf32, #hivm.address_space<ub>>, %arg3: memref<1x64xf32, #hivm.address_space<ub>>, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index, %arg14: memref<?xf32, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NOT: arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
  %cst_1 = arith.constant dense<7.812500e-03> : vector<64xf32>
  %cst_2 = arith.constant dense<1.000000e-01> : vector<64xf32>
  %cst_3 = arith.constant dense<1.000000e+00> : vector<64xf32>
  scf.for %arg15 = %c0 to %arg0 step %c64 {
    %0 = affine.min #map(%arg15)[%arg0]
    %subview = memref.subview %arg1[%arg15, 0] [%0, 128] [1, 1] : memref<?x128xf32, #hivm.address_space<ub>> to memref<?x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_4 = memref.subview %arg2[%arg15] [%0] [1] : memref<?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    scf.for %arg16 = %c0 to %0 step %c1 {
      %subview_6 = memref.subview %subview[%arg16, 0] [1, 128] [1, 1] : memref<?x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_7 = memref.subview %subview_4[%arg16] [1] [1] : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %7 = scf.for %arg17 = %c0 to %c128 step %c64 iter_args(%arg18 = %cst_0) -> (vector<1x64xf32>) {
        %subview_8 = memref.subview %subview_6[0, %arg17] [1, 64] [1, 1] : memref<1x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %10 = vector.transfer_read %subview_8[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %11 = arith.mulf %10, %10 : vector<1x64xf32>
        %12 = arith.addf %arg18, %11 : vector<1x64xf32>
        scf.yield %12 : vector<1x64xf32>
      }
      %8 = vector.transfer_read %subview_7[%c0], %cst {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %9 = vector.multi_reduction <add>, %7, %8 [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %9, %subview_7[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    %subview_5 = memref.subview %arg14[%arg15] [%0] [1] : memref<?xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %1 = vector.create_mask %0 : vector<64xi1>
    %2 = vector.transfer_read %subview_4[%c0], %cst, %1 {in_bounds = [true]} : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xf32>
    %3 = arith.mulf %2, %cst_1 : vector<64xf32>
    %4 = arith.addf %3, %cst_2 : vector<64xf32>
    %5 = math.sqrt %4 : vector<64xf32>
    %6 = arith.divf %cst_3, %5 : vector<64xf32>
    vector.transfer_write %6, %subview_5[%c0], %1 {in_bounds = [true]} : vector<64xf32>, memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  }
  return
}