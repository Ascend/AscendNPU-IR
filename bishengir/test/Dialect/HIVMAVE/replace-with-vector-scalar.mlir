// RUN: bishengir-opt -ave-replace-with-vector-scalar %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_vector_scalar_shift_op
func.func @test_vector_scalar_shift_op(%arg0: memref<1024xi8, #hivm.address_space<ub>>, %arg1: memref<1024xi8, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %c2_i16 = arith.constant 2 : i16
  %0 = ave.hir.pge <ALL> : vector<256xi1>
  %1 = ave.hir.broadcast %c2_i16, %0 : i16, vector<256xi1> -> vector<256xi8>
  %c256 = arith.constant 256 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c1024 step %c256 {
    %subview = memref.subview %arg0[%arg2] [256] [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [256] [1] : memref<1024xi8, #hivm.address_space<ub>> to memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %2 = ave.hir.vload <NORM> %subview[%c0] : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<256xi8>
    %3 = ave.hir.pge <ALL> : vector<256xi1>
    %true = arith.constant true
    // CHECK: ave.hir.vshrs
    %4 = ave.hir.vshr %2, %1, %3, %true : vector<256xi8>, vector<256xi1>
    // CHECK: ave.hir.vshls
    %5 = ave.hir.vshl %2, %1, %3, %true : vector<256xi8>, vector<256xi1>
    %6 = ave.hir.pge <ALL> : vector<256xi1>
    ave.hir.masked_store <NORM_B8> %subview_0[%c0], %6, %4 : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
    ave.hir.masked_store <NORM_B8> %subview_0[%c0], %6, %5 : memref<256xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<256xi1>, vector<256xi8>
  
  }
  return
}

// -----

// CHECK-LABEL: func.func @test_vector_scalar_binary_op
func.func @test_vector_scalar_binary_op(%arg0: memref<1024xf16, #hivm.address_space<ub>>, %arg1: memref<1024xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant 3.000000e+00 : f16
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  %1 = ave.hir.broadcast %cst, %0 : f16, vector<128xi1> -> vector<128xf16>
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c1024 step %c128 {
    %subview = memref.subview %arg0[%arg2] [128] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [128] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %2 = ave.hir.vload <NORM> %subview[%c0] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xf16>
    %3 = ave.hir.pge <ALL> : vector<128xi1>
    // CHECK: ave.hir.vadds
    %4 = ave.hir.vadd %2, %1, %3 : vector<128xf16>, vector<128xi1>
    // CHECK: ave.hir.vmuls
    %5 = ave.hir.vmul %2, %1, %3 : vector<128xf16>, vector<128xi1>
    // CHECK: ave.hir.vmaxs
    %6 = ave.hir.vmax %2, %1, %3 : vector<128xf16>, vector<128xi1>
    // CHECK: ave.hir.vmins
    %7 = ave.hir.vmin %2, %1, %3 : vector<128xf16>, vector<128xi1>
    %8 = ave.hir.pge <ALL> : vector<128xi1>
    ave.hir.masked_store <NORM_B16> %subview_0[%c0], %8, %4 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    ave.hir.masked_store <NORM_B16> %subview_0[%c0], %8, %5 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    ave.hir.masked_store <NORM_B16> %subview_0[%c0], %8, %6 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    ave.hir.masked_store <NORM_B16> %subview_0[%c0], %8, %7 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>

  }
  return
}

// CHECK-LABEL: func.func @test_fold_scalar_broadcast
func.func @test_fold_scalar_broadcast(%arg0: memref<1024xf16, #hivm.address_space<ub>>, %arg1: memref<1024xf16, #hivm.address_space<ub>>) {
  %cst = arith.constant 3.000000e+00 : f16
  %0 = ave.hir.pge <ALL> : vector<128xi1>
  %1 = ave.hir.scalar_broadcast %cst : f16 -> vector<128xf16>
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c1024 step %c128 {
    %subview = memref.subview %arg0[%arg2] [128] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg2] [128] [1] : memref<1024xf16, #hivm.address_space<ub>> to memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %2 = ave.hir.vload <NORM> %subview[%c0] : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<128xf16>
    %3 = ave.hir.pge <ALL> : vector<128xi1>
    // CHECK: ave.hir.vadds
    %4 = ave.hir.vadd %2, %1, %3 : vector<128xf16>, vector<128xi1>
    // CHECK: ave.hir.vmuls
    %5 = ave.hir.vmul %2, %1, %3 : vector<128xf16>, vector<128xi1>
    // CHECK: ave.hir.vmaxs
    %6 = ave.hir.vmax %2, %1, %3 : vector<128xf16>, vector<128xi1>
    // CHECK: ave.hir.vmins
    %7 = ave.hir.vmin %2, %1, %3 : vector<128xf16>, vector<128xi1>
    %8 = ave.hir.pge <ALL> : vector<128xi1>
    ave.hir.masked_store <NORM_B16> %subview_0[%c0], %8, %4 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    ave.hir.masked_store <NORM_B16> %subview_0[%c0], %8, %5 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    ave.hir.masked_store <NORM_B16> %subview_0[%c0], %8, %6 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    ave.hir.masked_store <NORM_B16> %subview_0[%c0], %8, %7 : memref<128xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>

  }
  return
}