// RUN: bishengir-opt %s -expand-strided-metadata --convert-hivm-to-llvm | FileCheck %s

#map0 = affine_map<(d0) -> (2*d0+7)>
func.func @memref_lowering_static_subview16(%arg1: memref<32xf32>) -> memref<16xf32,#map0> {

  // CHECK: llvm.inttoptr
  %addr = arith.constant 1234 : i64
  %tmp_base = hivm.hir.pointer_cast(%addr)[] : memref<64xf32>
  %tmp = memref.subview %tmp_base[7][16][2]: memref<64xf32> to memref<16xf32, #map0>

  %const = arith.constant 0.33333 : f32
  affine.store %const, %tmp[0] : memref<16xf32,#map0>
  affine.for %i = 1 to 16 {
    %1 = affine.load %tmp[%i - 1] : memref<16xf32,#map0>
    %2 = affine.load %arg1[%i] : memref<32xf32>
    %3 = arith.addf %1, %2 : f32
    affine.store %3, %tmp[%i] : memref<16xf32,#map0>
  }
  return %tmp : memref<16xf32,#map0>
}

func.func @memref_lowering_dynamic(%arg0: memref<?xf32,#hivm.address_space<ub>>, %addr: i64)
  -> memref<?xf32,#hivm.address_space<ub>> {

  // CHECK: llvm.inttoptr
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?xf32,#hivm.address_space<ub>>
  %tmp = hivm.hir.pointer_cast(%addr)[%0] : memref<?xf32,#hivm.address_space<ub>>

  %const = arith.constant 0.33333 : f32
  affine.store %const, %tmp[0] : memref<?xf32,#hivm.address_space<ub>>
  affine.for %i = 1 to 32 {
    %1 = affine.load %tmp[%i - 1] : memref<?xf32,#hivm.address_space<ub>>
    %2 = affine.load %arg0[%i] : memref<?xf32,#hivm.address_space<ub>>
    %3 = arith.addf %1, %2 : f32
    affine.store %3, %tmp[%i] : memref<?xf32,#hivm.address_space<ub>>
  }
  return %tmp : memref<?xf32,#hivm.address_space<ub>>
}

func.func @memref_lowering_static(%arg1: memref<32xf32>) -> memref<32xf32> {

  // CHECK: llvm.inttoptr
  %addr = arith.constant 1234 : i64
  %tmp = hivm.hir.pointer_cast(%addr)[] : memref<32xf32>

  %const = arith.constant 0.33333 : f32
  affine.store %const, %tmp[0] : memref<32xf32>
  affine.for %i = 1 to 32 {
    %1 = affine.load %tmp[%i - 1] : memref<32xf32>
    %2 = affine.load %arg1[%i] : memref<32xf32>
    %3 = arith.addf %1, %2 : f32
    affine.store %3, %tmp[%i] : memref<32xf32>
  }
  return %tmp : memref<32xf32>
}
