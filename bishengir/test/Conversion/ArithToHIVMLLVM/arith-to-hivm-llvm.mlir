// RUN: bishengir-opt %s -convert-arith-to-hivm-llvm -convert-arith-to-llvm | FileCheck %s

// CHECK-LABEL: func @arith_select_to_llvm
#map = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
module {
  func.func @arith_select_to_llvm(%arg0: memref<16xf32>) {
    %alloca = memref.alloca() : memref<f32, #hivm.address_space<ub>>
    %alloca_0 = memref.alloca() : memref<f32, #hivm.address_space<ub>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %st = arith.constant 4.5 : f32
    scf.for %arg1 = %c0 to %c16 step %c4 {
      %0 = affine.apply #map(%arg1)
      %1 = arith.index_cast %0 : index to i1
      %2 = arith.select %1, %alloca, %alloca_0 : memref<f32, #hivm.address_space<ub>>
      // CHECK: %[[new_select:.*]] = llvm.select %[[cond:.*]], %[[ptr1:.*]], %[[ptr2:.*]] : i1, !llvm.ptr<6>
      // CHECK: %[[ucc:.*]] = builtin.unrealized_conversion_cast %[[descr:.*]] : !llvm.struct<(ptr<6>, ptr<6>, i64)> to memref<f32, #hivm.address_space<ub>>
      memref.store %st, %2[] : memref<f32, #hivm.address_space<ub>>
    }
    return
  }
}