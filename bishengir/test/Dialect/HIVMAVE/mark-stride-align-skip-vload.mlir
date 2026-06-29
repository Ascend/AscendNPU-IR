// RUN: bishengir-opt -hivm-mark-stride-align -split-input-file %s | FileCheck %s

// A buffer read by TransferReadOp (vload) in a vf must not get stride
// alignment; the padding would create non-contiguous strides that the
// vldsx1 flat-pointer load cannot handle.
// The vsstb output buffer still gets alignment for bank-conflict avoidance.

// CHECK-LABEL: func.func @test_vload_arg_skipped
// CHECK-NOT: annotation.mark %alloc  {hivm.stride_align_dims
// CHECK:     annotation.mark %alloc_0 {hivm.stride_align_dims

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @vf(%arg0: memref<3x11x8xf16, #hivm.address_space<ub>>,
                %arg1: memref<11x3x8xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %c11 = arith.constant 11 : index
    scf.for %i = %c0 to %c3 step %c1 {
      scf.for %j = %c0 to %c11 step %c8 {
        %n = affine.min affine_map<(d0) -> (-d0 + 11, 8)>(%j)
        %s = memref.subview %arg0[%i, %j, 0] [1, %n, 8] [1, 1, 1]
            : memref<3x11x8xf16, #hivm.address_space<ub>>
            to memref<1x?x8xf16, strided<[88, 8, 1], offset: ?>, #hivm.address_space<ub>>
        %m = vector.create_mask %c1, %n, %c8 : vector<1x8x8xi1>
        %v = vector.transfer_read %s[%c0, %c0, %c0], %cst, %m {in_bounds = [true, true, true]}
            : memref<1x?x8xf16, strided<[88, 8, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x8x8xf16>
        %e = arith.extf %v : vector<1x8x8xf16> to vector<1x8x8xf32>
        %so = memref.subview %arg1[%j, %i, 0] [%n, 1, 8] [1, 1, 1]
            : memref<11x3x8xf32, #hivm.address_space<ub>>
            to memref<?x1x8xf32, strided<[24, 8, 1], offset: ?>, #hivm.address_space<ub>>
        %sc = vector.shape_cast %e : vector<1x8x8xf32> to vector<8x1x8xf32>
        %m2 = vector.create_mask %n, %c1, %c8 : vector<8x1x8xi1>
        vector.transfer_write %sc, %so[%c0, %c0, %c0], %m2 {in_bounds = [true, true, true]}
            : vector<8x1x8xf32>, memref<?x1x8xf32, strided<[24, 8, 1], offset: ?>, #hivm.address_space<ub>>
      } {unroll_for_vsstb}
    }
    return
  }

  func.func @test_vload_arg_skipped() {
    %arg0 = memref.alloc() : memref<3x11x8xf16, #hivm.address_space<ub>>
    %arg1 = memref.alloc() : memref<11x3x8xf32, #hivm.address_space<ub>>
    func.call @vf(%arg0, %arg1) {hivm.vector_function, no_inline}
        : (memref<3x11x8xf16, #hivm.address_space<ub>>,
           memref<11x3x8xf32, #hivm.address_space<ub>>) -> ()
    return
  }
}
