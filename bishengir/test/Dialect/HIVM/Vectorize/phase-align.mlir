// RUN: bishengir-opt %s \
// RUN:   --hacc-append-device-spec=target=Ascend960DT_968B7 \
// RUN:   --hivm-vectorize=phase=align --verify-each | FileCheck %s

// The align phase protects vector-function allocation layouts from stride
// padding. Only rank-2-or-higher allocations passed to a vector function are
// marked; unrelated allocations are left alone.

module {
  func.func private @vf(
      %src: memref<8x16xf32, #hivm.address_space<ub>>,
      %dst: memref<8x16xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %v = memref.load %src[%c0, %c0]
        : memref<8x16xf32, #hivm.address_space<ub>>
    memref.store %v, %dst[%c0, %c0]
        : memref<8x16xf32, #hivm.address_space<ub>>
    return
  }

  func.func @caller() {
    %wide = memref.alloc() : memref<8x16xf32, #hivm.address_space<ub>>
    %unused = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    call @vf(%wide, %wide)
        : (memref<8x16xf32, #hivm.address_space<ub>>,
           memref<8x16xf32, #hivm.address_space<ub>>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @caller(
// CHECK: %[[WIDE:.*]] = memref.alloc()
// CHECK-SAME: memref<8x16xf32, #hivm.address_space<ub>>
// CHECK-NEXT: annotation.mark %[[WIDE]]
// CHECK-SAME: hivm.skip_stride_align_for_vload
// CHECK: %[[UNUSED:.*]] = memref.alloc()
// CHECK-SAME: memref<16xf32, #hivm.address_space<ub>>
// CHECK-NOT: annotation.mark %[[UNUSED]]
// CHECK: call @vf(%[[WIDE]], %[[WIDE]])
