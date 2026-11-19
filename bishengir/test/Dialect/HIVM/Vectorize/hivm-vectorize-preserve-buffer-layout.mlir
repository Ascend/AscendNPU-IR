// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize=phase=align --hivm-enable-stride-align --verify-each | FileCheck %s --implicit-check-not=unrealized_conversion_cast

module {
  // CHECK-LABEL: func.func private @reduce
  // CHECK-SAME: memref<69x1xf32, #hivm.address_space<ub>>
  // CHECK: memref.collapse_shape
  func.func private @reduce(
      %src: memref<69x128xf32, #hivm.address_space<ub>>,
      %dst: memref<69x1xf32, #hivm.address_space<ub>>)
      attributes {hivm.vector_function} {
    %flat = memref.collapse_shape %dst [[0, 1]]
        : memref<69x1xf32, #hivm.address_space<ub>>
        into memref<69xf32, #hivm.address_space<ub>>
    %zero = arith.constant 0 : index
    %value = arith.constant 0.0 : f32
    memref.store %value, %flat[%zero]
        : memref<69xf32, #hivm.address_space<ub>>
    return
  }

  // CHECK-LABEL: func.func @caller
  // CHECK: memref.alloc() : memref<69x128xf32, #hivm.address_space<ub>>
  // CHECK: hivm.skip_stride_align_for_vload
  // CHECK: memref.alloc() : memref<69x1xf32, #hivm.address_space<ub>>
  // CHECK: hivm.skip_stride_align_for_vload
  // CHECK: call @reduce
  // CHECK-SAME: memref<69x1xf32, #hivm.address_space<ub>>
  func.func @caller() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %src = memref.alloc() : memref<69x128xf32, #hivm.address_space<ub>>
    %dst = memref.alloc() : memref<69x1xf32, #hivm.address_space<ub>>
    func.call @reduce(%src, %dst)
        : (memref<69x128xf32, #hivm.address_space<ub>>,
           memref<69x1xf32, #hivm.address_space<ub>>) -> ()
    return
  }
}
