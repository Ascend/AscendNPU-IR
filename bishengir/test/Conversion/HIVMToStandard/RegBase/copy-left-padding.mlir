// RUN: bishengir-opt %s -convert-hivm-to-std -split-input-file | FileCheck %s

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @copyop2d_load_left_padding() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1 = arith.constant 1 : index
    %pad = arith.constant 0.000000e+00 : f32
    %src_storage = memref.alloc() : memref<16xf32, #hivm.address_space<gm>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [2, 8], strides: [8, 1]
      : memref<16xf32, #hivm.address_space<gm>> to memref<2x8xf32, strided<[8, 1]>, #hivm.address_space<gm>>
    %dst_storage = memref.alloc() : memref<34xf32, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [2, 16], strides: [17, 1]
      : memref<34xf32, #hivm.address_space<ub>> to memref<2x16xf32, strided<[17, 1]>, #hivm.address_space<ub>>
    // CHECK-LABEL: @copyop2d_load_left_padding
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: call @load_gm_to_ubuf_2d_float(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C1]], %{{.*}})
    hivm.hir.load ins(%src : memref<2x8xf32, strided<[8, 1]>, #hivm.address_space<gm>>)
                  outs(%dst : memref<2x16xf32, strided<[17, 1]>, #hivm.address_space<ub>>)
                  pad_mode = #hivm.padmode<PadValue>
                  pad_value = %pad : f32
                  left_padding_num = %c1 : index
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @copyop3d_load_left_padding() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c1 = arith.constant 1 : index
    %pad = arith.constant 0.000000e+00 : f32
    %src_storage = memref.alloc() : memref<32xf32, #hivm.address_space<gm>>
    %src = memref.reinterpret_cast %src_storage to offset: [0], sizes: [2, 2, 8], strides: [16, 8, 1]
      : memref<32xf32, #hivm.address_space<gm>> to memref<2x2x8xf32, strided<[16, 8, 1]>, #hivm.address_space<gm>>
    %dst_storage = memref.alloc() : memref<68xf32, #hivm.address_space<ub>>
    %dst = memref.reinterpret_cast %dst_storage to offset: [0], sizes: [2, 2, 16], strides: [34, 17, 1]
      : memref<68xf32, #hivm.address_space<ub>> to memref<2x2x16xf32, strided<[34, 17, 1]>, #hivm.address_space<ub>>
    // CHECK-LABEL: @copyop3d_load_left_padding
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: call @load_gm_to_ubuf_3d_float(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C1]], %{{.*}})
    hivm.hir.load ins(%src : memref<2x2x8xf32, strided<[16, 8, 1]>, #hivm.address_space<gm>>)
                  outs(%dst : memref<2x2x16xf32, strided<[34, 17, 1]>, #hivm.address_space<ub>>)
                  pad_mode = #hivm.padmode<PadValue>
                  pad_value = %pad : f32
                  left_padding_num = %c1 : index
    return
  }
}
