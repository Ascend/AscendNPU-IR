// RUN: not bishengir-opt %s --convert-hivm-to-tritongpu 2>&1 | FileCheck %s

// CHECK: error: 'memref.subview' op parent memref.reinterpret_cast before memref.subview is not supported: parent view offset is not yet composed

module {
  func.func @nested_reinterpret_subview(
      %arg0: memref<?xi64, #hivm.address_space<ub>>, %arg1: index,
      %arg2: memref<2xi64, #hivm.address_space<ub>>)
      attributes {no_inline, outline, vector_function,
                  vf_mode = #hivm.vf_mode<SIMT>} {
    %view = memref.reinterpret_cast %arg0 to offset: [3], sizes: [8], strides: [1] : memref<?xi64, #hivm.address_space<ub>> to memref<8xi64, strided<[1], offset: 3>, #hivm.address_space<ub>>
    %subview = memref.subview %view[%arg1] [2] [1] : memref<8xi64, strided<[1], offset: 3>, #hivm.address_space<ub>> to memref<2xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = bufferization.to_tensor %subview restrict writable : memref<2xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg2 : memref<2xi64, #hivm.address_space<ub>>, %0 : tensor<2xi64>)
    return
  }
}
