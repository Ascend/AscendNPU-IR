// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960 --hivm-insert-cv-tight-coupled-buffer | FileCheck %s
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910_9589 --hivm-insert-cv-tight-coupled-buffer | FileCheck %s
// RUN: bishengir-opt %s --pass-pipeline="builtin.module(hivm-v2-pre-vectorize-pipeline{target=Ascend960})" | FileCheck %s

// UB placement must preserve masked subview types.
module {
  // CHECK-LABEL: func.func @masked_load
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[ALLOC]] : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
  // CHECK: %[[VIEW:.*]] = memref.subview %[[CAST]]
  // CHECK: hivm.hir.load
  // CHECK-SAME: outs(%[[VIEW]] : memref<?x16xf32, strided<[16, 1]>>)
  // CHECK: bufferization.to_tensor %[[CAST]]
  // CHECK: hivm.hir.mmadL1
  func.func @masked_load(%src: memref<16x16xf32, #hivm.address_space<gm>>,
                         %b: tensor<16x16xf32>, %rows: index)
      -> tensor<16x16xf32> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix"} {
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %alloc = memref.alloc() : memref<16x16xf32>
    %src_view = memref.subview %src[0, 0] [%rows, 16] [1, 1]
        : memref<16x16xf32, #hivm.address_space<gm>>
          to memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<gm>>
    %dst_view = memref.subview %alloc[0, 0] [%rows, 16] [1, 1]
        : memref<16x16xf32> to memref<?x16xf32, strided<[16, 1]>>
    hivm.hir.load ins(%src_view : memref<?x16xf32, strided<[16, 1]>, #hivm.address_space<gm>>)
        outs(%dst_view : memref<?x16xf32, strided<[16, 1]>>)
    %a = bufferization.to_tensor %alloc restrict writable : memref<16x16xf32>
    annotation.mark %a {MayImplicitTransposeWithLastAxis} : tensor<16x16xf32>
    %empty = tensor.empty() : tensor<16x16xf32>
    %result = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16
        : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index)
        outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %result : tensor<16x16xf32>
  }
}
