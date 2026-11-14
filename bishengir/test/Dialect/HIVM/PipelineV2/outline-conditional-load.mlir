// RUN: bishengir-opt %s -outline-cv-scopes -verify-each -split-input-file | FileCheck %s
// RUN: bishengir-opt %s -outline-cv-scopes -merge-cv-scopes -memref-dse -verify-each -split-input-file | FileCheck %s --check-prefix=DSE

// A conditional load must remain visible to memref-dse after its allocation
// is hoisted to export a tensor descriptor. Cover both branches, an address
// space cast, a subview destination, and annotations on the tensor descriptor.
// CHECK-LABEL: func.func @conditional_load_then
// CHECK: scf.for
// CHECK-NOT: hivm.cv_pipelined_loop
// CHECK: return
// DSE-LABEL: func.func @conditional_load_then
// DSE-NOT: hivm.cv_pipelined_loop
// DSE: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @conditional_load_then(%src: memref<16x16xf16>, %b: tensor<1x1x16x16xf16>, %dst: memref<16x16xf32>, %ub: i32, %cond: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.if %cond {
        %buffer = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
        %cast = memref.memory_space_cast %buffer : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
        %view = memref.subview %cast[0, 0] [16, 16] [1, 1] : memref<16x16xf16> to memref<16x16xf16, strided<[16, 1]>>
        hivm.hir.load ins(%src : memref<16x16xf16>) outs(%view : memref<16x16xf16, strided<[16, 1]>>) core_type = <VECTOR>
        annotation.mark %cast {MayImplicitTransposeWithLastAxis} : memref<16x16xf16>
        %tensor = bufferization.to_tensor %cast restrict writable : memref<16x16xf16>
        annotation.mark %tensor {MayImplicitTransposeWithLastAxis} : tensor<16x16xf16>
        %empty = tensor.empty() : tensor<16x16xf16>
        %transposed = hivm.hir.vtranspose ins(%tensor : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
        %expanded = tensor.expand_shape %transposed [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %l1 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded : tensor<1x1x16x16xf16>) outs(%l1 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        %a = bufferization.to_tensor %l1 restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%dst : memref<16x16xf32>)
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @conditional_load_else
// CHECK: scf.for
// CHECK-NOT: hivm.cv_pipelined_loop
// CHECK: return
// DSE-LABEL: func.func @conditional_load_else
// DSE-NOT: hivm.cv_pipelined_loop
// DSE: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @conditional_load_else(%src: memref<16x16xf16>, %b: tensor<1x1x16x16xf16>, %dst: memref<16x16xf32>, %ub: i32, %cond: i1)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    scf.for %i = %c0 to %ub step %c1 : i32 {
      scf.if %cond {
      } else {
        %buffer = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
        %cast = memref.memory_space_cast %buffer : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
        %view = memref.subview %cast[0, 0] [16, 16] [1, 1] : memref<16x16xf16> to memref<16x16xf16, strided<[16, 1]>>
        hivm.hir.load ins(%src : memref<16x16xf16>) outs(%view : memref<16x16xf16, strided<[16, 1]>>) core_type = <VECTOR>
        annotation.mark %cast {MayImplicitTransposeWithLastAxis} : memref<16x16xf16>
        %tensor = bufferization.to_tensor %cast restrict writable : memref<16x16xf16>
        annotation.mark %tensor {MayImplicitTransposeWithLastAxis} : tensor<16x16xf16>
        %empty = tensor.empty() : tensor<16x16xf16>
        %transposed = hivm.hir.vtranspose ins(%tensor : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>) permutation = [1, 0] -> tensor<16x16xf16>
        %expanded = tensor.expand_shape %transposed [[0, 1, 2], [3]] output_shape [1, 1, 16, 16] : tensor<16x16xf16> into tensor<1x1x16x16xf16>
        %l1 = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded : tensor<1x1x16x16xf16>) outs(%l1 : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
        %a = bufferization.to_tensor %l1 restrict writable : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
        %init = tensor.empty() : tensor<1x1x16x16xf32>
        %dot = hivm.hir.mmadL1 ins(%a, %b, %true, %c16, %c16, %c16 : tensor<1x1x16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%init : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%dst : memref<16x16xf32>)
      }
    }
    return
  }
}
