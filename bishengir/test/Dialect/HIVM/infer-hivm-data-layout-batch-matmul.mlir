// RUN: bishengir-opt %s -allow-unregistered-dialect -hivm-infer-data-layout='batch-matmul=true' -split-input-file | FileCheck %s

// Regbase keeps the batch dimension on a single ND2NZ so that the library call
// can fold it into one MTE2 descriptor, instead of emitting one per batch.
// CHECK-LABEL: regbase_batch_nd2nz
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @regbase_batch_nd2nz(
      %gmA: memref<8x64x64xf16, #hivm.address_space<gm>>,
      %gmB: memref<8x64x64xf16, #hivm.address_space<gm>>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %true = arith.constant true
    %c64 = arith.constant 64 : index
    %l1A = memref.alloc() : memref<8x64x64xf16, #hivm.address_space<cbuf>>
    %l1B = memref.alloc() : memref<8x64x64xf16, #hivm.address_space<cbuf>>
    %l0C = memref.alloc() : memref<8x64x64xf32, #hivm.address_space<cc>>
    // CHECK-NOT: scf.for
    // CHECK: hivm.hir.nd2nz
    hivm.hir.load ins(%gmA : memref<8x64x64xf16, #hivm.address_space<gm>>)
                  outs(%l1A : memref<8x64x64xf16, #hivm.address_space<cbuf>>)
                  init_out_buffer = false
    // CHECK: hivm.hir.nd2nz
    hivm.hir.load ins(%gmB : memref<8x64x64xf16, #hivm.address_space<gm>>)
                  outs(%l1B : memref<8x64x64xf16, #hivm.address_space<cbuf>>)
                  init_out_buffer = false
    hivm.hir.batchMmadL1 ins(%l1A, %l1B, %true, %c64, %c64, %c64 :
        memref<8x64x64xf16, #hivm.address_space<cbuf>>,
        memref<8x64x64xf16, #hivm.address_space<cbuf>>,
        i1, index, index, index)
        outs(%l0C : memref<8x64x64xf32, #hivm.address_space<cc>>)
    return
  }
}
