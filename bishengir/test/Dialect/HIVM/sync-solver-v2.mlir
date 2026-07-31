// RUN: bishengir-opt -hivm-graph-sync-solver -hivm-lower-multi-buffer-counter %s | FileCheck %s

module {
  func.func @skip_covered_processing_orders(
      %arg0: memref<16xf32, #hivm.address_space<gm>>,
      %arg1: memref<256xf32, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index

    %0 = hivm.hir.pointer_cast(%c0_i64) :
        memref<16xf32, #hivm.address_space<cbuf>>
    // CHECK-LABEL: func.func @skip_covered_processing_orders
    // CHECK: hivm.hir.nd2nz
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>
    hivm.hir.nd2nz {dst_continuous}
        ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
        outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)

    %1 = hivm.hir.pointer_cast(%c64_i64) :
        memref<16xf32, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.nd2nz
    // CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>
    hivm.hir.nd2nz {dst_continuous}
        ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
        outs(%1 : memref<16xf32, #hivm.address_space<cbuf>>)

    %2 = hivm.hir.pointer_cast(%c0_i64) :
        memref<256xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.mmadL1
    hivm.hir.mmadL1
        ins(%0, %1, %true, %c16, %c256, %c16 :
            memref<16xf32, #hivm.address_space<cbuf>>,
            memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index)
        outs(%2 : memref<256xf32, #hivm.address_space<cc>>)

    hivm.hir.fixpipe {enable_nz2nd}
        ins(%2 : memref<256xf32, #hivm.address_space<cc>>)
        outs(%arg1 : memref<256xf32, #hivm.address_space<gm>>)
    return
  }
}
