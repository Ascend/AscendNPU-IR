// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-inject-sync{sync-mode=barrier-all}))" -split-input-file %s | FileCheck %s

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9362">} {
  func.func @test_mmadl1_m_mte1(%arg0: memref<16xf32, #hivm.address_space<gm>>,
                                %arg1: memref<16xf32, #hivm.address_space<gm>>,
                                %arg2: memref<256xf32, #hivm.address_space<gm>>) {
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                   outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    %1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                   outs(%1 : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK-NOT: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>,
                        memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index)
                        outs(%2 : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK-NOT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.fixpipe {enable_nz2nd} ins(%2 : memref<256xf32, #hivm.address_space<cc>>)
                     outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_mmadl1_m_mte1(%arg0: memref<16xf32, #hivm.address_space<gm>>,
                                %arg1: memref<16xf32, #hivm.address_space<gm>>,
                                %arg2: memref<256xf32, #hivm.address_space<gm>>) {
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                   outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    %1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                   outs(%1 : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>,
                        memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index)
                        outs(%2 : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    hivm.hir.fixpipe {enable_nz2nd} ins(%2 : memref<256xf32, #hivm.address_space<cc>>)
                     outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
