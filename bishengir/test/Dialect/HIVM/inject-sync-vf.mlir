// RUN: bishengir-opt -hivm-inject-sync -split-input-file %s | FileCheck %s

// -----
module {
  func.func @outlined_vf(%arg0: memref<16x16x16xf16, #hivm.address_space<ub>>,
                 %arg1: memref<16x16x16xf16, #hivm.address_space<ub>>) attributes {hivm.vector_function} {

    %tmp = memref.alloc() {alignment = 64 : i64} : memref<16x16x16xf16, #hivm.address_space<ub>>
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %subview = memref.subview %arg1[%c0, %c0, %c0] [1, 1, 1] [1, 1, 1] : memref<16x16x16xf16, #hivm.address_space<ub>> to memref<1x1x1xf16, strided<[256, 16, 1], offset: ?>, #hivm.address_space<ub>>
    %0 = vector.transfer_read %subview[%c0, %c0, %c0], %cst {in_bounds = [true]} : memref<1x1x1xf16, strided<[256, 16, 1], offset : ?>, #hivm.address_space<ub>>, vector<1xf16>
    vector.transfer_write %0, %arg0[%c0, %c0, %c0] {in_bounds = [true]} : vector<1xf16>, memref<16x16x16xf16, #hivm.address_space<ub>> 
    return
  }



  func.func @test_mem_inject_sync_basic(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                        %arg3: memref<16x16x16xf16, #hivm.address_space<gm>>)
  attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c8192_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c16384_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @outlined_vf(%0, %4) : (memref<16x16x16xf16, #hivm.address_space<ub>>, memref<16x16x16xf16, #hivm.address_space<ub>>) -> ()
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                   outs(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
