// RUN: bishengir-opt -hivm-inject-sync -split-input-file %s | FileCheck %s

// -----
module {
  func.func @test_mem_injcet_sync_basic(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
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
    hivm.hir.vadd ins(%0, %4 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                  memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                   outs(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

// -----
module {
  func.func @test_injcet_sync_loop(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                   %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>){
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg4 = %c0 to %c1024 step %c128 {
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                    outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                     outs(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    return
  }
}

// -----

module {
  func.func @test_injcet_sync_loop_nest(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                        %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>){
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg2 = %c0 to %c1024 step %c128 {
      scf.for %arg4 = %c0 to %c1024 step %c128 {
        // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                          outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
         // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
        hivm.hir.store ins(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                           outs(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>)
        // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      }
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    return
  }
}

// -----

module {
  func.func @test_injcet_sync_if_else(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg2: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg3: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg4: i1) {
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c8192_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c16384_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_MTE2>]
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%4 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%5 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %6 = scf.if %arg4 -> (memref<16x16x16xf16, #hivm.address_space<ub>>) {
      hivm.hir.vadd ins(%0, %4 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                    memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      scf.yield %0 : memref<16x16x16xf16, #hivm.address_space<ub>>
    } else {
      hivm.hir.vadd ins(%2, %5 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                    memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      scf.yield %2 : memref<16x16x16xf16, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%6 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_injcet_sync_if(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg2: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg3: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg4: i1) {
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
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    scf.if %arg4 {
      hivm.hir.vadd ins(%0, %4 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                    memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    }
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                   outs(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_injcet_sync_collapse_shape(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %arg2: memref<256x16xf16, #hivm.address_space<gm>>) {
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c8192_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vadd ins(%0, %1 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                               memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %collapse_shape = memref.collapse_shape %2 [[0, 1], [2]] :
                      memref<16x16x16xf16, #hivm.address_space<ub>> into
                      memref<256x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%collapse_shape : memref<256x16xf16, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<256x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_injcet_sync_expand_shape(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                           %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                           %arg2: memref<2x8x16x16xf16, #hivm.address_space<gm>>) {
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c8192_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vadd ins(%0, %1 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                               memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %expand_shape = memref.expand_shape %2 [[0, 1], [2], [3]] output_shape [2, 8, 16, 16] :
                    memref<16x16x16xf16, #hivm.address_space<ub>> into
                    memref<2x8x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%expand_shape : memref<2x8x16x16xf16, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<2x8x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_injcet_sync_two_event_id(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg2: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                 %arg3: memref<16x16x16xf16, #hivm.address_space<gm>>) {
    %c24576_i64 = arith.constant 24576 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c8192_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c16384_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c24576_i64) [] : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vadd ins(%0, %4 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                  memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.vadd ins(%2, %5 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                  memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.store ins(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_inject_sync_dyn_view_and_reinterpret_cast(%arg0: memref<1x?xi16, #hivm.address_space<gm>>,
                                                            %arg1: memref<?x1xi16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c1 : memref<1x?xi16, #hivm.address_space<gm>>
    %0 = hivm.hir.pointer_cast(%c0_i64) [] : memref<2048xi8, #hivm.address_space<ub>>
    %view = memref.view %0[%c0][%dim] : memref<2048xi8, #hivm.address_space<ub>> to
                                        memref<1x?xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<1x?xi16, #hivm.address_space<gm>>)
                  outs(%view : memref<1x?xi16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast = memref.reinterpret_cast %view to
                                  offset: [0], sizes: [%dim, 1], strides: [1, 1] :
                        memref<1x?xi16, #hivm.address_space<ub>> to memref<?x1xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%reinterpret_cast : memref<?x1xi16, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<?x1xi16, #hivm.address_space<gm>>)
    return
  }
}

// -----

// CHECK: #map = affine_map<()[s0] -> ((s0 floordiv 4) mod 2)>
module {
  func.func @test_db_two_address(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                 %arg1: memref<16xf16, #hivm.address_space<gm>>) {
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[c4:.*]] = arith.constant 4 : index
    // CHECK: %[[c16:.*]] = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: scf.for %[[arg2:.*]] = %[[c0]] to %[[c16]] step %[[c4]] {
    scf.for %arg2 = %c0 to %c16 step %c4 {
      // CHECK: %[[T0:.*]] = affine.apply #map()[%[[arg2]]]
      // CHECK: %[[T1:.*]] = arith.index_cast %[[T0]] : index to i1
      // CHECK: %[[T2:.*]] = arith.constant 0 : i64
      // CHECK: %[[T3:.*]] = arith.constant 1 : i64
      // CHECK: %[[T4:.*]] = arith.select %[[T1]], %[[T2]], %[[T3]] : i64
      %0 = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %[[T4]]]
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%0 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %[[T4]]]

    }
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    return
  }
}

// -----
module {
 func.func @test_db_two_address_two_buffer(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg1: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg2: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg3: memref<16xf16, #hivm.address_space<gm>>) {
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c96_i64 = arith.constant 96 : i64
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[c4:.*]] = arith.constant 4 : index
    // CHECK: %[[c16:.*]] = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    // CHECK: scf.for %[[arg2:.*]] = %[[c0]] to %[[c16]] step %[[c4]] {
    scf.for %arg4 = %c0 to %c16 step %c4 {
      // CHECK: %[[T0:.*]] = affine.apply #map()[%[[arg2]]]
      // CHECK: %[[T1:.*]] = arith.index_cast %[[T0]] : index to i1
      // CHECK: %[[T2:.*]] = arith.constant 2 : i64
      // CHECK: %[[T3:.*]] = arith.constant 3 : i64
      // CHECK: %[[T4:.*]] = arith.select %[[T1]], %[[T2]], %[[T3]] : i64
      // CHECK: %[[T5:.*]] = arith.constant 0 : i64
      // CHECK: %[[T6:.*]] = arith.constant 1 : i64
      // CHECK: %[[T7:.*]] = arith.select %[[T1]], %[[T5]], %[[T6]] : i64

      %0 = hivm.hir.pointer_cast(%c0_i64, %c32_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %[[T7]]]
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.store ins(%0 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %[[T7]]]

      %1 = hivm.hir.pointer_cast(%c64_i64, %c96_i64) : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %1 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %[[T4]]]
      hivm.hir.load ins(%arg2 : memref<16xf16, #hivm.address_space<gm>>) outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID1>]
      hivm.hir.store ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg3 : memref<16xf16, #hivm.address_space<gm>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %[[T4]]]

    }
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID2>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID3>]
    return
  }
}

// -----
module {
  func.func @test_mem_memref_load_store(%arg0: memref<1024xf16, #hivm.address_space<gm>>,
                                        %arg1: memref<1024xf16, #hivm.address_space<gm>>) {
    %c2048_i64 = arith.constant 2048 : i64
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf16, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c2048_i64) : memref<1024xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>)
                  outs(%0 : memref<1024xf16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
    scf.for %arg2 = %c0 to %c1024 step %c1 {
      %2 = memref.load %0[%arg2] : memref<1024xf16, #hivm.address_space<ub>>
      %3 = memref.load %0[%arg2] : memref<1024xf16, #hivm.address_space<ub>>
      %4 = arith.addf %2, %3 : f16
      memref.store %4, %1[%arg2] : memref<1024xf16, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%1 : memref<1024xf16, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<1024xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_widen_sync(%arg0: memref<1024xf16, #hivm.address_space<gm>>,
                             %arg1: memref<1024xf16, #hivm.address_space<gm>>,
                             %arg2: memref<1024xf16, #hivm.address_space<gm>>,
                             %arg3: memref<1024xf16, #hivm.address_space<gm>>,
                             %arg4: memref<1024xf16, #hivm.address_space<gm>>,
                             %arg5: memref<1024xf16, #hivm.address_space<gm>>,
                             %arg6: memref<1024xf16, #hivm.address_space<gm>>,
                             %arg7: memref<1024xf16, #hivm.address_space<gm>>) {
    %c14336_i64 = arith.constant 14336 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c6144_i64 = arith.constant 6144 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID2>]
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID3>]
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID4>]
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID5>]
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID6>]
    // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID7>]
    scf.for %arg8 = %c0 to %c1024 step %c1 {
      %0 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf16, #hivm.address_space<ub>>
      %1 = hivm.hir.pointer_cast(%c2048_i64) : memref<1024xf16, #hivm.address_space<ub>>
      %2 = hivm.hir.pointer_cast(%c4096_i64) : memref<1024xf16, #hivm.address_space<ub>>
      %3 = hivm.hir.pointer_cast(%c6144_i64) : memref<1024xf16, #hivm.address_space<ub>>
      %4 = hivm.hir.pointer_cast(%c8192_i64) : memref<1024xf16, #hivm.address_space<ub>>
      %5 = hivm.hir.pointer_cast(%c10240_i64) : memref<1024xf16, #hivm.address_space<ub>>
      %6 = hivm.hir.pointer_cast(%c12288_i64) : memref<1024xf16, #hivm.address_space<ub>>
      %7 = hivm.hir.pointer_cast(%c14336_i64) : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>) outs(%0 : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>) outs(%1 : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID1>]
      // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>) outs(%2 : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID2>]
      // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>) outs(%3 : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID3>]
      // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID4>]
      hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>) outs(%4 : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID4>]
      // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID5>]
      hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>) outs(%5 : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID5>]
      // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID6>]
      hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>) outs(%6 : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID6>]
      // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID7>]
      hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>) outs(%7 : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID7>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]
      %8 = memref.load %0[%c0] : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID1>]
      %9 = memref.load %1[%c0] : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID2>]
      %10 = memref.load %2[%c0] : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID2>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID3>]
      %11 = memref.load %3[%c0] : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID3>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID4>]
      %12 = memref.load %4[%c0] : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID4>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID5>]
      %13 = memref.load %5[%c0] : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID5>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID6>]
      %14 = memref.load %6[%c0] : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID6>]
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID7>]
      %15 = memref.load %7[%c0] : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID7>]
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID1>]
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID2>]
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID3>]
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID4>]
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID5>]
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID6>]
    // CHECK: hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID7>]
    return
  }
}

// -----
module {

  func.func @test_sync_for_cube(%arg0: memref<16xf32, #hivm.address_space<gm>>,
                               %arg1: memref<16xf32, #hivm.address_space<gm>>,
                               %arg2: memref<256xf32, #hivm.address_space<gm>>) {
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                   outs(%0 : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %1 = hivm.hir.pointer_cast(%c64_i64) : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                   outs(%1 : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK:   hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: sync_related_args(%c1_i64, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64
    hivm.hir.mmadL1 ins(%0, %1, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>,
                        memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index)
                        outs(%2 : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {enable_nz2nd} ins(%2 : memref<256xf32, #hivm.address_space<cc>>)
                     outs(%arg2 : memref<256xf32, #hivm.address_space<gm>>)
    return
  }
}


// -----
#map = affine_map<()[s0] -> (s0 + 32)>
#map1 = affine_map<()[s0, s1] -> (s0 - s1)>
#map2 = affine_map<()[s0] -> ((s0 + 15) floordiv 16)>
#map3 = affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1 + 32)>
module {
  func.func @matmul_x_w_bias_down_up_fused_layer_1_kernel_mix_aic(
      %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, 
      %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, 
      %arg2: memref<?xf16, #hivm.address_space<gm>>, %arg3: memref<?xf16, #hivm.address_space<gm>>, 
      %arg4: memref<?xf16, #hivm.address_space<gm>>, %arg5: memref<?xf16, #hivm.address_space<gm>>, 
      %arg6: memref<?xf16, #hivm.address_space<gm>>, %arg7: memref<?xf16, #hivm.address_space<gm>>, %arg8: i32, %arg9: i32) 
    {
    %c1 = arith.constant 1 : index
    %c20480_i64 = arith.constant 20480 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c18432_i64 = arith.constant 18432 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c0_i64 = arith.constant 0 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c1_i32 = arith.constant 1 : i32
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    hivm.hir.set_ffts_base_addr %arg0
    hivm.hir.set_mask_norm
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [32, 32], strides: [%c1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %cast = memref.cast %reinterpret_cast : memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%c0], sizes: [32, 32], strides: [%c1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
    %cast_1 = memref.cast %reinterpret_cast_0 : memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [32, 64], strides: [%c1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x64xf16, strided<[?, 1]>, #hivm.address_space<gm>>
    %cast_3 = memref.cast %reinterpret_cast_2 : memref<32x64xf16, strided<[?, 1]>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
    %0 = hivm.hir.pointer_cast(%c8192_i64) : memref<2x2x16x16xf32, #hivm.address_space<cc>>
    %cast_4 = memref.cast %0 : memref<2x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<4x2x16x16xf32, #hivm.address_space<cc>>
    %cast_5 = memref.cast %1 : memref<4x2x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    %2:9 = scf.for %arg10 = %c0_i32 to %c0_i32 step %c1_i32 iter_args(%arg11 = %cast, %arg12 = %cast_1, %arg13 = %cast_3, %arg14 = %c0, %arg15 = %c0, %arg16 = %c0, %arg17 = %c0, %arg18 = %c0, %arg19 = %c0) -> (memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, index, index, index, index, index)  : i32 {
      %3 = arith.muli %arg10, %c32_i32 : i32
      %4 = hivm.hir.pointer_cast(%c0_i64, %c16384_i64) : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %4 {hivm.multi_buffer = 2 : i32} : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
      %cast_6 = memref.cast %4 : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
      %5 = arith.index_cast %3 : i32 to index
      %6 = affine.apply #map()[%5]
      %7 = arith.maxsi %5, %c0 : index
      %8 = arith.minsi %6, %7 : index
      %9 = affine.apply #map1()[%8, %5]
      %10 = arith.minsi %9, %c32 : index
      %subview = memref.subview %arg11[0, 0] [0, %10] [1, 1] : memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<0x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_7 = memref.cast %subview : memref<0x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %11 = affine.apply #map2()[%10]
      %subview_8 = memref.subview %4[0, 0, 0, 0] [%11, 0, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x0x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
      %cast_9 = memref.cast %subview_8 : memref<?x0x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %
      hivm.hir.nd2nz {dst_continuous} ins(%cast_7 : memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%cast_9 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      %12 = hivm.hir.pointer_cast(%c2048_i64, %c18432_i64) : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %12 {hivm.multi_buffer = 2 : i32} : memref<2x2x16x16xf16, #hivm.address_space<cbuf>>
      %cast_10 = memref.cast %12 : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
      %subview_11 = memref.subview %arg12[0, 0] [%10, 0] [1, 1] : memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x0xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %cast_12 = memref.cast %subview_11 : memref<?x0xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %subview_13 = memref.subview %12[0, 0, 0, 0] [0, %11, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16, #hivm.address_space<cbuf>> to memref<0x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
      %cast_14 = memref.cast %subview_13 : memref<0x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %
      hivm.hir.nd2nz {dst_continuous} ins(%cast_12 : memref<?x?xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%cast_14 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      %13 = hivm.hir.pointer_cast(%c4096_i64, %c20480_i64) : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %13 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
      %cast_15 = memref.cast %13 : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, #hivm.address_space<cbuf>>
      %subview_16 = memref.subview %arg13[0, 0] [%9, 64] [1, 1] : memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %14 = affine.apply #map3()[%8, %5]
      %subview_17 = memref.subview %13[0, 0, 0, 0] [4, %14, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xf16, #hivm.address_space<cbuf>> to memref<4x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>>
      %cast_18 = memref.cast %subview_17 : memref<4x?x16x16xf16, strided<[512, 256, 16, 1]>, #hivm.address_space<cbuf>> to memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
      // CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %
      hivm.hir.nd2nz {dst_continuous} ins(%subview_16 : memref<?x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>)
      // CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]
      %15 = arith.cmpi eq, %arg10, %c0_i32 : i32
      // CHECK: %c{{[0-1]}}_i64{{(_[0-9]+)?}}, %c{{[0-1]}}_i64{{(_[0-9]+)?}} : i64
      hivm.hir.mmadL1 ins(%cast_6, %cast_10, %15, %c0, %10, %c0 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_4 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      // CHECK: %c{{[0-1]}}_i64{{(_[0-9]+)?}}, %c{{[0-1]}}_i64{{(_[0-9]+)?}} : i64
      hivm.hir.mmadL1 ins(%cast_6, %cast_15, %15, %c0, %10, %c64 : memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_5 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
      %16 = affine.apply #map4()[%arg15, %arg14]
      %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [32, 32], strides: [%c1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_20 = memref.cast %reinterpret_cast_19 : memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %17 = affine.apply #map4()[%arg17, %arg16]
      %reinterpret_cast_21 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [32, 32], strides: [%c1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_22 = memref.cast %reinterpret_cast_21 : memref<32x32xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      %18 = affine.apply #map4()[%arg19, %arg18]
      %reinterpret_cast_23 = memref.reinterpret_cast %arg5 to offset: [%18], sizes: [32, 64], strides: [%c1, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<32x64xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>
      %cast_24 = memref.cast %reinterpret_cast_23 : memref<32x64xf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>> to memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>
      scf.yield %cast_20, %cast_22, %cast_24, %16, %c0, %17, %c0, %18, %c0 : memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<32x32xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, memref<32x64xf16, strided<[?, ?], offset: ?>, #hivm.address_space<gm>>, index, index, index, index, index, index
    }
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    return
  }
}
