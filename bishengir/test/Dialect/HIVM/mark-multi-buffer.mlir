// RUN: bishengir-opt -allow-unregistered-dialect %s             \
// RUN:   -pass-pipeline="builtin.module(                        \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true}),cse)" \
// RUN:   -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt -allow-unregistered-dialect %s             \
// RUN:   -pass-pipeline="builtin.module(                        \
// RUN:     func.func(hivm-mark-multi-buffer{enable-auto=true limit-auto-multi-buffer-only-for-local-buffer=true}),cse)" \
// RUN:   -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=LIMIT-LOCAL

// -----
// CHECK-LABEL: func.func @test_mark_multi_buffer(
func.func @test_mark_multi_buffer(%d : index, %in : memref<8xf32, #hivm.address_space<gm>>, %out : memref<8xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %tmp4 = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
  // CHECK: memref.alloca
  scf.for %i0 = %c0 to %c16 step %c4 {
    %tmp1 = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[alloc:.*]] = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK-NOT: annotation.mark
    %tmp2 = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: %[[alloca_0:.*]] = memref.alloca() : memref<8xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %alloca_0 {hivm.multi_buffer = 2 : i32} : memref<8xf32, #hivm.address_space<ub>>
    %tmp3 = memref.alloca(%d) : memref<?xf32, #hivm.address_space<ub>>
    // CHECK: %[[alloca_1:.*]] = memref.alloca(%arg0) : memref<?xf32, #hivm.address_space<ub>>
    // CHECK-NOT: annotation.mark
    %tmp_l0c = memref.alloca() : memref<8xf32, #hivm.address_space<cc>>
    // CHECK-NOT: annotation.mark
    "some_use"(%tmp1) : (memref<8xf32, #hivm.address_space<ub>>) -> ()
    hivm.hir.load ins(%in : memref<8xf32, #hivm.address_space<gm>>) outs(%tmp2 : memref<8xf32, #hivm.address_space<ub>>)
    "some_use"(%tmp3) : (memref<?xf32, #hivm.address_space<ub>>) -> ()
    "some_use"(%tmp_l0c) : (memref<8xf32, #hivm.address_space<cc>>) -> ()
    hivm.hir.store ins(%tmp4 : memref<8xf32, #hivm.address_space<ub>>) outs(%out : memref<8xf32, #hivm.address_space<gm>>)
  }
  return
}

// -----
module {
  // CHECK-LABEL: func.func @test_fa_bwd_cube_return_to_tensor_distance_two
  func.func @test_fa_bwd_cube_return_to_tensor_distance_two(
      %q_gm: memref<64x128xf16, #hivm.address_space<gm>>,
      %do_gm: memref<64x128xf16, #hivm.address_space<gm>>)
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %p = tensor.empty() : tensor<4x4x16x16xf16>
    %acc = tensor.empty() : tensor<8x4x16x16xf32>

    scf.for %i = %c0_i32 to %c16_i32 step %c1_i32 : i32 {
      %p2:2 = scope.scope : () -> (tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>) {
        // FA bwd preload_num=2 CUBE produces Q and dO as to_tensor(alloc).
        // CHECK: %[[Q_ALLOC:.*]] = memref.alloc() : memref<8x4x16x16xf16>
        // CHECK-NEXT: annotation.mark %[[Q_ALLOC]] {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32}
        %q_alloc = memref.alloc() : memref<8x4x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%q_gm : memref<64x128xf16, #hivm.address_space<gm>>) outs(%q_alloc : memref<8x4x16x16xf16>)
        %q = bufferization.to_tensor %q_alloc restrict writable : memref<8x4x16x16xf16>

        // CHECK: %[[DO_ALLOC:.*]] = memref.alloc() : memref<8x4x16x16xf16>
        // CHECK-NEXT: annotation.mark %[[DO_ALLOC]] {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32}
        %do_alloc = memref.alloc() : memref<8x4x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%do_gm : memref<64x128xf16, #hivm.address_space<gm>>) outs(%do_alloc : memref<8x4x16x16xf16>)
        %do = bufferization.to_tensor %do_alloc restrict writable : memref<8x4x16x16xf16>

        scope.return %q, %do : tensor<8x4x16x16xf16>, tensor<8x4x16x16xf16>
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 2 : i32, no_inline}

      scope.scope : () -> () {
        // preload_num=0 consumes them two stages later, so distance + 1 = 3.
        %dv = hivm.hir.mmadL1 {a_transpose, hivm.remain_in_l0c} ins(%p, %p2#1, %true, %c64, %c64, %c128 : tensor<4x4x16x16xf16>, tensor<8x4x16x16xf16>, i1, index, index, index) outs(%acc : tensor<8x4x16x16xf32>) -> tensor<8x4x16x16xf32>
        %dk = hivm.hir.mmadL1 {hivm.remain_in_l0c} ins(%p, %p2#0, %true, %c64, %c64, %c128 : tensor<4x4x16x16xf16>, tensor<8x4x16x16xf16>, i1, index, index, index) outs(%acc : tensor<8x4x16x16xf32>) -> tensor<8x4x16x16xf32>
        "test.use"(%dv, %dk) : (tensor<8x4x16x16xf32>, tensor<8x4x16x16xf32>) -> ()
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 3 : i32, hivm.preload_num = 0 : i32, no_inline}
    }

    return
  }
}

// -----
module {
  func.func @test_for_yield(
      %arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    %29 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
    scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
      %39 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
      // %39 is yielded by the for but is also loaded into inside it, so the
      // for is its rotation anchor and it gets marked. (Before the
      // isConsumedInLoop guard getParentLoop climbed past the loop result to
      // null and the buffer was silently left un-multi-buffered.)
      // CHECK: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
      hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

      scf.yield %39 : memref<1x2048xf16, #hivm.address_space<ub>>
    }

    return
  }
}

// -----
module {
  func.func @test_2for_yield(
      %arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32  : i32 {
      %29 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>

      %31 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        %39 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
        // CHECK: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
        hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

        scf.yield %39 : memref<1x2048xf16, #hivm.address_space<ub>>
      }
    }

    return
  }
}

// -----
module {
  // LIMIT-LOCAL-LABEL: func.func @test_mark_workspace(
  func.func @test_mark_workspace(
      %arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
      %arg2: memref<64x16xf32>)
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %1 = tensor.empty() : tensor<16x16xf32>
    %2 = tensor.empty() : tensor<16x16xf32>
    scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 : i32 {
      %3 = tensor.empty() : tensor<16x16xf32>
      %4 = hivm.hir.mmadL1 ins(%1, %2, %true, %c16, %c16, %c16 : tensor<16x16xf32>, tensor<16x16xf32>, i1, index, index, index) outs(%3 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %5 = memref_ext.alloc_workspace() from %arg1 : from memref<?xi8> to memref<16x16xf32>
      // CHECK: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
      // LIMIT-LOCAL: %[[WS:.*]] = memref_ext.alloc_workspace
      // LIMIT-LOCAL-NOT: annotation.mark %[[WS]]
      // LIMIT-LOCAL: bufferization.to_tensor %[[WS]]
      %6 = bufferization.to_tensor %5 restrict writable : memref<16x16xf32>
      %7 = hivm.hir.fixpipe {enable_nz2nd} ins(%4 : tensor<16x16xf32>) outs(%6 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %8 = tensor.empty() : tensor<16x16xf32>
      %9 = hivm.hir.load ins(%7 : tensor<16x16xf32>) outs(%8 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %10 = tensor.empty() : tensor<16x16xf32>
      %11 = tensor.empty() : tensor<16x16xf32>
      %12 = hivm.hir.vadd ins(%9, %10 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %13 = arith.index_cast %arg3 : i32 to index
      %14 = arith.muli %13, %c16 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [16, 16], strides: [16, 1] : memref<64x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
      hivm.hir.store ins(%12 : tensor<16x16xf32>) outs(%reinterpret_cast : memref<16x16xf32, strided<[16, 1], offset: ?>>)
    }
    return
  }
}

// -----
module {
  func.func @test_3for_yield(
      %arg0: memref<1x2048xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32})
      attributes {global_kernel = "local", hacc.entry = "", hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c49152_i32 = arith.constant 49152 : i32

    %29 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
    scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg9 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>) : i32 {

      %31:2 = scf.for %arg8 = %c0_i32 to %c49152_i32 step %c2048_i32 iter_args(%arg10 = %29, %arg11 = %29) -> (memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>)  : i32 {
        %39 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
        // %39 is yielded up the loop nest but is loaded into in the innermost
        // for, so it anchors there and is marked. %40 is only threaded out via
        // the yield (never consumed), so it is left unmarked.
        // CHECK: annotation.mark %{{.*}} {hivm.multi_buffer = 2 : i32}
        // CHECK-NOT: annotation.mark
        %40 = memref.alloc() : memref<1x2048xf16, #hivm.address_space<ub>>
        hivm.hir.load ins(%arg0 : memref<1x2048xf16, #hivm.address_space<gm>>) outs(%39 : memref<1x2048xf16, #hivm.address_space<ub>>)

        scf.yield %40, %39 : memref<1x2048xf16, #hivm.address_space<ub>>, memref<1x2048xf16, #hivm.address_space<ub>>
      }

      scf.yield %31#1 : memref<1x2048xf16, #hivm.address_space<ub>>
    }

    return
  }
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_for_scope_markmultibuffer_for_preload(%arg0: i32, %arg1: tensor<128xf32>, %arg2: tensor<128x128xf32>, %arg3 : tensor<8x8x16x16xf32>) -> tensor<128x128xf32> {
    %c128_i32 = arith.constant 128 : i32
    %alloc = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = scope.scope : () -> i32 {
      // CHECK: annotation.mark
      // CHECK-SAME: hivm.multi_buffer = 2 : i32
      // CHECK-SAME: hivm.preload_local_buffer = 1 : i32
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<128x128xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%arg3 : tensor<8x8x16x16xf32>) outs(%alloc : memref<128x128xf32, #hivm.address_space<ub>>)
      %2 = arith.addi %arg0, %c128_i32 : i32
      scope.return %2 : i32
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
    %memspacecast = memref.memory_space_cast %alloc : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
    %3 = bufferization.to_tensor %memspacecast restrict writable : memref<128x128xf32>
    %4 = scope.scope : () -> tensor<128x128xf32> {
      %expanded = tensor.expand_shape %arg1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %5 = hivm.hir.vmul ins(%arg2, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%0 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
      %6 = hivm.hir.vadd ins(%3, %5 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      scope.return %6 : tensor<128x128xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}

    return %4 : tensor<128x128xf32>
  }
}
