// RUN: bishengir-opt -pass-pipeline="builtin.module(func.func(hivm-cross-core-gss{force-is-reg-based=true}))" -split-input-file -verify-diagnostics %s | FileCheck %s

// Regression for cv-pipelining when the outer cv-unrolled (K) loop is folded
// away (e.g. its trip count is 1). The producer (VECTOR) and consumer (CUBE)
// multibuffer-unroll loops then become siblings directly under the block loop,
// with no enclosing loop above them. The cross-core WAR (buffer-reuse) handshake
// must keep the multibuffer pair on multiple distinct event ids (here 0 and 1),
// instead of collapsing to a single id, and must size the flag-id repeat by the
// producer/consumer loop trip count (3) rather than the multibuffer depth (4):
// the bug over-permitted with `depth` (4) sets per id, overwriting unread slots
// across the folded outer loop. checkRepeatMultiBufferFlagId must therefore emit
// exactly `loopCount` (3) sets/waits per id and keep both ids.

// CHECK-LABEL: func.func @dot_scale_kernel_2D

// Preamble: WAR buffer-free init. Two distinct ids (0 and 1) prove multi-id is
// kept; each id repeats exactly the loop trip count (3), not the depth (4).
// CHECK:      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
// CHECK-NEXT: scf.for

// Producer (VECTOR) loop: each store waits on its own buffer-free id (1 then 0)
// before reusing the slot, then signals data-ready (RAW) on PIPE_MTE3.
// CHECK:      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
// CHECK-NEXT: hivm.hir.store
// CHECK:      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 3
// CHECK:      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: hivm.hir.store
// CHECK:      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 2

// Consumer (CUBE) loop: after consuming, frees the slot back with matching
// distinct ids (1 then 0).
// CHECK:      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
// CHECK:      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 0

// Epilogue: drain the buffer-free waits, again two ids each repeated 3x.
// CHECK:      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE3>] flag = 1
module {
  func.func @dot_scale_kernel_2D(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: i32, %arg8: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, false, true, true, false, true, false, false, false, false]> : vector<13xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.set_ffts_base_addr %arg0
    %c3 = arith.constant 3 : index
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c0_i8 = arith.constant 0 : i8
    %c127_i16 = arith.constant 127 : i16
    %c7_i32 = arith.constant 7 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %c20_i32 = arith.constant 20 : i32
    %0 = arith.muli %arg10, %arg11 : i32
    %1 = arith.muli %0, %arg12 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    scf.for %arg13 = %3 to %1 step %c20_i32  : i32 {
      hivm.hir.set_mask_norm
      %4 = arith.divsi %arg13, %arg12 : i32
      %5 = arith.remsi %4, %arg11 : i32
      %6 = arith.muli %arg12, %arg11 : i32
      %7 = arith.divsi %arg13, %6 : i32
      %8 = arith.remsi %7, %arg10 : i32
      %9 = arith.muli %8, %c16_i32 : i32
      %10 = arith.muli %5, %c16_i32 : i32
      %11 = tensor.empty() : tensor<16x16xf32>
      %12 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<4x16x32xf16>
      %13 = memref_ext.alloc_workspace() from %arg2 offset = [%c4096] : from memref<?xi8> to memref<4x16x32xf16>
      scf.for %arg14 = %c0 to %c3 step %c1 {
        %27 = arith.index_cast %9 : i32 to index
        %28 = arith.index_cast %arg4 : i32 to index
        %29 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * 32 + s0 * s1)>(%arg14)[%27, %28]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [16, 32], strides: [%28, 1] : memref<?xf16> to memref<16x32xf16, strided<[?, 1], offset: ?>>
        %alloc = memref.alloc() : memref<16x32xf16>
        %30 = affine.max affine_map<()[s0] -> (124, s0)>()[%27]
        %31 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%30, %27]
        %32 = affine.min affine_map<(d0) -> (80, d0 * 32 + 32)>(%arg14)
        %33 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%31, %27]
        %34 = affine.min affine_map<(d0)[s0] -> (d0 * -32 + s0, 32)>(%arg14)[%32]
        %35 = arith.cmpi slt, %33, %c16 : index
        %36 = arith.cmpi slt, %34, %c32 : index
        %37 = arith.ori %35, %36 : i1
        %subview_1 = memref.subview %reinterpret_cast_0[0, 0] [%33, %34] [1, 1] : memref<16x32xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_2 = memref.subview %alloc[0, 0] [%33, %34] [1, 1] : memref<16x32xf16> to memref<?x?xf16, strided<[32, 1]>>
        hivm.hir.load ins(%subview_1 : memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_2 : memref<?x?xf16, strided<[32, 1]>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, vector_producer_to_fuse_0} pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %37 : i1 may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        %38 = bufferization.to_tensor %alloc restrict writable : memref<16x32xf16>
        %39 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0 * 80)>(%arg14)[%27]
        %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%39], sizes: [16, 32], strides: [80, 1] : memref<?xi8> to memref<16x32xi8, strided<[80, 1], offset: ?>>
        %alloc_4 = memref.alloc() : memref<16x32xi8>
        %subview_5 = memref.subview %reinterpret_cast_3[0, 0] [%33, %34] [1, 1] : memref<16x32xi8, strided<[80, 1], offset: ?>> to memref<?x?xi8, strided<[80, 1], offset: ?>>
        %subview_6 = memref.subview %alloc_4[0, 0] [%33, %34] [1, 1] : memref<16x32xi8> to memref<?x?xi8, strided<[32, 1]>>
        hivm.hir.load ins(%subview_5 : memref<?x?xi8, strided<[80, 1], offset: ?>>) outs(%subview_6 : memref<?x?xi8, strided<[32, 1]>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, vector_producer_to_fuse_0} pad_mode = <PadValue> pad_value = %c0_i8 : i8 left_padding_num = %c0 : index init_out_buffer = true init_condition = %37 : i1 may_implicit_transpose_with_last_axis = false core_type = <VECTOR>
        %40 = bufferization.to_tensor %alloc_4 restrict writable : memref<16x32xi8>
        %41 = tensor.empty() : tensor<16x32xi16>
        %42 = tensor.empty() : tensor<16x32xf16>
        %43 = hivm.hir.vcast {vector_producer_to_fuse_0} ins(%40 : tensor<16x32xi8>) outs(%42 : tensor<16x32xf16>) -> tensor<16x32xf16>
        %44 = hivm.hir.vcast {vector_producer_to_fuse_0} ins(%43 : tensor<16x32xf16>) outs(%41 : tensor<16x32xi16>) round_mode = <trunc> -> tensor<16x32xi16>
        %45 = hivm.hir.vadd {vector_producer_to_fuse_0} ins(%44, %c127_i16 : tensor<16x32xi16>, i16) outs(%41 : tensor<16x32xi16>) -> tensor<16x32xi16>
        %46 = tensor.empty() : tensor<16x32xf32>
        %47 = hivm.hir.vcast {vector_producer_to_fuse_0} ins(%45 : tensor<16x32xi16>) outs(%46 : tensor<16x32xf32>) -> tensor<16x32xf32>
        %48 = tensor.empty() : tensor<16x32xi32>
        %49 = hivm.hir.vcast {vector_producer_to_fuse_0} ins(%47 : tensor<16x32xf32>) outs(%48 : tensor<16x32xi32>) round_mode = <trunc> -> tensor<16x32xi32>
        %50 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>, vector_producer_to_fuse_0} ins(%c7_i32 : i32) outs(%48 : tensor<16x32xi32>) -> tensor<16x32xi32>
        %51 = hivm.hir.vmax {vector_producer_to_fuse_0} ins(%50, %c0_i32 : tensor<16x32xi32>, i32) outs(%48 : tensor<16x32xi32>) -> tensor<16x32xi32>
        %52 = tensor.empty() : tensor<16x32xi1>
        %53 = hivm.hir.vcmp {vector_producer_to_fuse_0} ins(%51, %50 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%52 : tensor<16x32xi1>) -> tensor<16x32xi1>
        %54 = hivm.hir.vmax {vector_producer_to_fuse_0} ins(%50, %c32_i32 : tensor<16x32xi32>, i32) outs(%48 : tensor<16x32xi32>) -> tensor<16x32xi32>
        %55 = hivm.hir.vcmp {vector_producer_to_fuse_0} ins(%54, %50 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%52 : tensor<16x32xi1>) -> tensor<16x32xi1>
        %56 = hivm.hir.vnot {vector_producer_to_fuse_0} ins(%55 : tensor<16x32xi1>) outs(%52 : tensor<16x32xi1>) -> tensor<16x32xi1>
        %57 = hivm.hir.vand {vector_producer_to_fuse_0} ins(%53, %56 : tensor<16x32xi1>, tensor<16x32xi1>) outs(%52 : tensor<16x32xi1>) -> tensor<16x32xi1>
        %58 = hivm.hir.vsel {vector_producer_to_fuse_0} ins(%57, %c7_i32, %c0_i32 : tensor<16x32xi1>, i32, i32) outs(%48 : tensor<16x32xi32>) -> tensor<16x32xi32>
        %59 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>, vector_producer_to_fuse_0} ins(%c2_i32 : i32) outs(%48 : tensor<16x32xi32>) -> tensor<16x32xi32>
        %60 = hivm.hir.vpow {vector_producer_to_fuse_0} ins(%59, %58 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%48 : tensor<16x32xi32>) -> tensor<16x32xi32>
        %61 = hivm.hir.vmul {vector_producer_to_fuse_0} ins(%49, %60 : tensor<16x32xi32>, tensor<16x32xi32>) outs(%48 : tensor<16x32xi32>) -> tensor<16x32xi32>
        %62 = hivm.hir.vsel {vector_producer_to_fuse_0} ins(%57, %61, %c0_i32 : tensor<16x32xi1>, tensor<16x32xi32>, i32) outs(%48 : tensor<16x32xi32>) -> tensor<16x32xi32>
        %63 = hivm.hir.vcast {vector_producer_to_fuse_0} ins(%62 : tensor<16x32xi32>) outs(%41 : tensor<16x32xi16>) round_mode = <truncwithoverflow> -> tensor<16x32xi16>
        %64 = hivm.hir.bitcast %63 : tensor<16x32xi16> -> tensor<16x32xbf16>
        %65 = hivm.hir.vcast {vector_producer_to_fuse_0} ins(%64 : tensor<16x32xbf16>) outs(%46 : tensor<16x32xf32>) -> tensor<16x32xf32>
        %66 = hivm.hir.vcast {vector_producer_to_fuse_0} ins(%65 : tensor<16x32xf32>) outs(%42 : tensor<16x32xf16>) -> tensor<16x32xf16>
        %67 = hivm.hir.vmul {vector_producer_to_fuse_0} ins(%38, %66 : tensor<16x32xf16>, tensor<16x32xf16>) outs(%42 : tensor<16x32xf16>) -> tensor<16x32xf16>
        %subview_7 = memref.subview %12[%arg14, 0, 0] [1, 16, 32] [1, 1, 1] : memref<4x16x32xf16> to memref<1x16x32xf16, strided<[512, 32, 1], offset: ?>>
        %collapse_shape = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x16x32xf16, strided<[512, 32, 1], offset: ?>> into memref<16x32xf16, strided<[32, 1], offset: ?>>
        hivm.hir.store ins(%67 : tensor<16x32xf16>) outs(%collapse_shape : memref<16x32xf16, strided<[32, 1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, op_to_tile_0_0}
        %subview_8 = memref.subview %13[%arg14, 0, 0] [1, 16, 32] [1, 1, 1] : memref<4x16x32xf16> to memref<1x16x32xf16, strided<[512, 32, 1], offset: ?>>
        %collapse_shape_9 = memref.collapse_shape %subview_8 [[0, 1], [2]] : memref<1x16x32xf16, strided<[512, 32, 1], offset: ?>> into memref<16x32xf16, strided<[32, 1], offset: ?>>
        hivm.hir.store ins(%67 : tensor<16x32xf16>) outs(%collapse_shape_9 : memref<16x32xf16, strided<[32, 1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, op_to_tile_0_1}
      } {cv_pipeline_idx = 1 : i64, hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 4 : i32}
      %14 = bufferization.to_tensor %13 restrict : memref<4x16x32xf16>
      %15 = bufferization.to_tensor %12 restrict : memref<4x16x32xf16>
      %16 = scf.for %arg14 = %c0 to %c3 step %c1 iter_args(%arg15 = %11) -> (tensor<16x16xf32>) {
        %27 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
        %28 = arith.index_cast %27 : index to i32
        %29 = arith.index_cast %arg7 : i32 to index
        %30 = arith.index_cast %10 : i32 to index
        %31 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 * s1) * 32 + s0)>(%arg14)[%30, %29]
        %reinterpret_cast_0 = memref.reinterpret_cast %arg6 to offset: [%31], sizes: [32, 16], strides: [%29, 1] : memref<?xf16> to memref<32x16xf16, strided<[?, 1], offset: ?>>
        %32 = affine.min affine_map<(d0) -> (80, d0 * 32 + 32)>(%arg14)
        %33 = affine.min affine_map<(d0)[s0] -> (d0 * -32 + s0, 32)>(%arg14)[%32]
        %34 = arith.cmpi slt, %33, %c32 : index
        %alloc = memref.alloc() : memref<32x16xf16>
        %35 = affine.max affine_map<()[s0] -> (248, s0)>()[%30]
        %36 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%35, %30]
        %37 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%36, %30]
        %38 = arith.cmpi slt, %37, %c16 : index
        %39 = arith.ori %34, %38 : i1
        %subview_1 = memref.subview %reinterpret_cast_0[0, 0] [%33, %37] [1, 1] : memref<32x16xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        %subview_2 = memref.subview %alloc[0, 0] [%33, %37] [1, 1] : memref<32x16xf16> to memref<?x?xf16, strided<[16, 1]>>
        hivm.hir.load ins(%subview_1 : memref<?x?xf16, strided<[?, 1], offset: ?>>) outs(%subview_2 : memref<?x?xf16, strided<[16, 1]>>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} pad_mode = <PadValue> pad_value = %cst : f16 left_padding_num = %c0 : index init_out_buffer = true init_condition = %39 : i1 may_implicit_transpose_with_last_axis = false core_type = <CUBE>
        %40 = bufferization.to_tensor %alloc restrict writable : memref<32x16xf16>
        %41 = tensor.empty() : tensor<16x32xf16>
        %extracted_slice_3 = tensor.extract_slice %15[%arg14, 0, 0] [1, 16, 32] [1, 1, 1] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : tensor<4x16x32xf16> to tensor<16x32xf16>
        %42 = hivm.hir.load ins(%extracted_slice_3 : tensor<16x32xf16>) outs(%41 : tensor<16x32xf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>, "inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <CUBE> -> tensor<16x32xf16>
        %extracted_slice_4 = tensor.extract_slice %14[%arg14, 0, 0] [1, 16, 32] [1, 1, 1] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : tensor<4x16x32xf16> to tensor<16x32xf16>
        %43 = hivm.hir.load ins(%extracted_slice_4 : tensor<16x32xf16>) outs(%41 : tensor<16x32xf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>, "inserted-load"} init_out_buffer = false may_implicit_transpose_with_last_axis = false core_type = <CUBE> -> tensor<16x32xf16>
        %44 = arith.cmpi eq, %28, %c0_i32 : i32
        %45 = hivm.hir.mmadL1 ins(%42, %40, %44, %c16, %c32, %37 : tensor<16x32xf16>, tensor<32x16xf16>, i1, index, index, index) outs(%arg15 : tensor<16x16xf32>) -> tensor<16x16xf32>
        %46 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%43, %40, %false, %c16, %c32, %37 : tensor<16x32xf16>, tensor<32x16xf16>, i1, index, index, index) outs(%45 : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.yield %46 : tensor<16x16xf32>
      } {cv_pipeline_idx = 1 : i64, hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 4 : i32}
      %17 = arith.index_cast %9 : i32 to index
      %18 = arith.index_cast %arg9 : i32 to index
      %19 = arith.index_cast %10 : i32 to index
      %20 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * s2)>()[%19, %17, %18]
      %reinterpret_cast = memref.reinterpret_cast %arg8 to offset: [%20], sizes: [16, 16], strides: [%18, 1] : memref<?xf16> to memref<16x16xf16, strided<[?, 1], offset: ?>>
      %21 = affine.max affine_map<()[s0] -> (124, s0)>()[%17]
      %22 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%21, %17]
      %23 = affine.max affine_map<()[s0] -> (248, s0)>()[%19]
      %24 = affine.min affine_map<()[s0, s1] -> (s1 + 16, s0)>()[%23, %19]
      %25 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%22, %17]
      %26 = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%24, %19]
      %extracted_slice = tensor.extract_slice %16[0, 0] [%25, %26] [1, 1] {hivm.tcore_type = #hivm.tcore_type<CUBE>} : tensor<16x16xf32> to tensor<?x?xf32>
      %subview = memref.subview %reinterpret_cast[0, 0] [%25, %26] [1, 1] : memref<16x16xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%extracted_slice : tensor<?x?xf32>) outs(%subview : memref<?x?xf16, strided<[?, 1], offset: ?>>)
    } {cv_unrolled_loop, autoblockify.subloop}
    return
  }
}
