// RUN: bishengir-opt %s --hivm-bind-sub-block 2>&1 | FileCheck %s

// Regression test for the ExtractSlice bubble-up rank mismatch on multi-buffered
// CV-pipelined block loops (commit "[HIVM] Fix ExtractSlice bubble-up rank
// mismatch for multi-buffered CV-pipelined block loops").
//
// This is real post-CVPipelining IR (the AIV/vector function of a mix-cv triton
// kernel) compiled with --enable-auto-blockify-loop + --set-workspace-multibuffer=2.
// CVPipelining double-buffers the AutoBlockify grid-stride block loop, so the
// on-chip UB buffers gain a leading multi-buffer (slot) dimension
// (memref<2x32x128xf32, #ub>, annotated hivm.cv_pipelined_multi_buffer). The
// to-be-bubbled tensor.extract_slice carries that leading slot dim
// (e.g. [%slot, 0, 0] [1, 32, 128]) and is bubbled, inside hivm-bind-sub-block,
// onto the rank-2 GM memref.reinterpret_cast (memref<32x128xf32>).
//
// Before the fix, BufferizationBubbleUpStrategy built a rank-3 memref.subview
// directly on the rank-2 GM cast, tripping
//   SubViewOp::inferResultType: assert(staticOffsets.size() == rank)
// and aborting the compiler. The fix projects the slice onto the cast's rank on
// the GM source side and rank-reduces the leading size-1 slot dim on the UB
// destination side, so the bubbled hivm.hir.load has matching src/dst ranks.
//
// The test passes simply by NOT crashing, and by the bubbled multi-buffered load
// having rank-reduced (rank-2) src and dst operands.

// CHECK-LABEL: func.func @chunk_transform_qk_fwd_kernel_mix_aiv
// CHECK: memref.alloc() : memref<2x32x128xf32, #hivm.address_space<ub>>
// CHECK: annotation.mark {{.*}} {hivm.cv_pipelined_multi_buffer} : memref<2x32x128xf32, #hivm.address_space<ub>>
// The bubbled GM->UB load: both operands rank-2 (slot dim dropped on GM,
// rank-reduced on UB), so src and dst dimensions match.
// CHECK: hivm.hir.load ins(%{{.*}} : memref<?x128xf32, strided<[128, 1], offset: ?>>) outs(%{{.*}} : memref<?x128xf32, strided<[128, 1], offset: ?>>)

#map = affine_map<(d0, d1) -> ((d0 - d1) ceildiv 28)>
#map1 = affine_map<(d0)[s0] -> (d0 * 28 + s0)>
#map2 = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map3 = affine_map<()[s0, s1] -> (s0 - s1)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1)>
#map5 = affine_map<()[s0, s1] -> ((s0 - s1) ceildiv 16)>
#map6 = affine_map<()[s0] -> (s0 floordiv 16)>
#map7 = affine_map<()[s0] -> (s0 mod 16)>
#map8 = affine_map<()[s0, s1] -> (s0 + s1 * 32)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @chunk_transform_qk_fwd_kernel_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: f32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false]> : vector<16xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c10 = arith.constant 10 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c56_i32 = arith.constant 56 : i32
    hivm.hir.anchor {id = 0 : i64}
    %0 = arith.muli %arg13, %arg14 : i32
    %1 = arith.muli %0, %arg15 : i32
    hivm.hir.anchor {id = 1 : i64}
    annotation.mark %1 {logical_block_num} : i32
    hivm.hir.anchor {id = 2 : i64}
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
    hivm.hir.anchor {id = 3 : i64}
    scf.for %arg16 = %3 to %1 step %c56_i32  : i32 {
      hivm.hir.anchor_block {id_end = 171 : i64, id_start = 4 : i64}
      hivm.hir.anchor {id = 5 : i64}
      %alloc = memref.alloc() : memref<2x32x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<2x32x128xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 6 : i64}
      annotation.mark %alloc {hivm.cv_pipelined_multi_buffer} : memref<2x32x128xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 7 : i64}
      %alloc_0 = memref.alloc() : memref<2x32x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<2x32x128xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 8 : i64}
      annotation.mark %alloc_0 {hivm.cv_pipelined_multi_buffer} : memref<2x32x128xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 9 : i64}
      %alloc_1 = memref.alloc() : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 10 : i64}
      annotation.mark %alloc_1 {hivm.cv_pipelined_multi_buffer} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 11 : i64}
      %alloc_2 = memref.alloc() : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 12 : i64}
      annotation.mark %alloc_2 {hivm.cv_pipelined_multi_buffer} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 13 : i64}
      %alloc_3 = memref.alloc() : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 14 : i64}
      annotation.mark %alloc_3 {hivm.cv_pipelined_multi_buffer} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 15 : i64}
      %alloc_4 = memref.alloc() : memref<2x32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>} : memref<2x32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 16 : i64}
      annotation.mark %alloc_4 {hivm.cv_pipelined_multi_buffer} : memref<2x32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 17 : i64}
      %alloc_5 = memref.alloc() : memref<2x32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>} : memref<2x32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 18 : i64}
      annotation.mark %alloc_5 {hivm.cv_pipelined_multi_buffer} : memref<2x32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 19 : i64}
      %alloc_6 = memref.alloc() : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 20 : i64}
      annotation.mark %alloc_6 {hivm.cv_pipelined_multi_buffer} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 21 : i64}
      %alloc_7 = memref.alloc() : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 22 : i64}
      annotation.mark %alloc_7 {hivm.cv_pipelined_multi_buffer} : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 23 : i64}
      %alloc_8 = memref.alloc() : memref<2x32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<9>} : memref<2x32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 24 : i64}
      annotation.mark %alloc_8 {hivm.cv_pipelined_multi_buffer} : memref<2x32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 25 : i64}
      %alloc_9 = memref.alloc() : memref<2x32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>} : memref<2x32x32xf32, #hivm.address_space<ub>>
      hivm.hir.anchor {id = 26 : i64}
      annotation.mark %alloc_9 {hivm.cv_pipelined_multi_buffer} : memref<2x32x32xf32, #hivm.address_space<ub>>
      %4 = tensor.empty() : tensor<2x32x128xf32>
      hivm.hir.anchor {id = 27 : i64}
      %alloc_10 = memref.alloc() : memref<2x32x128xf32>
      hivm.hir.anchor {id = 28 : i64}
      annotation.mark %alloc_10 {hivm.cv_pipelined_multi_buffer} : memref<2x32x128xf32>
      %5 = tensor.empty() : tensor<2x32x32xi1>
      hivm.hir.anchor {id = 29 : i64}
      %alloc_11 = memref.alloc() : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<11>} : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 30 : i64}
      annotation.mark %alloc_11 {hivm.cv_pipelined_multi_buffer} : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 31 : i64}
      %alloc_12 = memref.alloc() : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>} : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 32 : i64}
      annotation.mark %alloc_12 {hivm.cv_pipelined_multi_buffer} : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 33 : i64}
      %alloc_13 = memref.alloc() : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<13>} : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 34 : i64}
      annotation.mark %alloc_13 {hivm.cv_pipelined_multi_buffer} : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>>
      %6 = arith.index_cast %1 : i32 to index
      %7 = arith.index_cast %arg16 : i32 to index
      %8 = affine.apply #map(%6, %7)
      %9 = arith.minui %8, %c2 : index
      hivm.hir.anchor {id = 35 : i64}
      %10:3 = scf.for %arg17 = %c0 to %9 step %c1 iter_args(%arg18 = %4, %arg19 = %5, %arg20 = %5) -> (tensor<2x32x128xf32>, tensor<2x32x32xi1>, tensor<2x32x32xi1>) {
        hivm.hir.anchor_block {id_end = 65 : i64, id_start = 36 : i64}
        hivm.hir.anchor {id = 37 : i64}
        %18 = arith.addi %arg17, %c6 : index
        %19 = arith.index_cast %18 : index to i64
        %20 = arith.addi %arg17, %c8 : index
        %21 = arith.index_cast %20 : index to i64
        %22 = arith.addi %arg17, %c10 : index
        %23 = arith.index_cast %22 : index to i64
        %subview = memref.subview %alloc_13[%arg17, 0, 0, 0, 0] [1, 16, 2, 16, 8] [1, 1, 1, 1, 1] : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>> to memref<16x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        %subview_19 = memref.subview %alloc_12[%arg17, 0, 0, 0, 0] [1, 16, 2, 16, 8] [1, 1, 1, 1, 1] : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>> to memref<16x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        %subview_20 = memref.subview %alloc_11[%arg17, 0, 0, 0, 0] [1, 16, 2, 16, 8] [1, 1, 1, 1, 1] : memref<2x16x2x16x8xf32, #hivm.address_space<cbuf>> to memref<16x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        %subview_21 = memref.subview %alloc_10[%arg17, 0, 0] [1, 32, 128] [1, 1, 1] : memref<2x32x128xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>>
        %24 = affine.apply #map1(%arg17)[%7]
        %25 = arith.index_cast %24 : index to i32
        %26 = arith.remsi %25, %arg13 : i32
        %27 = arith.divsi %25, %arg13 : i32
        %28 = arith.remsi %27, %arg14 : i32
        %29 = tensor.empty() : tensor<32x128xf32>
        %30 = arith.muli %28, %arg12 : i32
        %31 = arith.muli %30, %c128_i32 : i32
        %32 = arith.index_cast %31 : i32 to index
        %33 = arith.muli %26, %c32_i32 : i32
        %34 = arith.maxsi %33, %c0_i32 : i32
        %35 = arith.index_cast %34 : i32 to index
        %36 = affine.apply #map2()[%32, %35]
        %37 = arith.index_cast %arg12 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [32, 128], strides: [128, 1] : memref<?xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>>
        %reinterpret_cast_22 = memref.reinterpret_cast %arg4 to offset: [%36], sizes: [32, 128], strides: [128, 1] : memref<?xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>>
        hivm.hir.anchor {id = 38 : i64}
        %alloc_23 = memref.alloc() : memref<32x128xf32>
        %38 = affine.apply #map3()[%37, %35]
        %39 = arith.maxsi %38, %c0 : index
        %40 = arith.minsi %39, %c32 : index
        %41 = arith.subi %c0_i32, %33 : i32
        %42 = arith.maxsi %41, %c0_i32 : i32
        %43 = arith.index_cast %42 : i32 to index
        %44 = arith.minsi %43, %40 : index
        %45 = affine.apply #map3()[%40, %44]
        %46 = arith.cmpi slt, %45, %c32 : index
        %subview_24 = memref.subview %reinterpret_cast[0, 0] [%45, 128] [1, 1] : memref<32x128xf32, strided<[128, 1], offset: ?>> to memref<?x128xf32, strided<[128, 1], offset: ?>>
        %subview_25 = memref.subview %alloc_23[%44, 0] [%45, 128] [1, 1] : memref<32x128xf32> to memref<?x128xf32, strided<[128, 1], offset: ?>>
        hivm.hir.anchor {id = 39 : i64}
        hivm.hir.load ins(%subview_24 : memref<?x128xf32, strided<[128, 1], offset: ?>>) outs(%subview_25 : memref<?x128xf32, strided<[128, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %46 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
        hivm.hir.anchor {id = 40 : i64}
        %47 = bufferization.to_tensor %alloc_23 restrict writable : memref<32x128xf32>
        hivm.hir.anchor {id = 41 : i64}
        %extracted_slice = tensor.extract_slice %arg18[%arg17, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<2x32x128xf32> to tensor<32x128xf32>
        hivm.hir.anchor {id = 42 : i64}
        %48 = hivm.hir.vmul ins(%47, %arg11 : tensor<32x128xf32>, f32) outs(%extracted_slice : tensor<32x128xf32>) -> tensor<32x128xf32>
        hivm.hir.anchor {id = 43 : i64}
        %inserted_slice = tensor.insert_slice %48 into %arg18[%arg17, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<32x128xf32> into tensor<2x32x128xf32>
        %subview_26 = memref.subview %reinterpret_cast_22[0, 0] [%45, 128] [1, 1] : memref<32x128xf32, strided<[128, 1], offset: ?>> to memref<?x128xf32, strided<[128, 1], offset: ?>>
        %subview_27 = memref.subview %alloc_10[%arg17, %44, 0] [1, %45, 128] [1, 1, 1] : memref<2x32x128xf32> to memref<?x128xf32, strided<[128, 1], offset: ?>>
        hivm.hir.anchor {id = 44 : i64}
        hivm.hir.load ins(%subview_26 : memref<?x128xf32, strided<[128, 1], offset: ?>>) outs(%subview_27 : memref<?x128xf32, strided<[128, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true init_condition = %46 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
        hivm.hir.anchor {id = 45 : i64}
        %49 = bufferization.to_tensor %subview_21 restrict writable : memref<32x128xf32, strided<[128, 1], offset: ?>>
        %50 = tensor.empty() : tensor<32xi32>
        hivm.hir.anchor {id = 46 : i64}
        %51 = hivm.hir.varange offset[%c0] strides[%c1] outs(%50 : tensor<32xi32>) -> tensor<32xi32>
        %52 = tensor.empty() : tensor<32x32xi32>
        %expanded = tensor.expand_shape %51 [[0, 1]] output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
        hivm.hir.anchor {id = 47 : i64}
        %53 = hivm.hir.vbrc ins(%expanded : tensor<32x1xi32>) outs(%52 : tensor<32x32xi32>) broadcast_dims = [1] -> tensor<32x32xi32>
        %expanded_28 = tensor.expand_shape %51 [[0, 1]] output_shape [1, 32] : tensor<32xi32> into tensor<1x32xi32>
        hivm.hir.anchor {id = 48 : i64}
        %54 = hivm.hir.vbrc ins(%expanded_28 : tensor<1x32xi32>) outs(%52 : tensor<32x32xi32>) broadcast_dims = [0] -> tensor<32x32xi32>
        hivm.hir.anchor {id = 49 : i64}
        %extracted_slice_29 = tensor.extract_slice %arg19[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xi1> to tensor<32x32xi1>
        hivm.hir.anchor {id = 50 : i64}
        %55 = hivm.hir.vcmp ins(%53, %54 : tensor<32x32xi32>, tensor<32x32xi32>) outs(%extracted_slice_29 : tensor<32x32xi1>) compare_mode = <ge> -> tensor<32x32xi1>
        hivm.hir.anchor {id = 51 : i64}
        %inserted_slice_30 = tensor.insert_slice %55 into %arg19[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<32x32xi1> into tensor<2x32x32xi1>
        %expanded_31 = tensor.expand_shape %48 [[0], [1, 2]] output_shape [32, 16, 8] : tensor<32x128xf32> into tensor<32x16x8xf32>
        %56 = tensor.empty() : tensor<16x32x8xf32>
        hivm.hir.anchor {id = 52 : i64}
        %57 = hivm.hir.vtranspose ins(%expanded_31 : tensor<32x16x8xf32>) outs(%56 : tensor<16x32x8xf32>) permutation = [1, 0, 2] -> tensor<16x32x8xf32>
        %expanded_32 = tensor.expand_shape %57 [[0], [1, 2], [3]] output_shape [16, 2, 16, 8] : tensor<16x32x8xf32> into tensor<16x2x16x8xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %19
        hivm.hir.anchor {id = 53 : i64}
        hivm.hir.copy ins(%expanded_32 : tensor<16x2x16x8xf32>) outs(%subview_20 : memref<16x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>) {"inserted-copy"}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 13
        hivm.hir.anchor {id = 54 : i64}
        %extracted_slice_33 = tensor.extract_slice %arg20[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xi1> to tensor<32x32xi1>
        hivm.hir.anchor {id = 55 : i64}
        %58 = hivm.hir.vcmp ins(%53, %54 : tensor<32x32xi32>, tensor<32x32xi32>) outs(%extracted_slice_33 : tensor<32x32xi1>) compare_mode = <gt> -> tensor<32x32xi1>
        hivm.hir.anchor {id = 56 : i64}
        %inserted_slice_34 = tensor.insert_slice %58 into %arg20[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<32x32xi1> into tensor<2x32x32xi1>
        %59 = arith.index_cast %30 : i32 to index
        %60 = affine.apply #map4()[%35, %59]
        %reinterpret_cast_35 = memref.reinterpret_cast %arg6 to offset: [%60], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
        hivm.hir.anchor {id = 57 : i64}
        %alloc_36 = memref.alloc() : memref<32xf32>
        %subview_37 = memref.subview %reinterpret_cast_35[0] [%45] [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
        %subview_38 = memref.subview %alloc_36[%44] [%45] [1] : memref<32xf32> to memref<?xf32, strided<[1], offset: ?>>
        %61 = arith.remui %44, %c8 : index
        hivm.hir.anchor {id = 58 : i64}
        hivm.hir.load ins(%subview_37 : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_38 : memref<?xf32, strided<[1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %61 : index init_out_buffer = true init_condition = %46 : i1 eviction_policy = <EvictFirst> core_type = <VECTOR>
        hivm.hir.anchor {id = 59 : i64}
        %62 = bufferization.to_tensor %alloc_36 restrict writable : memref<32xf32>
        %expanded_39 = tensor.expand_shape %62 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        hivm.hir.anchor {id = 60 : i64}
        %63 = hivm.hir.vmul ins(%49, %expanded_39 : tensor<32x128xf32>, tensor<32x1xf32>) outs(%29 : tensor<32x128xf32>) broadcast = [1] -> tensor<32x128xf32>
        %expanded_40 = tensor.expand_shape %63 [[0], [1, 2]] output_shape [32, 16, 8] : tensor<32x128xf32> into tensor<32x16x8xf32>
        hivm.hir.anchor {id = 61 : i64}
        %64 = hivm.hir.vtranspose ins(%expanded_40 : tensor<32x16x8xf32>) outs(%56 : tensor<16x32x8xf32>) permutation = [1, 0, 2] -> tensor<16x32x8xf32>
        %expanded_41 = tensor.expand_shape %64 [[0], [1, 2], [3]] output_shape [16, 2, 16, 8] : tensor<16x32x8xf32> into tensor<16x2x16x8xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %23
        hivm.hir.anchor {id = 62 : i64}
        hivm.hir.copy ins(%expanded_41 : tensor<16x2x16x8xf32>) outs(%subview_19 : memref<16x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>) {"inserted-copy"}
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %21
        hivm.hir.anchor {id = 63 : i64}
        hivm.hir.copy ins(%expanded_41 : tensor<16x2x16x8xf32>) outs(%subview : memref<16x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>) {"inserted-copy"}
        hivm.hir.anchor {id = 64 : i64}
        scf.yield %inserted_slice, %inserted_slice_30, %inserted_slice_34 : tensor<2x32x128xf32>, tensor<2x32x32xi1>, tensor<2x32x32xi1>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      hivm.hir.anchor {id = 66 : i64}
      hivm.hir.anchor {id = 67 : i64}
      hivm.hir.anchor {id = 68 : i64}
      hivm.hir.anchor {id = 69 : i64}
      %11 = bufferization.to_tensor %alloc_10 restrict writable : memref<2x32x128xf32>
      hivm.hir.anchor {id = 70 : i64}
      scf.for %arg17 = %c0 to %9 step %c1 {
        hivm.hir.anchor_block {id_end = 82 : i64, id_start = 71 : i64}
        hivm.hir.anchor {id = 72 : i64}
        hivm.hir.anchor {id = 73 : i64}
        hivm.hir.anchor {id = 74 : i64}
        hivm.hir.anchor {id = 75 : i64}
        hivm.hir.anchor {id = 76 : i64}
        hivm.hir.anchor {id = 77 : i64}
        hivm.hir.anchor {id = 78 : i64}
        hivm.hir.anchor {id = 79 : i64}
        hivm.hir.anchor {id = 80 : i64}
        hivm.hir.anchor {id = 81 : i64}
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      %memspacecast = memref.memory_space_cast %alloc_9 : memref<2x32x32xf32, #hivm.address_space<ub>> to memref<2x32x32xf32>
      hivm.hir.anchor {id = 83 : i64}
      %12 = bufferization.to_tensor %memspacecast restrict writable : memref<2x32x32xf32>
      %memspacecast_14 = memref.memory_space_cast %alloc_8 : memref<2x32x32xf32, #hivm.address_space<ub>> to memref<2x32x32xf32>
      hivm.hir.anchor {id = 84 : i64}
      %13 = bufferization.to_tensor %memspacecast_14 restrict writable : memref<2x32x32xf32>
      hivm.hir.anchor {id = 85 : i64}
      scf.for %arg17 = %c0 to %9 step %c1 {
        hivm.hir.anchor_block {id_end = 99 : i64, id_start = 86 : i64}
        hivm.hir.anchor {id = 87 : i64}
        %subview = memref.subview %alloc_7[%arg17, 0, 0, 0, 0] [1, 4, 2, 16, 8] [1, 1, 1, 1, 1] : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        %subview_19 = memref.subview %alloc_6[%arg17, 0, 0, 0, 0] [1, 4, 2, 16, 8] [1, 1, 1, 1, 1] : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        %18 = tensor.empty() : tensor<32x32xf32>
        hivm.hir.anchor {id = 88 : i64}
        %extracted_slice = tensor.extract_slice %10#1[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xi1> to tensor<32x32xi1>
        hivm.hir.anchor {id = 89 : i64}
        %extracted_slice_20 = tensor.extract_slice %13[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xf32> to tensor<32x32xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 12
        hivm.hir.anchor {id = 90 : i64}
        %19 = hivm.hir.vsel ins(%extracted_slice, %extracted_slice_20, %cst : tensor<32x32xi1>, tensor<32x32xf32>, f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.anchor {id = 91 : i64}
        %extracted_slice_21 = tensor.extract_slice %10#2[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xi1> to tensor<32x32xi1>
        hivm.hir.anchor {id = 92 : i64}
        %extracted_slice_22 = tensor.extract_slice %12[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xf32> to tensor<32x32xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 12
        hivm.hir.anchor {id = 93 : i64}
        %20 = hivm.hir.vsel ins(%extracted_slice_21, %extracted_slice_22, %cst : tensor<32x32xi1>, tensor<32x32xf32>, f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
        %expanded = tensor.expand_shape %19 [[0], [1, 2]] output_shape [32, 4, 8] : tensor<32x32xf32> into tensor<32x4x8xf32>
        %21 = tensor.empty() : tensor<4x32x8xf32>
        hivm.hir.anchor {id = 94 : i64}
        %22 = hivm.hir.vtranspose ins(%expanded : tensor<32x4x8xf32>) outs(%21 : tensor<4x32x8xf32>) permutation = [1, 0, 2] -> tensor<4x32x8xf32>
        %expanded_23 = tensor.expand_shape %22 [[0], [1, 2], [3]] output_shape [4, 2, 16, 8] : tensor<4x32x8xf32> into tensor<4x2x16x8xf32>
        hivm.hir.anchor {id = 95 : i64}
        hivm.hir.copy ins(%expanded_23 : tensor<4x2x16x8xf32>) outs(%subview_19 : memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>) {"inserted-copy"}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 13
        %expanded_24 = tensor.expand_shape %20 [[0], [1, 2]] output_shape [32, 4, 8] : tensor<32x32xf32> into tensor<32x4x8xf32>
        hivm.hir.anchor {id = 96 : i64}
        %23 = hivm.hir.vtranspose ins(%expanded_24 : tensor<32x4x8xf32>) outs(%21 : tensor<4x32x8xf32>) permutation = [1, 0, 2] -> tensor<4x32x8xf32>
        %expanded_25 = tensor.expand_shape %23 [[0], [1, 2], [3]] output_shape [4, 2, 16, 8] : tensor<4x32x8xf32> into tensor<4x2x16x8xf32>
        hivm.hir.anchor {id = 97 : i64}
        hivm.hir.copy ins(%expanded_25 : tensor<4x2x16x8xf32>) outs(%subview : memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>) {"inserted-copy"}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 13
        hivm.hir.anchor {id = 98 : i64}
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      hivm.hir.anchor {id = 100 : i64}
      hivm.hir.anchor {id = 101 : i64}
      hivm.hir.anchor {id = 102 : i64}
      scf.for %arg17 = %c0 to %9 step %c1 {
        hivm.hir.anchor_block {id_end = 115 : i64, id_start = 103 : i64}
        hivm.hir.anchor {id = 104 : i64}
        hivm.hir.anchor {id = 105 : i64}
        hivm.hir.anchor {id = 106 : i64}
        hivm.hir.anchor {id = 107 : i64}
        hivm.hir.anchor {id = 108 : i64}
        hivm.hir.anchor {id = 109 : i64}
        hivm.hir.anchor {id = 110 : i64}
        hivm.hir.anchor {id = 111 : i64}
        hivm.hir.anchor {id = 112 : i64}
        hivm.hir.anchor {id = 113 : i64}
        hivm.hir.anchor {id = 114 : i64}
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      %memspacecast_15 = memref.memory_space_cast %alloc_5 : memref<2x32x32xf32, #hivm.address_space<ub>> to memref<2x32x32xf32>
      hivm.hir.anchor {id = 116 : i64}
      %14 = bufferization.to_tensor %memspacecast_15 restrict writable : memref<2x32x32xf32>
      %memspacecast_16 = memref.memory_space_cast %alloc_4 : memref<2x32x32xf32, #hivm.address_space<ub>> to memref<2x32x32xf32>
      hivm.hir.anchor {id = 117 : i64}
      %15 = bufferization.to_tensor %memspacecast_16 restrict writable : memref<2x32x32xf32>
      hivm.hir.anchor {id = 118 : i64}
      scf.for %arg17 = %c0 to %9 step %c1 {
        hivm.hir.anchor_block {id_end = 129 : i64, id_start = 119 : i64}
        hivm.hir.anchor {id = 120 : i64}
        %subview = memref.subview %alloc_3[%arg17, 0, 0, 0, 0] [1, 4, 2, 16, 8] [1, 1, 1, 1, 1] : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        %subview_19 = memref.subview %alloc_2[%arg17, 0, 0, 0, 0] [1, 4, 2, 16, 8] [1, 1, 1, 1, 1] : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        %subview_20 = memref.subview %alloc_1[%arg17, 0, 0, 0, 0] [1, 4, 2, 16, 8] [1, 1, 1, 1, 1] : memref<2x4x2x16x8xf32, #hivm.address_space<cbuf>> to memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>
        %18 = tensor.empty() : tensor<4x32x8xf32>
        hivm.hir.anchor {id = 121 : i64}
        %extracted_slice = tensor.extract_slice %15[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xf32> to tensor<32x32xf32>
        %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [32, 4, 8] : tensor<32x32xf32> into tensor<32x4x8xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 12
        hivm.hir.anchor {id = 122 : i64}
        %19 = hivm.hir.vtranspose ins(%expanded : tensor<32x4x8xf32>) outs(%18 : tensor<4x32x8xf32>) permutation = [1, 0, 2] -> tensor<4x32x8xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        %expanded_21 = tensor.expand_shape %19 [[0], [1, 2], [3]] output_shape [4, 2, 16, 8] : tensor<4x32x8xf32> into tensor<4x2x16x8xf32>
        hivm.hir.anchor {id = 123 : i64}
        hivm.hir.copy ins(%expanded_21 : tensor<4x2x16x8xf32>) outs(%subview_20 : memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>) {"inserted-copy"}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 13
        hivm.hir.anchor {id = 124 : i64}
        hivm.hir.copy ins(%expanded_21 : tensor<4x2x16x8xf32>) outs(%subview_19 : memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>) {"inserted-copy"}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 13
        hivm.hir.anchor {id = 125 : i64}
        %extracted_slice_22 = tensor.extract_slice %14[%arg17, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<2x32x32xf32> to tensor<32x32xf32>
        %expanded_23 = tensor.expand_shape %extracted_slice_22 [[0], [1, 2]] output_shape [32, 4, 8] : tensor<32x32xf32> into tensor<32x4x8xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 12
        hivm.hir.anchor {id = 126 : i64}
        %20 = hivm.hir.vtranspose ins(%expanded_23 : tensor<32x4x8xf32>) outs(%18 : tensor<4x32x8xf32>) permutation = [1, 0, 2] -> tensor<4x32x8xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        %expanded_24 = tensor.expand_shape %20 [[0], [1, 2], [3]] output_shape [4, 2, 16, 8] : tensor<4x32x8xf32> into tensor<4x2x16x8xf32>
        hivm.hir.anchor {id = 127 : i64}
        hivm.hir.copy ins(%expanded_24 : tensor<4x2x16x8xf32>) outs(%subview : memref<4x2x16x8xf32, strided<[256, 128, 8, 1], offset: ?>, #hivm.address_space<cbuf>>) {"inserted-copy"}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 13
        hivm.hir.anchor {id = 128 : i64}
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      hivm.hir.anchor {id = 130 : i64}
      hivm.hir.anchor {id = 131 : i64}
      hivm.hir.anchor {id = 132 : i64}
      hivm.hir.anchor {id = 133 : i64}
      scf.for %arg17 = %c0 to %9 step %c1 {
        hivm.hir.anchor_block {id_end = 152 : i64, id_start = 134 : i64}
        hivm.hir.anchor {id = 135 : i64}
        hivm.hir.anchor {id = 136 : i64}
        hivm.hir.anchor {id = 137 : i64}
        hivm.hir.anchor {id = 138 : i64}
        hivm.hir.anchor {id = 139 : i64}
        hivm.hir.anchor {id = 140 : i64}
        hivm.hir.anchor {id = 141 : i64}
        hivm.hir.anchor {id = 142 : i64}
        hivm.hir.anchor {id = 143 : i64}
        hivm.hir.anchor {id = 144 : i64}
        hivm.hir.anchor {id = 145 : i64}
        hivm.hir.anchor {id = 146 : i64}
        hivm.hir.anchor {id = 147 : i64}
        hivm.hir.anchor {id = 148 : i64}
        hivm.hir.anchor {id = 149 : i64}
        hivm.hir.anchor {id = 150 : i64}
        hivm.hir.anchor {id = 151 : i64}
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, multibuffer_unroll_factor = 2 : i32}
      %memspacecast_17 = memref.memory_space_cast %alloc_0 : memref<2x32x128xf32, #hivm.address_space<ub>> to memref<2x32x128xf32>
      hivm.hir.anchor {id = 153 : i64}
      %16 = bufferization.to_tensor %memspacecast_17 restrict writable : memref<2x32x128xf32>
      %memspacecast_18 = memref.memory_space_cast %alloc : memref<2x32x128xf32, #hivm.address_space<ub>> to memref<2x32x128xf32>
      hivm.hir.anchor {id = 154 : i64}
      %17 = bufferization.to_tensor %memspacecast_18 restrict writable : memref<2x32x128xf32>
      hivm.hir.anchor {id = 155 : i64}
      scf.for %arg17 = %c0 to %9 step %c1 {
        hivm.hir.anchor_block {id_end = 169 : i64, id_start = 156 : i64}
        hivm.hir.anchor {id = 157 : i64}
        %18 = affine.apply #map1(%arg17)[%7]
        %19 = arith.index_cast %18 : index to i32
        %20 = arith.remsi %19, %arg13 : i32
        %21 = arith.divsi %19, %arg13 : i32
        %22 = arith.remsi %21, %arg14 : i32
        %23 = tensor.empty() : tensor<32x128xf32>
        %24 = arith.muli %22, %arg12 : i32
        %25 = arith.muli %24, %c128_i32 : i32
        %26 = arith.index_cast %25 : i32 to index
        %27 = arith.muli %20, %c32_i32 : i32
        %28 = arith.maxsi %27, %c0_i32 : i32
        %29 = arith.index_cast %28 : i32 to index
        %30 = affine.apply #map2()[%26, %29]
        %31 = arith.index_cast %arg12 : i32 to index
        %32 = affine.apply #map3()[%31, %29]
        %33 = arith.maxsi %32, %c0 : index
        %34 = arith.minsi %33, %c32 : index
        %35 = arith.subi %c0_i32, %27 : i32
        %36 = arith.maxsi %35, %c0_i32 : i32
        %37 = arith.index_cast %36 : i32 to index
        %38 = arith.minsi %37, %34 : index
        %39 = affine.apply #map3()[%34, %38]
        %reinterpret_cast = memref.reinterpret_cast %arg9 to offset: [%30], sizes: [32, 128], strides: [128, 1] : memref<?xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>>
        hivm.hir.anchor {id = 158 : i64}
        %extracted_slice = tensor.extract_slice %10#0[%arg17, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<2x32x128xf32> to tensor<32x128xf32>
        hivm.hir.anchor {id = 159 : i64}
        %extracted_slice_19 = tensor.extract_slice %17[%arg17, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<2x32x128xf32> to tensor<32x128xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 12
        hivm.hir.anchor {id = 160 : i64}
        %40 = hivm.hir.vsub ins(%extracted_slice, %extracted_slice_19 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%23 : tensor<32x128xf32>) -> tensor<32x128xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.anchor {id = 161 : i64}
        %extracted_slice_20 = tensor.extract_slice %40[%38, 0] [%39, 128] [1, 1] : tensor<32x128xf32> to tensor<?x128xf32>
        %subview = memref.subview %reinterpret_cast[0, 0] [%39, 128] [1, 1] : memref<32x128xf32, strided<[128, 1], offset: ?>> to memref<?x128xf32, strided<[128, 1], offset: ?>>
        hivm.hir.anchor {id = 162 : i64}
        hivm.hir.store ins(%extracted_slice_20 : tensor<?x128xf32>) outs(%subview : memref<?x128xf32, strided<[128, 1], offset: ?>>)
        %reinterpret_cast_21 = memref.reinterpret_cast %arg10 to offset: [%30], sizes: [32, 128], strides: [128, 1] : memref<?xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>>
        hivm.hir.anchor {id = 163 : i64}
        %extracted_slice_22 = tensor.extract_slice %11[%arg17, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<2x32x128xf32> to tensor<32x128xf32>
        hivm.hir.anchor {id = 164 : i64}
        %extracted_slice_23 = tensor.extract_slice %16[%arg17, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<2x32x128xf32> to tensor<32x128xf32>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 12
        hivm.hir.anchor {id = 165 : i64}
        %41 = hivm.hir.vsub ins(%extracted_slice_22, %extracted_slice_23 : tensor<32x128xf32>, tensor<32x128xf32>) outs(%23 : tensor<32x128xf32>) -> tensor<32x128xf32>
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
        hivm.hir.anchor {id = 166 : i64}
        %extracted_slice_24 = tensor.extract_slice %41[%38, 0] [%39, 128] [1, 1] : tensor<32x128xf32> to tensor<?x128xf32>
        %subview_25 = memref.subview %reinterpret_cast_21[0, 0] [%39, 128] [1, 1] : memref<32x128xf32, strided<[128, 1], offset: ?>> to memref<?x128xf32, strided<[128, 1], offset: ?>>
        hivm.hir.anchor {id = 167 : i64}
        hivm.hir.store ins(%extracted_slice_24 : tensor<?x128xf32>) outs(%subview_25 : memref<?x128xf32, strided<[128, 1], offset: ?>>)
        hivm.hir.anchor {id = 168 : i64}
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, multibuffer_unroll_factor = 2 : i32}
      hivm.hir.anchor {id = 170 : i64}
    } {cv_unrolled_loop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 6
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 7
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 8
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 9
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 10
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 11
    hivm.hir.anchor {id = 172 : i64}
    return
  }
}
