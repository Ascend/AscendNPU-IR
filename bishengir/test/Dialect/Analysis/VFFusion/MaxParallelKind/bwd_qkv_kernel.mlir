// RUN: bishengir-opt --hacc-append-device-spec="target=Ascend950PR_9579" --vf-fusion="fusion-mode=max-parallel" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @bwd_qkv_kernel_mix_aic

// CHECK-LABEL: func.func private @bwd_qkv_kernel_mix_aiv_fused_0(
// CHECK: linalg.fill
// CHECK: linalg.mul
// CHECK: linalg.fill
// CHECK: linalg.mul
// CHECK: tensor.collapse_shape
// CHECK: linalg.broadcast
// CHECK: linalg.sub
// CHECK: linalg.fill
// CHECK: linalg.mul
// CHECK: linalg.exp
// CHECK: hfusion.compare
// CHECK: linalg.select
// CHECK: hfusion.cast
// CHECK: tensor.expand_shape
// CHECK: linalg.transpose
// CHECK: return

// CHECK-LABEL: func.func private @bwd_qkv_kernel_mix_aiv_fused_1(
// CHECK: linalg.broadcast
// CHECK: linalg.sub
// CHECK: linalg.mul
// CHECK: linalg.select
// CHECK: hfusion.cast
// CHECK: return

// CHECK-LABEL: func.func private @bwd_qkv_kernel_mix_aiv_fused_2(
// CHECK: tensor.expand_shape
// CHECK: linalg.transpose
// CHECK: return

// CHECK-LABEL: func.func private @bwd_qkv_kernel_mix_aiv_fused_3(
// CHECK: linalg.fill
// CHECK: linalg.mul
// CHECK: return

// CHECK-LABEL: func.func private @bwd_qkv_kernel_mix_aiv_fused_4(
// CHECK: linalg.fill
// CHECK: linalg.mul
// CHECK: return

// CHECK-LABEL: func.func private @bwd_qkv_kernel_mix_aiv_fused_5(
// CHECK: linalg.fill
// CHECK: linalg.mul
// CHECK: return

// CHECK-LABEL: func.func @bwd_qkv_kernel_mix_aiv(
// CHECK: func.call @bwd_qkv_kernel_mix_aiv_fused_4
// CHECK: func.call @bwd_qkv_kernel_mix_aiv_fused_0
// CHECK: func.call @bwd_qkv_kernel_mix_aiv_fused_1
// CHECK: func.call @bwd_qkv_kernel_mix_aiv_fused_2
// CHECK: func.call @bwd_qkv_kernel_mix_aiv_fused_3
// CHECK: func.call @bwd_qkv_kernel_mix_aiv_fused_5
// CHECK: return
#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0, s1, s2, s3] -> (s0 * s1 + s2 + s3)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
#map3 = affine_map<()[s0, s1] -> ((s0 - s1) ceildiv 16)>
#map4 = affine_map<(d0) -> (d0 floordiv 16)>
#map5 = affine_map<(d0) -> (d0 mod 16)>
#map6 = affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 * s3)>
#map7 = affine_map<()[s0, s1, s2, s3] -> (s0 - (s2 + s3 * s1) floordiv s1)>
#map8 = affine_map<()[s0, s1, s2, s3] -> (s0 - (s2 + s3 * s1) mod s1)>
#map9 = affine_map<()[s0] -> (s0 * 64)>
#map10 = affine_map<()[s0] -> (s0 + 32)>
#map11 = affine_map<()[s0] -> (s0 * 16)>
#map12 = affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2)>
#map13 = affine_map<()[s0] -> (s0 * 16 + 16)>
#map14 = affine_map<()[s0, s1] -> (s0 - s1 * 16)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>, ssbuffer.insertionOptimization} {
  func.func @bwd_qkv_kernel_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 2 : i32}, %arg6: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %arg7: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %arg8: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi8> {tt.tensor_kind = 0 : i32}, %arg14: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg15: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: f32, %arg22: i32, %arg23: i32, %arg24: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<25xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c28_i32 = arith.constant 28 : i32
    %c31_i32 = arith.constant 31 : i32
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    hivm.hir.anchor {id = 0 : i64}
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %0 = arith.muli %arg22, %arg23 : i32
    %1 = arith.muli %0, %arg24 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.remsi %3, %arg22 : i32
    %5 = arith.addi %arg18, %c31_i32 : i32
    %6 = arith.divsi %5, %c32_i32 : i32
    %7 = arith.extsi %arg19 : i32 to i64
    %8 = arith.muli %6, %arg19 : i32
    %9 = arith.muli %8, %arg16 : i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg25 = %4 to %9 step %c28_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      %10 = arith.divsi %arg25, %6 : i32
      %11 = arith.remsi %arg25, %6 : i32
      %12 = arith.divsi %10, %arg19 : i32
      %13 = arith.remsi %10, %arg19 : i32
      %14 = arith.divsi %arg19, %arg20 : i32
      %15 = arith.divsi %13, %14 : i32
      %16 = arith.index_cast %12 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg15 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      hivm.hir.anchor {id = 3 : i64}
      %17 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
      hivm.hir.anchor {id = 4 : i64}
      %18 = affine.apply #map()[%16]
      %reinterpret_cast_0 = memref.reinterpret_cast %arg15 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
      %19 = memref.load %reinterpret_cast_0[%c0] : memref<1xi32, strided<[1], offset: ?>>
      hivm.hir.anchor {id = 5 : i64}
      %20 = arith.extsi %17 : i32 to i64
      %21 = arith.subi %19, %17 : i32
      %22 = arith.muli %11, %c32_i32 : i32
      %23 = arith.cmpi slt, %22, %21 : i32
      scf.if %23 {
        hivm.hir.anchor {id = 6 : i64}
        %reinterpret_cast_1 = memref.reinterpret_cast %arg14 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        hivm.hir.anchor {id = 7 : i64}
        %24 = memref.load %reinterpret_cast_1[%c0] : memref<1xi32, strided<[1], offset: ?>>
        hivm.hir.anchor {id = 8 : i64}
        %reinterpret_cast_2 = memref.reinterpret_cast %arg14 to offset: [%18], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %25 = memref.load %reinterpret_cast_2[%c0] : memref<1xi32, strided<[1], offset: ?>>
        hivm.hir.anchor {id = 9 : i64}
        %26 = arith.extsi %24 : i32 to i64
        %27 = arith.subi %25, %24 : i32
        %28 = arith.divsi %22, %c32_i32 : i32
        %29 = arith.muli %28, %c32_i32 : i32
        %30 = arith.addi %29, %c32_i32 : i32
        %31 = arith.minsi %30, %27 : i32
        hivm.hir.anchor {id = 10 : i64}
        hivm.hir.anchor {id = 11 : i64}
        hivm.hir.anchor {id = 12 : i64}
        %32 = arith.muli %26, %7 : i64
        %33 = arith.muli %32, %c64_i64 : i64
        %34 = arith.muli %13, %c64_i32 : i32
        %35 = arith.index_cast %33 : i64 to index
        %36 = arith.index_cast %34 : i32 to index
        %37 = arith.muli %arg19, %c64_i32 : i32
        %38 = arith.extsi %arg20 : i32 to i64
        %39 = arith.muli %20, %38 : i64
        %40 = arith.muli %39, %c64_i64 : i64
        %41 = arith.muli %15, %c64_i32 : i32
        %42 = arith.index_cast %40 : i64 to index
        %43 = arith.index_cast %41 : i32 to index
        %44 = arith.muli %arg20, %c64_i32 : i32
        %45 = arith.muli %20, %7 : i64
        %46 = arith.muli %45, %c64_i64 : i64
        %47 = arith.index_cast %46 : i64 to index
        %48 = arith.maxsi %22, %c0_i32 : i32
        %49 = arith.index_cast %48 : i32 to index
        %50 = arith.index_cast %37 : i32 to index
        %51 = affine.apply #map1()[%49, %50, %47, %36]
        %52 = arith.index_cast %21 : i32 to index
        %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%51], sizes: [32, 64], strides: [%50, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
        %53 = arith.index_cast %44 : i32 to index
        %54 = affine.apply #map1()[%49, %53, %42, %43]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%54], sizes: [32, 64], strides: [%53, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
        %55 = affine.apply #map2()[%52, %49]
        %56 = arith.maxsi %55, %c0 : index
        %57 = arith.minsi %56, %c32 : index
        %58 = arith.subi %c0_i32, %22 : i32
        %59 = arith.maxsi %58, %c0_i32 : i32
        %60 = arith.index_cast %59 : i32 to index
        %61 = arith.minsi %60, %57 : index
        %62 = affine.apply #map2()[%57, %61]
        %63 = arith.cmpi slt, %62, %c32 : index
        %subview = memref.subview %reinterpret_cast_4[0, 0] [%62, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
        %64 = affine.apply #map3()[%57, %61]
        %65 = affine.apply #map4(%61)
        %66 = affine.apply #map5(%61)
        %alloc = memref.alloc() : memref<4x2x16x16xbf16>
        %subview_5 = memref.subview %alloc[0, %65, %66, 0] [4, %64, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
        hivm.hir.anchor {id = 13 : i64}
        hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x64xbf16, strided<[?, 1], offset: ?>>) outs(%subview_5 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) init_out_buffer = true pad_value = %cst : bf16 init_condition = %63 : i1
        hivm.hir.anchor {id = 14 : i64}
        %67 = bufferization.to_tensor %alloc restrict writable : memref<4x2x16x16xbf16>
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [32, 64], strides: [%53, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
        %subview_7 = memref.subview %reinterpret_cast_6[0, 0] [%62, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
        %alloc_8 = memref.alloc() : memref<4x2x16x16xbf16>
        %subview_9 = memref.subview %alloc_8[0, %65, %66, 0] [4, %64, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
        hivm.hir.anchor {id = 15 : i64}
        hivm.hir.nd2nz {dst_continuous} ins(%subview_7 : memref<?x64xbf16, strided<[?, 1], offset: ?>>) outs(%subview_9 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) init_out_buffer = true pad_value = %cst : bf16 init_condition = %63 : i1
        hivm.hir.anchor {id = 16 : i64}
        %68 = bufferization.to_tensor %alloc_8 restrict writable : memref<4x2x16x16xbf16>
        %alloca = memref.alloca() {normalize_matmul_counter = 6 : i32} : memref<i32>
        hivm.hir.anchor {id = 17 : i64}
        memref.store %c0_i32, %alloca[] : memref<i32>
        hivm.hir.anchor {id = 18 : i64}
        %alloca_10 = memref.alloca() {normalize_matmul_counter = 5 : i32} : memref<i32>
        hivm.hir.anchor {id = 19 : i64}
        memref.store %c0_i32, %alloca_10[] : memref<i32>
        hivm.hir.anchor {id = 20 : i64}
        %69 = tensor.empty() : tensor<4x2x16x16xf32>
        %70:6 = scf.for %arg26 = %c0_i32 to %31 step %c32_i32 iter_args(%arg27 = %c0_i32, %arg28 = %c0_i32, %arg29 = %69, %arg30 = %69, %arg31 = %c0_i32, %arg32 = %c0_i32) -> (i32, i32, tensor<4x2x16x16xf32>, tensor<4x2x16x16xf32>, i32, i32)  : i32 {
          hivm.hir.anchor {id = 21 : i64}
          %alloc_13 = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_13 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x64xf32, #hivm.address_space<ub>>
          %alloc_14 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_14 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
          %alloc_15 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_15 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
          %alloc_16 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_16 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
          %alloc_17 = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
          %alloc_18 = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_18 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
          hivm.hir.anchor {id = 22 : i64}
          %75:2 = scope.scope : () -> (i32, i32) {
            hivm.hir.anchor {id = 23 : i64}
            %80 = arith.maxsi %arg27, %c0_i32 : i32
            %81 = arith.index_cast %80 : i32 to index
            %82 = affine.apply #map1()[%81, %50, %35, %36]
            %83 = arith.index_cast %27 : i32 to index
            %reinterpret_cast_21 = memref.reinterpret_cast %arg2 to offset: [%82], sizes: [32, 64], strides: [%50, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
            %84 = arith.maxsi %arg28, %c0_i32 : i32
            %85 = arith.index_cast %84 : i32 to index
            %86 = affine.apply #map1()[%85, %50, %35, %36]
            %reinterpret_cast_22 = memref.reinterpret_cast %arg8 to offset: [%86], sizes: [32, 64], strides: [%50, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
            %87 = affine.apply #map2()[%83, %81]
            %88 = arith.maxsi %87, %c0 : index
            %89 = arith.minsi %88, %c32 : index
            %90 = arith.subi %c0_i32, %arg27 : i32
            %91 = arith.maxsi %90, %c0_i32 : i32
            %92 = arith.index_cast %91 : i32 to index
            %93 = arith.minsi %92, %89 : index
            %94 = affine.apply #map2()[%89, %93]
            %95 = arith.cmpi slt, %94, %c32 : index
            %subview_23 = memref.subview %reinterpret_cast_21[0, 0] [%94, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
            %96 = affine.apply #map3()[%89, %93]
            %97 = affine.apply #map4(%93)
            %98 = affine.apply #map5(%93)
            %alloc_24 = memref.alloc() : memref<4x2x16x16xbf16>
            %subview_25 = memref.subview %alloc_24[0, %97, %98, 0] [4, %96, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
            hivm.hir.anchor {id = 24 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%subview_23 : memref<?x64xbf16, strided<[?, 1], offset: ?>>) outs(%subview_25 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) init_out_buffer = true pad_value = %cst : bf16 init_condition = %95 : i1
            hivm.hir.anchor {id = 25 : i64}
            %99 = bufferization.to_tensor %alloc_24 restrict writable : memref<4x2x16x16xbf16>
            hivm.hir.anchor {id = 26 : i64}
            %100 = tensor.empty() : tensor<2x2x16x16xf32>
            %101 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%99, %67, %true, %c32, %c64, %c32 : tensor<4x2x16x16xbf16>, tensor<4x2x16x16xbf16>, i1, index, index, index) outs(%100 : tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32>
            hivm.hir.anchor {id = 27 : i64}
            annotation.mark %alloc_18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x32xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 28 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%101 : tensor<2x2x16x16xf32>) outs(%alloc_18 : memref<16x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 29 : i64}
            %102 = affine.apply #map2()[%83, %85]
            %103 = arith.maxsi %102, %c0 : index
            %104 = arith.minsi %103, %c32 : index
            %105 = arith.subi %c0_i32, %arg28 : i32
            %106 = arith.maxsi %105, %c0_i32 : i32
            %107 = arith.index_cast %106 : i32 to index
            %108 = arith.minsi %107, %104 : index
            %109 = affine.apply #map2()[%104, %108]
            %110 = arith.cmpi slt, %109, %c32 : index
            %subview_26 = memref.subview %reinterpret_cast_22[0, 0] [%109, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
            %111 = affine.apply #map3()[%104, %108]
            %112 = affine.apply #map4(%108)
            %113 = affine.apply #map5(%108)
            %alloc_27 = memref.alloc() : memref<4x2x16x16xbf16>
            %subview_28 = memref.subview %alloc_27[0, %112, %113, 0] [4, %111, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
            hivm.hir.anchor {id = 30 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<?x64xbf16, strided<[?, 1], offset: ?>>) outs(%subview_28 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) init_out_buffer = true pad_value = %cst : bf16 init_condition = %110 : i1
            hivm.hir.anchor {id = 31 : i64}
            %114 = bufferization.to_tensor %alloc_27 restrict writable : memref<4x2x16x16xbf16>
            hivm.hir.anchor {id = 32 : i64}
            %115 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%114, %68, %true, %c32, %c64, %c32 : tensor<4x2x16x16xbf16>, tensor<4x2x16x16xbf16>, i1, index, index, index) outs(%100 : tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32>
            hivm.hir.anchor {id = 33 : i64}
            annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x32xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 34 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%115 : tensor<2x2x16x16xf32>) outs(%alloc_17 : memref<16x32xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 35 : i64}
            %116 = arith.addi %arg27, %c32_i32 : i32
            %117 = arith.addi %arg28, %c32_i32 : i32
            scope.return %116, %117 : i32, i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
          hivm.hir.anchor {id = 36 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 37 : i64}
          scope.scope : () -> () {
            hivm.hir.anchor {id = 38 : i64}
            hivm.hir.anchor {id = 39 : i64}
            hivm.hir.anchor {id = 40 : i64}
            hivm.hir.anchor {id = 41 : i64}
            hivm.hir.anchor {id = 42 : i64}
            hivm.hir.anchor {id = 43 : i64}
            hivm.hir.anchor {id = 44 : i64}
            hivm.hir.anchor {id = 45 : i64}
            hivm.hir.anchor {id = 46 : i64}
            hivm.hir.anchor {id = 47 : i64}
            hivm.hir.anchor {id = 48 : i64}
            hivm.hir.anchor {id = 49 : i64}
            hivm.hir.anchor {id = 50 : i64}
            %80 = tensor.empty() : tensor<32x32xbf16>
            hivm.hir.anchor {id = 51 : i64}
            hivm.hir.anchor {id = 52 : i64}
            annotation.mark %alloc_16 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, hivm.tiling_dim = -1 : index} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 53 : i64}
            hivm.hir.anchor {id = 54 : i64}
            hivm.hir.anchor {id = 55 : i64}
            hivm.hir.anchor {id = 56 : i64}
            hivm.hir.anchor {id = 57 : i64}
            hivm.hir.anchor {id = 58 : i64}
            hivm.hir.anchor {id = 59 : i64}
            hivm.hir.anchor {id = 60 : i64}
            hivm.hir.anchor {id = 61 : i64}
            annotation.mark %80 {break_vf} : tensor<32x32xbf16>
            hivm.hir.anchor {id = 62 : i64}
            hivm.hir.anchor {id = 63 : i64}
            annotation.mark %alloc_15 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, hivm.tiling_dim = -1 : index} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 64 : i64}
            hivm.hir.anchor {id = 65 : i64}
            annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, hivm.tiling_dim = -1 : index} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 66 : i64}
            hivm.hir.anchor {id = 67 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
          hivm.hir.anchor {id = 68 : i64}
          %memspacecast = memref.memory_space_cast %alloc_16 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
          %76 = bufferization.to_tensor %memspacecast restrict writable : memref<2x2x16x16xbf16>
          %memspacecast_19 = memref.memory_space_cast %alloc_15 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
          %77 = bufferization.to_tensor %memspacecast_19 restrict writable : memref<2x2x16x16xbf16>
          %memspacecast_20 = memref.memory_space_cast %alloc_14 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
          %78 = bufferization.to_tensor %memspacecast_20 restrict writable : memref<2x2x16x16xbf16>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          hivm.hir.anchor {id = 69 : i64}
          %79:4 = scope.scope : () -> (tensor<4x2x16x16xf32>, tensor<4x2x16x16xf32>, i32, i32) {
            hivm.hir.anchor {id = 70 : i64}
            %80 = arith.maxsi %arg31, %c0_i32 : i32
            %81 = arith.index_cast %80 : i32 to index
            %82 = affine.apply #map1()[%81, %50, %35, %36]
            %83 = arith.index_cast %27 : i32 to index
            %reinterpret_cast_21 = memref.reinterpret_cast %arg2 to offset: [%82], sizes: [32, 64], strides: [%50, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
            %84 = arith.maxsi %arg32, %c0_i32 : i32
            %85 = arith.index_cast %84 : i32 to index
            %86 = affine.apply #map1()[%85, %50, %35, %36]
            %reinterpret_cast_22 = memref.reinterpret_cast %arg8 to offset: [%86], sizes: [32, 64], strides: [%50, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
            %87 = affine.apply #map2()[%83, %81]
            %88 = arith.maxsi %87, %c0 : index
            %89 = arith.minsi %88, %c32 : index
            %90 = arith.subi %c0_i32, %arg31 : i32
            %91 = arith.maxsi %90, %c0_i32 : i32
            %92 = arith.index_cast %91 : i32 to index
            %93 = arith.minsi %92, %89 : index
            %94 = affine.apply #map2()[%89, %93]
            %95 = arith.cmpi slt, %94, %c32 : index
            %subview_23 = memref.subview %reinterpret_cast_21[0, 0] [%94, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
            %96 = affine.apply #map3()[%89, %93]
            %97 = affine.apply #map4(%93)
            %98 = affine.apply #map5(%93)
            %alloc_24 = memref.alloc() : memref<4x2x16x16xbf16>
            %subview_25 = memref.subview %alloc_24[0, %97, %98, 0] [4, %96, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
            hivm.hir.anchor {id = 71 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%subview_23 : memref<?x64xbf16, strided<[?, 1], offset: ?>>) outs(%subview_25 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) init_out_buffer = true pad_value = %cst : bf16 init_condition = %95 : i1
            hivm.hir.anchor {id = 72 : i64}
            %99 = bufferization.to_tensor %alloc_24 restrict writable : memref<4x2x16x16xbf16>
            %100 = affine.apply #map2()[%83, %85]
            %101 = arith.maxsi %100, %c0 : index
            %102 = arith.minsi %101, %c32 : index
            %103 = arith.subi %c0_i32, %arg32 : i32
            %104 = arith.maxsi %103, %c0_i32 : i32
            %105 = arith.index_cast %104 : i32 to index
            %106 = arith.minsi %105, %102 : index
            %107 = affine.apply #map2()[%102, %106]
            %108 = arith.cmpi slt, %107, %c32 : index
            %subview_26 = memref.subview %reinterpret_cast_22[0, 0] [%107, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
            %109 = affine.apply #map3()[%102, %106]
            %110 = affine.apply #map4(%106)
            %111 = affine.apply #map5(%106)
            %alloc_27 = memref.alloc() : memref<4x2x16x16xbf16>
            %subview_28 = memref.subview %alloc_27[0, %110, %111, 0] [4, %109, 16, 16] [1, 1, 1, 1] : memref<4x2x16x16xbf16> to memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
            hivm.hir.anchor {id = 73 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%subview_26 : memref<?x64xbf16, strided<[?, 1], offset: ?>>) outs(%subview_28 : memref<4x?x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) init_out_buffer = true pad_value = %cst : bf16 init_condition = %108 : i1
            hivm.hir.anchor {id = 74 : i64}
            %112 = bufferization.to_tensor %alloc_27 restrict writable : memref<4x2x16x16xbf16>
            hivm.hir.anchor {id = 75 : i64}
            %113 = memref.load %alloca[] : memref<i32>
            hivm.hir.anchor {id = 76 : i64}
            %114 = arith.cmpi eq, %113, %c0_i32 : i32
            %115 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, hivm.remain_in_l0c, normalized_in_L0C} ins(%76, %112, %114, %c32, %c32, %c64 : tensor<2x2x16x16xbf16>, tensor<4x2x16x16xbf16>, i1, index, index, index) outs(%arg30 : tensor<4x2x16x16xf32>) -> tensor<4x2x16x16xf32>
            hivm.hir.anchor {id = 77 : i64}
            %116 = arith.addi %113, %c1_i32 : i32
            memref.store %116, %alloca[] : memref<i32>
            hivm.hir.anchor {id = 78 : i64}
            %117 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%77, %67, %true, %c32, %c32, %c64 : tensor<2x2x16x16xbf16>, tensor<4x2x16x16xbf16>, i1, index, index, index) outs(%69 : tensor<4x2x16x16xf32>) -> tensor<4x2x16x16xf32>
            hivm.hir.anchor {id = 79 : i64}
            annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 80 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%117 : tensor<4x2x16x16xf32>) outs(%alloc_13 : memref<16x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 81 : i64}
            %118 = memref.load %alloca_10[] : memref<i32>
            hivm.hir.anchor {id = 82 : i64}
            %119 = arith.cmpi eq, %118, %c0_i32 : i32
            %120 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, hivm.remain_in_l0c, normalized_in_L0C} ins(%78, %99, %119, %c32, %c32, %c64 : tensor<2x2x16x16xbf16>, tensor<4x2x16x16xbf16>, i1, index, index, index) outs(%arg29 : tensor<4x2x16x16xf32>) -> tensor<4x2x16x16xf32>
            hivm.hir.anchor {id = 83 : i64}
            %121 = arith.addi %118, %c1_i32 : i32
            memref.store %121, %alloca_10[] : memref<i32>
            hivm.hir.anchor {id = 84 : i64}
            %122 = arith.addi %arg31, %c32_i32 : i32
            %123 = arith.addi %arg32, %c32_i32 : i32
            scope.return %115, %120, %122, %123 : tensor<4x2x16x16xf32>, tensor<4x2x16x16xf32>, i32, i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
          hivm.hir.anchor {id = 85 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
          hivm.hir.anchor {id = 86 : i64}
          scope.scope : () -> () {
            hivm.hir.anchor {id = 87 : i64}
            hivm.hir.anchor {id = 88 : i64}
            hivm.hir.anchor {id = 89 : i64}
            hivm.hir.anchor {id = 90 : i64}
            hivm.hir.anchor {id = 91 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
          hivm.hir.anchor {id = 92 : i64}
          scf.yield %75#0, %75#1, %79#1, %79#0, %79#2, %79#3 : i32, i32, tensor<4x2x16x16xf32>, tensor<4x2x16x16xf32>, i32, i32
        } {fixpipe_for_mmad_result_already_inserted = true}
        hivm.hir.anchor {id = 93 : i64}
        %71 = memref.load %alloca[] : memref<i32>
        hivm.hir.anchor {id = 94 : i64}
        %72 = arith.cmpi eq, %71, %c0_i32 : i32
        %subview_11 = memref.subview %reinterpret_cast_3[0, 0] [%62, 64] [1, 1] : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
        scf.if %72 {
          hivm.hir.anchor {id = 95 : i64}
          hivm.hir.anchor {id = 96 : i64}
          hivm.hir.anchor {id = 97 : i64}
          hivm.hir.anchor {id = 98 : i64}
          hivm.hir.anchor {id = 99 : i64}
          hivm.hir.anchor {id = 100 : i64}
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_FIX>] flag = 0
        } else {
          hivm.hir.anchor {id = 101 : i64}
          %extracted_slice = tensor.extract_slice %70#3[0, %65, %66, 0] [4, %64, 16, 16] [1, 1, 1, 1] : tensor<4x2x16x16xf32> to tensor<4x?x16x16xf32>
          hivm.hir.anchor {id = 102 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322BF16>} ins(%extracted_slice : tensor<4x?x16x16xf32>) outs(%subview_11 : memref<?x64xbf16, strided<[?, 1], offset: ?>>)
          hivm.hir.anchor {id = 103 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
        } {fallback_not_exec}
        hivm.hir.anchor {id = 104 : i64}
        %73 = memref.load %alloca_10[] : memref<i32>
        hivm.hir.anchor {id = 105 : i64}
        %74 = arith.cmpi eq, %73, %c0_i32 : i32
        %alloc_12 = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
        hivm.hir.anchor {id = 106 : i64}
        scf.if %74 {
          hivm.hir.anchor {id = 107 : i64}
          hivm.hir.anchor {id = 108 : i64}
          hivm.hir.anchor {id = 109 : i64}
        } else {
          hivm.hir.anchor {id = 110 : i64}
          hivm.hir.anchor {id = 111 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%70#2 : tensor<4x2x16x16xf32>) outs(%alloc_12 : memref<16x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.anchor {id = 112 : i64}
        }
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.anchor {id = 113 : i64}
        hivm.hir.anchor {id = 114 : i64}
        hivm.hir.anchor {id = 115 : i64}
        hivm.hir.anchor {id = 116 : i64}
        hivm.hir.anchor {id = 117 : i64}
      }
      hivm.hir.anchor {id = 118 : i64}
    }
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
    hivm.hir.set_ctrl true at ctrl[60]
    hivm.hir.anchor {id = 119 : i64}
    return
  }
  func.func @bwd_qkv_kernel_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 2 : i32}, %arg6: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %arg7: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %arg8: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %arg9: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg11: memref<?xi32>, %arg12: memref<?xi32>, %arg13: memref<?xi8> {tt.tensor_kind = 0 : i32}, %arg14: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg15: memref<?xi32> {tt.tensor_kind = 0 : i32}, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: f32, %arg22: i32, %arg23: i32, %arg24: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false, false, false]> : vector<25xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : bf16
    %c16 = arith.constant 16 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.693147182 : f32
    %c32 = arith.constant 32 : index
    %c31_i32 = arith.constant 31 : i32
    %c28_i32 = arith.constant 28 : i32
    %c0_i8 = arith.constant 0 : i8
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_2 = arith.constant 1.44269502 : f32
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.muli %arg22, %arg23 : i32
    %1 = arith.muli %0, %arg24 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.remsi %3, %arg22 : i32
    %5 = arith.addi %arg18, %c31_i32 : i32
    %6 = arith.divsi %5, %c32_i32 : i32
    %7 = arith.extsi %arg19 : i32 to i64
    %8 = arith.muli %6, %arg19 : i32
    %9 = arith.muli %8, %arg16 : i32
    scf.for %arg25 = %c0 to %c2 step %c1 {
      %10 = affine.apply #map11()[%arg25]
      hivm.hir.anchor {id = 0 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      annotation.mark %1 {logical_block_num} : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
      hivm.hir.anchor {id = 1 : i64}
      scf.for %arg26 = %4 to %9 step %c28_i32  : i32 {
        hivm.hir.anchor {id = 2 : i64}
        %11 = arith.divsi %arg26, %6 : i32
        %12 = arith.remsi %arg26, %6 : i32
        %13 = arith.divsi %11, %arg19 : i32
        %14 = arith.remsi %11, %arg19 : i32
        %15 = arith.index_cast %13 : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg15 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %16 = affine.apply #map()[%15]
        hivm.hir.anchor {id = 3 : i64}
        %17 = memref.load %reinterpret_cast[%c0] : memref<1xi32, strided<[1], offset: ?>>
        hivm.hir.anchor {id = 4 : i64}
        %reinterpret_cast_3 = memref.reinterpret_cast %arg15 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %18 = memref.load %reinterpret_cast_3[%c0] : memref<1xi32, strided<[1], offset: ?>>
        hivm.hir.anchor {id = 5 : i64}
        %19 = arith.extsi %17 : i32 to i64
        %20 = arith.subi %18, %17 : i32
        %21 = arith.muli %12, %c32_i32 : i32
        %22 = arith.cmpi slt, %21, %20 : i32
        scf.if %22 {
          hivm.hir.anchor {id = 6 : i64}
          %reinterpret_cast_4 = memref.reinterpret_cast %arg14 to offset: [%15], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          hivm.hir.anchor {id = 7 : i64}
          %23 = memref.load %reinterpret_cast_4[%c0] : memref<1xi32, strided<[1], offset: ?>>
          hivm.hir.anchor {id = 8 : i64}
          %reinterpret_cast_5 = memref.reinterpret_cast %arg14 to offset: [%16], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %24 = memref.load %reinterpret_cast_5[%c0] : memref<1xi32, strided<[1], offset: ?>>
          hivm.hir.anchor {id = 9 : i64}
          %25 = arith.extsi %23 : i32 to i64
          %26 = arith.subi %24, %23 : i32
          %27 = arith.divsi %21, %c32_i32 : i32
          %28 = arith.muli %27, %c32_i32 : i32
          %29 = arith.addi %28, %c32_i32 : i32
          %30 = arith.minsi %29, %26 : i32
          %31 = tensor.empty() : tensor<1xf32>
          %inserted = tensor.insert %arg21 into %31[%c0] : tensor<1xf32>
          hivm.hir.anchor {id = 10 : i64}
          %32 = linalg.fill ins(%cst_2 : f32) outs(%31 : tensor<1xf32>) -> tensor<1xf32>
          %33 = linalg.mul ins(%inserted, %32 : tensor<1xf32>, tensor<1xf32>) outs(%31 : tensor<1xf32>) -> tensor<1xf32>
          hivm.hir.anchor {id = 11 : i64}
          %extracted = tensor.extract %33[%c0] {"DuplicateTensorExtractForCube::visitedLabel" = 1 : i32} : tensor<1xf32>
          hivm.hir.anchor {id = 12 : i64}
          %34 = arith.muli %25, %7 : i64
          %35 = arith.muli %14, %c64_i32 : i32
          %36 = arith.index_cast %35 : i32 to index
          %37 = arith.muli %arg19, %c64_i32 : i32
          %38 = arith.muli %19, %7 : i64
          %39 = arith.muli %38, %c64_i64 : i64
          %40 = arith.index_cast %39 : i64 to index
          %41 = arith.maxsi %21, %c0_i32 : i32
          %42 = arith.index_cast %41 : i32 to index
          %43 = arith.index_cast %37 : i32 to index
          %44 = affine.apply #map1()[%42, %43, %40, %36]
          %45 = arith.index_cast %20 : i32 to index
          %46 = affine.apply #map2()[%45, %42]
          %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%44], sizes: [32, 64], strides: [%43, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
          %subview = memref.subview %reinterpret_cast_6[%10, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<16x64xbf16, strided<[?, 1], offset: ?>>
          %reinterpret_cast_7 = memref.reinterpret_cast %arg7 to offset: [%44], sizes: [32, 64], strides: [%43, 1] : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
          %subview_8 = memref.subview %reinterpret_cast_7[%10, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<16x64xbf16, strided<[?, 1], offset: ?>>
          %47 = arith.index_cast %34 : i64 to index
          %48 = arith.index_cast %14 : i32 to index
          %49 = arith.muli %13, %arg17 : i32
          %50 = arith.muli %49, %arg18 : i32
          %51 = arith.index_cast %50 : i32 to index
          %52 = arith.maxsi %46, %c0 : index
          %53 = arith.minsi %52, %c32 : index
          %54 = arith.subi %c0_i32, %21 : i32
          %55 = arith.maxsi %54, %c0_i32 : i32
          %56 = arith.index_cast %55 : i32 to index
          %57 = arith.minsi %56, %53 : index
          %58 = affine.apply #map2()[%53, %57]
          hivm.hir.anchor {id = 13 : i64}
          hivm.hir.anchor {id = 14 : i64}
          hivm.hir.anchor {id = 15 : i64}
          hivm.hir.anchor {id = 16 : i64}
          %59 = arith.extsi %35 : i32 to i64
          %alloca = memref.alloca() {normalize_matmul_counter = 6 : i32} : memref<i32>
          hivm.hir.anchor {id = 17 : i64}
          memref.store %c0_i32, %alloca[] : memref<i32>
          hivm.hir.anchor {id = 18 : i64}
          %alloca_9 = memref.alloca() {normalize_matmul_counter = 5 : i32} : memref<i32>
          hivm.hir.anchor {id = 19 : i64}
          memref.store %c0_i32, %alloca_9[] : memref<i32>
          hivm.hir.anchor {id = 20 : i64}
          %60:3 = scf.for %arg27 = %c0_i32 to %30 step %c32_i32 iter_args(%arg28 = %c0_i32, %arg29 = %c0_i32, %arg30 = %c0_i32) -> (i32, i32, i32)  : i32 {
            hivm.hir.anchor {id = 21 : i64}
            %alloc_12 = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
            annotation.mark %alloc_12 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x64xf32, #hivm.address_space<ub>>
            %alloc_13 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %alloc_13 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            %alloc_14 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %alloc_14 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            %alloc_15 = memref.alloc() : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            annotation.mark %alloc_15 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
            %alloc_16 = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
            annotation.mark %alloc_16 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
            %alloc_17 = memref.alloc() : memref<16x32xf32, #hivm.address_space<ub>>
            annotation.mark %alloc_17 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x32xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 22 : i64}
            scope.scope : () -> () {
              hivm.hir.anchor {id = 23 : i64}
              hivm.hir.anchor {id = 24 : i64}
              hivm.hir.anchor {id = 25 : i64}
              hivm.hir.anchor {id = 26 : i64}
              hivm.hir.anchor {id = 27 : i64}
              annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x32xf32, #hivm.address_space<ub>>
              hivm.hir.anchor {id = 28 : i64}
              hivm.hir.anchor {id = 29 : i64}
              hivm.hir.anchor {id = 30 : i64}
              hivm.hir.anchor {id = 31 : i64}
              hivm.hir.anchor {id = 32 : i64}
              hivm.hir.anchor {id = 33 : i64}
              annotation.mark %alloc_16 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x32xf32, #hivm.address_space<ub>>
              hivm.hir.anchor {id = 34 : i64}
              hivm.hir.anchor {id = 35 : i64}
              scope.return
            } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
            hivm.hir.anchor {id = 36 : i64}
            %memspacecast = memref.memory_space_cast %alloc_17 : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
            %80 = bufferization.to_tensor %memspacecast restrict writable : memref<16x32xf32>
            %memspacecast_18 = memref.memory_space_cast %alloc_16 : memref<16x32xf32, #hivm.address_space<ub>> to memref<16x32xf32>
            %81 = bufferization.to_tensor %memspacecast_18 restrict writable : memref<16x32xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
            hivm.hir.anchor {id = 37 : i64}
            %82:3 = scope.scope : () -> (i32, i32, i32) {
              hivm.hir.anchor {id = 38 : i64}
              %84 = arith.index_cast %26 : i32 to index
              %85 = arith.maxsi %arg28, %c0_i32 : i32
              %86 = arith.index_cast %85 : i32 to index
              %87 = affine.apply #map2()[%84, %86]
              %88 = arith.index_cast %arg19 : i32 to index
              %89 = affine.apply #map1()[%86, %88, %47, %48]
              %reinterpret_cast_20 = memref.reinterpret_cast %arg9 to offset: [%89], sizes: [32], strides: [%88] : memref<?xf32> to memref<32xf32, strided<[?], offset: ?>>
              %subview_21 = memref.subview %reinterpret_cast_20[%10] [16] [1] : memref<32xf32, strided<[?], offset: ?>> to memref<16xf32, strided<[?], offset: ?>>
              %90 = arith.maxsi %arg29, %c0_i32 : i32
              %91 = arith.index_cast %90 : i32 to index
              %92 = affine.apply #map1()[%91, %88, %47, %48]
              %93 = affine.apply #map2()[%84, %91]
              %reinterpret_cast_22 = memref.reinterpret_cast %arg10 to offset: [%92], sizes: [32], strides: [%88] : memref<?xf32> to memref<32xf32, strided<[?], offset: ?>>
              %subview_23 = memref.subview %reinterpret_cast_22[%10] [16] [1] : memref<32xf32, strided<[?], offset: ?>> to memref<16xf32, strided<[?], offset: ?>>
              %94 = arith.maxsi %arg30, %c0_i32 : i32
              %95 = arith.index_cast %94 : i32 to index
              %96 = arith.index_cast %arg18 : i32 to index
              %97 = affine.apply #map6()[%42, %51, %95, %96]
              %reinterpret_cast_24 = memref.reinterpret_cast %arg13 to offset: [%97], sizes: [32, 32], strides: [%96, 1] : memref<?xi8> to memref<32x32xi8, strided<[?, 1], offset: ?>>
              %98 = affine.apply #map8()[%45, %96, %42, %95]
              %99 = affine.apply #map7()[%84, %96, %42, %95]
              %subview_25 = memref.subview %reinterpret_cast_24[%10, 0] [16, 32] [1, 1] : memref<32x32xi8, strided<[?, 1], offset: ?>> to memref<16x32xi8, strided<[?, 1], offset: ?>>
              %alloc_26 = memref.alloc() : memref<16x32xi8>
              %100 = arith.maxsi %99, %c0 : index
              %101 = arith.minsi %100, %c32 : index
              %102 = arith.maxsi %98, %c0 : index
              %103 = arith.minsi %102, %c32 : index
              %104 = arith.subi %c0_i32, %arg30 : i32
              %105 = arith.maxsi %104, %c0_i32 : i32
              %106 = arith.index_cast %105 : i32 to index
              %107 = arith.minsi %106, %101 : index
              %108 = affine.apply #map2()[%101, %107]
              %109 = arith.minsi %56, %103 : index
              %110 = affine.apply #map2()[%103, %109]
              %111 = arith.cmpi slt, %108, %c32 : index
              %112 = arith.cmpi slt, %110, %c32 : index
              %113 = arith.ori %111, %112 : i1
              %114 = arith.minsi %10, %108 : index
              %115 = affine.apply #map12()[%114, %101, %107]
              %116 = arith.minsi %115, %c16 : index
              %subview_27 = memref.subview %subview_25[0, 0] [%116, %110] [1, 1] : memref<16x32xi8, strided<[?, 1], offset: ?>> to memref<?x?xi8, strided<[?, 1], offset: ?>>
              %117 = affine.apply #map13()[%arg25]
              %118 = arith.maxsi %107, %10 : index
              %119 = arith.minsi %101, %117 : index
              %120 = arith.maxsi %118, %119 : index
              %121 = affine.apply #map2()[%120, %118]
              %122 = affine.apply #map14()[%118, %arg25]
              %subview_28 = memref.subview %alloc_26[%122, %109] [%121, %110] [1, 1] : memref<16x32xi8> to memref<?x?xi8, strided<[32, 1], offset: ?>>
              %123 = arith.remui %109, %c32 : index
              hivm.hir.anchor {id = 39 : i64}
              scf.if %113 {
                linalg.fill ins(%c0_i8 : i8) outs(%alloc_26 : memref<16x32xi8>)
              } {hivm.unlikely_condition}
              hivm.hir.load ins(%subview_27 : memref<?x?xi8, strided<[?, 1], offset: ?>>) outs(%subview_28 : memref<?x?xi8, strided<[32, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %c0_i8 : i8 left_padding_num = %123 : index
              hivm.hir.anchor {id = 40 : i64}
              %124 = bufferization.to_tensor %alloc_26 restrict writable : memref<16x32xi8>
              %alloc_29 = memref.alloc() : memref<16xf32>
              %125 = arith.maxsi %87, %c0 : index
              %126 = arith.minsi %125, %c32 : index
              %127 = arith.subi %c0_i32, %arg28 : i32
              %128 = arith.maxsi %127, %c0_i32 : i32
              %129 = arith.index_cast %128 : i32 to index
              %130 = arith.minsi %129, %126 : index
              %131 = affine.apply #map2()[%126, %130]
              %132 = arith.cmpi slt, %131, %c32 : index
              %133 = arith.minsi %10, %131 : index
              %134 = affine.apply #map12()[%133, %126, %130]
              %135 = arith.minsi %134, %c16 : index
              %subview_30 = memref.subview %subview_21[0] [%135] [1] : memref<16xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
              %136 = arith.maxsi %130, %10 : index
              %137 = arith.minsi %126, %117 : index
              %138 = arith.maxsi %136, %137 : index
              %139 = affine.apply #map2()[%138, %136]
              %140 = affine.apply #map14()[%136, %arg25]
              %subview_31 = memref.subview %alloc_29[%140] [%139] [1] : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>>
              %141 = arith.remui %130, %c8 : index
              hivm.hir.anchor {id = 41 : i64}
              scf.if %132 {
                linalg.fill ins(%cst_1 : f32) outs(%alloc_29 : memref<16xf32>)
              } {hivm.unlikely_condition}
              hivm.hir.load ins(%subview_30 : memref<?xf32, strided<[?], offset: ?>>) outs(%subview_31 : memref<?xf32, strided<[1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %141 : index
              hivm.hir.anchor {id = 42 : i64}
              %142 = bufferization.to_tensor %alloc_29 restrict writable : memref<16xf32>
              hivm.hir.anchor {id = 43 : i64}
              %143 = tensor.empty() : tensor<16x32xf32>
              %144 = linalg.fill ins(%extracted : f32) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              %145 = linalg.mul ins(%80, %144 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              hivm.hir.anchor {id = 44 : i64}
              %expanded = tensor.expand_shape %142 [[0, 1]] output_shape [16, 1] : tensor<16xf32> into tensor<16x1xf32>
              %146 = tensor.empty() : tensor<16x1xf32>
              %147 = linalg.fill ins(%cst_2 : f32) outs(%146 : tensor<16x1xf32>) -> tensor<16x1xf32>
              %148 = linalg.mul ins(%expanded, %147 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%146 : tensor<16x1xf32>) -> tensor<16x1xf32>
              hivm.hir.anchor {id = 45 : i64}
              %collapsed = tensor.collapse_shape %148 [[0, 1]] : tensor<16x1xf32> into tensor<16xf32>
              %broadcasted = linalg.broadcast ins(%collapsed : tensor<16xf32>) outs(%143 : tensor<16x32xf32>) dimensions = [1]
              %149 = linalg.sub ins(%145, %broadcasted : tensor<16x32xf32>, tensor<16x32xf32>) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              hivm.hir.anchor {id = 46 : i64}
              %150 = linalg.fill ins(%cst_0 : f32) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              %151 = linalg.mul ins(%149, %150 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              hivm.hir.anchor {id = 47 : i64}
              %152 = linalg.exp ins(%151 : tensor<16x32xf32>) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              hivm.hir.anchor {id = 48 : i64}
              %153 = tensor.empty() : tensor<16x32xi1>
              %154 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>} ins(%124, %c0_i8 : tensor<16x32xi8>, i8) outs(%153 : tensor<16x32xi1>) -> tensor<16x32xi1>
              hivm.hir.anchor {id = 49 : i64}
              %155 = linalg.select ins(%154, %152, %cst_1 : tensor<16x32xi1>, tensor<16x32xf32>, f32) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              hivm.hir.anchor {id = 50 : i64}
              %156 = tensor.empty() : tensor<16x32xbf16>
              %157 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%155 : tensor<16x32xf32>) outs(%156 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
              hivm.hir.anchor {id = 51 : i64}
              hivm.hir.anchor {id = 52 : i64}
              %expanded_32 = tensor.expand_shape %157 [[0], [1, 2]] output_shape [16, 2, 16] : tensor<16x32xbf16> into tensor<16x2x16xbf16>
              %158 = tensor.empty() : tensor<2x16x16xbf16>
              %transposed = linalg.transpose ins(%expanded_32 : tensor<16x2x16xbf16>) outs(%158 : tensor<2x16x16xbf16>) permutation = [1, 0, 2]
              %expanded_33 = tensor.expand_shape %transposed [[0], [1, 2], [3]] output_shape [2, 1, 16, 16] : tensor<2x16x16xbf16> into tensor<2x1x16x16xbf16>
              annotation.mark %alloc_15 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, hivm.tiling_dim = 1 : index} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
              %memspacecast_34 = memref.memory_space_cast %alloc_15 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
              %subview_35 = memref.subview %memspacecast_34[0, %arg25, 0, 0] [2, 1, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<2x2x16x16xbf16> to memref<2x1x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
              hivm.hir.anchor {id = 53 : i64}
              hivm.hir.copy ins(%expanded_33 : tensor<2x1x16x16xbf16>) outs(%subview_35 : memref<2x1x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) {"hivm.inserted-copy", tiled_op}
              hivm.hir.anchor {id = 54 : i64}
              %alloc_36 = memref.alloc() : memref<16xf32>
              %159 = arith.maxsi %93, %c0 : index
              %160 = arith.minsi %159, %c32 : index
              %161 = arith.subi %c0_i32, %arg29 : i32
              %162 = arith.maxsi %161, %c0_i32 : i32
              %163 = arith.index_cast %162 : i32 to index
              %164 = arith.minsi %163, %160 : index
              %165 = affine.apply #map2()[%160, %164]
              %166 = arith.cmpi slt, %165, %c32 : index
              %167 = arith.minsi %10, %165 : index
              %168 = affine.apply #map12()[%167, %160, %164]
              %169 = arith.minsi %168, %c16 : index
              %subview_37 = memref.subview %subview_23[0] [%169] [1] : memref<16xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
              %170 = arith.maxsi %164, %10 : index
              %171 = arith.minsi %160, %117 : index
              %172 = arith.maxsi %170, %171 : index
              %173 = affine.apply #map2()[%172, %170]
              %174 = affine.apply #map14()[%170, %arg25]
              %subview_38 = memref.subview %alloc_36[%174] [%173] [1] : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>>
              %175 = arith.remui %164, %c8 : index
              hivm.hir.anchor {id = 55 : i64}
              scf.if %166 {
                linalg.fill ins(%cst_1 : f32) outs(%alloc_36 : memref<16xf32>)
              } {hivm.unlikely_condition}
              hivm.hir.load ins(%subview_37 : memref<?xf32, strided<[?], offset: ?>>) outs(%subview_38 : memref<?xf32, strided<[1], offset: ?>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %175 : index
              hivm.hir.anchor {id = 56 : i64}
              %176 = bufferization.to_tensor %alloc_36 restrict writable : memref<16xf32>
              hivm.hir.anchor {id = 57 : i64}
              %broadcasted_39 = linalg.broadcast ins(%176 : tensor<16xf32>) outs(%143 : tensor<16x32xf32>) dimensions = [1]
              %177 = linalg.sub ins(%81, %broadcasted_39 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              hivm.hir.anchor {id = 58 : i64}
              %178 = linalg.mul ins(%155, %177 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              hivm.hir.anchor {id = 59 : i64}
              %179 = linalg.select ins(%154, %178, %cst_1 : tensor<16x32xi1>, tensor<16x32xf32>, f32) outs(%143 : tensor<16x32xf32>) -> tensor<16x32xf32>
              hivm.hir.anchor {id = 60 : i64}
              %180 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%179 : tensor<16x32xf32>) outs(%156 : tensor<16x32xbf16>) -> tensor<16x32xbf16>
              hivm.hir.anchor {id = 61 : i64}
              annotation.mark %180 {break_vf} : tensor<16x32xbf16>
              hivm.hir.anchor {id = 62 : i64}
              hivm.hir.anchor {id = 63 : i64}
              %expanded_40 = tensor.expand_shape %180 [[0], [1, 2]] output_shape [16, 2, 16] : tensor<16x32xbf16> into tensor<16x2x16xbf16>
              %transposed_41 = linalg.transpose ins(%expanded_40 : tensor<16x2x16xbf16>) outs(%158 : tensor<2x16x16xbf16>) permutation = [1, 0, 2]
              %expanded_42 = tensor.expand_shape %transposed_41 [[0], [1, 2], [3]] output_shape [2, 1, 16, 16] : tensor<2x16x16xbf16> into tensor<2x1x16x16xbf16>
              annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, hivm.tiling_dim = 1 : index} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
              %memspacecast_43 = memref.memory_space_cast %alloc_14 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
              %subview_44 = memref.subview %memspacecast_43[0, %arg25, 0, 0] [2, 1, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<2x2x16x16xbf16> to memref<2x1x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
              hivm.hir.anchor {id = 64 : i64}
              hivm.hir.copy ins(%expanded_42 : tensor<2x1x16x16xbf16>) outs(%subview_44 : memref<2x1x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) {"hivm.inserted-copy", tiled_op}
              hivm.hir.anchor {id = 65 : i64}
              annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, hivm.tiling_dim = 1 : index} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
              %memspacecast_45 = memref.memory_space_cast %alloc_13 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<2x2x16x16xbf16>
              %subview_46 = memref.subview %memspacecast_45[0, %arg25, 0, 0] [2, 1, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<2x2x16x16xbf16> to memref<2x1x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>
              hivm.hir.anchor {id = 66 : i64}
              hivm.hir.copy ins(%expanded_42 : tensor<2x1x16x16xbf16>) outs(%subview_46 : memref<2x1x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>>) {"hivm.inserted-copy", tiled_op}
              hivm.hir.anchor {id = 67 : i64}
              %181 = arith.addi %arg28, %c32_i32 : i32
              %182 = arith.addi %arg29, %c32_i32 : i32
              %183 = arith.addi %arg30, %c32_i32 : i32
              scope.return %181, %182, %183 : i32, i32, i32
            } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
            hivm.hir.anchor {id = 68 : i64}
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
            hivm.hir.anchor {id = 69 : i64}
            scope.scope : () -> () {
              hivm.hir.anchor {id = 70 : i64}
              hivm.hir.anchor {id = 71 : i64}
              hivm.hir.anchor {id = 72 : i64}
              hivm.hir.anchor {id = 73 : i64}
              hivm.hir.anchor {id = 74 : i64}
              hivm.hir.anchor {id = 75 : i64}
              %84 = memref.load %alloca[] : memref<i32>
              hivm.hir.anchor {id = 76 : i64}
              hivm.hir.anchor {id = 77 : i64}
              %85 = arith.addi %84, %c1_i32 : i32
              memref.store %85, %alloca[] : memref<i32>
              hivm.hir.anchor {id = 78 : i64}
              hivm.hir.anchor {id = 79 : i64}
              annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x64xf32, #hivm.address_space<ub>>
              hivm.hir.anchor {id = 80 : i64}
              hivm.hir.anchor {id = 81 : i64}
              %86 = memref.load %alloca_9[] : memref<i32>
              hivm.hir.anchor {id = 82 : i64}
              hivm.hir.anchor {id = 83 : i64}
              %87 = arith.addi %86, %c1_i32 : i32
              memref.store %87, %alloca_9[] : memref<i32>
              hivm.hir.anchor {id = 84 : i64}
              scope.return
            } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
            hivm.hir.anchor {id = 85 : i64}
            %memspacecast_19 = memref.memory_space_cast %alloc_12 : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x64xf32>
            %83 = bufferization.to_tensor %memspacecast_19 restrict writable : memref<16x64xf32>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
            hivm.hir.anchor {id = 86 : i64}
            scope.scope : () -> () {
              hivm.hir.anchor {id = 87 : i64}
              %84 = arith.index_cast %arg19 : i32 to index
              %85 = affine.apply #map9()[%84]
              %86 = arith.extsi %arg27 : i32 to i64
              %87 = arith.addi %25, %86 : i64
              %88 = arith.muli %87, %7 : i64
              %89 = arith.muli %88, %c64_i64 : i64
              %90 = arith.addi %89, %59 : i64
              hivm.hir.anchor {id = 88 : i64}
              %91 = tensor.empty() : tensor<16x64xf32>
              %92 = linalg.fill ins(%arg21 : f32) outs(%91 : tensor<16x64xf32>) -> tensor<16x64xf32>
              %93 = linalg.mul ins(%83, %92 : tensor<16x64xf32>, tensor<16x64xf32>) outs(%91 : tensor<16x64xf32>) -> tensor<16x64xf32>
              hivm.hir.anchor {id = 89 : i64}
              %94 = arith.index_cast %90 : i64 to index
              %reinterpret_cast_20 = memref.reinterpret_cast %arg5 to offset: [%94], sizes: [32, 64], strides: [%85, 1] : memref<?xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
              %subview_21 = memref.subview %reinterpret_cast_20[%10, 0] [16, 64] [1, 1] {to_be_bubbled_slice} : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<16x64xf32, strided<[?, 1], offset: ?>>
              %95 = arith.index_cast %arg27 : i32 to index
              %96 = affine.apply #map10()[%95]
              %97 = arith.index_cast %30 : i32 to index
              %98 = arith.maxsi %95, %97 : index
              %99 = arith.minsi %96, %98 : index
              %100 = affine.apply #map2()[%99, %95]
              %101 = arith.minsi %10, %100 : index
              %102 = affine.apply #map12()[%101, %99, %95]
              %103 = arith.minsi %102, %c16 : index
              %subview_22 = memref.subview %subview_21[0, 0] [%103, 64] [1, 1] : memref<16x64xf32, strided<[?, 1], offset: ?>> to memref<?x64xf32, strided<[?, 1], offset: ?>>
              %extracted_slice_23 = tensor.extract_slice %93[0, 0] [%103, 64] [1, 1] : tensor<16x64xf32> to tensor<?x64xf32>
              hivm.hir.anchor {id = 90 : i64}
              hivm.hir.store ins(%extracted_slice_23 : tensor<?x64xf32>) outs(%subview_22 : memref<?x64xf32, strided<[?, 1], offset: ?>>) {tiled_op} atomic = <add>
              hivm.hir.anchor {id = 91 : i64}
              scope.return
            } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
            hivm.hir.anchor {id = 92 : i64}
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
            scf.yield %82#0, %82#1, %82#2 : i32, i32, i32
          } {fixpipe_for_mmad_result_already_inserted = true}
          hivm.hir.anchor {id = 93 : i64}
          %61 = memref.load %alloca[] : memref<i32>
          hivm.hir.anchor {id = 94 : i64}
          %62 = arith.cmpi eq, %61, %c0_i32 : i32
          %63 = arith.minsi %10, %58 : index
          %64 = affine.apply #map12()[%63, %53, %57]
          %65 = arith.minsi %64, %c16 : index
          %subview_10 = memref.subview %subview_8[0, 0] [%65, 64] [1, 1] : memref<16x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
          scf.if %62 {
            hivm.hir.anchor {id = 95 : i64}
            hivm.hir.anchor {id = 96 : i64}
            hivm.hir.anchor {id = 97 : i64}
            %80 = tensor.empty() : tensor<16x64xbf16>
            %81 = linalg.fill ins(%cst : bf16) outs(%80 : tensor<16x64xbf16>) -> tensor<16x64xbf16>
            hivm.hir.anchor {id = 98 : i64}
            %82 = affine.apply #map13()[%arg25]
            %83 = arith.maxsi %57, %10 : index
            %84 = arith.minsi %53, %82 : index
            %85 = arith.maxsi %83, %84 : index
            %86 = affine.apply #map2()[%85, %83]
            %87 = affine.apply #map14()[%83, %arg25]
            %extracted_slice_12 = tensor.extract_slice %81[%87, 0] [%86, 64] [1, 1] : tensor<16x64xbf16> to tensor<?x64xbf16>
            hivm.hir.anchor {id = 99 : i64}
            hivm.hir.store ins(%extracted_slice_12 : tensor<?x64xbf16>) outs(%subview_10 : memref<?x64xbf16, strided<[?, 1], offset: ?>>) {tiled_op}
            hivm.hir.anchor {id = 100 : i64}
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_FIX>] flag = 0
          } else {
            hivm.hir.anchor {id = 101 : i64}
            hivm.hir.anchor {id = 102 : i64}
            hivm.hir.anchor {id = 103 : i64}
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>] flag = 0
          } {fallback_not_exec}
          hivm.hir.anchor {id = 104 : i64}
          %66 = memref.load %alloca_9[] : memref<i32>
          hivm.hir.anchor {id = 105 : i64}
          %67 = arith.cmpi eq, %66, %c0_i32 : i32
          %alloc = memref.alloc() : memref<16x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<16x64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 106 : i64}
          %68 = scf.if %67 -> (tensor<16x64xf32>) {
            hivm.hir.anchor {id = 107 : i64}
            hivm.hir.anchor {id = 108 : i64}
            %80 = tensor.empty() : tensor<16x64xf32>
            %81 = linalg.fill ins(%cst_1 : f32) outs(%80 : tensor<16x64xf32>) -> tensor<16x64xf32>
            hivm.hir.anchor {id = 109 : i64}
            scf.yield %81 : tensor<16x64xf32>
          } else {
            hivm.hir.anchor {id = 110 : i64}
            %memspacecast = memref.memory_space_cast %alloc : memref<16x64xf32, #hivm.address_space<ub>> to memref<16x64xf32>
            %80 = bufferization.to_tensor %memspacecast restrict writable : memref<16x64xf32>
            hivm.hir.anchor {id = 111 : i64}
            hivm.hir.anchor {id = 112 : i64}
            scf.yield %80 : tensor<16x64xf32>
          } {fallback_not_exec}
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 113 : i64}
          %69 = tensor.empty() : tensor<16x64xf32>
          %70 = linalg.fill ins(%arg21 : f32) outs(%69 : tensor<16x64xf32>) -> tensor<16x64xf32>
          %71 = linalg.mul ins(%68, %70 : tensor<16x64xf32>, tensor<16x64xf32>) outs(%69 : tensor<16x64xf32>) -> tensor<16x64xf32>
          hivm.hir.anchor {id = 114 : i64}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
          %72 = tensor.empty() : tensor<16x64xbf16>
          %73 = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = true, enable_saturate = false, hfusion.unsigned_mode = #hfusion.unsigned_mode<si2si>, round_mode = #hfusion.round_mode<rint>} ins(%71 : tensor<16x64xf32>) outs(%72 : tensor<16x64xbf16>) -> tensor<16x64xbf16>
          hivm.hir.anchor {id = 115 : i64}
          %74 = affine.apply #map13()[%arg25]
          %75 = arith.maxsi %57, %10 : index
          %76 = arith.minsi %53, %74 : index
          %77 = arith.maxsi %75, %76 : index
          %78 = affine.apply #map2()[%77, %75]
          %79 = affine.apply #map14()[%75, %arg25]
          %extracted_slice = tensor.extract_slice %73[%79, 0] [%78, 64] [1, 1] : tensor<16x64xbf16> to tensor<?x64xbf16>
          hivm.hir.anchor {id = 116 : i64}
          %subview_11 = memref.subview %subview[0, 0] [%65, 64] [1, 1] : memref<16x64xbf16, strided<[?, 1], offset: ?>> to memref<?x64xbf16, strided<[?, 1], offset: ?>>
          hivm.hir.store ins(%extracted_slice : tensor<?x64xbf16>) outs(%subview_11 : memref<?x64xbf16, strided<[?, 1], offset: ?>>) {tiled_op}
          hivm.hir.anchor {id = 117 : i64}
        }
        hivm.hir.anchor {id = 118 : i64}
      }
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 119 : i64}
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}
