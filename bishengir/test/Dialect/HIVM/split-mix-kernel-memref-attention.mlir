// RUN: bishengir-opt %s --hivm-v2-split-kernel | FileCheck %s
// RUN: bishengir-opt %s --hivm-v2-split-kernel --canonicalize | FileCheck %s --check-prefix=CANON
// RUN: (bishengir-compile %s --enable-tla-mode --target=Ascend910_9589 --enable-triton-kernel-compile=true --enable-tuning-mode --enable-lir-compile=false --disable-hivm-auto-inject-sync=true --mlir-disable-threading --mlir-print-ir-after=hivm-v2-split-kernel -o %t.ll 2>&1 || true) | FileCheck %s --check-prefix=DRIVER

// Issue: https://gitcode.com/liupengcheng2012/AscendNPU-IR/issues/108
// CHECK-LABEL: func.func @attention_stages_mix_aic(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: hivm.part_of_mix
// CHECK-NOT: memref<8x144xbf16
// CHECK-NOT: get_sub_block_idx
// CHECK: hivm.hir.nd2nz
// CHECK: hivm.required_stage = 0 : i32, hivm.task_name = "load_qk"
// CHECK: hivm.task_name = "load_v"
// CHECK: hivm.hir.mmadL1 {b_transpose}
// CHECK: hivm.required_stage = 1 : i32, hivm.task_name = "qk"
// CHECK: hivm.hir.fixpipe
// CHECK: hivm.required_stage = 1 : i32, hivm.task_name = "score"
// CHECK-NOT: ave.hir
// CHECK-NOT: hivm.hir.copy
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.required_stage = 3 : i32, hivm.task_name = "pv"
// CHECK: hivm.hir.fixpipe
// CHECK: hivm.required_stage = 3 : i32, hivm.task_name = "output"
// CHECK-NOT: hivm.hir.store
// CHECK: hivm.schedule_domain = "auto", hivm.schedule_state = "logical"
// CHECK: return
// CHECK-LABEL: func.func @attention_stages_mix_aiv(
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.part_of_mix
// CHECK-NOT: address_space<cc>
// CHECK: memref.alloc() : memref<8x1x16x16xbf16, #hivm.address_space<cbuf>>
// CHECK: memref.alloc() : memref<8x128xf32, #hivm.address_space<ub>>
// CHECK: memref.alloc() : memref<8x144xbf16, #hivm.address_space<ub>>
// CHECK: memref.alloc() : memref<8x64xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.get_sub_block_idx
// CHECK-NOT: hivm.hir.nd2nz
// CHECK-NOT: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: ave.hir.vload
// CHECK: ave.hir.reduction <max>
// CHECK: ave.hir.reduction <add>
// CHECK: ave.hir.store_with_stride
// CHECK: hivm.required_stage = 2 : i32, hivm.task_name = "softmax"
// CHECK: memref.collapse_shape
// CHECK: memref.subview
// CHECK: hivm.hir.copy
// CHECK: hivm.required_stage = 2 : i32, hivm.task_name = "prob"
// CHECK-NOT: hivm.hir.mmadL1
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: hivm.hir.store
// CHECK: hivm.required_stage = 4 : i32, hivm.task_name = "store"
// CHECK: hivm.schedule_domain = "auto", hivm.schedule_state = "logical"
// CHECK: return

// CANON-LABEL: func.func @attention_stages_mix_aic(
// CANON-NOT: get_sub_block_idx
// CANON: hivm.hir.mmadL1
// CANON: hivm.hir.fixpipe
// CANON-NOT: ave.hir
// CANON-NOT: hivm.hir.copy
// CANON-NOT: hivm.hir.store
// CANON-LABEL: func.func @attention_stages_mix_aiv(
// CANON-NOT: address_space<cc>
// CANON: ave.hir.vload
// CANON: ave.hir.store_with_stride
// CANON: hivm.hir.copy
// CANON-NOT: hivm.hir.mmadL1
// CANON-NOT: hivm.hir.fixpipe
// CANON: hivm.hir.store

// Check the affinity split before optional device-code generation.
// DRIVER: IR Dump After V2SplitKernel (hivm-v2-split-kernel)
// DRIVER: func.func @attention_stages_mix_aic(
// DRIVER: hivm.hir.mmadL1
// DRIVER: hivm.hir.fixpipe
// DRIVER-NOT: ave.hir
// DRIVER: func.func @attention_stages_mix_aiv(
// DRIVER: ave.hir.vload
// DRIVER: ave.hir.store_with_stride
// DRIVER: hivm.hir.copy
// DRIVER: hivm.hir.store

module attributes {hacc.mode = "tla", hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @attention_stages(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16>, %arg3: memref<?xbf16>, %arg4: memref<?xbf16>, %arg5: memref<?xf32>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c512_i64 = arith.constant 512 : i64
    %c128_i64 = arith.constant 128 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c144 = arith.constant 144 : index
    %cst = arith.constant 1.250000e-01 : f32
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %c16_i32 = arith.constant 16 : i32
    scope.scope : () -> () {
      scf.for %arg12 = %c0_i32 to %c3_i32 step %c1_i32  : i32 {
        %alloc = memref.alloc() : memref<4x1x16x16xbf16, #hivm.address_space<cbuf>>
        %alloc_0 = memref.alloc() : memref<4x8x16x16xbf16, #hivm.address_space<cbuf>>
        %alloc_1 = memref.alloc() : memref<4x8x16x16xbf16, #hivm.address_space<cbuf>>
        %alloc_2 = memref.alloc() : memref<8x1x16x16xbf16, #hivm.address_space<cbuf>>
        %alloc_3 = memref.alloc() : memref<8x1x16x16xf32, #hivm.address_space<cc>>
        %alloc_4 = memref.alloc() : memref<4x1x16x16xf32, #hivm.address_space<cc>>
        %alloc_5 = memref.alloc() : memref<8x128xf32, #hivm.address_space<ub>>
        %alloc_6 = memref.alloc() : memref<8x144xbf16, #hivm.address_space<ub>>
        %alloc_7 = memref.alloc() : memref<8x64xf32, #hivm.address_space<ub>>
        %0 = hivm.hir.get_sub_block_idx -> i64
        scope.scope : () -> () {
          %1 = arith.muli %arg12, %c1024_i32 : i32
          %2 = arith.index_cast %1 : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%2], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
          %intptr = memref.extract_aligned_pointer_as_index %reinterpret_cast : memref<1xbf16, strided<[1], offset: ?>> -> index
          %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %reinterpret_cast : memref<1xbf16, strided<[1], offset: ?>> -> memref<bf16>, index, index, index
          %3 = arith.muli %offset, %c2 : index
          %4 = arith.addi %intptr, %3 : index
          %5 = arith.index_cast %4 : index to i64
          %c64_8 = arith.constant 64 : index
          %6 = arith.index_cast %c0 : index to i64
          %c2_i64 = arith.constant 2 : i64
          %7 = arith.muli %6, %c2_i64 : i64
          %8 = arith.addi %5, %7 : i64
          %9 = hivm.hir.pointer_cast(%8) [%c64_8] : memref<?xbf16, #hivm.address_space<gm>>
          annotation.mark %9 {address_space = #hivm.address_space<gm>} : memref<?xbf16, #hivm.address_space<gm>>
          %reinterpret_cast_9 = memref.reinterpret_cast %9 to offset: [0], sizes: [16, 64], strides: [64, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<16x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_9 : memref<16x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<4x1x16x16xbf16, #hivm.address_space<cbuf>>)
          %intptr_10 = memref.extract_aligned_pointer_as_index %arg3 : memref<?xbf16> -> index
          %10 = arith.index_cast %intptr_10 : index to i64
          %c64_11 = arith.constant 64 : index
          %11 = arith.index_cast %c0 : index to i64
          %c2_i64_12 = arith.constant 2 : i64
          %12 = arith.muli %11, %c2_i64_12 : i64
          %13 = arith.addi %10, %12 : i64
          %14 = hivm.hir.pointer_cast(%13) [%c64_11] : memref<?xbf16, #hivm.address_space<gm>>
          annotation.mark %14 {address_space = #hivm.address_space<gm>} : memref<?xbf16, #hivm.address_space<gm>>
          %reinterpret_cast_13 = memref.reinterpret_cast %14 to offset: [0], sizes: [128, 64], strides: [64, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_13 : memref<128x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_0 : memref<4x8x16x16xbf16, #hivm.address_space<cbuf>>)
          scope.return
        } {hivm.required_stage = 0 : i32, hivm.task_name = "load_qk", hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %intptr = memref.extract_aligned_pointer_as_index %arg4 : memref<?xbf16> -> index
          %1 = arith.index_cast %intptr : index to i64
          %c64_8 = arith.constant 64 : index
          %2 = arith.index_cast %c0 : index to i64
          %c2_i64 = arith.constant 2 : i64
          %3 = arith.muli %2, %c2_i64 : i64
          %4 = arith.addi %1, %3 : i64
          %5 = hivm.hir.pointer_cast(%4) [%c64_8] : memref<?xbf16, #hivm.address_space<gm>>
          annotation.mark %5 {address_space = #hivm.address_space<gm>} : memref<?xbf16, #hivm.address_space<gm>>
          %reinterpret_cast = memref.reinterpret_cast %5 to offset: [0], sizes: [128, 64], strides: [64, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_1 : memref<4x8x16x16xbf16, #hivm.address_space<cbuf>>)
          scope.return
        } {hivm.task_name = "load_v", hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          hivm.hir.mmadL1 {b_transpose} ins(%alloc, %alloc_0, %true, %c16, %c64, %c128 : memref<4x1x16x16xbf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_3 : memref<8x1x16x16xf32, #hivm.address_space<cc>>)
          scope.return
        } {hivm.required_stage = 1 : i32, hivm.task_name = "qk", hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          hivm.hir.fixpipe {c0_pad_en = true, dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_3 : memref<8x1x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_5 : memref<8x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          scope.return
        } {hivm.required_stage = 1 : i32, hivm.task_name = "score", hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          scf.for %arg13 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
            %1 = arith.index_cast %arg13 : i32 to index
            %res = ave.hir.vload <NORM> %alloc_5[%1, %c0] : memref<8x128xf32, #hivm.address_space<ub>> into vector<64xf32>
            %2 = ave.hir.pge <ALL> : vector<64xi1>
            %3 = ave.hir.vmuls %res, %cst, %2 : vector<64xf32>, f32, vector<64xi1>
            %res_8 = ave.hir.vload <NORM> %alloc_5[%1, %c64] : memref<8x128xf32, #hivm.address_space<ub>> into vector<64xf32>
            %4 = ave.hir.vmuls %res_8, %cst, %2 : vector<64xf32>, f32, vector<64xi1>
            %5 = ave.hir.vmax %3, %4, %2 : vector<64xf32>, vector<64xi1>
            %6 = ave.hir.reduction <max>, %5, %2 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
            %7 = ave.hir.vector_broadcast %6, %2, true : vector<64xf32>, vector<64xi1> -> vector<64xf32>
            %8 = ave.hir.vsub %3, %7, %2 : vector<64xf32>, vector<64xi1>
            %9 = ave.hir.vexp %8, %2 : vector<64xf32>, vector<64xi1>
            %10 = ave.hir.vsub %4, %7, %2 : vector<64xf32>, vector<64xi1>
            %11 = ave.hir.vexp %10, %2 : vector<64xf32>, vector<64xi1>
            %12 = ave.hir.vadd %9, %11, %2 : vector<64xf32>, vector<64xi1>
            %13 = ave.hir.reduction <add>, %12, %2 : vector<64xf32>, vector<64xi1> -> vector<64xf32>
            %14 = ave.hir.vector_broadcast %13, %2, true : vector<64xf32>, vector<64xi1> -> vector<64xf32>
            %15 = ave.hir.vdiv %9, %14, %2 : vector<64xf32>, vector<64xi1>
            %16 = ave.hir.vtruncf %15, <round>, true, <part_even>, %2 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %17 = ave.hir.vdiv %11, %14, %2 : vector<64xf32>, vector<64xi1>
            %18 = ave.hir.vtruncf %17, <round>, true, <part_even>, %2 : vector<64xf32>, vector<128xbf16>, vector<64xi1>
            %res1, %res2 = ave.hir.vdintlv %16, %18 {layout_change = #ave<layout_change SPARSE>} : vector<128xbf16>, vector<128xbf16>
            %19 = arith.muli %arg13, %c16_i32 : i32
            %20 = ave.hir.pge <ALL> : vector<128xi1>
            %21 = arith.index_cast %19 : i32 to index
            ave.hir.store_with_stride %alloc_6[%c0, %21], %c144, %20, %res1 : memref<8x144xbf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xbf16>
          }
          scope.return
        } {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.required_stage = 2 : i32, hivm.task_name = "softmax", hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vector_function, hivm.vf_mode = #hivm.vf_mode<SIMD>, no_inline, outline, vector_mode = "vec"}
        scope.scope : () -> () {
          %subview = memref.subview %alloc_6[0, 0] [8, 128] [1, 1] : memref<8x144xbf16, #hivm.address_space<ub>> to memref<8x128xbf16, strided<[144, 1]>, #hivm.address_space<ub>>
          %1 = arith.muli %0, %c128_i64 : i64
          %collapse_shape = memref.collapse_shape %alloc_2 [[0], [1, 2, 3]] : memref<8x1x16x16xbf16, #hivm.address_space<cbuf>> into memref<8x256xbf16, #hivm.address_space<cbuf>>
          %2 = arith.index_cast %1 : i64 to index
          %subview_8 = memref.subview %collapse_shape[0, %2] [8, 128] [1, 1] : memref<8x256xbf16, #hivm.address_space<cbuf>> to memref<8x128xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%subview : memref<8x128xbf16, strided<[144, 1]>, #hivm.address_space<ub>>) outs(%subview_8 : memref<8x128xbf16, strided<[256, 1], offset: ?>, #hivm.address_space<cbuf>>)
          scope.return
        } {hivm.required_stage = 2 : i32, hivm.task_name = "prob", hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        scope.scope : () -> () {
          hivm.hir.mmadL1 ins(%alloc_2, %alloc_1, %true, %c16, %c128, %c64 : memref<8x1x16x16xbf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_4 : memref<4x1x16x16xf32, #hivm.address_space<cc>>)
          scope.return
        } {hivm.required_stage = 3 : i32, hivm.task_name = "pv", hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          hivm.hir.fixpipe {c0_pad_en = true, dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_4 : memref<4x1x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_7 : memref<8x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          scope.return
        } {hivm.required_stage = 3 : i32, hivm.task_name = "output", hivm.tcore_type = #hivm.tcore_type<CUBE>}
        scope.scope : () -> () {
          %1 = arith.muli %arg12, %c1024_i32 : i32
          %2 = arith.index_cast %1 : i32 to index
          %3 = arith.muli %0, %c512_i64 : i64
          %4 = arith.index_cast %3 : i64 to index
          %5 = arith.addi %2, %4 : index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%5], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1], offset: ?>>
          %intptr = memref.extract_aligned_pointer_as_index %reinterpret_cast : memref<1xf32, strided<[1], offset: ?>> -> index
          %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %reinterpret_cast : memref<1xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
          %6 = arith.muli %offset, %c4 : index
          %7 = arith.addi %intptr, %6 : index
          %8 = arith.index_cast %7 : index to i64
          %c64_8 = arith.constant 64 : index
          %9 = arith.index_cast %c0 : index to i64
          %c4_i64 = arith.constant 4 : i64
          %10 = arith.muli %9, %c4_i64 : i64
          %11 = arith.addi %8, %10 : i64
          %12 = hivm.hir.pointer_cast(%11) [%c64_8] : memref<?xf32, #hivm.address_space<gm>>
          annotation.mark %12 {address_space = #hivm.address_space<gm>} : memref<?xf32, #hivm.address_space<gm>>
          %reinterpret_cast_9 = memref.reinterpret_cast %12 to offset: [0], sizes: [8, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.store ins(%alloc_7 : memref<8x64xf32, #hivm.address_space<ub>>) outs(%reinterpret_cast_9 : memref<8x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>)
          scope.return
        } {hivm.required_stage = 4 : i32, hivm.task_name = "store", hivm.tcore_type = #hivm.tcore_type<VECTOR>}
      }
      scope.return
    } {hivm.schedule_domain = "auto", hivm.schedule_state = "logical"}
    return
  }
}
