// RUN: bishengir-opt -hivm-cross-core-gss --debug-only=hivm-cross-core-gss -split-input-file %s 2>&1 | FileCheck %s

// CHECK: after:
// CHECK-NEXT: Function0 {
// Function0 {
//   Scope1 {
//     FunctionBlock2 {
//       PlaceHolder3
//       %8 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>4 [<VECTOR>] [<PIPE_S>] 
//       %11 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%10 : tensor<128xf32>) -> tensor<128xf32>5 [<VECTOR>] [<PIPE_S>] 
//       %12 = hivm.hir.vbrc ins(%cst : f32) outs(%10 : tensor<128xf32>) -> tensor<128xf32>6 [<VECTOR>] [<PIPE_S>] 
//       PlaceHolder81
//       CHECK: SetFlagOp{{.*}} [<VECTOR>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID8, EVENT_ID9, EVENT_ID10, EVENT_ID11)] all-at-once
//       CHECK: SetFlagOp{{.*}} [<CUBE>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7)] all-at-once
//       CHECK-NEXT: Loop7 {
//       Loop7 {
//         Scope8 {
//           PlaceHolder9
//           %30 = hivm.hir.load ins(%28 : tensor<128x128xf16>) outs(%29 : tensor<128x128xf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf16>10 [<CUBE>] [<PIPE_MTE2>] 
//             read: MemInfo(<block argument> of type 'memref<?xf16>' at index: 3)
//           PlaceHolder72
//           Loop11 {
//             Scope12 {
//               PlaceHolder13
//               CHECK: Scope14 max-preload-num=4 preload-num=3 {
//                 Scope15 {
//                   PlaceHolder16
//                   CHECK: WaitFlagOp{{.*}} [<CUBE>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID8, EVENT_ID9, EVENT_ID10, EVENT_ID11)]
//                   CHECK-NEXT: PlaceHolder24
//                   Loop17 {
//                     Scope18 {
//                       PlaceHolder19
//                       %84 = hivm.hir.load ins(%82 : tensor<128x128xf16>) outs(%83 : tensor<128x128xf16>) {cube_producer_to_fuse_0, hivm.tcore_type = #hivm.tcore_type<CUBE>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf16>20 [<CUBE>] [<PIPE_MTE2>] 
//                         read: MemInfo(<block argument> of type 'memref<?xf16>' at index: 4)
//                       %86 = hivm.hir.mmadL1 {b_transpose, cube_producer_to_fuse_0, fixpipe_already_inserted = true} ins(%30, %84, %true, %c128, %c128, %85 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index) outs(%extracted_slice : tensor<128x128xf32>) -> tensor<128x128xf32>21 [<CUBE>] [<PIPE_MTE1>, <PIPE_MTE1>] 
//                       hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_0} ins(%86 : tensor<128x128xf32>) outs(%subview_15 : memref<128x128xf32, strided<[512, 1], offset: ?>>)22 [<CUBE>] [<PIPE_FIX>] 
//                         write: MemInfo(%70 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<4x128x512xf32>, PointerLikeInfo(gm, [0], 8388608))
//                       PlaceHolder23
//                     }
//                   }
//                   PlaceHolder25
//                   CHECK: SetFlagOp{{.*}} [<CUBE>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID12)]
//                   CHECK-NEXT: PlaceHolder26
//                 }
//               }
//               CHECK: Scope27 max-preload-num=4 preload-num=2 {
//                 Scope28 {
//                   PlaceHolder29
//                   CHECK: WaitFlagOp{{.*}} [<VECTOR>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7)]
//                   CHECK: WaitFlagOp{{.*}} [<VECTOR>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID12)]
//                   CHECK-NEXT: PlaceHolder47
//                   Loop30 {
//                     Scope31 {
//                       PlaceHolder32
//                       %81 = hivm.hir.load ins(%extracted_slice_13 : tensor<32x512xf32>) outs(%extracted_slice_14 : tensor<32x512xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, vector_producer_to_fuse_1} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<32x512xf32>33 [<VECTOR>] [<PIPE_MTE2>] 
//                         read: MemInfo(%70 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<4x128x512xf32>, PointerLikeInfo(gm, [0], 8388608))
//                       %82 = hivm.hir.vmul {vector_producer_to_fuse_1} ins(%81, %cst_1 : tensor<32x512xf32>, f32) outs(%extracted_slice_14 : tensor<32x512xf32>) -> tensor<32x512xf32>34 [<VECTOR>] [<PIPE_V>] 
//                       %83 = hivm.hir.vreduce {vector_producer_to_fuse_1} <max> ins(%82 : tensor<32x512xf32>) outs(%extracted_slice_15 : tensor<32x1xf32>) reduce_dims = [1] -> tensor<32x1xf32>35 [<VECTOR>] [<PIPE_V>] 
//                       %84 = hivm.hir.vmax {vector_producer_to_fuse_1} ins(%extracted_slice_12, %collapsed : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_17 : tensor<32xf32>) -> tensor<32xf32>36 [<VECTOR>] [<PIPE_V>] 
//                       %85 = hivm.hir.vsub {vector_producer_to_fuse_1} ins(%82, %expanded_18 : tensor<32x512xf32>, tensor<32x1xf32>) outs(%extracted_slice_14 : tensor<32x512xf32>) broadcast = [1] -> tensor<32x512xf32>37 [<VECTOR>] [<PIPE_V>] 
//                       %86 = hivm.hir.vexp {vector_producer_to_fuse_1} ins(%85 : tensor<32x512xf32>) outs(%extracted_slice_14 : tensor<32x512xf32>) -> tensor<32x512xf32>38 [<VECTOR>] [<PIPE_V>] 
//                       %87 = hivm.hir.vcast {vector_producer_to_fuse_1} ins(%86 : tensor<32x512xf32>) outs(%extracted_slice_19 : tensor<32x512xf16>) -> tensor<32x512xf16>39 [<VECTOR>] [<PIPE_V>] 
//                       hivm.hir.store ins(%87 : tensor<32x512xf16>) outs(%subview_20 : memref<32x512xf16, strided<[512, 1], offset: ?>>) {op_to_tile_1_0}40 [<VECTOR>] [<PIPE_MTE3>] 
//                         write: MemInfo(%68 = memref_ext.alloc_workspace() from %arg2 offset = [%c1048576] : from memref<?xi8> to memref<4x128x512xf16>, PointerLikeInfo(gm, [8388608], 4194304))
//                       %88 = hivm.hir.vsub {vector_producer_to_fuse_1} ins(%extracted_slice_12, %84 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_16 : tensor<32xf32>) -> tensor<32xf32>41 [<VECTOR>] [<PIPE_V>] 
//                       %89 = hivm.hir.vexp {vector_producer_to_fuse_1} ins(%88 : tensor<32xf32>) outs(%extracted_slice_21 : tensor<32xf32>) -> tensor<32xf32>42 [<VECTOR>] [<PIPE_V>] 
//                       %90 = hivm.hir.vmul {vector_producer_to_fuse_1} ins(%extracted_slice_23, %89 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_16 : tensor<32xf32>) -> tensor<32xf32>43 [<VECTOR>] [<PIPE_V>] 
//                       %91 = hivm.hir.vreduce {vector_producer_to_fuse_1} <sum> ins(%86 : tensor<32x512xf32>) outs(%extracted_slice_15 : tensor<32x1xf32>) reduce_dims = [1] -> tensor<32x1xf32>44 [<VECTOR>] [<PIPE_V>] 
//                       %92 = hivm.hir.vadd {vector_producer_to_fuse_1} ins(%90, %collapsed_24 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_25 : tensor<32xf32>) -> tensor<32xf32>45 [<VECTOR>] [<PIPE_V>] 
//                       PlaceHolder46
//                     }
//                   }
//                   CHECK: PlaceHolder48
//                   CHECK-NEXT: SetFlagOp{{.*}} [<VECTOR>, <PIPE_MTE3>, <PIPE_S>, (EVENT_ID13)]
//                   CHECK-NEXT: SetFlagOp{{.*}} [<VECTOR>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID8, EVENT_ID9, EVENT_ID10, EVENT_ID11)]
//                   PlaceHolder49
//                 }
//               }
//               CHECK: Scope50 max-preload-num=4 preload-num=1 {
//                 Scope51 {
//                   PlaceHolder52
//                   %83 = hivm.hir.load ins(%81 : tensor<512x128xf16>) outs(%82 : tensor<512x128xf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<512x128xf16>53 [<CUBE>] [<PIPE_MTE2>] 
//                     read: MemInfo(<block argument> of type 'memref<?xf16>' at index: 5)
//                   CHECK: WaitFlagOp{{.*}} [<CUBE>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3)]
//                   CHECK: WaitFlagOp{{.*}} [<CUBE>, <PIPE_MTE3>, <PIPE_S>, (EVENT_ID13)]
//                   CHECK-NEXT: PlaceHolder61
//                   Loop54 {
//                     Scope55 {
//                       PlaceHolder56
//                       %86 = hivm.hir.load ins(%extracted_slice_14 : tensor<64x512xf16>) outs(%extracted_slice_15 : tensor<64x512xf16>) {cube_producer_to_fuse_2, hivm.tcore_type = #hivm.tcore_type<CUBE>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<64x512xf16>57 [<CUBE>] [<PIPE_MTE2>] 
//                         read: MemInfo(%68 = memref_ext.alloc_workspace() from %arg2 offset = [%c1048576] : from memref<?xi8> to memref<4x128x512xf16>, PointerLikeInfo(gm, [8388608], 4194304))
//                       %88 = hivm.hir.mmadL1 {cube_producer_to_fuse_2, fixpipe_already_inserted = true, hivm.tile_mix_cube_num = 2 : i32} ins(%86, %83, %true, %87, %c512, %c128 : tensor<64x512xf16>, tensor<512x128xf16>, i1, index, index, index) outs(%extracted_slice_16 : tensor<64x128xf32>) -> tensor<64x128xf32>58 [<CUBE>] [<PIPE_MTE1>, <PIPE_MTE1>] 
//                       hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_2_0} ins(%88 : tensor<64x128xf32>) outs(%subview_17 : memref<64x128xf32, strided<[128, 1], offset: ?>>)59 [<CUBE>] [<PIPE_FIX>] 
//                         write: MemInfo(%69 = memref_ext.alloc_workspace() from %arg2 offset = [%c1572864] : from memref<?xi8> to memref<4x128x128xf32>, PointerLikeInfo(gm, [12582912], 2097152))
//                       PlaceHolder60
//                     }
//                   }
//                   CHECK: PlaceHolder62
//                   CHECK-NEXT: SetFlagOp{{.*}} [<CUBE>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID12)]
//                   CHECK-NEXT: SetFlagOp{{.*}} [<CUBE>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7)]
//                   PlaceHolder63
//                 }
//               }
//               CHECK: Scope64 max-preload-num=4 preload-num=0 {
//                 Scope65 {
//                   PlaceHolder66
//                   CHECK: WaitFlagOp{{.*}} [<VECTOR>, <PIPE_FIX>, <PIPE_S>, (EVENT_ID12)]
//                   CHECK-NEXT: {{.*}}hivm.hir.load ins(%extracted_slice : tensor<128x128xf32>){{.*}}
//                   %78 = hivm.hir.load ins(%extracted_slice : tensor<128x128xf32>) outs(%7 : tensor<128x128xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf32>67 [<VECTOR>] [<PIPE_MTE2>] 
//                     read: MemInfo(%69 = memref_ext.alloc_workspace() from %arg2 offset = [%c1572864] : from memref<?xi8> to memref<4x128x128xf32>, PointerLikeInfo(gm, [12582912], 2097152))
//                   CHECK: SetFlagOp{{.*}} [<VECTOR>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3)]
//                   %79 = hivm.hir.vmul ins(%arg14, %expanded_12 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>68 [<VECTOR>] [<PIPE_V>] 
//                   %80 = hivm.hir.vadd ins(%79, %78 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>69 [<VECTOR>] [<PIPE_V>] 
//                   CHECK: PlaceHolder70
//                 }
//               }
//               PlaceHolder71
//             }
//           }
//           PlaceHolder73
//           %32 = hivm.hir.vln ins(%31#0 : tensor<128xf32>) outs(%10 : tensor<128xf32>) -> tensor<128xf32>74 [<VECTOR>] [<PIPE_V>] 
//           %33 = hivm.hir.vadd ins(%31#2, %32 : tensor<128xf32>, tensor<128xf32>) outs(%10 : tensor<128xf32>) -> tensor<128xf32>75 [<VECTOR>] [<PIPE_V>] 
//           %34 = hivm.hir.vdiv ins(%31#1, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>76 [<VECTOR>] [<PIPE_V>] 
//           hivm.hir.store ins(%33 : tensor<128xf32>) outs(%reinterpret_cast_4 : memref<128xf32, strided<[1], offset: ?>>)77 [<VECTOR>] [<PIPE_MTE3>] 
//             write: MemInfo(<block argument> of type 'memref<?xf32>' at index: 6)
//           %40 = hivm.hir.vcast ins(%34 : tensor<128x128xf32>) outs(%39 : tensor<128x128xf16>) -> tensor<128x128xf16>78 [<VECTOR>] [<PIPE_V>] 
//           hivm.hir.store ins(%40 : tensor<128x128xf16>) outs(%reinterpret_cast_3 : memref<128x128xf16, strided<[128, 1], offset: ?>>)79 [<VECTOR>] [<PIPE_MTE3>] 
//             write: MemInfo(<block argument> of type 'memref<?xf16>' at index: 7)
//           PlaceHolder80
//         }
//       }
//       CHECK: WaitFlagOp{{.*}} [<VECTOR>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7)] all-at-once
//       CHECK-NEXT: WaitFlagOp{{.*}} [<CUBE>, <PIPE_MTE2>, <PIPE_S>, (EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID8, EVENT_ID9, EVENT_ID10, EVENT_ID11)] all-at-once
//       CHECK-NEXT: PlaceHolder82
//       PlaceHolder83
//     }
//   }
// }

// CHECK: func.func @_attn_fwd{{.*}}
func.func @_attn_fwd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, false, false, false]> : vector<11xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %c1572864 = arith.constant 1572864 : index
  %c1048576 = arith.constant 1048576 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %true = arith.constant true
  %c128 = arith.constant 128 : index
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %cst_1 = arith.constant 5.000000e-01 : f32
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c8388608_i64 = arith.constant 8388608 : i64
  %c1048576_i64 = arith.constant 1048576 : i64
  %c128_i32 = arith.constant 128 : i32
  %c8192_i32 = arith.constant 8192 : i32
  %c512_i32 = arith.constant 512 : i32
  %c20_i32 = arith.constant 20 : i32
  %cst_2 = arith.constant 0.000000e+00 : f32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg8, %arg9 : i32
  %1 = arith.muli %0, %arg10 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg10, %arg9 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg8 : i32
  %7 = tensor.empty() : tensor<128x128xf32>
  %8 = hivm.hir.vbrc ins(%cst_2 : f32) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %9 = tensor.empty() : tensor<128x512xf32>
  %10 = tensor.empty() : tensor<128xf32>
  %11 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%10 : tensor<128xf32>) -> tensor<128xf32>
  %12 = hivm.hir.vbrc ins(%cst : f32) outs(%10 : tensor<128xf32>) -> tensor<128xf32>
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 10
  // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 11
  // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  // CHECK-NEXT: scf.for %arg11{{.*}}
  scf.for %arg11 = %6 to %c512_i32 step %c20_i32  : i32 {
    %13 = arith.divsi %arg11, %c64_i32 : i32
    %14 = arith.remsi %arg11, %c64_i32 : i32
    %15 = arith.divsi %13, %c8_i32 : i32
    %16 = arith.remsi %13, %c8_i32 : i32
    %17 = arith.extsi %15 : i32 to i64
    %18 = arith.muli %17, %c8388608_i64 : i64
    %19 = arith.extsi %16 : i32 to i64
    %20 = arith.muli %19, %c1048576_i64 : i64
    %21 = arith.addi %18, %20 : i64
    %22 = arith.index_cast %21 : i64 to index
    %23 = arith.muli %14, %c128_i32 : i32
    %24 = arith.maxsi %23, %c0_i32 : i32
    %25 = arith.index_cast %24 : i32 to index
    %26 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%25]
    %27 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%26, %22]
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg7 to offset: [%27], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xf16>
    %28 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<128x128xf16, strided<[128, 1], offset: ?>>
    %29 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16>
    %30 = hivm.hir.load ins(%28 : tensor<128x128xf16>) outs(%29 : tensor<128x128xf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf16>
    %31:5 = scf.for %arg12 = %c0_i32 to %c8192_i32 step %c512_i32 iter_args(%arg13 = %12, %arg14 = %8, %arg15 = %11, %arg16 = %c0_i32, %arg17 = %c0_i32) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32)  : i32 {
      %41 = memref_ext.alloc_workspace() from %arg2 offset = [%c1048576] : from memref<?xi8> to memref<4x128x512xf16>
      %42 = memref_ext.alloc_workspace() from %arg2 offset = [%c1572864] : from memref<?xi8> to memref<4x128x128xf32>
      %43 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8> to memref<4x128x512xf32>
      annotation.mark %43 : memref<4x128x512xf32>
      annotation.mark %41 : memref<4x128x512xf16>
      annotation.mark %42 : memref<4x128x128xf32>
      %44 = scope.scope : () -> i32 {
        %51 = arith.index_cast %arg17 : i32 to index
        %52 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%51]
        %53 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%52, %22]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [512, 128], strides: [128, 1] : memref<?xf16> to memref<512x128xf16, strided<[128, 1], offset: ?>>
        %subview = memref.subview %43[0, 0, 0] [1, 128, 512] [1, 1, 1] {hivm.preload_workspace} : memref<4x128x512xf32> to memref<1x128x512xf32, strided<[65536, 512, 1]>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x512xf32, strided<[65536, 512, 1]>> into memref<128x512xf32, strided<[512, 1]>>
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = [[EVENT10:%[0-9]]]
        // CHECK-NEXT: {{.*}}scf.for %arg18{{.*}}
        scf.for %arg18 = %c0 to %c512 step %c128 {
          %subview_6 = memref.subview %reinterpret_cast_5[%arg18, 0] [128, 128] [1, 1] : memref<512x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %55 = bufferization.to_tensor %subview_6 restrict writable : memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_7 = memref.alloc() : memref<128x128xf16>
          %56 = bufferization.to_tensor %alloc_7 restrict writable : memref<128x128xf16>
          %57 = hivm.hir.load ins(%55 : tensor<128x128xf16>) outs(%56 : tensor<128x128xf16>) {cube_producer_to_fuse_0, hivm.tcore_type = #hivm.tcore_type<CUBE>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf16>
          %extracted_slice = tensor.extract_slice %9[0, %arg18] [128, 128] [1, 1] : tensor<128x512xf32> to tensor<128x128xf32>
          %58 = affine.min affine_map<(d0) -> (-d0 + 512, 128)>(%arg18)
          %59 = hivm.hir.mmadL1 {b_transpose, cube_producer_to_fuse_0, fixpipe_already_inserted = true} ins(%30, %57, %true, %c128, %c128, %58 : tensor<128x128xf16>, tensor<128x128xf16>, i1, index, index, index) outs(%extracted_slice : tensor<128x128xf32>) -> tensor<128x128xf32>
          %subview_8 = memref.subview %collapse_shape[0, %arg18] [128, 128] [1, 1] : memref<128x512xf32, strided<[512, 1]>> to memref<128x128xf32, strided<[512, 1], offset: ?>>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_0_0} ins(%59 : tensor<128x128xf32>) outs(%subview_8 : memref<128x128xf32, strided<[512, 1], offset: ?>>)
        }
        // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = [[EVENT0:[0-9]+]]
        %54 = arith.addi %arg17, %c512_i32 : i32
        scope.return %54 : i32
      // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
      %45 = bufferization.to_tensor %43 restrict : memref<4x128x512xf32>
      %46:3 = scope.scope : () -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
        %51 = tensor.empty() : tensor<128x512xf16>
        %extracted_slice = tensor.extract_slice %45[0, 0, 0] [1, 128, 512] [1, 1, 1] {hivm.preload_workspace} : tensor<4x128x512xf32> to tensor<128x512xf32>
        %52 = tensor.empty() : tensor<128x1xf32>
        %subview = memref.subview %41[0, 0, 0] [1, 128, 512] [1, 1, 1] {hivm.preload_workspace} : memref<4x128x512xf16> to memref<1x128x512xf16, strided<[65536, 512, 1]>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x512xf16, strided<[65536, 512, 1]>> into memref<128x512xf16, strided<[512, 1]>>
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = [[EVENT11:%[0-9]]]
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = [[EVENT0]]
        // CHECK-NEXT: {{.*}}scf.for %arg18{{.*}}
        %53:3 = scf.for %arg18 = %c0 to %c128 step %c32 iter_args(%arg19 = %10, %arg20 = %10, %arg21 = %10) -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
          %extracted_slice_5 = tensor.extract_slice %arg15[%arg18] [32] [1] : tensor<128xf32> to tensor<32xf32>
          %extracted_slice_6 = tensor.extract_slice %extracted_slice[%arg18, 0] [32, 512] [1, 1] : tensor<128x512xf32> to tensor<32x512xf32>
          %extracted_slice_7 = tensor.extract_slice %9[%arg18, 0] [32, 512] [1, 1] : tensor<128x512xf32> to tensor<32x512xf32>
          %54 = hivm.hir.load ins(%extracted_slice_6 : tensor<32x512xf32>) outs(%extracted_slice_7 : tensor<32x512xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>, vector_producer_to_fuse_1} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<32x512xf32>
          %55 = hivm.hir.vmul {vector_producer_to_fuse_1} ins(%54, %cst_1 : tensor<32x512xf32>, f32) outs(%extracted_slice_7 : tensor<32x512xf32>) -> tensor<32x512xf32>
          %extracted_slice_8 = tensor.extract_slice %52[%arg18, 0] [32, 1] [1, 1] : tensor<128x1xf32> to tensor<32x1xf32>
          %56 = hivm.hir.vreduce {vector_producer_to_fuse_1} <max> ins(%55 : tensor<32x512xf32>) outs(%extracted_slice_8 : tensor<32x1xf32>) reduce_dims = [1] -> tensor<32x1xf32>
          %collapsed = tensor.collapse_shape %56 [[0, 1]] {vector_producer_to_fuse_1} : tensor<32x1xf32> into tensor<32xf32>
          %extracted_slice_9 = tensor.extract_slice %10[%arg18] [32] [1] : tensor<128xf32> to tensor<32xf32>
          %extracted_slice_10 = tensor.extract_slice %arg19[%arg18] [32] [1] : tensor<128xf32> to tensor<32xf32>
          %57 = hivm.hir.vmax {vector_producer_to_fuse_1} ins(%extracted_slice_5, %collapsed : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_10 : tensor<32xf32>) -> tensor<32xf32>
          %inserted_slice = tensor.insert_slice %57 into %arg19[%arg18] [32] [1] : tensor<32xf32> into tensor<128xf32>
          %expanded_11 = tensor.expand_shape %57 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
          %58 = hivm.hir.vsub {vector_producer_to_fuse_1} ins(%55, %expanded_11 : tensor<32x512xf32>, tensor<32x1xf32>) outs(%extracted_slice_7 : tensor<32x512xf32>) broadcast = [1] -> tensor<32x512xf32>
          %59 = hivm.hir.vexp {vector_producer_to_fuse_1} ins(%58 : tensor<32x512xf32>) outs(%extracted_slice_7 : tensor<32x512xf32>) -> tensor<32x512xf32>
          %extracted_slice_12 = tensor.extract_slice %51[%arg18, 0] [32, 512] [1, 1] : tensor<128x512xf16> to tensor<32x512xf16>
          %60 = hivm.hir.vcast {vector_producer_to_fuse_1} ins(%59 : tensor<32x512xf32>) outs(%extracted_slice_12 : tensor<32x512xf16>) -> tensor<32x512xf16>
          %subview_13 = memref.subview %collapse_shape[%arg18, 0] [32, 512] [1, 1] : memref<128x512xf16, strided<[512, 1]>> to memref<32x512xf16, strided<[512, 1], offset: ?>>
          hivm.hir.store ins(%60 : tensor<32x512xf16>) outs(%subview_13 : memref<32x512xf16, strided<[512, 1], offset: ?>>) {op_to_tile_1_0}
          %61 = hivm.hir.vsub {vector_producer_to_fuse_1} ins(%extracted_slice_5, %57 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_9 : tensor<32xf32>) -> tensor<32xf32>
          %extracted_slice_14 = tensor.extract_slice %arg20[%arg18] [32] [1] : tensor<128xf32> to tensor<32xf32>
          %62 = hivm.hir.vexp {vector_producer_to_fuse_1} ins(%61 : tensor<32xf32>) outs(%extracted_slice_14 : tensor<32xf32>) -> tensor<32xf32>
          %inserted_slice_15 = tensor.insert_slice %62 into %arg20[%arg18] [32] [1] : tensor<32xf32> into tensor<128xf32>
          %extracted_slice_16 = tensor.extract_slice %arg13[%arg18] [32] [1] : tensor<128xf32> to tensor<32xf32>
          %63 = hivm.hir.vmul {vector_producer_to_fuse_1} ins(%extracted_slice_16, %62 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_9 : tensor<32xf32>) -> tensor<32xf32>
          %64 = hivm.hir.vreduce {vector_producer_to_fuse_1} <sum> ins(%59 : tensor<32x512xf32>) outs(%extracted_slice_8 : tensor<32x1xf32>) reduce_dims = [1] -> tensor<32x1xf32>
          %collapsed_17 = tensor.collapse_shape %64 [[0, 1]] {vector_producer_to_fuse_1} : tensor<32x1xf32> into tensor<32xf32>
          %extracted_slice_18 = tensor.extract_slice %arg21[%arg18] [32] [1] : tensor<128xf32> to tensor<32xf32>
          %65 = hivm.hir.vadd {vector_producer_to_fuse_1} ins(%63, %collapsed_17 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice_18 : tensor<32xf32>) -> tensor<32xf32>
          %inserted_slice_19 = tensor.insert_slice %65 into %arg21[%arg18] [32] [1] : tensor<32xf32> into tensor<128xf32>
          scf.yield %inserted_slice, %inserted_slice_15, %inserted_slice_19 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
        }
        // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = [[EVENT1:[0-9]+]]
        // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = [[EVENT12:%[0-9]]]
        scope.return %53#1, %53#0, %53#2 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
      // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
      %47 = bufferization.to_tensor %41 restrict : memref<4x128x512xf16>
      %48 = scope.scope : () -> i32 {
        %51 = arith.index_cast %arg16 : i32 to index
        %52 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%51]
        %53 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%52, %22]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%53], sizes: [512, 128], strides: [128, 1] : memref<?xf16> to memref<512x128xf16, strided<[128, 1], offset: ?>>
        %alloc_6 = memref.alloc() : memref<512x128xf16>
        %54 = bufferization.to_tensor %reinterpret_cast_5 restrict writable : memref<512x128xf16, strided<[128, 1], offset: ?>>
        %55 = bufferization.to_tensor %alloc_6 restrict writable : memref<512x128xf16>
        %56 = hivm.hir.load ins(%54 : tensor<512x128xf16>) outs(%55 : tensor<512x128xf16>) {hivm.tcore_type = #hivm.tcore_type<CUBE>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<512x128xf16>
        %subview = memref.subview %42[0, 0, 0] [1, 128, 128] [1, 1, 1] {hivm.preload_workspace} : memref<4x128x128xf32> to memref<1x128x128xf32, strided<[16384, 128, 1]>>
        %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x128xf32, strided<[16384, 128, 1]>> into memref<128x128xf32, strided<[128, 1]>>
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = [[EVENT13:%[0-9]]]
        // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = [[EVENT1]]
        // CHECK-NEXT: {{.*}}scf.for %arg18{{.*}}
        scf.for %arg18 = %c0 to %c128 step %c64 {
          %extracted_slice = tensor.extract_slice %47[0, 0, 0] [1, 128, 512] [1, 1, 1] {cube_producer_to_fuse_2, hivm.preload_workspace} : tensor<4x128x512xf16> to tensor<128x512xf16>
          %extracted_slice_7 = tensor.extract_slice %extracted_slice[%arg18, 0] [64, 512] [1, 1] : tensor<128x512xf16> to tensor<64x512xf16>
          %58 = tensor.empty() {cube_producer_to_fuse_2} : tensor<128x512xf16>
          %extracted_slice_8 = tensor.extract_slice %58[%arg18, 0] [64, 512] [1, 1] : tensor<128x512xf16> to tensor<64x512xf16>
          %59 = hivm.hir.load ins(%extracted_slice_7 : tensor<64x512xf16>) outs(%extracted_slice_8 : tensor<64x512xf16>) {cube_producer_to_fuse_2, hivm.tcore_type = #hivm.tcore_type<CUBE>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<64x512xf16>
          %extracted_slice_9 = tensor.extract_slice %7[%arg18, 0] [64, 128] [1, 1] : tensor<128x128xf32> to tensor<64x128xf32>
          %60 = affine.min affine_map<(d0) -> (-d0 + 128, 64)>(%arg18)
          %61 = hivm.hir.mmadL1 {cube_producer_to_fuse_2, fixpipe_already_inserted = true, hivm.tile_mix_cube_num = 2 : i32} ins(%59, %56, %true, %60, %c512, %c128 : tensor<64x512xf16>, tensor<512x128xf16>, i1, index, index, index) outs(%extracted_slice_9 : tensor<64x128xf32>) -> tensor<64x128xf32>
          %subview_10 = memref.subview %collapse_shape[%arg18, 0] [64, 128] [1, 1] : memref<128x128xf32, strided<[128, 1]>> to memref<64x128xf32, strided<[128, 1], offset: ?>>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, op_to_tile_2_0} ins(%61 : tensor<64x128xf32>) outs(%subview_10 : memref<64x128xf32, strided<[128, 1], offset: ?>>)
        }
        // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = [[EVENT2:[0-9]+]]
        // CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = [[EVENT14:%[0-9]]]
        %57 = arith.addi %arg16, %c512_i32 : i32
        scope.return %57 : i32
      // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
      } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
      %49 = bufferization.to_tensor %42 restrict : memref<4x128x128xf32>
      %50 = scope.scope : () -> tensor<128x128xf32> {
        %extracted_slice = tensor.extract_slice %49[0, 0, 0] [1, 128, 128] [1, 1, 1] {hivm.preload_workspace} : tensor<4x128x128xf32> to tensor<128x128xf32>
        // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = [[EVENT2]]
        // CHECK-NEXT: {{.*}}hivm.hir.load ins(%extracted_slice : tensor<128x128xf32>){{.*}}
        %51 = hivm.hir.load ins(%extracted_slice : tensor<128x128xf32>) outs(%7 : tensor<128x128xf32>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>} init_out_buffer = false may_implicit_transpose_with_last_axis = false -> tensor<128x128xf32>
        // CHECK-NEXT: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = [[EVENT15:%[0-9]]]
        %expanded_5 = tensor.expand_shape %46#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %52 = hivm.hir.vmul ins(%arg14, %expanded_5 : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
        %53 = hivm.hir.vadd ins(%52, %51 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scope.return %53 : tensor<128x128xf32>
      // CHECK: } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
      scf.yield %46#2, %50, %46#1, %48, %44 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32
    }
    %32 = hivm.hir.vln ins(%31#0 : tensor<128xf32>) outs(%10 : tensor<128xf32>) -> tensor<128xf32>
    %33 = hivm.hir.vadd ins(%31#2, %32 : tensor<128xf32>, tensor<128xf32>) outs(%10 : tensor<128xf32>) -> tensor<128xf32>
    %expanded = tensor.expand_shape %31#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
    %34 = hivm.hir.vdiv ins(%31#1, %expanded : tensor<128x128xf32>, tensor<128x1xf32>) outs(%7 : tensor<128x128xf32>) broadcast = [1] -> tensor<128x128xf32>
    %35 = arith.muli %13, %c8192_i32 : i32
    %36 = arith.index_cast %35 : i32 to index
    %37 = arith.index_cast %23 : i32 to index
    %38 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%36, %37]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%38], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%33 : tensor<128xf32>) outs(%reinterpret_cast_4 : memref<128xf32, strided<[1], offset: ?>>)
    %39 = tensor.empty() : tensor<128x128xf16>
    %40 = hivm.hir.vcast ins(%34 : tensor<128x128xf32>) outs(%39 : tensor<128x128xf16>) -> tensor<128x128xf16>
    hivm.hir.store ins(%40 : tensor<128x128xf16>) outs(%reinterpret_cast_3 : memref<128x128xf16, strided<[128, 1], offset: ?>>)
  }
  // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
  // CHECK-NEXT: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
  // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
  // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 0
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 1
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 10
  // CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 11
  // CHECK-NEXT: return
  return
}

