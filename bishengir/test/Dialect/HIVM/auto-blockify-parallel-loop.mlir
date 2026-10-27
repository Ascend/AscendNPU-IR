// RUN: bishengir-opt %s -auto-blockify-parallel-loop -split-input-file -verify-diagnostics | FileCheck %s
// CHECK: "VECTOR_CORE_COUNT", [[Physical_num:[0-9]+]]
// CHECK: annotation.mark %[[Upper_bound:.*]] {logical_block_num}
// CHECK: %[[Lower_bound_i64:.*]] = hivm.hir.get_block_idx
// CHECK: %[[Step:.*]] = arith.constant [[Physical_num]]
// CHECK: %[[Lower_bound_i32:.*]] = arith.trunci %[[Lower_bound_i64]]
// CHECK: scf.for %{{.*}} = %[[Lower_bound_i32]] to %[[Upper_bound]] step %[[Step]]
// CHECK: hivm.hir.store {{.*}} {autoblockify.dma}
// CHECK: autoblockify.subloop
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @add_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {WorkspaceArgIdx = 0 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg6, %arg7 : i32
    %1 = arith.muli %0, %arg8 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg8, %arg7 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg6 : i32
    %7 = arith.muli %6, %c1024_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xf32>
    %9 = arith.addi %8, %c1024 : index
    %10 = arith.index_cast %arg5 : i32 to index
    %11 = arith.maxsi %8, %10 : index
    %12 = arith.minsi %9, %11 : index
    %13 = arith.subi %12, %8 : index
    %subview = memref.subview %reinterpret_cast[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%13] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_0 : memref<?xf32, strided<[1]>>) left_padding_num = %c0 : index init_out_buffer = false
    %14 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%13] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    hivm.hir.load ins(%subview_3 : memref<?xf32, strided<[1], offset: ?>>) outs(%subview_4 : memref<?xf32, strided<[1]>>) left_padding_num = %c0 : index init_out_buffer = false
    %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32>
    %16 = tensor.empty() : tensor<1024xf32>
    %17 = hivm.hir.vadd ins(%14, %15 : tensor<1024xf32>, tensor<1024xf32>) outs(%16 : tensor<1024xf32>) -> tensor<1024xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%8], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %17[0] [%13] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%13] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%extracted_slice : tensor<?xf32>) outs(%subview_6 : memref<?xf32, strided<[1], offset: ?>>)
    return
  }
}
// -----

// Regression: this kernel was AIV before hivm.core_ratio forced it to MIX,
// and it must keep stepping by the Vector count.
// CHECK-LABEL: func.func @blockify_ratio_0_1(
// CHECK:         %[[STEP:.*]] = arith.constant 56 : i32
// CHECK:         scf.for %{{.*}} step %[[STEP]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @blockify_ratio_0_1(%n: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<0, 1>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %n {logical_block_num} : i32
    %0 = hivm.hir.get_block_idx -> i64
    return
  }
}

// -----

// Cube-only blocks.
// CHECK-LABEL: func.func @blockify_ratio_1_0(
// CHECK:         %[[STEP:.*]] = arith.constant 28 : i32
// CHECK:         scf.for %{{.*}} step %[[STEP]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @blockify_ratio_1_0(%n: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<1, 0>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %n {logical_block_num} : i32
    %0 = hivm.hir.get_block_idx -> i64
    return
  }
}

// -----

// One Vector core per block, but blocks are still Cube-counted.
// CHECK-LABEL: func.func @blockify_ratio_1_1(
// CHECK:         %[[STEP:.*]] = arith.constant 28 : i32
// CHECK:         scf.for %{{.*}} step %[[STEP]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @blockify_ratio_1_1(%n: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<1, 1>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %n {logical_block_num} : i32
    %0 = hivm.hir.get_block_idx -> i64
    return
  }
}

// -----

// Two Vector cores per block, addressed by sub-block index.
// CHECK-LABEL: func.func @blockify_ratio_1_2(
// CHECK:         %[[STEP:.*]] = arith.constant 28 : i32
// CHECK:         scf.for %{{.*}} step %[[STEP]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @blockify_ratio_1_2(%n: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.core_ratio = #hivm.core_ratio<1, 2>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %n {logical_block_num} : i32
    %0 = hivm.hir.get_block_idx -> i64
    return
  }
}

// -----

// A MIX func with no ratio keeps the historical 1:2 default, so nothing changes
// for kernels predating this attribute.
// CHECK-LABEL: func.func @blockify_mix_no_ratio(
// CHECK:         %[[STEP:.*]] = arith.constant 28 : i32
// CHECK:         scf.for %{{.*}} step %[[STEP]]
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @blockify_mix_no_ratio(%n: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    annotation.mark %n {logical_block_num} : i32
    %0 = hivm.hir.get_block_idx -> i64
    return
  }
}

// -----

// Only non-atomic stores and fixpipes at a block-indexed offset are marked.
// CHECK-LABEL: func.func @mark_blockify_dmas(
// CHECK:         scf.for
// CHECK:         hivm.hir.store ins({{[^{]*}}) {autoblockify.dma}
// CHECK:         hivm.hir.store ins({{[^{]*}}){{$}}
// CHECK:         hivm.hir.store ins({{[^{]*}}) atomic = <add>
// CHECK:         hivm.hir.set_atomic kind = <add>
// CHECK-NEXT:    hivm.hir.store ins({{[^{]*}}){{$}}
// CHECK-NEXT:    hivm.hir.set_atomic kind = <none>
// CHECK:         hivm.hir.fixpipe {autoblockify.dma, {{.*}}}
// CHECK:         hivm.hir.set_atomic kind = <add>
// CHECK-NEXT:    hivm.hir.fixpipe {dma_mode
// CHECK-NEXT:    hivm.hir.set_atomic kind = <none>
// CHECK:         hivm.hir.store ins({{[^{]*}}){{$}}
// CHECK:         autoblockify.subloop
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 1572864 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>>>, hivm.module_core_type = #hivm.module_core_type<AIC>} {
  func.func @mark_blockify_dmas(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>, %n: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    annotation.mark %n {logical_block_num} : i32
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.muli %1, %c256_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %src = tensor.empty() : tensor<256xf32>
    %acc = tensor.empty() : tensor<16x16xf32>
    // Block-indexed store.
    %indexed = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%src : tensor<256xf32>) outs(%indexed : memref<256xf32, strided<[1], offset: ?>>)
    // Every block stores to the same GM.
    %fixed = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%src : tensor<256xf32>) outs(%fixed : memref<256xf32, strided<[1], offset: ?>>)
    // Atomic store.
    %atomic = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%src : tensor<256xf32>) outs(%atomic : memref<256xf32, strided<[1], offset: ?>>) atomic = <add>
    // Plain store in atomic mode.
    %mode = memref.reinterpret_cast %arg3 to offset: [%3], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    hivm.hir.set_atomic kind = <add>[type = f32]
    hivm.hir.store ins(%src : tensor<256xf32>) outs(%mode : memref<256xf32, strided<[1], offset: ?>>)
    hivm.hir.set_atomic kind = <none>[type = f32]
    // Block-indexed fixpipe.
    %fix = memref.reinterpret_cast %arg4 to offset: [%3], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%acc : tensor<16x16xf32>) outs(%fix : memref<16x16xf32, strided<[16, 1], offset: ?>>)
    // Fixpipe in atomic mode.
    %fix_atomic = memref.reinterpret_cast %arg5 to offset: [%3], sizes: [16, 16], strides: [16, 1] : memref<?xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
    hivm.hir.set_atomic kind = <add>[type = f32]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%acc : tensor<16x16xf32>) outs(%fix_atomic : memref<16x16xf32, strided<[16, 1], offset: ?>>)
    hivm.hir.set_atomic kind = <none>[type = f32]
    // Block-indexed view of a buffer that is not a kernel argument.
    %alloc = memref.alloc() : memref<4096xf32>
    %local = memref.subview %alloc[%3] [256] [1] : memref<4096xf32> to memref<256xf32, strided<[1], offset: ?>>
    hivm.hir.store ins(%src : tensor<256xf32>) outs(%local : memref<256xf32, strided<[1], offset: ?>>)
    return
  }
}
