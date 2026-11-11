// RUN: bishengir-opt %s --hfusion-prepare-l1-gather --canonicalize | FileCheck %s --check-prefix=PREPARE --implicit-check-not=tensor.extract --implicit-check-not=tensor.insert
// RUN: bishengir-opt %s --hfusion-prepare-l1-gather --canonicalize --convert-arith-to-hfusion --convert-hfusion-to-hivm="mm-map-mode=macro_instr" --convert-to-hivm-op --hivm-normalize-matmul --canonicalize --hivm-promote-l1-gather --canonicalize | FileCheck %s --check-prefix=PROMOTE
// RUN: sed 's/enable_l1_gather = true/enable_l1_gather = false/' %s | bishengir-opt --hfusion-prepare-l1-gather --canonicalize | FileCheck %s --check-prefix=DISABLED
// RUN: sed '/annotation.mark .*enable_l1_gather/d' %s | bishengir-opt --hfusion-prepare-l1-gather --canonicalize | FileCheck %s --check-prefix=DISABLED
// RUN: sed 's/%%element_offset = arith.addi %%row_base, %%column_i64/%%twice_column = arith.addi %%column_i64, %%column_i64 : i64\n        %%element_offset = arith.addi %%row_base, %%twice_column/' %s | bishengir-opt --hfusion-prepare-l1-gather --canonicalize | FileCheck %s --check-prefix=DISABLED

// Fresh sparse_gqa_prefill K lowering: nested scalar loops build [column, row].
// Normalize the hinted load to contiguous rows while retaining the i32 address
// multiplication, clipped metadata reads, and negative-token mask. The final
// RUN changes the column stride to two, which must keep the scalar fallback.
// PREPARE-LABEL: func.func @sparse_gqa_key_implicit
// PREPARE: linalg.fill {{.*}} outs({{.*}} : memref<16x256xbf16>)
// PREPARE: scf.for
// PREPARE: arith.cmpi ult
// PREPARE: %[[TOKEN:.*]] = scf.if
// PREPARE: memref.load {{.*}} : memref<?xi32
// PREPARE: %[[VALID:.*]] = arith.cmpi sge, %[[TOKEN]]
// PREPARE: scf.if %[[VALID]]
// PREPARE: %[[ROWOFFSET:.*]] = arith.muli %[[TOKEN]], {{.*}} : i32
// PREPARE: arith.extsi %[[ROWOFFSET]] : i32 to i64
// PREPARE: memref.copy {{.*}} to memref<1x256xbf16
// PREPARE: %[[ROWS:.*]] = bufferization.to_tensor {{.*}} : memref<16x256xbf16>
// PREPARE: annotation.mark %[[ROWS]] {hivm.enable_l1_gather = true} : tensor<16x256xbf16>
// PREPARE: linalg.transpose ins(%[[ROWS]] : tensor<16x256xbf16>)
// PREPARE: linalg.matmul
// PROMOTE-LABEL: func.func @sparse_gqa_key_implicit
// PROMOTE: memref.alloc() : memref<16x1x16x16xbf16, #hivm.address_space<cbuf>>
// PROMOTE: hivm.hir.vbrc {{.*}} #hivm.address_space<cbuf>
// PROMOTE: scf.for
// PROMOTE: arith.cmpi ult
// PROMOTE: memref.load {{.*}} : memref<?xi32
// PROMOTE: arith.cmpi sge
// PROMOTE: scf.if
// PROMOTE: hivm.hir.nd2nz {{.*}} memref<1x256xbf16
// PROMOTE-NOT: hivm.hir.load
// PROMOTE-NOT: hivm.hir.vtranspose
// PROMOTE-NOT: annotation.mark
// PROMOTE: hivm.hir.mmadL1 {{.*}}b_transpose
// DISABLED-LABEL: func.func @sparse_gqa_key_implicit
// DISABLED: scf.for {{.*}} to {{.*}} iter_args
// DISABLED: scf.for {{.*}} to {{.*}} iter_args
// DISABLED: tensor.extract
// DISABLED: arith.muli {{.*}} : i32
// DISABLED: arith.extsi {{.*}} : i32 to i64
// DISABLED: memref.load {{.*}} : memref<1xbf16
// DISABLED: tensor.insert
// DISABLED: arith.select
// DISABLED: linalg.matmul

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sparse_gqa_key_implicit(
      %ids: memref<?xi32>, %k: memref<?xbf16>,
      %q: tensor<16x256xbf16>, %acc: tensor<16x16xf32>,
      %query: i64, %seq_start: i64, %tile_start: index, %topk: i32)
      -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %minus_one = arith.constant -1 : i32
    %zeros = arith.constant dense<0> : tensor<16xi32>
    %other = arith.constant dense<0.0> : tensor<256x16xbf16>
    %init = tensor.empty() : tensor<256x16xbf16>

    %seq_base = arith.muli %seq_start, %c256_i64 : i64
    %query_offset = arith.muli %query, %c512_i64 : i64
    %query_index = arith.index_cast %query_offset : i64 to index
    %metadata_offset = arith.addi %query_index, %tile_start : index
    %metadata = memref.reinterpret_cast %ids to offset: [%metadata_offset], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1], offset: ?>>
    %indices = memref.alloc() : memref<16xi32>
    %tile_end = arith.addi %tile_start, %c16 : index
    %topk_index = arith.index_cast %topk : i32 to index
    %clamped_end = arith.maxsi %tile_start, %topk_index : index
    %live_end = arith.minsi %tile_end, %clamped_end : index
    %live = arith.subi %live_end, %tile_start : index
    %partial = arith.cmpi slt, %live, %c16 : index
    scf.if %partial {
      linalg.fill ins(%minus_one : i32) outs(%indices : memref<16xi32>)
    } {hivm.unlikely_condition}
    %src_prefix = memref.subview %metadata[0] [%live] [1] : memref<16xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
    %dst_prefix = memref.subview %indices[0] [%live] [1] : memref<16xi32> to memref<?xi32, strided<[1]>>
    memref.copy %src_prefix, %dst_prefix : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
    %tokens = bufferization.to_tensor %indices restrict writable : memref<16xi32>
    %valid = arith.cmpi sge, %tokens, %zeros : tensor<16xi32>
    %mask_init = tensor.empty() : tensor<256x16xi1>
    %mask = linalg.broadcast ins(%valid : tensor<16xi1>) outs(%mask_init : tensor<256x16xi1>) dimensions = [0]
    %tile = scf.for %column = %c0 to %c256 step %c1 iter_args(%outer_tile = %init) -> (tensor<256x16xbf16>) {
      %next = scf.for %row = %c0 to %c16 step %c1 iter_args(%inner_tile = %outer_tile) -> (tensor<256x16xbf16>) {
        %token = tensor.extract %tokens[%row] {DiscreteMemAccess} : tensor<16xi32>
        %row_offset_i32 = arith.muli %token, %c256_i32 : i32
        %row_offset_i64 = arith.extsi %row_offset_i32 : i32 to i64
        %row_base = arith.addi %seq_base, %row_offset_i64 : i64
        %column_i32 = arith.index_cast %column : index to i32
        %column_i64 = arith.extsi %column_i32 : i32 to i64
        %element_offset = arith.addi %row_base, %column_i64 : i64
        %element_index = arith.index_cast %element_offset : i64 to index
        %element = memref.reinterpret_cast %k to offset: [%element_index], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
        %value = memref.load %element[%c0] : memref<1xbf16, strided<[1], offset: ?>>
        %inserted = tensor.insert %value into %inner_tile[%column, %row] : tensor<256x16xbf16>
        scf.yield {DiscreteMemAccess} %inserted : tensor<256x16xbf16>
      } {ExtractedLoadOrStore}
      scf.yield %next : tensor<256x16xbf16>
    } {ExtractedLoadOrStore}
    %selected = arith.select %mask, %tile, %other : tensor<256x16xi1>, tensor<256x16xbf16>
    annotation.mark %selected {hivm.enable_l1_gather = true} : tensor<256x16xbf16>
    %result = linalg.matmul {input_precision = "ieee"} ins(%q, %selected : tensor<16x256xbf16>, tensor<256x16xbf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %result : tensor<16x16xf32>
  }
}
