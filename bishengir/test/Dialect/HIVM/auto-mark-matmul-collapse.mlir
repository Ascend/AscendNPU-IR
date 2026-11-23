// RUN: bishengir-opt %s -hivm-mark-matmul-row-collapses -hivm-insert-load-store-for-mix-cv -canonicalize-ext | FileCheck %s
// RUN: bishengir-opt %s -hivm-mark-matmul-row-collapses -hivm-insert-load-store-for-mix-cv='enable-legacy=true' -canonicalize-ext | FileCheck %s
// RUN: bishengir-opt %s -hivm-mark-matmul-row-collapses -hivm-insert-load-store-for-mix-cv -hivm-mark-matmul-row-collapses -hivm-insert-load-store-for-mix-cv -canonicalize-ext | FileCheck %s
// RUN: bishengir-opt %s -verify-diagnostics -hacc-append-device-spec='target=Ascend910_9382' -optimize-hivm-pipeline='target=Ascend910_9382 enable-triton-kernel-compile=true enable-layout-optimization=false tile-mix-cube-loop=1 tile-mix-vector-loop=1' | FileCheck %s --check-prefix=FULL --implicit-check-not='memref<2x4x2x16x16xf16'

// FULL-LABEL: func.func @dynamic_row_merge_mix_aic(
// FULL: memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
// FULL: call @nd2nz_half
// FULL-SAME: : (memref<?x?xf16,
// FULL: call @mma_tile_half_to_float
// FULL-SAME: : (memref<?x?x?x?xf16,

// The descriptor copies disjoint GM row groups into one contiguous ND buffer.
// Preserve the dynamic bounds and initialization on that copy, then transfer
// the collapsed matrix to Cube. Converting the groups independently to NZ
// would change the element order of the 64x64 MMAD operand (issue #886).
// CHECK-LABEL: func.func @dynamic_row_merge
// CHECK: %[[SRC:.*]] = memref.subview {{.*}} [2, {{.*}}, {{.*}}] [1, 1, 1]
// CHECK: %[[DST:.*]] = memref.subview {{.*}} [2, {{.*}}, {{.*}}] [1, 1, 1]
// CHECK: hivm.hir.load ins(%[[SRC]] : memref<2x?x?xf16, strided<[16384, 128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%[[DST]]
// CHECK-SAME: pad_value = {{.*}} : f16
// CHECK-SAME: init_out_buffer = true init_condition = {{.*}} : i1
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape {{.*}} {{\[\[}}0, 1], [2]] : tensor<2x32x64xf16> into tensor<64x64xf16>
// CHECK-DAG: annotation.mark %[[COLLAPSED]] {maybeUnCollapsibleReshape} : tensor<64x64xf16>
// CHECK-DAG: %[[STORED:.*]] = hivm.hir.store ins(%[[COLLAPSED]] : tensor<64x64xf16>)
// CHECK-DAG: %[[LOADED:.*]] = hivm.hir.load ins(%[[STORED]] : tensor<64x64xf16>)
// CHECK: hivm.hir.mmadL1 ins(%[[LOADED]],
// CHECK-NOT: maybeUnCollapsibleReshape
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  func.func @dynamic_row_merge(
      %ffts: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>},
      %src: memref<2x32x64xf16, strided<[16384, 128, 1]>, #hivm.address_space<gm>>,
      %rhs: memref<64x64xf16, #hivm.address_space<gm>>, %m_start: index, %k_start: index,
      %rows: index, %cols: index, %initialize: i1, %out: memref<64x64xf32, #hivm.address_space<gm>>,
      %workspace: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>})
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %true = arith.constant true
    %zero = arith.constant 0.0 : f16
    %c64 = arith.constant 64 : index
    %local = memref.alloc() : memref<2x32x64xf16>
    %src_view = memref.subview %src[0, %m_start, %k_start]
        [2, %rows, %cols] [1, 1, 1]
        : memref<2x32x64xf16, strided<[16384, 128, 1]>, #hivm.address_space<gm>>
       to memref<2x?x?xf16, strided<[16384, 128, 1], offset: ?>, #hivm.address_space<gm>>
    %dst_view = memref.subview %local[0, %m_start, %k_start]
        [2, %rows, %cols] [1, 1, 1]
        : memref<2x32x64xf16>
       to memref<2x?x?xf16, strided<[2048, 64, 1], offset: ?>>
    hivm.hir.load ins(%src_view : memref<2x?x?xf16, strided<[16384, 128, 1], offset: ?>, #hivm.address_space<gm>>)
        outs(%dst_view : memref<2x?x?xf16, strided<[2048, 64, 1], offset: ?>>)
        pad_mode = <PadValue> pad_value = %zero : f16
        init_out_buffer = true init_condition = %initialize : i1
    %t = bufferization.to_tensor %local restrict writable : memref<2x32x64xf16>
    %a = tensor.collapse_shape %t [[0, 1], [2]]
        : tensor<2x32x64xf16> into tensor<64x64xf16>
    %rhs_local = memref.alloc() : memref<64x64xf16>
    hivm.hir.load ins(%rhs : memref<64x64xf16, #hivm.address_space<gm>>) outs(%rhs_local : memref<64x64xf16>)
    %b = bufferization.to_tensor %rhs_local restrict writable : memref<64x64xf16>
    %empty = tensor.empty() : tensor<64x64xf32>
    %result = hivm.hir.mmadL1 ins(%a, %b, %true, %c64, %c64, %c64
        : tensor<64x64xf16>, tensor<64x64xf16>, i1, index, index, index)
        outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    hivm.hir.store ins(%result : tensor<64x64xf32>) outs(%out : memref<64x64xf32, #hivm.address_space<gm>>)
    return
  }
}
