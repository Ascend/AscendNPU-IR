// RUN: bishengir-opt %s -create-preload -split-input-file | FileCheck %s --check-prefix=PRELOAD
// RUN: bishengir-opt %s -create-preload -hivm-graph-sync-solver -split-input-file | FileCheck %s --check-prefix=SYNC

// Each page writes a disjoint L1 slice. Preserve the existing parallel marker
// when rebuilding the loop inside a preload scope, so sync analysis does not
// insert a loop-carried MTE2 barrier before each page transfer.
// PRELOAD-LABEL: func.func @preload_parallel_pages(
// PRELOAD: scf.for
// PRELOAD: scf.if
// PRELOAD: scf.for
// PRELOAD: hivm.hir.nd2nz
// PRELOAD-NEXT: } {hivm.parallel_loop}
// SYNC-LABEL: func.func @preload_parallel_pages(
// SYNC: scf.for
// SYNC: scf.if
// SYNC: scf.for
// SYNC-NOT: hivm.hir.pipe_barrier
// SYNC: hivm.hir.nd2nz
// SYNC-NEXT: } {hivm.parallel_loop}

#block = affine_map<(d0) -> (d0 floordiv 16)>
#row = affine_map<(d0) -> (d0 mod 16)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>} {
  func.func @preload_parallel_pages(%src: memref<128x192xbf16, #hivm.address_space<gm>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %addr = arith.constant 0 : i64
    %dst = hivm.hir.pointer_cast(%addr) : memref<12x8x16x16xbf16, #hivm.address_space<cbuf>>
    scf.for %tile = %c0 to %n step %c1 {
      scope.scope : () -> () {
        scf.for %page = %c0 to %c16 step %c1 {
          %offset = arith.muli %page, %c8 : index
          %block = affine.apply #block(%offset)
          %row = affine.apply #row(%offset)
          %in = memref.subview %src[%offset, 0] [8, 192] [1, 1] : memref<128x192xbf16, #hivm.address_space<gm>> to memref<8x192xbf16, strided<[192, 1], offset: ?>, #hivm.address_space<gm>>
          %out = memref.subview %dst[0, %block, %row, 0] [12, 1, 8, 16] [1, 1, 1, 1] : memref<12x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<12x1x8x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%in : memref<8x192xbf16, strided<[192, 1], offset: ?>, #hivm.address_space<gm>>) outs(%out : memref<12x1x8x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
        } {hivm.parallel_loop}
        scope.return
      } {no_inline, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}
    }
    return
  }
}

// -----

// Without the parallel guarantee, retain conservative loop-carried sync.
// PRELOAD-LABEL: func.func @preload_unmarked_pages(
// PRELOAD-NOT: hivm.parallel_loop
// PRELOAD: return
// SYNC-LABEL: func.func @preload_unmarked_pages(
// SYNC: scf.for
// SYNC: scf.if
// SYNC: scf.for
// SYNC: hivm.hir.pipe_barrier[<PIPE_MTE2>]
// SYNC-NEXT: hivm.hir.nd2nz

#block = affine_map<(d0) -> (d0 floordiv 16)>
#row = affine_map<(d0) -> (d0 mod 16)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIC>} {
  func.func @preload_unmarked_pages(%src: memref<128x192xbf16, #hivm.address_space<gm>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %addr = arith.constant 0 : i64
    %dst = hivm.hir.pointer_cast(%addr) : memref<12x8x16x16xbf16, #hivm.address_space<cbuf>>
    scf.for %tile = %c0 to %n step %c1 {
      scope.scope : () -> () {
        scf.for %page = %c0 to %c16 step %c1 {
          %offset = arith.muli %page, %c8 : index
          %block = affine.apply #block(%offset)
          %row = affine.apply #row(%offset)
          %in = memref.subview %src[%offset, 0] [8, 192] [1, 1] : memref<128x192xbf16, #hivm.address_space<gm>> to memref<8x192xbf16, strided<[192, 1], offset: ?>, #hivm.address_space<gm>>
          %out = memref.subview %dst[0, %block, %row, 0] [12, 1, 8, 16] [1, 1, 1, 1] : memref<12x8x16x16xbf16, #hivm.address_space<cbuf>> to memref<12x1x8x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.nd2nz {dst_continuous} ins(%in : memref<8x192xbf16, strided<[192, 1], offset: ?>, #hivm.address_space<gm>>) outs(%out : memref<12x1x8x16xbf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>)
        }
        scope.return
      } {no_inline, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}
    }
    return
  }
}
