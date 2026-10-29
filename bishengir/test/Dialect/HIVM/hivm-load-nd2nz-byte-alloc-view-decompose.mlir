// RUN: bishengir-opt %s -hivm-aggregated-decompose-op="decompose-phase=before-hivm-align" -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=LOAD
// RUN: bishengir-opt %s -hivm-aggregated-decompose-op="decompose-phase=after-infer-hivm-data-layout" -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=ND2NZ

//===----------------------------------------------------------------------===//
// LoadOp: dst is a typed memref.view over a byte alloc; VBrc pad buffer needs
// a 1-D view sized from the byte alloc, not the typed dst shape.
//===----------------------------------------------------------------------===//

// -----
// LOAD-LABEL: func @test_load_byte_alloc_view_static
func.func @test_load_byte_alloc_view_static(%arg0: memref<256x128xf32, #hivm.address_space<gm>>) {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<131072xi8, #hivm.address_space<ub>>
  %view = memref.view %alloc[%c0][] : memref<131072xi8, #hivm.address_space<ub>> to memref<256x128xf32, #hivm.address_space<ub>>
  // LOAD: %[[PAD_VIEW:.*]] = memref.view %[[ALLOC:.*]][{{.*}}][] : memref<131072xi8, #hivm.address_space<ub>> to memref<32768xf32, #hivm.address_space<ub>>
  // LOAD: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[PAD_VIEW]] : memref<32768xf32, #hivm.address_space<ub>>)
  // LOAD: hivm.hir.load ins(%arg0 : memref<256x128xf32, #hivm.address_space<gm>>) outs(%view : memref<256x128xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = {{.*}} : f32 left_padding_num = {{.*}} : index
  hivm.hir.load ins(%arg0: memref<256x128xf32, #hivm.address_space<gm>>) outs(%view: memref<256x128xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true
  return
}

// -----
// LOAD-LABEL: func @test_load_byte_alloc_view_dyn
func.func @test_load_byte_alloc_view_dyn(%arg0: memref<?x?xf32, #hivm.address_space<gm>>, %m: index, %n: index) attributes { enable_auto_mark_buffer_size } {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %byte_elems = arith.muli %m, %n : index
  %byte_size = arith.muli %byte_elems, %c32 : index
  %alloc = memref.alloc(%byte_size) : memref<?xi8, #hivm.address_space<ub>>
  %view = memref.view %alloc[%c0][%m, %n] : memref<?xi8, #hivm.address_space<ub>> to memref<?x?xf32, #hivm.address_space<ub>>
  // LOAD: %[[NUM_ELEMS:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
  // LOAD: %[[PAD_VIEW:.*]] = memref.view %[[ALLOC:.*]][{{.*}}][%[[NUM_ELEMS]]] : memref<?xi8, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
  // LOAD: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[PAD_VIEW]] : memref<?xf32, #hivm.address_space<ub>>)
  // LOAD: hivm.hir.load ins(%arg0 : memref<?x?xf32, #hivm.address_space<gm>>) outs(%view : memref<?x?xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = {{.*}} : f32 left_padding_num = {{.*}} : index
  hivm.hir.load ins(%arg0: memref<?x?xf32, #hivm.address_space<gm>>) outs(%view: memref<?x?xf32, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true
  return
}

// -----
// LOAD-LABEL: func @test_load_byte_alloc_view_subview
func.func @test_load_byte_alloc_view_subview(%arg0: memref<256x128xf32, #hivm.address_space<gm>>) {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<131072xi8, #hivm.address_space<ub>>
  %view = memref.view %alloc[%c0][] : memref<131072xi8, #hivm.address_space<ub>> to memref<256x128xf32, #hivm.address_space<ub>>
  %subview = memref.subview %view[0, 0] [256, 128] [1, 1] : memref<256x128xf32, #hivm.address_space<ub>> to memref<256x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>
  // LOAD: %[[PAD_VIEW:.*]] = memref.view %[[ALLOC:.*]][{{.*}}][] : memref<131072xi8, #hivm.address_space<ub>> to memref<32768xf32, #hivm.address_space<ub>>
  // LOAD: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[PAD_VIEW]] : memref<32768xf32, #hivm.address_space<ub>>)
  // LOAD: hivm.hir.load ins(%arg0 : memref<256x128xf32, #hivm.address_space<gm>>) outs(%subview : memref<256x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = {{.*}} : f32 left_padding_num = {{.*}} : index
  hivm.hir.load ins(%arg0: memref<256x128xf32, #hivm.address_space<gm>>) outs(%subview: memref<256x128xf32, strided<[128, 1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = true
  return
}

//===----------------------------------------------------------------------===//
// ND2NZOp: same byte-alloc + typed view pattern for init_out_buffer VBrc.
//===----------------------------------------------------------------------===//

// -----
// ND2NZ-LABEL: func @test_nd2nz_byte_alloc_view_static
func.func @test_nd2nz_byte_alloc_view_static(%arg0: memref<8x16x16x8xf32, #hivm.address_space<gm>>) {
  %cst = arith.constant 2.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<16384xi8, #hivm.address_space<cbuf>>
  %view = memref.view %alloc[%c0][] : memref<16384xi8, #hivm.address_space<cbuf>> to memref<8x16x16x8xf32, #hivm.address_space<cbuf>>
  // ND2NZ: %[[PAD_VIEW:.*]] = memref.view %[[ALLOC:.*]][{{.*}}][] : memref<16384xi8, #hivm.address_space<cbuf>> to memref<4096xf32, #hivm.address_space<cbuf>>
  // ND2NZ: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[PAD_VIEW]] : memref<4096xf32, #hivm.address_space<cbuf>>)
  // ND2NZ: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<8x16x16x8xf32, #hivm.address_space<gm>>) outs(%view : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>)
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<8x16x16x8xf32, #hivm.address_space<gm>>) outs(%view : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = %cst : f32
  return
}

// -----
// ND2NZ-LABEL: func @test_nd2nz_byte_alloc_view_dyn
func.func @test_nd2nz_byte_alloc_view_dyn(%arg0: memref<?x?x?x?xf32, #hivm.address_space<gm>>, %d0: index, %d1: index, %d2: index, %d3: index) attributes { enable_auto_mark_buffer_size } {
  %cst = arith.constant 2.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %tmp0 = arith.muli %d0, %d1 : index
  %tmp1 = arith.muli %tmp0, %d2 : index
  %elem_count = arith.muli %tmp1, %d3 : index
  %byte_size = arith.muli %elem_count, %c32 : index
  %alloc = memref.alloc(%byte_size) : memref<?xi8, #hivm.address_space<cbuf>>
  %view = memref.view %alloc[%c0][%d0, %d1, %d2, %d3] : memref<?xi8, #hivm.address_space<cbuf>> to memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
  // ND2NZ: %[[NUM_ELEMS:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
  // ND2NZ: %[[PAD_VIEW:.*]] = memref.view %[[ALLOC:.*]][{{.*}}][%[[NUM_ELEMS]]] : memref<?xi8, #hivm.address_space<cbuf>> to memref<?xf32, #hivm.address_space<cbuf>>
  // ND2NZ: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[PAD_VIEW]] : memref<?xf32, #hivm.address_space<cbuf>>)
  // ND2NZ: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<?x?x?x?xf32, #hivm.address_space<gm>>) outs(%view : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>)
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<?x?x?x?xf32, #hivm.address_space<gm>>) outs(%view : memref<?x?x?x?xf32, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = %cst : f32
  return
}

// -----
// ND2NZ-LABEL: func @test_nd2nz_byte_alloc_view_if
func.func @test_nd2nz_byte_alloc_view_if(%arg0: memref<8x16x16x8xf32, #hivm.address_space<gm>>, %cond: i1) {
  %cst = arith.constant 2.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<16384xi8, #hivm.address_space<cbuf>>
  %view = memref.view %alloc[%c0][] : memref<16384xi8, #hivm.address_space<cbuf>> to memref<8x16x16x8xf32, #hivm.address_space<cbuf>>
  // ND2NZ: %[[PAD_VIEW:.*]] = memref.view %[[ALLOC:.*]][{{.*}}][] : memref<16384xi8, #hivm.address_space<cbuf>> to memref<4096xf32, #hivm.address_space<cbuf>>
  // ND2NZ: scf.if
  // ND2NZ: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[PAD_VIEW]] : memref<4096xf32, #hivm.address_space<cbuf>>)
  // ND2NZ: } {hivm.unlikely_condition}
  // ND2NZ: hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<8x16x16x8xf32, #hivm.address_space<gm>>) outs(%view : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>)
  hivm.hir.nd2nz {dst_continuous} ins(%arg0 : memref<8x16x16x8xf32, #hivm.address_space<gm>>) outs(%view : memref<8x16x16x8xf32, #hivm.address_space<cbuf>>) init_out_buffer = true pad_value = %cst : f32 init_condition = %cond : i1
  return
}
