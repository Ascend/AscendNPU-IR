// RUN: bishengir-opt -canonicalize %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_propagate_convert_layout_down_scf_if(
// CHECK: %[[select:.*]] = arith.select
// CHECK: hivm.hir.convert_layout %[[select]]
module {
  func.func @test_propagate_convert_layout_down_scf_if(%arg0: i32){
    %true = arith.constant true
    %c128 = arith.constant 128 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %alloc_0 = memref.alloc() : memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>
    %alloc_1 = memref.alloc() : memref<128x128xbf16, #hivm.address_space<cbuf>>
    %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %1 = scf.if %0 -> (memref<128x128xbf16, #hivm.address_space<cbuf>>) {
      %2 = hivm.hir.convert_layout %alloc {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<128x128xbf16, #hivm.address_space<cbuf>>
      scf.yield %2 : memref<128x128xbf16, #hivm.address_space<cbuf>>
    } else {
      %2 = hivm.hir.convert_layout %alloc_0 {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<ND>} : (memref<8x8x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<128x128xbf16, #hivm.address_space<cbuf>>
      scf.yield %2 : memref<128x128xbf16, #hivm.address_space<cbuf>>
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 {already_set_real_mkn} ins(%1, %alloc_1, %true, %c128, %c128, %c128 : memref<128x128xbf16, #hivm.address_space<cbuf>>, memref<128x128xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_2 : memref<128x128xf32, #hivm.address_space<cc>>)
    return
  }
}
