// REQUIRES: hivmc, enable-lir-compile
// Keep the DMA instructions in the entry functions for per-slice checks.
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-hfusion-compile=false \
// RUN:   --enable-hivm-compile=true --disable-ffts --enable-static-bare-ptr \
// RUN:   --enable-lib-call-no-inline=false -o %t.o
// RUN: sh -c 'exec "$("$1" -print-prog-name=llvm-objdump)" -d "$2"' sh "%bisheng_compiler" "%t.o" | FileCheck %s

// CHECK-LABEL: <short_rows{{(\$local)?}}>:
// CHECK-NOT: MOV_OUT_TO_UB_ALIGN_V2
// CHECK: ND_DMA_OUT_TO_UB
// CHECK-NOT: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// A03: 8B rows at a 16B GM pitch.
!short_rows_src = memref<7680x4xi16, strided<[8, 1]>, #hivm.address_space<gm>>
!short_rows_tile = memref<7680x4xi16, #hivm.address_space<ub>>
!short_rows_out = memref<7680x4xi16, #hivm.address_space<gm>>
func.func @short_rows(%src: !short_rows_src, %out: !short_rows_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !short_rows_tile
  hivm.hir.load ins(%src : !short_rows_src) outs(%tile : !short_rows_tile)
  hivm.hir.store ins(%tile : !short_rows_tile) outs(%out : !short_rows_out)
  return
}

// CHECK-LABEL: <short_rows_b8{{(\$local)?}}>:
// CHECK-NOT: MOV_OUT_TO_UB_ALIGN_V2
// CHECK: ND_DMA_OUT_TO_UB
// CHECK-NOT: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// 8B rows at a 12B GM pitch with 8-bit elements, at the 128-row threshold.
!short_rows_b8_src = memref<128x8xi8, strided<[12, 1]>, #hivm.address_space<gm>>
!short_rows_b8_tile = memref<128x8xi8, #hivm.address_space<ub>>
!short_rows_b8_out = memref<128x8xi8, #hivm.address_space<gm>>
func.func @short_rows_b8(%src: !short_rows_b8_src, %out: !short_rows_b8_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !short_rows_b8_tile
  hivm.hir.load ins(%src : !short_rows_b8_src) outs(%tile : !short_rows_b8_tile)
  hivm.hir.store ins(%tile : !short_rows_b8_tile) outs(%out : !short_rows_b8_out)
  return
}

// CHECK-LABEL: <short_rows_b32{{(\$local)?}}>:
// CHECK-NOT: MOV_OUT_TO_UB_ALIGN_V2
// CHECK: ND_DMA_OUT_TO_UB
// CHECK-NOT: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// A 32B GM pitch reaches the larger-pitch threshold at 320 rows.
!short_rows_b32_src = memref<320x2xi32, strided<[8, 1]>, #hivm.address_space<gm>>
!short_rows_b32_tile = memref<320x2xi32, #hivm.address_space<ub>>
!short_rows_b32_out = memref<320x2xi32, #hivm.address_space<gm>>
func.func @short_rows_b32(%src: !short_rows_b32_src, %out: !short_rows_b32_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !short_rows_b32_tile
  hivm.hir.load ins(%src : !short_rows_b32_src) outs(%tile : !short_rows_b32_tile)
  hivm.hir.store ins(%tile : !short_rows_b32_tile) outs(%out : !short_rows_b32_out)
  return
}

// CHECK-LABEL: <few_rows{{(\$local)?}}>:
// CHECK-NOT: ND_DMA_OUT_TO_UB
// CHECK: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// A 12B GM pitch is one row below its 128-row threshold.
!few_rows_src = memref<127x4xi16, strided<[6, 1]>, #hivm.address_space<gm>>
!few_rows_tile = memref<127x4xi16, #hivm.address_space<ub>>
!few_rows_out = memref<127x4xi16, #hivm.address_space<gm>>
func.func @few_rows(%src: !few_rows_src, %out: !few_rows_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !few_rows_tile
  hivm.hir.load ins(%src : !few_rows_src) outs(%tile : !few_rows_tile)
  hivm.hir.store ins(%tile : !few_rows_tile) outs(%out : !few_rows_out)
  return
}

// CHECK-LABEL: <larger_pitch{{(\$local)?}}>:
// CHECK-NOT: ND_DMA_OUT_TO_UB
// CHECK: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// At 128 rows, increasing the GM pitch from 12B to 13B selects MOV.
!larger_pitch_src = memref<128x8xi8, strided<[13, 1]>, #hivm.address_space<gm>>
!larger_pitch_tile = memref<128x8xi8, #hivm.address_space<ub>>
!larger_pitch_out = memref<128x8xi8, #hivm.address_space<gm>>
func.func @larger_pitch(%src: !larger_pitch_src, %out: !larger_pitch_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !larger_pitch_tile
  hivm.hir.load ins(%src : !larger_pitch_src) outs(%tile : !larger_pitch_tile)
  hivm.hir.store ins(%tile : !larger_pitch_tile) outs(%out : !larger_pitch_out)
  return
}

// CHECK-LABEL: <below_320_rows{{(\$local)?}}>:
// CHECK-NOT: ND_DMA_OUT_TO_UB
// CHECK: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// Eight i32 elements give a 32B GM pitch; 319 rows still use MOV.
!below_320_rows_src = memref<319x2xi32, strided<[8, 1]>, #hivm.address_space<gm>>
!below_320_rows_tile = memref<319x2xi32, #hivm.address_space<ub>>
!below_320_rows_out = memref<319x2xi32, #hivm.address_space<gm>>
func.func @below_320_rows(%src: !below_320_rows_src, %out: !below_320_rows_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !below_320_rows_tile
  hivm.hir.load ins(%src : !below_320_rows_src) outs(%tile : !below_320_rows_tile)
  hivm.hir.store ins(%tile : !below_320_rows_tile) outs(%out : !below_320_rows_out)
  return
}

// CHECK-LABEL: <contiguous{{(\$local)?}}>:
// CHECK-NOT: ND_DMA_OUT_TO_UB
// CHECK: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// The GM rows have no gap.
!contiguous_src = memref<7680x4xi16, strided<[4, 1]>, #hivm.address_space<gm>>
!contiguous_tile = memref<7680x4xi16, #hivm.address_space<ub>>
!contiguous_out = memref<7680x4xi16, #hivm.address_space<gm>>
func.func @contiguous(%src: !contiguous_src, %out: !contiguous_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !contiguous_tile
  hivm.hir.load ins(%src : !contiguous_src) outs(%tile : !contiguous_tile)
  hivm.hir.store ins(%tile : !contiguous_tile) outs(%out : !contiguous_out)
  return
}

// CHECK-LABEL: <large_pitch{{(\$local)?}}>:
// CHECK-NOT: ND_DMA_OUT_TO_UB
// CHECK: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// The 64B GM pitch exceeds the 32B limit.
!large_pitch_src = memref<7680x4xi16, strided<[32, 1]>, #hivm.address_space<gm>>
!large_pitch_tile = memref<7680x4xi16, #hivm.address_space<ub>>
!large_pitch_out = memref<7680x4xi16, #hivm.address_space<gm>>
func.func @large_pitch(%src: !large_pitch_src, %out: !large_pitch_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !large_pitch_tile
  hivm.hir.load ins(%src : !large_pitch_src) outs(%tile : !large_pitch_tile)
  hivm.hir.store ins(%tile : !large_pitch_tile) outs(%out : !large_pitch_out)
  return
}

// CHECK-LABEL: <wide_rows{{(\$local)?}}>:
// CHECK-NOT: ND_DMA_OUT_TO_UB
// CHECK: MOV_OUT_TO_UB_ALIGN_V2
// CHECK-NOT: ND_DMA_OUT_TO_UB
// The 24B rows exceed the 16B limit.
!wide_rows_src = memref<1920x12xi16, strided<[16, 1]>, #hivm.address_space<gm>>
!wide_rows_tile = memref<1920x12xi16, #hivm.address_space<ub>>
!wide_rows_out = memref<1920x12xi16, #hivm.address_space<gm>>
func.func @wide_rows(%src: !wide_rows_src, %out: !wide_rows_out)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
                hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tile = memref.alloc() : !wide_rows_tile
  hivm.hir.load ins(%src : !wide_rows_src) outs(%tile : !wide_rows_tile)
  hivm.hir.store ins(%tile : !wide_rows_tile) outs(%out : !wide_rows_out)
  return
}
