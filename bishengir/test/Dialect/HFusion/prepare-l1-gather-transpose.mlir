// RUN: bishengir-opt %s --hfusion-prepare-l1-gather --canonicalize | FileCheck %s --check-prefix=PREPARE --implicit-check-not=tensor.extract
// RUN: bishengir-opt %s --hfusion-prepare-l1-gather --canonicalize --convert-arith-to-hfusion --convert-hfusion-to-hivm="mm-map-mode=macro_instr" --convert-to-hivm-op --hivm-normalize-matmul --canonicalize --hivm-promote-l1-gather --canonicalize | FileCheck %s --check-prefix=PROMOTE
// RUN: sed '/annotation.mark /{s/%%keys /%%selected /;s/256x16/16x256/;}' %s | bishengir-opt --hfusion-prepare-l1-gather --canonicalize --convert-arith-to-hfusion --convert-hfusion-to-hivm="mm-map-mode=macro_instr" --convert-to-hivm-op --hivm-normalize-matmul --canonicalize --hivm-promote-l1-gather --canonicalize | FileCheck %s --check-prefix=PROMOTE
// RUN: sed 's/enable_l1_gather = true/enable_l1_gather = false/' %s | bishengir-opt --hfusion-prepare-l1-gather --canonicalize --convert-arith-to-hfusion --convert-hfusion-to-hivm="mm-map-mode=macro_instr" --convert-to-hivm-op --hivm-normalize-matmul --canonicalize --hivm-promote-l1-gather --canonicalize | FileCheck %s --check-prefix=DISABLED
// RUN: sed '/annotation.mark .*enable_l1_gather/d' %s | bishengir-opt --hfusion-prepare-l1-gather --canonicalize --convert-arith-to-hfusion --convert-hfusion-to-hivm="mm-map-mode=macro_instr" --convert-to-hivm-op --hivm-normalize-matmul --canonicalize --hivm-promote-l1-gather --canonicalize | FileCheck %s --check-prefix=DISABLED

// Reduced from sparse_gqa_prefill's original K load. The frontend gathers
// contiguous rows, transposes the result, then attaches the compile hint.
// The explicit tl.trans spelling instead attaches the hint before transpose.
// Both must preserve the row mask and reach L1 after transpose folds into MMAD.
// Keep the expanded/reshaped index calculation from the original spelling.
// PREPARE-LABEL: func.func @sparse_gqa_key
// PREPARE: linalg.fill {{.*}} outs({{.*}} : memref<16x256xbf16>)
// PREPARE: scf.for
// PREPARE: %[[ID:.*]] = scf.if
// PREPARE: memref.load
// PREPARE: %[[VALID:.*]] = arith.cmpi sge, %[[ID]]
// PREPARE: arith.muli %[[ID]], {{.*}} : i32
// PREPARE: scf.if %[[VALID]]
// PREPARE: memref.copy {{.*}} to memref<1x256xbf16
// PREPARE: %[[ROWS:.*]] = bufferization.to_tensor {{.*}} : memref<16x256xbf16>
// PREPARE-NOT: arith.select
// PREPARE: linalg.transpose ins(%[[ROWS]] : tensor<16x256xbf16>)
// PREPARE: annotation.mark %[[ROWS]] {hivm.enable_l1_gather = true} : tensor<16x256xbf16>
// PREPARE: linalg.matmul
// PROMOTE-LABEL: func.func @sparse_gqa_key
// PROMOTE: memref.alloc() : memref<16x1x16x16xbf16, #hivm.address_space<cbuf>>
// PROMOTE: hivm.hir.vbrc {{.*}} #hivm.address_space<cbuf>
// PROMOTE: scf.for
// PROMOTE: memref.load
// PROMOTE: scf.if
// PROMOTE: hivm.hir.nd2nz {{.*}} memref<1x256xbf16
// PROMOTE-NOT: hivm.hir.load
// PROMOTE-NOT: hivm.hir.vtranspose
// PROMOTE-NOT: annotation.mark
// PROMOTE: hivm.hir.mmadL1 {{.*}}b_transpose
// DISABLED-LABEL: func.func @sparse_gqa_key
// DISABLED-NOT: hivm.hir.nd2nz
// DISABLED: hivm.hir.load
// DISABLED-NOT: hivm.hir.nd2nz
// DISABLED: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @sparse_gqa_key(%ids: memref<16xi32>, %k: memref<?xbf16>,
                            %q: tensor<16x256xbf16>, %acc: tensor<16x16xf32>)
      -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0 : i32
    %zeros = arith.constant dense<0> : tensor<16xi32>
    %stride = arith.constant dense<256> : tensor<1x16xi32>
    %shape = arith.constant dense<[16, 1]> : tensor<2xi64>
    %other = arith.constant dense<0.0> : tensor<16x256xbf16>
    %indices = memref.alloc() : memref<16xi32>
    linalg.fill ins(%zero : i32) outs(%indices : memref<16xi32>)
    %prefix = memref.subview %indices[0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
    memref.copy %ids, %prefix : memref<16xi32> to memref<16xi32, strided<[1]>>
    %tokens = bufferization.to_tensor %indices restrict writable : memref<16xi32>
    %valid = arith.cmpi sge, %tokens, %zeros : tensor<16xi32>
    %expanded = tensor.expand_shape %tokens [[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
    %offsets = arith.muli %expanded, %stride : tensor<1x16xi32>
    %reshaped = tensor.reshape %offsets(%shape) : (tensor<1x16xi32>, tensor<2xi64>) -> tensor<16x1xi32>
    %mask_init = tensor.empty() : tensor<16x256xi1>
    %mask = linalg.broadcast ins(%valid : tensor<16xi1>) outs(%mask_init : tensor<16x256xi1>) dimensions = [1]
    %buffer = memref.alloc() : memref<16x256xbf16>
    scf.for %i = %c0 to %c16 step %c1 {
      %offset = tensor.extract %reshaped[%i, %c0] : tensor<16x1xi32>
      %index = arith.index_cast %offset : i32 to index
      %src = memref.reinterpret_cast %k to offset: [%index], sizes: [1, 256], strides: [256, 1] : memref<?xbf16> to memref<1x256xbf16, strided<[256, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 256] [1, 1] : memref<16x256xbf16> to memref<1x256xbf16, strided<[256, 1], offset: ?>>
      memref.copy %src, %row : memref<1x256xbf16, strided<[256, 1], offset: ?>> to memref<1x256xbf16, strided<[256, 1], offset: ?>>
    } {ExtractedLoadOrStore, hivm.parallel_loop}
    %tile = bufferization.to_tensor %buffer restrict writable : memref<16x256xbf16>
    %selected = arith.select %mask, %tile, %other : tensor<16x256xi1>, tensor<16x256xbf16>
    %init = tensor.empty() : tensor<256x16xbf16>
    %keys = linalg.transpose ins(%selected : tensor<16x256xbf16>) outs(%init : tensor<256x16xbf16>) permutation = [1, 0]
    annotation.mark %keys {hivm.enable_l1_gather = true} : tensor<256x16xbf16>
    %result = linalg.matmul ins(%q, %keys : tensor<16x256xbf16>, tensor<256x16xbf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %result : tensor<16x16xf32>
  }
}
