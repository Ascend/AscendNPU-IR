// RUN: bishengir-opt %s --hivm-promote-l1-gather | FileCheck %s
// RUN: sed 's/enable_l1_gather = true/enable_l1_gather = false/g' %s | bishengir-opt --hivm-promote-l1-gather | FileCheck %s --check-prefix=DISABLED
// RUN: sed '/annotation.mark.*hivm.enable_l1_gather/d' %s | bishengir-opt --hivm-promote-l1-gather | FileCheck %s --check-prefix=DISABLED
// RUN: sed '/annotation.mark %%tile1 /d' %s | bishengir-opt --hivm-promote-l1-gather | FileCheck %s --check-prefix=ONE-HINT
// RUN: sed 's/left_padding_num = %%c0/left_padding_num = %%c1/g' %s | bishengir-opt --hivm-promote-l1-gather | FileCheck %s --check-prefix=DISABLED
// RUN: sed '/^    %%buffer1 = memref.alloc/i\    memref.store %%id_zero, %%ids[%%c0] : memref<16xi32>' %s | bishengir-opt --hivm-promote-l1-gather | FileCheck %s --check-prefix=SEPARATE

// Reduced from adjacent MLA column gathers. Preserve the guarded metadata
// reads and payload loads, and combine the two rows into one native DMA.
// Keep the surviving score-mask conversion beside its predicate, so fusion
// does not spill i1 between the metadata and score vector functions.
// CHECK-LABEL: func.func @adjacent_columns
// CHECK: %[[MASK:.*]] = hivm.hir.vand
// CHECK-NEXT: %[[ENCODED:.*]] = hivm.hir.vsel ins(%[[MASK]]
// CHECK: memref.alloc() : memref<2x1x16x16xbf16, #hivm.address_space<cbuf>>
// CHECK: scf.for
// CHECK: memref.load
// CHECK: scf.if
// CHECK: hivm.hir.nd2nz {{.*}} memref<1x32xbf16
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: annotation.mark
// CHECK-NOT: ExtractedLoadOrStore
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.mmadL1
// CHECK: tensor.expand_shape %[[ENCODED]]
// CHECK: hivm.hir.vbrc
// DISABLED-LABEL: func.func @adjacent_columns
// DISABLED: hivm.hir.vand
// DISABLED: hivm.hir.load
// DISABLED: hivm.hir.load
// DISABLED-NOT: hivm.hir.nd2nz
// DISABLED: hivm.hir.vsel
// ONE-HINT-LABEL: func.func @adjacent_columns
// ONE-HINT: hivm.hir.nd2nz {{.*}} memref<1x16xbf16
// ONE-HINT: hivm.hir.load
// ONE-HINT-NOT: hivm.hir.nd2nz
// ONE-HINT: hivm.hir.mmadL1
// ONE-HINT: hivm.hir.mmadL1
// SEPARATE-LABEL: func.func @adjacent_columns
// SEPARATE: hivm.hir.nd2nz {{.*}} memref<1x16xbf16
// SEPARATE: memref.store
// SEPARATE: hivm.hir.nd2nz {{.*}} memref<1x16xbf16
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @adjacent_columns(%ids: memref<16xi32>, %gm: memref<?xbf16>,
                              %q: tensor<16x16xbf16>, %acc: tensor<16x16xf32>,
                              %in_range: tensor<16xi1>, %nonnegative: tensor<16xi1>)
      -> (tensor<16x16xf32>, tensor<16x16xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant 0.0 : bf16
    %id_zero = arith.constant 0 : i32
    %true = arith.constant true
    %false = arith.constant false
    %mask_true = arith.constant -1.0 : f16
    %mask_false = arith.constant 0.0 : f16
    %mask_empty = tensor.empty() : tensor<16xi1>
    %mask = hivm.hir.vand ins(%in_range, %nonnegative : tensor<16xi1>, tensor<16xi1>) outs(%mask_empty : tensor<16xi1>) -> tensor<16xi1>
    %buffer0 = memref.alloc() : memref<16x16xbf16>
    hivm.hir.vbrc ins(%zero : bf16) outs(%buffer0 : memref<16x16xbf16>)
    scf.for %i = %c0 to %c16 step %c1 {
      %id = memref.load %ids[%i] : memref<16xi32>
      %valid = arith.cmpi sge, %id, %id_zero : i32
      %index = arith.index_cast %id : i32 to index
      %offset = arith.muli %index, %c32 : index
      %src = memref.reinterpret_cast %gm to offset: [%offset], sizes: [1, 16], strides: [16, 1] : memref<?xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %buffer0[%i, 0] [1, 16] [1, 1] : memref<16x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      scf.if %valid {
        hivm.hir.load ins(%src : memref<1x16xbf16, strided<[16, 1], offset: ?>>) outs(%row : memref<1x16xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %zero : bf16 left_padding_num = %c0 : index
      }
    } {ExtractedLoadOrStore, hivm.parallel_loop}
    %tile0 = bufferization.to_tensor %buffer0 restrict writable : memref<16x16xbf16>
    annotation.mark %tile0 {hivm.enable_l1_gather = true} keys = ["write"] : tensor<16x16xbf16>
    %buffer1 = memref.alloc() : memref<16x16xbf16>
    hivm.hir.vbrc ins(%zero : bf16) outs(%buffer1 : memref<16x16xbf16>)
    scf.for %i = %c0 to %c16 step %c1 {
      %id = memref.load %ids[%i] : memref<16xi32>
      %valid = arith.cmpi sge, %id, %id_zero : i32
      %index = arith.index_cast %id : i32 to index
      %offset = arith.muli %index, %c32 : index
      %column = arith.addi %offset, %c16 : index
      %src = memref.reinterpret_cast %gm to offset: [%column], sizes: [1, 16], strides: [16, 1] : memref<?xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %buffer1[%i, 0] [1, 16] [1, 1] : memref<16x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      scf.if %valid {
        hivm.hir.load ins(%src : memref<1x16xbf16, strided<[16, 1], offset: ?>>) outs(%row : memref<1x16xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %zero : bf16 left_padding_num = %c0 : index
      }
    } {ExtractedLoadOrStore, hivm.parallel_loop}
    %tile1 = bufferization.to_tensor %buffer1 restrict writable : memref<16x16xbf16>
    annotation.mark %tile1 {hivm.enable_l1_gather = true} keys = ["write"] : tensor<16x16xbf16>
    %r0 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, normalized_in_L0C} ins(%q, %tile0, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r1 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, normalized_in_L0C} ins(%q, %tile1, %false, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%r0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %encoded_empty = tensor.empty() : tensor<16xf16>
    %encoded = hivm.hir.vsel ins(%mask, %mask_true, %mask_false : tensor<16xi1>, f16, f16) outs(%encoded_empty : tensor<16xf16>) -> tensor<16xf16>
    %expanded = tensor.expand_shape %encoded [[0, 1]] output_shape [1, 16] : tensor<16xf16> into tensor<1x16xf16>
    %broadcast_empty = tensor.empty() : tensor<16x16xf16>
    %broadcast = hivm.hir.vbrc ins(%expanded : tensor<1x16xf16>) outs(%broadcast_empty : tensor<16x16xf16>) broadcast_dims = [0] -> tensor<16x16xf16>
    return %r1, %broadcast : tensor<16x16xf32>, tensor<16x16xf16>
  }

  // Paged decode gathers a dynamic prefix. Clear L1 only for partial tiles.
  // CHECK-LABEL: func.func @row_prefix
  // CHECK: memref.alloc() : memref<1x1x16x16xbf16, #hivm.address_space<cbuf>>
  // CHECK: %[[PARTIAL:.*]] = arith.cmpi ne
  // CHECK: scf.if %[[PARTIAL]]
  // CHECK: hivm.hir.vbrc {{.*}} #hivm.address_space<cbuf>
  // CHECK: scf.for
  // CHECK: memref.load
  // CHECK-NOT: scf.if
  // CHECK: hivm.hir.nd2nz
  // CHECK: hivm.hir.mmadL1
  func.func @row_prefix(%ids: memref<16xi32>, %gm: memref<?xbf16>,
                        %count: index, %q: tensor<16x16xbf16>,
                        %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.0 : bf16
    %true = arith.constant true
    %buffer = memref.alloc() : memref<16x16xbf16>
    hivm.hir.vbrc ins(%zero : bf16) outs(%buffer : memref<16x16xbf16>)
    scf.for %i = %c0 to %count step %c1 {
      %id = memref.load %ids[%i] : memref<16xi32>
      %index = arith.index_cast %id : i32 to index
      %offset = arith.muli %index, %c16 : index
      %src = memref.reinterpret_cast %gm to offset: [%offset], sizes: [1, 16], strides: [16, 1] : memref<?xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 16] [1, 1] : memref<16x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xbf16, strided<[16, 1], offset: ?>>) outs(%row : memref<1x16xbf16, strided<[16, 1], offset: ?>>) pad_mode = <PadValue> pad_value = %zero : bf16 left_padding_num = %c0 : index
    } {hivm.parallel_loop}
    %tile = bufferization.to_tensor %buffer restrict writable : memref<16x16xbf16>
    annotation.mark %tile {hivm.enable_l1_gather = true} keys = ["write"] : tensor<16x16xbf16>
    %result = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, normalized_in_L0C} ins(%q, %tile, %true, %c16, %c16, %c16 : tensor<16x16xbf16>, tensor<16x16xbf16>, i1, index, index, index) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %result : tensor<16x16xf32>
  }
}
