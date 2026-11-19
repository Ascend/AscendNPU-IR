// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize | FileCheck %s --check-prefix=SINGLE
// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize | FileCheck %s --check-prefix=PAIR
// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize | FileCheck %s --check-prefix=HOIST
// RUN: bishengir-opt %s -hivm-optimize-row-loads="fallback-loop=true" -canonicalize | FileCheck %s --check-prefix=HOIST-LANE
// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize | FileCheck %s --check-prefix=FILL
// RUN: bishengir-opt %s -hivm-optimize-row-loads | FileCheck %s --check-prefix=GUARD

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Pairing preserves the exact fallback, including its sign.
  // Source subviews below exercise mask sinking without pairable descriptors.
  // PAIR-LABEL: func.func @masked_negative_zero(
  // PAIR: scf.for {{.*}} step %c2
  // PAIR: %[[BOTH:.*]] = arith.andi
  // PAIR: scf.if %[[BOTH]] {
  // PAIR: sizes: [2, 16], strides: [{{.*}}, 1]
  // PAIR: hivm.hir.load
  // PAIR: } else {
  // PAIR: scf.if
  // PAIR: hivm.hir.load
  // PAIR-NOT: } else {
  // PAIR-NOT: hivm.hir.vbrc
  // PAIR: scf.if
  // PAIR: hivm.hir.load
  // PAIR-NOT: } else {
  // PAIR-NOT: hivm.hir.vbrc
  // PAIR-NOT: arith.select
  // PAIR: hivm.hir.vadd
  // PAIR: return
  // HOIST-LABEL: func.func @masked_negative_zero(
  // HOIST: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<4x16xf16>)
  // HOIST: scf.for {{.*}} step %c2
  // HOIST: scf.if {{.*}} {
  // HOIST: sizes: [2, 16], strides: [{{.*}}, 1]
  // HOIST: } else {
  // HOIST: scf.if {{.*}} {
  // HOIST: hivm.hir.load
  // HOIST-NOT: hivm.hir.vbrc
  // HOIST: scf.if {{.*}} {
  // HOIST: hivm.hir.load
  // HOIST-NOT: hivm.hir.vbrc
  // HOIST: } {hivm.parallel_loop}
  // HOIST: hivm.hir.vadd
  // HOIST-LANE-LABEL: func.func @masked_negative_zero(
  // HOIST-LANE: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<4x16xf16>)
  // HOIST-LANE: scf.for {{.*}} step %c2
  // HOIST-LANE: sizes: [2, 16], strides: [{{.*}}, 1]
  // HOIST-LANE: } else {
  // HOIST-LANE: scf.for {{.*}} step %c1 {
  // HOIST-LANE: arith.select
  // HOIST-LANE: scf.if {{.*}} {
  // HOIST-LANE: hivm.hir.load
  // HOIST-LANE-NOT: hivm.hir.vbrc
  // HOIST-LANE: } {hivm.parallel_loop}
  // HOIST-LANE: hivm.hir.vadd
  // FILL-LABEL: func.func @masked_negative_zero(
  // FILL: %[[NEGZERO:.*]] = arith.constant -0.000000e+00 : f16
  // FILL: hivm.hir.vbrc ins(%[[NEGZERO]] : f16)
  // FILL: return
  func.func @masked_negative_zero(%base: memref<?xf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %fill = arith.constant -0.0 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x16xi1>) dimensions = [1]
    %fill_init = tensor.empty() : tensor<4x16xf16>
    %fallback = hivm.hir.vbrc ins(%fill : f16) outs(%fill_init : tensor<4x16xf16>) -> tensor<4x16xf16>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<4x16xi1>, tensor<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

  // An odd trip count pairs only its full prefix; the final row keeps its
  // predicate and must not extract a nonexistent fourth ID.
  // PAIR-LABEL: func.func @odd_masked_vsel(
  // PAIR: sizes: [2, 16], strides: [{{.*}}, 1]
  // PAIR: hivm.hir.load
  // PAIR: tensor.extract %arg1[%c2]
  // PAIR: sizes: [1, 16]
  // PAIR: scf.if
  // PAIR: hivm.hir.load
  // PAIR-NOT: } else {
  // PAIR-NOT: hivm.hir.vbrc
  // PAIR-NOT: tensor.extract %arg1[%c3]
  // PAIR-NOT: hivm.hir.vsel
  // PAIR: hivm.hir.vadd
  // PAIR: return
  func.func @odd_masked_vsel(%base: memref<?xf16>, %ids: tensor<3xi32>,
      %other: tensor<3x16xf16>) -> tensor<3x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %fill = arith.constant 7.5 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 3 : index
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<3x16xf16>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<3xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<3x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<3x16xf16>
    %mask_init = tensor.empty() : tensor<3xi1>
    %mask = hivm.hir.vcmp ins(%ids, %zero : tensor<3xi32>, i32)
      outs(%mask_init : tensor<3xi1>) compare_mode = <ge> -> tensor<3xi1>
    %expanded = tensor.expand_shape %mask [[0, 1]] output_shape [3, 1]
      : tensor<3xi1> into tensor<3x1xi1>
    %broadcast_init = tensor.empty() : tensor<3x16xi1>
    %full_mask = hivm.hir.vbrc ins(%expanded : tensor<3x1xi1>)
      outs(%broadcast_init : tensor<3x16xi1>) broadcast_dims = [1] -> tensor<3x16xi1>
    %select_init = tensor.empty() : tensor<3x16xf16>
    %selected = hivm.hir.vsel ins(%full_mask, %loaded, %fill : tensor<3x16xi1>, tensor<3x16xf16>, f16)
      outs(%select_init : tensor<3x16xf16>) -> tensor<3x16xf16>
    %empty = tensor.empty() : tensor<3x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<3x16xf16>, tensor<3x16xf16>)
      outs(%empty : tensor<3x16xf16>) -> tensor<3x16xf16>
    return %sum : tensor<3x16xf16>
  }

  // A select with gathered data in its false branch inverts the row predicate.
  // SINGLE-LABEL: func.func @inverse_mask(
  // SINGLE: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<4x16xf16>)
  // SINGLE: scf.for {{.*}} step %c1
  // SINGLE: arith.cmpi slt
  // SINGLE: scf.if
  // SINGLE: hivm.hir.load
  // SINGLE-NOT: } else {
  // SINGLE-NOT: hivm.hir.vbrc
  // SINGLE-NOT: arith.select
  // SINGLE: hivm.hir.vadd
  // SINGLE: return
  func.func @inverse_mask(%base: memref<?x16xf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %fallback = arith.constant dense<1.0> : tensor<4x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 16] [1, 1]
        : memref<?x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x16xi1>) dimensions = [1]
    %selected = arith.select %full_mask, %fallback, %loaded : tensor<4x16xi1>, tensor<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

  // A different predicate for each column cannot guard a whole-row DMA.
  // SINGLE-LABEL: func.func @column_mask(
  // SINGLE: scf.for {{.*}} step %c1
  // SINGLE-NOT: scf.if
  // SINGLE: hivm.hir.load
  // SINGLE: } {hivm.parallel_loop{{.*}}}
  // SINGLE: arith.select
  // SINGLE: hivm.hir.vadd
  // SINGLE: return
  func.func @column_mask(%base: memref<?x16xf16>, %ids: tensor<4xi32>,
      %mask: tensor<4x16xi1>, %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %fallback = arith.constant dense<0.0> : tensor<4x16xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 16] [1, 1]
        : memref<?x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %selected = arith.select %mask, %loaded, %fallback : tensor<4x16xi1>, tensor<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

  // BF16 selects are legalized through F32. Sinking removes the select while
  // retaining the original casts and preserving the fallback's BF16 -0.
  // PAIR-LABEL: func.func @bf16_legalized_mask(
  // PAIR: scf.for {{.*}} step %c2
  // PAIR: sizes: [2, 16]
  // PAIR: hivm.hir.load
  // PAIR: hivm.hir.vcast {{.*}} : tensor<4x16xbf16>) {{.*}} : tensor<4x16xf32>)
  // PAIR-NOT: hivm.hir.vsel
  // PAIR: hivm.hir.vcast {{.*}} : tensor<4x16xf32>) {{.*}} : tensor<4x16xbf16>)
  // PAIR: hivm.hir.vmax
  // PAIR: return
  // FILL-LABEL: func.func @bf16_legalized_mask(
  // FILL: %[[BF16NEGZERO:.*]] = arith.constant -0.000000e+00 : bf16
  // FILL: hivm.hir.vbrc ins(%[[BF16NEGZERO]] : bf16)
  // FILL: return
  func.func @bf16_legalized_mask(%base: memref<?xbf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xbf16>) -> tensor<4x16xbf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %fill = arith.constant -0.0 : f32
    %dst = memref.alloc() : memref<4x16xbf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xbf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xbf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xbf16>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x16xi1>) dimensions = [1]
    %cast_init = tensor.empty() : tensor<4x16xf32>
    %wide = hivm.hir.vcast ins(%loaded : tensor<4x16xbf16>)
      outs(%cast_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    %select_init = tensor.empty() : tensor<4x16xf32>
    %masked = hivm.hir.vsel ins(%full_mask, %wide, %fill : tensor<4x16xi1>, tensor<4x16xf32>, f32)
      outs(%select_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    %narrow_init = tensor.empty() : tensor<4x16xbf16>
    %selected = hivm.hir.vcast ins(%masked : tensor<4x16xf32>)
      outs(%narrow_init : tensor<4x16xbf16>) -> tensor<4x16xbf16>
    %empty = tensor.empty() : tensor<4x16xbf16>
    %sum = hivm.hir.vmax ins(%selected, %other : tensor<4x16xbf16>, tensor<4x16xbf16>)
      outs(%empty : tensor<4x16xbf16>) -> tensor<4x16xbf16>
    return %sum : tensor<4x16xbf16>
  }


  // A square transpose preserves the type but changes the data. It cannot
  // disappear as part of the BF16 mask legalization.
  // SINGLE-LABEL: func.func @bf16_transposed_cast(
  // SINGLE: scf.for {{.*}} step %c1
  // SINGLE-NOT: scf.if
  // SINGLE: hivm.hir.load
  // SINGLE: hivm.hir.vcast {{.*}} transpose = [1, 0]
  // SINGLE: hivm.hir.vsel
  // SINGLE: hivm.hir.vcast
  // SINGLE: hivm.hir.vmax
  // SINGLE: return
  func.func @bf16_transposed_cast(%base: memref<?x16xbf16>, %ids: tensor<16xi32>,
      %other: tensor<16x16xbf16>) -> tensor<16x16xbf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : i32
    %fill = arith.constant 0.0 : f32
    %dst = memref.alloc() : memref<16x16xbf16>
    scf.for %i = %c0 to %c16 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<16xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 16] [1, 1]
        : memref<?x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<16x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xbf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xbf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<16x16xbf16>
    %zeros = arith.constant dense<0> : tensor<16xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<16xi32>
    %mask_init = tensor.empty() : tensor<16x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<16xi1>)
      outs(%mask_init : tensor<16x16xi1>) dimensions = [1]
    %cast_init = tensor.empty() : tensor<16x16xf32>
    %wide = hivm.hir.vcast ins(%loaded : tensor<16x16xbf16>)
      outs(%cast_init : tensor<16x16xf32>) transpose = [1, 0] -> tensor<16x16xf32>
    %select_init = tensor.empty() : tensor<16x16xf32>
    %masked = hivm.hir.vsel ins(%full_mask, %wide, %fill : tensor<16x16xi1>, tensor<16x16xf32>, f32)
      outs(%select_init : tensor<16x16xf32>) -> tensor<16x16xf32>
    %narrow_init = tensor.empty() : tensor<16x16xbf16>
    %selected = hivm.hir.vcast ins(%masked : tensor<16x16xf32>)
      outs(%narrow_init : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    %empty = tensor.empty() : tensor<16x16xbf16>
    %result = hivm.hir.vmax ins(%selected, %other : tensor<16x16xbf16>, tensor<16x16xbf16>)
      outs(%empty : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    return %result : tensor<16x16xbf16>
  }

  // Scalarizing only one result of an extended arithmetic operation would
  // change its result signature. Keep this unsupported mask after the loop.
  // SINGLE-LABEL: func.func @multi_result_mask(
  // SINGLE: scf.for {{.*}} step %c1
  // SINGLE-NOT: scf.if
  // SINGLE: hivm.hir.load
  // SINGLE: arith.addui_extended
  // SINGLE: hivm.hir.vsel
  // SINGLE: hivm.hir.vadd
  // SINGLE: return
  func.func @multi_result_mask(%base: memref<?x16xf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %fill = arith.constant 0.0 : f16
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 16] [1, 1]
        : memref<?x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %ones = arith.constant dense<1> : tensor<4xi32>
    %sum, %mask = arith.addui_extended %ids, %ones : tensor<4xi32>, tensor<4xi1>
    %mask_init = tensor.empty() : tensor<4x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x16xi1>) dimensions = [1]
    %select_init = tensor.empty() : tensor<4x16xf16>
    %selected = hivm.hir.vsel ins(%full_mask, %loaded, %fill : tensor<4x16xi1>, tensor<4x16xf16>, f16)
      outs(%select_init : tensor<4x16xf16>) -> tensor<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %result = hivm.hir.vadd ins(%selected, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %result : tensor<4x16xf16>
  }

  // BF16 legalization can keep subsequent vector arithmetic in F32. Move
  // only the mask into the row loop and preserve both real conversions.
  // SINGLE-LABEL: func.func @bf16_fused_vector_mask(
  // SINGLE: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<4x16xbf16>)
  // SINGLE: scf.for {{.*}} step %c1
  // SINGLE: scf.if
  // SINGLE: hivm.hir.load
  // SINGLE-NOT: } else {
  // SINGLE-NOT: hivm.hir.vbrc
  // SINGLE-NOT: hivm.hir.vsel
  // SINGLE: hivm.hir.vcast
  // SINGLE: hivm.hir.vmul
  // SINGLE: hivm.hir.vcast
  // SINGLE: hivm.hir.vmax
  // SINGLE: return
  func.func @bf16_fused_vector_mask(%base: memref<?x16xbf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xbf16>) -> tensor<4x16xbf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %zero = arith.constant 0 : i32
    %fill = arith.constant -0.0 : f32
    %scale = arith.constant 1.5 : f32
    %dst = memref.alloc() : memref<4x16xbf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 16] [1, 1]
        : memref<?x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xbf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xbf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xbf16>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x16xi1>) dimensions = [1]
    %cast_init = tensor.empty() : tensor<4x16xf32>
    %wide = hivm.hir.vcast ins(%loaded : tensor<4x16xbf16>)
      outs(%cast_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    %select_init = tensor.empty() : tensor<4x16xf32>
    %masked = hivm.hir.vsel ins(%full_mask, %wide, %fill : tensor<4x16xi1>, tensor<4x16xf32>, f32)
      outs(%select_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    %mul_init = tensor.empty() : tensor<4x16xf32>
    %multiplied = hivm.hir.vmul ins(%masked, %scale : tensor<4x16xf32>, f32)
      outs(%mul_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    %narrow_init = tensor.empty() : tensor<4x16xbf16>
    %selected = hivm.hir.vcast ins(%multiplied : tensor<4x16xf32>)
      outs(%narrow_init : tensor<4x16xbf16>) -> tensor<4x16xbf16>
    %empty = tensor.empty() : tensor<4x16xbf16>
    %sum = hivm.hir.vmax ins(%selected, %other : tensor<4x16xbf16>, tensor<4x16xbf16>)
      outs(%empty : tensor<4x16xbf16>) -> tensor<4x16xbf16>
    return %sum : tensor<4x16xbf16>
  }

  // A negated row predicate is scalarized into the row guard.
  // GUARD-LABEL: func.func @masked_vnot(
  // GUARD: scf.for {{.*}} step %c1
  // GUARD: %[[CMP:.*]] = arith.cmpi eq
  // GUARD: %[[NOT:.*]] = arith.xori %[[CMP]], %true
  // GUARD: scf.if %[[NOT]] {
  // GUARD: hivm.hir.load
  // GUARD-NOT: } else {
  // GUARD-NOT: hivm.hir.vbrc
  // GUARD: } {hivm.parallel_loop}
  // SINGLE-LABEL: func.func @masked_vnot(
  // SINGLE: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<4x16xf16>)
  // SINGLE: scf.for {{.*}} step %c1
  // SINGLE: scf.if
  // SINGLE: hivm.hir.load
  // SINGLE-NOT: } else {
  // SINGLE-NOT: hivm.hir.vbrc
  // SINGLE: } {hivm.parallel_loop}
  // SINGLE-NOT: hivm.hir.vsel
  // SINGLE: hivm.hir.vadd
  // SINGLE: return
  func.func @masked_vnot(%base: memref<?x16xf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %fill = arith.constant 0.0 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %c-1 = arith.constant -1 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 16] [1, 1]
        : memref<?x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %cmp_init = tensor.empty() : tensor<4xi1>
    %cmp = hivm.hir.vcmp ins(%ids, %c-1 : tensor<4xi32>, i32)
      outs(%cmp_init : tensor<4xi1>) compare_mode = <eq> -> tensor<4xi1>
    %not_init = tensor.empty() : tensor<4xi1>
    %not = hivm.hir.vnot ins(%cmp : tensor<4xi1>)
      outs(%not_init : tensor<4xi1>) -> tensor<4xi1>
    %mask_init = tensor.empty() : tensor<4x16xi1>
    %full_mask = linalg.broadcast ins(%not : tensor<4xi1>)
      outs(%mask_init : tensor<4x16xi1>) dimensions = [1]
    %select_init = tensor.empty() : tensor<4x16xf16>
    %selected = hivm.hir.vsel ins(%full_mask, %loaded, %fill : tensor<4x16xi1>, tensor<4x16xf16>, f16)
      outs(%select_init : tensor<4x16xf16>) -> tensor<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

  // An external per-row mask cannot be rebuilt as scalar arithmetic, so the
  // general mask path reads it one element at a time in the loop.
  // GUARD-LABEL: func.func @masked_external(
  // GUARD: %[[BYTES:.*]] = hivm.hir.vcast ins(%arg2 : tensor<4xi1>)
  // GUARD: scf.for {{.*}} step %c1
  // GUARD: %[[BYTE:.*]] = tensor.extract %[[BYTES]][{{.*}}] : tensor<4xi8>
  // GUARD: %[[M:.*]] = arith.cmpi ne, %[[BYTE]]
  // GUARD: scf.if %[[M]] {
  // GUARD: hivm.hir.load
  // GUARD-NOT: } else {
  // GUARD-NOT: hivm.hir.vbrc
  // GUARD: } {hivm.parallel_loop}
  // GUARD-NOT: arith.select
  // GUARD: hivm.hir.vadd
  // GUARD: return
  func.func @masked_external(%base: memref<?x16xf16>, %ids: tensor<4xi32>,
      %mask: tensor<4xi1>, %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %fill = arith.constant 0.0 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %rows step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 16] [1, 1]
        : memref<?x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %mask_init = tensor.empty() : tensor<4x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x16xi1>) dimensions = [1]
    %fill_init = tensor.empty() : tensor<4x16xf16>
    %fallback = hivm.hir.vbrc ins(%fill : f16) outs(%fill_init : tensor<4x16xf16>) -> tensor<4x16xf16>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<4x16xi1>, tensor<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }
}
