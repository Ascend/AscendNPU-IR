// RUN: bishengir-opt %s -hivm-optimize-row-loads | FileCheck %s

// Source subviews exercise mask sinking without a pairable source descriptor.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Sinking replaces only the select; the two widening casts and the distinct
  // integer suffix retain their order and rounding.
  // CHECK-LABEL: func.func @multilevel_i16_suffix(
  // CHECK: %[[FILL:.*]] = arith.constant 0 : i8
  // CHECK: hivm.hir.vbrc ins(%[[FILL]] : i8) outs({{.*}} : memref<4x32xi8>)
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK-NOT: } else {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK-NOT: arith.select
  // CHECK-NOT: hivm.hir.vsel
  // CHECK: %[[SUFFIX:.*]] = hivm.hir.vcast ins(%[[WIDE]] : tensor<4x32xf32>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xi16>)
  // CHECK: hivm.hir.vadd ins(%[[SUFFIX]]
  // CHECK: return
  func.func @multilevel_i16_suffix(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xi16>) -> tensor<4x32xi16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x32xf32>
    %selected = arith.select %full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>
    %suffix_init = tensor.empty() : tensor<4x32xi16>
    %suffix = hivm.hir.vcast ins(%selected : tensor<4x32xf32>)
      outs(%suffix_init : tensor<4x32xi16>) round_mode = <trunc> -> tensor<4x32xi16>
    %sum_init = tensor.empty() : tensor<4x32xi16>
    %sum = hivm.hir.vadd ins(%suffix, %other : tensor<4x32xi16>, tensor<4x32xi16>)
      outs(%sum_init : tensor<4x32xi16>) -> tensor<4x32xi16>
    return %sum : tensor<4x32xi16>
  }

  // A floating suffix with a different type also survives sinking through a
  // scalar-fallback vsel.
  // CHECK-LABEL: func.func @multilevel_f16_suffix(
  // CHECK: %[[FILL:.*]] = arith.constant 0 : i8
  // CHECK: hivm.hir.vbrc ins(%[[FILL]] : i8) outs({{.*}} : memref<4x32xi8>)
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK-NOT: } else {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK-NOT: arith.select
  // CHECK-NOT: hivm.hir.vsel
  // CHECK: %[[SUFFIX:.*]] = hivm.hir.vcast ins(%[[WIDE]] : tensor<4x32xf32>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>)
  // CHECK: hivm.hir.vadd ins(%[[SUFFIX]]
  // CHECK: return
  func.func @multilevel_f16_suffix(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xf16>) -> tensor<4x32xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant 0.0 : f32
    %select_init = tensor.empty() : tensor<4x32xf32>
    %selected = hivm.hir.vsel ins(%full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>, f32)
      outs(%select_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %suffix_init = tensor.empty() : tensor<4x32xf16>
    %suffix = hivm.hir.vcast ins(%selected : tensor<4x32xf32>)
      outs(%suffix_init : tensor<4x32xf16>) -> tensor<4x32xf16>
    %sum_init = tensor.empty() : tensor<4x32xf16>
    %sum = hivm.hir.vadd ins(%suffix, %other : tensor<4x32xf16>, tensor<4x32xf16>)
      outs(%sum_init : tensor<4x32xf16>) -> tensor<4x32xf16>
    return %sum : tensor<4x32xf16>
  }

  // A selected wide value needs no matching cast back to the loaded type.
  // CHECK-LABEL: func.func @multilevel_without_suffix(
  // CHECK: %[[FILL:.*]] = arith.constant 0 : i8
  // CHECK: hivm.hir.vbrc ins(%[[FILL]] : i8) outs({{.*}} : memref<4x32xi8>)
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK-NOT: } else {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK-NOT: arith.select
  // CHECK-NOT: hivm.hir.vsel
  // CHECK: hivm.hir.vadd ins(%[[WIDE]]
  // CHECK: return
  func.func @multilevel_without_suffix(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xf32>) -> tensor<4x32xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x32xf32>
    %selected = arith.select %full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>
    %sum_init = tensor.empty() : tensor<4x32xf32>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x32xf32>, tensor<4x32xf32>)
      outs(%sum_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    return %sum : tensor<4x32xf32>
  }

  // The fallback has the exact signed i8 preimage -3 through both casts.
  // CHECK-LABEL: func.func @exact_nonzero_fallback(
  // CHECK: %[[FILL:.*]] = arith.constant -3 : i8
  // CHECK: hivm.hir.vbrc ins(%[[FILL]] : i8) outs({{.*}} : memref<4x32xi8>)
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK-NOT: } else {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK-NOT: arith.select
  // CHECK-NOT: hivm.hir.vsel
  // CHECK: hivm.hir.vadd ins(%[[WIDE]]
  // CHECK: return
  func.func @exact_nonzero_fallback(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xf32>) -> tensor<4x32xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant -3.0 : f32
    %select_init = tensor.empty() : tensor<4x32xf32>
    %selected = hivm.hir.vsel ins(%full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>, f32)
      outs(%select_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %sum_init = tensor.empty() : tensor<4x32xf32>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x32xf32>, tensor<4x32xf32>)
      outs(%sum_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    return %sum : tensor<4x32xf32>
  }

  // One half has no i8 preimage. Classification succeeds without changing the
  // load or select.
  // CHECK-LABEL: func.func @fractional_fallback_keeps_select(
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  // CHECK: hivm.hir.load
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK: %[[SELECTED:.*]] = arith.select {{.*}}, %[[WIDE]],
  // CHECK: hivm.hir.vadd ins(%[[SELECTED]]
  // CHECK: return
  func.func @fractional_fallback_keeps_select(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xf32>) -> tensor<4x32xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.5> : tensor<4x32xf32>
    %selected = arith.select %full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>
    %sum_init = tensor.empty() : tensor<4x32xf32>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x32xf32>, tensor<4x32xf32>)
      outs(%sum_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    return %sum : tensor<4x32xf32>
  }

  // A separate user observes the unmasked intermediate cast, so guarding its
  // load would change that observation.
  // CHECK-LABEL: func.func @prefix_multiuser_keeps_select(
  // CHECK: scf.for
  // CHECK-NOT: scf.if
  // CHECK: hivm.hir.load
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK: %[[SELECTED:.*]] = arith.select {{.*}}, %[[WIDE]],
  // CHECK: %[[SUFFIX:.*]] = hivm.hir.vcast ins(%[[SELECTED]] : tensor<4x32xf32>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xi16>)
  // CHECK: hivm.hir.vadd ins(%[[SUFFIX]]
  // CHECK: hivm.hir.vadd ins(%[[HALF]]
  // CHECK: return
  func.func @prefix_multiuser_keeps_select(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xi16>, %half_other: tensor<4x32xf16>) -> (tensor<4x32xi16>, tensor<4x32xf16>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x32xf32>
    %selected = arith.select %full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>
    %suffix_init = tensor.empty() : tensor<4x32xi16>
    %suffix = hivm.hir.vcast ins(%selected : tensor<4x32xf32>)
      outs(%suffix_init : tensor<4x32xi16>) round_mode = <trunc> -> tensor<4x32xi16>
    %sum_init = tensor.empty() : tensor<4x32xi16>
    %sum = hivm.hir.vadd ins(%suffix, %other : tensor<4x32xi16>, tensor<4x32xi16>)
      outs(%sum_init : tensor<4x32xi16>) -> tensor<4x32xi16>
    %observed_init = tensor.empty() : tensor<4x32xf16>
    %observed = hivm.hir.vadd ins(%half, %half_other : tensor<4x32xf16>, tensor<4x32xf16>)
      outs(%observed_init : tensor<4x32xf16>) -> tensor<4x32xf16>
    return %sum, %observed : tensor<4x32xi16>, tensor<4x32xf16>
  }

  // Both users observe the masked wide result and can share the guarded data
  // path.
  // CHECK-LABEL: func.func @selected_multiuser(
  // CHECK: %[[FILL:.*]] = arith.constant 0 : i8
  // CHECK: hivm.hir.vbrc ins(%[[FILL]] : i8) outs({{.*}} : memref<4x32xi8>)
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK-NOT: } else {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK-NOT: arith.select
  // CHECK-NOT: hivm.hir.vsel
  // CHECK: %[[SUFFIX:.*]] = hivm.hir.vcast ins(%[[WIDE]] : tensor<4x32xf32>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xi16>)
  // CHECK: hivm.hir.vadd ins(%[[SUFFIX]]
  // CHECK: hivm.hir.vadd ins(%[[WIDE]]
  // CHECK: return
  func.func @selected_multiuser(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xi16>, %wide_other: tensor<4x32xf32>) -> (tensor<4x32xi16>, tensor<4x32xf32>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x32xf32>
    %selected = arith.select %full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>
    %suffix_init = tensor.empty() : tensor<4x32xi16>
    %suffix = hivm.hir.vcast ins(%selected : tensor<4x32xf32>)
      outs(%suffix_init : tensor<4x32xi16>) round_mode = <trunc> -> tensor<4x32xi16>
    %sum_init = tensor.empty() : tensor<4x32xi16>
    %sum = hivm.hir.vadd ins(%suffix, %other : tensor<4x32xi16>, tensor<4x32xi16>)
      outs(%sum_init : tensor<4x32xi16>) -> tensor<4x32xi16>
    %second_init = tensor.empty() : tensor<4x32xf32>
    %second = hivm.hir.vadd ins(%selected, %wide_other : tensor<4x32xf32>, tensor<4x32xf32>)
      outs(%second_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    return %sum, %second : tensor<4x32xi16>, tensor<4x32xf32>
  }

  // Sharing the suffix result does not make a masked data prefix observable.
  // CHECK-LABEL: func.func @suffix_multiuser(
  // CHECK: %[[FILL:.*]] = arith.constant 0 : i8
  // CHECK: hivm.hir.vbrc ins(%[[FILL]] : i8) outs({{.*}} : memref<4x32xi8>)
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK-NOT: } else {
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK-NOT: arith.select
  // CHECK-NOT: hivm.hir.vsel
  // CHECK: %[[SUFFIX:.*]] = hivm.hir.vcast ins(%[[WIDE]] : tensor<4x32xf32>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>)
  // CHECK: hivm.hir.vadd ins(%[[SUFFIX]]
  // CHECK: hivm.hir.vadd ins(%[[SUFFIX]]
  // CHECK: return
  func.func @suffix_multiuser(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xf16>) -> (tensor<4x32xf16>, tensor<4x32xf16>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant 0.0 : f32
    %select_init = tensor.empty() : tensor<4x32xf32>
    %selected = hivm.hir.vsel ins(%full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>, f32)
      outs(%select_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %suffix_init = tensor.empty() : tensor<4x32xf16>
    %suffix = hivm.hir.vcast ins(%selected : tensor<4x32xf32>)
      outs(%suffix_init : tensor<4x32xf16>) -> tensor<4x32xf16>
    %sum_init = tensor.empty() : tensor<4x32xf16>
    %sum = hivm.hir.vadd ins(%suffix, %other : tensor<4x32xf16>, tensor<4x32xf16>)
      outs(%sum_init : tensor<4x32xf16>) -> tensor<4x32xf16>
    %second_init = tensor.empty() : tensor<4x32xf16>
    %second = hivm.hir.vadd ins(%suffix, %other : tensor<4x32xf16>, tensor<4x32xf16>)
      outs(%second_init : tensor<4x32xf16>) -> tensor<4x32xf16>
    return %sum, %second : tensor<4x32xf16>, tensor<4x32xf16>
  }

  // AIV mask sinking is independent of the eventual consumer kind.
  // CHECK-LABEL: func.func @cast_select_cast_to_cube(
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: hivm.hir.load
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK-NOT: arith.select
  // CHECK: %[[SUFFIX:.*]] = hivm.hir.vcast ins(%[[WIDE]] : tensor<4x32xf32>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>)
  // CHECK: hivm.hir.mmadL1 ins(%[[SUFFIX]]
  // CHECK: return
  func.func @cast_select_cast_to_cube(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %rhs: tensor<32x32xf16>) -> tensor<4x32xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x32xf32>
    %selected = arith.select %full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xf32>
    %suffix_init = tensor.empty() : tensor<4x32xf16>
    %suffix = hivm.hir.vcast ins(%selected : tensor<4x32xf32>)
      outs(%suffix_init : tensor<4x32xf16>) -> tensor<4x32xf16>
    %true = arith.constant true
    %dot_init = tensor.empty() : tensor<4x32xf32>
    %dot = hivm.hir.mmadL1 ins(%suffix, %rhs, %true, %c4, %c32, %c32
      : tensor<4x32xf16>, tensor<32x32xf16>, i1, index, index, index)
      outs(%dot_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    return %dot : tensor<4x32xf32>
  }

  // A widening-only chain still reaches the terminal Vector consumer and marks
  // the gather.
  // CHECK-LABEL: func.func @cast_chain_to_vector(
  // CHECK: scf.for
  // CHECK: hivm.hir.load
  // CHECK: } {hivm.parallel_loop}
  // CHECK: %[[LOADED:.*]] = bufferization.to_tensor
  // CHECK: %[[HALF:.*]] = hivm.hir.vcast ins(%[[LOADED]] : tensor<4x32xi8>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf16>) round_mode = <round>
  // CHECK: %[[WIDE:.*]] = hivm.hir.vcast ins(%[[HALF]] : tensor<4x32xf16>)
  // CHECK-SAME: outs({{.*}} : tensor<4x32xf32>)
  // CHECK: hivm.hir.vadd ins(%[[WIDE]]
  // CHECK: return
  func.func @cast_chain_to_vector(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %other: tensor<4x32xf32>) -> tensor<4x32xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %sum_init = tensor.empty() : tensor<4x32xf32>
    %sum = hivm.hir.vadd ins(%wide, %other : tensor<4x32xf32>, tensor<4x32xf32>)
      outs(%sum_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    return %sum : tensor<4x32xf32>
  }

  // A shape bridge must not hide the select behind the numeric prefix.
  // CHECK-LABEL: func.func @cast_view_select_to_cube(
  // CHECK: hivm.hir.mmadL1
  // CHECK: return
  func.func @cast_view_select_to_cube(%base: memref<?x32xi8>, %ids: tensor<4xi32>,
      %rhs: tensor<32x32xf16>) -> tensor<4x32xf32>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %stride = arith.constant 64 : i32
    %zero = arith.constant 0 : i32
    %dst = memref.alloc() : memref<4x32xi8>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %safe = arith.maxsi %id, %zero : i32
      %scaled = arith.muli %safe, %stride : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.subview %base[%offset, 0] [1, 32] [1, 1]
        : memref<?x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xi8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xi8, strided<[32, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xi8>
    %half_init = tensor.empty() : tensor<4x32xf16>
    %half = hivm.hir.vcast ins(%loaded : tensor<4x32xi8>)
      outs(%half_init : tensor<4x32xf16>) round_mode = <round> -> tensor<4x32xf16>
    %wide_init = tensor.empty() : tensor<4x32xf32>
    %wide = hivm.hir.vcast ins(%half : tensor<4x32xf16>)
      outs(%wide_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    %expanded = tensor.expand_shape %wide [[0], [1, 2]] output_shape [4, 1, 32] : tensor<4x32xf32> into tensor<4x1x32xf32>
    %view = tensor.collapse_shape %expanded [[0], [1, 2]] : tensor<4x1x32xf32> into tensor<4x32xf32>
    %zeros = arith.constant dense<0> : tensor<4xi32>
    %mask = arith.cmpi sge, %ids, %zeros : tensor<4xi32>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<0.0> : tensor<4x32xf32>
    %selected = arith.select %full_mask, %view, %fallback : tensor<4x32xi1>, tensor<4x32xf32>
    %suffix_init = tensor.empty() : tensor<4x32xf16>
    %suffix = hivm.hir.vcast ins(%selected : tensor<4x32xf32>)
      outs(%suffix_init : tensor<4x32xf16>) -> tensor<4x32xf16>
    %true = arith.constant true
    %dot_init = tensor.empty() : tensor<4x32xf32>
    %dot = hivm.hir.mmadL1 ins(%suffix, %rhs, %true, %c4, %c32, %c32
      : tensor<4x32xf16>, tensor<32x32xf16>, i1, index, index, index)
      outs(%dot_init : tensor<4x32xf32>) -> tensor<4x32xf32>
    return %dot : tensor<4x32xf32>
  }

  // CHECK-LABEL: func.func @unsigned_unrepresentable(
  // CHECK-NOT: scf.if
  // CHECK: } {hivm.parallel_loop}
  // CHECK: arith.select
  // CHECK: return
  func.func @unsigned_unrepresentable(%base: memref<?x32xui8>, %mask: tensor<4xi1>,
      %out: memref<4x32xui16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %dst = memref.alloc() : memref<4x32xui8>
    scf.for %i = %c0 to %c4 step %c1 {
      %off = arith.muli %i, %c32 : index
      %src = memref.subview %base[%off, 0] [1, 32] [1, 1]
        : memref<?x32xui8> to memref<1x32xui8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xui8> to memref<1x32xui8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xui8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xui8, strided<[32, 1], offset: ?>>)
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xui8>
    %init = tensor.empty() : tensor<4x32xui16>
    %wide = hivm.hir.vcast ins(%loaded : tensor<4x32xui8>) outs(%init : tensor<4x32xui16>) -> tensor<4x32xui16>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>) outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<65535> : tensor<4x32xui16>
    %selected = arith.select %full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xui16>
    bufferization.materialize_in_destination %selected in writable %out : (tensor<4x32xui16>, memref<4x32xui16>) -> ()
    return
  }

  // CHECK-LABEL: func.func @unsigned_exact(
  // The exact preimage still cannot be materialized as an arith scalar ui8.
  // CHECK-NOT: scf.if
  // CHECK: } {hivm.parallel_loop}
  // CHECK: arith.select
  // CHECK: return
  func.func @unsigned_exact(%base: memref<?x32xui8>, %mask: tensor<4xi1>,
      %out: memref<4x32xui16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %dst = memref.alloc() : memref<4x32xui8>
    scf.for %i = %c0 to %c4 step %c1 {
      %off = arith.muli %i, %c32 : index
      %src = memref.subview %base[%off, 0] [1, 32] [1, 1]
        : memref<?x32xui8> to memref<1x32xui8, strided<[32, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 32] [1, 1]
        : memref<4x32xui8> to memref<1x32xui8, strided<[32, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x32xui8, strided<[32, 1], offset: ?>>)
        outs(%row : memref<1x32xui8, strided<[32, 1], offset: ?>>)
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x32xui8>
    %init = tensor.empty() : tensor<4x32xui16>
    %wide = hivm.hir.vcast ins(%loaded : tensor<4x32xui8>) outs(%init : tensor<4x32xui16>) -> tensor<4x32xui16>
    %mask_init = tensor.empty() : tensor<4x32xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>) outs(%mask_init : tensor<4x32xi1>) dimensions = [1]
    %fallback = arith.constant dense<255> : tensor<4x32xui16>
    %selected = arith.select %full_mask, %wide, %fallback : tensor<4x32xi1>, tensor<4x32xui16>
    bufferization.materialize_in_destination %selected in writable %out : (tensor<4x32xui16>, memref<4x32xui16>) -> ()
    return
  }
}
