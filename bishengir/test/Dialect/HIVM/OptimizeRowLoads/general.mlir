// RUN: bishengir-opt %s -hivm-optimize-row-loads -canonicalize | FileCheck %s --check-prefixes=CHECK,PREP

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-LABEL: func.func @multiple_index_tensors(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 16], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
  func.func @multiple_index_tensors(%base: memref<?xf16>, %ids: tensor<4xi32>, %pages: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %page = tensor.extract %pages[%i] : tensor<4xi32>
      %combined = arith.addi %id, %page : i32
      %scaled = arith.muli %combined, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

// CHECK-LABEL: func.func @different_index_domain(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 16]
// CHECK: return
  func.func @different_index_domain(%base: memref<?xf16>, %ids: tensor<4xi32>, %pages: tensor<8xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %page = tensor.extract %pages[%i] : tensor<8xi32>
      %combined = arith.addi %id, %page : i32
      %scaled = arith.muli %combined, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

// CHECK-LABEL: func.func @shifted_index_read(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 16]
// CHECK: return
  func.func @shifted_index_read(%base: memref<?xf16>, %ids: tensor<4xi32>, %pages: tensor<5xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %next = arith.addi %i, %c1 : index
      %page = tensor.extract %pages[%next] : tensor<5xi32>
      %combined = arith.addi %id, %page : i32
      %scaled = arith.muli %combined, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

// CHECK-LABEL: func.func @indices_defined_after_alloc(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 16], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
  func.func @indices_defined_after_alloc(%base: memref<?xf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    %zero_ids = arith.constant dense<0> : tensor<4xi32>
    %safe_ids = arith.maxsi %ids, %zero_ids : tensor<4xi32>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %safe_ids[%i] : tensor<4xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

// CHECK-LABEL: func.func @multiple_vector_observations(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 16], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: return
  func.func @multiple_vector_observations(%base: memref<?xf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst : memref<4x16xf16>
    %loaded_again = bufferization.to_tensor %dst : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %loaded_again : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

// Observations after the loop do not prevent unmasked row pairing.
// CHECK-LABEL: func.func @unknown_observer(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 16]
// CHECK: hivm.hir.load
// CHECK: bufferization.to_tensor
// CHECK: hivm.hir.vadd
// CHECK: return {{.*}}, {{.*}} : tensor<4x16xf16>, tensor<4x16xf16>
  func.func @unknown_observer(%base: memref<?xf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> (tensor<4x16xf16>, tensor<4x16xf16>)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst : memref<4x16xf16>
    %loaded_again = bufferization.to_tensor %dst : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %loaded_again : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum, %loaded_again : tensor<4x16xf16>, tensor<4x16xf16>
  }

// CHECK-LABEL: func.func @side_effect_in_address(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 16]
// CHECK: return
  func.func @side_effect_in_address(%base: memref<?xf16>, %ids: tensor<4xi32>, %pages: tensor<4xi32>,
      %side: memref<4xi32>, %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %page = tensor.extract %pages[%i] : tensor<4xi32>
      %combined = arith.addi %id, %page : i32
      memref.store %page, %side[%i] : memref<4xi32>
      %scaled = arith.muli %combined, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }


// CHECK-LABEL: func.func @shared_mask_observation(
// CHECK: scf.for {{.*}} step %c2
// CHECK-NOT: scf.if
// CHECK: sizes: [2, 16], strides: [{{.*}}, 1]
// CHECK: hivm.hir.load
// CHECK: arith.select
// CHECK: hivm.hir.vadd
// CHECK: return
  func.func @shared_mask_observation(%base: memref<?xf16>, %ids: tensor<4xi32>,
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
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst : memref<4x16xf16>
    %raw = bufferization.to_tensor %dst : memref<4x16xf16>
    %mask_init = tensor.empty() : tensor<4x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<4xi1>)
      outs(%mask_init : tensor<4x16xi1>) dimensions = [1]
    %fill_init = tensor.empty() : tensor<4x16xf16>
    %fallback = hivm.hir.vbrc ins(%fill : f16) outs(%fill_init : tensor<4x16xf16>) -> tensor<4x16xf16>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<4x16xi1>, tensor<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %raw : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

// PREP-LABEL: func.func @multiple_mask_inputs(
// PREP: scf.for {{.*}} step %c2
// PREP: arith.cmpi sge
// PREP: arith.cmpi sge
// PREP: scf.if
// PREP: sizes: [2, 16], strides: [{{.*}}, 1]
// PREP: hivm.hir.load
// PREP-NOT: arith.select
// PREP: hivm.hir.vadd
// PREP: return
  func.func @multiple_mask_inputs(%base: memref<?xf16>, %ids: tensor<4xi32>, %pages: tensor<4xi32>,
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
      %page = tensor.extract %pages[%i] : tensor<4xi32>
      %combined = arith.addi %id, %page : i32
      %safe = arith.maxsi %combined, %zero : i32
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
    %combined_ids = arith.addi %ids, %pages : tensor<4xi32>
    %mask = arith.cmpi sge, %combined_ids, %zeros : tensor<4xi32>
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

// The exact -0 fallback can cross the widening cast; retain its saturation
// attribute and both numerical casts while removing only the select.
// CHECK-LABEL: func.func @saturating_cast_stays(
// CHECK: scf.for {{.*}} step %c2
// CHECK: scf.if
// CHECK: hivm.hir.vcast
// CHECK-SAME: enable_saturate = true
// CHECK-NOT: hivm.hir.vsel
// CHECK: hivm.hir.vcast
// CHECK: return
  func.func @saturating_cast_stays(%base: memref<?xbf16>, %ids: tensor<4xi32>,
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
    %wide = hivm.hir.vcast {enable_saturate = true} ins(%loaded : tensor<4x16xbf16>)
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

// A space-less buffer produced inside the function is local memory, not GM,
// even though genuine GM arguments also omit the memory space. A row copy from
// it must stay a plain load.
// CHECK-LABEL: func.func @space_less_local_source(
// CHECK: scf.for {{.*}} step %c1
// CHECK-NOT: sizes: [2, 16]
// CHECK: return
  func.func @space_less_local_source(%ids: tensor<4xi32>, %fill: f16,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    %local = memref.alloc() : memref<128x256xf16>
    hivm.hir.vbrc ins(%fill : f16) outs(%local : memref<128x256xf16>)
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %local to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<128x256xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }

// A structurally valid gather is paired without a preparatory pass.
// CHECK-LABEL: func.func @unmarked_loop_pairs(
// CHECK: scf.for {{.*}} step %c2
// CHECK: sizes: [2, 16]
// CHECK: return
  func.func @unmarked_loop_pairs(%base: memref<?xf16>, %ids: tensor<4xi32>,
      %other: tensor<4x16xf16>) -> tensor<4x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : i32
    %dst = memref.alloc() : memref<4x16xf16>
    scf.for %i = %c0 to %c4 step %c1 {
      %id = tensor.extract %ids[%i] : tensor<4xi32>
      %scaled = arith.muli %id, %c32 : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %row = memref.subview %dst[%i, 0] [1, 16] [1, 1]
        : memref<4x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%loaded, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }
}
