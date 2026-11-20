// RUN: bishengir-opt %s -hivm-optimize-row-loads -split-input-file | FileCheck %s --check-prefixes=CHECK,PAIR

// A reversed read is valid address data, but is not the mask for this row.
// The post-loop predicate does not dominate the loop, so byte fallback cannot
// move it either. The select must stay intact.
// CHECK-LABEL: func.func @non_direct_read_not_reused(
// CHECK: scf.for
// CHECK: arith.subi
// CHECK: tensor.extract
// CHECK-NOT: scf.if
// CHECK: hivm.hir.load
// CHECK: arith.select
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @non_direct_read_not_reused(%base: memref<?x16xf16>, %ids: tensor<4xi32>,
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
      %last = arith.constant 3 : index
      %reverse = arith.subi %last, %i : index
      %id = tensor.extract %ids[%reverse] : tensor<4xi32>
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
    %fill_init = tensor.empty() : tensor<4x16xf16>
    %fallback = hivm.hir.vbrc ins(%fill : f16) outs(%fill_init : tensor<4x16xf16>) -> tensor<4x16xf16>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<4x16xi1>, tensor<4x16xf16>
    %empty = tensor.empty() : tensor<4x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%empty : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %sum : tensor<4x16xf16>
  }
}

// -----

// Address replay uses global iterations 2 and 3; the local mask uses rows 0 and 1.
// PAIR-LABEL: func.func @post_tiling_local_mask(
// PAIR: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<2x16xf16>)
// PAIR: tensor.extract %arg1[%[[GLOBAL:.*]]] : tensor<4xi32>
// PAIR: %[[LOCAL:.*]] = arith.subi %[[GLOBAL]], {{.*}} : index
// PAIR: tensor.extract {{.*}}[%[[LOCAL]]] : tensor<2xi8>
// PAIR: tensor.extract %arg1[%[[NEXT:.*]]] : tensor<4xi32>
// PAIR: %[[LOCAL_NEXT:.*]] = arith.subi %[[NEXT]], {{.*}} : index
// PAIR: tensor.extract {{.*}}[%[[LOCAL_NEXT]]] : tensor<2xi8>
// PAIR: arith.andi
// PAIR: scf.if
// PAIR: sizes: [2, 16], strides:
// PAIR: hivm.hir.load
// PAIR: } else {
// PAIR: scf.if
// PAIR: hivm.hir.load
// PAIR-NOT: } else {
// PAIR-NOT: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<
// PAIR: scf.if
// PAIR: hivm.hir.load
// PAIR-NOT: } else {
// PAIR-NOT: hivm.hir.vbrc {{.*}} outs({{.*}} : memref<
// PAIR-NOT: arith.select
// PAIR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @post_tiling_local_mask(%base: memref<?xf16>, %ids: tensor<4xi32>,
      %mask: tensor<2xi1>, %other: tensor<2x16xf16>) -> tensor<2x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %pitch = arith.constant 32 : i32
    %fill = arith.constant -0.0 : f16
    %dst = memref.alloc() : memref<2x16xf16>
    scf.for %iv = %c2 to %c4 step %c1 {
      %id = tensor.extract %ids[%iv] : tensor<4xi32>
      %scaled = arith.muli %id, %pitch : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %local = arith.subi %iv, %c2 : index
      %row = memref.subview %dst[%local, 0] [1, 16] [1, 1]
        : memref<2x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst restrict writable : memref<2x16xf16>
    %mask_init = tensor.empty() : tensor<2x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<2xi1>)
      outs(%mask_init : tensor<2x16xi1>) dimensions = [1]
    %fill_init = tensor.empty() : tensor<2x16xf16>
    %fallback = hivm.hir.vbrc ins(%fill : f16) outs(%fill_init : tensor<2x16xf16>) -> tensor<2x16xf16>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<2x16xi1>, tensor<2x16xf16>
    %empty = tensor.empty() : tensor<2x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %other : tensor<2x16xf16>, tensor<2x16xf16>)
      outs(%empty : tensor<2x16xf16>) -> tensor<2x16xf16>
    return %sum : tensor<2x16xf16>
  }
}

// -----

// A separate raw observation forbids mask sinking but permits row pairing.
// PAIR-LABEL: func.func @post_tiling_raw_observation(
// PAIR: tensor.extract %arg1[{{.*}}] : tensor<4xi32>
// PAIR-NOT: scf.if
// PAIR: sizes: [2, 16], strides:
// PAIR: hivm.hir.load
// PAIR: arith.select
// PAIR: hivm.hir.vadd
// PAIR: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @post_tiling_raw_observation(%base: memref<?xf16>, %ids: tensor<4xi32>,
      %mask: tensor<2xi1>, %other: tensor<2x16xf16>) -> tensor<2x16xf16>
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %pitch = arith.constant 32 : i32
    %fill = arith.constant -0.0 : f16
    %dst = memref.alloc() : memref<2x16xf16>
    scf.for %iv = %c2 to %c4 step %c1 {
      %id = tensor.extract %ids[%iv] : tensor<4xi32>
      %scaled = arith.muli %id, %pitch : i32
      %offset = arith.index_cast %scaled : i32 to index
      %src = memref.reinterpret_cast %base to offset: [%offset], sizes: [1, 16], strides: [16, 1]
        : memref<?xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      %local = arith.subi %iv, %c2 : index
      %row = memref.subview %dst[%local, 0] [1, 16] [1, 1]
        : memref<2x16xf16> to memref<1x16xf16, strided<[16, 1], offset: ?>>
      hivm.hir.load ins(%src : memref<1x16xf16, strided<[16, 1], offset: ?>>)
        outs(%row : memref<1x16xf16, strided<[16, 1], offset: ?>>) core_type = <VECTOR>
    } {hivm.parallel_loop}
    %loaded = bufferization.to_tensor %dst : memref<2x16xf16>
    %raw = bufferization.to_tensor %dst : memref<2x16xf16>
    %mask_init = tensor.empty() : tensor<2x16xi1>
    %full_mask = linalg.broadcast ins(%mask : tensor<2xi1>)
      outs(%mask_init : tensor<2x16xi1>) dimensions = [1]
    %fill_init = tensor.empty() : tensor<2x16xf16>
    %fallback = hivm.hir.vbrc ins(%fill : f16) outs(%fill_init : tensor<2x16xf16>) -> tensor<2x16xf16>
    %selected = arith.select %full_mask, %loaded, %fallback : tensor<2x16xi1>, tensor<2x16xf16>
    %empty = tensor.empty() : tensor<2x16xf16>
    %sum = hivm.hir.vadd ins(%selected, %raw : tensor<2x16xf16>, tensor<2x16xf16>)
      outs(%empty : tensor<2x16xf16>) -> tensor<2x16xf16>
    return %sum : tensor<2x16xf16>
  }
}
