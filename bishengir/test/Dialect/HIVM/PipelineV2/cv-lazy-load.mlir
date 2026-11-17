// RUN: bishengir-opt %s -cv-lazy-load -split-input-file | FileCheck %s

// Distinct GM arguments are independent; a store to %out does not block
// reloading %src. Preserve the loop-dependent address and zero padding.
// CHECK-LABEL: func.func @late_load
// CHECK: scope.scope : () -> ()
// CHECK: hivm.hir.load
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.copy
// CHECK: scope.return
// CHECK: hivm.preload_num = 2
// CHECK: hivm.preload_num = 1
// CHECK: scope.scope : () -> ()
// CHECK: %[[ADDR:.*]] = memref.reinterpret_cast
// CHECK: %[[BUF:.*]] = memref.alloc
// CHECK: hivm.hir.load ins(%[[ADDR]] {{.*}} outs(%[[BUF]] {{.*}} init_out_buffer = true init_condition = %{{.*}}
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[BUF]]
// CHECK: %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[T]]
// CHECK: hivm.hir.vmul ins(%[[TRANSPOSE]], %[[TRANSPOSE]]
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @late_load(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @same_stage
// CHECK: %[[SNAPSHOT:.*]] = scope.scope : () -> tensor<4x2xf32>
// CHECK: hivm.hir.load
// CHECK: scope.return %{{.*}} : tensor<4x2xf32>
// CHECK-NOT: hivm.hir.load
// CHECK: hivm.hir.vmul ins(%[[SNAPSHOT]], %[[SNAPSHOT]]
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @same_stage(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @adjacent_stage
// CHECK: scope.scope : () -> ()
// CHECK: hivm.hir.load
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.copy
// CHECK: scope.return
// CHECK: hivm.preload_num = 2
// CHECK: hivm.preload_num = 1
// CHECK: scope.scope : () -> ()
// CHECK: %[[ADDR:.*]] = memref.reinterpret_cast
// CHECK: %[[BUF:.*]] = memref.alloc
// CHECK: hivm.hir.load ins(%[[ADDR]] {{.*}} outs(%[[BUF]] {{.*}} init_out_buffer = true init_condition = %{{.*}}
// CHECK: %[[T:.*]] = bufferization.to_tensor %[[BUF]]
// CHECK: %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[T]]
// CHECK: hivm.hir.vmul ins(%[[TRANSPOSE]], %[[TRANSPOSE]]
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @adjacent_stage(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 1 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @written_input
// CHECK: %[[SNAPSHOT:.*]] = scope.scope : () -> tensor<4x2xf32>
// CHECK: hivm.hir.load
// CHECK: scope.return %{{.*}} : tensor<4x2xf32>
// CHECK-NOT: hivm.hir.load
// CHECK: hivm.hir.vmul ins(%[[SNAPSHOT]], %[[SNAPSHOT]]
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @written_input(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      %ptr = memref.reinterpret_cast %src to offset: [0], sizes: [4, 2], strides: [2, 1] : memref<?xf32> to memref<4x2xf32>
      hivm.hir.store ins(%v : tensor<4x2xf32>) outs(%ptr : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @mutated_local_buffer
// CHECK: %[[SNAPSHOT:.*]] = scope.scope : () -> tensor<4x2xf32>
// CHECK: hivm.hir.load
// CHECK: scope.return %{{.*}} : tensor<4x2xf32>
// CHECK-NOT: hivm.hir.load
// CHECK: hivm.hir.vmul ins(%[[SNAPSHOT]], %[[SNAPSHOT]]
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @mutated_local_buffer(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      memref.store %zero, %buf[%c0, %c0] : memref<2x4xf32>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// CHECK-LABEL: func.func @missing_stage
// CHECK: %[[SNAPSHOT:.*]] = scope.scope : () -> tensor<4x2xf32>
// CHECK: hivm.hir.load
// CHECK: scope.return %{{.*}} : tensor<4x2xf32>
// CHECK-NOT: hivm.hir.load
// CHECK: hivm.hir.vmul ins(%[[SNAPSHOT]], %[[SNAPSHOT]]
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @missing_stage(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// Retain the producer result for same-stage users; only the later consumer
// gets a private reload, even when it uses the tensor more than once.
// CHECK-LABEL: func.func @mixed_stage_consumers
// CHECK: %[[SNAPSHOT:.*]] = scope.scope : () -> tensor<4x2xf32>
// CHECK: hivm.hir.load
// CHECK: scope.return %{{.*}} : tensor<4x2xf32>
// CHECK-NOT: hivm.hir.load
// CHECK: hivm.hir.vmul ins(%[[SNAPSHOT]], %[[SNAPSHOT]]
// CHECK: hivm.preload_num = 2
// CHECK: hivm.preload_num = 1
// CHECK: hivm.hir.load
// CHECK: %[[RELOADED:.*]] = hivm.hir.vtranspose
// CHECK: hivm.hir.vmul ins(%[[RELOADED]], %[[RELOADED]]
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @mixed_stage_consumers(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// A cast following the load is rematerialized as part of the load chain.
// CHECK-LABEL: func.func @load_cast
// CHECK: scope.scope : () -> ()
// CHECK: hivm.hir.load
// CHECK: hivm.hir.vcast
// CHECK: scope.return
// CHECK: hivm.preload_num = 2
// CHECK: hivm.preload_num = 1
// CHECK: scope.scope : () -> ()
// CHECK: hivm.hir.load
// CHECK: %[[WIDE:.*]] = hivm.hir.vcast
// CHECK: %[[TRANSPOSE:.*]] = hivm.hir.vtranspose ins(%[[WIDE]]
// CHECK: hivm.hir.vmul ins(%[[TRANSPOSE]], %[[TRANSPOSE]]
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @load_cast(%src: memref<?xbf16>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : bf16
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xbf16> to memref<2x4xbf16, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xbf16>
      hivm.hir.load ins(%ptr : memref<2x4xbf16, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xbf16>) pad_mode = <PadValue> pad_value = %zero : bf16 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xbf16>
      %wide = tensor.empty() : tensor<2x4xf32>
      %widened = hivm.hir.vcast ins(%t : tensor<2x4xbf16>) outs(%wide : tensor<2x4xf32>) -> tensor<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%widened : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}

// -----

// Reload after unrelated work within the consuming stage to shorten the local
// buffer lifetime as well as eliminating the cross-stage tensor lifetime.
// CHECK-LABEL: func.func @first_use_in_stage
// CHECK: scope.scope : () -> ()
// CHECK: hivm.hir.load
// CHECK: hivm.preload_num = 2
// CHECK: hivm.preload_num = 1
// CHECK: scope.scope : () -> ()
// CHECK-NOT: hivm.hir.load
// CHECK: hivm.hir.vbrc
// CHECK: hivm.hir.store
// CHECK: hivm.hir.load
// CHECK: hivm.hir.vtranspose
// CHECK: hivm.hir.vmul
module attributes {hacc.target = #hacc.target<"Ascend950">} {
func.func @first_use_in_stage(%src: memref<?xf32>, %out: memref<4x2xf32>, %pad: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = scope.scope : () -> tensor<4x2xf32> {
      %offset = arith.muli %i, %c4 : index
      %ptr = memref.reinterpret_cast %src to offset: [%offset], sizes: [2, 4], strides: [4, 1] : memref<?xf32> to memref<2x4xf32, strided<[4, 1], offset: ?>>
      %buf = memref.alloc() : memref<2x4xf32>
      hivm.hir.load ins(%ptr : memref<2x4xf32, strided<[4, 1], offset: ?>>) outs(%buf : memref<2x4xf32>) pad_mode = <PadValue> pad_value = %zero : f32 init_out_buffer = true init_condition = %pad : i1 core_type = <VECTOR>
      %t = bufferization.to_tensor %buf restrict writable : memref<2x4xf32>
      %empty = tensor.empty() : tensor<4x2xf32>
      %trans = hivm.hir.vtranspose ins(%t : tensor<2x4xf32>) outs(%empty : tensor<4x2xf32>) permutation = [1, 0] -> tensor<4x2xf32>
      %bf = tensor.empty() : tensor<4x2xbf16>
      %cast = hivm.hir.vcast ins(%trans : tensor<4x2xf32>) outs(%bf : tensor<4x2xbf16>) -> tensor<4x2xbf16>
      %l1 = memref.alloc() : memref<4x2xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%cast : tensor<4x2xbf16>) outs(%l1 : memref<4x2xbf16, #hivm.address_space<cbuf>>)
      scope.return %trans : tensor<4x2xf32>
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32}
    scope.scope : () -> () {
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.preload_num = 1 : i32}
    scope.scope : () -> () {
      %e = tensor.empty() : tensor<4x2xf32>
      %work = hivm.hir.vbrc ins(%zero : f32) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%work : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      %r = hivm.hir.vmul ins(%v, %v : tensor<4x2xf32>, tensor<4x2xf32>) outs(%e : tensor<4x2xf32>) -> tensor<4x2xf32>
      hivm.hir.store ins(%r : tensor<4x2xf32>) outs(%out : memref<4x2xf32>)
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32}
  } {hivm.cv_pipelined_loop}
  return
}
}
