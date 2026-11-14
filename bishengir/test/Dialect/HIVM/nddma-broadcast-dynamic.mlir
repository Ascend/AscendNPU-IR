// RUN: bishengir-opt %s -split-input-file -verify-diagnostics | FileCheck %s
// CHECK-LABEL: func.func @bounded2axis0(
// CHECK: broadcast_axes = [0]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @bounded2axis0(%src: memref<1x32xf32, #hivm.address_space<gm>>, %k: index) {
  %limit = arith.constant 4 : index
  %n = arith.minui %k, %limit : index
  %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  %view = memref.subview %ub[0, 0][%n, 32][1, 1] : memref<4x32xf32, #hivm.address_space<ub>> to memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%view : memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}
}
// -----
// CHECK-LABEL: func.func @bounded2axis1(
// CHECK: broadcast_axes = [1]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @bounded2axis1(%src: memref<4x1xf32, #hivm.address_space<gm>>, %k: index) {
  %limit = arith.constant 32 : index
  %n = arith.minui %k, %limit : index
  %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
  %view = memref.subview %ub[0, 0][4, %n][1, 1] : memref<4x32xf32, #hivm.address_space<ub>> to memref<4x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<4x1xf32, #hivm.address_space<gm>>) outs(%view : memref<4x?xf32, strided<[32, 1]>, #hivm.address_space<ub>>) broadcast_axes = [1]
  return
}
}
// -----
// CHECK-LABEL: func.func @bounded3axis0(
// CHECK: broadcast_axes = [0]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @bounded3axis0(%src: memref<1x3x32xf32, #hivm.address_space<gm>>, %k: index) {
  %limit = arith.constant 4 : index
  %n = arith.minui %k, %limit : index
  %ub = memref.alloc() : memref<4x3x32xf32, #hivm.address_space<ub>>
  %view = memref.subview %ub[0, 0, 0][%n, 3, 32][1, 1, 1] : memref<4x3x32xf32, #hivm.address_space<ub>> to memref<?x3x32xf32, strided<[96, 32, 1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<1x3x32xf32, #hivm.address_space<gm>>) outs(%view : memref<?x3x32xf32, strided<[96, 32, 1]>, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}
}
// -----
// CHECK-LABEL: func.func @bounded3axis2(
// CHECK: broadcast_axes = [2]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @bounded3axis2(%src: memref<4x3x1xf32, #hivm.address_space<gm>>, %k: index) {
  %limit = arith.constant 32 : index
  %n = arith.minui %k, %limit : index
  %ub = memref.alloc() : memref<4x3x32xf32, #hivm.address_space<ub>>
  %view = memref.subview %ub[0, 0, 0][4, 3, %n][1, 1, 1] : memref<4x3x32xf32, #hivm.address_space<ub>> to memref<4x3x?xf32, strided<[96, 32, 1]>, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<4x3x1xf32, #hivm.address_space<gm>>) outs(%view : memref<4x3x?xf32, strided<[96, 32, 1]>, #hivm.address_space<ub>>) broadcast_axes = [2]
  return
}
}
// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @unbounded(%src: memref<1x32xf32, #hivm.address_space<gm>>, %dst: memref<?x32xf32, #hivm.address_space<ub>>) {
  // expected-error @+1 {{dynamic broadcast requires a bounded static-backed view; allocation length is unavailable in the legacy ABI}}
  hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<?x32xf32, #hivm.address_space<ub>>) broadcast_axes = [0]
  return
}
}

// -----
// CHECK-LABEL: func.func @dynamic_kept_and_broadcast(
// CHECK: broadcast_axes = [0]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @dynamic_kept_and_broadcast(%gm: memref<1x3x32xf32, #hivm.address_space<gm>>, %k: index, %w: index) {
 %c4 = arith.constant 4 : index
 %c32 = arith.constant 32 : index
 %n = arith.minui %k, %c4 : index
 %width = arith.minui %w, %c32 : index
 %src = memref.subview %gm[0,0,0][1,3,%width][1,1,1] : memref<1x3x32xf32, #hivm.address_space<gm>> to memref<1x3x?xf32, strided<[96,32,1]>, #hivm.address_space<gm>>
 %ub = memref.alloc() : memref<4x3x32xf32, #hivm.address_space<ub>>
 %dst = memref.subview %ub[0,0,0][%n,3,%width][1,1,1] : memref<4x3x32xf32, #hivm.address_space<ub>> to memref<?x3x?xf32, strided<[96,32,1]>, #hivm.address_space<ub>>
 hivm.hir.load ins(%src : memref<1x3x?xf32, strided<[96,32,1]>, #hivm.address_space<gm>>) outs(%dst : memref<?x3x?xf32, strided<[96,32,1]>, #hivm.address_space<ub>>) broadcast_axes = [0]
 return
}
}
// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @oversized_dynamic_bound(%src: memref<1x32xf32, #hivm.address_space<gm>>, %k: index) {
 %limit = arith.constant 5 : index
 %n = arith.minui %k, %limit : index
 %ub = memref.alloc() : memref<4x32xf32, #hivm.address_space<ub>>
 %dst = memref.subview %ub[0,0][%n,32][1,1] : memref<4x32xf32, #hivm.address_space<ub>> to memref<?x32xf32, strided<[32,1]>, #hivm.address_space<ub>>
 // expected-error @+1 {{dynamic broadcast requires a bounded static-backed view; allocation length is unavailable in the legacy ABI}}
 hivm.hir.load ins(%src : memref<1x32xf32, #hivm.address_space<gm>>) outs(%dst : memref<?x32xf32, strided<[32,1]>, #hivm.address_space<ub>>) broadcast_axes = [0]
 return
}
}

// -----
// CHECK-LABEL: func.func @suffix_bound(
// CHECK: broadcast_axes = [1]
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @suffix_bound(%gm: memref<32xf32, #hivm.address_space<gm>>, %raw: index, %off: index) {
 %c32 = arith.constant 32 : index
 %n = arith.minui %raw, %c32 : index
 %suffix = arith.subi %c32, %n : index
 %sv = memref.subview %gm[0][%n][1] : memref<32xf32, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<gm>>
 %src = memref.expand_shape %sv [[0,1]] output_shape [%n,1] : memref<?xf32, strided<[1]>, #hivm.address_space<gm>> into memref<?x1xf32, #hivm.address_space<gm>>
 %root = memref.alloc() : memref<32x4xf32, #hivm.address_space<ub>>
 %dst = memref.subview %root[%suffix,0][%n,4][1,1] : memref<32x4xf32, #hivm.address_space<ub>> to memref<?x4xf32, strided<[4,1], offset: ?>, #hivm.address_space<ub>>
 
 hivm.hir.load ins(%src : memref<?x1xf32, #hivm.address_space<gm>>) outs(%dst : memref<?x4xf32, strided<[4,1], offset: ?>, #hivm.address_space<ub>>) broadcast_axes = [1]
 return
}
}

// -----
module attributes {hacc.target = #hacc.target<"Ascend960">} {
func.func @unknown_offset(%gm: memref<32xf32, #hivm.address_space<gm>>, %raw: index, %off: index) {
 %c32 = arith.constant 32 : index
 %n = arith.minui %raw, %c32 : index
 %suffix = arith.subi %c32, %n : index
 %sv = memref.subview %gm[0][%n][1] : memref<32xf32, #hivm.address_space<gm>> to memref<?xf32, strided<[1]>, #hivm.address_space<gm>>
 %src = memref.expand_shape %sv [[0,1]] output_shape [%n,1] : memref<?xf32, strided<[1]>, #hivm.address_space<gm>> into memref<?x1xf32, #hivm.address_space<gm>>
 %root = memref.alloc() : memref<32x4xf32, #hivm.address_space<ub>>
 %dst = memref.subview %root[%off,0][%n,4][1,1] : memref<32x4xf32, #hivm.address_space<ub>> to memref<?x4xf32, strided<[4,1], offset: ?>, #hivm.address_space<ub>>
 // expected-error @+1 {{dynamic broadcast requires a bounded static-backed view}}
 hivm.hir.load ins(%src : memref<?x1xf32, #hivm.address_space<gm>>) outs(%dst : memref<?x4xf32, strided<[4,1], offset: ?>, #hivm.address_space<ub>>) broadcast_axes = [1]
 return
}
}
