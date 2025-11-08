// RUN: bishengir-opt --hivm-constantize-buffer-size -split-input-file -allow-unregistered-dialect %s | FileCheck %s

#map = affine_map<(d0) -> (d0, 32)>
func.func @test0(%arg0 : index) {
  %size = affine.min #map(%arg0)
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2048xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[ALLOC]]
  // CHECK: "some_use"(%[[VIEW]])
  %alloc = memref.alloc(%size) : memref<?x16xi32>
  annotation.mark %alloc {buffer_size_in_byte = 16384 : i64} : memref<?x16xi32>
  "some_use"(%alloc) : (memref<?x16xi32>) -> ()
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<2048xi8>
  // CHECK: %[[VIEW1:.*]] = memref.view %[[ALLOCA]]
  // CHECK: "some_use"(%[[VIEW1]])
  %alloca = memref.alloca(%size) : memref<?x16xi32>
  annotation.mark %alloca {buffer_size_in_byte = 16384 : i64} : memref<?x16xi32>
  "some_use"(%alloca) : (memref<?x16xi32>) -> ()
}

// -----

#map = affine_map<(d0) -> (d0, 32)>
#map1 = affine_map<(d0) -> (d0, 64)>
func.func @test0(%arg0 : index, %arg1 : index) {
  %size = affine.min #map(%arg0)
  %size1 = affine.min #map1(%arg1)
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2048xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[ALLOC]]
  // CHECK: "some_use"(%[[VIEW]])
  %alloc = memref.alloc(%size) : memref<16x?xi32>
  annotation.mark %alloc {buffer_size_in_byte = 16384 : i64} : memref<16x?xi32>
  "some_use"(%alloc) : (memref<16x?xi32>) -> ()
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<19922944xi8>
  // CHECK: %[[VIEW1:.*]] = memref.view %[[ALLOCA]]
  // CHECK: "some_use"(%[[VIEW1]])
  %alloca = memref.alloca(%size, %size1) : memref<16x?x38x4x?xi32>
  annotation.mark %alloca {buffer_size_in_byte = 159383552 : i64} : memref<16x?x38x4x?xi32>
  "some_use"(%alloca) : (memref<16x?x38x4x?xi32>) -> ()
}

// -----

// Cannot compute upper bound, no effect.
// CHECK-NOT: memref.view
#map = affine_map<(d0) -> (d0, 32)>
func.func @counter_test0(%arg0 : index) {
  %size = affine.max #map(%arg0)
  %alloc = memref.alloc(%size) : memref<?x16xi32>
  "some_use"(%alloc) : (memref<?x16xi32>) -> ()
}

// -----

// Static shape, no effect.
// CHECK-NOT: memref.view
func.func @counter_test1() {
  %alloc = memref.alloc() : memref<32x16xi32>
  "some_use"(%alloc) : (memref<32x16xi32>) -> ()
}

// -----

// Partially constantized dynamic shape, no effect.
// CHECK-NOT: memref.view
#map = affine_map<()[s0] -> (-s0 + 11264)>
#map1 = affine_map<()[s0, s1] -> (s0 * -19 - s1 * 19 + ((s0 + s1) floordiv 11) * 209 + 196, 19)>
#map2 = affine_map<()[s0, s1] -> (((s0 + s1) floordiv 11) * -16 + (((s0 + s1) floordiv 11) floordiv 8) * 128 + 116, 16)>
module {
  func.func @partially_constantized() {
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c0 = arith.constant 0 : index
    %c48 = arith.constant 48 : index
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.index_cast %0 : i64 to index
    %2 = affine.apply #map()[%1]
    scf.for %arg2 = %c0 to %2 step %c48 {
      %3 = affine.min #map1()[%1, %arg2]
      %4 = affine.min #map2()[%1, %arg2]
      %5 = arith.addi %3, %c7 : index
      %6 = arith.remsi %5, %c8 : index
      %7 = arith.subi %5, %6 : index
      %alloc = memref.alloc(%4, %7) : memref<1x2x?x?x1xf32, #hivm.address_space<ub>>
      %subview = memref.subview %alloc[0, 0, 0, 0, 0] [1, 2, %4, %3, 1] [1, 1, 1, 1, 1] : memref<1x2x?x?x1xf32, #hivm.address_space<ub>> to memref<1x2x?x?xf32, strided<[?, ?, ?, 1]>, #hivm.address_space<ub>>
      "some_use"(%subview) : (memref<1x2x?x?xf32, strided<[?, ?, ?, 1]>, #hivm.address_space<ub>>) -> ()
    } {__tiled_for___5}
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0, 32)>
func.func @no_annotation(%arg0 : index) -> (memref<?x16xi32>) {
  %size = affine.min #map(%arg0)
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2048xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[ALLOC]]
  %alloc = memref.alloc(%size) : memref<?x16xi32>
  return %alloc : memref<?x16xi32>
}

// -----

#map = affine_map<(d0) -> (d0, 32)>
func.func @alloc_excceds_marked_size(%arg0 : index) -> (memref<?x16xi32>) {
  %size = affine.min #map(%arg0)
  // CHECK: memref.alloc({{.*}}) : memref<?x16xi32>
  %alloc = memref.alloc(%size) : memref<?x16xi32>
  annotation.mark %alloc {buffer_size_in_byte = 100 : i64} : memref<?x16xi32>
  return %alloc : memref<?x16xi32>
}

