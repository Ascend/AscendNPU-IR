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

// -----

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  // CHECK: affine_map<()[s0, s1] -> (0, s0 - s1)>
  // CHECK: affine_map<()[s0] -> (16, s0)>
  func.func @distribute_affine_max_over_min(%arg0 : index, %arg1 : index) -> (memref<?xf32>) {
    %min = affine.min affine_map<()[s0, s1] -> (16, s0 - s1)>()[%arg0, %arg1]
    %size = affine.max affine_map<()[s0] -> (0, s0)>()[%min]
    // CHECK: affine.max #map{{.*}}()[%arg0, %arg1]
    // CHECK: affine.min #map{{.*}}()[%{{.*}}]
    %alloc = memref.alloc(%size) : memref<?xf32>
    return %alloc : memref<?xf32>
  }
}

// -----

// affine.min with a constant branch that is the minimum: UB = 16.
// "blockade" (unregistered op) breaks ValueBoundsConstraintSet, forcing
// fallback to resolveDynamicDimToBoundImpl, which recurses through
// arith.index_cast (line 351) into the affine.min branch (248-275).
// CHECK-LABEL: func.func @affine_min_constant_branch
func.func @affine_min_constant_branch(%arg0 : index) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[ALLOC]]
  // CHECK: "some_use"(%[[VIEW]])
  %size = affine.min affine_map<(d0) -> (16, d0)>(%arg0)
  %ic = arith.index_cast %size : index to i64
  %size2 = arith.index_cast %ic : i64 to index
  %alloc = memref.alloc(%size2) : memref<?x16xi32>
  annotation.mark %alloc {buffer_size_in_byte = 16384 : i64} : memref<?x16xi32>
  "some_use"(%alloc) : (memref<?x16xi32>) -> ()
}

// -----

// affine.max with every non-constant branch bounded by affine.min:
// max(min(16,d0), min(32,d1)) -> UB = max(16, 32) = 32.
// arith.index_cast breaks ValueBoundsConstraintSet, forcing fallback to
// resolveDynamicDimToBoundImpl -> affine.max branch (279-307).
// CHECK-LABEL: func.func @affine_max_all_branches_bounded
func.func @affine_max_all_branches_bounded(%arg0 : index, %arg1 : index) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2048xi8>
  // CHECK: %[[VIEW:.*]] = memref.view %[[ALLOC]]
  // CHECK: "some_use"(%[[VIEW]])
  %m0 = affine.min affine_map<(d0) -> (16, d0)>(%arg0)
  %m1 = affine.min affine_map<(d0) -> (32, d0)>(%arg1)
  %size = affine.max affine_map<(d0, d1) -> (d0, d1)>(%m0, %m1)
  %ic = arith.index_cast %size : index to i64
  %size2 = arith.index_cast %ic : i64 to index
  %alloc = memref.alloc(%size2) : memref<?x16xi32>
  annotation.mark %alloc {buffer_size_in_byte = 16384 : i64} : memref<?x16xi32>
  "some_use"(%alloc) : (memref<?x16xi32>) -> ()
}
