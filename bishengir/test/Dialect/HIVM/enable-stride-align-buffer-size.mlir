// RUN: bishengir-opt -split-input-file -hivm-enable-stride-align %s | FileCheck %s

// Enabling stride align re-allocates a buffer with padded extents, so a
// `buffer_size_in_byte` annotation computed on the unpadded shape no longer
// bounds the allocation. The annotation has to grow by the ratio of the padded
// to the unpadded per-dimension upper bounds.

// -----

// The padded extent is dynamic and its bound is recoverable through the
// index_cast/minsi/maxsi clamp chain: ub = 4, AlignUp(4, 8) = 8, so 512 -> 1024.
// This is the shape a Cube-to-UB fixpipe result buffer has in a mix kernel.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @padded_dynamic_extent
  func.func @padded_dynamic_extent(%arg0: i32, %arg1: i32,
      %src: memref<?x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %i0 = arith.index_cast %arg0 : i32 to index
    %m0 = arith.maxsi %i0, %c0 : index
    %d0 = arith.minsi %m0, %c32 : index
    %i1 = arith.index_cast %arg1 : i32 to index
    %n1 = arith.minsi %i1, %c4 : index
    %d1 = arith.maxsi %n1, %c0 : index

    %alloc = memref.alloc(%d0, %d1) : memref<?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 512 : i64}
      : memref<?x?xf32, #hivm.address_space<ub>>

    // CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?x1xf32
    // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 1024 : i64}
    // CHECK-NOT: buffer_size_in_byte = 512
    hivm.hir.copy ins(%src : memref<?x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<?x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Only the padded extent matters: the other extent being dynamic does not
// change the factor, so an outer static extent behaves the same way.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @padded_dynamic_extent_static_outer
  func.func @padded_dynamic_extent_static_outer(%arg0: i32,
      %src: memref<32x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %i1 = arith.index_cast %arg0 : i32 to index
    %n1 = arith.minsi %i1, %c4 : index
    %d1 = arith.maxsi %n1, %c0 : index

    %alloc = memref.alloc(%d1) : memref<32x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<32x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 512 : i64}
      : memref<32x?xf32, #hivm.address_space<ub>>

    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 1024 : i64}
    hivm.hir.copy ins(%src : memref<32x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<32x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Two padded extents, one static and one dynamic: both factors have to be
// applied. Scaling by the static volume alone would stop at 1024, which looks
// like the annotation was updated while it is still half of what is needed.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @padded_static_and_dynamic_extents
  func.func @padded_static_and_dynamic_extents(%arg0: i32, %d0: index,
      %src: memref<?x4x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c4 = arith.constant 4 : index
    %i2 = arith.index_cast %arg0 : i32 to index
    %d2 = arith.minsi %i2, %c4 : index

    %alloc = memref.alloc(%d0, %d2) : memref<?x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1, 2>,
      hivm.stride_align_value_in_byte = array<i32: 32, 32>
    } : memref<?x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 512 : i64}
      : memref<?x4x?xf32, #hivm.address_space<ub>>

    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 2048 : i64}
    // CHECK-NOT: buffer_size_in_byte = 1024
    hivm.hir.copy ins(%src : memref<?x4x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<?x4x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A padded extent whose bound is already a multiple of the alignment unit
// grows by a factor of one, so the annotation stays as it is.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @padded_extent_already_aligned
  func.func @padded_extent_already_aligned(%arg0: i32, %d0: index,
      %src: memref<?x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c8 = arith.constant 8 : index
    %i1 = arith.index_cast %arg0 : i32 to index
    %d1 = arith.minsi %i1, %c8 : index

    %alloc = memref.alloc(%d0, %d1) : memref<?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 1024 : i64}
      : memref<?x?xf32, #hivm.address_space<ub>>

    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 1024 : i64}
    hivm.hir.copy ins(%src : memref<?x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<?x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Regression: a padded static extent keeps the result of the previous
// static-volume ratio.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @padded_static_extent
  func.func @padded_static_extent(%d0: index,
      %src: memref<?x4xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc(%d0) : memref<?x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<?x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 512 : i64}
      : memref<?x4xf32, #hivm.address_space<ub>>

    // CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) : memref<?x8x1xf32
    // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 1024 : i64}
    hivm.hir.copy ins(%src : memref<?x4xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<?x4xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Regression: an extent that needs no padding leaves the annotation alone.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @unpadded_extent
  func.func @unpadded_extent(%d0: index,
      %src: memref<?x8xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc(%d0) : memref<?x8xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<?x8xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 512 : i64}
      : memref<?x8xf32, #hivm.address_space<ub>>

    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 512 : i64}
    hivm.hir.copy ins(%src : memref<?x8xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<?x8xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// A padded dynamic extent with no recoverable bound contributes no factor, and
// here it is the only padded extent, so no annotation changes.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @padded_extent_without_bound
  func.func @padded_extent_without_bound(%d0: index, %d1: index,
      %src: memref<?x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc(%d0, %d1) : memref<?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 512 : i64}
      : memref<?x?xf32, #hivm.address_space<ub>>

    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 512 : i64}
    hivm.hir.copy ins(%src : memref<?x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<?x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// An unbounded padded extent contributes no factor, but the factor of the
// extent next to it is known regardless: a static 4 padded to 8 doubles the
// buffer whatever the other extent does. Abandoning the rewrite over the
// unbounded one would leave the annotation at a value this pass is able to
// improve on, and below what it produced from the static extents alone.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @unbounded_extent_keeps_other_factors
  func.func @unbounded_extent_keeps_other_factors(%n: index,
      %src: memref<4x4x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc(%n) : memref<4x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1, 2>,
      hivm.stride_align_value_in_byte = array<i32: 32, 32>
    } : memref<4x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 512 : i64}
      : memref<4x4x?xf32, #hivm.address_space<ub>>

    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 1024 : i64}
    // CHECK-NOT: buffer_size_in_byte = 512
    hivm.hir.copy ins(%src : memref<4x4x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x4x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Two extents bounded by a clamp against a large constant. Accumulating the
// bounds into one fraction overflows int64 -- 8 * 2^25 * 2^25 is 2^53, and
// multiplying the recorded size by it needs 66 bits -- even though the ratio it
// stands for is barely above 8. Applying one factor at a time keeps every
// intermediate product well inside the range: 4096 -> 8192 -> 8193 -> 8194.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @wide_bounds_do_not_overflow
  func.func @wide_bounds_do_not_overflow(%arg0: i32, %arg1: i32,
      %src: memref<4x4x?x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %big = arith.constant 33554431 : index
    %i2 = arith.index_cast %arg0 : i32 to index
    %l2 = arith.maxsi %i2, %c0 : index
    %d2 = arith.minsi %l2, %big : index
    %i3 = arith.index_cast %arg1 : i32 to index
    %l3 = arith.maxsi %i3, %c0 : index
    %d3 = arith.minsi %l3, %big : index

    %alloc = memref.alloc(%d2, %d3)
      : memref<4x4x?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1, 2, 3>,
      hivm.stride_align_value_in_byte = array<i32: 32, 32, 32>
    } : memref<4x4x?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 4096 : i64}
      : memref<4x4x?x?xf32, #hivm.address_space<ub>>

    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 8194 : i64}
    // CHECK-NOT: buffer_size_in_byte = 4096
    hivm.hir.copy ins(%src : memref<4x4x?x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x4x?x?xf32, #hivm.address_space<ub>>)
    return
  }
}
