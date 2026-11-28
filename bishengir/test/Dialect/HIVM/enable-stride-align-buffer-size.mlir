// RUN: bishengir-opt -split-input-file -hivm-enable-stride-align %s | FileCheck %s
// RUN: bishengir-opt -split-input-file -hivm-infer-buffer-size-from-slice -hivm-enable-stride-align %s | FileCheck %s --check-prefix=INFER

// Enabling stride align re-allocates a buffer with padded extents, so a
// `buffer_size_in_byte` annotation computed on the unpadded shape no longer
// bounds the allocation. These tests preserve the existing extent-bound ratio
// estimate and its per-dimension rounding when intermediate values exceed i64.

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
// stands for is barely above 2. Preserve rounding at each step: the resulting
// estimate is 8194.

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

// -----

// A single factor whose intermediate product leaves the 64-bit range must not
// cost the other extents their factors. Here the middle extent is only bounded
// by a clamp against 2^52-1, so 2 KiB times its padded bound is 2^63,
// while the quotient needs 12 bits; the innermost extent grows from 4 to 8.
// Apply both factors with the existing per-dimension rounding: 2048 -> 2049 ->
// 4098. An overflow in the middle dimension must not skip the innermost factor.

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @wide_intermediate_keeps_other_factors
  func.func @wide_intermediate_keeps_other_factors(%arg0: i32,
      %src: memref<4x?x4xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %big = arith.constant 4503599627370495 : index
    %i1 = arith.index_cast %arg0 : i32 to index
    %l1 = arith.maxsi %i1, %c0 : index
    %rows = arith.minsi %l1, %big : index

    %alloc = memref.alloc(%rows) : memref<4x?x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1, 2>,
      hivm.stride_align_value_in_byte = array<i32: 64, 32>
    } : memref<4x?x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 2048 : i64}
      : memref<4x?x4xf32, #hivm.address_space<ub>>

    // The middle extent rounds to 2049 before the innermost factor doubles it.
    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 4098 : i64}
    // CHECK-NOT: buffer_size_in_byte = 2048
    hivm.hir.copy ins(%src : memref<4x?x4xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x?x4xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Swapping the factors still applies both. Per-dimension rounding now produces
// 2048 -> 4096 -> 4097 because the static factor comes first.
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @wide_intermediate_reordered
  func.func @wide_intermediate_reordered(%n: index,
      %src: memref<4x4x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %big = arith.constant 4503599627370495 : index
    %positive = arith.maxsi %n, %c0 : index
    %rows = arith.minsi %positive, %big : index
    %alloc = memref.alloc(%rows) : memref<4x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1, 2>,
      hivm.stride_align_value_in_byte = array<i32: 32, 64>
    } : memref<4x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 2048 : i64}
      : memref<4x4x?xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 4097 : i64}
    hivm.hir.copy ins(%src : memref<4x4x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x4x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// The padded capacity product of three wide bounds needs more than 128 bits.
// The actual shape 4x4x8x8x8 needs 65536 B after padding; per-dimension rounding
// adds one byte for each of the three dynamic factors.
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  func.func private @eight() -> index {
    %c8 = arith.constant 8 : index
    return %c8 : index
  }
  // CHECK-LABEL: func.func @product_exceeds_128_bits
  func.func @product_exceeds_128_bits(
      %src: memref<4x4x?x?x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %n = func.call @eight() : () -> index
    %c0 = arith.constant 0 : index
    %big = arith.constant 4503599627370495 : index
    %positive = arith.maxsi %n, %c0 : index
    %rows = arith.minsi %positive, %big : index
    %alloc = memref.alloc(%rows, %rows, %rows)
      : memref<4x4x?x?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1, 2, 3, 4>,
      hivm.stride_align_value_in_byte = array<i32: 32, 32, 32, 32>
    } : memref<4x4x?x?x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 32768 : i64}
      : memref<4x4x?x?x?xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 65539 : i64}
    hivm.hir.copy ins(%src : memref<4x4x?x?x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x4x?x?x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Aligning the analysis bound INT64_MAX-8 to 16 produces 2^63. It must
// remain positive and contribute its factor. This covers alloca and two marks.
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @padded_bound_exceeds_int64
  func.func @padded_bound_exceeds_int64(%n: index,
      %src: memref<4x?x4xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %big = arith.constant 9223372036854775799 : index
    %positive = arith.maxsi %n, %c0 : index
    %rows = arith.minsi %positive, %big : index
    %alloc = memref.alloca(%rows) : memref<4x?x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1, 2>,
      hivm.stride_align_value_in_byte = array<i32: 64, 32>
    } : memref<4x?x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 2048 : i64}
      : memref<4x?x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 4096 : i64}
      : memref<4x?x4xf32, #hivm.address_space<ub>>
    // CHECK: memref.alloca
    // CHECK-DAG: annotation.mark %{{.*}} {buffer_size_in_byte = 4098 : i64}
    // CHECK-DAG: annotation.mark %{{.*}} {buffer_size_in_byte = 8194 : i64}
    hivm.hir.copy ins(%src : memref<4x?x4xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x?x4xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// Apply the padded capacity cap before checking the final int64_t range.
// This loose annotation already covers the padded allocation and stays intact.
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @cap_before_range_check
  func.func @cap_before_range_check(
      %src: memref<4x4xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc() : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<4x4xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 4611686018427387904 : i64}
      : memref<4x4xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 4611686018427387904 : i64}
    hivm.hir.copy ins(%src : memref<4x4xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x4xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// If the final estimate cannot fit in int64_t, stop updating the size marks.
// Preserve the existing annotation and continue the pass for alloc and alloca.
module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @alloc_size_overflow
  func.func @alloc_size_overflow(%n: index,
      %src: memref<4x4x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloc(%n) : memref<4x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<4x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 4611686018427387904 : i64}
      : memref<4x4x?xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 4611686018427387904 : i64}
    hivm.hir.copy ins(%src : memref<4x4x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x4x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

module attributes {
  hacc.target = #hacc.target<"Ascend950PR_9579">,
  hivm.module_core_type = #hivm.module_core_type<AIV>
} {
  // CHECK-LABEL: func.func @alloca_size_overflow
  func.func @alloca_size_overflow(%n: index,
      %src: memref<4x4x?xf32, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %alloc = memref.alloca(%n) : memref<4x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {
      hivm.stride_align_dims = array<i32: 1>,
      hivm.stride_align_value_in_byte = array<i32: 32>
    } : memref<4x4x?xf32, #hivm.address_space<ub>>
    annotation.mark %alloc {buffer_size_in_byte = 4611686018427387904 : i64}
      : memref<4x4x?xf32, #hivm.address_space<ub>>
    // CHECK: annotation.mark %{{.*}} {buffer_size_in_byte = 4611686018427387904 : i64}
    hivm.hir.copy ins(%src : memref<4x4x?xf32, #hivm.address_space<ub>>)
                  outs(%alloc : memref<4x4x?xf32, #hivm.address_space<ub>>)
    return
  }
}

// -----

// InferBufferSizeFromSlice produces a 36 B mark from the static source.
// Preserve per-dimension rounding even when that basis differs from the bounds.
// After padding: 1x16x8x16x8x1. Last logical f32 occupies bytes 256..259.
// Old per-dimension ceil: 36 -> 48 -> 54 -> 65 -> 260.
// One final ceil: 36*(32/24)*(40/36)*(24/20)*(8/2) = 256.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">,
                   hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func private @one() -> index {
    %v = arith.constant 1 : index
    return %v : index
  }
  func.func private @nine() -> index {
    %v = arith.constant 9 : index
    return %v : index
  }
  // INFER-LABEL: func.func @rounding_boundary
  func.func @rounding_boundary(%src: memref<1x1x1x9x1xf32, #hivm.address_space<gm>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %one = func.call @one() : () -> index
    %nine = func.call @nine() : () -> index
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c36 = arith.constant 36 : index
    %c20 = arith.constant 20 : index
    %c2 = arith.constant 2 : index
    %p1 = arith.maxsi %one, %c0 : index
    %p9 = arith.maxsi %nine, %c0 : index
    %d1 = arith.minsi %p1, %c24 : index
    %d2 = arith.minsi %p1, %c36 : index
    %d3 = arith.minsi %p9, %c20 : index
    %d4 = arith.minsi %p1, %c2 : index
    %dst = memref.alloc(%d1, %d2, %d3, %d4)
      : memref<1x?x?x?x?xf32, #hivm.address_space<ub>>
    annotation.mark %dst {hivm.stride_align_dims = array<i32: 1, 2, 3, 4>,
                          hivm.stride_align_value_in_byte = array<i32: 64, 32, 32, 32>}
      : memref<1x?x?x?x?xf32, #hivm.address_space<ub>>
    // INFER: annotation.mark %{{.*}} {buffer_size_in_byte = 260 : i64}
    %slice = memref.subview %src[0, 0, 0, 0, 0] [1, %d1, %d2, %d3, %d4] [1, 1, 1, 1, 1]
      : memref<1x1x1x9x1xf32, #hivm.address_space<gm>>
        to memref<1x?x?x?x?xf32, strided<[9, 9, 9, 1, 1]>, #hivm.address_space<gm>>
    hivm.hir.load ins(%slice : memref<1x?x?x?x?xf32, strided<[9, 9, 9, 1, 1]>, #hivm.address_space<gm>>)
                  outs(%dst : memref<1x?x?x?x?xf32, #hivm.address_space<ub>>)
    return
  }
}
