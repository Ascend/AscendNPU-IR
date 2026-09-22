// RUN: bishengir-opt %s -loop-restructure-arange-optimization | FileCheck %s
// RUN: bishengir-opt %s -loop-restructure-arange-optimization | FileCheck %s --check-prefix=ATOMIC

// ATOMIC-LABEL: tt.func public @preserve_reduce_region
// ATOMIC-NOT: group_id
// ATOMIC-LABEL: tt.func public @preserve_non_reduce_region
// ATOMIC-NOT: group_id
// ATOMIC-LABEL: tt.func public @optimize_with_unrelated_reduce

// CHECK-LABEL: tt.func public @preserve_reduce_region
// CHECK: tt.make_range {end = 8 : i32, start = 0 : i32}
// CHECK: "tt.reduce"{{.*}}({
// CHECK: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32):
// CHECK:   %[[MAX:.*]] = arith.maxsi %[[LHS]], %[[RHS]] : i32
// CHECK:   tt.reduce.return %[[MAX]] : i32
// CHECK: })
// CHECK: tt.return

// CHECK-LABEL: tt.func public @preserve_non_reduce_region
// CHECK: scf.if
// CHECK: tt.return

// CHECK-LABEL: tt.func public @optimize_with_unrelated_reduce
// Safe stores are rewritten in place, before the unrelated reduction.
// CHECK: tt.store {{.*}} {group_id = 0 : i32} : tensor<2x4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}} {group_id = 1 : i32} : tensor<2x2x!tt.ptr<i32>>
// CHECK: "tt.reduce"{{.*}}({
// CHECK: tt.return

module attributes {"ttg.simt-optimization-mode" = 200000 : i32} {
  // This function contains one unsupported reduction-dependent candidate and
  // one otherwise valid candidate. The pass is intentionally function-atomic:
  // neither candidate is rewritten until partial rewriting can prove aliasing
  // and memory order safe.
  tt.func public @preserve_reduce_region(
      %src: !tt.ptr<i32>, %dst: !tt.ptr<i32>,
      %valid_src: !tt.ptr<i32>, %valid_dst: !tt.ptr<i32>) {
    %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %indices = tt.broadcast %expanded : tensor<1x8xi32> -> tensor<2x8xi32>
    %src_splat = tt.splat %src : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %src_ptrs = tt.addptr %src_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %loaded = tt.load %src_ptrs : tensor<2x8x!tt.ptr<i32>>

    %four = arith.constant dense<4> : tensor<2x8xi32>
    %minus_one = arith.constant dense<-1> : tensor<2x8xi32>
    %within_embedding = arith.cmpi slt, %indices, %four : tensor<2x8xi32>
    %selected = arith.select %within_embedding, %indices, %minus_one : tensor<2x8xi1>, tensor<2x8xi32>
    %maximum = "tt.reduce"(%selected) <{axis = 1 : i32}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %max = arith.maxsi %lhs, %rhs : i32
      tt.reduce.return %max : i32
    }) : (tensor<2x8xi32>) -> tensor<2xi32>
    %maximum_expanded = tt.expand_dims %maximum {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %maximum_broadcast = tt.broadcast %maximum_expanded : tensor<2x1xi32> -> tensor<2x8xi32>
    %zero = arith.constant dense<0> : tensor<2x8xi32>
    %store_mask = arith.cmpi sge, %maximum_broadcast, %zero : tensor<2x8xi32>

    %dst_splat = tt.splat %dst : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dst_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    tt.store %dst_ptrs, %loaded, %store_mask : tensor<2x8x!tt.ptr<i32>>

    %valid_src_splat = tt.splat %valid_src : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %valid_src_ptrs = tt.addptr %valid_src_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %valid_loaded = tt.load %valid_src_ptrs : tensor<2x8x!tt.ptr<i32>>
    %valid_mask = arith.cmpi slt, %indices, %four : tensor<2x8xi32>
    %valid_dst_splat = tt.splat %valid_dst : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %valid_dst_ptrs = tt.addptr %valid_dst_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    tt.store %valid_dst_ptrs, %valid_loaded, %valid_mask : tensor<2x8x!tt.ptr<i32>>
    tt.return
  }

  // Region-bearing dependencies other than tt.reduce are equally unsupported
  // by the regionless clone path.
  tt.func public @preserve_non_reduce_region(
      %src: !tt.ptr<i32>, %dst: !tt.ptr<i32>, %condition: i1) {
    %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %indices = tt.broadcast %expanded : tensor<1x8xi32> -> tensor<2x8xi32>
    %src_splat = tt.splat %src : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %src_ptrs = tt.addptr %src_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %loaded = tt.load %src_ptrs : tensor<2x8x!tt.ptr<i32>>
    %four = arith.constant dense<4> : tensor<2x8xi32>
    %mask = arith.cmpi slt, %indices, %four : tensor<2x8xi32>
    %region_mask = scf.if %condition -> (tensor<2x8xi1>) {
      scf.yield %mask : tensor<2x8xi1>
    } else {
      %false = arith.constant dense<false> : tensor<2x8xi1>
      scf.yield %false : tensor<2x8xi1>
    }
    %dst_splat = tt.splat %dst : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %dst_ptrs = tt.addptr %dst_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    tt.store %dst_ptrs, %loaded, %region_mask : tensor<2x8x!tt.ptr<i32>>
    tt.return
  }

  tt.func public @optimize_with_unrelated_reduce(
      %src0: !tt.ptr<i32>, %dst0: !tt.ptr<i32>,
      %src1: !tt.ptr<i32>, %dst1: !tt.ptr<i32>) {
    %range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %indices = tt.broadcast %expanded : tensor<1x8xi32> -> tensor<2x8xi32>

    %src0_splat = tt.splat %src0 : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %src0_ptrs = tt.addptr %src0_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %loaded0 = tt.load %src0_ptrs : tensor<2x8x!tt.ptr<i32>>
    %four = arith.constant dense<4> : tensor<2x8xi32>
    %mask0 = arith.cmpi slt, %indices, %four : tensor<2x8xi32>
    %dst0_splat = tt.splat %dst0 : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %dst0_ptrs = tt.addptr %dst0_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    tt.store %dst0_ptrs, %loaded0, %mask0 : tensor<2x8x!tt.ptr<i32>>

    %src1_splat = tt.splat %src1 : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %src1_ptrs = tt.addptr %src1_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    %loaded1 = tt.load %src1_ptrs : tensor<2x8x!tt.ptr<i32>>
    %two = arith.constant dense<2> : tensor<2x8xi32>
    %mask1 = arith.cmpi slt, %indices, %two : tensor<2x8xi32>
    %dst1_splat = tt.splat %dst1 : !tt.ptr<i32> -> tensor<2x8x!tt.ptr<i32>>
    %dst1_ptrs = tt.addptr %dst1_splat, %indices : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
    tt.store %dst1_ptrs, %loaded1, %mask1 : tensor<2x8x!tt.ptr<i32>>

    %unused = "tt.reduce"(%indices) <{axis = 1 : i32}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      tt.reduce.return %sum : i32
    }) : (tensor<2x8xi32>) -> tensor<2xi32>
    tt.return
  }
}
