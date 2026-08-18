// RUN: bishengir-opt %s --convert-hivm-to-tritongpu | FileCheck %s

// A memref.subview stacked on a memref.reinterpret_cast. Under the 1:N MemRef
// descriptor conversion each view op rewrites the descriptor in place, so the
// two offsets compose:
//
//   reinterpret_cast  ->  base = root,  offset = 3          (RESET)
//   subview           ->  base = root,  offset = 3 + %arg1  (COMPOSE)

module {
  // CHECK-LABEL: tt.func @nested_reinterpret_subview
  // CHECK-SAME:  %{{arg[0-9]+}}: !tt.ptr<i64, 6>, %[[ROOT:arg[0-9]+]]: !tt.ptr<i64, 6>,

  // The source stride is 1, so the index*stride term folds away and the
  // composition is just the subview's dynamic offset + the cast's 3.
  // CHECK:       %[[DYN:.*]] = arith.index_cast %{{.*}} : index to i64
  // CHECK:       %[[OFF:.*]] = arith.addi %[[DYN]], %{{c3_i64[_0-9]*}} : i64

  // CHECK:       %[[RANGE:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32}
  // CHECK:       %[[OFF32:.*]] = arith.trunci %[[OFF]] : i64 to i32
  // CHECK:       %[[SOFF:.*]] = tt.splat %[[OFF32]] : i32 -> tensor<2xi32>
  // CHECK:       %[[IDX:.*]] = arith.addi %[[RANGE]], %[[SOFF]] : tensor<2xi32>
  // CHECK:       %[[BASE:.*]] = tt.splat %[[ROOT]] : !tt.ptr<i64, 6>
  // CHECK:       %[[PTRS:.*]] = tt.addptr %[[BASE]], %[[IDX]]
  // CHECK:       tt.load %[[PTRS]]

  // CHECK-NOT:   memref.reinterpret_cast
  // CHECK-NOT:   memref.subview
  // CHECK-NOT:   unrealized_conversion_cast
  func.func @nested_reinterpret_subview(
      %arg0: memref<?xi64, #hivm.address_space<ub>>, %arg1: index,
      %arg2: memref<2xi64, #hivm.address_space<ub>>)
      attributes {no_inline, outline, vector_function,
                  vf_mode = #hivm.vf_mode<SIMT>} {
    %view = memref.reinterpret_cast %arg0 to offset: [3], sizes: [8], strides: [1] : memref<?xi64, #hivm.address_space<ub>> to memref<8xi64, strided<[1], offset: 3>, #hivm.address_space<ub>>
    %subview = memref.subview %view[%arg1] [2] [1] : memref<8xi64, strided<[1], offset: 3>, #hivm.address_space<ub>> to memref<2xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %0 = bufferization.to_tensor %subview restrict writable : memref<2xi64, strided<[1], offset: ?>, #hivm.address_space<ub>>
    hivm.hir.local_store ins(%arg2 : memref<2xi64, #hivm.address_space<ub>>, %0 : tensor<2xi64>)
    return
  }
}
