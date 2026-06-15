// RUN: bishengir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @test_fold_memref_fixpipe
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4x1xf32, #hivm.address_space<ub>>
// CHECK: %[[OUT:.*]] = memref.reinterpret_cast %[[ALLOC]]
// CHECK-NOT: memref.cast
// CHECK: hivm.hir.fixpipe
// CHECK-SAME: outs(%[[OUT]] : memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>)
// CHECK-NOT: memref.cast
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_fold_memref_fixpipe() attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %c0 = arith.constant 0 : index

    %alloc = memref.alloc()
      : memref<4x1xf32, #hivm.address_space<ub>>

    %alloc_0 = memref.alloc() {alignment = 64 : i64}
      : memref<4x1xf32, #hivm.address_space<cc>>

    %out_dyn_offset = memref.reinterpret_cast %alloc
      to offset: [%c0], sizes: [4, 1], strides: [1, 1]
      : memref<4x1xf32, #hivm.address_space<ub>>
      to memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>

    %out_static_offset = memref.cast %out_dyn_offset
      : memref<4x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      to memref<4x1xf32, strided<[1, 1], offset: 0>, #hivm.address_space<ub>>

    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>}
      ins(%alloc_0 : memref<4x1xf32, #hivm.address_space<cc>>)
      outs(%out_static_offset : memref<4x1xf32, strided<[1, 1], offset: 0>, #hivm.address_space<ub>>)

    return
  }
}
