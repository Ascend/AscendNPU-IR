// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1 enable-unit-flag=true" %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2 enable-unit-flag=true" %s | FileCheck %s

// Focus on FIX destination hazards. Pool alloc/release enqueue FIX waits
// for both AIV completion notifications before recycling slots and IDs.
// Descriptor publication and consumer notifications are omitted here to
// isolate the WAW decision; these are compiler tests, not runnable kernels.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @distinct_allocations
  func.func @distinct_allocations(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a0, %id0 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    %a1, %id1 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 1 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst1 = hivm.hir.pointer_cast(%a1) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst1 : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: return
    return
  }

  // Independent pools cannot share the reclamation guarantee. Their full
  // ranges [0, 4096) and [2048, 6144) overlap, so keep the FIX WAW barrier.
  // CHECK-LABEL: func.func @overlapping_pools
  func.func @overlapping_pools(%meta0: memref<44xi32, #hivm.address_space<ssbuf>>, %meta1: memref<44xi32, #hivm.address_space<ssbuf>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %overlap = arith.constant 2048 : i64
    %pool0 = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %pool1 = hivm.hir.pointer_cast(%overlap) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %a0, %id0 = hivm.hir.tcb_pool_alloc %meta0, %pool0 {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    %a1, %id1 = hivm.hir.tcb_pool_alloc %meta1, %pool1 {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst1 = hivm.hir.pointer_cast(%a1) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK-NEXT: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst1 : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK: return
    return
  }

  // Independent pools with adjacent ranges [0, 4096) and [4096, 8192)
  // cannot alias, so no FIX barrier is needed.
  // CHECK-LABEL: func.func @disjoint_pools
  func.func @disjoint_pools(%meta0: memref<44xi32, #hivm.address_space<ssbuf>>, %meta1: memref<44xi32, #hivm.address_space<ssbuf>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %outside = arith.constant 4096 : i64
    %pool0 = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %pool1 = hivm.hir.pointer_cast(%outside) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %a0, %id0 = hivm.hir.tcb_pool_alloc %meta0, %pool0 {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    %a1, %id1 = hivm.hir.tcb_pool_alloc %meta1, %pool1 {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst1 = hivm.hir.pointer_cast(%a1) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst1 : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @mixed_pool_and_raw
  func.func @mixed_pool_and_raw(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %n: index, %cond: i1) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a0, %id0 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
    %raw = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<ub>>
    %mixed = arith.select %cond, %dst0, %raw : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%mixed : memref<16x16xf32, #hivm.address_space<ub>>)
    %a1, %id1 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 1 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst1 = hivm.hir.pointer_cast(%a1) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK-NEXT: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst1 : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @same_allocation
  func.func @same_allocation(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a0, %id0 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK-NEXT: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @allocate_each_iteration
  func.func @allocate_each_iteration(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %a0, %id0 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
      %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
      // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
      // CHECK: hivm.hir.fixpipe
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    }
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @allocation_outside_loop
  func.func @allocation_outside_loop(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a0, %id0 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: scf.for
    scf.for %i = %c0 to %n step %c1 {
      // CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_FIX>]
      // CHECK-NEXT: hivm.hir.fixpipe
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    }
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @allocation_outside_inner_loop
  func.func @allocation_outside_inner_loop(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: scf.for
    scf.for %outer = %c0 to %n step %c1 {
      %a0, %id0 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
      %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
      // CHECK: scf.for
      scf.for %i = %c0 to %n step %c1 {
        // CHECK-NEXT: hivm.hir.pipe_barrier[<PIPE_FIX>]
        // CHECK-NEXT: hivm.hir.fixpipe
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
      }
    }
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @raw_pool_alias
  func.func @raw_pool_alias(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a0, %id0 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    %inside = arith.constant 2048 : i64
    %raw = hivm.hir.pointer_cast(%inside) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK-NEXT: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%raw : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @nonoverlapping_raw_address
  func.func @nonoverlapping_raw_address(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %n: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %zero = arith.constant 0 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %src = hivm.hir.pointer_cast(%zero) : memref<16x16xf32, #hivm.address_space<cc>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a0, %id0 = hivm.hir.tcb_pool_alloc %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %dst0 = hivm.hir.pointer_cast(%a0) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%dst0 : memref<16x16xf32, #hivm.address_space<ub>>)
    %far = arith.constant 4096 : i64
    %raw = hivm.hir.pointer_cast(%far) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: hivm.hir.fixpipe
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%src : memref<16x16xf32, #hivm.address_space<cc>>) outs(%raw : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK-NOT: hivm.hir.pipe_barrier[<PIPE_FIX>]
    // CHECK: return
    return
  }

  // A get address may name any slot. Keep the V-to-MTE3 hazard against a raw view
  // inside the pool even when it lies beyond the returned tensor's byte size.
  // CHECK-LABEL: func.func @vector_get_alias
  func.func @vector_get_alias(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %out: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %zero = arith.constant 0 : i64
    %inside = arith.constant 2048 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %address, %id = hivm.hir.tcb_pool_get %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %src = hivm.hir.pointer_cast(%address) : memref<16x16xf32, #hivm.address_space<ub>>
    %raw = hivm.hir.pointer_cast(%inside) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.vadd
    hivm.hir.vadd ins(%src, %src : memref<16x16xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%src : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK-NEXT: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    // CHECK-NEXT: hivm.hir.store
    hivm.hir.store ins(%raw : memref<16x16xf32, #hivm.address_space<ub>>) outs(%out : memref<16x16xf32, #hivm.address_space<gm>>)
    // CHECK: return
    return
  }

  // The ordinary source starts at the end of the pool, so the V write and
  // MTE3 read do not overlap. No intra-core event is needed.
  // CHECK-LABEL: func.func @vector_get_no_alias
  func.func @vector_get_no_alias(%meta: memref<44xi32, #hivm.address_space<ssbuf>>, %out: memref<16x16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %zero = arith.constant 0 : i64
    %outside = arith.constant 4096 : i64
    %pool = hivm.hir.pointer_cast(%zero) : memref<4096xi8, #hivm.address_space<ub>>
    %address, %id = hivm.hir.tcb_pool_get %meta, %pool {segment = 0 : i32} : memref<44xi32, #hivm.address_space<ssbuf>>, memref<4096xi8, #hivm.address_space<ub>> -> (i64, i32)
    %src = hivm.hir.pointer_cast(%address) : memref<16x16xf32, #hivm.address_space<ub>>
    %raw = hivm.hir.pointer_cast(%outside) : memref<16x16xf32, #hivm.address_space<ub>>
    // CHECK-NOT: hivm.hir.set_flag
    // CHECK-NOT: hivm.hir.wait_flag
    // CHECK: hivm.hir.vadd
    hivm.hir.vadd ins(%src, %src : memref<16x16xf32, #hivm.address_space<ub>>, memref<16x16xf32, #hivm.address_space<ub>>) outs(%src : memref<16x16xf32, #hivm.address_space<ub>>)
    // CHECK-NOT: hivm.hir.set_flag
    // CHECK-NOT: hivm.hir.wait_flag
    // CHECK: hivm.hir.store
    hivm.hir.store ins(%raw : memref<16x16xf32, #hivm.address_space<ub>>) outs(%out : memref<16x16xf32, #hivm.address_space<gm>>)
    // CHECK-NOT: hivm.hir.set_flag
    // CHECK-NOT: hivm.hir.wait_flag
    // CHECK: return
    return
  }

}
