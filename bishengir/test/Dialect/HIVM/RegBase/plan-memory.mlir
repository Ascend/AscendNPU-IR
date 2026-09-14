// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend910B1 -hivm-plan-memory-regbase -split-input-file | FileCheck %s

// Regression (Ascend910B1 UB=192KB / 1572864 bits): independently planned
// multi-buffer "other" slots at a higher speculative level than the first
// buffer must be valid rollback stop points so they can retry at level-0.
// 1D shapes; end VFs need minimal memref uses so VF inplace-reuse analysis
// matches the production transpose pipeline (middle VF may be empty).
// CHECK: warning: [hivm-plan-memory] There reused some dma buffers in ub address space, which may stall pipe. Not reusing dma buffer needs 2224896 bits while 1572864 bits available!
// CHECK-LABEL: func.func @mb_other_rollback_level0_retry(
// CHECK-DAG: %[[STORE_STRIDE:.*]] = arith.constant 57600 : i64
// CHECK-DAG: %[[MID_OFF:.*]] = arith.constant 115200 : i64
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[LOAD_STRIDE:.*]] = arith.constant 55456 : i64
// CHECK: hivm.hir.pointer_cast(%[[ZERO]], %[[LOAD_STRIDE]])
// CHECK-SAME: memref<6932xi64, #hivm.address_space<ub>>
// CHECK: annotation.mark {{.*}} {hivm.multi_buffer = 2 : i32}
// CHECK: hivm.hir.pointer_cast(%[[ZERO]], %[[LOAD_STRIDE]])
// CHECK-SAME: memref<5985xi64, #hivm.address_space<ub>>
// CHECK: hivm.hir.pointer_cast(%[[MID_OFF]])
// CHECK-SAME: memref<6500xi64, #hivm.address_space<ub>>
// CHECK: hivm.hir.pointer_cast(%[[ZERO]], %[[STORE_STRIDE]])
// CHECK-SAME: memref<7200xi64, #hivm.address_space<ub>>
// CHECK: annotation.mark {{.*}} {hivm.multi_buffer = 2 : i32}
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func private @mb_other_rollback_vf0(
      %a: memref<5985xi64, strided<[1]>, #hivm.address_space<ub>>,
      %b: memref<5985xi64, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %mask = vector.constant_mask [1] : vector<64xi1>
    %src = memref.subview %a[0] [1] [1]
      : memref<5985xi64, strided<[1]>, #hivm.address_space<ub>>
      to memref<1xi64, strided<[1]>, #hivm.address_space<ub>>
    %dst = memref.subview %b[0] [1] [1]
      : memref<5985xi64, #hivm.address_space<ub>>
      to memref<1xi64, #hivm.address_space<ub>>
    %v = vector.transfer_read %src[%c0], %c0_i64, %mask {in_bounds = [true]}
      : memref<1xi64, strided<[1]>, #hivm.address_space<ub>>, vector<64xi64>
    vector.transfer_write %v, %dst[%c0], %mask {in_bounds = [true]}
      : vector<64xi64>, memref<1xi64, #hivm.address_space<ub>>
    return
  }

  func.func private @mb_other_rollback_vf1(
      %a: memref<5985xi64, #hivm.address_space<ub>>,
      %b: memref<6500xi64, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.vector_function, no_inline} {
    return
  }

  func.func private @mb_other_rollback_vf2(
      %a: memref<6500xi64, #hivm.address_space<ub>>,
      %b: memref<6500xi64, strided<[1]>, #hivm.address_space<ub>>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %mask = vector.constant_mask [1] : vector<64xi1>
    %src = memref.subview %a[0] [1] [1]
      : memref<6500xi64, #hivm.address_space<ub>>
      to memref<1xi64, #hivm.address_space<ub>>
    %dst = memref.subview %b[0] [1] [1]
      : memref<6500xi64, strided<[1]>, #hivm.address_space<ub>>
      to memref<1xi64, strided<[1]>, #hivm.address_space<ub>>
    %v = vector.transfer_read %src[%c0], %c0_i64, %mask {in_bounds = [true]}
      : memref<1xi64, #hivm.address_space<ub>>, vector<64xi64>
    vector.transfer_write %v, %dst[%c0], %mask {in_bounds = [true]}
      : vector<64xi64>, memref<1xi64, strided<[1]>, #hivm.address_space<ub>>
    return
  }

  func.func @mb_other_rollback_level0_retry(
      %src: memref<5985xi64, #hivm.address_space<gm>>,
      %dst: memref<6500xi64, #hivm.address_space<gm>>) {
    %load_ub = memref.alloc() : memref<6932xi64, #hivm.address_space<ub>>
    annotation.mark %load_ub {hivm.multi_buffer = 2 : i32}
      : memref<6932xi64, #hivm.address_space<ub>>
    %load_v = memref.subview %load_ub[0] [5985] [1]
      : memref<6932xi64, #hivm.address_space<ub>>
      to memref<5985xi64, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.load
        ins(%src : memref<5985xi64, #hivm.address_space<gm>>)
        outs(%load_v : memref<5985xi64, strided<[1]>, #hivm.address_space<ub>>)

    %mid0 = memref.alloc() : memref<5985xi64, #hivm.address_space<ub>>
    func.call @mb_other_rollback_vf0(%load_v, %mid0)
      {hivm.vector_function, no_inline}
      : (memref<5985xi64, strided<[1]>, #hivm.address_space<ub>>,
         memref<5985xi64, #hivm.address_space<ub>>) -> ()

    %mid1 = memref.alloc() : memref<6500xi64, #hivm.address_space<ub>>
    func.call @mb_other_rollback_vf1(%mid0, %mid1)
      {hivm.vector_function, no_inline}
      : (memref<5985xi64, #hivm.address_space<ub>>,
         memref<6500xi64, #hivm.address_space<ub>>) -> ()

    %store_ub = memref.alloc() : memref<7200xi64, #hivm.address_space<ub>>
    annotation.mark %store_ub {hivm.multi_buffer = 2 : i32}
      : memref<7200xi64, #hivm.address_space<ub>>
    %store_v = memref.subview %store_ub[0] [6500] [1]
      : memref<7200xi64, #hivm.address_space<ub>>
      to memref<6500xi64, strided<[1]>, #hivm.address_space<ub>>
    func.call @mb_other_rollback_vf2(%mid1, %store_v)
      {hivm.vector_function, no_inline}
      : (memref<6500xi64, #hivm.address_space<ub>>,
         memref<6500xi64, strided<[1]>, #hivm.address_space<ub>>) -> ()
    hivm.hir.store
        ins(%store_v : memref<6500xi64, strided<[1]>, #hivm.address_space<ub>>)
        outs(%dst : memref<6500xi64, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  // Loop-carried vadd dest whose init is arith.select must not share a UB
  // offset with the sibling src (chunk_abc alloc_24 / alloc_20 pattern).
  // CHECK-LABEL: func.func @test_select_init_iter_no_reuse_yield
  func.func @test_select_init_iter_no_reuse_yield(
      %src0: memref<64xf32, #hivm.address_space<gm>>,
      %src1: memref<64xf32, #hivm.address_space<gm>>,
      %tmp_src: memref<64xf32, #hivm.address_space<gm>>,
      %dst: memref<64xf32, #hivm.address_space<gm>>,
      %cond: i1) {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
    // CHECK-DAG: %[[C256:.*]] = arith.constant 256 : i64
    // CHECK-DAG: %[[C512:.*]] = arith.constant 512 : i64
    // CHECK-DAG: %[[C768:.*]] = arith.constant 768 : i64
    // CHECK: hivm.hir.pointer_cast(%[[C0]]) : memref<64xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[C256]]) : memref<64xf32, #hivm.address_space<ub>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %init0 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    %init1 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src0 : memref<64xf32, #hivm.address_space<gm>>)
                  outs(%init0 : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%src1 : memref<64xf32, #hivm.address_space<gm>>)
                  outs(%init1 : memref<64xf32, #hivm.address_space<ub>>)
    %init = arith.select %cond, %init0, %init1 : memref<64xf32, #hivm.address_space<ub>>
    %acc = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %init)
        -> (memref<64xf32, #hivm.address_space<ub>>) {
      // Sibling src (tmp) and yielded dest (out) must get distinct offsets.
      // CHECK: %[[TMP:.*]] = hivm.hir.pointer_cast(%[[C512]]) : memref<64xf32, #hivm.address_space<ub>>
      // CHECK: %[[OUT:.*]] = hivm.hir.pointer_cast(%[[C768]]) : memref<64xf32, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vadd ins(%{{.*}}, %[[TMP]] : {{.*}}) outs(%[[OUT]] :
      %tmp = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%tmp_src : memref<64xf32, #hivm.address_space<gm>>)
                    outs(%tmp : memref<64xf32, #hivm.address_space<ub>>)
      %out = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%iter, %tmp : memref<64xf32, #hivm.address_space<ub>>,
                                      memref<64xf32, #hivm.address_space<ub>>)
                    outs(%out : memref<64xf32, #hivm.address_space<ub>>)
      scf.yield %out : memref<64xf32, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%acc : memref<64xf32, #hivm.address_space<ub>>)
                   outs(%dst : memref<64xf32, #hivm.address_space<gm>>)
    return
  }
}
