// RUN: bishengir-opt %s -hivm-plan-memory -split-input-file -verify-diagnostics | FileCheck %s

// -----
module {
  func.func @test_mem_allocate_basic(%src : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                    %dst3 : memref<16x16x16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %copy_in_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %dst1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %dst2 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %copy_out_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%copy_in_ub : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%copy_in_ub, %copy_in_ub : memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%dst1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%dst1, %dst1 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%dst2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%dst2, %dst2 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%copy_out_ub : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%copy_out_ub : memref<16x16x16xf16,#hivm.address_space<ub>>)
                   outs(%dst3: memref<16x16x16xf16,#hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_infer_mem_allocate_loop_conflict(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                                   %alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                                   %alloc6 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                                   %alloc8 : memref<16x16x16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %start = arith.constant 0 : index
    %end = arith.constant 1024 : index
    %step = arith.constant 128 : index
    scf.for %iv = %start to %end step %step {
      // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
      %alloc1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
      // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
      %alloc3 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
      // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST1]])
      %alloc5 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
      // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST1]])
      %alloc7 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                     outs(%alloc1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc1, %alloc1 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                        memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%alloc3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                     outs(%alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>)
      hivm.hir.load ins(%alloc6 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                    outs(%alloc5 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc5, %alloc5 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                        memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%alloc7 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc7 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                     outs(%alloc8 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----
module {
    func.func @vadd_inplace(%lhs_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                             %rhs_gm :  memref<16x16xf16, #hivm.address_space<gm>>,
                             %vadd_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST1:.*]] = arith.constant 512 : i64
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
      %lhs_ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%lhs_gm : memref<16x16xf16, #hivm.address_space<gm>>)
                    outs(%lhs_ub : memref<16x16xf16, #hivm.address_space<ub>>)
      // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST1]])
      %rhs_ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%rhs_gm : memref<16x16xf16, #hivm.address_space<gm>>)
                    outs(%rhs_ub : memref<16x16xf16, #hivm.address_space<ub>>)
      // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
      %vadd_res_ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%lhs_ub, %rhs_ub : memref<16x16xf16, #hivm.address_space<ub>>,
                        memref<16x16xf16, #hivm.address_space<ub>>)
                    outs(%vadd_res_ub : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%vadd_res_ub : memref<16x16xf16, #hivm.address_space<ub>>)
                     outs(%vadd_gm : memref<16x16xf16, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_2d_small_to_large_invalid(%arg0_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                                                       %vcast_gm :memref<16x16xf32, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST512:.*]] = arith.constant 512 : i64
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<16x16xf16, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<16x16xf16, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST512]])
      %vcast_res_ub = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<16x16xf16, #hivm.address_space<ub>>) outs(%[[DST]] : memref<16x16xf32, #hivm.address_space<ub>>)
      hivm.hir.vcast ins(%arg0_ub : memref<16x16xf16, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<16x16xf32, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<rint>
      hivm.hir.store ins(%vcast_res_ub : memref<16x16xf32, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<16x16xf32, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_2d_equal_valid(%arg0_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                                            %vcast_gm :memref<16x16xi16, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<16x16xf16, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<16x16xf16, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %vcast_res_ub = memref.alloc() : memref<16x16xi16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<16x16xf16, #hivm.address_space<ub>>) outs(%[[DST]] : memref<16x16xi16, #hivm.address_space<ub>>) round_mode = <trunc>
      hivm.hir.vcast ins(%arg0_ub : memref<16x16xf16, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<16x16xi16, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<trunc>
      hivm.hir.store ins(%vcast_res_ub : memref<16x16xi16, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<16x16xi16, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_2d_large_to_small_invalid(%arg0_gm : memref<16x16xf32, #hivm.address_space<gm>>,
                                                       %vcast_gm :memref<16x16xf16, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST1024:.*]] = arith.constant 1024 : i64
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<16x16xf32, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<16x16xf32, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST1024]])
      %vcast_res_ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<16x16xf32, #hivm.address_space<ub>>) outs(%[[DST]] : memref<16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vcast ins(%arg0_ub : memref<16x16xf32, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<16x16xf16, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<trunc>
      hivm.hir.store ins(%vcast_res_ub : memref<16x16xf16, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<16x16xf16, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_1d_small_to_large_invalid(%arg0_gm : memref<1024xf16, #hivm.address_space<gm>>,
                                                       %vcast_gm :memref<1024xf32, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST2048:.*]] = arith.constant 2048 : i64
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<1024xf16, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST2048]])
      %vcast_res_ub = memref.alloc() : memref<1024xf32, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<1024xf16, #hivm.address_space<ub>>) outs(%[[DST]] : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.vcast ins(%arg0_ub : memref<1024xf16, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<1024xf32, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<rint>
      hivm.hir.store ins(%vcast_res_ub : memref<1024xf32, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<1024xf32, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_1d_equal_valid(%arg0_gm : memref<1024xf16, #hivm.address_space<gm>>,
                                            %vcast_gm :memref<1024xi16, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<1024xf16, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<1024xf16, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %vcast_res_ub = memref.alloc() : memref<1024xi16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<1024xf16, #hivm.address_space<ub>>) outs(%[[DST]] : memref<1024xi16, #hivm.address_space<ub>>) round_mode = <trunc>
      hivm.hir.vcast ins(%arg0_ub : memref<1024xf16, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<1024xi16, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<trunc>
      hivm.hir.store ins(%vcast_res_ub : memref<1024xi16, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<1024xi16, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_1d_large_to_small_valid(%arg0_gm : memref<1024xf32, #hivm.address_space<gm>>,
                                                     %vcast_gm :memref<1024xf16, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<1024xf32, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<1024xf32, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %vcast_res_ub = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<1024xf32, #hivm.address_space<ub>>) outs(%[[DST]] : memref<1024xf16, #hivm.address_space<ub>>)
      hivm.hir.vcast ins(%arg0_ub : memref<1024xf32, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<1024xf16, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<trunc>
      hivm.hir.store ins(%vcast_res_ub : memref<1024xf16, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<1024xf16, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_1d_s2l_stride2_invalid(%arg0_gm : memref<1024xf16, strided<[2]>, #hivm.address_space<gm>>,
                                                    %vcast_gm :memref<1024xf32, strided<[2]>, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST2048:.*]] = arith.constant 2048 : i64
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<1024xf16, strided<[2]>, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST2048]])
      %vcast_res_ub = memref.alloc() : memref<1024xf32, strided<[2]>, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>) outs(%[[DST]] : memref<1024xf32, strided<[2]>, #hivm.address_space<ub>>)
      hivm.hir.vcast ins(%arg0_ub : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<1024xf32, strided<[2]>, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<rint>
      hivm.hir.store ins(%vcast_res_ub : memref<1024xf32, strided<[2]>, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<1024xf32, strided<[2]>, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_1d_equal_stride2_valid(%arg0_gm : memref<1024xf16, strided<[2]>, #hivm.address_space<gm>>,
                                                    %vcast_gm :memref<1024xi16, strided<[2]>, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<1024xf16, strided<[2]>, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %vcast_res_ub = memref.alloc() : memref<1024xi16, strided<[2]>, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>) outs(%[[DST]] : memref<1024xi16, strided<[2]>, #hivm.address_space<ub>>) round_mode = <trunc>
      hivm.hir.vcast ins(%arg0_ub : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<1024xi16, strided<[2]>, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<trunc>
      hivm.hir.store ins(%vcast_res_ub : memref<1024xi16, strided<[2]>, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<1024xi16, strided<[2]>, #hivm.address_space<gm>>)
      return
    }
}

// -----
module {
    func.func @vcast_inplace_1d_l2s_stride2_invalid(%arg0_gm : memref<1024xf32, strided<[2]>, #hivm.address_space<gm>>,
                                                    %vcast_gm :memref<1024xf16, strided<[2]>, #hivm.address_space<gm>>) {
      // CHECK-NOT: memref.alloc()
      // CHECK: %[[CONST4096:.*]] = arith.constant 4096 : i64
      // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
      // CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[CONST0]])
      %arg0_ub = memref.alloc() : memref<1024xf32, strided<[2]>, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0_gm : memref<1024xf32, strided<[2]>, #hivm.address_space<gm>>)
                    outs(%arg0_ub : memref<1024xf32, strided<[2]>, #hivm.address_space<ub>>)
      // CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[CONST4096]])
      %vcast_res_ub = memref.alloc() : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>
      // CHECK: hivm.hir.vcast ins(%[[SRC]] : memref<1024xf32, strided<[2]>, #hivm.address_space<ub>>) outs(%[[DST]] : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>)
      hivm.hir.vcast ins(%arg0_ub : memref<1024xf32, strided<[2]>, #hivm.address_space<ub>>)
      outs(%vcast_res_ub : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<trunc>
      hivm.hir.store ins(%vcast_res_ub : memref<1024xf16, strided<[2]>, #hivm.address_space<ub>>)
                     outs(%vcast_gm : memref<1024xf16, strided<[2]>, #hivm.address_space<gm>>)
      return
    }
}

// -----

module {
  func.func @test_infer_plan_memory_if_yield(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %alloc6 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %alloc8 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %cond: i1) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST5:.*]] = arith.constant 24576 : i64
    // CHECK: %[[CONST4:.*]] = arith.constant 16384 : i64
    // CHECK: %[[CONST3:.*]] = arith.constant 40960 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 32768 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %alloc3 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %alloc5 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
    %alloc7 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
    %alloc9 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST5]])
    %alloc10 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                 outs(%alloc1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc5 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%alloc6 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc9 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%alloc8 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc10 : memref<16x16x16xf16, #hivm.address_space<ub>>)

    %0 = scf.if %cond -> (memref<16x16x16xf16, #hivm.address_space<ub>>) {
      hivm.hir.vadd ins(%alloc1, %alloc9 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                        memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%alloc3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      scf.yield %alloc3: memref<16x16x16xf16, #hivm.address_space<ub>>
    } else {
      hivm.hir.vadd ins(%alloc5, %alloc10 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                        memref<16x16x16xf16, #hivm.address_space<ub>>)
          outs(%alloc7 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      scf.yield %alloc7 : memref<16x16x16xf16, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                   outs(%alloc8 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_plan_memory_for_result(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                         %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                         %arg2: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                         %arg3: memref<16x16x16xf16, #hivm.address_space<gm>>) ->
                                         memref<16x16x16xf16, #hivm.address_space<ub>> {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST2:.*]] = arith.constant 16384 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK:  hivm.hir.pointer_cast(%[[CONST0]])
    %0 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %2 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %3 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %4 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                      outs(%1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                      outs(%4 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%0, %1 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                      outs(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %5 = scf.for %arg4 = %c0 to %c1024 step %c128 iter_args(%arg5 = %4) ->
         (memref<16x16x16xf16, #hivm.address_space<ub>>) {
      hivm.hir.vadd ins(%0, %arg5 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                        memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      scf.yield %3 : memref<16x16x16xf16, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    return %5 : memref<16x16x16xf16, #hivm.address_space<ub>>
  }
}

// -----

module {
  func.func @test_plan_memory_subview(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                      %alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                      %alloc6 : memref<16x2x16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %alloc3 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc5 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc3 : memref<16x16x16xf16, #hivm.address_space<ub>>)

    hivm.hir.vadd ins(%alloc1, %alloc3: memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
          outs(%alloc5: memref<16x16x16xf16, #hivm.address_space<ub>>)
    %0 = memref.subview %alloc5[0, 0, 0] [16, 2, 16] [1, 1, 1] :
         memref<16x16x16xf16, #hivm.address_space<ub>> to
         memref<16x2x16xf16, strided<[256, 16, 1]>, #hivm.address_space<ub>>

    hivm.hir.store ins(%0: memref<16x2x16xf16, strided<[256, 16, 1]>, #hivm.address_space<ub>>)
                   outs(%alloc6: memref<16x2x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_plan_memory_collapse_shape(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %alloc6 : memref<256x16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %alloc3 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc5 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc3 : memref<16x16x16xf16, #hivm.address_space<ub>>)

    hivm.hir.vadd ins(%alloc1, %alloc3: memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%alloc5: memref<16x16x16xf16, #hivm.address_space<ub>>)
    %0 = memref.collapse_shape %alloc5 [[0, 1], [2]] :
         memref<16x16x16xf16, #hivm.address_space<ub>> into memref<256x16xf16, #hivm.address_space<ub>>
    hivm.hir.store ins(%0: memref<256x16xf16, #hivm.address_space<ub>>)
                   outs(%alloc6: memref<256x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_plan_memory_expand_shape(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                           %alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                           %alloc6 : memref<2x8x16x16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %alloc3 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc5 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%alloc3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%alloc1, %alloc3: memref<16x16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16x16xf16, #hivm.address_space<ub>>)
                  outs(%alloc5: memref<16x16x16xf16, #hivm.address_space<ub>>)

    %0 = memref.expand_shape %alloc5 [[0, 1], [2], [3]] output_shape [2, 8, 16, 16] :
         memref<16x16x16xf16, #hivm.address_space<ub>> into memref<2x8x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.store ins(%0: memref<2x8x16x16xf16, #hivm.address_space<ub>>)
                   outs(%alloc6: memref<2x8x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_plan_memory_temp_buffer(%arg0: memref<1x10xi16, #hivm.address_space<gm>>,
                                          %arg1: memref<8x10xi16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST0:.*]] = arith.constant 192 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 32 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %alloc = memref.alloc() : memref<1x10xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<1x10xi16, #hivm.address_space<gm>>)
                  outs(%alloc : memref<1x10xi16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %alloc_0 = memref.alloc() : memref<8x10xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc_1 = memref.alloc() : memref<80xi16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%alloc : memref<1x10xi16, #hivm.address_space<ub>>)
                  outs(%alloc_0 : memref<8x10xi16, #hivm.address_space<ub>>)
                  temp_buffer(%alloc_1 : memref<80xi16, #hivm.address_space<ub>>) broadcast_dims = [0]
    hivm.hir.store ins(%alloc_0 : memref<8x10xi16, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<8x10xi16, #hivm.address_space<gm>>)
    return
  }
}

// -----
#map = affine_map<()[s0] -> (s0 * 1572864)>
#map1 = affine_map<(d0) -> ((d0 floordiv 2048) mod 2)>
module {
  func.func @test_plan_memory_select(%arg0: memref<31457280xf32, #hivm.address_space<gm>>,
                                     %arg1: memref<31457280xf32, #hivm.address_space<gm>>,
                                     %arg2: memref<31457280xf32, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST5:.*]] = arith.constant 40960 : i64
    // CHECK: %[[CONST4:.*]] = arith.constant 32768 : i64
    // CHECK: %[[CONST3:.*]] = arith.constant 24576 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 16384 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1572864 = arith.constant 1572864 : index
    %c2048 = arith.constant 2048 : index
    %0 = hivm.hir.get_block_idx -> i64
    %1 = arith.index_cast %0 : i64 to index
    %2 = affine.apply #map()[%1]
    %subview = memref.subview %arg0[%2] [1572864] [1] : memref<31457280xf32, #hivm.address_space<gm>> to
               memref<1572864xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_0 = memref.subview %arg2[%2] [1572864] [1] : memref<31457280xf32, #hivm.address_space<gm>> to
                 memref<1572864xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_1 = memref.subview %arg1[%2] [1572864] [1] : memref<31457280xf32, #hivm.address_space<gm>> to
                 memref<1572864xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    // CHECK-NOT: memref.alloc()
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2048xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST5]])
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32, #hivm.address_space<ub>>
    scf.for %arg3 = %c0 to %c1572864 step %c2048 {
      %3 = affine.apply #map1(%arg3)
      %4 = arith.index_cast %3 : index to i1
      %5 = arith.select %4, %alloc_5, %alloc_6 : memref<2048xf32, #hivm.address_space<ub>>
      %6 = affine.apply #map1(%arg3)
      %7 = arith.index_cast %6 : index to i1
      %8 = arith.select %7, %alloc_3, %alloc_4 : memref<2048xf32, #hivm.address_space<ub>>
      %9 = affine.apply #map1(%arg3)
      %10 = arith.index_cast %9 : index to i1
      %11 = arith.select %10, %alloc, %alloc_2 : memref<2048xf32, #hivm.address_space<ub>>
      %subview_7 = memref.subview %subview[%arg3] [2048] [1] :
                   memref<1572864xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to
                   memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%subview_7 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
                    outs(%11 : memref<2048xf32, #hivm.address_space<ub>>)
      %subview_8 = memref.subview %subview_1[%arg3] [2048] [1] :
                   memref<1572864xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to
                   memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.load ins(%subview_8 :
                    memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
                    outs(%8 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%11, %8 : memref<2048xf32, #hivm.address_space<ub>>,
                    memref<2048xf32, #hivm.address_space<ub>>)
                    outs(%5 : memref<2048xf32, #hivm.address_space<ub>>)
      %subview_9 = memref.subview %subview_0[%arg3] [2048] [1] :
                   memref<1572864xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to
                   memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.store ins(%5 : memref<2048xf32, #hivm.address_space<ub>>)
                     outs(%subview_9 : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----
module {
  func.func @test_plan_memory_memref_view_and_reinterpret_cast(%arg0: memref<1x?xi16, #hivm.address_space<gm>>,
                                                                 %arg1: memref<?x1xi16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c1 : memref<1x?xi16, #hivm.address_space<gm>>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x1xi16, #hivm.address_space<gm>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc = memref.alloc() : memref<2048xi8, #hivm.address_space<ub>>
    %view = memref.view %alloc[%c0][%dim] : memref<2048xi8, #hivm.address_space<ub>> to
                                            memref<1x?xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<1x?xi16, #hivm.address_space<gm>>)
                  outs(%view : memref<1x?xi16, #hivm.address_space<ub>>)
    %reinterpret_cast = memref.reinterpret_cast %view to offset: [0], sizes: [%dim, 1], strides: [1, 1] :
                        memref<1x?xi16, #hivm.address_space<ub>> to
                        memref<?x1xi16, #hivm.address_space<ub>>
    hivm.hir.store ins(%reinterpret_cast : memref<?x1xi16, #hivm.address_space<ub>>)
                   outs(%arg1 : memref<?x1xi16, #hivm.address_space<gm>>)
      return
  }

}

// -----
module {
  func.func @test_mem_plan_pipe_opt(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                    %arg2 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                    %arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                    %arg4 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                    %arg5 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                    %arg6 : memref<16x16x16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST2:.*]] = arith.constant 16384 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %0 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %2 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%0 : memref<16x16x16xf16,#hivm.address_space<ub>>)
                   outs(%arg2 : memref<16x16x16xf16, #hivm.address_space<gm>>)

    hivm.hir.load ins(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%1 : memref<16x16x16xf16,#hivm.address_space<ub>>)
                   outs(%arg4: memref<16x16x16xf16,#hivm.address_space<gm>>)

    hivm.hir.load ins(%arg5 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                  outs(%2 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%2 : memref<16x16x16xf16,#hivm.address_space<ub>>)
                   outs(%arg6: memref<16x16x16xf16,#hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_mem_plan_db_two_address(%src_gm: memref<16xf16, #hivm.address_space<gm>>,
                                          %dst_gm: memref<16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 32 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %i0 = %c0 to %c16 step %c4 {
      // CHECK: hivm.hir.pointer_cast(%[[CONST0]], %[[CONST1]])
      %src_ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %src_ub {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%src_gm : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%src_ub : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%src_ub : memref<16xf16,#hivm.address_space<ub>>)
                     outs(%dst_gm: memref<16xf16,#hivm.address_space<gm>>)
    }
    return
  }
}

// -----
module {
  func.func @test_inplace_single_and_db(%src_gm: memref<16xf16, #hivm.address_space<gm>>,
                                        %dst_gm: memref<16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 32 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %i0 = %c0 to %c16 step %c4 {
      // CHECK: hivm.hir.pointer_cast(%[[CONST0]], %[[CONST1]])
      %src_ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %src_ub {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.pointer_cast(%[[CONST0]], %[[CONST1]])
      %dst_ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%src_gm : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%src_ub : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%src_ub, %src_ub : memref<16xf16, #hivm.address_space<ub>>,
                        memref<16xf16, #hivm.address_space<ub>>)
                    outs(%dst_ub : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%dst_ub : memref<16xf16,#hivm.address_space<ub>>)
                     outs(%dst_gm: memref<16xf16,#hivm.address_space<gm>>)
    }
    return
  }
}

// -----
module {
  func.func @test_mem_inplace_db_and_db(%src_gm: memref<16xf16, #hivm.address_space<gm>>,
                                        %dst_gm: memref<16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 32 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    scf.for %i0 = %c0 to %c16 step %c4 {
      // CHECK: hivm.hir.pointer_cast(%[[CONST0]], %[[CONST1]])
      %src_ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %src_ub {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      // CHECK: hivm.hir.pointer_cast(%[[CONST0]], %[[CONST1]])
      %dst_ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %dst_ub {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%src_gm : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%src_ub : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%src_ub, %src_ub : memref<16xf16, #hivm.address_space<ub>>,
                        memref<16xf16, #hivm.address_space<ub>>)
                    outs(%dst_ub : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%dst_ub : memref<16xf16,#hivm.address_space<ub>>)
                     outs(%dst_gm: memref<16xf16,#hivm.address_space<gm>>)
    }
    return
  }
}

// -----
module {
  func.func @test_infer_mem_allocate_loop_conflict(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>,
                                                   %alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %start = arith.constant 0 : index
    %end = arith.constant 1024 : index
    %step = arith.constant 128 : index
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc3 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    scf.for %iv = %start to %end step %step {
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%alloc2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
                     outs(%alloc1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc1, %alloc1 : memref<16x16x16xf16, #hivm.address_space<ub>>,
                        memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%alloc3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    }
    hivm.hir.store ins(%alloc3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
                   outs(%alloc4 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
memref.global @__constant_3xi64 : memref<3xi64>

func.func @test_plan_memory_memref_reshape(%arg0: memref<2x8xi16, #hivm.address_space<gm>>,
                                           %arg1: memref<2x2x4xi16, #hivm.address_space<gm>>) {
  %0 = memref.get_global @__constant_3xi64 : memref<3xi64>
  // CHECK-NOT: memref.alloc()
  // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
  // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
  %alloc = memref.alloc() : memref<2x8xi16, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0 : memref<2x8xi16, #hivm.address_space<gm>>)
                outs(%alloc : memref<2x8xi16, #hivm.address_space<ub>>)
  %reshape = memref.reshape %alloc(%0) : (memref<2x8xi16, #hivm.address_space<ub>>, memref<3xi64>) ->
                                         memref<2x2x4xi16, #hivm.address_space<ub>>
  hivm.hir.store ins(%reshape : memref<2x2x4xi16, #hivm.address_space<ub>>)
                 outs(%arg1 : memref<2x2x4xi16, #hivm.address_space<gm>>)
    return
}
}

// -----
module {
  func.func @test_mem_memref_load_store(%alloc2 : memref<1024xf16, #hivm.address_space<gm>>,
                                        %alloc4 : memref<1024xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 2048 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %start = arith.constant 0 : index
    %end = arith.constant 1024 : index
    %step = arith.constant 1 : index
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc1 = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %alloc3 = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%alloc2 : memref<1024xf16, #hivm.address_space<gm>>)
                      outs(%alloc1 : memref<1024xf16, #hivm.address_space<ub>>)
    scf.for %iv = %start to %end step %step {
      %0 = memref.load %alloc1[%iv] : memref<1024xf16, #hivm.address_space<ub>>
      %1 = memref.load %alloc1[%iv] : memref<1024xf16, #hivm.address_space<ub>>
      %2 = arith.addf %0, %1 : f16
      memref.store %2, %alloc3[%iv] : memref<1024xf16, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%alloc3 : memref<1024xf16, #hivm.address_space<ub>>)
                   outs(%alloc4 : memref<1024xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_mem_enough(%arg1 : memref<1xf32, #hivm.address_space<gm>>,
                             %arg2 : memref<1xf32, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST128:.*]] = arith.constant 128 : i64
    // CHECK: %[[CONST96:.*]] = arith.constant 96 : i64
    // CHECK: %[[CONST64:.*]] = arith.constant 64 : i64
    // CHECK: %[[CONST32:.*]] = arith.constant 32 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.44140625E-4 : f32
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %1 = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<1xf32, #hivm.address_space<gm>>)
                  outs(%1 : memref<1xf32, #hivm.address_space<ub>>)
    %4 = memref.load %1[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %5 = arith.mulf %4, %cst_1 : f32
    // CHECK: hivm.hir.pointer_cast(%[[CONST32]])
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
    memref.store %5, %alloc_9[%c0] : memref<1xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST64]])
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
    %6 = memref.load %alloc_9[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %7 = arith.addf %6, %cst_0 : f32
    memref.store %7, %alloc_10[%c0] : memref<1xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST96]])
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
    %8 = memref.load %alloc_10[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %9 = math.sqrt %8 : f32
    memref.store %9, %alloc_11[%c0] : memref<1xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST128]])
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1xf32, #hivm.address_space<ub>>
    memref.store %cst, %alloc_12[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %10 = memref.load %alloc_12[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %11 = memref.load %alloc_11[%c0] : memref<1xf32, #hivm.address_space<ub>>
    %12 = arith.divf %10, %11 : f32
    memref.store %12, %alloc_12[%c0] : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.store ins(%alloc_12 : memref<1xf32, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<1xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_mem_for_cube(%arg1 : memref<16xf32, #hivm.address_space<gm>>,
                               %arg2 : memref<16xf32, #hivm.address_space<gm>>,
                               %arg3 : memref<256xf32, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 64 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc_1 = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>)
                    outs(%alloc_1 : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %alloc_2 = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>)
                    outs(%alloc_2 : memref<16xf32, #hivm.address_space<cbuf>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %alloc3 = memref.alloc() : memref<256xf32, #hivm.address_space<cc>>
    hivm.hir.mmadL1 ins(%alloc_1, %alloc_2, %true, %c16, %c256, %c16 : memref<16xf32, #hivm.address_space<cbuf>>,
                        memref<16xf32, #hivm.address_space<cbuf>>, i1, index, index, index)
                        outs(%alloc3 : memref<256xf32, #hivm.address_space<cc>>)
    hivm.hir.fixpipe {enable_nz2nd} ins(%alloc3 : memref<256xf32, #hivm.address_space<cc>>)
                      outs(%arg3 : memref<256xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----
// expected-error@+1 {{ub overflow, requires 2560000 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)}}
func.func @test_one_mem_not_enough(%arg0_gm : memref<80000xf32, #hivm.address_space<gm>>,
                                   %vcast_gm :memref<80000xf16, #hivm.address_space<gm>>) {
  %arg0_ub = memref.alloc() : memref<80000xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg0_gm : memref<80000xf32, #hivm.address_space<gm>>)
                outs(%arg0_ub : memref<80000xf32, #hivm.address_space<ub>>)
  %vcast_res_ub = memref.alloc() : memref<80000xf16, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%arg0_ub : memref<80000xf32, #hivm.address_space<ub>>)
  outs(%vcast_res_ub : memref<80000xf16, #hivm.address_space<ub>>) round_mode = #hivm.round_mode<trunc>
  hivm.hir.store ins(%vcast_res_ub : memref<80000xf16, #hivm.address_space<ub>>)
                  outs(%vcast_gm : memref<80000xf16, #hivm.address_space<gm>>)
  return
}

// -----
module {
  // expected-error@below {{ub overflow, requires 2379264 bits while 1572864 bits available! (possible reason: tiling basic block is too large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer.)}}
  func.func @test_two_mem_not_enough(%arg1 : memref<37172xf32, #hivm.address_space<gm>>,
                                     %arg2 : memref<37172xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 4 : index
    %1 = memref.alloc() : memref<37172xf32, #hivm.address_space<ub>>
    %2 = memref.alloc() : memref<37172xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<37172xf32, #hivm.address_space<gm>>) outs(%1 : memref<37172xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<37172xf32, #hivm.address_space<gm>>) outs(%2 : memref<37172xf32, #hivm.address_space<ub>>)
    %3 = memref.load %1[%c0] : memref<37172xf32, #hivm.address_space<ub>>
    %4 = memref.load %2[%c1] : memref<37172xf32, #hivm.address_space<ub>>
    %5 = arith.mulf %3, %4 : f32
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<37172xf32, #hivm.address_space<ub>>
    memref.store %5, %alloc_9[%c0] : memref<37172xf32, #hivm.address_space<ub>>
    return
  }
}

// -----
module {
  func.func @test_if_else(%arg1 : memref<11520xi32, #hivm.address_space<gm>>,
                          %arg2 : memref<11520xi32, #hivm.address_space<gm>>,
                          %arg3 : memref<11520xf32, #hivm.address_space<gm>>,
                          %arg4: i8 {tt.divisibility = 16 : i32},
                          %arg5 : memref<11520xi64, #hivm.address_space<gm>>,
                          %arg6 : memref<11520xi64, #hivm.address_space<gm>>) {
    %cst_1 = arith.constant 4.6566126E-10 : f32
    %1 = arith.trunci %arg4 : i8 to i1
    // CHECK: %[[ARG1:.*]] = hivm.hir.pointer_cast(%[[CONST0:.*]]) : memref<11520xi32, #hivm.address_space<ub>>
    // CHECK: %[[ARG2:.*]] = hivm.hir.pointer_cast(%[[CONST1:.*]]) : memref<11520xi32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<11520xi32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<11520xi32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<11520xi32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<11520xi32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<11520xi32, #hivm.address_space<gm>>) outs(%alloc_1 : memref<11520xi32, #hivm.address_space<ub>>)
    // CHECK: %[[ARG3:.*]] = hivm.hir.pointer_cast(%[[CONST0]]) : memref<11520xi32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<11520xi32, #hivm.address_space<ub>>
    hivm.hir.vand ins(%alloc_0, %alloc_1 : memref<11520xi32, #hivm.address_space<ub>>, memref<11520xi32, #hivm.address_space<ub>>) outs(%alloc_2 : memref<11520xi32, #hivm.address_space<ub>>)
    // CHECK: scf.if %[[ARG0:.*]] {
    scf.if %1 {
      // CHECK: %[[ARG4:.*]] = hivm.hir.pointer_cast(%[[CONST2:.*]]) : memref<11520xi64, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() : memref<11520xi64, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_2 : memref<11520xi32, #hivm.address_space<ub>>) outs(%alloc_3 : memref<11520xi64, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc_3 : memref<11520xi64, #hivm.address_space<ub>>) outs(%arg5 : memref<11520xi64, #hivm.address_space<gm>>)
    // CHECK: } else {
    } else {
      // CHECK: %[[ARG4:.*]] = hivm.hir.pointer_cast(%[[CONST1]]) : memref<11520xf32, #hivm.address_space<ub>>
      // CHECK: %[[ARG5:.*]] = hivm.hir.pointer_cast(%[[CONST3:.*]]) : memref<11520xf32, #hivm.address_space<ub>>
      // CHECK: %[[ARG6:.*]] = hivm.hir.pointer_cast(%[[CONST1]]) : memref<11520xi64, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() : memref<11520xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg3 : memref<11520xf32, #hivm.address_space<gm>>) outs(%alloc_4 : memref<11520xf32, #hivm.address_space<ub>>)
      %alloc_5 = memref.alloc() : memref<11520xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%alloc_4, %cst_1 : memref<11520xf32, #hivm.address_space<ub>>, f32) outs(%alloc_5 : memref<11520xf32, #hivm.address_space<ub>>)
      %alloc_6 = memref.alloc() : memref<11520xi64, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%alloc_2 : memref<11520xi32, #hivm.address_space<ub>>) outs(%alloc_6 : memref<11520xi64, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc_6 : memref<11520xi64, #hivm.address_space<ub>>) outs(%arg6 : memref<11520xi64, #hivm.address_space<gm>>)
    }
    return
  }
}

// -----
module {
  func.func @test_ub_memory_without_load(%arg0: memref<5x1xf32, #hivm.address_space<gm>>) {
    // expected-error@+1 {{'memref.alloc' op error: read before first write}}
    %alloc = memref.alloc() : memref<5x2xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<5x1xf32, #hivm.address_space<ub>>
    hivm.hir.vdeinterleave ins(%alloc : memref<5x2xf32, #hivm.address_space<ub>>) outs(%alloc_1 : memref<5x1xf32, #hivm.address_space<ub>>) channel_num = 2 index_mode = <CHANNEL_0>
    hivm.hir.store ins(%alloc_1 : memref<5x1xf32, #hivm.address_space<ub>>) outs(%arg0 : memref<5x1xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----
module {
  func.func @test_two_ub_memory_without_load(%arg0: memref<5x1xf32, #hivm.address_space<gm>>, %arg1: memref<5x1xf32, #hivm.address_space<ub>>) {
    // expected-error@+1 {{'memref.alloc' op error: read before first write}}
    %alloc = memref.alloc() : memref<5x2xf32, #hivm.address_space<ub>>
    // expected-error@+1 {{'memref.alloc' op error: read before first write}}
    %alloc_1 = memref.alloc() : memref<5x1xf32, #hivm.address_space<ub>>
    hivm.hir.vdeinterleave ins(%alloc : memref<5x2xf32, #hivm.address_space<ub>>) outs(%arg1 : memref<5x1xf32, #hivm.address_space<ub>>) channel_num = 2 index_mode = <CHANNEL_0>
    hivm.hir.store ins(%alloc_1 : memref<5x1xf32, #hivm.address_space<ub>>) outs(%arg0 : memref<5x1xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vsub_inline_broadcast_inplace(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>,
                                                %arg2 : memref<64x1xf32, #hivm.address_space<gm>>,
                                                %arg3 : memref<64x128xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 33024 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 32768 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<64x1xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<64x1xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<64x1xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<64x1xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<512xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%alloc, %alloc_0 : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>)
      outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.store ins(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<64x128xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vsub_inline_brc_inplace_second_src(%arg1 : memref<64x1xf32, #hivm.address_space<gm>>,
                                                     %arg2 : memref<64x128xf32, #hivm.address_space<gm>>,
                                                     %arg3 : memref<64x128xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 33024 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 256 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x1xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x1xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<64x1xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64x1xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<64x128xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<64x128xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<512xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%alloc, %alloc_0 : memref<64x1xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>)
      outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.store ins(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<64x128xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vsub_inline_brc_inplace_dim0(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>,
                                               %arg2 : memref<1x128xf32, #hivm.address_space<gm>>,
                                               %arg3 : memref<64x128xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 33280 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 32768 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<1x128xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<1x128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<1x128xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<1x128xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<512xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%alloc, %alloc_0 : memref<64x128xf32, #hivm.address_space<ub>>, memref<1x128xf32, #hivm.address_space<ub>>)
      outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.store ins(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<64x128xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vadd_inline_broadcast_inplace(%arg1 : memref<128xf32, #hivm.address_space<gm>>,
                                                %arg2 : memref<1xf32, #hivm.address_space<gm>>,
                                                %arg3 : memref<128xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 544 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 512 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<1xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<128xf32, #hivm.address_space<gm>>) outs(%alloc : memref<128xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<1xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<1xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<128xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<128xf32, #hivm.address_space<ub>>, memref<1xf32, #hivm.address_space<ub>>)
      outs(%alloc_1 : memref<128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<128xf32, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.store ins(%alloc_1 : memref<128xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<128xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vmul_inline_broadcast_inplace(%arg1 : memref<4x64x128xf32, #hivm.address_space<gm>>,
                                                %arg2 : memref<4x1x128xf32, #hivm.address_space<gm>>,
                                                %arg3 : memref<4x64x128xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 133120 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 131072 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<4x64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<4x1x128xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<4x64x128xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<4x1x128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<4x64x128xf32, #hivm.address_space<gm>>) outs(%alloc : memref<4x64x128xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<4x1x128xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<4x1x128xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<4x64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<4096xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<4x64x128xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc, %alloc_0 : memref<4x64x128xf32, #hivm.address_space<ub>>, memref<4x1x128xf32, #hivm.address_space<ub>>)
      outs(%alloc_1 : memref<4x64x128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<4096xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.store ins(%alloc_1 : memref<4x64x128xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<4x64x128xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vmax_inline_broadcast_inplace(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>,
                                                %arg2 : memref<64x1xf32, #hivm.address_space<gm>>,
                                                %arg3 : memref<64x128xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 33024 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 32768 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<64x1xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<64x1xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<64x1xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<64x1xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<512xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.vmax ins(%alloc, %alloc_0 : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>)
      outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.store ins(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<64x128xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vdiv_inline_brc_inplace(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>,
                                          %arg2 : memref<64x1xf32, #hivm.address_space<gm>>,
                                          %arg3 : memref<64x128xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 33024 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 32768 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<64x1xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<64x1xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<64x1xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<64x1xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<512xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.vdiv ins(%alloc, %alloc_0 : memref<64x128xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>)
      outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.store ins(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<64x128xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vdiv_tempbuffer_vs_inplace(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>,
                                             %arg2 : memref<64x128xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST1:.*]] = arith.constant 32768 : i64
    // CHECK: %[[CST:.*]] = arith.constant 0.72134751 : f32
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %cst = arith.constant 0.72134751 : f32
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<64x128xf32, #hivm.address_space<gm>>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<64x128xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<16xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vdiv ins(%alloc, %cst : memref<64x128xf32, #hivm.address_space<ub>>, f32)
      outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>) outs(%arg2 : memref<64x128xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vdiv_tempbuffer_samesize_inplace_diff(%arg1 : memref<16xf32, #hivm.address_space<gm>>,
                                                        %arg2 : memref<16xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST1:.*]] = arith.constant 64 : i64
    // CHECK: %[[CST:.*]] = arith.constant 0.72134751 : f32
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %cst = arith.constant 0.72134751 : f32
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<16xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<16xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<16xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vdiv ins(%alloc, %cst : memref<16xf32, #hivm.address_space<ub>>, f32)
      outs(%alloc_1 : memref<16xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg2 : memref<16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vmul_inline_brc_bigger_not_inplace(%arg1 : memref<4x1x64xf32, #hivm.address_space<gm>>,
                                                     %arg2 : memref<4x1x64xf32, #hivm.address_space<gm>>,
                                                     %arg3 : memref<4x64x64xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST3:.*]] = arith.constant 67584 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 2048 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 1024 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<4x1x64xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<4x1x64xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<4x1x64xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<4x1x64xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<4x1x64xf32, #hivm.address_space<gm>>) outs(%alloc : memref<4x1x64xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<4x1x64xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<4x1x64xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<4x64x64xf32, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST3]]) : memref<4096xf32, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<4x64x64xf32, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%alloc, %alloc_0 : memref<4x1x64xf32, #hivm.address_space<ub>>, memref<4x1x64xf32, #hivm.address_space<ub>>)
      outs(%alloc_1 : memref<4x64x64xf32, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<4096xf32, #hivm.address_space<ub>>) broadcast = [1]
    hivm.hir.store ins(%alloc_1 : memref<4x64x64xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<4x64x64xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vmul_dyn_same_shape_inplace(%arg1 : memref<?xi16, #hivm.address_space<gm>>,
                                              %arg2 : memref<?xi16, #hivm.address_space<gm>>,
                                              %arg3 : memref<?xi16, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 1024 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 512 : i64
    // CHECK: %[[CST256:.*]] = arith.constant 256 : index
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<256xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<256xi16, #hivm.address_space<ub>>
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256xi16, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<256xi16, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0] [%c256] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<?xi16, #hivm.address_space<ub>>
    %subview_0 = memref.subview %alloc_0[0] [%c256] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<?xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<?xi16, #hivm.address_space<gm>>) outs(%subview : memref<?xi16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<?xi16, #hivm.address_space<gm>>) outs(%subview_0 : memref<?xi16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<256xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<256xi16, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<256xi16, #hivm.address_space<ub>>
    %subview_1 = memref.subview %alloc_1[0] [%c256] [1] : memref<256xi16, #hivm.address_space<ub>> to memref<?xi16, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<256xi16, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%subview, %subview_0 : memref<?xi16, #hivm.address_space<ub>>, memref<?xi16, #hivm.address_space<ub>>)
      outs(%subview_1 : memref<?xi16, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<256xi16, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.store ins(%subview_1 : memref<?xi16, #hivm.address_space<ub>>) outs(%arg3 : memref<?xi16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vmul_dyn_same_2d_shape_inplace(%arg1 : memref<?x?xi16, #hivm.address_space<gm>>,
                                                 %arg2 : memref<?x?xi16, #hivm.address_space<gm>>,
                                                 %arg3 : memref<?x?xi16, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 16384 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CST16:.*]] = arith.constant 16 : index
    // CHECK: %[[CST256:.*]] = arith.constant 256 : index
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<256x16xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<256x16xi16, #hivm.address_space<ub>>
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x16xi16, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<256x16xi16, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0] [%c256, %c16] [1, 1] : memref<256x16xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %alloc_0[0, 0] [%c256, %c16] [1, 1] : memref<256x16xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<?x?xi16, #hivm.address_space<gm>>) outs(%subview : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<?x?xi16, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<256x16xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<1024xi16, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<256x16xi16, #hivm.address_space<ub>>
    %subview_1 = memref.subview %alloc_1[0, 0] [%c256, %c16] [1, 1] : memref<256x16xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<1024xi16, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%subview, %subview_0 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>, memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>)
      outs(%subview_1 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<1024xi16, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.store ins(%subview_1 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%arg3 : memref<?x?xi16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vmul_dyn_same_2d_shape_inplace_one(%arg1 : memref<?x?xi16, #hivm.address_space<gm>>,
                                                     %arg2 : memref<?x?xi16, #hivm.address_space<gm>>,
                                                     %arg3 : memref<?x?xi16, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 8704 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 512 : i64
    // CHECK: %[[CST1:.*]] = arith.constant 1 : index
    // CHECK: %[[CST16:.*]] = arith.constant 16 : index
    // CHECK: %[[CST256:.*]] = arith.constant 256 : index
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<256x1xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<256x16xi16, #hivm.address_space<ub>>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x1xi16, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<256x16xi16, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0] [%c256, %c1] [1, 1] : memref<256x1xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %alloc_0[0, 0] [%c256, %c16] [1, 1] : memref<256x16xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<?x?xi16, #hivm.address_space<gm>>) outs(%subview : memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<?x?xi16, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<256x16xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<1024xi16, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<256x16xi16, #hivm.address_space<ub>>
    %subview_1 = memref.subview %alloc_1[0, 0] [%c256, %c16] [1, 1] : memref<256x16xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<1024xi16, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%subview, %subview_0 : memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>, memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>)
      outs(%subview_1 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<1024xi16, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.store ins(%subview_1 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%arg3 : memref<?x?xi16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vmul_dyn_diff_2d_shape_not_inplace(%arg1 : memref<?x?xi16, #hivm.address_space<gm>>,
                                                     %arg2 : memref<?x?xi16, #hivm.address_space<gm>>,
                                                     %arg3 : memref<?x?xi16, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST3:.*]] = arith.constant 9216 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 1024 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 512 : i64
    // CHECK: %[[CST1:.*]] = arith.constant 1 : index
    // CHECK: %[[CST16:.*]] = arith.constant 16 : index
    // CHECK: %[[CST256:.*]] = arith.constant 256 : index
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<256x1xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<256x1xi16, #hivm.address_space<ub>>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<256x1xi16, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<256x1xi16, #hivm.address_space<ub>>
    %subview = memref.subview %alloc[0, 0] [%c256, %c1] [1, 1] : memref<256x1xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %alloc_0[0, 0] [%c256, %c1] [1, 1] : memref<256x1xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<?x?xi16, #hivm.address_space<gm>>) outs(%subview : memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<?x?xi16, #hivm.address_space<gm>>) outs(%subview_0 : memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<256x16xi16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST3]]) : memref<1024xi16, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<256x16xi16, #hivm.address_space<ub>>
    %subview_1 = memref.subview %alloc_1[0, 0] [%c256, %c16] [1, 1] : memref<256x16xi16, #hivm.address_space<ub>> to memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<1024xi16, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%subview, %subview_0 : memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>, memref<?x?xi16, strided<[1, 1]>, #hivm.address_space<ub>>)
      outs(%subview_1 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<1024xi16, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.store ins(%subview_1 : memref<?x?xi16, strided<[16, 1]>, #hivm.address_space<ub>>) outs(%arg3 : memref<?x?xi16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vmul_one_dyn_same_2d_shape_inplace_one(%arg1 : memref<4x?xi16, #hivm.address_space<gm>>,
                                                         %arg2 : memref<?x64xi16, #hivm.address_space<gm>>,
                                                         %arg3 : memref<4x?xi16, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST2:.*]] = arith.constant 1024 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 512 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: %[[CST4:.*]] = arith.constant 4 : index
    // CHECK: %[[CST1:.*]] = arith.constant 1 : index
    // CHECK: %[[CST0:.*]] = arith.constant 0 : index
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<512xi8, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<512xi8, #hivm.address_space<ub>>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %dim = memref.dim %arg1, %c1 : memref<4x?xi16, #hivm.address_space<gm>>
    %dim_0 = memref.dim %arg2, %c0 : memref<?x64xi16, #hivm.address_space<gm>>
    %alloc = memref.alloc() : memref<512xi8, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<512xi8, #hivm.address_space<ub>>
    %view = memref.view %alloc[%c0][%dim] : memref<512xi8, #hivm.address_space<ub>> to memref<4x?xi16, #hivm.address_space<ub>>
    %view_0 = memref.view %alloc_0[%c0][%dim_0] : memref<512xi8, #hivm.address_space<ub>> to memref<?x64xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<4x?xi16, #hivm.address_space<gm>>) outs(%view : memref<4x?xi16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg2 : memref<?x64xi16, #hivm.address_space<gm>>) outs(%view_0 : memref<?x64xi16, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<512xi8, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<256xi16, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<512xi8, #hivm.address_space<ub>>
    %dim_1 = memref.dim %arg3, %c0 : memref<4x?xi16, #hivm.address_space<gm>>
    %view_1 = memref.view %alloc_1[%c0][%dim_1] : memref<512xi8, #hivm.address_space<ub>> to memref<4x?xi16, #hivm.address_space<ub>>
    %alloc_2 = memref.alloc() : memref<256xi16, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%view, %view_0 : memref<4x?xi16, #hivm.address_space<ub>>, memref<?x64xi16, #hivm.address_space<ub>>)
      outs(%view_1 : memref<4x?xi16, #hivm.address_space<ub>>) temp_buffer(%alloc_2 : memref<256xi16, #hivm.address_space<ub>>) broadcast = [0]
    hivm.hir.store ins(%view_1 : memref<4x?xi16, #hivm.address_space<ub>>) outs(%arg3 : memref<4x?xi16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_copy_iter_arg_before_yield(%arg0: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %arg1: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %arg2: memref<16x16x16xf16, #hivm.address_space<gm>>,
                                             %arg3: memref<16x16x16xf16, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST4:.*]] = arith.constant 32768 : i64
    // CHECK: %[[CONST3:.*]] = arith.constant 24576 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 16384 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 8192 : i64
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CST0:.*]] = arith.constant 0 : index
    // CHECK: %[[CST1024:.*]] = arith.constant 1024 : index
    // CHECK: %[[CST128:.*]] = arith.constant 128 : index
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %0 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %2 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16x16x16xf16, #hivm.address_space<gm>>) outs(%0 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<16x16x16xf16, #hivm.address_space<gm>>) outs(%1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %4 = scf.for %arg4 = %c0 to %c1024 step %c128 iter_args(%arg5 = %2) -> (memref<16x16x16xf16, #hivm.address_space<ub>>) {
      // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
      %3 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
      scf.if %true {
        hivm.hir.vadd ins(%0, %arg5 : memref<16x16x16xf16, #hivm.address_space<ub>>, memref<16x16x16xf16, #hivm.address_space<ub>>)
          outs(%3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      }
      hivm.hir.vadd ins(%1, %arg5 : memref<16x16x16xf16, #hivm.address_space<ub>>, memref<16x16x16xf16, #hivm.address_space<ub>>)
                    outs(%3 : memref<16x16x16xf16, #hivm.address_space<ub>>)
      // hivm.hir.copy
      scf.yield %3 : memref<16x16x16xf16, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
    %5 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg3 : memref<16x16x16xf16, #hivm.address_space<gm>>) outs(%5 : memref<16x16x16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%5 : memref<16x16x16xf16, #hivm.address_space<ub>>) outs(%arg2 : memref<16x16x16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_load_arg_after_yield(%arg0: memref<256xi64, #hivm.address_space<gm>>,
                                       %arg1: memref<256xi64, #hivm.address_space<gm>>,
                                       %arg2: memref<256xi64, #hivm.address_space<gm>>,
                                       %arg3: memref<256xi64, #hivm.address_space<gm>>,
                                       %arg4: memref<?xi64, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST5:.*]] = arith.constant 8224 : i64
    // CHECK: %[[CONST4:.*]] = arith.constant 8192 : i64
    // CHECK: %[[CONST3:.*]] = arith.constant 6144 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 4096 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 2048 : i64
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CST0:.*]] = arith.constant 0 : index
    // CHECK: %[[CST1:.*]] = arith.constant 1 : i64
    // CHECK: %[[CST1024:.*]] = arith.constant 1024 : index
    // CHECK: %[[CST128:.*]] = arith.constant 128 : index
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %0 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %1 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %2 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<256xi64, #hivm.address_space<gm>>) outs(%0 : memref<256xi64, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<256xi64, #hivm.address_space<gm>>) outs(%1 : memref<256xi64, #hivm.address_space<ub>>)
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %4 = scf.for %arg5 = %c0 to %c1024 step %c128 iter_args(%arg6 = %2) -> (memref<256xi64, #hivm.address_space<ub>>) {
      // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
      %3 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
      scf.if %true {
        hivm.hir.vadd ins(%0, %arg6 : memref<256xi64, #hivm.address_space<ub>>, memref<256xi64, #hivm.address_space<ub>>)
          outs(%3 : memref<256xi64, #hivm.address_space<ub>>)
      }
      %4 = memref.load %arg6[%arg5] : memref<256xi64, #hivm.address_space<ub>>
      %5 = arith.subi %4, %c1_i64 : i64
      %6 = arith.index_cast %5 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%6], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      scf.if %true {
        // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
        %7 = memref.alloc() {alignment = 64 : i64} : memref<1xi64, #hivm.address_space<ub>>
        memref.store %4, %7[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.store ins(%7 : memref<1xi64, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      }
      // hivm.hir.copy
      scf.yield %3 : memref<256xi64, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.pointer_cast(%[[CONST5]])
    %5 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg3 : memref<256xi64, #hivm.address_space<gm>>) outs(%5 : memref<256xi64, #hivm.address_space<ub>>)
    hivm.hir.store ins(%5 : memref<256xi64, #hivm.address_space<ub>>) outs(%arg2 : memref<256xi64, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_load_yield_before_use_arg(%arg0: memref<256xi64, #hivm.address_space<gm>>,
                                            %arg1: memref<256xi64, #hivm.address_space<gm>>,
                                            %arg2: memref<256xi64, #hivm.address_space<gm>>,
                                            %arg3: memref<256xi64, #hivm.address_space<gm>>,
                                            %arg4: memref<?xi64, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST5:.*]] = arith.constant 4192 : i64
    // CHECK: %[[CONST4:.*]] = arith.constant 4160 : i64
    // CHECK: %[[CONST3:.*]] = arith.constant 4128 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 4096 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 2048 : i64
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CST0:.*]] = arith.constant 0 : index
    // CHECK: %[[CST1:.*]] = arith.constant 1 : i64
    // CHECK: %[[CST1024:.*]] = arith.constant 1024 : index
    // CHECK: %[[CST128:.*]] = arith.constant 128 : index
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %0 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %1 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %2 = memref.alloc() : memref<1xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<256xi64, #hivm.address_space<gm>>) outs(%0 : memref<256xi64, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<256xi64, #hivm.address_space<gm>>) outs(%1 : memref<256xi64, #hivm.address_space<ub>>)
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %3 = scf.for %arg5 = %c0 to %c1024 step %c128 iter_args(%arg6 = %2) -> (memref<1xi64, #hivm.address_space<ub>>) {
      // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
      %5 = memref.alloc() : memref<1xi64, #hivm.address_space<ub>>
      %6 = memref.load %0[%arg5] : memref<256xi64, #hivm.address_space<ub>>
      %7 = arith.addi %6, %c1_i64 : i64
      memref.store %7, %5[%c0] : memref<1xi64, #hivm.address_space<ub>>
      %12 = memref.load %5[%c0] : memref<1xi64, #hivm.address_space<ub>>
      %8 = memref.load %arg6[%c0] : memref<1xi64, #hivm.address_space<ub>>
      %9 = arith.subi %8, %12 : i64
      %10 = arith.index_cast %9 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%10], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      scf.if %true {
        // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
        %11 = memref.alloc() {alignment = 64 : i64} : memref<1xi64, #hivm.address_space<ub>>
        memref.store %8, %11[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.store ins(%11 : memref<1xi64, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      }
      // hivm.hir.copy
      scf.yield %5 : memref<1xi64, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.pointer_cast(%[[CONST5]])
    %4 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg3 : memref<256xi64, #hivm.address_space<gm>>) outs(%4 : memref<256xi64, #hivm.address_space<ub>>)
    hivm.hir.store ins(%4 : memref<256xi64, #hivm.address_space<ub>>) outs(%arg2 : memref<256xi64, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_store_yield_before_use_arg(%arg0: memref<256xi64, #hivm.address_space<gm>>,
                                             %arg1: memref<256xi64, #hivm.address_space<gm>>,
                                             %arg2: memref<256xi64, #hivm.address_space<gm>>,
                                             %arg3: memref<256xi64, #hivm.address_space<gm>>,
                                             %arg4: memref<?xi64, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST5:.*]] = arith.constant 4192 : i64
    // CHECK: %[[CONST4:.*]] = arith.constant 4160 : i64
    // CHECK: %[[CONST3:.*]] = arith.constant 4128 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 4096 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 2048 : i64
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CST0:.*]] = arith.constant 0 : index
    // CHECK: %[[CST1:.*]] = arith.constant 1 : i64
    // CHECK: %[[CST1024:.*]] = arith.constant 1024 : index
    // CHECK: %[[CST128:.*]] = arith.constant 128 : index
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
    %0 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
    %1 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
    %2 = memref.alloc() : memref<1xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<256xi64, #hivm.address_space<gm>>) outs(%0 : memref<256xi64, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<256xi64, #hivm.address_space<gm>>) outs(%1 : memref<256xi64, #hivm.address_space<ub>>)
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %3 = scf.for %arg5 = %c0 to %c1024 step %c128 iter_args(%arg6 = %2) -> (memref<1xi64, #hivm.address_space<ub>>) {
      // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
      %5 = memref.alloc() : memref<1xi64, #hivm.address_space<ub>>
      %6 = memref.load %0[%arg5] : memref<256xi64, #hivm.address_space<ub>>
      %7 = arith.addi %6, %c1_i64 : i64
      memref.store %7, %5[%c0] : memref<1xi64, #hivm.address_space<ub>>
      %8 = memref.load %arg6[%c0] : memref<1xi64, #hivm.address_space<ub>>
      %9 = arith.subi %8, %c1_i64 : i64
      %10 = arith.index_cast %9 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%10], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
      scf.if %true {
        // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
        %11 = memref.alloc() {alignment = 64 : i64} : memref<1xi64, #hivm.address_space<ub>>
        memref.store %8, %11[%c0] : memref<1xi64, #hivm.address_space<ub>>
        hivm.hir.store ins(%11 : memref<1xi64, #hivm.address_space<ub>>) outs(%reinterpret_cast : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      }
      // hivm.hir.copy
      scf.yield %5 : memref<1xi64, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.pointer_cast(%[[CONST5]])
    %4 = memref.alloc() : memref<256xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg3 : memref<256xi64, #hivm.address_space<gm>>) outs(%4 : memref<256xi64, #hivm.address_space<ub>>)
    hivm.hir.store ins(%4 : memref<256xi64, #hivm.address_space<ub>>) outs(%arg2 : memref<256xi64, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_multi_if_yield_not_alias(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg1: i1,
                                           %arg2: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg3: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg4: i1,
                                           %arg5: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg6: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg7: memref<16xf32, #hivm.address_space<gm>>,
                                           %arg8: memref<16xf32, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST6:.*]] = arith.constant 224 : i64
    // CHECK: %[[CONST5:.*]] = arith.constant 160 : i64
    // CHECK: %[[CONST4:.*]] = arith.constant 128 : i64
    // CHECK: %[[CONST3:.*]] = arith.constant 96 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 64 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 32 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK:  hivm.hir.pointer_cast(%[[CONST0]])
    %alloc = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16xf16, #hivm.address_space<ub>>)
    %0 = scf.if %arg1 -> (memref<16xf16, #hivm.address_space<ub>>) {
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc_0 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg2 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
      %alloc_1 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg3 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc_2 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%alloc_0, %alloc_1 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_2 : memref<16xf16, #hivm.address_space<ub>>)
      scf.yield %alloc_2 : memref<16xf16, #hivm.address_space<ub>>
    } else {
      scf.yield %alloc : memref<16xf16, #hivm.address_space<ub>>
    }
    %1 = scf.if %arg4 -> (memref<16xf16, #hivm.address_space<ub>>) {
      // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
      %alloc_3 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg5 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_3 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
      %alloc_4 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg6 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_4 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
      %alloc_5 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vsub ins(%alloc_3, %alloc_4 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_5 : memref<16xf16, #hivm.address_space<ub>>)
      scf.yield %alloc_5 : memref<16xf16, #hivm.address_space<ub>>
    } else {
      scf.yield %alloc : memref<16xf16, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.pointer_cast(%[[CONST5]])
    %alloc_6 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%0 : memref<16xf16, #hivm.address_space<ub>>) outs(%alloc_6 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_6 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg7 : memref<16xf32, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST6]])
    %alloc_7 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%1 : memref<16xf16, #hivm.address_space<ub>>) outs(%alloc_7 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_7 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg8 : memref<16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_if_multi_yield_not_alias(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg1: i1,
                                           %arg2: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg3: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg4: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg5: memref<16xf16, #hivm.address_space<gm>>,
                                           %arg6: memref<16xf32, #hivm.address_space<gm>>,
                                           %arg7: memref<16xf32, #hivm.address_space<gm>>,
                                           %arg8: memref<16xf32, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST8:.*]] = arith.constant 320 : i64
    // CHECK: %[[CONST7:.*]] = arith.constant 256 : i64
    // CHECK: %[[CONST6:.*]] = arith.constant 192 : i64
    // CHECK: %[[CONST5:.*]] = arith.constant 160 : i64
    // CHECK: %[[CONST4:.*]] = arith.constant 128 : i64
    // CHECK: %[[CONST3:.*]] = arith.constant 96 : i64
    // CHECK: %[[CONST2:.*]] = arith.constant 64 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 32 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK:  hivm.hir.pointer_cast(%[[CONST0]])
    %alloc = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16xf16, #hivm.address_space<ub>>)
    %0:3 = scf.if %arg1 -> (memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) {
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc_0 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg2 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
      %alloc_1 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg3 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST3]])
      %alloc_2 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%alloc_0, %alloc_1 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_2 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc_3 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vsub ins(%alloc_0, %alloc_1 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_3 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
      %alloc_4 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg4 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_4 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST5]])
      %alloc_5 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg5 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_5 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST4]])
      %alloc_6 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vdiv ins(%alloc_4, %alloc_5 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_6 : memref<16xf16, #hivm.address_space<ub>>)
      scf.yield %alloc_2, %alloc_3, %alloc_6 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>
    } else {
      scf.yield %alloc, %alloc, %alloc : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>
    }
    // CHECK: hivm.hir.pointer_cast(%[[CONST6]])
    %alloc_7 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%0#0 : memref<16xf16, #hivm.address_space<ub>>) outs(%alloc_7 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_7 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg6 : memref<16xf32, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST7]])
    %alloc_8 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%0#1 : memref<16xf16, #hivm.address_space<ub>>) outs(%alloc_8 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_8 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg7 : memref<16xf32, #hivm.address_space<gm>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST8]])
    %alloc_9 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%0#2 : memref<16xf16, #hivm.address_space<ub>>) outs(%alloc_9 : memref<16xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_9 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg8 : memref<16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_scfwhileop(%arg0 : memref<16xf32, #hivm.address_space<gm>>,
                             %arg1 : memref<16xf32, #hivm.address_space<ub>>,
                             %arg2 : memref<16xf32, #hivm.address_space<gm>>) {
    // CHECK: %[[CONST1:.*]] = arith.constant 64 : i64
    // CHECK: %[[CST0:.*]] = arith.constant 0 : i32
    // CHECK: %[[CST1:.*]] = arith.constant 1 : i32
    // CHECK: %[[CST100:.*]] = arith.constant 100 : i32
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    // CHECK: hivm.hir.pointer_cast(%[[CONST0]]) : memref<16xf32, #hivm.address_space<ub>>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c100 = arith.constant 100 : i32
    %alloc_1 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    %0 = scf.while (%arg3 = %c0) : (i32) -> i32 {
      %1 = arith.cmpi eq, %arg3, %c100 : i32
      scf.condition(%1) %arg3 : i32
    } do {
    ^bb0(%arg3: i32):
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<16xf32, #hivm.address_space<ub>>
      %alloc_0 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xf32, #hivm.address_space<ub>>)
      hivm.hir.vadd ins(%alloc_0, %arg1 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16xf32, #hivm.address_space<ub>>)
      %2 = arith.addi %arg3, %c1 : i32
      scf.yield %2 : i32
    }
    hivm.hir.store ins(%alloc_1 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg2 : memref<16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_scfwhileop_yield_inplace(%arg0 : memref<16xf32, #hivm.address_space<gm>>,
                                           %arg1 : memref<16xf32, #hivm.address_space<gm>>,
                                           %arg2 : memref<16xf32, #hivm.address_space<gm>>,
                                           %arg3 : memref<16xf32, #hivm.address_space<gm>>,
                                           %arg4 : i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c100 = arith.constant 100 : i32
    // CHECK: hivm.hir.pointer_cast(%[[CONST2:.*]]) : memref<16xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
    // CHECK: hivm.hir.pointer_cast(%[[CONST0:.*]]) : memref<16xf32, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xf32, #hivm.address_space<ub>>)
    %0 = scf.while (%arg5 = %alloc) : (memref<16xf32, #hivm.address_space<ub>>) -> memref<16xf32, #hivm.address_space<ub>> {
      %1 = arith.cmpi eq, %arg4, %c100 : i32
      scf.condition(%1) %alloc_0 : memref<16xf32, #hivm.address_space<ub>>
    } do {
    ^bb0(%arg6: memref<16xf32, #hivm.address_space<ub>>):
      // CHECK: hivm.hir.pointer_cast(%[[CONST1:.*]]) : memref<16xf32, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg2 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16xf32, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<16xf32, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg6, %alloc_1 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_2 : memref<16xf32, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<16xf32, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg6, %alloc_2 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_3 : memref<16xf32, #hivm.address_space<ub>>)
      scf.yield %alloc_3 : memref<16xf32, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%0#0 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_dowhile_yield_inplace(%arg0 : memref<16xf32, #hivm.address_space<gm>>,
                                        %arg1 : memref<16xf32, #hivm.address_space<gm>>,
                                        %arg2 : memref<16xf32, #hivm.address_space<gm>>,
                                        %arg3 : memref<16xf32, #hivm.address_space<gm>>,
                                        %arg4 : i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c100 = arith.constant 100 : i32
    // CHECK: hivm.hir.pointer_cast(%[[CONST2:.*]]) : memref<16xf32, #hivm.address_space<ub>>
    %alloc = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc : memref<16xf32, #hivm.address_space<ub>>)
    %0 = scf.while (%arg5 = %alloc) : (memref<16xf32, #hivm.address_space<ub>>) -> memref<16xf32, #hivm.address_space<ub>> {
      // CHECK: hivm.hir.pointer_cast(%[[CONST0:.*]]) : memref<16xf32, #hivm.address_space<ub>>
      %alloc_0 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xf32, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST1:.*]]) : memref<16xf32, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg2 : memref<16xf32, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16xf32, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]]) : memref<16xf32, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg5, %alloc_1 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_2 : memref<16xf32, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST2]]) : memref<16xf32, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%alloc_0, %alloc_2 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>) outs(%alloc_3 : memref<16xf32, #hivm.address_space<ub>>)
      %1 = arith.cmpi eq, %arg4, %c100 : i32
      scf.condition(%1) %alloc_3 : memref<16xf32, #hivm.address_space<ub>>
    } do {
    ^bb0(%arg6: memref<16xf32, #hivm.address_space<ub>>):
      scf.yield %arg6 : memref<16xf32, #hivm.address_space<ub>>
    }
    hivm.hir.store ins(%0#0 : memref<16xf32, #hivm.address_space<ub>>) outs(%arg3 : memref<16xf32, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_branchop(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                           %arg1: memref<16xf16, #hivm.address_space<gm>>,
                           %arg2: memref<16xf16, #hivm.address_space<gm>>,
                           %arg3: i1) {
    // CHECK-NOT: memref.alloc()
    // CHECK:  hivm.hir.pointer_cast(%[[CONST0:.*]])
    %alloc = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16xf16, #hivm.address_space<ub>>)
    cf.cond_br %arg3, ^bb1(%alloc : memref<16xf16, #hivm.address_space<ub>>), ^bb2(%alloc : memref<16xf16, #hivm.address_space<ub>>)
    ^bb1(%arg10 : memref<16xf16, #hivm.address_space<ub>>):
      // CHECK: hivm.hir.pointer_cast(%[[CONST1:.*]])
      %alloc_0 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc_1 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg10, %alloc_0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_1 : memref<16xf16, #hivm.address_space<ub>>)
      cf.br ^bb3(%alloc_1 : memref<16xf16, #hivm.address_space<ub>>)
    ^bb2(%arg11 : memref<16xf16, #hivm.address_space<ub>>):
      // CHECK: hivm.hir.pointer_cast(%[[CONST2:.*]])
      %alloc_2 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_2 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST0]])
      %alloc_3 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vsub ins(%arg11, %alloc_2 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_3 : memref<16xf16, #hivm.address_space<ub>>)
      cf.br ^bb3(%alloc_3 : memref<16xf16, #hivm.address_space<ub>>)
    ^bb3(%arg12 : memref<16xf16, #hivm.address_space<ub>>):
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc_4 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg12, %arg12 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
                  outs(%alloc_4 : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_4 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_branchop_inplace(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                                   %arg1: memref<16xf16, #hivm.address_space<gm>>,
                                   %arg2: memref<16xf16, #hivm.address_space<gm>>,
                                   %arg3: i1) {
    // CHECK-NOT: memref.alloc()
    // CHECK:  hivm.hir.pointer_cast(%[[CONST0:.*]])
    %alloc = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc : memref<16xf16, #hivm.address_space<ub>>)
    cf.cond_br %arg3, ^bb1(%alloc : memref<16xf16, #hivm.address_space<ub>>), ^bb2(%alloc : memref<16xf16, #hivm.address_space<ub>>)
    ^bb1(%arg10 : memref<16xf16, #hivm.address_space<ub>>):
      // CHECK: hivm.hir.pointer_cast(%[[CONST1:.*]])
      %alloc_0 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc_1 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg10, %alloc_0 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_1 : memref<16xf16, #hivm.address_space<ub>>)
      cf.br ^bb3(%alloc_1, %arg10 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
    ^bb2(%arg11 : memref<16xf16, #hivm.address_space<ub>>):
      // CHECK: hivm.hir.pointer_cast(%[[CONST2:.*]])
      %alloc_2 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>) outs(%alloc_2 : memref<16xf16, #hivm.address_space<ub>>)
      // CHECK: hivm.hir.pointer_cast(%[[CONST2]])
      %alloc_3 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vsub ins(%arg11, %alloc_2 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
        outs(%alloc_3 : memref<16xf16, #hivm.address_space<ub>>)
      cf.br ^bb3(%alloc_3, %arg11 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
    ^bb3(%arg12 : memref<16xf16, #hivm.address_space<ub>>, %arg13 : memref<16xf16, #hivm.address_space<ub>>):
      // CHECK: hivm.hir.pointer_cast(%[[CONST1]])
      %alloc_4 = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%arg12, %arg13 : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>)
                  outs(%alloc_4 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%alloc_4 : memref<16xf16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
      return
  }
}

// -----

module {
  func.func @test_vsel_i64_reuse(%arg0: memref<16xi64, #hivm.address_space<gm>>,
                                   %arg1: memref<16xi64, #hivm.address_space<gm>>,
                                   %arg2: memref<16xi64, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST2:.*]] = arith.constant 256 : i64
    // CHECK: %[[CONST1:.*]] = arith.constant 128 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c0_i64 = arith.constant 0 : i64
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %alloc = memref.alloc() : memref<16xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xi64, #hivm.address_space<gm>>) outs(%alloc : memref<16xi64, #hivm.address_space<ub>>)
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST1]])
    %alloc_1 = memref.alloc() : memref<16xi64, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xi64, #hivm.address_space<gm>>) outs(%alloc_1 : memref<16xi64, #hivm.address_space<ub>>)
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST2]])
    %alloc_2 = memref.alloc() : memref<16xi1, #hivm.address_space<ub>>
    hivm.hir.vcmp ins(%alloc, %c0_i64 : memref<16xi64, #hivm.address_space<ub>>, i64) outs(%alloc_2 : memref<16xi1, #hivm.address_space<ub>>)
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %alloc_3 = memref.alloc() : memref<16xi64, #hivm.address_space<ub>>
    hivm.hir.vsel ins(%alloc_2, %alloc, %alloc_1 : memref<16xi1, #hivm.address_space<ub>>, memref<16xi64, #hivm.address_space<ub>>, memref<16xi64, #hivm.address_space<ub>>) outs(%alloc_3 : memref<16xi64, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_3 : memref<16xi64, #hivm.address_space<ub>>) outs(%arg2 : memref<16xi64, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vsel_i64_no_reuse(%arg0: memref<16xi64, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST1:.*]] = arith.constant 256 : i64
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %alloc = memref.alloc() : memref<2048xi1, #hivm.address_space<ub>>
    linalg.fill ins(%c1_i64: i64) outs(%alloc : memref<2048xi1, #hivm.address_space<ub>>)
    %subview = memref.subview %alloc[0] [16] [1] : memref<2048xi1, #hivm.address_space<ub>> to memref<16xi1, #hivm.address_space<ub>>
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST1]])
    %alloc_1 = memref.alloc() : memref<16xi64, #hivm.address_space<ub>>
    hivm.hir.vsel ins(%subview, %c1_i64, %c2_i64 : memref<16xi1, #hivm.address_space<ub>>, i64, i64) outs(%alloc_1 : memref<16xi64, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xi64, #hivm.address_space<ub>>) outs(%arg0 : memref<16xi64, #hivm.address_space<gm>>)
    return
  }
}

// -----

module {
  func.func @test_vsel_i32_reuse_cond(%arg0: memref<16xi32, #hivm.address_space<gm>>) {
    // CHECK-NOT: memref.alloc()
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %alloc = memref.alloc() : memref<2048xi1, #hivm.address_space<ub>>
    linalg.fill ins(%c1_i32: i32) outs(%alloc : memref<2048xi1, #hivm.address_space<ub>>)
    %subview = memref.subview %alloc[0] [16] [1] : memref<2048xi1, #hivm.address_space<ub>> to memref<16xi1, #hivm.address_space<ub>>
    // CHECK: {{.*}} = hivm.hir.pointer_cast(%[[CONST0]])
    %alloc_1 = memref.alloc() : memref<16xi32, #hivm.address_space<ub>>
    hivm.hir.vsel ins(%subview, %c1_i32, %c2_i32 : memref<16xi1, #hivm.address_space<ub>>, i32, i32) outs(%alloc_1 : memref<16xi32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xi32, #hivm.address_space<ub>>) outs(%arg0 : memref<16xi32, #hivm.address_space<gm>>)
    return
  }
}