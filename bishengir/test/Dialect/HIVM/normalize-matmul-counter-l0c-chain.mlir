// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend910B4 -hivm-normalize-matmul %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s
// RUN: bishengir-opt -hacc-append-device-spec=target=Ascend950PR_9579 -hivm-normalize-matmul %s -split-input-file -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

// Test counterPrevious chain across mixed CCF types:
// CCF1: for+if, variable bounds (mayNotExec=true) → NoBias, addTailFallback, counterPrevious=true
// CCF2: for, variable bounds (mayNotExec=true) → ReuseL0C from CCF1's IfOp, counterPrevious=runtime → andi
// CCF3: bare mmad (no for) → ReuseL0C from CCF2's IfOp, no counter/addTailFallback
// CCF4: for, constant bounds (mayNotExec=false) → ReuseL0C from CCF3's MmadL1Op, counterPrevious=false (folded)

// CHECK-LABEL: func.func @test_counter_previous_mixed_ccf
module {
  func.func @test_counter_previous_mixed_ccf(%lb: i32, %ub: i32) -> tensor<64x32xf32> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<32x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf32>
    %empty = tensor.empty() : tensor<64x32xf32>

    // CCF1: for+if with variable bounds → addTailFallback, counterPrevious=true
    // CHECK: %[[ALLOCA1:.*]] = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: scf.if {{.*}} -> (tensor<64x32xf32>)
    // CCF1 mmad inside if: initCondition = firstIter (counter == 0)
    // CHECK: %[[FIRST1:.*]] = arith.cmpi eq, {{.*}}, %c0_i32 : i32
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[FIRST1]], %c64, %c32, %c32
    // counterPrevious1: counterPrevIn=true → m_One → CmpIOp{counter_previous}
    //   compare: load(counter1) == 0  (did CCF1's mmad ever execute?)
    // CHECK: %[[CNT1_ZERO:.*]] = arith.cmpi eq, {{.*}}, %c0_i32 {counter_previous} : i32
    %for1 = scf.for %i1 = %lb to %ub step %c1_i32 iter_args(%acc1 = %empty) -> (tensor<64x32xf32>) : i32 {
      %cond = arith.cmpi eq, %i1, %c0_i32 : i32
      %if1 = scf.if %cond -> (tensor<64x32xf32>) {
        %mmad1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc1 : tensor<64x32xf32>) -> tensor<64x32xf32>
        scf.yield %mmad1 : tensor<64x32xf32>
      } else {
        scf.yield %acc1 : tensor<64x32xf32>
      }
      scf.yield %if1 : tensor<64x32xf32>
    }

    // CCF2: for with variable bounds → ReuseL0C, no IfOp (init not replaced)
    // CHECK: %[[ALLOCA2:.*]] = memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CCF2 mmad inside for: initCondition = andi(counterPrevious1, firstIter2), remain_in_l0c
    // CHECK: %[[INIT2:.*]] = arith.andi %[[CNT1_ZERO]], {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT2]], %c64, %c32, %c32
    // counterPrevious2: counterPrevIn=counterPrevious1(runtime) → andi(counter1==0, counter2==0)
    //   compare: andi(counter1==0, counter2==0)  (did CCF1 and CCF2 both never execute?)
    // CHECK: %[[CNT_PREV2:.*]] = arith.andi %[[CNT1_ZERO]], {{.*}} {counter_previous} : i1
    // CHECK-NOT: scf.if
    %for2 = scf.for %i2 = %lb to %ub step %c1_i32 iter_args(%acc2 = %for1) -> (tensor<64x32xf32>) : i32 {
      %mmad2 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc2 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad2 : tensor<64x32xf32>
    }

    // CCF3: bare mmad (no for) → ReuseL0C from CCF2's IfOp, no counter/addTailFallback
    // CCF3 mmad: initCondition = counterPrevious2 (AndIOp{counter_previous}), remain_in_l0c
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[CNT_PREV2]], %c64, %c32, %c32
    // CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    %mmad3 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%for2 : tensor<64x32xf32>) -> tensor<64x32xf32>

    // CCF4: for with constant bounds (mayNotExec=false) → ReuseL0C, counterPrevious=false (folded)
    // CCF4 mmad: initCondition = false (always accumulate), remain_in_l0c, no may_not_exec
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %false, %c64, %c32, %c32
    // CHECK-NOT: may_not_exec
    // CHECK-NOT: scf.if {{.*}} -> (tensor<64x32xf32>)
    // CHECK: return
    %for4 = scf.for %i4 = %c0 to %c8 step %c1 iter_args(%acc4 = %mmad3) -> (tensor<64x32xf32>) {
      %mmad4 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc4 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad4 : tensor<64x32xf32>
    }

    return %for4 : tensor<64x32xf32>
  }
}
// -----

// Test: when counterPrevious is compile-time false (previous CCF definitely
// executed), ReuseL0C skips addTailFallback and kMayNotExec.
// CCF1: bare mmadL1 (definitely executes) → result in L0C
// CCF2: for with variable bounds, ReuseL0C from CCF1 → counterPrevious=false
//        → no addTailFallback, no kMayNotExec, initCondition=false (always accumulate)

// CHECK-LABEL: func.func @test_reuse_l0c_skip_tail_fallback
module {
  func.func @test_reuse_l0c_skip_tail_fallback(%lb: i32, %ub: i32) -> tensor<64x32xf32> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<32x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf32>
    %empty = tensor.empty() : tensor<64x32xf32>

    // CCF1: bare mmadL1 (definitely executes) → kNormalizedInL0C
    // CHECK: hivm.hir.mmadL1 {{.*}} normalized_in_L0C
    %mmad1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%empty : tensor<64x32xf32>) -> tensor<64x32xf32>

    // CCF2: for with variable bounds, ReuseL0C from CCF1
    // counterPrevious = false (from MmadL1Op) → skip addTailFallback
    // CHECK: scf.for
    // CHECK-NOT: may_not_exec
    // CHECK-NOT: scf.if {{.*}} -> (tensor<64x32xf32>, i1)
    // CHECK: return
    %for2 = scf.for %i = %lb to %ub step %c1_i32 iter_args(%acc = %mmad1) -> (tensor<64x32xf32>) : i32 {
      %mmad2 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad2 : tensor<64x32xf32>
    }

    return %for2 : tensor<64x32xf32>
  }
}
// -----

// Test: all CCFs are mayNotExec (variable bounds), 2 mmads per for.
// CCF1 is ZeroInit, CCF2/CCF3 are ReuseL0C. AddIf should create IfOp only at
// the last CCF (CCF3) for each mmad, providing vbrc(0) fallback when all
// didn't execute. Also verifies counterBuf result index matching across the
// chain.

// CHECK-LABEL: func.func @test_all_may_not_exec_chain
module {
  func.func @test_all_may_not_exec_chain(%lb: i32, %ub: i32) -> (tensor<64x32xf32>, tensor<64x32xf32>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %false = arith.constant false
    %cst_zero = arith.constant 0.000000e+00 : f32
    %alloc_a = memref.alloc() : memref<64x32xf32>
    %a = bufferization.to_tensor %alloc_a restrict writable : memref<64x32xf32>
    %alloc_b = memref.alloc() : memref<32x32xf32>
    %b = bufferization.to_tensor %alloc_b restrict writable : memref<32x32xf32>
    %empty1 = tensor.empty() : tensor<64x32xf32>
    %empty2 = tensor.empty() : tensor<64x32xf32>
    %vbrc_zero1 = hivm.hir.vbrc ins(%cst_zero : f32) outs(%empty1 : tensor<64x32xf32>) -> tensor<64x32xf32>
    %vbrc_zero2 = hivm.hir.vbrc ins(%cst_zero : f32) outs(%empty2 : tensor<64x32xf32>) -> tensor<64x32xf32>

    // CCF1: 2 ZeroInit mmads, variable bounds
    // CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: memref.alloca() {normalize_matmul_counter = 1 : i32} : memref<i32>
    // CCF1 mmad: initCondition = firstIter (CmpIOp, counter == 0), remain_in_l0c
    // CHECK: %[[FIRST1_A:.*]] = arith.cmpi eq, {{.*}}, %c0_i32 : i32
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[FIRST1_A]], %c64, %c32, %c32
    // CHECK: %[[FIRST1_B:.*]] = arith.cmpi eq, {{.*}}, %c0_i32 : i32
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[FIRST1_B]], %c64, %c32, %c32
    %for1:2 = scf.for %i1 = %lb to %ub step %c1_i32 iter_args(%acc1_0 = %vbrc_zero1, %acc1_1 = %vbrc_zero2) -> (tensor<64x32xf32>, tensor<64x32xf32>) : i32 {
      %mmad1_0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc1_0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %mmad1_1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc1_1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad1_0, %mmad1_1 : tensor<64x32xf32>, tensor<64x32xf32>
    }

    // CCF2: 2 ReuseL0C mmads (init from CCF1), variable bounds
    // CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: memref.alloca() {normalize_matmul_counter = 1 : i32} : memref<i32>
    // CCF2 mmad: initCondition = andi(counterPrevious1, firstIter2) (AndIOp), remain_in_l0c
    // CHECK: %[[INIT2_A:.*]] = arith.andi {{.*}}, {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT2_A]], %c64, %c32, %c32
    // CHECK: %[[INIT2_B:.*]] = arith.andi {{.*}}, {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT2_B]], %c64, %c32, %c32
    %for2:2 = scf.for %i2 = %lb to %ub step %c1_i32 iter_args(%acc2_0 = %for1#0, %acc2_1 = %for1#1) -> (tensor<64x32xf32>, tensor<64x32xf32>) : i32 {
      %mmad2_0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc2_0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %mmad2_1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc2_1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad2_0, %mmad2_1 : tensor<64x32xf32>, tensor<64x32xf32>
    }

    // CCF3: 2 ReuseL0C mmads (init from CCF2), variable bounds — last mayNotExec
    // CHECK: memref.alloca() {normalize_matmul_counter = 0 : i32} : memref<i32>
    // CHECK: memref.alloca() {normalize_matmul_counter = 1 : i32} : memref<i32>
    // CCF3 mmad: initCondition = andi(counterPrevious2, firstIter3) (AndIOp), remain_in_l0c
    // CHECK: %[[INIT3_A:.*]] = arith.andi {{.*}}, {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT3_A]], %c64, %c32, %c32
    // CHECK: %[[INIT3_B:.*]] = arith.andi {{.*}}, {{.*}} : i1
    // CHECK: hivm.hir.mmadL1 {already_set_real_mkn, hivm.remain_in_l0c, normalized_in_L0C} ins({{.*}}, {{.*}}, %[[INIT3_B]], %c64, %c32, %c32
    // CHECK-COUNT-2: scf.if {{.*}} -> (tensor<64x32xf32>)
    // CHECK: hivm.hir.vbrc
    // CHECK: return
    %for3:2 = scf.for %i3 = %lb to %ub step %c1_i32 iter_args(%acc3_0 = %for2#0, %acc3_1 = %for2#1) -> (tensor<64x32xf32>, tensor<64x32xf32>) : i32 {
      %mmad3_0 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc3_0 : tensor<64x32xf32>) -> tensor<64x32xf32>
      %mmad3_1 = hivm.hir.mmadL1 ins(%a, %b, %false, %c16, %c16, %c16 : tensor<64x32xf32>, tensor<32x32xf32>, i1, index, index, index) outs(%acc3_1 : tensor<64x32xf32>) -> tensor<64x32xf32>
      scf.yield %mmad3_0, %mmad3_1 : tensor<64x32xf32>, tensor<64x32xf32>
    }

    return %for3#0, %for3#1 : tensor<64x32xf32>, tensor<64x32xf32>
  }
}
