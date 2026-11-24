// RUN: bishengir-opt %s -hivm-enable-multi-buffer -split-input-file | FileCheck %s --check-prefix=ENABLE
// RUN: bishengir-opt %s -hivm-lower-multi-buffer-counter -split-input-file | FileCheck %s --check-prefix=LOWER

// -----

module {
// ENABLE-LABEL: func.func @reuse_counter_op(
  func.func @reuse_counter_op(%arg0: memref<16xf16, #hivm.address_space<gm>>,
                              %arg1: memref<16xf16, #hivm.address_space<gm>>,
                              %arg2: memref<16xf16, #hivm.address_space<gm>>) {
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c144_i64 = arith.constant 144 : i64
    %c256_i64 = arith.constant 256 : i64
    %c272_i64 = arith.constant 272 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // ENABLE: scf.for
    scf.for %iv = %c0 to %c4 step %c1 {
      // ENABLE: %[[CTR:.*]] = hivm.hir.multi_buffer_counter -> i64
      // ENABLE: arith.remui %[[CTR]], %{{.*}} : i64
      // ENABLE: arith.remui %[[CTR]], %{{.*}} : i64
      // ENABLE-NOT: hivm.hir.multi_buffer_counter
      %0 = hivm.hir.pointer_cast(%c0_i64, %c16_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %0 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      %1 = hivm.hir.pointer_cast(%c128_i64, %c144_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      annotation.mark %1 {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
      %2 = hivm.hir.pointer_cast(%c256_i64, %c272_i64) [] : memref<16xf16, #hivm.address_space<ub>>
      hivm.hir.load ins(%arg0 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%0 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.load ins(%arg1 : memref<16xf16, #hivm.address_space<gm>>)
                    outs(%1 : memref<16xf16, #hivm.address_space<ub>>)
      hivm.hir.store ins(%2 : memref<16xf16, #hivm.address_space<ub>>)
                     outs(%arg2 : memref<16xf16, #hivm.address_space<gm>>)
    }
    // ENABLE: return
    return
  }
}

// -----

module {
// LOWER-LABEL: func.func @lower_counter_op(
// LOWER: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
// LOWER: memref.store %{{.*}}, %[[CTR]][%{{.*}}] : memref<1xi64>
  func.func @lower_counter_op() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i64 = arith.constant 1 : i64
    // LOWER: scf.for
    scf.for %iv = %c0 to %c4 step %c1 {
      // LOWER: %[[CUR:.*]] = memref.load %[[CTR]][%{{.*}}] : memref<1xi64>
      // LOWER-NOT: hivm.hir.multi_buffer_counter
      %counter = hivm.hir.multi_buffer_counter -> i64
      %use = arith.addi %counter, %c1_i64 : i64
      // LOWER: arith.addi %[[CUR]], %{{.*}} : i64
      // LOWER: memref.store %{{.*}}, %[[CTR]][%{{.*}}] : memref<1xi64>
    }
    return
  }
}

// -----

// Multiple counter ops in one loop are deduplicated onto a single alloca; only
// one memref.alloca / load / increment triplet is emitted for the loop.
module {
// LOWER-LABEL: func.func @dedup_counter_ops(
// LOWER: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
// LOWER-NOT: memref.alloca()
  func.func @dedup_counter_ops() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // LOWER: scf.for
    scf.for %iv = %c0 to %c4 step %c1 {
      // LOWER: %[[CUR:.*]] = memref.load %[[CTR]][%{{.*}}] : memref<1xi64>
      // LOWER-NOT: hivm.hir.multi_buffer_counter
      %a = hivm.hir.multi_buffer_counter -> i64
      %b = hivm.hir.multi_buffer_counter -> i64
      %sum = arith.addi %a, %b : i64
    }
    return
  }
}

// -----

// counter op in a conditional block -> load inside that block
//   increment before its own terminator => the block counts its entries
// alloca -> function scope
module {
// LOWER-LABEL: func.func @lower_counter_in_conditional_block(
// LOWER: %[[CTR:.*]] = memref.alloca() : memref<1xi64>
// LOWER-NOT: memref.alloca()
  func.func @lower_counter_in_conditional_block(%cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2_i64 = arith.constant 2 : i64
    // LOWER: scf.for
    scf.for %iv = %c0 to %c4 step %c1 {
      // LOWER-NOT: memref.load
      // LOWER: scf.if
      scf.if %cond {
        // LOWER: %[[CUR:.*]] = memref.load %[[CTR]][%{{.*}}] : memref<1xi64>
        // LOWER-NEXT: %{{.*}} = arith.remui %[[CUR]], %{{.*}} : i64
        %a = hivm.hir.multi_buffer_counter -> i64
        %idx = arith.remui %a, %c2_i64 : i64
        // LOWER: arith.addi %[[CUR]], %{{.*}} : i64
        // LOWER: memref.store %{{.*}}, %[[CTR]][%{{.*}}] : memref<1xi64>
        // LOWER-NEXT: }
        scf.yield
      }
      scf.yield
    }
    return
  }
}

// -----

// two counters in one loop (body + guard) -> two allocas
//   one shared storage would advance each on the other's entries.
module {
// LOWER-LABEL: func.func @lower_two_counters_one_loop(
// One alloca and one load per counter; shared storage would show one of each.
// LOWER-COUNT-2: memref.alloca() : memref<1xi64>
// LOWER-COUNT-2: memref.load
  func.func @lower_two_counters_one_loop(%cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %iv = %c0 to %c4 step %c1 {
      %a = hivm.hir.multi_buffer_counter -> i64
      scf.if %cond {
        %b = hivm.hir.multi_buffer_counter -> i64
        scf.yield
      }
      scf.yield
    }
    return
  }
}
