// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v1" -hivm-lower-multi-buffer-counter -split-input-file %s | FileCheck %s
// RUN: bishengir-opt "-hivm-graph-sync-solver=solver-version=v2" -hivm-lower-multi-buffer-counter -split-input-file %s | FileCheck %s

// GraphSyncSolver assigns one function-scoped L0 ping-pong signal to every
// MmadL1Op. When the first MMAD lives in a nested region, the alloca must still
// be hoisted to the function entry block so it dominates all MMAD users.

// -----
module {
  // The first MMAD in walk order is inside scf.for, so the signal is created at
  // the start of the outer function block (not inside the loop).
  // CHECK-LABEL: func.func @test_ping_pong_signal_nested_for_and_if
  func.func @test_ping_pong_signal_nested_for_and_if(
      %gmA : memref<16xf32, #hivm.address_space<gm>>,
      %gmB : memref<16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // CHECK-NEXT: %[[SIGNAL:.*]] = memref.alloca() {hivm.ping_pong_signal} : memref<i32>
    // CHECK: memref.store %{{.*}}, %[[SIGNAL]][] : memref<i32>
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %bufA = hivm.hir.pointer_cast(%c0_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmA : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufA : memref<16xf32, #hivm.address_space<cbuf>>)
    %bufB = hivm.hir.pointer_cast(%c64_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmB : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufB : memref<16xf32, #hivm.address_space<cbuf>>)
    %out = hivm.hir.pointer_cast(%c0_i64)
        : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: scf.for
    scf.for %iv = %c0 to %c2 step %c1 {
      // CHECK: hivm.hir.mmadL1
      // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
      hivm.hir.mmadL1
          ins(%bufA, %bufB, %true, %c16, %c256, %c16 :
              memref<16xf32, #hivm.address_space<cbuf>>,
              memref<16xf32, #hivm.address_space<cbuf>>,
              i1, index, index, index)
          outs(%out : memref<256xf32, #hivm.address_space<cc>>)
    }
    // CHECK: scf.if
    scf.if %true {
      // CHECK: hivm.hir.mmadL1
      // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
      hivm.hir.mmadL1
          ins(%bufB, %bufA, %true, %c16, %c256, %c16 :
              memref<16xf32, #hivm.address_space<cbuf>>,
              memref<16xf32, #hivm.address_space<cbuf>>,
              i1, index, index, index)
          outs(%out : memref<256xf32, #hivm.address_space<cc>>)
    }
    // CHECK-NOT: memref.alloca() {hivm.ping_pong_signal}
    return
  }
}

// -----
module {
  // Share the entry-block signal with MMAD ops inside nested control flow.
  // CHECK-LABEL: func.func @test_ping_pong_signal_entry_and_nested_loop
  func.func @test_ping_pong_signal_entry_and_nested_loop(
      %gmA : memref<16xf32, #hivm.address_space<gm>>,
      %gmB : memref<16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // CHECK: %[[SIGNAL:.*]] = memref.alloca() {hivm.ping_pong_signal} : memref<i32>
    // CHECK: memref.store %{{.*}}, %[[SIGNAL]][] : memref<i32>
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %bufA = hivm.hir.pointer_cast(%c0_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmA : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufA : memref<16xf32, #hivm.address_space<cbuf>>)
    %bufB = hivm.hir.pointer_cast(%c64_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmB : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufB : memref<16xf32, #hivm.address_space<cbuf>>)
    %out = hivm.hir.pointer_cast(%c0_i64)
        : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    // CHECK: hivm.hir.mmadL1
    // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
    hivm.hir.mmadL1
        ins(%bufA, %bufB, %true, %c16, %c256, %c16 :
            memref<16xf32, #hivm.address_space<cbuf>>,
            memref<16xf32, #hivm.address_space<cbuf>>,
            i1, index, index, index)
        outs(%out : memref<256xf32, #hivm.address_space<cc>>)
    // CHECK: scf.for
    scf.for %iv = %c0 to %c2 step %c1 {
      // CHECK: hivm.hir.mmadL1
      // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
      hivm.hir.mmadL1
          ins(%bufB, %bufA, %true, %c16, %c256, %c16 :
              memref<16xf32, #hivm.address_space<cbuf>>,
              memref<16xf32, #hivm.address_space<cbuf>>,
              i1, index, index, index)
          outs(%out : memref<256xf32, #hivm.address_space<cc>>)
    }
    // CHECK-NOT: memref.alloca() {hivm.ping_pong_signal}
    return
  }
}

// -----
module {
  // MMAD ops only appear under scf.for/scf.if nesting; still one entry-block alloca.
  // CHECK-LABEL: func.func @test_ping_pong_signal_deeply_nested_scopes
  func.func @test_ping_pong_signal_deeply_nested_scopes(
      %gmA : memref<16xf32, #hivm.address_space<gm>>,
      %gmB : memref<16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // CHECK-NEXT: %[[SIGNAL:.*]] = memref.alloca() {hivm.ping_pong_signal} : memref<i32>
    // CHECK: memref.store %{{.*}}, %[[SIGNAL]][] : memref<i32>
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %bufA = hivm.hir.pointer_cast(%c0_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmA : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufA : memref<16xf32, #hivm.address_space<cbuf>>)
    %bufB = hivm.hir.pointer_cast(%c64_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmB : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufB : memref<16xf32, #hivm.address_space<cbuf>>)
    %out = hivm.hir.pointer_cast(%c0_i64)
        : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: scf.for
    scf.for %iv = %c0 to %c2 step %c1 {
      // CHECK: scf.if
      scf.if %true {
        // CHECK: hivm.hir.mmadL1
        // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
        hivm.hir.mmadL1
            ins(%bufA, %bufB, %true, %c16, %c256, %c16 :
                memref<16xf32, #hivm.address_space<cbuf>>,
                memref<16xf32, #hivm.address_space<cbuf>>,
                i1, index, index, index)
            outs(%out : memref<256xf32, #hivm.address_space<cc>>)
      }
    }
    // CHECK: scf.if
    scf.if %true {
      // CHECK: scf.for
      scf.for %jv = %c0 to %c2 step %c1 {
        // CHECK: hivm.hir.mmadL1
        // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
        hivm.hir.mmadL1
            ins(%bufB, %bufA, %true, %c16, %c256, %c16 :
                memref<16xf32, #hivm.address_space<cbuf>>,
                memref<16xf32, #hivm.address_space<cbuf>>,
                i1, index, index, index)
            outs(%out : memref<256xf32, #hivm.address_space<cc>>)
      }
    }
    // CHECK-NOT: memref.alloca() {hivm.ping_pong_signal}
    return
  }
}

// -----
module {
  // The first MMAD in walk order is inside scf.for, so the signal is created at
  // the start of the outer function block (not inside the loop).
  // CHECK-LABEL: func.func @test_ping_pong_signal_preassigned_nested
  func.func @test_ping_pong_signal_preassigned_nested(
      %gmA : memref<16xf32, #hivm.address_space<gm>>,
      %gmB : memref<16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // CHECK-NEXT: %[[SIGNAL:.*]] = memref.alloca() {hivm.ping_pong_signal} : memref<i32>
    // CHECK: memref.store %{{.*}}, %[[SIGNAL]][] : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %bufA = hivm.hir.pointer_cast(%c0_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmA : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufA : memref<16xf32, #hivm.address_space<cbuf>>)
    %bufB = hivm.hir.pointer_cast(%c64_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmB : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufB : memref<16xf32, #hivm.address_space<cbuf>>)
    %out = hivm.hir.pointer_cast(%c0_i64)
        : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: scf.for
    scf.for %iv = %c0 to %c2 step %c1 {
      %inner_signal = memref.alloca() : memref<i32>
      memref.store %c0_i32, %inner_signal[] : memref<i32>
      // CHECK: hivm.hir.mmadL1
      // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
      hivm.hir.mmadL1
          ins(%bufA, %bufB, %true, %c16, %c256, %c16 :
              memref<16xf32, #hivm.address_space<cbuf>>,
              memref<16xf32, #hivm.address_space<cbuf>>,
              i1, index, index, index)
          outs(%out : memref<256xf32, #hivm.address_space<cc>>)
          ping_pong_signal(%inner_signal : memref<i32>)
    }
    // CHECK: scf.if
    scf.if %true {
      // CHECK: hivm.hir.mmadL1
      // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
      hivm.hir.mmadL1
          ins(%bufB, %bufA, %true, %c16, %c256, %c16 :
              memref<16xf32, #hivm.address_space<cbuf>>,
              memref<16xf32, #hivm.address_space<cbuf>>,
              i1, index, index, index)
          outs(%out : memref<256xf32, #hivm.address_space<cc>>)
    }
    // CHECK-NOT: memref.alloca() {hivm.ping_pong_signal}
    return
  }
}


// -----
module {
  // Replace preassigned signals with one signal at function entry.
  // CHECK-LABEL: func.func @test_ping_pong_signal_replace_preassigned
  func.func @test_ping_pong_signal_replace_preassigned(
      %signal : memref<i32>,
      %gmA : memref<16xf32, #hivm.address_space<gm>>,
      %gmB : memref<16xf32, #hivm.address_space<gm>>)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIC>} {
    // CHECK-NEXT: %[[SIGNAL:.*]] = memref.alloca() {hivm.ping_pong_signal} : memref<i32>
    // CHECK: memref.store %{{.*}}, %[[SIGNAL]][] : memref<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %bufA = hivm.hir.pointer_cast(%c0_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmA : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufA : memref<16xf32, #hivm.address_space<cbuf>>)
    %bufB = hivm.hir.pointer_cast(%c64_i64)
        : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous}
        ins(%gmB : memref<16xf32, #hivm.address_space<gm>>)
        outs(%bufB : memref<16xf32, #hivm.address_space<cbuf>>)
    %out = hivm.hir.pointer_cast(%c0_i64)
        : memref<256xf32, #hivm.address_space<cc>>
    // CHECK: scf.for
    scf.for %iv = %c0 to %c2 step %c1 {
      %inner_signal = memref.alloca() : memref<i32>
      memref.store %c0_i32, %inner_signal[] : memref<i32>
      // CHECK: hivm.hir.mmadL1
      // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
      hivm.hir.mmadL1
          ins(%bufA, %bufB, %true, %c16, %c256, %c16 :
              memref<16xf32, #hivm.address_space<cbuf>>,
              memref<16xf32, #hivm.address_space<cbuf>>,
              i1, index, index, index)
          outs(%out : memref<256xf32, #hivm.address_space<cc>>)
          ping_pong_signal(%signal : memref<i32>)
    }
    // CHECK: scf.if
    scf.if %true {
      %other_signal = memref.alloca() : memref<i32>
      // CHECK: hivm.hir.mmadL1
      // CHECK-SAME: ping_pong_signal(%[[SIGNAL]] : memref<i32>)
      hivm.hir.mmadL1
          ins(%bufB, %bufA, %true, %c16, %c256, %c16 :
              memref<16xf32, #hivm.address_space<cbuf>>,
              memref<16xf32, #hivm.address_space<cbuf>>,
              i1, index, index, index)
          outs(%out : memref<256xf32, #hivm.address_space<cc>>)
          ping_pong_signal(%other_signal : memref<i32>)
    }
    // CHECK-NOT: memref.alloca() {hivm.ping_pong_signal}
    return
  }
}
