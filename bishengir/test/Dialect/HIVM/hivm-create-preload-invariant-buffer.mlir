// RUN: bishengir-opt %s -allow-unregistered-dialect -create-preload | FileCheck %s

// A loop-carried state aliases the same outer-loop buffer on every iteration.
// Its pointer_cast must stay in the outer loop: moving a copy into the preload
// loop changes which loop selects the address from the multi-buffer ring.
// CHECK-LABEL: func.func @preserve_outer_buffer_selection
// CHECK: scf.for
// CHECK: %[[STATE:.*]] = hivm.hir.pointer_cast
// CHECK: scf.for
// CHECK-NOT: hivm.hir.pointer_cast
// CHECK: "test.update"(%[[STATE]], %[[STATE]])
// CHECK-NOT: hivm.hir.pointer_cast
// CHECK: "test.result"(%[[STATE]], %[[STATE]])
func.func @preserve_outer_buffer_selection(%outer_n: i32, %inner_n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %addr0 = arith.constant 0 : i64
  %addr1 = arith.constant 8192 : i64
  scf.for %outer = %c0 to %outer_n step %c1 : i32 {
    %state = hivm.hir.pointer_cast(%addr0, %addr1)
        : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %state {hivm.multi_buffer = 2 : i32}
        : memref<32x64xf32, #hivm.address_space<ub>>
    %result = scf.for %inner = %c0 to %inner_n step %c1
        iter_args(%current = %state)
        -> memref<32x64xf32, #hivm.address_space<ub>> : i32 {
      scope.scope : () -> () {
        "test.update"(%current, %state)
            : (memref<32x64xf32, #hivm.address_space<ub>>,
               memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        scope.return
      } {hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32,
         no_inline}
      scf.yield %state : memref<32x64xf32, #hivm.address_space<ub>>
    }
    "test.result"(%result, %state)
        : (memref<32x64xf32, #hivm.address_space<ub>>,
           memref<32x64xf32, #hivm.address_space<ub>>) -> ()
  }
  return
}

// Invariant annotated buffers must be removed before searching for a writer.
// CHECK-LABEL: func.func @preserve_annotated_outer_buffer_selection
// CHECK: scf.for
// CHECK: %[[STATE:.*]] = hivm.hir.pointer_cast
// CHECK: scf.for
// CHECK-NOT: hivm.hir.pointer_cast
// CHECK: "test.update"(%[[STATE]], %[[STATE]])
// CHECK-NOT: hivm.hir.pointer_cast
// CHECK: "test.result"(%[[STATE]], %[[STATE]])
func.func @preserve_annotated_outer_buffer_selection(%outer_n: i32, %inner_n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %addr0 = arith.constant 0 : i64
  %addr1 = arith.constant 8192 : i64
  scf.for %outer = %c0 to %outer_n step %c1 : i32 {
    %state = hivm.hir.pointer_cast(%addr0, %addr1)
        : memref<32x64xf32, #hivm.address_space<ub>>
    annotation.mark %state {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer}
        : memref<32x64xf32, #hivm.address_space<ub>>
    %result = scf.for %inner = %c0 to %inner_n step %c1
        iter_args(%current = %state)
        -> memref<32x64xf32, #hivm.address_space<ub>> : i32 {
      scope.scope : () -> () {
        "test.update"(%current, %state)
            : (memref<32x64xf32, #hivm.address_space<ub>>,
               memref<32x64xf32, #hivm.address_space<ub>>) -> ()
        scope.return
      } {hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32,
         no_inline}
      scf.yield %state : memref<32x64xf32, #hivm.address_space<ub>>
    }
    "test.result"(%result, %state)
        : (memref<32x64xf32, #hivm.address_space<ub>>,
           memref<32x64xf32, #hivm.address_space<ub>>) -> ()
  }
  return
}
