// RUN: bishengir-opt %s -hivm-lower-to-cpu-backend | FileCheck %s
// RUN: bishengir-opt %s -hivm-lower-to-cpu-backend -convert-hivm-to-llvm -convert-arith-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

// A zero address can also name a real device buffer. Only the marked null
// signal descriptor must bypass CPU allocation.
// CHECK-LABEL: func.func @null_signal(
// LLVM-LABEL: llvm.func @null_signal(
func.func @null_signal() {
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
  // LLVM: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64)
  %zero = arith.constant 0 : i64

  // CHECK: %[[SIGNAL:.*]] = hivm.hir.pointer_cast(%[[ZERO]]) {hivm.ping_pong_signal} : memref<i32>
  // LLVM: %[[NULL:.*]] = llvm.inttoptr %[[ZERO]] : i64 to !llvm.ptr
  // LLVM: llvm.insertvalue %[[NULL]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64)>
  // LLVM: llvm.insertvalue %[[NULL]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64)>
  %signal = hivm.hir.pointer_cast(%zero) {hivm.ping_pong_signal} : memref<i32>

  // CHECK: %[[BUFFER:.*]] = memref.alloca() {{.*}} : memref<i32>
  // LLVM: llvm.alloca
  %buffer = hivm.hir.pointer_cast(%zero) : memref<i32>

  // CHECK: call @consume(%[[SIGNAL]], %[[BUFFER]])
  // LLVM: llvm.call @consume
  call @consume(%signal, %buffer) : (memref<i32>, memref<i32>) -> ()
  return
}

func.func private @consume(memref<i32>, memref<i32>)
