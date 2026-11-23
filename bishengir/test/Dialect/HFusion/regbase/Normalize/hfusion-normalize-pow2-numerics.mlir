// REQUIRES: execution-engine
//
// Numeric execution test for the pow2 strength reduction (review round 1,
// comment: zero numeric verification). Both RUN lines must print identical,
// correct results: the first exercises the rewritten shift/and sequence,
// the second the original lowering (enable-div-rem-pow2=false).
//
// Inputs (i32, DIV=8): 100, -100, 7, -7, 0, INT_MIN, INT_MAX, 65535 --
// negative operands pin the truncation semantics of the signed correction.
//
// RUN: bishengir-opt %s --hfusion-normalize-ops="use-regbase=true" \
// RUN:   --convert-hfusion-to-hivm \
// RUN:   --execution-engine-convert-hivm-to-upstream="convert-to-named-op=true" \
// RUN:   --execution-engine-convert-hfusion-to-upstream \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: sed -e "s/module attributes {[^}]*}/module/" | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s
//
// RUN: bishengir-opt %s --hfusion-normalize-ops="use-regbase=true enable-div-rem-pow2=false" \
// RUN:   --convert-hfusion-to-hivm \
// RUN:   --execution-engine-convert-hivm-to-upstream="convert-to-named-op=true" \
// RUN:   --execution-engine-convert-hfusion-to-upstream \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN: sed -e "s/module attributes {[^}]*}/module/" | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {
  func.func @pow2_kernel(%x: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %c8 = arith.constant 8 : i32
    %q0 = tensor.empty() : tensor<8xi32>
    %q = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
      ins(%x, %c8 : tensor<8xi32>, i32) outs(%q0 : tensor<8xi32>) -> tensor<8xi32>
    %r0 = tensor.empty() : tensor<8xi32>
    %r = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>}
      ins(%x, %c8 : tensor<8xi32>, i32) outs(%r0 : tensor<8xi32>) -> tensor<8xi32>
    return %q, %r : tensor<8xi32>, tensor<8xi32>
  }

  func.func @main() {
    %x = memref.alloc() : memref<8xi32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index

    %v0 = arith.constant 100 : i32
    %v1 = arith.constant -100 : i32
    %v2 = arith.constant 7 : i32
    %v3 = arith.constant -7 : i32
    %v4 = arith.constant 0 : i32
    %v5 = arith.constant -2147483648 : i32
    %v6 = arith.constant 2147483647 : i32
    %v7 = arith.constant 65535 : i32

    memref.store %v0, %x[%c0] : memref<8xi32>
    memref.store %v1, %x[%c1] : memref<8xi32>
    memref.store %v2, %x[%c2] : memref<8xi32>
    memref.store %v3, %x[%c3] : memref<8xi32>
    memref.store %v4, %x[%c4] : memref<8xi32>
    memref.store %v5, %x[%c5] : memref<8xi32>
    memref.store %v6, %x[%c6] : memref<8xi32>
    memref.store %v7, %x[%c7] : memref<8xi32>

    %xt = bufferization.to_tensor %x restrict writable : memref<8xi32>
    %q, %r = func.call @pow2_kernel(%xt) : (tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>)

    %qm = bufferization.to_memref %q : memref<8xi32>
    %q_unranked = memref.cast %qm : memref<8xi32> to memref<*xi32>
    func.call @printMemrefI32(%q_unranked) : (memref<*xi32>) -> ()

    %rm = bufferization.to_memref %r : memref<8xi32>
    %r_unranked = memref.cast %rm : memref<8xi32> to memref<*xi32>
    func.call @printMemrefI32(%r_unranked) : (memref<*xi32>) -> ()

    memref.dealloc %x : memref<8xi32>
    return
  }

  func.func private @printMemrefI32(memref<*xi32>)
}

// quotient:  trunc(x / 8)
// CHECK: [12, -12, 0, 0, 0, -268435456, 268435455, 8191]
// remainder: C-style trunc semantics (negative operands stay negative)
// CHECK: [4, -4, 7, -7, 0, 0, 7, 7]
