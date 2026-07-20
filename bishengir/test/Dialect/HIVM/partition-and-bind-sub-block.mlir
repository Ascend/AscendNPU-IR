// RUN: bishengir-opt %s -hivm-partition-and-bind-sub-block=enable-load-balanced=false -split-input-file -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,PIN
// RUN: bishengir-opt %s -hivm-partition-and-bind-sub-block=enable-load-balanced=true  -split-input-file -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,BAL

// Test 1 -- bail out + inline (legality failure).

// CHECK-LABEL: func.func @cross_core_mul_inlined
// CHECK-NOT:     scope.scope
// CHECK:         hivm.hir.gather_load ins(%arg0
// CHECK:         hivm.hir.gather_load ins(%arg1
// CHECK:         arith.mulf
// CHECK-NOT:     scope.scope
// CHECK-NOT:     scf.if
func.func @cross_core_mul_inlined(%arg0: memref<?xf16>, %arg1: memref<?xf16>) attributes {hacc.entry} {
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %ia = tensor.empty() : tensor<32x64xi64>
  %ea = tensor.empty() : tensor<32x64xf16>
  // sub_block=0 result (core V0).
  %a = scope.scope : () -> tensor<32x64xf16> {
    %ga = hivm.hir.gather_load ins(%arg0 : memref<?xf16>, %ia : tensor<32x64xi64>, %c1_i32 : i32) outs(%ea : tensor<32x64xf16>) -> tensor<32x64xf16>
    scope.return %ga : tensor<32x64xf16>
  } {sub_block = 0 : i64}
  // sub_block=1 result (core V1).
  %b = scope.scope : () -> tensor<32x64xf16> {
    %gb = hivm.hir.gather_load ins(%arg1 : memref<?xf16>, %ia : tensor<32x64xi64>, %c1_i32 : i32) outs(%ea : tensor<32x64xf16>) -> tensor<32x64xf16>
    scope.return %gb : tensor<32x64xf16>
  } {sub_block = 1 : i64}
  // ILLEGAL: a non-cube vector op mixing both sub-blocks -> origin `Both`.
  // expected-warning @+1 {{legality check failed}}
  %prod = arith.mulf %a, %b : tensor<32x64xf16>
  // A cube op keeps the func past the "has cube" gate.
  %da = tensor.empty() : tensor<32x64xf16>
  %eb = tensor.empty() : tensor<64x16xf16>
  %ec = tensor.empty() : tensor<32x16xf32>
  %mm = hivm.hir.mmadL1 ins(%da, %eb, %true, %c32, %c64, %c16 : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%ec : tensor<32x16xf32>) -> tensor<32x16xf32>
  return
}

// -----

// Test 2 -- default-pin (enable-load-balanced=false)

// CHECK-LABEL: func.func @free_node_default_pin
// CHECK:         hivm.hir.gather_load ins(%arg0
// CHECK:         %[[RC:.+]] = memref.reinterpret_cast %arg1
// PIN:           %[[C:.+]] = arith.constant 0 : index
// PIN:           arith.cmpi eq, %{{.+}}, %[[C]]
// PIN:           scf.if
// PIN:             hivm.hir.store {{.*}} outs(%[[RC]] :
func.func @free_node_default_pin(%arg0: memref<?xf16>, %arg1: memref<?xf16>) attributes {hacc.entry} {
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  // matmul opens the cube stage that both the scope and the free node live in.
  %da0 = tensor.empty() : tensor<32x64xf16>
  %db0 = tensor.empty() : tensor<64x16xf16>
  %ec0 = tensor.empty() : tensor<32x16xf32>
  %mm0 = hivm.hir.mmadL1 ins(%da0, %db0, %true, %c32, %c64, %c16 : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%ec0 : tensor<32x16xf32>) -> tensor<32x16xf32>
  // sub_block=0 scope -> V0 load = 1 for this stage.
  %ia = tensor.empty() : tensor<32x64xi64>
  %ea = tensor.empty() : tensor<32x64xf16>
  %a = scope.scope : () -> tensor<32x64xf16> {
    %ga = hivm.hir.gather_load ins(%arg0 : memref<?xf16>, %ia : tensor<32x64xi64>, %c1_i32 : i32) outs(%ea : tensor<32x64xf16>) -> tensor<32x64xf16>
    scope.return %ga : tensor<32x64xf16>
  } {sub_block = 0 : i64}
  // FREE NODE (fresh tensor -> GM store): pinned to core 0.
  %fb = tensor.empty() : tensor<32x16xf16>
  %rcb = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf16> to memref<32x16xf16, strided<[16, 1]>>
  hivm.hir.store ins(%fb : tensor<32x16xf16>) outs(%rcb : memref<32x16xf16, strided<[16, 1]>>)
  return
}

// -----

// Test 3 -- load-balanced (enable-load-balanced=true)
//
// Same input as Test 2. The per-stage balancer sees the stage already loaded on
// V0 (the sub_block=0 scope) and sends the free node to the lighter core, V1.

// CHECK-LABEL: func.func @free_node_load_balanced
// CHECK:         hivm.hir.gather_load ins(%arg0
// CHECK:         %[[RC:.+]] = memref.reinterpret_cast %arg1
// BAL:           %[[C:.+]] = arith.constant 1 : index
// BAL:           arith.cmpi eq, %{{.+}}, %[[C]]
// BAL:           scf.if
// BAL:             hivm.hir.store {{.*}} outs(%[[RC]] :
func.func @free_node_load_balanced(%arg0: memref<?xf16>, %arg1: memref<?xf16>) attributes {hacc.entry} {
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  // matmul opens the cube stage that both the scope and the free node live in.
  %da0 = tensor.empty() : tensor<32x64xf16>
  %db0 = tensor.empty() : tensor<64x16xf16>
  %ec0 = tensor.empty() : tensor<32x16xf32>
  %mm0 = hivm.hir.mmadL1 ins(%da0, %db0, %true, %c32, %c64, %c16 : tensor<32x64xf16>, tensor<64x16xf16>, i1, index, index, index) outs(%ec0 : tensor<32x16xf32>) -> tensor<32x16xf32>
  // sub_block=0 scope -> V0 load = 1 for this stage.
  %ia = tensor.empty() : tensor<32x64xi64>
  %ea = tensor.empty() : tensor<32x64xf16>
  %a = scope.scope : () -> tensor<32x64xf16> {
    %ga = hivm.hir.gather_load ins(%arg0 : memref<?xf16>, %ia : tensor<32x64xi64>, %c1_i32 : i32) outs(%ea : tensor<32x64xf16>) -> tensor<32x64xf16>
    scope.return %ga : tensor<32x64xf16>
  } {sub_block = 0 : i64}
  // FREE NODE (fresh tensor -> GM store): balanced onto core 1.
  %fb = tensor.empty() : tensor<32x16xf16>
  %rcb = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32, 16], strides: [16, 1] : memref<?xf16> to memref<32x16xf16, strided<[16, 1]>>
  hivm.hir.store ins(%fb : tensor<32x16xf16>) outs(%rcb : memref<32x16xf16, strided<[16, 1]>>)
  return
}
