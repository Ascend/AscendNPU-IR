// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s
// RUN: bishengir-opt %s -convert-hivm-to-std | FileCheck %s --check-prefix=DECL
// RUN: bishengir-opt %s -hivm-flatten-ops -convert-hivm-to-std | FileCheck %s
// RUN: bishengir-opt %s | bishengir-opt -convert-hivm-to-std | FileCheck %s
// RUN: bishengir-opt %s -mlir-print-op-generic | bishengir-opt -convert-hivm-to-std | FileCheck %s
// RUN: bishengir-opt %s -one-shot-bufferize -convert-hivm-to-std | FileCheck %s
// RUN: bishengir-opt %s -convert-hivm-to-std -canonicalize -cse | FileCheck %s --check-prefix=INIT
// RUN: sed 's/Ascend950PR_9589/Ascend960/g' %s | not bishengir-opt -convert-hivm-to-std 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: sed 's/2x4x16x16xbf16/2x4x16x16xf16/g' %s | not bishengir-opt -convert-hivm-to-std 2>&1 | FileCheck %s --check-prefix=MIXED

// MIXED: A5 mmadL0 requires matching f16 or bf16 inputs and an f32 accumulator

// Operands are already-backed memrefs. Allocation planning and automatic
// synchronization are supplied outside this conversion test. Bufferization
// still checks that the op's existing pure-buffer interface is preserved.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9589">} {

  // CHECK-LABEL: func.func @clear_then_accumulate
  // INIT-LABEL: func.func @clear_then_accumulate
  func.func @clear_then_accumulate(%a: memref<2x2x16x16xbf16, #hivm.address_space<ca>>, %b: memref<2x4x16x16xbf16, #hivm.address_space<cb>>, %c: memref<4x2x16x16xf32, #hivm.address_space<cc>>) {
    %m = arith.constant 32 : index
    %k = arith.constant 32 : index
    %n = arith.constant 64 : index
    // Check the ABI flag values after folding the i1-to-i64 extensions.
    // INIT-DAG: %[[ONE:.*]] = arith.constant 1 : i64
    // INIT-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
    %false = arith.constant false
    %true = arith.constant true
    // REJECT: failed to legalize operation 'hivm.hir.mmadL0'
    // CHECK: call @mmadl0_a5_bfloat16_t_to_float(
    // INIT: call @mmadl0_a5_bfloat16_t_to_float({{.*}}, %[[ONE]])
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %true :
      memref<2x2x16x16xbf16, #hivm.address_space<ca>>, memref<2x4x16x16xbf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    // CHECK: call @mmadl0_a5_bfloat16_t_to_float(
    // INIT: call @mmadl0_a5_bfloat16_t_to_float({{.*}}, %[[ZERO]])
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %false :
      memref<2x2x16x16xbf16, #hivm.address_space<ca>>, memref<2x4x16x16xbf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    // CHECK-NOT: hivm.hir.mmadL0
    // INIT: return
    return
  }

  // CHECK-LABEL: func.func @dynamic_clear(
  // CHECK-SAME: %[[A:[^:]+]]: memref<3x2x16x16xbf16, #hivm.address_space<ca>>,
  // CHECK-SAME: %[[B:[^:]+]]: memref<3x4x16x16xbf16, #hivm.address_space<cb>>,
  // CHECK-SAME: %[[C:[^:]+]]: memref<4x2x16x16xf32, #hivm.address_space<cc>>,
  // CHECK-SAME: %[[CLEAR:[^:]+]]: i1
  func.func @dynamic_clear(%a: memref<3x2x16x16xbf16, #hivm.address_space<ca>>, %b: memref<3x4x16x16xbf16, #hivm.address_space<cb>>, %c: memref<4x2x16x16xf32, #hivm.address_space<cc>>, %clear: i1) {
    // CHECK-DAG: %[[M:.*]] = arith.constant 32 : i64
    %m = arith.constant 32 : index
    // CHECK-DAG: %[[K:.*]] = arith.constant 48 : i64
    %k = arith.constant 48 : index
    // CHECK-DAG: %[[N:.*]] = arith.constant 64 : i64
    %n = arith.constant 64 : index
    %false = arith.constant false
    // CHECK-DAG: %[[CLEAR64:.*]] = arith.extui %[[CLEAR]] : i1 to i64
    // CHECK-DAG: %[[ADESC:.*]] = memref.cast %[[A]]
    // CHECK-DAG: %[[BDESC:.*]] = memref.cast %[[B]]
    // CHECK-DAG: %[[CDESC:.*]] = memref.cast %[[C]]
    // CHECK: call @mmadl0_a5_bfloat16_t_to_float(%[[ADESC]], %[[BDESC]], %[[CDESC]], %[[M]], %[[K]], %[[N]], %[[CLEAR64]])
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %clear :
      memref<3x2x16x16xbf16, #hivm.address_space<ca>>, memref<3x4x16x16xbf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    return
  }

  // Element offsets are preserved in the descriptors for the template:
  // A=256 f16 (512 bytes), B=512 f16 (1024 bytes), C=256 f32 (1024 bytes).
  // CHECK-LABEL: func.func @nonzero_offsets
  func.func @nonzero_offsets(%a: memref<2x2x16x16xbf16, strided<[512, 256, 16, 1], offset: 256>, #hivm.address_space<ca>>, %b: memref<2x4x16x16xbf16, strided<[1024, 256, 16, 1], offset: 512>, #hivm.address_space<cb>>, %c: memref<4x2x16x16xf32, strided<[512, 256, 16, 1], offset: 256>, #hivm.address_space<cc>>) {
    %m = arith.constant 32 : index
    %k = arith.constant 32 : index
    %n = arith.constant 64 : index
    %false = arith.constant false
    %true = arith.constant true
    // CHECK-DAG: %[[AO:.*]] = memref.cast %arg0 : memref<2x2x16x16xbf16, strided<[512, 256, 16, 1], offset: 256>, #hivm.address_space<ca>>
    // CHECK-DAG: %[[BO:.*]] = memref.cast %arg1 : memref<2x4x16x16xbf16, strided<[1024, 256, 16, 1], offset: 512>, #hivm.address_space<cb>>
    // CHECK-DAG: %[[CO:.*]] = memref.cast %arg2 : memref<4x2x16x16xf32, strided<[512, 256, 16, 1], offset: 256>, #hivm.address_space<cc>>
    // CHECK: call @mmadl0_a5_bfloat16_t_to_float(%[[AO]], %[[BO]], %[[CO]],
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %true :
      memref<2x2x16x16xbf16, strided<[512, 256, 16, 1], offset: 256>, #hivm.address_space<ca>>, memref<2x4x16x16xbf16, strided<[1024, 256, 16, 1], offset: 512>, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, strided<[512, 256, 16, 1], offset: 256>, #hivm.address_space<cc>>)
    // CHECK: call @mmadl0_a5_bfloat16_t_to_float(
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %false :
      memref<2x2x16x16xbf16, strided<[512, 256, 16, 1], offset: 256>, #hivm.address_space<ca>>, memref<2x4x16x16xbf16, strided<[1024, 256, 16, 1], offset: 512>, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, strided<[512, 256, 16, 1], offset: 256>, #hivm.address_space<cc>>)
    return
  }

  // CHECK-LABEL: func.func @dynamic_l0_offset
  func.func @dynamic_l0_offset(%a: memref<2x2x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<ca>>, %b: memref<2x4x16x16xbf16, #hivm.address_space<cb>>, %c: memref<4x2x16x16xf32, #hivm.address_space<cc>>) {
    %m = arith.constant 32 : index
    %k = arith.constant 32 : index
    %n = arith.constant 64 : index
    %false = arith.constant false
    %true = arith.constant true
    // CHECK: call @mmadl0_a5_bfloat16_t_to_float(
    hivm.hir.mmadL0 ins(%a, %b, %m, %k, %n, %false, %false, %true :
      memref<2x2x16x16xbf16, strided<[512, 256, 16, 1], offset: ?>, #hivm.address_space<ca>>, memref<2x4x16x16xbf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
      outs(%c : memref<4x2x16x16xf32, #hivm.address_space<cc>>)
    return
  }

  // DECL: func.func private @mmadl0_a5_bfloat16_t_to_float(memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<ca>>, memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cb>>, memref<?x?x?x?xf32, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<cc>>, i64, i64, i64, i64){{.*}}llvm.emit_c_interface
}
