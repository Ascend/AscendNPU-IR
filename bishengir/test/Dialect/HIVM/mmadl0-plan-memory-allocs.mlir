// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-plan-memory -split-input-file -verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s -hacc-append-device-spec=target=Ascend950PR_9589 -hivm-plan-memory=plan-memory-strategy=largest-first -split-input-file -verify-diagnostics | FileCheck %s

// Planner-only fixtures. Scalar stores below establish allocation lifetimes;
// they do not claim scalar hardware access to L0 or executable kernels.
// MMAD L0 uses caller-provided tiles to enable the per-function planning policy
// without changing the allocation sizes/lifetimes under test.
// Rounded sizes and addresses are bytes, not elements or planner bits.
// CHECK-LABEL: func.func @rounded_slots
func.func @rounded_slots(%mmad_a: memref<1x1x16x16xf16, #hivm.address_space<ca>>, %mmad_b: memref<1x1x16x16xf16, #hivm.address_space<cb>>, %mmad_c: memref<1x1x16x16xf32, #hivm.address_space<cc>>) {
  %mmad_dim = arith.constant 16 : index
  %mmad_false = arith.constant false
  %mmad_true = arith.constant true
  hivm.hir.mmadL0 ins(%mmad_a, %mmad_b, %mmad_dim, %mmad_dim, %mmad_dim, %mmad_false, %mmad_false, %mmad_true :
    memref<1x1x16x16xf16, #hivm.address_space<ca>>, memref<1x1x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
    outs(%mmad_c : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
  %idx = arith.constant 0 : index
  %half = arith.constant 1.0 : f16
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // CHECK-DAG: %[[AB:.*]] = arith.constant 512 : i64
  // CHECK: hivm.hir.pointer_cast(%[[ZERO]]) : memref<1xf16, #hivm.address_space<ca>>
  %a0 = memref.alloc() : memref<1xf16, #hivm.address_space<ca>>
  memref.store %half, %a0[%idx] : memref<1xf16, #hivm.address_space<ca>>
  // CHECK: hivm.hir.pointer_cast(%[[AB]]) : memref<1xf16, #hivm.address_space<ca>>
  %a1 = memref.alloc() : memref<1xf16, #hivm.address_space<ca>>
  memref.store %half, %a1[%idx] : memref<1xf16, #hivm.address_space<ca>>
  // CHECK: hivm.hir.pointer_cast(%[[ZERO]]) : memref<1xf16, #hivm.address_space<cb>>
  %b0 = memref.alloc() : memref<1xf16, #hivm.address_space<cb>>
  memref.store %half, %b0[%idx] : memref<1xf16, #hivm.address_space<cb>>
  // CHECK: hivm.hir.pointer_cast(%[[AB]]) : memref<1xf16, #hivm.address_space<cb>>
  %b1 = memref.alloc() : memref<1xf16, #hivm.address_space<cb>>
  memref.store %half, %b1[%idx] : memref<1xf16, #hivm.address_space<cb>>
  return
}

// -----

// A static view retains its element offset; only its backing alloc receives
// an address. Two explicitly requested multibuffer slots remain independent.
// CHECK-LABEL: func.func @view_and_multibuffer
func.func @view_and_multibuffer(%mmad_a: memref<1x1x16x16xf16, #hivm.address_space<ca>>, %mmad_b: memref<1x1x16x16xf16, #hivm.address_space<cb>>, %mmad_c: memref<1x1x16x16xf32, #hivm.address_space<cc>>) {
  %mmad_dim = arith.constant 16 : index
  %mmad_false = arith.constant false
  %mmad_true = arith.constant true
  hivm.hir.mmadL0 ins(%mmad_a, %mmad_b, %mmad_dim, %mmad_dim, %mmad_dim, %mmad_false, %mmad_false, %mmad_true :
    memref<1x1x16x16xf16, #hivm.address_space<ca>>, memref<1x1x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
    outs(%mmad_c : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // CHECK-DAG: %[[NEXT:.*]] = arith.constant 2048 : i64
  // CHECK: %[[A:.*]] = hivm.hir.pointer_cast(%[[ZERO]], %[[NEXT]]) : memref<1024xf16, #hivm.address_space<ca>>
  %a = memref.alloc() : memref<1024xf16, #hivm.address_space<ca>>
  annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<1024xf16, #hivm.address_space<ca>>
  // CHECK: memref.subview %[[A]][256] [256] [1]
  %view = memref.subview %a[256] [256] [1] : memref<1024xf16, #hivm.address_space<ca>> to memref<256xf16, strided<[1], offset: 256>, #hivm.address_space<ca>>
  %idx = arith.constant 0 : index
  %value = arith.constant 1.0 : f16
  memref.store %value, %view[%idx] : memref<256xf16, strided<[1], offset: 256>, #hivm.address_space<ca>>
  return
}

// -----

// A small allocation before the MMAD output must not leave C at byte 512.
// CHECK-LABEL: func.func @accumulator_alignment
func.func @accumulator_alignment(%a: memref<1x1x16x16xf16, #hivm.address_space<ca>>, %b: memref<1x1x16x16xf16, #hivm.address_space<cb>>) {
  %idx = arith.constant 0 : index
  %value = arith.constant 1.0 : f32
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
  // CHECK-DAG: %[[C:.*]] = arith.constant 1024 : i64
  // CHECK: hivm.hir.pointer_cast(%[[ZERO]]) : memref<1xf32, #hivm.address_space<cc>>
  %padding = memref.alloc() : memref<1xf32, #hivm.address_space<cc>>
  memref.store %value, %padding[%idx] : memref<1xf32, #hivm.address_space<cc>>
  // CHECK: hivm.hir.pointer_cast(%[[C]]) : memref<1x1x16x16xf32, #hivm.address_space<cc>>
  %c = memref.alloc() : memref<1x1x16x16xf32, #hivm.address_space<cc>>
  %dim = arith.constant 16 : index
  %false = arith.constant false
  %true = arith.constant true
  hivm.hir.mmadL0 ins(%a, %b, %dim, %dim, %dim, %false, %false, %true :
    memref<1x1x16x16xf16, #hivm.address_space<ca>>, memref<1x1x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
    outs(%c : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
  return
}

// No MMAD at all, still in the same module. The new byte-addressable-element
// validation must not apply to legacy L0C allocations. This is planner-only IR.
// CHECK-LABEL: func.func @legacy_without_mmad
func.func @legacy_without_mmad() {
  %idx = arith.constant 0 : index
  %value = arith.constant true
  // CHECK: hivm.hir.pointer_cast({{.*}}) : memref<1xi1, #hivm.address_space<cc>>
  %c = memref.alloc() : memref<1xi1, #hivm.address_space<cc>>
  memref.store %value, %c[%idx] : memref<1xi1, #hivm.address_space<cc>>
  return
}

// -----

// Exact capacity is accepted independently in L0A and L0B.
// CHECK-LABEL: func.func @capacity_boundary
func.func @capacity_boundary(%mmad_a: memref<1x1x16x16xf16, #hivm.address_space<ca>>, %mmad_b: memref<1x1x16x16xf16, #hivm.address_space<cb>>, %mmad_c: memref<1x1x16x16xf32, #hivm.address_space<cc>>) {
  %mmad_dim = arith.constant 16 : index
  %mmad_false = arith.constant false
  %mmad_true = arith.constant true
  hivm.hir.mmadL0 ins(%mmad_a, %mmad_b, %mmad_dim, %mmad_dim, %mmad_dim, %mmad_false, %mmad_false, %mmad_true :
    memref<1x1x16x16xf16, #hivm.address_space<ca>>, memref<1x1x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
    outs(%mmad_c : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
  %idx = arith.constant 0 : index
  %half = arith.constant 1.0 : f16
  // CHECK: hivm.hir.pointer_cast({{.*}}) : memref<32768xf16, #hivm.address_space<ca>>
  %a = memref.alloc() : memref<32768xf16, #hivm.address_space<ca>>
  memref.store %half, %a[%idx] : memref<32768xf16, #hivm.address_space<ca>>
  // CHECK: hivm.hir.pointer_cast({{.*}}) : memref<32768xf16, #hivm.address_space<cb>>
  %b = memref.alloc() : memref<32768xf16, #hivm.address_space<cb>>
  memref.store %half, %b[%idx] : memref<32768xf16, #hivm.address_space<cb>>
  return
}

// -----

// Even sequentially used explicit slots must not be reused to hide overflow.
// expected-error@below {{ca overflow, requires 1048576 bits while 524288 bits available!}}
func.func @no_time_reuse_a(%mmad_a: memref<1x1x16x16xf16, #hivm.address_space<ca>>, %mmad_b: memref<1x1x16x16xf16, #hivm.address_space<cb>>, %mmad_c: memref<1x1x16x16xf32, #hivm.address_space<cc>>) {
  %mmad_dim = arith.constant 16 : index
  %mmad_false = arith.constant false
  %mmad_true = arith.constant true
  hivm.hir.mmadL0 ins(%mmad_a, %mmad_b, %mmad_dim, %mmad_dim, %mmad_dim, %mmad_false, %mmad_false, %mmad_true :
    memref<1x1x16x16xf16, #hivm.address_space<ca>>, memref<1x1x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
    outs(%mmad_c : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
  %idx = arith.constant 0 : index
  %value = arith.constant 1.0 : f16
  %a0 = memref.alloc() : memref<32768xf16, #hivm.address_space<ca>>
  memref.store %value, %a0[%idx] : memref<32768xf16, #hivm.address_space<ca>>
  %a1 = memref.alloc() : memref<32768xf16, #hivm.address_space<ca>>
  memref.store %value, %a1[%idx] : memref<32768xf16, #hivm.address_space<ca>>
  return
}

// -----

// expected-error@below {{cb overflow, requires 1048576 bits while 524288 bits available!}}
func.func @no_time_reuse_b(%mmad_a: memref<1x1x16x16xf16, #hivm.address_space<ca>>, %mmad_b: memref<1x1x16x16xf16, #hivm.address_space<cb>>, %mmad_c: memref<1x1x16x16xf32, #hivm.address_space<cc>>) {
  %mmad_dim = arith.constant 16 : index
  %mmad_false = arith.constant false
  %mmad_true = arith.constant true
  hivm.hir.mmadL0 ins(%mmad_a, %mmad_b, %mmad_dim, %mmad_dim, %mmad_dim, %mmad_false, %mmad_false, %mmad_true :
    memref<1x1x16x16xf16, #hivm.address_space<ca>>, memref<1x1x16x16xf16, #hivm.address_space<cb>>, index, index, index, i1, i1, i1)
    outs(%mmad_c : memref<1x1x16x16xf32, #hivm.address_space<cc>>)
  %idx = arith.constant 0 : index
  %value = arith.constant 1.0 : f16
  %b0 = memref.alloc() : memref<32768xf16, #hivm.address_space<cb>>
  memref.store %value, %b0[%idx] : memref<32768xf16, #hivm.address_space<cb>>
  %b1 = memref.alloc() : memref<32768xf16, #hivm.address_space<cb>>
  memref.store %value, %b1[%idx] : memref<32768xf16, #hivm.address_space<cb>>
  return
}
