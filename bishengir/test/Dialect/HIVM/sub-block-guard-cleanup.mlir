// RUN: bishengir-opt --hivm-sub-block-guard-cleanup -split-input-file %s | FileCheck %s

func.func private @vf(memref<32x64xf16>, memref<32x64xf16>)

// CHECK-LABEL: func @detected_by_predicate
//       CHECK:   scf.if %{{.*}} {
//       CHECK:     func.call @vf(%{{.*}}, %[[OUT:.*]]) :
//       CHECK:   }
//   CHECK-NOT:   else
//       CHECK:   memref.copy %[[OUT]], %{{.*}}
func.func @detected_by_predicate(%in: memref<32x64xf16>, %dst: memref<32x64xf16>) {
  %c0 = arith.constant 0 : index
  %i = hivm.hir.get_sub_block_idx -> i64
  %idx = arith.index_cast %i : i64 to index
  %cond = arith.cmpi eq, %idx, %c0 : index
  %init = memref.alloc() : memref<32x64xf16>
  %r = scf.if %cond -> (memref<32x64xf16>) {
    %out = memref.alloc() : memref<32x64xf16>
    func.call @vf(%in, %out) : (memref<32x64xf16>, memref<32x64xf16>) -> ()
    scf.yield %out : memref<32x64xf16>
  } else {
    scf.yield %init : memref<32x64xf16>
  }
  memref.copy %r, %dst : memref<32x64xf16> to memref<32x64xf16>
  return
}

// -----

func.func private @vf(memref<32x64xf16>, memref<32x64xf16>)

// CHECK-LABEL: func @already_result_free
//       CHECK:   scf.if %{{.*}} {
//       CHECK:     func.call @vf
//   CHECK-NOT:   else
func.func @already_result_free(%in: memref<32x64xf16>, %out: memref<32x64xf16>) {
  %c0 = arith.constant 0 : index
  %i = hivm.hir.get_sub_block_idx -> i64
  %idx = arith.index_cast %i : i64 to index
  %cond = arith.cmpi eq, %idx, %c0 : index
  scf.if %cond {
    func.call @vf(%in, %out) : (memref<32x64xf16>, memref<32x64xf16>) -> ()
  }
  return
}

// -----

func.func private @vf(memref<32x64xf16>, memref<32x64xf16>)
// CHECK-LABEL: func @preserves_else_anchor
//   CHECK-NOT:   = scf.if
//       CHECK:   scf.if %{{.*}} {
//       CHECK:     hivm.hir.anchor {id = 100
//       CHECK:     func.call @vf
//       CHECK:     hivm.hir.anchor {id = 101
//       CHECK:   } else {
//       CHECK:     hivm.hir.anchor {id = 102
//       CHECK:   }
func.func @preserves_else_anchor(%in: memref<32x64xf16>, %dst: memref<32x64xf16>) {
  %c0 = arith.constant 0 : index
  %i = hivm.hir.get_sub_block_idx -> i64
  %idx = arith.index_cast %i : i64 to index
  %cond = arith.cmpi eq, %idx, %c0 : index
  %init = memref.alloc() : memref<32x64xf16>
  %r = scf.if %cond -> (memref<32x64xf16>) {
    hivm.hir.anchor {id = 100 : i64}
    %out = memref.alloc() : memref<32x64xf16>
    func.call @vf(%in, %out) : (memref<32x64xf16>, memref<32x64xf16>) -> ()
    hivm.hir.anchor {id = 101 : i64}
    scf.yield %out : memref<32x64xf16>
  } else {
    hivm.hir.anchor {id = 102 : i64}
    scf.yield %init : memref<32x64xf16>
  }
  memref.copy %r, %dst : memref<32x64xf16> to memref<32x64xf16>
  return
}
