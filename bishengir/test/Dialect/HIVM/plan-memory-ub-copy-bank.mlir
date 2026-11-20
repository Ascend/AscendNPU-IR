// MEMBASED: error: UB copy bank planning is not supported on memory-based architectures
// RUN: split-file %s %t
// RUN: %if asserts %{ not bishengir-opt %t/bank-head-rollback.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=require -debug-only=hivm-plan-memory 2>&1 | FileCheck %t/bank-head-rollback.mlir --check-prefix=ROLLBACK %}
// RUN: bash -o pipefail -c 'for name in bank-bias bank-conflict bank-cycle-bad bank-isolated-multibuffer bank-multibuffer bank-different-slots bank-unique-special bank-special bank-view bank bank-reshape; do for mode in prefer require; do prefix=$(echo $mode | tr a-z A-Z); bishengir-opt %t/$name.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=$mode | FileCheck %t/$name.mlir --check-prefix=$prefix || exit 1; done; done'
// RUN: bash -o pipefail -c 'for name in bank-filter bank-reuse bank-special-first bank-special-reuse; do for mode in prefer require; do bishengir-opt %t/$name.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=$mode | FileCheck %t/$name.mlir || exit 1; done; done'
// RUN: bash -o pipefail -c 'for mode in prefer require; do for name in bank-shared-cycle bank-cycle-bad bank-conflict; do bishengir-opt %t/$name.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=$mode 2>&1 >/dev/null | FileCheck %t/$name.mlir --check-prefix=WARNING || exit 1; done; bishengir-opt %t/bank-self-bad.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=$mode | FileCheck %t/bank-self-bad.mlir --check-prefix=OK || exit 1; bishengir-opt %t/bank-level1-reuse.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=$mode | FileCheck %t/bank-level1-reuse.mlir --check-prefix=LEVEL1 || exit 1; done'
// RUN: bash -o pipefail -c 'for name in bank-bias-chain bank-level1-padded bank-level1-unaligned bank-cycle-good bank-self-good; do bishengir-opt %t/$name.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=require | FileCheck %t/$name.mlir || exit 1; done'
// RUN: bash -o pipefail -c 'for mode in prefer require; do not bishengir-opt %t/bank-fallback.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=$mode 2>&1 | FileCheck %t/bank-fallback.mlir --check-prefix=REQUIRE || exit 1; done'
// RUN: bishengir-opt %t/bank-anchor-late.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=prefer | FileCheck %t/bank-anchor-late.mlir --check-prefix=ON
// RUN: bishengir-opt %t/bank-anchor-late.mlir -hacc-append-device-spec=target=Ascend950PR_950z -hivm-plan-memory=plan-memory-ub-copy-bank-mode=require | FileCheck %t/bank-anchor-late.mlir --check-prefix=REQUIRE
// RUN: not bishengir-opt %t/bank.mlir -hacc-append-device-spec=target=Ascend910B1 -hivm-plan-memory=plan-memory-ub-copy-bank-mode=require 2>&1 | FileCheck %s --check-prefix=MEMBASED

//--- bank-bias.mlir
// The source subview is shifted by 128 bytes: the destination must have the
// same residue as the accessed address, not the source allocation base.
// PREFER-LABEL: func.func @static_bias
// PREFER-DAG: %[[PREF_DST:.*]] = arith.constant 640 : i64
// PREFER: %[[PREF_DST_ROOT:.*]] = hivm.hir.pointer_cast(%[[PREF_DST]])
// PREFER: %[[PREF_DST_VIEW:.*]] = memref.subview %[[PREF_DST_ROOT]]
// PREFER: hivm.hir.copy ins({{.*}}) outs(%[[PREF_DST_VIEW]] :
// REQUIRE-LABEL: func.func @static_bias
// REQUIRE-DAG: %[[REQ_DST:.*]] = arith.constant 640 : i64
// REQUIRE: %[[REQ_ROOT:.*]] = hivm.hir.pointer_cast(%[[REQ_DST]])
// REQUIRE: %[[REQ_VIEW:.*]] = memref.subview %[[REQ_ROOT]]
// REQUIRE: hivm.hir.copy ins({{.*}}) outs(%[[REQ_VIEW]] :
func.func @static_bias(%src: memref<128xf32, #hivm.address_space<gm>>,
                       %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                       %out0: memref<128xf32, #hivm.address_space<gm>>,
                       %out1: memref<128xf32, #hivm.address_space<gm>>,
                       %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%a : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  %as = memref.subview %a[32] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 32>, #hivm.address_space<ub>>
  %bs = memref.subview %b[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%as : memref<64xf32, strided<[1], offset: 32>, #hivm.address_space<ub>>) outs(%bs : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%out0 : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%out1 : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

// A destination bias of 32 bytes requires base residue -32 mod 256 = 224.
// PREFER-LABEL: func.func @destination_bias
// PREFER-DAG: %[[PREF_B:.*]] = arith.constant 480 : i64
// PREFER: %[[PREF_B_ROOT:.*]] = hivm.hir.pointer_cast(%[[PREF_B]])
// PREFER: %[[PREF_B_VIEW:.*]] = memref.subview %[[PREF_B_ROOT]]
// PREFER: hivm.hir.copy ins({{.*}}) outs(%[[PREF_B_VIEW]] :
// REQUIRE-LABEL: func.func @destination_bias
// REQUIRE-DAG: %[[REQ_B:.*]] = arith.constant 480 : i64
// REQUIRE: %[[REQ_B_ROOT:.*]] = hivm.hir.pointer_cast(%[[REQ_B]])
// REQUIRE: %[[REQ_B_VIEW:.*]] = memref.subview %[[REQ_B_ROOT]]
// REQUIRE: hivm.hir.copy ins({{.*}}) outs(%[[REQ_B_VIEW]] :
func.func @destination_bias(%src: memref<64xf32, #hivm.address_space<gm>>,
                            %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                            %dstSrc: memref<80xf32, #hivm.address_space<gm>>,
                            %out0: memref<64xf32, #hivm.address_space<gm>>,
                            %out1: memref<80xf32, #hivm.address_space<gm>>,
                            %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<80xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%dstSrc : memref<80xf32, #hivm.address_space<gm>>) outs(%b : memref<80xf32, #hivm.address_space<ub>>)
  %bs = memref.subview %b[8] [64] [1] : memref<80xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%bs : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<80xf32, #hivm.address_space<ub>>) outs(%out1 : memref<80xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-conflict.mlir
// The 512-byte copy requires equal residues; the 256-byte copy requires the
// destination residue to differ by 128 bytes. Prefer follows the larger copy.
// PREFER-LABEL: func.func @incompatible_copies
// PREFER-DAG: %[[P_SRC:.*]] = arith.constant 0 : i64
// PREFER-DAG: %[[P_DST:.*]] = arith.constant 768 : i64
// PREFER: %[[P_A:.*]] = hivm.hir.pointer_cast(%[[P_SRC]])
// PREFER: %[[P_B:.*]] = hivm.hir.pointer_cast(%[[P_DST]])
// PREFER: hivm.hir.copy ins(%[[P_A]] : {{.*}}) outs(%[[P_B]] :
// REQUIRE-LABEL: func.func @incompatible_copies
// REQUIRE-DAG: %[[R_SRC:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[R_DST:.*]] = arith.constant 768 : i64
// REQUIRE: %[[R_A:.*]] = hivm.hir.pointer_cast(%[[R_SRC]])
// REQUIRE: %[[R_B:.*]] = hivm.hir.pointer_cast(%[[R_DST]])
// REQUIRE: hivm.hir.copy ins(%[[R_A]] : {{.*}}) outs(%[[R_B]] :
// WARNING: warning: ignoring UB copy bank constraint in a cycle
func.func @incompatible_copies(%src: memref<128xf32, #hivm.address_space<gm>>,
                               %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                               %out0: memref<128xf32, #hivm.address_space<gm>>,
                               %out1: memref<128xf32, #hivm.address_space<gm>>,
                               %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%a : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%b : memref<128xf32, #hivm.address_space<ub>>)
  %as = memref.subview %a[64] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>>
  %bs = memref.subview %b[32] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 32>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%as : memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>>) outs(%bs : memref<64xf32, strided<[1], offset: 32>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%out0 : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%out1 : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-filter.mlir
// 128-byte copies do not meet the minimum transfer size. No bank padding is
// inserted even though the source and destination have different residues.
// CHECK-LABEL: func.func @short_copy
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[DST:.*]] = arith.constant 160 : i64
// CHECK: %[[A:.*]] = hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: %[[B:.*]] = hivm.hir.pointer_cast(%[[DST]])
// CHECK: hivm.hir.copy ins(%[[A]] : {{.*}}) outs(%[[B]] :
func.func @short_copy(%src: memref<32xf32, #hivm.address_space<gm>>,
                      %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                      %out0: memref<32xf32, #hivm.address_space<gm>>,
                      %out1: memref<32xf32, #hivm.address_space<gm>>,
                      %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<32xf32, #hivm.address_space<gm>>) outs(%a : memref<32xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<32xf32, #hivm.address_space<ub>>) outs(%b : memref<32xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<32xf32, #hivm.address_space<ub>>) outs(%out0 : memref<32xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<32xf32, #hivm.address_space<ub>>) outs(%out1 : memref<32xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

// A 2D copy is excluded even when its rows are contiguous.
// CHECK-LABEL: func.func @copy_2d
// CHECK-DAG: %[[C2D:.*]] = arith.constant 544 : i64
// CHECK: %[[C2D_DST:.*]] = hivm.hir.pointer_cast(%[[C2D]])
// CHECK: hivm.hir.copy ins({{.*}}) outs(%[[C2D_DST]] :
func.func @copy_2d(%src: memref<2x64xf32, #hivm.address_space<gm>>,
                   %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                   %out0: memref<2x64xf32, #hivm.address_space<gm>>,
                   %out1: memref<2x64xf32, #hivm.address_space<gm>>,
                   %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<2x64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<2x64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<2x64xf32, #hivm.address_space<gm>>) outs(%a : memref<2x64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<2x64xf32, #hivm.address_space<ub>>) outs(%b : memref<2x64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<2x64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<2x64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<2x64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<2x64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

// A dynamic-length 1D copy is also excluded.
// CHECK-LABEL: func.func @dynamic_copy
// CHECK-DAG: %[[CDYN:.*]] = arith.constant 544 : i64
// CHECK: %[[CDYN_DST:.*]] = hivm.hir.pointer_cast(%[[CDYN]])
// CHECK: %[[CDYN_VIEW:.*]] = memref.subview %[[CDYN_DST]]
// CHECK: hivm.hir.copy ins({{.*}}) outs(%[[CDYN_VIEW]] :
func.func @dynamic_copy(%src: memref<128xf32, #hivm.address_space<gm>>,
                        %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                        %out0: memref<128xf32, #hivm.address_space<gm>>,
                        %out1: memref<128xf32, #hivm.address_space<gm>>,
                        %padOut: memref<8xf32, #hivm.address_space<gm>>,
                        %n: index) {
  %a = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%a : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  %as = memref.subview %a[0] [%n] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
  %bs = memref.subview %b[0] [%n] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<?xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%as : memref<?xf32, #hivm.address_space<ub>>) outs(%bs : memref<?xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%out0 : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%out1 : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

// A valid affine layout that cannot be represented by constant strides must
// be skipped, rather than asserting in the bank copy collector.
// CHECK-LABEL: func.func @non_strided_layout
// CHECK-DAG: %[[CNON:.*]] = arith.constant 544 : i64
// CHECK: %[[CNON_DST:.*]] = hivm.hir.pointer_cast(%[[CNON]])
// CHECK: hivm.hir.copy ins({{.*}}) outs(%[[CNON_DST]] :
func.func @non_strided_layout(%src: memref<128xf32, #hivm.address_space<gm>>,
                              %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                              %out0: memref<128xf32, #hivm.address_space<gm>>,
                              %out1: memref<128xf32, #hivm.address_space<gm>>,
                              %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<128xf32, affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%a : memref<128xf32, affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<128xf32, affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>, #hivm.address_space<ub>>) outs(%b : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<128xf32, affine_map<(d0) -> (d0 floordiv 2, d0 mod 2)>, #hivm.address_space<ub>>) outs(%out0 : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%out1 : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

// A reinterpret_cast rebuilds descriptor metadata and is deliberately not
// traced. All modes therefore retain the same no-reuse addresses.
// CHECK-LABEL: func.func @reinterpret_cast_skipped
// CHECK-DAG: %[[RAW_ADDR:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[DST_ADDR:.*]] = arith.constant 288 : i64
// CHECK: %[[RAW:.*]] = hivm.hir.pointer_cast(%[[RAW_ADDR]])
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %[[RAW]]
// CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[DST_ADDR]])
// CHECK: hivm.hir.copy ins(%[[VIEW]] : {{.*}}) outs(%[[DST]] :
func.func @reinterpret_cast_skipped(%src: memref<72xf32, #hivm.address_space<gm>>, %out: memref<64xf32, #hivm.address_space<gm>>) {
  %raw = memref.alloc() : memref<72xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<72xf32, #hivm.address_space<gm>>) outs(%raw : memref<72xf32, #hivm.address_space<ub>>)
  %view = memref.reinterpret_cast %raw to offset: [8], sizes: [64], strides: [1] : memref<72xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%view : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>) outs(%dst : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%dst : memref<64xf32, #hivm.address_space<ub>>) outs(%out : memref<64xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-level1-reuse.mlir
// The first pair is multibuffered but has no UB copy edge. The later copy
// reuses all four slots in both modes.
// LEVEL1-LABEL: func.func @level1_bank_reuse
// LEVEL1: %[[FIRST:.*]] = hivm.hir.pointer_cast(%[[A0:[^, )]+]], %[[A1:[^, )]+]], %[[A2:[^, )]+]], %[[A3:[^, )]+]])
// LEVEL1: %[[SECOND:.*]] = hivm.hir.pointer_cast(%[[B0:[^, )]+]], %[[B1:[^, )]+]], %[[B2:[^, )]+]], %[[B3:[^, )]+]])
// LEVEL1: hivm.hir.pointer_cast(%[[B0]], %[[B1]], %[[B2]], %[[B3]])
// LEVEL1: hivm.hir.pointer_cast(%[[A0]], %[[A1]], %[[A2]], %[[A3]])
// LEVEL1: hivm.hir.copy ins(%[[SECOND]] : {{.*}}) outs(%[[FIRST]] :
func.func @level1_bank_reuse(%src1: memref<7168xf32, #hivm.address_space<gm>>,
                             %src2: memref<7168xf32, #hivm.address_space<gm>>,
                             %dst1: memref<7168xf32, #hivm.address_space<gm>>,
                             %dst2: memref<7168xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c4 {
    %a = memref.alloc() : memref<7168xf32, #hivm.address_space<ub>>
    annotation.mark %a {hivm.multi_buffer = 4 : i32} : memref<7168xf32, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<7168xf32, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<7168xf32, #hivm.address_space<ub>>
    annotation.mark %c {hivm.multi_buffer = 4 : i32} : memref<7168xf32, #hivm.address_space<ub>>
    %d = memref.alloc() : memref<7168xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src1 : memref<7168xf32, #hivm.address_space<gm>>) outs(%a : memref<7168xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%a, %a : memref<7168xf32, #hivm.address_space<ub>>, memref<7168xf32, #hivm.address_space<ub>>) outs(%c : memref<7168xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%c : memref<7168xf32, #hivm.address_space<ub>>) outs(%dst1 : memref<7168xf32, #hivm.address_space<gm>>)
    hivm.hir.load ins(%src2 : memref<7168xf32, #hivm.address_space<gm>>) outs(%b : memref<7168xf32, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%b : memref<7168xf32, #hivm.address_space<ub>>) outs(%d : memref<7168xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%d : memref<7168xf32, #hivm.address_space<ub>>) outs(%dst2 : memref<7168xf32, #hivm.address_space<gm>>)
  }
  return
}

//--- bank-level1-padded.mlir
// A 28800-byte buffer needs slot padding. The copy destination reuses all
// four slots of an isolated multibuffer; both operands have residue zero.
// CHECK-LABEL: func.func @level1_padded_reuse
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C28928:.*]] = arith.constant 28928 : i64
// CHECK-DAG: %[[C57856:.*]] = arith.constant 57856 : i64
// CHECK-DAG: %[[C86784:.*]] = arith.constant 86784 : i64
// CHECK-DAG: %[[C144384:.*]] = arith.constant 144384 : i64
// CHECK: hivm.hir.pointer_cast
// CHECK: %[[MULTI:.*]] = hivm.hir.pointer_cast(%[[C0]], %[[C28928]], %[[C57856]], %[[C86784]])
// CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[C144384]])
// CHECK: hivm.hir.pointer_cast(%[[C0]], %[[C28928]], %[[C57856]], %[[C86784]])
// CHECK: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[MULTI]] :
func.func @level1_padded_reuse(%src1: memref<7200xf32, #hivm.address_space<gm>>,
                             %src2: memref<7200xf32, #hivm.address_space<gm>>,
                             %dst1: memref<7200xf32, #hivm.address_space<gm>>,
                             %dst2: memref<7200xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c4 {
    %a = memref.alloc() : memref<7200xf32, #hivm.address_space<ub>>
    annotation.mark %a {hivm.multi_buffer = 4 : i32} : memref<7200xf32, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<7200xf32, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<7200xf32, #hivm.address_space<ub>>
    annotation.mark %c {hivm.multi_buffer = 4 : i32} : memref<7200xf32, #hivm.address_space<ub>>
    %d = memref.alloc() : memref<7200xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src1 : memref<7200xf32, #hivm.address_space<gm>>) outs(%a : memref<7200xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%a, %a : memref<7200xf32, #hivm.address_space<ub>>, memref<7200xf32, #hivm.address_space<ub>>) outs(%c : memref<7200xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%c : memref<7200xf32, #hivm.address_space<ub>>) outs(%dst1 : memref<7200xf32, #hivm.address_space<gm>>)
    hivm.hir.load ins(%src2 : memref<7200xf32, #hivm.address_space<gm>>) outs(%b : memref<7200xf32, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%b : memref<7200xf32, #hivm.address_space<ub>>) outs(%d : memref<7200xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%d : memref<7200xf32, #hivm.address_space<ub>>) outs(%dst2 : memref<7200xf32, #hivm.address_space<gm>>)
  }
  return
}

//--- bank-level1-unaligned.mlir
// Each 28704-byte slot needs padding. Both isolated multibuffers must have
// one residue per group, and the copy operands must share their residue.
// CHECK-LABEL: func.func @level1_unaligned_slots
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C28704:.*]] = arith.constant 28704 : i64
// CHECK-DAG: %[[C28928:.*]] = arith.constant 28928 : i64
// CHECK-DAG: %[[C57408:.*]] = arith.constant 57408 : i64
// CHECK-DAG: %[[C57632:.*]] = arith.constant 57632 : i64
// CHECK-DAG: %[[C57856:.*]] = arith.constant 57856 : i64
// CHECK-DAG: %[[C86560:.*]] = arith.constant 86560 : i64
// CHECK-DAG: %[[C86784:.*]] = arith.constant 86784 : i64
// CHECK-DAG: %[[C115488:.*]] = arith.constant 115488 : i64
// CHECK-DAG: %[[C144192:.*]] = arith.constant 144192 : i64
// CHECK: hivm.hir.pointer_cast(%[[C115488]], %[[C28704]], %[[C57632]], %[[C86560]])
// CHECK: hivm.hir.pointer_cast(%[[C0]], %[[C28928]], %[[C57856]], %[[C86784]])
// CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[C144192]])
// CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[C57408]])
// CHECK: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[DST]] :
func.func @level1_unaligned_slots(%src1: memref<7176xf32, #hivm.address_space<gm>>,
                             %src2: memref<7176xf32, #hivm.address_space<gm>>,
                             %dst1: memref<7176xf32, #hivm.address_space<gm>>,
                             %dst2: memref<7176xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c4 {
    %a = memref.alloc() : memref<7176xf32, #hivm.address_space<ub>>
    annotation.mark %a {hivm.multi_buffer = 4 : i32} : memref<7176xf32, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<7176xf32, #hivm.address_space<ub>>
    %c = memref.alloc() : memref<7176xf32, #hivm.address_space<ub>>
    annotation.mark %c {hivm.multi_buffer = 4 : i32} : memref<7176xf32, #hivm.address_space<ub>>
    %d = memref.alloc() : memref<7176xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src1 : memref<7176xf32, #hivm.address_space<gm>>) outs(%a : memref<7176xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%a, %a : memref<7176xf32, #hivm.address_space<ub>>, memref<7176xf32, #hivm.address_space<ub>>) outs(%c : memref<7176xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%c : memref<7176xf32, #hivm.address_space<ub>>) outs(%dst1 : memref<7176xf32, #hivm.address_space<gm>>)
    hivm.hir.load ins(%src2 : memref<7176xf32, #hivm.address_space<gm>>) outs(%b : memref<7176xf32, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%b : memref<7176xf32, #hivm.address_space<ub>>) outs(%d : memref<7176xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%d : memref<7176xf32, #hivm.address_space<ub>>) outs(%dst2 : memref<7176xf32, #hivm.address_space<gm>>)
  }
  return
}

//--- bank-isolated-multibuffer.mlir
// Even without a copy edge, REQUIRE must align every slot for later reuse.
// PREFER-LABEL: func.func @isolated_multibuffer
// PREFER-DAG: %[[O0:.*]] = arith.constant 0 : i64
// PREFER-DAG: %[[O288:.*]] = arith.constant 288 : i64
// PREFER: hivm.hir.pointer_cast(%[[O0]], %[[O288]])
// REQUIRE-LABEL: func.func @isolated_multibuffer
// REQUIRE-DAG: %[[R0:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[R512:.*]] = arith.constant 512 : i64
// REQUIRE: hivm.hir.pointer_cast(%[[R0]], %[[R512]])
func.func @isolated_multibuffer(%src: memref<72xf32, #hivm.address_space<gm>>,
                               %dst: memref<72xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %a = memref.alloc() : memref<72xf32, #hivm.address_space<ub>>
    annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<72xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<72xf32, #hivm.address_space<gm>>) outs(%a : memref<72xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%a : memref<72xf32, #hivm.address_space<ub>>) outs(%dst : memref<72xf32, #hivm.address_space<gm>>)
  }
  return
}

//--- bank-multibuffer.mlir
// Both multibuffer slots and the normal peer have the same residue.
// Two compatible copies share the same special peer.
// PREFER-LABEL: func.func @multi_anchor
// PREFER-DAG: %[[FIRST:.*]] = arith.constant 0 : i64
// PREFER-DAG: %[[OTHER:.*]] = arith.constant 256 : i64
// PREFER-DAG: %[[NORMAL:.*]] = arith.constant 768 : i64
// PREFER: %[[MULTI:.*]] = hivm.hir.pointer_cast(%[[FIRST]], %[[OTHER]])
// PREFER: %[[DST:.*]] = hivm.hir.pointer_cast(%[[NORMAL]])
// PREFER: hivm.hir.copy ins(%[[MULTI]] : {{.*}}) outs(%[[DST]] :
// PREFER: hivm.hir.copy ins(%[[MULTI]] : {{.*}}) outs(%[[DST]] :
// REQUIRE-LABEL: func.func @multi_anchor
// REQUIRE-DAG: %[[R_FIRST:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[R_OTHER:.*]] = arith.constant 256 : i64
// REQUIRE-DAG: %[[R_NORMAL:.*]] = arith.constant 768 : i64
// REQUIRE: %[[R_MULTI:.*]] = hivm.hir.pointer_cast(%[[R_FIRST]], %[[R_OTHER]])
// REQUIRE: %[[R_DST:.*]] = hivm.hir.pointer_cast(%[[R_NORMAL]])
// REQUIRE: hivm.hir.copy ins(%[[R_MULTI]] : {{.*}}) outs(%[[R_DST]] :
// REQUIRE: hivm.hir.copy ins(%[[R_MULTI]] : {{.*}}) outs(%[[R_DST]] :
func.func @multi_anchor(%src: memref<64xf32, #hivm.address_space<gm>>,
                        %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                        %postSrc: memref<8xf32, #hivm.address_space<gm>>,
                        %out0: memref<64xf32, #hivm.address_space<gm>>,
                        %out1: memref<64xf32, #hivm.address_space<gm>>,
                        %padOut: memref<8xf32, #hivm.address_space<gm>>,
                        %postOut: memref<8xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
    %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    %post = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%postSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%post : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%post : memref<8xf32, #hivm.address_space<ub>>) outs(%postOut : memref<8xf32, #hivm.address_space<gm>>)
  }
  return
}

//--- bank-different-slots.mlir
// 288-byte slots necessarily have different residues when laid out adjacent.
// PREFER-LABEL: func.func @different_slot_residues
// REQUIRE-DAG: %[[S0:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[S1:.*]] = arith.constant 512 : i64
// REQUIRE-DAG: %[[D:.*]] = arith.constant 1024 : i64
// REQUIRE: %[[MULTI:.*]] = hivm.hir.pointer_cast(%[[S0]], %[[S1]])
// REQUIRE: %[[DST:.*]] = hivm.hir.pointer_cast(%[[D]])
// REQUIRE: %[[VIEW:.*]] = memref.subview %[[MULTI]]
// REQUIRE: hivm.hir.copy ins(%[[VIEW]] : {{.*}}) outs(%[[DST]] :
func.func @different_slot_residues(%src: memref<72xf32, #hivm.address_space<gm>>,
                        %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                        %postSrc: memref<8xf32, #hivm.address_space<gm>>,
                        %out0: memref<72xf32, #hivm.address_space<gm>>,
                        %out1: memref<64xf32, #hivm.address_space<gm>>,
                        %padOut: memref<8xf32, #hivm.address_space<gm>>,
                        %postOut: memref<8xf32, #hivm.address_space<gm>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %a = memref.alloc() : memref<72xf32, #hivm.address_space<ub>>
    annotation.mark %a {hivm.multi_buffer = 2 : i32} : memref<72xf32, #hivm.address_space<ub>>
    %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    %post = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%src : memref<72xf32, #hivm.address_space<gm>>) outs(%a : memref<72xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
    %view = memref.subview %a[0] [64] [1] : memref<72xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%view : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%postSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%post : memref<8xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%a : memref<72xf32, #hivm.address_space<ub>>) outs(%out0 : memref<72xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
    hivm.hir.store ins(%post : memref<8xf32, #hivm.address_space<ub>>) outs(%postOut : memref<8xf32, #hivm.address_space<gm>>)
  }
  return
}


//--- bank-reuse.mlir
// No-reuse exceeds UB capacity. The first buffer is dead before the copy,
// so the allocator must enter reuse planning while keeping the copy aligned.
// CHECK-LABEL: func.func @reuse_and_align
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[COPY_DST:.*]] = arith.constant 16384 : i64
// CHECK: %[[TMP:.*]] = hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[COPY_DST]])
// CHECK: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[DST]] :
func.func @reuse_and_align(%large: memref<60000xf32, #hivm.address_space<gm>>,
                           %largeOut: memref<60000xf32, #hivm.address_space<gm>>,
                           %src: memref<4096xf32, #hivm.address_space<gm>>,
                           %out0: memref<4096xf32, #hivm.address_space<gm>>,
                           %out1: memref<4096xf32, #hivm.address_space<gm>>) {
  %tmp = memref.alloc() : memref<60000xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%large : memref<60000xf32, #hivm.address_space<gm>>) outs(%tmp : memref<60000xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%tmp : memref<60000xf32, #hivm.address_space<ub>>) outs(%largeOut : memref<60000xf32, #hivm.address_space<gm>>)
  %a = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<4096xf32, #hivm.address_space<gm>>) outs(%a : memref<4096xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<4096xf32, #hivm.address_space<ub>>) outs(%b : memref<4096xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<4096xf32, #hivm.address_space<ub>>) outs(%out0 : memref<4096xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<4096xf32, #hivm.address_space<ub>>) outs(%out1 : memref<4096xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-unique-special.mlir
// A memory-unique endpoint excludes the copy from bank planning. The special
// view has a 32-byte bias, so the two access residues deliberately differ.
// PREFER-LABEL: func.func @unique_normal_after_special
// PREFER-DAG: %[[P_ZERO:.*]] = arith.constant 0 : i64
// PREFER-DAG: %[[P_BANK:.*]] = arith.constant 320 : i64
// PREFER: %[[P_NORMAL:.*]] = hivm.hir.pointer_cast(%[[P_BANK]])
// PREFER: %[[P_SPECIAL:.*]] = hivm.hir.pointer_cast(%[[P_ZERO]])
// PREFER: %[[P_VIEW:.*]] = memref.subview %[[P_SPECIAL]][8]
// PREFER: hivm.hir.copy ins(%[[P_VIEW]] : {{.*}}) outs(%[[P_NORMAL]] :
// REQUIRE-LABEL: func.func @unique_normal_after_special
// REQUIRE-DAG: %[[R_ZERO:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[R_BANK:.*]] = arith.constant 320 : i64
// REQUIRE: %[[R_NORMAL:.*]] = hivm.hir.pointer_cast(%[[R_BANK]])
// REQUIRE: %[[R_SPECIAL:.*]] = hivm.hir.pointer_cast(%[[R_ZERO]])
// REQUIRE: %[[R_VIEW:.*]] = memref.subview %[[R_SPECIAL]][8]
// REQUIRE: hivm.hir.copy ins(%[[R_VIEW]] : {{.*}}) outs(%[[R_NORMAL]] :
func.func @unique_normal_after_special(%src: memref<80xf32, #hivm.address_space<gm>>,
                                       %out: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %normal = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  annotation.mark %normal {mem_unique} : memref<64xf32, #hivm.address_space<ub>>
  %special = memref.alloc() : memref<80xf32, #hivm.address_space<ub>>
  annotation.mark %special {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<80xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<80xf32, #hivm.address_space<gm>>) outs(%special : memref<80xf32, #hivm.address_space<ub>>)
  %view = memref.subview %special[8] [64] [1] : memref<80xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%view : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>) outs(%normal : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%normal : memref<64xf32, #hivm.address_space<ub>>) outs(%out : memref<64xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-special-first.mlir
// The large special entry is dead before the second copy, but the bank-aware
// normal entry must not reuse its address. Other compatible space is available.
// CHECK-LABEL: func.func @normal_after_special
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C253216:.*]] = arith.constant 253216 : i64
// CHECK-DAG: %[[C253472:.*]] = arith.constant 253472 : i64
// CHECK-DAG: %[[C253696:.*]] = arith.constant 253696 : i64
// CHECK: %[[T:.*]] = hivm.hir.pointer_cast(%[[C0]])
// CHECK: %[[X:.*]] = hivm.hir.pointer_cast(%[[C253696]])
// CHECK: hivm.hir.copy
// CHECK: %[[S:.*]] = hivm.hir.pointer_cast(%[[C253216]])
// CHECK: %[[N:.*]] = hivm.hir.pointer_cast(%[[C253472]])
// CHECK: hivm.hir.copy ins(%[[S]] : {{.*}}) outs(%[[N]] :
func.func @normal_after_special(%large: memref<63300xf32, #hivm.address_space<gm>>,
                                %small: memref<64xf32, #hivm.address_space<gm>>,
                                %out0: memref<64xf32, #hivm.address_space<gm>>,
                                %out1: memref<64xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %t = memref.alloc() : memref<63300xf32, #hivm.address_space<ub>>
  annotation.mark %t {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<63300xf32, #hivm.address_space<ub>>
  %x = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%large : memref<63300xf32, #hivm.address_space<gm>>) outs(%t : memref<63300xf32, #hivm.address_space<ub>>)
  %ts = memref.subview %t[0] [64] [1] : memref<63300xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%ts : memref<64xf32, #hivm.address_space<ub>>) outs(%x : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%x : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  %s = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  annotation.mark %s {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<64xf32, #hivm.address_space<ub>>
  %n = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%small : memref<64xf32, #hivm.address_space<gm>>) outs(%s : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%s : memref<64xf32, #hivm.address_space<ub>>) outs(%n : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%n : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-special-reuse.mlir
// Current policy checks special addresses only when a normal entry is being
// planned. Here the bank-connected normals are planned before the unrelated
// TCB, so the later TCB can reuse the normal's address.
// CHECK-LABEL: func.func @special_after_normal
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C16384:.*]] = arith.constant 16384 : i64
// CHECK: %[[TCB:.*]] = hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: %[[SRC:.*]] = hivm.hir.pointer_cast(%[[ZERO]])
// CHECK: %[[DST:.*]] = hivm.hir.pointer_cast(%[[C16384]])
// CHECK: hivm.hir.copy ins(%[[SRC]] : {{.*}}) outs(%[[DST]] :
func.func @special_after_normal(%large: memref<60000xf32, #hivm.address_space<gm>>,
                           %largeOut: memref<60000xf32, #hivm.address_space<gm>>,
                           %src: memref<4096xf32, #hivm.address_space<gm>>,
                           %out0: memref<4096xf32, #hivm.address_space<gm>>,
                           %out1: memref<4096xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %tmp = memref.alloc() : memref<60000xf32, #hivm.address_space<ub>>
  annotation.mark %tmp {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<60000xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%large : memref<60000xf32, #hivm.address_space<gm>>) outs(%tmp : memref<60000xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%tmp : memref<60000xf32, #hivm.address_space<ub>>) outs(%largeOut : memref<60000xf32, #hivm.address_space<gm>>)
  %a = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<4096xf32, #hivm.address_space<gm>>) outs(%a : memref<4096xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<4096xf32, #hivm.address_space<ub>>) outs(%b : memref<4096xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<4096xf32, #hivm.address_space<ub>>) outs(%out0 : memref<4096xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<4096xf32, #hivm.address_space<ub>>) outs(%out1 : memref<4096xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-special.mlir
// Bank-aware ordering places the two copy operands at equal residues.
// PREFER-LABEL: func.func @bank_tcb_anchor
// PREFER-DAG: %[[P0:.*]] = arith.constant 0 : i64
// PREFER-DAG: %[[P256:.*]] = arith.constant 256 : i64
// PREFER-DAG: %[[P512:.*]] = arith.constant 512 : i64
// PREFER: %[[PA:.*]] = hivm.hir.pointer_cast(%[[P0]])
// PREFER: %[[PP:.*]] = hivm.hir.pointer_cast(%[[P256]])
// PREFER: %[[PB:.*]] = hivm.hir.pointer_cast(%[[P512]])
// PREFER: hivm.hir.copy ins(%[[PA]] : {{.*}}) outs(%[[PB]] :
// REQUIRE-LABEL: func.func @bank_tcb_anchor
// REQUIRE-DAG: %[[R0:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[R512:.*]] = arith.constant 512 : i64
// REQUIRE: %[[RA:.*]] = hivm.hir.pointer_cast(%[[R0]])
// REQUIRE: %[[RB:.*]] = hivm.hir.pointer_cast(%[[R512]])
// REQUIRE: hivm.hir.copy ins(%[[RA]] : {{.*}}) outs(%[[RB]] :
func.func @bank_tcb_anchor(%src: memref<64xf32, #hivm.address_space<gm>>,
                        %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                        %out0: memref<64xf32, #hivm.address_space<gm>>,
                        %out1: memref<64xf32, #hivm.address_space<gm>>,
                        %padOut: memref<8xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  annotation.mark %a {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-view.mlir
// memref.view has identity layout but a nonzero byte shift. The actual copy
// starts 32 bytes after the allocation, so the destination has residue 32.
// PREFER-LABEL: func.func @view_bias
// PREFER-DAG: %[[PREF_B:.*]] = arith.constant 544 : i64
// PREFER: %[[PREF_B_ROOT:.*]] = hivm.hir.pointer_cast(%[[PREF_B]])
// PREFER: hivm.hir.copy ins({{.*}}) outs(%[[PREF_B_ROOT]] :
// REQUIRE-LABEL: func.func @view_bias
// REQUIRE-DAG: %[[REQ_B:.*]] = arith.constant 544 : i64
// REQUIRE: %[[REQ_B_ROOT:.*]] = hivm.hir.pointer_cast(%[[REQ_B]])
// REQUIRE: hivm.hir.copy ins({{.*}}) outs(%[[REQ_B_ROOT]] :
func.func @view_bias(%src: memref<64xf32, #hivm.address_space<gm>>,
                     %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                     %out0: memref<64xf32, #hivm.address_space<gm>>,
                     %out1: memref<64xf32, #hivm.address_space<gm>>,
                     %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %c32 = arith.constant 32 : index
  %raw = memref.alloc() : memref<288xi8, #hivm.address_space<ub>>
  %a = memref.view %raw[%c32][] : memref<288xi8, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}


// A 32-byte view shift plus an 8-element f32 subview offset gives 64 bytes.
// PREFER-LABEL: func.func @view_subview_bias
// PREFER-DAG: %[[PREF_COMBINED:.*]] = arith.constant 576 : i64
// PREFER: %[[PREF_COMBINED_ROOT:.*]] = hivm.hir.pointer_cast(%[[PREF_COMBINED]])
// PREFER: hivm.hir.copy ins({{.*}}) outs(%[[PREF_COMBINED_ROOT]] :
// REQUIRE-LABEL: func.func @view_subview_bias
// REQUIRE-DAG: %[[REQ_COMBINED:.*]] = arith.constant 576 : i64
// REQUIRE: %[[REQ_COMBINED_ROOT:.*]] = hivm.hir.pointer_cast(%[[REQ_COMBINED]])
// REQUIRE: hivm.hir.copy ins({{.*}}) outs(%[[REQ_COMBINED_ROOT]] :
func.func @view_subview_bias(%src: memref<64xf32, #hivm.address_space<gm>>,
                     %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                     %out0: memref<64xf32, #hivm.address_space<gm>>,
                     %out1: memref<64xf32, #hivm.address_space<gm>>,
                     %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %c32 = arith.constant 32 : index
  %raw = memref.alloc() : memref<320xi8, #hivm.address_space<ub>>
  %view = memref.view %raw[%c32][] : memref<320xi8, #hivm.address_space<ub>> to memref<72xf32, #hivm.address_space<ub>>
  %a = memref.subview %view[8] [64] [1] : memref<72xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

// An unknown view byte shift cannot be used to derive a static bank residue.
// PREFER-LABEL: func.func @dynamic_view_bias
// PREFER-DAG: %[[PREF_DYNAMIC:.*]] = arith.constant 320 : i64
// PREFER: %[[PREF_DYNAMIC_ROOT:.*]] = hivm.hir.pointer_cast(%[[PREF_DYNAMIC]])
// PREFER: hivm.hir.copy ins({{.*}}) outs(%[[PREF_DYNAMIC_ROOT]] :
// REQUIRE-LABEL: func.func @dynamic_view_bias
// REQUIRE-DAG: %[[REQ_DYNAMIC:.*]] = arith.constant 320 : i64
// REQUIRE: %[[REQ_DYNAMIC_ROOT:.*]] = hivm.hir.pointer_cast(%[[REQ_DYNAMIC]])
// REQUIRE: hivm.hir.copy ins({{.*}}) outs(%[[REQ_DYNAMIC_ROOT]] :
func.func @dynamic_view_bias(%src: memref<64xf32, #hivm.address_space<gm>>,
                             %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                             %out0: memref<64xf32, #hivm.address_space<gm>>,
                             %out1: memref<64xf32, #hivm.address_space<gm>>,
                             %padOut: memref<8xf32, #hivm.address_space<gm>>,
                             %shift: index) {
  %raw = memref.alloc() : memref<288xi8, #hivm.address_space<ub>>
  %a = memref.view %raw[%shift][] : memref<288xi8, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

//--- bank.mlir
// OFF places the destination at 288 bytes. Bank-aware ordering gives
// the copy operands equal residues in both planning modes.
// PREFER-LABEL: func.func @bank_padding
// PREFER-DAG: %[[P0:.*]] = arith.constant 0 : i64
// PREFER-DAG: %[[P256:.*]] = arith.constant 256 : i64
// PREFER-DAG: %[[P512:.*]] = arith.constant 512 : i64
// PREFER: %[[PA:.*]] = hivm.hir.pointer_cast(%[[P0]])
// PREFER: %[[PP:.*]] = hivm.hir.pointer_cast(%[[P256]])
// PREFER: %[[PB:.*]] = hivm.hir.pointer_cast(%[[P512]])
// PREFER: hivm.hir.copy ins(%[[PA]] : {{.*}}) outs(%[[PB]] :
// REQUIRE-LABEL: func.func @bank_padding
// REQUIRE-DAG: %[[R0:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[R512:.*]] = arith.constant 512 : i64
// REQUIRE: %[[RA:.*]] = hivm.hir.pointer_cast(%[[R0]])
// REQUIRE: %[[RB:.*]] = hivm.hir.pointer_cast(%[[R512]])
// REQUIRE: hivm.hir.copy ins(%[[RA]] : {{.*}}) outs(%[[RB]] :
func.func @bank_padding(%src: memref<64xf32, #hivm.address_space<gm>>,
                        %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                        %out0: memref<64xf32, #hivm.address_space<gm>>,
                        %out1: memref<64xf32, #hivm.address_space<gm>>,
                        %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-self-bad.mlir
// OK-LABEL: func.func @self_edge
// OK: hivm.hir.copy
func.func @self_edge(%src: memref<256xf32, #hivm.address_space<gm>>, %dst: memref<256xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<256xf32, #hivm.address_space<gm>>) outs(%a : memref<256xf32, #hivm.address_space<ub>>)
  %a0 = memref.subview %a[0] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  %a1 = memref.subview %a[72] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 72>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%a0 : memref<64xf32, #hivm.address_space<ub>>) outs(%a1 : memref<64xf32, strided<[1], offset: 72>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<256xf32, #hivm.address_space<ub>>) outs(%dst : memref<256xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-self-good.mlir
// CHECK-LABEL: func.func @self_edge_aligned
// CHECK: hivm.hir.copy
func.func @self_edge_aligned(%src: memref<256xf32, #hivm.address_space<gm>>, %dst: memref<256xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<256xf32, #hivm.address_space<gm>>) outs(%a : memref<256xf32, #hivm.address_space<ub>>)
  %a0 = memref.subview %a[0] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  %a1 = memref.subview %a[64] [64] [1] : memref<256xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%a0 : memref<64xf32, #hivm.address_space<ub>>) outs(%a1 : memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<256xf32, #hivm.address_space<ub>>) outs(%dst : memref<256xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-reshape.mlir
// OFF places the destination at 288 bytes. Bank-aware ordering gives
// the copy operands equal residues in both planning modes.
// PREFER-LABEL: func.func @bank_reshape
// PREFER-DAG: %[[P0:.*]] = arith.constant 0 : i64
// PREFER-DAG: %[[P256:.*]] = arith.constant 256 : i64
// PREFER-DAG: %[[P512:.*]] = arith.constant 512 : i64
// PREFER: %[[PA:.*]] = hivm.hir.pointer_cast(%[[P0]])
// PREFER: %[[PP:.*]] = hivm.hir.pointer_cast(%[[P256]])
// PREFER: %[[PB:.*]] = hivm.hir.pointer_cast(%[[P512]])
// PREFER: %[[PA_RESHAPE:.*]] = memref.reshape %[[PA]]
// PREFER: hivm.hir.copy ins(%[[PA_RESHAPE]] : {{.*}}) outs(%[[PB]] :
// REQUIRE-LABEL: func.func @bank_reshape
// REQUIRE-DAG: %[[R0:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[R512:.*]] = arith.constant 512 : i64
// REQUIRE: %[[RA:.*]] = hivm.hir.pointer_cast(%[[R0]])
// REQUIRE: %[[RB:.*]] = hivm.hir.pointer_cast(%[[R512]])
// REQUIRE: %[[RA_RESHAPE:.*]] = memref.reshape %[[RA]]
// REQUIRE: hivm.hir.copy ins(%[[RA_RESHAPE]] : {{.*}}) outs(%[[RB]] :
func.func @bank_reshape(%src: memref<64xf32, #hivm.address_space<gm>>,
                        %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                        %out0: memref<64xf32, #hivm.address_space<gm>>,
                        %out1: memref<64xf32, #hivm.address_space<gm>>,
                        %padOut: memref<8xf32, #hivm.address_space<gm>>,
                        %shape: memref<1xindex>) {
  %a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  %reshaped = memref.reshape %a(%shape) : (memref<64xf32, #hivm.address_space<ub>>, memref<1xindex>) -> memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%reshaped : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-fallback.mlir
// Padding the destination to match its 32-byte access bias exceeds UB.
// REQUIRE: error: ub overflow
func.func @bank_fallback(%src: memref<64xf32, #hivm.address_space<gm>>,
                         %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                         %out0: memref<64xf32, #hivm.address_space<gm>>,
                         %out1: memref<63416xf32, #hivm.address_space<gm>>,
                         %padOut: memref<8xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<63416xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  %bs = memref.subview %b[8] [64] [1] : memref<63416xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%bs : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<63416xf32, #hivm.address_space<ub>>) outs(%out1 : memref<63416xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-anchor-late.mlir
// The normal is encountered before its TCB peer in the no-reuse layout.
// Bank-aware planning must not accept the unknown anchor as unconstrained.
// ON-LABEL: func.func @late_tcb_anchor
// ON-DAG: %[[C0:.*]] = arith.constant 0 : i64
// ON-DAG: %[[C256:.*]] = arith.constant 256 : i64
// ON: %[[N:.*]] = hivm.hir.pointer_cast(%[[C256]])
// ON: %[[T:.*]] = hivm.hir.pointer_cast(%[[C0]])
// ON: hivm.hir.copy ins(%[[N]] : {{.*}}) outs(%[[T]] :
// REQUIRE-LABEL: func.func @late_tcb_anchor
// REQUIRE-DAG: %[[R512:.*]] = arith.constant 512 : i64
// REQUIRE-DAG: %[[R0:.*]] = arith.constant 0 : i64
// REQUIRE: %[[RN:.*]] = hivm.hir.pointer_cast(%[[R0]])
// REQUIRE: %[[RT:.*]] = hivm.hir.pointer_cast(%[[R512]])
// REQUIRE: hivm.hir.copy ins(%[[RN]] : {{.*}}) outs(%[[RT]] :
func.func @late_tcb_anchor(%src: memref<64xf32, #hivm.address_space<gm>>,
                           %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                           %out0: memref<64xf32, #hivm.address_space<gm>>,
                           %out1: memref<64xf32, #hivm.address_space<gm>>,
                           %padOut: memref<8xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %n = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%n : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  %t = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  annotation.mark %t {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%n : memref<64xf32, #hivm.address_space<ub>>) outs(%t : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%n : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%t : memref<64xf32, #hivm.address_space<ub>>) outs(%out1 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-bias-chain.mlir
// The first copy has a 128-byte source bias; the second has a 32-byte
// source bias. Their transitive component must preserve both offsets.
// CHECK-LABEL: func.func @bank_bias_chain
// CHECK-DAG: %[[A_OFFSET:.*]] = arith.constant 640 : i64
// CHECK-DAG: %[[B_OFFSET:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[C_OFFSET:.*]] = arith.constant 1312 : i64
// CHECK: %[[A:.*]] = hivm.hir.pointer_cast(%[[A_OFFSET]])
// CHECK: %[[B:.*]] = hivm.hir.pointer_cast(%[[B_OFFSET]])
// CHECK: %[[C:.*]] = hivm.hir.pointer_cast(%[[C_OFFSET]])
// CHECK: %[[AV:.*]] = memref.subview %[[A]][32]
// CHECK: %[[B0:.*]] = memref.subview %[[B]][0]
// CHECK: %[[B8:.*]] = memref.subview %[[B]][8]
// CHECK: %[[C0:.*]] = memref.subview %[[C]][0]
// CHECK: hivm.hir.copy ins(%[[AV]] : {{.*}}) outs(%[[B0]] :
// CHECK: hivm.hir.copy ins(%[[B8]] : {{.*}}) outs(%[[C0]] :
func.func @bank_bias_chain(%src: memref<128xf32, #hivm.address_space<gm>>,
                           %out: memref<128xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  %c = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%a : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%b : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%c : memref<128xf32, #hivm.address_space<ub>>)
  %av = memref.subview %a[32] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 32>, #hivm.address_space<ub>>
  %b0 = memref.subview %b[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  %b8 = memref.subview %b[8] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>
  %c0 = memref.subview %c[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%av : memref<64xf32, strided<[1], offset: 32>, #hivm.address_space<ub>>) outs(%b0 : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%b8 : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>) outs(%c0 : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%c : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-cycle-good.mlir
// Three distinct entries form a cycle.
// CHECK-LABEL: func.func @cycle_good
func.func @cycle_good(%src: memref<128xf32, #hivm.address_space<gm>>, %out: memref<128xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%a : memref<128xf32, #hivm.address_space<ub>>)
  %b = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%b : memref<128xf32, #hivm.address_space<ub>>)
  %c = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%c : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%b : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%c : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%c : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%c : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-cycle-bad.mlir
// Three distinct entries form a cycle.
// PREFER-LABEL: func.func @cycle_bad
// PREFER-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// PREFER-DAG: %[[B512:.*]] = arith.constant 512 : i64
// PREFER-DAG: %[[C1024:.*]] = arith.constant 1024 : i64
// PREFER: %[[A:.*]] = hivm.hir.pointer_cast(%[[ZERO]])
// PREFER: %[[B:.*]] = hivm.hir.pointer_cast(%[[B512]])
// PREFER: %[[C:.*]] = hivm.hir.pointer_cast(%[[C1024]])
// PREFER: hivm.hir.copy ins(%[[A]] : {{.*}}) outs(%[[B]] :
// PREFER: hivm.hir.copy ins(%[[B]] : {{.*}}) outs(%[[C]] :
// REQUIRE-LABEL: func.func @cycle_bad
// REQUIRE-DAG: %[[R_ZERO:.*]] = arith.constant 0 : i64
// REQUIRE-DAG: %[[R_B512:.*]] = arith.constant 512 : i64
// REQUIRE-DAG: %[[R_C1024:.*]] = arith.constant 1024 : i64
// REQUIRE: %[[R_A:.*]] = hivm.hir.pointer_cast(%[[R_B512]])
// REQUIRE: %[[R_B:.*]] = hivm.hir.pointer_cast(%[[R_ZERO]])
// REQUIRE: %[[R_C:.*]] = hivm.hir.pointer_cast(%[[R_C1024]])
// REQUIRE: hivm.hir.copy ins(%[[R_A]] : {{.*}}) outs(%[[R_B]] :
// REQUIRE: hivm.hir.copy ins(%[[R_B]] : {{.*}}) outs(%[[R_C]] :
// WARNING: warning: ignoring UB copy bank constraint in a cycle
func.func @cycle_bad(%src: memref<128xf32, #hivm.address_space<gm>>, %out: memref<128xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%a : memref<128xf32, #hivm.address_space<ub>>)
  %b = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%b : memref<128xf32, #hivm.address_space<ub>>)
  %c = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<128xf32, #hivm.address_space<gm>>) outs(%c : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%b : memref<128xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%c : memref<128xf32, #hivm.address_space<ub>>)
  %av = memref.subview %a[32] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 32>, #hivm.address_space<ub>>
  %cv = memref.subview %c[0] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%av : memref<64xf32, strided<[1], offset: 32>, #hivm.address_space<ub>>) outs(%cv : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%c : memref<128xf32, #hivm.address_space<ub>>) outs(%out : memref<128xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-shared-cycle.mlir
// Two triangles share A-B; removing it leaves an outer cycle to prune.
// WARNING: warning: ignoring UB copy bank constraint
// WARNING-NEXT: hivm.hir.copy ins(%a : {{.*}}) outs(%b :
// WARNING: warning: ignoring UB copy bank constraint
// WARNING-NEXT: hivm.hir.copy ins(%d : {{.*}}) outs(%b :
// WARNING-NOT: warning: ignoring UB copy bank constraint
func.func @shared_cycle(%src: memref<64xf32, #hivm.address_space<gm>>,
                        %out: memref<64xf32, #hivm.address_space<gm>>) {
  %a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %c = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %d = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%c : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%c : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%d : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%d : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%b : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.store ins(%b : memref<64xf32, #hivm.address_space<ub>>) outs(%out : memref<64xf32, #hivm.address_space<gm>>)
  return
}

//--- bank-head-rollback.mlir
// Force a headed allocation to roll back and be allocated again.
// ROLLBACK: [UpdateOutline] split head, entry=[[ENTRY:[0-9]+]]
// ROLLBACK: [RollbackOutline] restore head, entry=[[ENTRY]]
// ROLLBACK: [PlanLocal] ApplyFailStrategy -> CONTINUE_PLAN
// ROLLBACK: [UpdateOutline] split head, entry=[[ENTRY]]
// ROLLBACK-NEXT: [MultiSpecPlan] SUCCESS at level=2 offsetByte=736
// ROLLBACK: error: ub overflow
func.func @bank_head_rollback(%src: memref<64xf32, #hivm.address_space<gm>>,
                         %padSrc: memref<8xf32, #hivm.address_space<gm>>,
                         %out0: memref<64xf32, #hivm.address_space<gm>>,
                         %out1: memref<80xf32, #hivm.address_space<gm>>,
                         %padOut: memref<8xf32, #hivm.address_space<gm>>, %v: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %huge = memref.alloc() : memref<63300xf32, #hivm.address_space<ub>>
  memref.store %v, %huge[%c0] : memref<63300xf32, #hivm.address_space<ub>>
  %a = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
  %pad = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  %b = memref.alloc() : memref<80xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<64xf32, #hivm.address_space<gm>>) outs(%a : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.load ins(%padSrc : memref<8xf32, #hivm.address_space<gm>>) outs(%pad : memref<8xf32, #hivm.address_space<ub>>)
  %bs = memref.subview %b[8] [64] [1] : memref<80xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>
  hivm.hir.copy ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%bs : memref<64xf32, strided<[1], offset: 8>, #hivm.address_space<ub>>)
  hivm.hir.store ins(%a : memref<64xf32, #hivm.address_space<ub>>) outs(%out0 : memref<64xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%b : memref<80xf32, #hivm.address_space<ub>>) outs(%out1 : memref<80xf32, #hivm.address_space<gm>>)
  hivm.hir.store ins(%pad : memref<8xf32, #hivm.address_space<ub>>) outs(%padOut : memref<8xf32, #hivm.address_space<gm>>)
  %r = memref.load %huge[%c0] : memref<63300xf32, #hivm.address_space<ub>>
  return %r : f32
}
