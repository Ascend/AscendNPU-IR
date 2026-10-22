// RUN: split-file %s %t
// RUN: bishengir-opt %t/a5.mlir -convert-hivm-to-std | FileCheck %s --check-prefix=STD --implicit-check-not=func.call
// RUN: bishengir-opt %t/a5.mlir -convert-hivm-to-llvm -verify-each | FileCheck %t/a5.mlir
// RUN: bishengir-opt %t/a5.mlir -convert-hivm-to-std -convert-hivm-to-llvm | mlir-translate --allow-unregistered-dialect --mlir-to-llvmir | FileCheck %t/a5.mlir --check-prefix=LLVM
// RUN: bishengir-opt %t/a5.mlir -convert-hivm-to-std -convert-hivm-to-llvm | mlir-translate --allow-unregistered-dialect --mlir-to-llvmir -o %t/queries.ll
// RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes='default<O2>' -S %t/queries.ll | FileCheck %s --check-prefix=OPT
// RUN: bishengir-opt %t/a5.mlir -convert-hivm-to-std -convert-hivm-to-llvm="use-bare-ptr-memref-call-conv=true" | mlir-translate --allow-unregistered-dialect --mlir-to-llvmir | FileCheck %t/a5.mlir --check-prefix=LLVM
// RUN: not bishengir-opt %t/collision.mlir -convert-hivm-to-llvm 2>&1 | FileCheck %s --check-prefix=COLLISION
// RUN: bishengir-opt %t/a6.mlir -convert-hivm-to-llvm | FileCheck %s --check-prefix=A6
// RUN: not bishengir-opt %t/a3.mlir -convert-hivm-to-llvm 2>&1 | FileCheck %s --check-prefix=REJECT

//--- a5.mlir

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, llvm.target_triple = "hiipu64-hisilicon-cce"} {
  // CHECK-LABEL: llvm.func @static_query
  func.func @static_query() -> i64 attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: %[[ID:.*]] = llvm.mlir.constant(31 : i64)
    // CHECK: %[[MODE:.*]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @llvm.hivm.TRY.WAIT.mode(%[[ID]], %[[MODE]])
    %counter = hivm.hir.try_wait flag = 31, sync_mode = <intra_block> : i64
    return %counter : i64
  }
  // CHECK-LABEL: llvm.func @dynamic_query
  func.func @dynamic_query(%flag: i64) -> i64 attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    // CHECK: %[[DYNAMIC_MODE:.*]] = llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @llvm.hivm.TRY.WAIT.mode(%arg0, %[[DYNAMIC_MODE]])
    %counter = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    return %counter : i64
  }
  // CHECK-LABEL: llvm.func @preserve_queries
  func.func @preserve_queries(%flag: i64) -> i64 {
    %a = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    %b = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    %unused = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    %sum = arith.addi %a, %b : i64
    return %sum : i64
  }
  func.func @poll_queries(%flag: i64, %limit: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    cf.br ^loop(%zero : i64)
  ^loop(%i: i64):
    %unused = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    %next = arith.addi %i, %one : i64
    %more = arith.cmpi ult, %next, %limit : i64
    cf.cond_br %more, ^loop(%next : i64), ^exit
  ^exit:
    return
  }

}
// LLVM-LABEL: define i64 @static_query()
// LLVM: call i64 @llvm.hivm.TRY.WAIT.mode(i64 31, i64 1)
// LLVM-LABEL: define i64 @dynamic_query(
// LLVM: call i64 @llvm.hivm.TRY.WAIT.mode(i64 %{{.*}}, i64 1)

//--- a3.mlir
module attributes {hacc.target = #hacc.target<"Ascend910B1">} {
  func.func @unsupported() -> i64 {
    // REJECT: 'hivm.hir.try_wait' op is not supported on memory-based (A3) architectures
    %v = hivm.hir.try_wait flag = 0, sync_mode = <intra_block> : i64
    return %v : i64
  }
}

//--- a6.mlir
module attributes {hacc.target = #hacc.target<"Ascend960">, llvm.target_triple = "hiipu64-hisilicon-cce"} {
  // A6: llvm.call @llvm.hivm.TRY.WAIT.mode(
  func.func @supported() -> i64 attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %v = hivm.hir.try_wait flag = 0, sync_mode = <intra_block> : i64
    return %v : i64
  }
}

// OPT-LABEL: define i64 @preserve_queries(
// OPT: call i64 @llvm.hivm.TRY.WAIT.mode
// OPT: call i64 @llvm.hivm.TRY.WAIT.mode
// OPT: call i64 @llvm.hivm.TRY.WAIT.mode
// OPT: ret i64

// STD: hivm.hir.try_wait flag = 31, sync_mode = <intra_block>
// STD: hivm.hir.try_wait flag = %arg0, sync_mode = <intra_block>

// OPT-LABEL: define void @poll_queries(
// OPT: br label
// OPT: call i64 @llvm.hivm.TRY.WAIT.mode
// OPT: br i1

//--- collision.mlir
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  llvm.func @llvm.hivm.TRY.WAIT.mode(i32, i32) -> i64
  func.func @invalid_declaration(%flag: i64) -> i64 {
    // COLLISION: incompatible declaration of llvm.hivm.TRY.WAIT.mode
    %v = hivm.hir.try_wait flag = %flag, sync_mode = <intra_block> : i64
    return %v : i64
  }
}
