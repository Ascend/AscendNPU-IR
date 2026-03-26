// RUN: bishengir-opt %s -cse -canonicalize | FileCheck %s

llvm.func @test_redundant_load(%arg0: !llvm.ptr<1>, %arg1: i1, %arg2: vector<1xi64>) -> i64 {
  %0 = ascend_dpx.load %arg0, %arg1, %arg2 : (!llvm.ptr<1>, i1, vector<1xi64>) -> vector<1xi64>
  %1 = ascend_dpx.load %arg0, %arg1, %arg2 : (!llvm.ptr<1>, i1, vector<1xi64>) -> vector<1xi64>
  %2 = ascend_dpx.load %arg0, %arg1, %arg2 : (!llvm.ptr<1>, i1, vector<1xi64>) -> vector<1xi64>
  %3 = llvm.bitcast %0 : vector<1xi64> to i64
  %4 = llvm.bitcast %1 : vector<1xi64> to i64
  %5 = llvm.add %3, %4 : i64
  llvm.return %5 : i64
}

// CHECK-LABEL: llvm.func @test_redundant_load
// CHECK-SAME:    (%[[PTR:.*]]: !llvm.ptr<1>, %[[PRED:.*]]: i1, %[[IDX:.*]]: vector<1xi64>) -> i64
// CHECK-NEXT:    %[[LOAD:.*]] = ascend_dpx.load %[[PTR]], %[[PRED]], %[[IDX]] : (!llvm.ptr<1>, i1, vector<1xi64>) -> vector<1xi64>
// CHECK-NEXT:    %[[CAST:.*]] = llvm.bitcast %[[LOAD]] : vector<1xi64> to i64
// CHECK-NEXT:    %[[ADD:.*]] = llvm.add %[[CAST]], %[[CAST]] : i64
// CHECK-NEXT:    llvm.return %[[ADD]] : i64