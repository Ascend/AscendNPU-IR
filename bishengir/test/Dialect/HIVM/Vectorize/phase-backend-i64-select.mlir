// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960DT_968B7 --hivm-vectorize=phase=backend --verify-each | FileCheck %s --check-prefixes=A6,COMMON
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend950PR_9579 --hivm-vectorize=phase=backend --verify-each | FileCheck %s --check-prefixes=A5,COMMON

// The single use of the condition selects one full-width arm, including for
// an undef condition. A poison value in the unselected arm cannot escape.
// A6-LABEL: func.func @choose_i64(
// A6-DAG: %[[T:.*]] = llvm.freeze %arg1 : i64
// A6-DAG: %[[F:.*]] = llvm.freeze %arg2 : i64
// A6-DAG: %[[MASK:.*]] = arith.extsi %arg0 : i1 to i64
// A6: %[[DIFF:.*]] = arith.xori %[[T]], %[[F]] : i64
// A6: %[[CHOSEN:.*]] = arith.andi %[[MASK]], %[[DIFF]] : i64
// A6: %[[RESULT:.*]] = arith.xori %[[F]], %[[CHOSEN]] : i64
// A6: return %[[RESULT]] : i64
// A5-LABEL: func.func @choose_i64(
// A5: arith.select %arg0, %arg1, %arg2 : i64
func.func @choose_i64(%p: i1, %t: i64, %f: i64) -> i64 attributes {hivm.vector_function} {
  %r = arith.select %p, %t, %f : i64
  return %r : i64
}

// COMMON-LABEL: func.func @choose_i32(
// COMMON: arith.select %arg0, %arg1, %arg2 : i32
func.func @choose_i32(%p: i1, %t: i32, %f: i32) -> i32 attributes {hivm.vector_function} {
  %r = arith.select %p, %t, %f : i32
  return %r : i32
}

// COMMON-LABEL: func.func @choose_si64(
// COMMON: arith.select %arg0, %arg1, %arg2 : si64
func.func @choose_si64(%p: i1, %t: si64, %f: si64) -> si64 attributes {hivm.vector_function} {
  %r = arith.select %p, %t, %f : si64
  return %r : si64
}

// COMMON-LABEL: func.func @scalar_caller(
// COMMON: arith.select %arg0, %arg1, %arg2 : i64
func.func @scalar_caller(%p: i1, %t: i64, %f: i64) -> i64 {
  %r = arith.select %p, %t, %f : i64
  return %r : i64
}
