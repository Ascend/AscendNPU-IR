// RUN: bishengir-opt %s -hfusion-simplify-vf-arg | FileCheck %s

module {
  // CHECK-LABEL: func.func @vf_e4m3fn(
  // CHECK-SAME: %[[E4_ARG:.*]]: i8)
  // CHECK-NEXT: %[[E4_VALUE:.*]] = arith.bitcast %[[E4_ARG]] : i8 to f8E4M3FN
  // CHECK-NEXT: vector.broadcast %[[E4_VALUE]] : f8E4M3FN to vector<256xf8E4M3FN>
  func.func @vf_e4m3fn(%arg0: f8E4M3FN) attributes {hivm.vector_function} {
    %0 = vector.broadcast %arg0 : f8E4M3FN to vector<256xf8E4M3FN>
    return
  }

  // CHECK-LABEL: func.func @vf_e5m2(
  // CHECK-SAME: %[[E5_ARG:.*]]: i8)
  // CHECK-NEXT: %[[E5_VALUE:.*]] = arith.bitcast %[[E5_ARG]] : i8 to f8E5M2
  // CHECK-NEXT: vector.broadcast %[[E5_VALUE]] : f8E5M2 to vector<256xf8E5M2>
  func.func @vf_e5m2(%arg0: f8E5M2) attributes {hivm.vector_function} {
    %0 = vector.broadcast %arg0 : f8E5M2 to vector<256xf8E5M2>
    return
  }

  // Non-VF functions are external ABI boundaries and must not be changed.
  // CHECK-LABEL: func.func @not_a_vf(
  // CHECK-SAME: %[[PLAIN_ARG:.*]]: f8E4M3FN)
  // CHECK-NEXT: vector.broadcast %[[PLAIN_ARG]] : f8E4M3FN to vector<256xf8E4M3FN>
  func.func @not_a_vf(%arg0: f8E4M3FN) {
    %0 = vector.broadcast %arg0 : f8E4M3FN to vector<256xf8E4M3FN>
    return
  }

  // CHECK-LABEL: func.func @caller(
  // CHECK-SAME: %[[CALL_E4:.*]]: f8E4M3FN, %[[CALL_E5:.*]]: f8E5M2)
  // CHECK: %[[E4_PAYLOAD:.*]] = arith.bitcast %[[CALL_E4]] : f8E4M3FN to i8
  // CHECK-NEXT: call @vf_e4m3fn(%[[E4_PAYLOAD]]) : (i8) -> ()
  // CHECK: %[[E5_PAYLOAD:.*]] = arith.bitcast %[[CALL_E5]] : f8E5M2 to i8
  // CHECK-NEXT: call @vf_e5m2(%[[E5_PAYLOAD]]) : (i8) -> ()
  func.func @caller(%arg0: f8E4M3FN, %arg1: f8E5M2) {
    func.call @vf_e4m3fn(%arg0) : (f8E4M3FN) -> ()
    func.call @vf_e5m2(%arg1) : (f8E5M2) -> ()
    return
  }
}
