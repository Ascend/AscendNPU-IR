// RUN: bishengir-opt %s --split-input-file --verify-diagnostics

module {
  func.func @invalid_is_bool(%arg0: i32) {
    // expected-error@+1 {{'ascend_dpx.print' op expected 1 isBool entries, but got 0}}
    "ascend_dpx.print"(%arg0) {hex = false, isBool = array<i32>, isSigned = array<i32: 1>, prefix = "value: "} : (i32) -> ()
    return
  }
}

// -----

module {
  func.func @invalid_is_signed(%arg0: i32) {
    // expected-error@+1 {{'ascend_dpx.print' op expected 1 isSigned entries, but got 0}}
    "ascend_dpx.print"(%arg0) {hex = false, isBool = array<i32: 0>, isSigned = array<i32>, prefix = "value: "} : (i32) -> ()
    return
  }
}
