// RUN: bishengir-opt %s -convert-proton-to-hivm -split-input-file -verify-diagnostics

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @crossed() {
    proton.record start "outer"
    proton.record start "inner"
    // expected-error @+1 {{scope end 'outer' does not match active scope 'inner'}}
    proton.record end "outer"
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // expected-error @+1 {{A5 Proton scope must end in the block where it starts}}
  func.func @unclosed() {
    proton.record start "outer"
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @closed_in_nested_region(%cond: i1) {
    proton.record start "outer"
    scf.if %cond {
      // expected-error @+1 {{scope end crosses a block or region boundary}}
      proton.record end "outer"
    }
    return
  }
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @name_reused_while_active() {
    proton.record start "outer"
    // expected-error @+1 {{scope name 'outer' is already active}}
    proton.record start "outer"
    return
  }
}
