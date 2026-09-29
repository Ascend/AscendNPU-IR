// RUN: bishengir-opt %s --infer-simt-vf-memory-scope-hint --split-input-file --verify-diagnostics

// A known allocation must not hide an unknown non-region root. The helper
// argument can actually be GM, as its device-entry caller demonstrates; the
// old pass skipped that root and incorrectly assigned UB to the selected value.
module {
  // expected-error @+1 {{failed to infer memory scope hint for simt argument #0}}
  func.func @simt_from_unknown_select(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @non_entry_caller(%unknown: memref<8xi64>, %cond: i1) {
    %local = memref.alloc() : memref<8xi64>
    %selected = arith.select %cond, %local, %unknown : memref<8xi64>
    call @simt_from_unknown_select(%selected) : (memref<8xi64>) -> ()
    return
  }
  func.func @entry(%gm: memref<8xi64>, %cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    call @non_entry_caller(%gm, %cond) : (memref<8xi64>, i1) -> ()
    return
  }
}

// -----

// Source order must not affect rejection of a known + unknown combination.
module {
  // expected-error @+1 {{failed to infer memory scope hint for simt argument #0}}
  func.func @simt_from_reversed_unknown_select(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @non_entry_caller(%unknown: memref<8xi64>, %cond: i1) {
    %local = memref.alloc() : memref<8xi64>
    %selected = arith.select %cond, %unknown, %local : memref<8xi64>
    call @simt_from_reversed_unknown_select(%selected) : (memref<8xi64>) -> ()
    return
  }
}
