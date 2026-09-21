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

// -----

// Tensor-valued while is not supported by the local region traversal yet.
// Both the pre-region-inference pass and the current pass reject this all-UB
// case. Do not assume UB merely because the initial value is tensor.empty:
// future support must trace condition forwarding and loop-carried sources.
module {
  // expected-error @+1 {{failed to infer memory scope hint for simt argument #0}}
  func.func @simt_from_tensor_while(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @while_caller(%cond: i1) {
    %local = tensor.empty() : tensor<8xi64>
    %result = scf.while (%iter = %local) : (tensor<8xi64>) -> tensor<8xi64> {
      scf.condition(%cond) %iter : tensor<8xi64>
    } do {
    ^bb0(%arg: tensor<8xi64>):
      scf.yield %arg : tensor<8xi64>
    }
    %buffer = bufferization.to_memref %result : memref<8xi64>
    call @simt_from_tensor_while(%buffer) : (memref<8xi64>) -> ()
    return
  }
}

// -----

module {
  // expected-error @+1 {{failed to infer memory scope hint for simt argument #0}}
  func.func @simt_from_mixed_region(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @mixed_caller(%gm: memref<8xi64>, %cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %global = bufferization.to_tensor %gm restrict writable : memref<8xi64>
    %local = tensor.empty() : tensor<8xi64>
    %result = scf.if %cond -> tensor<8xi64> {
      scf.yield %global : tensor<8xi64>
    } else {
      scf.yield %local : tensor<8xi64>
    }
    %buffer = bufferization.to_memref %result : memref<8xi64>
    call @simt_from_mixed_region(%buffer) : (memref<8xi64>) -> ()
    return
  }
}

// -----

// The known UB branch cannot hide an unknown source in a non-entry function.
module {
  // expected-error @+1 {{failed to infer memory scope hint for simt argument #0}}
  func.func @simt_from_unknown_region(%arg0: memref<8xi64>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline, outline, hivm.vf_mode = #hivm.vf_mode<SIMT>} {
    return
  }
  func.func @unknown_caller(%unknown: tensor<8xi64>, %cond: i1) {
    %local = tensor.empty() : tensor<8xi64>
    %result = scf.if %cond -> tensor<8xi64> {
      scf.yield %unknown : tensor<8xi64>
    } else {
      scf.yield %local : tensor<8xi64>
    }
    %buffer = bufferization.to_memref %result : memref<8xi64>
    call @simt_from_unknown_region(%buffer) : (memref<8xi64>) -> ()
    return
  }
}
