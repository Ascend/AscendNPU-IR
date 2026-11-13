// RUN: bishengir-opt %s --hivm-v2-split-kernel --split-input-file --verify-diagnostics

module {
  func.func private @vector_index() -> index
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>}
  func.func @cross_core_scalar(%result: memref<1xindex>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{filtered operation has live results on the other core}}
    %idx = func.call @vector_index() : () -> index
    memref.store %idx, %result[%c0] : memref<1xindex>
    return
  }
}

// -----

module {
  func.func @cross_core_load(%buffer: memref<1xi32>, %result: memref<1xi32>)
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c0 = arith.constant 0 : index
    %value = scope.scope : () -> i32 {
      // expected-error@+1 {{filtered operation has live results on the other core}}
      %value = memref.load %buffer[%c0] : memref<1xi32>
      scope.return %value : i32
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    memref.store %value, %result[%c0] : memref<1xi32>
    return
  }
}

// -----

module {
  func.func private @consume_vector(vector<64xf32>)
      attributes {hivm.func_core_type = #hivm.func_core_type<AIC>}
  func.func @cross_core_vector()
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // expected-error@+1 {{filtered operation has live results on the other core}}
    %value = scope.scope : () -> vector<64xf32> {
      %zero = arith.constant dense<0.0> : vector<64xf32>
      scope.return %zero : vector<64xf32>
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    func.call @consume_vector(%value) : (vector<64xf32>) -> ()
    return
  }
}

// -----

module {
  func.func private @unknown()
  func.func @unknown_callee_core()
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    // expected-error@+1 {{function core type is unknown for 'unknown'}}
    func.call @unknown() : () -> ()
    return
  }
}

// -----

module {
  // expected-error@+1 {{Currently, MIX kernels can only be called by host functions!}}
  func.func @indirect_mix() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func @address_of_mix() -> (() -> ()) {
    %addr = func.constant @indirect_mix : () -> ()
    return %addr : () -> ()
  }
}

// -----

module {
  // expected-error@+1 {{split kernel symbol already exists: colliding_mix_aiv}}
  func.func @colliding() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func private @colliding_mix_aiv()
}

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // expected-error@+1 {{hivm.core_ratio<1, 0> reserves no vector core, but the kernel has vector work}}
  func.func @missing_inline_vf_core(%buffer: memref<1xf32, #hivm.address_space<ub>>)
      attributes {hivm.core_ratio = #hivm.core_ratio<1, 0>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>} {
    scope.scope : () -> () {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1.0 : f32
      memref.store %c1, %buffer[%c0] : memref<1xf32, #hivm.address_space<ub>>
      scope.return
    } {hivm.vector_function, no_inline, outline}
    return
  }
}
