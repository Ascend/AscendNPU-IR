// RUN: bishengir-opt %s --hivm-v2-split-kernel --allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s
// RUN: bishengir-opt %s --hivm-split-mix-kernel --allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

// Both splitters preserve the host symbol and share the launch metadata.
// CHECK-LABEL: func.func private @called(
// CHECK-SAME: hacc.function_kind = #hacc.function_kind<DEVICE>
// CHECK-SAME: hacc.mix_entry
// CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 2>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<MIX>
// CHECK-LABEL: func.func @called_mix_aic(
// CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 2>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIC>
// CHECK-SAME: hivm.part_of_mix
// CHECK-LABEL: func.func @called_mix_aiv(
// CHECK-SAME: hivm.core_ratio = #hivm.core_ratio<1, 2>
// CHECK-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>
// CHECK-SAME: hivm.part_of_mix
// CHECK-LABEL: func.func @host(
// CHECK: call @called(
module {
  func.func @called(%arg: memref<1xi32>) attributes {
      hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>,
      hivm.core_ratio = #hivm.core_ratio<1, 2>,
      hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func @host(%arg: memref<1xi32>)
      attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    func.call @called(%arg) : (memref<1xi32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module {
// CHECK-NOT: @uncalled(
// CHECK: func.func @uncalled_mix_aic(
// CHECK: func.func @uncalled_mix_aiv(
module {
  func.func @uncalled()
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{split kernel symbol already exists: cube_collision_mix_aic}}
  func.func @cube_collision()
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func private @cube_collision_mix_aic()
}

// -----

module {
  // expected-error@+1 {{split kernel symbol already exists: vector_collision_mix_aiv}}
  func.func @vector_collision()
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func private @vector_collision_mix_aiv()
}

// -----

module {
  // expected-error@+1 {{Currently, MIX kernels can only be called by host functions!}}
  func.func @indirect()
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func @address_of_mix() -> (() -> ()) {
    %addr = func.constant @indirect : () -> ()
    return %addr : () -> ()
  }
}

// -----

module {
  // expected-error@+1 {{Currently, MIX kernels can only be called by host functions!}}
  func.func @device_called()
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func @device_caller()
      attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
    func.call @device_called() : () -> ()
    return
  }
}

// -----

module {
  // expected-error@+1 {{cannot determine MIX kernel symbol uses}}
  func.func @unknown_uses()
      attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  "unregistered.unknown_symbol_table"() ({
    cf.br ^loop
  ^loop:
    cf.br ^loop
  }) {callee = @unknown_uses} : () -> ()
}
