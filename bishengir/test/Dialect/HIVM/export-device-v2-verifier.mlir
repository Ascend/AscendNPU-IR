// RUN: bishengir-opt %s --split-input-file --hivm-v2-verify-stage=stage=planned --verify-diagnostics

// An export is the compilation root. Helpers and declarations are not roots.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @exported() attributes {export_device, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func private @helper() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    return
  }
  func.func private @external() attributes {export_device, hacc.function_kind = #hacc.function_kind<DEVICE>}
}

// -----

// Launch entries remain valid roots.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @entry() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
}

// -----

// An ordinary device function is not an independent compilation root.
// expected-error@+1 {{expected exactly one compilation root, found 0}}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @helper() attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
}

// -----

// expected-error@+1 {{expected exactly one compilation root, found 2}}
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @first() attributes {export_device, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
  func.func @second() attributes {export_device, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    return
  }
}
