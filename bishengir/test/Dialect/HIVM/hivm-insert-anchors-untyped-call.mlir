// RUN: bishengir-opt --hivm-insert-anchors-and-backup="insert-anchor-ops-before-all" --verify-diagnostics %s

func.func private @helper() {
  return
}

func.func private @either_core() attributes {hivm.func_core_type = #hivm.func_core_type<AIC_OR_AIV>} {
  return
}

// An unresolved op attribute still triggers lookup of the callee's core type.
func.func @untyped_call() attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
  // expected-error@+1 {{function core type is unknown for 'helper'}}
  call @helper() {hivm.tcore_type = #hivm.tcore_type<CUBE_OR_VECTOR>} : () -> ()
  // A known callee with flexible placement resolves without a diagnostic.
  call @either_core() : () -> ()
  return
}
