// RUN: bishengir-opt -fix-call-unknown-loc --mlir-print-debuginfo %s \
// RUN:   | FileCheck %s --check-prefix=NODEBUG
// RUN: bishengir-opt -fix-call-unknown-loc=assume-debug-info=true \
// RUN:   --mlir-print-debuginfo %s | FileCheck %s --check-prefix=DEBUG

// A module in which nothing at all carries a location, including the module
// itself. This is what Triton emits when line info is disabled: every op is
// printed as loc(unknown), so none of the inheritance priorities in
// findNonUnknownLocForFunc can find a location.
//
// Without assume-debug-info the step 2 shortcut applies and the call keeps its
// UnknownLoc: no DISubprogram would be attached, so no verifier error.
//
// With assume-debug-info the call must be given the synthetic location,
// because DIScopeForLLVMFuncOpPass attaches a DISubprogram unconditionally and
// an inlinable call under it needs a debug location.

module {
  func.func private @callee() -> () loc(unknown)

  func.func @all_locs_unknown() {
    func.call @callee() : () -> () loc(unknown)
    return loc(unknown)
  } loc(unknown)
} loc(unknown)

// NODEBUG: call @callee() : () -> () loc(#[[KEPT:.*]])
// NODEBUG-DAG: #[[KEPT]] = loc(unknown)

// DEBUG: call @callee() : () -> () loc(#[[SYNTH:.*]])
// DEBUG-DAG: #[[SYNTH]] = loc("<unknown>":0:0)
