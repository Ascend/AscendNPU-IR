// RUN: bishengir-opt --hacc-append-device-spec=target=Ascend950PR_9589 \
// RUN:               --hivm-infer-vf-mode -split-input-file %s | FileCheck %s

// enum VFMode : int64_t { SIMD = 0, SIMT, MIX };

// -----

// SIMD

func.func private @bar() attributes { hivm.vector_function }

// CHECK-LABEL: @foo()
// CHECK: hivm.vf_mode<SIMD>
func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  call @bar() : () -> ()
  return
}

// -----

// SIMT

func.func private @bar() attributes { hivm.vf_mode = #hivm.vf_mode<SIMT> }

// CHECK-LABEL: @foo()
// CHECK: hivm.vf_mode<SIMT>
func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  call @bar() : () -> ()
  return
}

// -----

// MIX

func.func private @bar() attributes { hivm.vector_function }
func.func private @xyz() attributes { hivm.vf_mode = #hivm.vf_mode<SIMT> }

// CHECK-LABEL: @foo()
// CHECK: hivm.vf_mode<MIX>
func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  call @bar() : () -> ()
  call @xyz() : () -> ()
  return
}

// -----

// MIX

func.func private @bar() attributes { hivm.vector_function }
func.func private @xyz() attributes { hivm.vf_mode = #hivm.vf_mode<SIMT> }

func.func private @foo2() {
  call @bar() : () -> ()
  call @xyz() : () -> ()
  return
}

// CHECK-LABEL: @foo()
// CHECK: hivm.vf_mode<MIX>
func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  call @foo2() : () -> ()
  return
}

// -----

// MIX

func.func private @bar() attributes { hivm.vector_function }
func.func private @xyz() attributes { hivm.vf_mode = #hivm.vf_mode<SIMT> }

func.func private @foo2() {
  call @bar() : () -> ()
  call @xyz() : () -> ()
  return
}

// CHECK-LABEL: @foo()
// CHECK: hivm.vf_mode<MIX>
func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  call @foo2() : () -> ()
  return
}

// -----

// SIMD

// CHECK-LABEL: @foo(%a
// CHECK: hivm.vf_mode<SIMD>
func.func @foo(%a: tensor<1x?x10xf32>, %c: tensor<5x?x10xf32>) attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %0 = hivm.hir.vabs ins(%a: tensor<1x?x10xf32>) outs(%c: tensor<5x?x10xf32>) broadcast = [0] -> tensor<5x?x10xf32>
  return
}

// -----

// MIX

// CHECK-LABEL: @foo()
// CHECK: hivm.vf_mode<MIX>
func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  %c0_i64 = arith.constant 0: i64
  %a = hivm.hir.pointer_cast(%c0_i64) { hivm.vector_function }              : memref<1xf32>
  %b = hivm.hir.pointer_cast(%c0_i64) { hivm.vf_mode = #hivm.vf_mode<SIMT> } : memref<1xf32>
  return
}
