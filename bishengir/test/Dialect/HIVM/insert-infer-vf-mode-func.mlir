// RUN: bishengir-opt --hacc-append-device-spec=target=Ascend950PR_9589 \
// RUN:               --hivm-insert-vf-mode-func -split-input-file %s | FileCheck %s

// SIMD

// CHECK-LABEL: @foo_infer_vf_mode_function() -> index
// CHECK: arith.constant 0
func.func private @bar() attributes { hivm.vector_function }

func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  call @bar() : () -> ()
  return
}

// -----

// SIMT
 
// CHECK-LABEL: @foo_infer_vf_mode_function() -> index
// CHECK: arith.constant 1
func.func private @bar() attributes { simt_parallel_loop }
 
func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  call @bar() : () -> ()
  return
}
 
// -----

// MIX

// CHECK-LABEL: @foo_infer_vf_mode_function() -> index
// CHECK: arith.constant 2
func.func private @bar() attributes { hivm.vector_function }
func.func private @xyz() attributes { simt_parallel_loop }

func.func private @foo2() {
  call @bar() : () -> ()
  call @xyz() : () -> ()
  return
}

func.func @foo() attributes { hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE> } {
  call @foo2() : () -> ()
  return
}
