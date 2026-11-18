// RUN: bishengir-opt %s -lower-triton-pipeline | FileCheck %s --check-prefix=ENABLED
// RUN: bishengir-opt %s "-lower-triton-pipeline=enable-optimize-atomic-reduction=false" | FileCheck %s --check-prefix=DISABLED

// With the default (enabled) setting, the select condition is folded into the
// atomic mask and the value select disappears.
// ENABLED-LABEL: llvm.func @atomic_add
// ENABLED-NOT: llvm.select %arg2
// ENABLED: %{{.*}} = llvm.and %{{.*}}, %arg2 : i1

// With the option disabled, the original select value remains and the
// condition is not folded into the mask.
// DISABLED-LABEL: llvm.func @atomic_add
// DISABLED: %{{.*}} = llvm.select %arg2, %arg1, %{{.*}} : i1, i32

module {
  tt.func public @atomic_add(%ptr: !tt.ptr<i32>, %a: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %s = arith.select %cond, %a, %c0 : i32
    %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %s : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }
}
