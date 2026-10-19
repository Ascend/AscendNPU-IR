// RUN: bishengir-opt %s -hivm-mark-multi-buffer="enable-auto=false enable-preload=true" | FileCheck %s
// RUN: bishengir-opt %s -hivm-mark-multi-buffer="enable-auto=false enable-preload=false" | FileCheck %s --check-prefix=DISABLED

// Unnumbered inner scopes must retain the outer producer/consumer stages.
// CHECK: %[[BUFFER:.*]] = memref.alloc
// CHECK-NEXT: annotation.mark %[[BUFFER]] {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32}
// DISABLED-NOT: hivm.multi_buffer
func.func @nested_preload(%input: memref<16xf32, #hivm.address_space<gm>>, %output: memref<16xf32, #hivm.address_space<gm>>) {
  %buffer = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  annotation.mark %buffer {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16xf32, #hivm.address_space<ub>>
  scope.scope : () -> () {
    scope.scope : () -> () {
      hivm.hir.load ins(%input : memref<16xf32, #hivm.address_space<gm>>) outs(%buffer : memref<16xf32, #hivm.address_space<ub>>)
      scope.return
    } {no_inline}
    scope.return
  } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 1 : i32}
  scope.scope : () -> () {
    scope.scope : () -> () {
      hivm.hir.store ins(%buffer : memref<16xf32, #hivm.address_space<ub>>) outs(%output : memref<16xf32, #hivm.address_space<gm>>)
      scope.return
    } {no_inline}
    scope.return
  } {hivm.max_preload_num = 2 : i32, hivm.preload_num = 0 : i32}
  return
}
