// RUN: bishengir-opt -convert-hivm-to-llvm="on-demand-bare-ptr-memref-call-conv" -canonicalize -allow-unregistered-dialect  -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK: {{.*}}: !llvm.ptr, {{.*}}: !llvm.ptr, {{.*}}: i64, {{.*}}: i64, {{.*}}: i64,
// CHECK: {{.*}}: !llvm.ptr {llvm.bareptr}
func.func @bare_ptr_calling_conv(%arg0: memref<1xf32>,
                                 %arg1: memref<1xf32> {llvm.bareptr}) -> () {
  "some_use"(%arg1) : (memref<1xf32>) -> ()
  return
}

// -----

// CHECK: llvm.func @bare_ptr_calling_conv({{.*}}: !llvm.ptr, {{.*}}: !llvm.ptr, {{.*}}: i64, {{.*}}: i64, {{.*}}: i64,
// CHECK:                                  %[[BARE_PTR:.*]]: !llvm.ptr {llvm.bareptr}) {
// CHECK:   %[[SIZE:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:   %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:   llvm.call @extern({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[BARE_PTR]], %[[BARE_PTR]], %[[OFFSET]], %[[SIZE]], %[[SIZE]])
// CHECK:   llvm.return
// CHECK: }
func.func @bare_ptr_calling_conv(%arg0: memref<1xf32>,
                                 %arg1: memref<1xf32> {llvm.bareptr}) -> () {

  call @extern(%arg0, %arg1) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

func.func private @extern(%arg0: memref<1xf32>, %arg1: memref<1xf32>) attributes {llvm.emit_c_interface}
// CHECK: llvm.func @_mlir_ciface_extern

// -----

// expected-error@below {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @bare_ptr_calling_conv(%arg0: memref<?xf32> {llvm.bareptr}) -> () {
  "some_use"(%arg0) : (memref<?xf32>) -> ()
  return
}
