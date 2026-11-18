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

// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: llvm.func @fp8_vector_function(
  // CHECK-SAME: %[[ARG:.*]]: i8)
  // CHECK: %[[VALUE:.*]] = llvm.bitcast %[[ARG]] : i8 to f8E4M3FN
  func.func @fp8_vector_function(%arg0: f8E4M3FN) attributes {hivm.vector_function} {
    "some_use"(%arg0) : (f8E4M3FN) -> ()
    return
  }

  // CHECK-LABEL: llvm.func @external_abi(
  // CHECK-SAME: %{{.*}}: f8E4M3FN)
  func.func @external_abi(%arg0: f8E4M3FN) {
    return
  }

  // CHECK-LABEL: llvm.func @fp8_caller(
  // CHECK-SAME: %[[ARG:.*]]: f8E4M3FN)
  // CHECK: %[[PAYLOAD:.*]] = llvm.bitcast %[[ARG]] : f8E4M3FN to i8
  // CHECK-NEXT: llvm.call @fp8_vector_function(%[[PAYLOAD]]) : (i8) -> ()
  func.func @fp8_caller(%arg0: f8E4M3FN) {
    func.call @fp8_vector_function(%arg0) : (f8E4M3FN) -> ()
    return
  }
}
