// RUN: bishengir-opt %s -hivm-set-buffer-size -allow-unregistered-dialect -split-input-file -verify-diagnostics | FileCheck %s

func.func @set_buffer_size(%arg0: index, %arg1: index) {
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<4000xi8, 1>
  // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
  // CHECK: %[[VIEW0:.*]] = memref.view %[[ALLOC0]][%[[CONST0]]][%[[INDEX0:.*]], %[[INDEX1:.*]]] : memref<4000xi8, 1> to memref<?x?xf32, 1>
  %alloc = memref.alloc(%arg0, %arg1) : memref<?x?xf32, 1>
  annotation.mark %alloc {buffer_size_in_byte = 4000 : i64} : memref<?x?xf32, 1>
  // CHECK: %[[ALLOC1:.*]] = memref.alloca() : memref<4000xi8, 1>
  // CHECK: %[[CONST1:.*]] = arith.constant 0 : index
  // CHECK: %[[VIEW1:.*]] = memref.view %[[ALLOC1]][%[[CONST1]]][%[[INDEX1]], %[[INDEX0]]] : memref<4000xi8, 1> to memref<?x?xf32, 1>
  %alloca = memref.alloca(%arg1, %arg0) : memref<?x?xf32, 1>
  annotation.mark %alloca {buffer_size_in_byte = 4000 : i64} : memref<?x?xf32, 1>
  // CHECK: "some_use"(%[[VIEW0]])
  "some_use"(%alloc) : (memref<?x?xf32, 1>) -> ()
  // CHECK: "some_other_use"(%[[VIEW1]])
  "some_other_use"(%alloca) : (memref<?x?xf32, 1>) -> ()
  return
}

// -----

func.func @set_buffer_size_static() {
  %alloc = memref.alloc() : memref<16xf32, 1>
  // CHECK-NOT: annotation.mark %alloc
  annotation.mark %alloc {buffer_size_in_byte = 4000 : i64} : memref<16xf32, 1>
  "some_use"(%alloc) : (memref<16xf32, 1>) -> ()
  return
}

// -----

func.func @set_buffer_size_error1(%arg0: index, %arg1: index) {
  %alloc = memref.alloc(%arg0, %arg1) : memref<?x?xf32, 1>
  annotation.mark %alloc {buffer_size_in_byte = 4000 : i64} : memref<?x?xf32, 1>
  // expected-error@+1 {{Found conflicting buffer size annotation on the same alloc!}}
  annotation.mark %alloc {buffer_size_in_byte = 3000 : i64} : memref<?x?xf32, 1>
  "some_use"(%alloc) : (memref<?x?xf32, 1>) -> ()
  return
}

// -----

func.func @set_buffer_size_error2() {
  %ret = "some_op"() : () -> (memref<?x?xf32>)
  // expected-warning@+1 {{Cannot find root memref alloc/alloca to set buffer size!}}
  annotation.mark %ret {buffer_size_in_byte = 4000 : i64} : memref<?x?xf32>
  return
}

// -----

func.func @set_buffer_size_static() {
  // CHECK-NOT: memref.alloc() : memref<4000xi8>
  %alloc = memref.alloc() : memref<16xf32>
  annotation.mark %alloc {buffer_size_in_byte = 4000 : i64} : memref<16xf32>
  return
}
