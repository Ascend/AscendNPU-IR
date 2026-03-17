// RUN: bishengir-opt %s -bind-buffer -split-input-file -allow-unregistered-dialect --verify-diagnostics | FileCheck %s

func.func @bind_buffer() {
  // CHECK-NOT: %[[SOURCE:.*]] = memref.alloc()
  %source = memref.alloc() : memref<16xf32>
  // CHECK: %[[TARGET:.*]] = memref.alloc
  %target = memref.alloc() : memref<16xf32>
  // CHECK: "some_op"(%[[TARGET]])
  "some_op"(%source) : (memref<16xf32>) -> ()
  // CHECK-NOT: annotation.mark
  annotation.mark %source keys = ["bind_buffer"] values = [%target : memref<16xf32>] : memref<16xf32>
  return
}

// -----

func.func @bind_buffer_with_address_space() {
  // CHECK-NOT: %[[SOURCE:.*]] = memref.alloc()
  %source = memref.alloc() {source} : memref<16xf32>
  // CHECK: %[[TARGET:.*]] = memref.alloc()
  // CHECK: %[[CAST:.*]] = memref.memory_space_cast %[[TARGET]]
  %target = memref.alloc() : memref<16xf32, 1>
  // CHECK: "some_op"(%[[CAST]])
  "some_op"(%source) : (memref<16xf32>) -> ()
  // CHECK-NOT: annotation.mark
  annotation.mark %source keys = ["bind_buffer"] values = [%target : memref<16xf32, 1>] : memref<16xf32>
  return
}

// -----

func.func @bind_buffer_with_cast(%size: index) {
  // CHECK-NOT: %[[SOURCE:.*]] = memref.alloc()
  %source = memref.alloc() {source} : memref<16xf32>
  // CHECK: %[[TARGET:.*]] = memref.alloc({{.*}})
  // CHECK: %[[CAST:.*]] = memref.cast %[[TARGET]]
  %target = memref.alloc(%size) : memref<?xf32>
  // CHECK: "some_op"(%[[CAST]])
  "some_op"(%source) : (memref<16xf32>) -> ()
  // CHECK-NOT: annotation.mark
  annotation.mark %source keys = ["bind_buffer"] values = [%target : memref<?xf32>] : memref<16xf32>
  return
}

// -----

// expected-error@below {{Failed to bind all allocs}}
func.func @not_cast_compatible() {
  %source = memref.alloc() : memref<16xf32>
  %target = memref.alloc() : memref<32xf32>
  "some_op"(%source) : (memref<16xf32>) -> ()
  annotation.mark %source keys = ["bind_buffer"] values = [%target : memref<32xf32>] : memref<16xf32>
  return
}

// -----

func.func @bind_if_else(%condition: i1) -> (memref<16xf32>) {
  // CHECK: %[[TARGET:.*]] = memref.alloc
  %target = memref.alloc() : memref<16xf32>
  %0 = scf.if %condition -> (memref<16xf32>) {
    // CHECK-NOT: memref.alloc
    %source = memref.alloc() : memref<16xf32>
    // CHECK: "some_op"(%[[TARGET]])
    "some_op"(%source) : (memref<16xf32>) -> ()
    scf.yield %source : memref<16xf32>
    } else {
    scf.yield %target : memref<16xf32>
  }
  annotation.mark %0 keys = ["bind_buffer"] values = [%target : memref<16xf32>] : memref<16xf32>
  return %0 : memref<16xf32>
}

// -----

func.func @bind_buffer_reshape_self() {
  // Reshape bound to itself: alloc -> expand_shape, annotation on reshape.
  // Should just remove annotation (replace alloc with itself).
  %alloc = memref.alloc() : memref<16xf32>
  %reshape = memref.expand_shape %alloc [[0, 1]] output_shape [4, 4] : memref<16xf32> into memref<4x4xf32>
  "some_op"(%reshape) : (memref<4x4xf32>) -> ()
  // CHECK-NOT: annotation.mark
  annotation.mark %reshape keys = ["bind_buffer"] values = [%reshape : memref<4x4xf32>] : memref<4x4xf32>
  return
}
