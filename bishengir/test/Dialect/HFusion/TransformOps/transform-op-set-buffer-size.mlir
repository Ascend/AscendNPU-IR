// RUN: bishengir-opt -transform-interpreter -split-input-file -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_element_mode(%arg0: index, %arg1: index) {
    // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<4000xi8, 1>
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW0:.*]] = memref.view %[[ALLOC0]][%[[CONST0]]][%[[INDEX0:.*]], %[[INDEX1:.*]]] : memref<4000xi8, 1> to memref<?x?xf32, 1>
    %alloc = memref.alloc(%arg0, %arg1) : memref<?x?xf32, 1>
    // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<4000xi8, 1>
    // CHECK: %[[CONST1:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW1:.*]] = memref.view %[[ALLOC1]][%[[CONST1]]][%[[INDEX1]], %[[INDEX0]]] : memref<4000xi8, 1> to memref<?x?xf32, 1>
    %alloc1 = memref.alloc(%arg1, %arg0) : memref<?x?xf32, 1>
    // CHECK: "some_use"(%[[VIEW0]])
    "some_use"(%alloc) : (memref<?x?xf32, 1>) -> ()
    // CHECK: "some_other_use"(%[[VIEW1]])
    "some_other_use"(%alloc1) : (memref<?x?xf32, 1>) -> ()
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.alloc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [1000] unit_mode = "per_element" : !transform.any_op 
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_element_mode(%arg0: index, %arg1: index) {
    // CHECK: %[[ALLOCA0:.*]] = memref.alloca() : memref<1000xi8>
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW0:.*]] = memref.view %[[ALLOCA0]][%[[CONST0]]][%[[INDEX0:.*]], %[[INDEX1:.*]]] : memref<1000xi8> to memref<?x?xf32>
    %alloca = memref.alloca(%arg0, %arg1) : memref<?x?xf32>
    // CHECK: %[[ALLOCA1:.*]] = memref.alloca() : memref<1000xi8>
    // CHECK: %[[CONST1:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW1:.*]] = memref.view %[[ALLOCA1]][%[[CONST1]]][%[[INDEX1]], %[[INDEX0]]] : memref<1000xi8> to memref<?x?xf32>
    %alloca1 = memref.alloca(%arg1, %arg0) : memref<?x?xf32>
    // CHECK: "some_use"(%[[VIEW0]])
    "some_use"(%alloca) : (memref<?x?xf32>) -> ()
    // CHECK: "some_other_use"(%[[VIEW1]])
    "some_other_use"(%alloca1) : (memref<?x?xf32>) -> ()
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.alloca"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [1000] unit_mode = "per_byte" : !transform.any_op 
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_variadic(%arg0: index, %arg1: index) {
    // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1000xi8>
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW0:.*]] = memref.view %[[ALLOC0]][%[[CONST0]]][%[[INDEX0:.*]], %[[INDEX1:.*]]] : memref<1000xi8> to memref<?x?xf32>
    %alloc = memref.alloc(%arg0, %arg1) {__a__} : memref<?x?xf32>
    // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2000xi8>
    // CHECK: %[[CONST1:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW1:.*]] = memref.view %[[ALLOC1]][%[[CONST1]]][%[[INDEX1]], %[[INDEX0]]] : memref<2000xi8> to memref<?x?xf32>
    %alloc1 = memref.alloc(%arg1, %arg0) {__b__}: memref<?x?xf32>
    // CHECK: "some_use"(%[[VIEW0]])
    "some_use"(%alloc) : (memref<?x?xf32>) -> ()
    // CHECK: "some_other_use"(%[[VIEW1]])
    "some_other_use"(%alloc1) : (memref<?x?xf32>) -> ()
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match attributes {__a__} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [1000] unit_mode = "per_byte" : !transform.any_op 
    %1 = transform.structured.match attributes {__b__} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %1 static_buffer_sizes = [2000] unit_mode = "per_byte" : !transform.any_op 
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_annotation(%arg0: index, %arg1: index) {
    // CHECK: %[[RET0:.*]] = "some_op"()
    %op1 = "some_op"() : () -> (tensor<?x?xf32>)
    // CHECK: annotation.mark %[[RET0]] {buffer_size_in_byte = 1000 : i64} : tensor<?x?xf32>
    // CHECK: %[[RET1:.*]]:2 = "some_other_op"()
    %op2:2 = "some_other_op"() : () -> (tensor<?x?xf32>, tensor<?x?xf32>)
    // CHECK-DAG: annotation.mark %[[RET1]]#0 {buffer_size_in_byte = 2000 : i64} : tensor<?x?xf32>
    // CHECK-DAG: annotation.mark %[[RET1]]#1 {buffer_size_in_byte = 2000 : i64} : tensor<?x?xf32>
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["some_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [1000] unit_mode = "per_byte" : !transform.any_op 
    %1 = transform.structured.match ops{["some_other_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %1 static_buffer_sizes = [2000] unit_mode = "per_byte" : !transform.any_op 
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_static_shape(%arg0: index, %arg1: index) {
    %op1 = "some_op"() : () -> (tensor<10xf32>)
    // CHECK-NOT: annotation.mark
    %alloc1 = memref.alloc() : memref<10xf32>
    // CHECK-NOT: annotation.mark
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["some_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [1000] unit_mode = "per_byte" : !transform.any_op 
    %1 = transform.structured.match ops{["memref.alloc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %1 static_buffer_sizes = [1000] unit_mode = "per_byte" : !transform.any_op 
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_reference_type(%arg0: index, %arg1: index) {
    // CHECK: %[[ALLOCA0:.*]] = memref.alloca() : memref<4000xi8>
    // CHECK: %[[CONST0:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW0:.*]] = memref.view %[[ALLOCA0]][%[[CONST0]]][%[[INDEX0:.*]], %[[INDEX1:.*]]] : memref<4000xi8> to memref<?x?xf32>
    %alloca = memref.alloca(%arg0, %arg1) : memref<?x?xf32>
    // CHECK: %[[ALLOCA1:.*]] = memref.alloca() : memref<4000xi8>
    // CHECK: %[[CONST1:.*]] = arith.constant 0 : index
    // CHECK: %[[VIEW1:.*]] = memref.view %[[ALLOCA1]][%[[CONST1]]][%[[INDEX1]], %[[INDEX0]]] : memref<4000xi8> to memref<?x?xf32>
    %alloca1 = memref.alloca(%arg1, %arg0) : memref<?x?xf32>
    // CHECK: "some_use"(%[[VIEW0]])
    "some_use"(%alloca) : (memref<?x?xf32>) -> ()
    // CHECK: "some_other_use"(%[[VIEW1]])
    "some_other_use"(%alloca1) : (memref<?x?xf32>) -> ()
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.alloca"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [1000] unit_mode = "per_byte" reference_type = i8 : !transform.any_op 
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_reference_type_tensor(%arg0: index, %arg1: index) {
    // CHECK: %[[RET:.*]]:2 = "some_op"()
    %op:2 = "some_op"() : () -> (tensor<?x?xf32>, tensor<?x?xf32>)
    // CHECK-DAG: annotation.mark %[[RET]]#0 {buffer_size_in_byte = 4000 : i64} : tensor<?x?xf32>
    // CHECK-DAG: annotation.mark %[[RET]]#1 {buffer_size_in_byte = 4000 : i64} : tensor<?x?xf32>
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["some_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [1000] unit_mode = "per_byte" reference_type = i8 : !transform.any_op 
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // CHECK: test_none_shaped_type
  func.func @test_none_shaped_type() {
    // CHECK-NOT: buffer_size_in_byte
    %op = "some_op"() : () -> (i32)
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["some_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [1000] unit_mode = "per_byte" : !transform.any_op 
    transform.yield 
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_multiple_results_with_different_type() {
    // CHECK: %[[RET:.*]]:3 = "some_op"()
    %op:3 = "some_op"() : () -> (tensor<?xi8>, tensor<?xi16>, tensor<?xi1>)
    // CHECK-DAG: annotation.mark %[[RET]]#0 {buffer_size_in_byte = 10000 : i64} : tensor<?xi8>
    // CHECK-DAG: annotation.mark %[[RET]]#1 {buffer_size_in_byte = 20000 : i64} : tensor<?xi16>
    // CHECK-DAG: annotation.mark %[[RET]]#2 {buffer_size_in_byte = 10000 : i64} : tensor<?xi1>
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["some_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [10000] unit_mode = "per_element" : !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func @set_buffer_size_multiple_results_i1_byte_mode() {
    // CHECK: %[[RET:.*]] = "some_op"()
    %op = "some_op"() : () -> (tensor<?xi1>)
    // CHECK: annotation.mark %[[RET]] {buffer_size_in_byte = 10000 : i64} : tensor<?xi1>
    return
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["some_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.set_buffer_size %0 static_buffer_sizes = [10000] unit_mode = "per_byte" : !transform.any_op
    transform.yield
  }
}