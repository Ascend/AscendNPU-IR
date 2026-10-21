// RUN: bishengir-opt %s -fold-extract-insert-pair -canonicalize | FileCheck %s

// Recover a standalone select, preserve the named consumer and other source
// users, and omit the predicate for the fully covered column dimension.
// CHECK-LABEL: func.func @prefix
// CHECK-SAME: %[[X:.*]]: tensor<4x1024xf32>, %[[A:.*]]: tensor<4x1024xf32>, %[[N:.*]]: index
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[N32:.*]] = arith.index_cast %[[N]] : index to i32
// CHECK: %[[MASK:.*]] = linalg.generic
// CHECK-SAME: ins(%[[X]] : tensor<4x1024xf32>)
// CHECK: ^bb0(%[[XV:.*]]: f32, %{{.*}}: f32):
// CHECK: %[[ROW:.*]] = linalg.index 0 : index
// CHECK-NEXT: %[[ROW32:.*]] = arith.index_cast %[[ROW]] : index to i32
// CHECK-NEXT: %[[P:.*]] = arith.cmpi ult, %[[ROW32]], %[[N32]] : i32
// CHECK-NEXT: %[[SEL:.*]] = arith.select %[[P]], %[[XV]], %[[ZERO]] : f32
// CHECK-NEXT: linalg.yield %[[SEL]] : f32
// CHECK-NEXT: } -> tensor<4x1024xf32>
// CHECK-NEXT: %[[OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK-SAME: ins(%[[A]], %[[MASK]]
// CHECK-NEXT: return %[[OUT]], %[[X]]
func.func @prefix(%x: tensor<4x1024xf32>, %a: tensor<4x1024xf32>, %n: index)
    -> (tensor<4x1024xf32>, tensor<4x1024xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<4x1024xf32>
  %base = linalg.fill ins(%zero : f32) outs(%empty : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  %slice = tensor.extract_slice %x[0, 0] [%n, 1024] [1, 1] : tensor<4x1024xf32> to tensor<?x1024xf32>
  %masked = tensor.insert_slice %slice into %base[0, 0] [%n, 1024] [1, 1] : tensor<?x1024xf32> into tensor<4x1024xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %masked : tensor<4x1024xf32>, tensor<4x1024xf32>) outs(%empty : tensor<4x1024xf32>) -> tensor<4x1024xf32>
  return %out, %x : tensor<4x1024xf32>, tensor<4x1024xf32>
}

// Combine the bounds of both dimensions and preserve an arbitrary scalar fill.
// CHECK-LABEL: func.func @rectangle
// CHECK-SAME: %[[X:.*]]: tensor<8x16xf32>, %{{.*}}: tensor<8x16xf32>, %[[R:.*]]: index, %[[C:.*]]: index, %[[ROWS:.*]]: index, %[[COLS:.*]]: index, %[[PAD:.*]]: f32
// CHECK-DAG: %[[R32:.*]] = arith.index_cast %[[R]] : index to i32
// CHECK-DAG: %[[ROWS32:.*]] = arith.index_cast %[[ROWS]] : index to i32
// CHECK-DAG: %[[C32:.*]] = arith.index_cast %[[C]] : index to i32
// CHECK-DAG: %[[COLS32:.*]] = arith.index_cast %[[COLS]] : index to i32
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[X]] : tensor<8x16xf32>)
// CHECK: ^bb0(%[[XV:.*]]: f32, %{{.*}}: f32):
// CHECK: %[[ROW:.*]] = linalg.index 0 : index
// CHECK: %[[ROW32:.*]] = arith.index_cast %[[ROW]] : index to i32
// CHECK: %[[DR:.*]] = arith.subi %[[ROW32]], %[[R32]] : i32
// CHECK: %[[RU:.*]] = arith.cmpi ult, %[[DR]], %[[ROWS32]] : i32
// CHECK: %[[RL:.*]] = arith.cmpi uge, %[[ROW32]], %[[R32]] : i32
// CHECK: %[[INROW:.*]] = arith.andi %[[RL]], %[[RU]] : i1
// CHECK: %[[COL:.*]] = linalg.index 1 : index
// CHECK: %[[COL32:.*]] = arith.index_cast %[[COL]] : index to i32
// CHECK: %[[DC:.*]] = arith.subi %[[COL32]], %[[C32]] : i32
// CHECK: %[[CU:.*]] = arith.cmpi ult, %[[DC]], %[[COLS32]] : i32
// CHECK: %[[CL:.*]] = arith.cmpi uge, %[[COL32]], %[[C32]] : i32
// CHECK: %[[INCOL:.*]] = arith.andi %[[CL]], %[[CU]] : i1
// CHECK: %[[INSIDE:.*]] = arith.andi %[[INROW]], %[[INCOL]] : i1
// CHECK-NEXT: %[[SEL:.*]] = arith.select %[[INSIDE]], %[[XV]], %[[PAD]] : f32
// CHECK-NEXT: linalg.yield %[[SEL]] : f32
func.func @rectangle(%x: tensor<8x16xf32>, %a: tensor<8x16xf32>,
    %r: index, %c: index, %rows: index, %cols: index, %pad: f32)
    -> tensor<8x16xf32> {
  %empty = tensor.empty() : tensor<8x16xf32>
  %base = linalg.fill ins(%pad : f32) outs(%empty : tensor<8x16xf32>) -> tensor<8x16xf32>
  %slice = tensor.extract_slice %x[%r, %c] [%rows, %cols] [1, 1] : tensor<8x16xf32> to tensor<?x?xf32>
  %masked = tensor.insert_slice %slice into %base[%r, %c] [%rows, %cols] [1, 1] : tensor<?x?xf32> into tensor<8x16xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%a, %masked : tensor<8x16xf32>, tensor<8x16xf32>) outs(%empty : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %out : tensor<8x16xf32>
}

// Preserve the implicit conversion from the fill scalar to the element type.
// CHECK-LABEL: func.func @different_fill_type
// CHECK: %[[BASE:.*]] = linalg.fill ins(%{{.*}} : i32)
// CHECK: tensor.insert_slice %{{.*}} into %[[BASE]]
// CHECK-NOT: arith.select
// CHECK: return
func.func @different_fill_type(%x: tensor<8xf32>, %a: tensor<8xf32>, %n: index, %pad: i32) -> tensor<8xf32> {
  %base = linalg.fill ins(%pad : i32) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  %slice = tensor.extract_slice %x[0] [%n] [1] : tensor<8xf32> to tensor<?xf32>
  %masked = tensor.insert_slice %slice into %base[0] [%n] [1] : tensor<?xf32> into tensor<8xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %masked : tensor<8xf32>, tensor<8xf32>) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// Equal sizes do not imply equal coordinates.
// CHECK-LABEL: func.func @different_offsets
// CHECK: tensor.insert_slice
func.func @different_offsets(%x: tensor<8xf32>, %a: tensor<8xf32>, %n: index) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %base = linalg.fill ins(%zero : f32) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  %slice = tensor.extract_slice %x[0] [%n] [1] : tensor<8xf32> to tensor<?xf32>
  %masked = tensor.insert_slice %slice into %base[1] [%n] [1] : tensor<?xf32> into tensor<8xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %masked : tensor<8xf32>, tensor<8xf32>) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// A full-domain read of a smaller source would be out of bounds.
// CHECK-LABEL: func.func @different_full_shapes
// CHECK: tensor.insert_slice
func.func @different_full_shapes(%x: tensor<4xf32>, %a: tensor<8xf32>, %n: index) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %base = linalg.fill ins(%zero : f32) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  %slice = tensor.extract_slice %x[0] [%n] [1] : tensor<4xf32> to tensor<?xf32>
  %masked = tensor.insert_slice %slice into %base[0] [%n] [1] : tensor<?xf32> into tensor<8xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %masked : tensor<8xf32>, tensor<8xf32>) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// Equal dynamic types do not prove equal runtime shapes.
// CHECK-LABEL: func.func @dynamic_full_shapes
// CHECK: tensor.insert_slice
func.func @dynamic_full_shapes(%x: tensor<?xf32>, %a: tensor<?xf32>, %n: index) -> tensor<?xf32> {
  %zero = arith.constant 0.0 : f32
  %base = linalg.fill ins(%zero : f32) outs(%a : tensor<?xf32>) -> tensor<?xf32>
  %slice = tensor.extract_slice %x[0] [%n] [1] : tensor<?xf32> to tensor<?xf32>
  %masked = tensor.insert_slice %slice into %base[0] [%n] [1] : tensor<?xf32> into tensor<?xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %masked : tensor<?xf32>, tensor<?xf32>) outs(%a : tensor<?xf32>) -> tensor<?xf32>
  return %out : tensor<?xf32>
}

// The initial implementation deliberately excludes strided regions.
// CHECK-LABEL: func.func @strided_slice
// CHECK: tensor.insert_slice
func.func @strided_slice(%x: tensor<8xf32>, %a: tensor<8xf32>, %n: index) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %base = linalg.fill ins(%zero : f32) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  %slice = tensor.extract_slice %x[0] [%n] [2] : tensor<8xf32> to tensor<?xf32>
  %masked = tensor.insert_slice %slice into %base[0] [%n] [2] : tensor<?xf32> into tensor<8xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %masked : tensor<8xf32>, tensor<8xf32>) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}

// Do not duplicate the selection or change other users.
// CHECK-LABEL: func.func @multiple_consumers
// CHECK: tensor.insert_slice
func.func @multiple_consumers(%x: tensor<8xf32>, %a: tensor<8xf32>, %n: index) -> tensor<8xf32> {
  %zero = arith.constant 0.0 : f32
  %base = linalg.fill ins(%zero : f32) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  %slice = tensor.extract_slice %x[0] [%n] [1] : tensor<8xf32> to tensor<?xf32>
  %masked = tensor.insert_slice %slice into %base[0] [%n] [1] : tensor<?xf32> into tensor<8xf32>
  %first = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %masked : tensor<8xf32>, tensor<8xf32>) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  %out = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%first, %masked : tensor<8xf32>, tensor<8xf32>) outs(%a : tensor<8xf32>) -> tensor<8xf32>
  return %out : tensor<8xf32>
}
