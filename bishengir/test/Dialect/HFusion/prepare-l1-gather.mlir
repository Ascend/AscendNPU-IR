// RUN: bishengir-opt %s --hfusion-prepare-l1-gather | FileCheck %s
// RUN: sed 's/annotation.mark %%selected/annotation.mark %%tile/' %s | bishengir-opt --hfusion-prepare-l1-gather | FileCheck %s
// RUN: sed 's/enable_l1_gather = true/enable_l1_gather = false/g' %s | bishengir-opt --hfusion-prepare-l1-gather | FileCheck %s --check-prefix=DISABLED
// RUN: sed '/annotation.mark.*hivm.enable_l1_gather/d' %s | bishengir-opt --hfusion-prepare-l1-gather | FileCheck %s --check-prefix=DISABLED
// RUN: sed 's/dense<0.0>/dense<1.0>/g' %s | bishengir-opt --hfusion-prepare-l1-gather | FileCheck %s --check-prefix=DISABLED
// RUN: sed '/^ *%%.* = tensor.extract /i\    memref.store %%id_zero, %%ids[%%i] : memref<16xi32>' %s | bishengir-opt --hfusion-prepare-l1-gather | FileCheck %s --check-prefix=DISABLED

// Reduced from the sparse MLA row mask. Arithmetic is scalarized through its
// operation traits, including operations outside the former add/and/cmp list.
// CHECK-LABEL: func.func @row_mask
// CHECK: linalg.fill {{.*}} outs({{.*}} : memref<16x16xbf16>)
// CHECK: scf.for
// CHECK: memref.load
// CHECK: arith.xori {{.*}} : i32
// CHECK: arith.cmpi sge
// CHECK: scf.if
// CHECK: memref.copy {{.*}} to memref<1x16xbf16
// CHECK-NOT: arith.select
// CHECK: annotation.mark
// CHECK: linalg.matmul
// DISABLED-LABEL: func.func @row_mask
// DISABLED: tensor.extract
// DISABLED: arith.select
func.func @row_mask(%ids: memref<16xi32>, %gm: memref<?xbf16>,
                    %q: tensor<16x16xbf16>, %acc: tensor<16x16xf32>)
    -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : bf16
  %zeros = arith.constant dense<0> : tensor<16xi32>
  %other = arith.constant dense<0.0> : tensor<16x16xbf16>
  %id_zero = arith.constant 0 : i32
  %indices = memref.alloc() : memref<16xi32>
  linalg.fill ins(%id_zero : i32) outs(%indices : memref<16xi32>)
  %prefix = memref.subview %indices[0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  memref.copy %ids, %prefix : memref<16xi32> to memref<16xi32, strided<[1]>>
  %id_tensor = bufferization.to_tensor %indices restrict writable : memref<16xi32>
  %adjusted = arith.xori %id_tensor, %zeros : tensor<16xi32>
  %valid = arith.cmpi sge, %adjusted, %zeros : tensor<16xi32>
  %empty = tensor.empty() : tensor<16x16xi1>
  %mask = linalg.broadcast ins(%valid : tensor<16xi1>) outs(%empty : tensor<16x16xi1>) dimensions = [1]
  %buffer = memref.alloc() : memref<16x16xbf16>
  scf.for %i = %c0 to %c16 step %c1 {
    %id = tensor.extract %adjusted[%i] : tensor<16xi32>
    %index = arith.index_cast %id : i32 to index
    %offset = arith.muli %index, %c16 : index
    %src = memref.reinterpret_cast %gm to offset: [%offset], sizes: [1, 16], strides: [16, 1] : memref<?xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
    %row = memref.subview %buffer[%i, 0] [1, 16] [1, 1] : memref<16x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
    memref.copy %src, %row : memref<1x16xbf16, strided<[16, 1], offset: ?>> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
  } {hivm.parallel_loop}
  %tile = bufferization.to_tensor %buffer restrict writable : memref<16x16xbf16>
  %selected = arith.select %mask, %tile, %other : tensor<16x16xi1>, tensor<16x16xbf16>
  annotation.mark %selected {hivm.enable_l1_gather = true} keys = ["write"] : tensor<16x16xbf16>
  %result = linalg.matmul ins(%q, %selected : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

// A dynamic prefix is zero outside the inserted slice. Limit its side-effect
// free row loop to the slice size, without matching the arithmetic that
// computes either value. Cached metadata from one branch must not be reused
// in its sibling, even when the tensor and indices are identical.
// CHECK-LABEL: func.func @row_prefix
// CHECK: arith.minsi
// CHECK: linalg.fill {{.*}} outs({{.*}} : memref<16x16xbf16>)
// CHECK: scf.for
// CHECK: %[[ID:.*]] = scf.if
// CHECK: arith.cmpi ult
// CHECK: %[[LEFT:.*]] = scf.if
// CHECK: memref.load
// CHECK: scf.yield %[[LEFT]]
// CHECK-NEXT: } else {
// CHECK: arith.cmpi ult
// CHECK: %[[RIGHT:.*]] = scf.if
// CHECK: memref.load
// CHECK: scf.yield %[[RIGHT]]
// CHECK-NEXT: }
// CHECK: arith.index_cast %[[ID]]
// CHECK: memref.copy {{.*}} to memref<1x16xbf16
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: annotation.mark
// DISABLED-LABEL: func.func @row_prefix
// DISABLED: tensor.extract_slice
// DISABLED: tensor.insert_slice
func.func @row_prefix(%ids: memref<16xi32>, %gm: memref<?xbf16>,
                      %bound: index, %count: index) -> tensor<16x16xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %id_zero = arith.constant 0 : i32
  %other = arith.constant dense<0.0> : tensor<16x16xbf16>
  %indices = memref.alloc() : memref<16xi32>
  linalg.fill ins(%id_zero : i32) outs(%indices : memref<16xi32>)
  %prefix = memref.subview %indices[0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  memref.copy %ids, %prefix : memref<16xi32> to memref<16xi32, strided<[1]>>
  %id_tensor = bufferization.to_tensor %indices restrict writable : memref<16xi32>
  %buffer = memref.alloc() : memref<16x16xbf16>
  scf.for %i = %c0 to %bound step %c1 {
    %choose = arith.cmpi slt, %i, %count : index
    %id = scf.if %choose -> (i32) {
      %left = tensor.extract %id_tensor[%i] : tensor<16xi32>
      scf.yield %left : i32
    } else {
      %right = tensor.extract %id_tensor[%i] : tensor<16xi32>
      scf.yield %right : i32
    }
    %index = arith.index_cast %id : i32 to index
    %offset = arith.muli %index, %c16 : index
    %src = memref.reinterpret_cast %gm to offset: [%offset], sizes: [1, 16], strides: [16, 1] : memref<?xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
    %row = memref.subview %buffer[%i, 0] [1, 16] [1, 1] : memref<16x16xbf16> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
    memref.copy %src, %row : memref<1x16xbf16, strided<[16, 1], offset: ?>> to memref<1x16xbf16, strided<[16, 1], offset: ?>>
  } {hivm.parallel_loop}
  %tile = bufferization.to_tensor %buffer restrict writable : memref<16x16xbf16>
  %active = tensor.extract_slice %tile[0, 0] [%count, 16] [1, 1] : tensor<16x16xbf16> to tensor<?x16xbf16>
  %result = tensor.insert_slice %active into %other[0, 0] [%count, 16] [1, 1] : tensor<?x16xbf16> into tensor<16x16xbf16>
  annotation.mark %result {hivm.enable_l1_gather = true} keys = ["write"] : tensor<16x16xbf16>
  return %result : tensor<16x16xbf16>
}
