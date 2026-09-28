// RUN: bishengir-opt %s --auto-scope --split-input-file | FileCheck %s

// Protect a tensor destination dependency, not just the indices path. Memref
// dependencies already stop at the default AutoScope boundary, while this
// tensor-valued region needs the dynamic-insert-slice subtree check.
// CHECK-LABEL: func.func @dynamic_slice_in_tensor_destination
// CHECK: %[[DST:.*]] = scf.if
// CHECK: tensor.insert_slice
// CHECK: scope.scope
// CHECK-NOT: scf.if
// CHECK-NOT: tensor.insert_slice
// CHECK: hivm.hir.gather_load {{.*}} outs(%[[DST]] : tensor<8xi32>)
// CHECK-NEXT: scope.return
module {
  func.func @dynamic_slice_in_tensor_destination(%base: memref<?xi32>, %indices: tensor<8xi32>, %a: tensor<8xi32>, %b: tensor<8xi32>, %size: index, %cond: i1) -> tensor<8xi32> {
    %c1 = arith.constant 1 : i32
    %dst = scf.if %cond -> tensor<8xi32> {
      %prefix = tensor.extract_slice %a[0] [%size] [1] : tensor<8xi32> to tensor<?xi32>
      %merged = tensor.insert_slice %prefix into %b[0] [%size] [1] : tensor<?xi32> into tensor<8xi32>
      scf.yield %merged : tensor<8xi32>
    } else {
      scf.yield %b : tensor<8xi32>
    }
    %result = hivm.hir.gather_load ins(%base : memref<?xi32>, %indices : tensor<8xi32>, %c1 : i32) outs(%dst : tensor<8xi32>) -> tensor<8xi32>
    return %result : tensor<8xi32>
  }
}

// -----

// MoE small_grouped computes scatter indices with a dynamic prefix update in
// scf.if. Cloning the whole if used to bypass the direct insert_slice boundary,
// leaving an unsupported dynamic slice in the outlined SIMT function.
// CHECK-LABEL: func.func @dynamic_slice_in_if
// CHECK: %[[INDICES:.*]] = scf.if
// CHECK: tensor.extract_slice
// CHECK: tensor.insert_slice
// CHECK: scope.scope
// CHECK-NOT: scf.if
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: hivm.hir.scatter_store ins(%[[INDICES]] : tensor<8xi32>
// CHECK-NEXT: scope.return
module {
  func.func @dynamic_slice_in_if(%base: memref<?xi32>, %a: tensor<8xi32>, %b: tensor<8xi32>, %size: index, %cond: i1) {
    %c1 = arith.constant 1 : i32
    %indices = scf.if %cond -> tensor<8xi32> {
      %prefix = tensor.extract_slice %a[0] [%size] [1] : tensor<8xi32> to tensor<?xi32>
      %merged = tensor.insert_slice %prefix into %b[0] [%size] [1] : tensor<?xi32> into tensor<8xi32>
      scf.yield %merged : tensor<8xi32>
    } else {
      scf.yield %b : tensor<8xi32>
    }
    hivm.hir.scatter_store ins(%indices : tensor<8xi32>, %a : tensor<8xi32>, %c1 : i32) outs(%base : memref<?xi32>)
    return
  }
}

// -----

// Check the whole cloned subtree, including an else region nested in a loop.
// CHECK-LABEL: func.func @dynamic_slice_in_nested_region
// CHECK: %[[INDICES:.*]] = scf.for
// CHECK: scf.if
// CHECK: tensor.insert_slice
// CHECK: scope.scope
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK-NOT: tensor.extract_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: hivm.hir.gather_load ins({{.*}}, %[[INDICES]] : tensor<8xi32>
// CHECK-NEXT: scope.return
module {
  func.func @dynamic_slice_in_nested_region(%base: memref<?xi32>, %a: tensor<8xi32>, %b: tensor<8xi32>, %size: index, %cond: i1, %count: index) -> tensor<8xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %indices = scf.for %i = %c0 to %count step %c1 iter_args(%acc = %b) -> tensor<8xi32> {
      %next = scf.if %cond -> tensor<8xi32> {
        scf.yield %acc : tensor<8xi32>
      } else {
        %prefix = tensor.extract_slice %a[0] [%size] [1] : tensor<8xi32> to tensor<?xi32>
        %merged = tensor.insert_slice %prefix into %acc[0] [%size] [1] : tensor<?xi32> into tensor<8xi32>
        scf.yield %merged : tensor<8xi32>
      }
      scf.yield %next : tensor<8xi32>
    }
    %empty = tensor.empty() : tensor<8xi32>
    %result = hivm.hir.gather_load ins(%base : memref<?xi32>, %indices : tensor<8xi32>, %c1_i32 : i32) outs(%empty : tensor<8xi32>) -> tensor<8xi32>
    return %result : tensor<8xi32>
  }
}

// -----

// Do not disable region cloning wholesale: a supported conditional dependency
// remains eligible for SIMT scope placement.
// CHECK-LABEL: func.func @supported_if_still_included
// CHECK: scope.scope
// CHECK: %[[INDICES:.*]] = scf.if
// CHECK: hivm.hir.gather_load ins({{.*}}, %[[INDICES]] : tensor<8xi32>
// CHECK-NEXT: scope.return
module {
  func.func @supported_if_still_included(%base: memref<?xi32>, %a: tensor<8xi32>, %b: tensor<8xi32>, %cond: i1) -> tensor<8xi32> {
    %c1 = arith.constant 1 : i32
    %indices = scf.if %cond -> tensor<8xi32> {
      scf.yield %a : tensor<8xi32>
    } else {
      scf.yield %b : tensor<8xi32>
    }
    %empty = tensor.empty() : tensor<8xi32>
    %result = hivm.hir.gather_load ins(%base : memref<?xi32>, %indices : tensor<8xi32>, %c1 : i32) outs(%empty : tensor<8xi32>) -> tensor<8xi32>
    return %result : tensor<8xi32>
  }
}

// -----

// Static slices are not the unsupported dynamic-sized case. Preserve the
// existing AutoScope eligibility policy for them, including within regions.
// CHECK-LABEL: func.func @static_slice_in_if_still_included
// CHECK: scope.scope
// CHECK: %[[INDICES:.*]] = scf.if
// CHECK: tensor.insert_slice
// CHECK: hivm.hir.gather_load ins({{.*}}, %[[INDICES]] : tensor<8xi32>
// CHECK-NEXT: scope.return
module {
  func.func @static_slice_in_if_still_included(%base: memref<?xi32>, %a: tensor<4xi32>, %b: tensor<8xi32>, %cond: i1) -> tensor<8xi32> {
    %c1 = arith.constant 1 : i32
    %indices = scf.if %cond -> tensor<8xi32> {
      %merged = tensor.insert_slice %a into %b[0] [4] [1] : tensor<4xi32> into tensor<8xi32>
      scf.yield %merged : tensor<8xi32>
    } else {
      scf.yield %b : tensor<8xi32>
    }
    %empty = tensor.empty() : tensor<8xi32>
    %result = hivm.hir.gather_load ins(%base : memref<?xi32>, %indices : tensor<8xi32>, %c1 : i32) outs(%empty : tensor<8xi32>) -> tensor<8xi32>
    return %result : tensor<8xi32>
  }
}
