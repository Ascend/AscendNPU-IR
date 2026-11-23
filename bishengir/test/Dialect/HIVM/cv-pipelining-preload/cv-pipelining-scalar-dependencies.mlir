// RUN: bishengir-opt -cv-pipelining="pipeline-mode=unroll set-depth-in-unroll-mode=2" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,UNROLL --enable-var-scope
// RUN: bishengir-opt -cv-pipelining="pipeline-mode=skew" -split-input-file %s | FileCheck %s --check-prefixes=CHECK,SKEW --enable-var-scope

// CHECK-LABEL: func.func @scalar_dependency_flat(
// UNROLL: %[[BUFFER:.*]] = memref.alloc() : memref<2x1x1xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.vcast
// CHECK-NOT: tensor.extract {{.*}} : tensor<1xf32>
// CHECK-NOT: arith.divf
// CHECK-NOT: hivm.hir.vadd
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// UNROLL: %[[VIEW:.*]] = memref.subview %[[BUFFER]]
// SKEW: %[[CUBE:.*]] = scope.scope : () -> tensor<1x1xf32>
// CHECK: hivm.hir.mmadL1
// UNROLL: hivm.hir.fixpipe {{.*}} outs(%[[VIEW]] :
// SKEW: hivm.hir.fixpipe
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// UNROLL: %[[TENSOR:.*]] = bufferization.to_tensor %[[BUFFER]] restrict writable
// UNROLL: %[[SLICE:.*]] = tensor.extract_slice %[[TENSOR]]
// UNROLL: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SLICE]]
// SKEW: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CUBE]]
// CHECK: %[[SCALAR:.*]] = tensor.extract %[[COLLAPSED]][
// CHECK: %[[SCALE:.*]] = arith.divf %[[SCALAR]], %{{.*}} : f32
// CHECK: hivm.hir.vadd ins({{.*}}, %[[SCALE]] : tensor<32xf32>, f32)
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.store
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @scalar_dependency_flat(%x: tensor<32xbf16>, %lhs: tensor<4x1x16x8xf32>, %rhs: tensor<1x2x16x8xf32>, %out: memref<32xbf16>, %n: index, %cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %divisor = arith.constant 3.300000e+01 : f32
    %true = arith.constant true
    %empty = tensor.empty() : tensor<32xf32>
    %output = tensor.empty() : tensor<32xbf16>
    %acc = tensor.empty() : tensor<1x1x16x16xf32>
    scf.for %i = %c0 to %n step %c1 {
      %xf = hivm.hir.vcast ins(%x : tensor<32xbf16>) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
      %dot = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c1, %c32, %c1 : tensor<4x1x16x8xf32>, tensor<1x2x16x8xf32>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub = memref.alloc() : memref<1x1xf32, #hivm.address_space<ub>>
      %t = bufferization.to_tensor %ub : memref<1x1xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub : memref<1x1xf32, #hivm.address_space<ub>>)
      %collapsed = tensor.collapse_shape %t [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
      %scalar = tensor.extract %collapsed[%c0] : tensor<1xf32>
      %scale = arith.divf %scalar, %divisor : f32
      %result = hivm.hir.vadd ins(%xf, %scale : tensor<32xf32>, f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
      %cast = hivm.hir.vcast ins(%result : tensor<32xf32>) outs(%output : tensor<32xbf16>) -> tensor<32xbf16>
      hivm.hir.store ins(%cast : tensor<32xbf16>) outs(%out : memref<32xbf16>)
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @scalar_dependency_nested_inner(
// UNROLL: %[[BUFFER:.*]] = memref.alloc() : memref<2x1x1xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.vcast
// CHECK-NOT: tensor.extract {{.*}} : tensor<1xf32>
// CHECK-NOT: arith.divf
// CHECK-NOT: hivm.hir.vadd
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// UNROLL: %[[VIEW:.*]] = memref.subview %[[BUFFER]]
// SKEW: %[[CUBE:.*]] = scope.scope : () -> tensor<1x1xf32>
// CHECK: hivm.hir.mmadL1
// UNROLL: hivm.hir.fixpipe {{.*}} outs(%[[VIEW]] :
// SKEW: hivm.hir.fixpipe
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// UNROLL: %[[TENSOR:.*]] = bufferization.to_tensor %[[BUFFER]] restrict writable
// UNROLL: %[[SLICE:.*]] = tensor.extract_slice %[[TENSOR]]
// UNROLL: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SLICE]]
// SKEW: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CUBE]]
// CHECK: %[[SCALAR:.*]] = tensor.extract %[[COLLAPSED]][
// CHECK: %[[SCALE:.*]] = arith.divf %[[SCALAR]], %{{.*}} : f32
// CHECK: hivm.hir.vadd ins({{.*}}, %[[SCALE]] : tensor<32xf32>, f32)
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.store
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @scalar_dependency_nested_inner(%x: tensor<32xbf16>, %lhs: tensor<4x1x16x8xf32>, %rhs: tensor<1x2x16x8xf32>, %out: memref<32xbf16>, %n: index, %cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %divisor = arith.constant 3.300000e+01 : f32
    %true = arith.constant true
    %empty = tensor.empty() : tensor<32xf32>
    %output = tensor.empty() : tensor<32xbf16>
    %acc = tensor.empty() : tensor<1x1x16x16xf32>
    scf.for %outer = %c0 to %c1 step %c1 {
      scf.if %cond {
        %done = scf.while (%w = %c0) : (index) -> index {
          %keep = arith.cmpi slt, %w, %c1 : index
          scf.condition(%keep) %w : index
        } do {
        ^bb0(%w: index):
          scf.forall (%p) in (1) {
            scf.for %i = %c0 to %n step %c1 {
              %xf = hivm.hir.vcast ins(%x : tensor<32xbf16>) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
              %dot = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c1, %c32, %c1 : tensor<4x1x16x8xf32>, tensor<1x2x16x8xf32>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
              %ub = memref.alloc() : memref<1x1xf32, #hivm.address_space<ub>>
              %t = bufferization.to_tensor %ub : memref<1x1xf32, #hivm.address_space<ub>>
              hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub : memref<1x1xf32, #hivm.address_space<ub>>)
              %collapsed = tensor.collapse_shape %t [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
              %scalar = tensor.extract %collapsed[%c0] : tensor<1xf32>
              %scale = arith.divf %scalar, %divisor : f32
              %result = hivm.hir.vadd ins(%xf, %scale : tensor<32xf32>, f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
              %cast = hivm.hir.vcast ins(%result : tensor<32xf32>) outs(%output : tensor<32xbf16>) -> tensor<32xbf16>
              hivm.hir.store ins(%cast : tensor<32xbf16>) outs(%out : memref<32xbf16>)
            }
          }
          %next = arith.addi %w, %c1 : index
          scf.yield %next : index
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @scalar_dependency_nested_consumer(
// UNROLL: %[[BUFFER:.*]] = memref.alloc() : memref<2x1x1xf32, #hivm.address_space<ub>>
// CHECK: hivm.hir.vcast
// CHECK-NOT: tensor.extract {{.*}} : tensor<1xf32>
// CHECK-NOT: arith.divf
// CHECK-NOT: hivm.hir.vadd
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
// UNROLL: %[[VIEW:.*]] = memref.subview %[[BUFFER]]
// SKEW: %[[CUBE:.*]] = scope.scope : () -> tensor<1x1xf32>
// CHECK: hivm.hir.mmadL1
// UNROLL: hivm.hir.fixpipe {{.*}} outs(%[[VIEW]] :
// SKEW: hivm.hir.fixpipe
// CHECK: hivm.loop_core_type = #hivm.tcore_type<CUBE>
// UNROLL: %[[TENSOR:.*]] = bufferization.to_tensor %[[BUFFER]] restrict writable
// UNROLL: %[[SLICE:.*]] = tensor.extract_slice %[[TENSOR]]
// UNROLL: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SLICE]]
// SKEW: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CUBE]]
// CHECK: %[[SCALAR:.*]] = tensor.extract %[[COLLAPSED]][
// CHECK: %[[SCALE:.*]] = arith.divf %[[SCALAR]], %{{.*}} : f32
// CHECK: hivm.hir.vadd ins({{.*}}, %[[SCALE]] : tensor<32xf32>, f32)
// CHECK: hivm.hir.vcast
// CHECK: hivm.hir.store
// CHECK: hivm.loop_core_type = #hivm.tcore_type<VECTOR>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @scalar_dependency_nested_consumer(%x: tensor<32xbf16>, %lhs: tensor<4x1x16x8xf32>, %rhs: tensor<1x2x16x8xf32>, %out: memref<32xbf16>, %n: index, %cond: i1) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %divisor = arith.constant 3.300000e+01 : f32
    %true = arith.constant true
    %empty = tensor.empty() : tensor<32xf32>
    %output = tensor.empty() : tensor<32xbf16>
    %acc = tensor.empty() : tensor<1x1x16x16xf32>
    scf.for %i = %c0 to %n step %c1 {
      %xf = hivm.hir.vcast ins(%x : tensor<32xbf16>) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
      %dot = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%lhs, %rhs, %true, %c1, %c32, %c1 : tensor<4x1x16x8xf32>, tensor<1x2x16x8xf32>, i1, index, index, index) outs(%acc : tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32>
      %ub = memref.alloc() : memref<1x1xf32, #hivm.address_space<ub>>
      %t = bufferization.to_tensor %ub : memref<1x1xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%dot : tensor<1x1x16x16xf32>) outs(%ub : memref<1x1xf32, #hivm.address_space<ub>>)
      %collapsed = tensor.collapse_shape %t [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
      %scalar = tensor.extract %collapsed[%c0] : tensor<1xf32>
      %scale = arith.divf %scalar, %divisor : f32
      scf.if %cond {
        %done = scf.while (%w = %c0) : (index) -> index {
          %keep = arith.cmpi slt, %w, %c1 : index
          scf.condition(%keep) %w : index
        } do {
        ^bb0(%w: index):
          scf.forall (%p) in (1) {
            scf.for %inner = %c0 to %c1 step %c1 {
              %result = hivm.hir.vadd ins(%xf, %scale : tensor<32xf32>, f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
              %cast = hivm.hir.vcast ins(%result : tensor<32xf32>) outs(%output : tensor<32xbf16>) -> tensor<32xbf16>
              hivm.hir.store ins(%cast : tensor<32xbf16>) outs(%out : memref<32xbf16>)
            }
          }
          %next = arith.addi %w, %c1 : index
          scf.yield %next : index
        }
      }
    }
    return
  }
}
