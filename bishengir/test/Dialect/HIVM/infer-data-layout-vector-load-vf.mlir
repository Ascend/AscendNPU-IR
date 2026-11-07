// RUN: bishengir-opt %s --hivm-infer-data-layout | FileCheck %s

// A vector load in an unsplit mixed kernel must preserve its outlined VF's
// rank-one argument. Only the cube load needs the rank-two layout expansion.
module attributes {hacc.target = #hacc.target<"Ascend960DT_968B7">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  // CHECK-LABEL: func.func private @vector_consumer(
  // CHECK-SAME: memref<16xf32, #hivm.address_space<ub>>
  func.func private @vector_consumer(%input: memref<16xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    return
  }

  // CHECK-LABEL: func.func @mixed_vector_load(
  // CHECK: %[[VECTOR:.*]] = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.load ins(%{{.*}} : memref<16xf32, #hivm.address_space<gm>>) outs(%[[VECTOR]] : memref<16xf32, #hivm.address_space<ub>>)
  // CHECK: call @vector_consumer(%[[VECTOR]])
  func.func @mixed_vector_load(%input: memref<16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %buffer = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%input : memref<16xf32, #hivm.address_space<gm>>) outs(%buffer : memref<16xf32, #hivm.address_space<ub>>) core_type = <VECTOR>
    func.call @vector_consumer(%buffer) {hivm.vector_function, no_inline} : (memref<16xf32, #hivm.address_space<ub>>) -> ()
    return
  }

  // CHECK-LABEL: func.func @cube_load(
  // CHECK: %[[CUBE:.*]] = memref.alloc() : memref<1x16xf32, #hivm.address_space<cbuf>>
  // CHECK: %[[EXPANDED:.*]] = memref.expand_shape %{{.*}}
  // CHECK: hivm.hir.load ins(%[[EXPANDED]] : memref<1x16xf32, #hivm.address_space<gm>>) outs(%[[CUBE]] : memref<1x16xf32, #hivm.address_space<cbuf>>)
  func.func @cube_load(%input: memref<16xf32, #hivm.address_space<gm>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>} {
    %buffer = memref.alloc() : memref<16xf32, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%input : memref<16xf32, #hivm.address_space<gm>>) outs(%buffer : memref<16xf32, #hivm.address_space<cbuf>>) core_type = <CUBE>
    return
  }
}
