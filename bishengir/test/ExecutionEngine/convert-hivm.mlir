// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-convert-hivm-to-upstream %s --split-input-file | FileCheck %s
// RUN: bishengir-opt --lower-for-cpu-runner-pipeline %s --split-input-file

func.func @tensor_direct_linalg_lowering(%a: tensor<1x?x10xf32>, %b: tensor<?x5x10xf32>, %c: tensor<5x?x10xf32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.abs
    %0 = hivm.hir.vabs ins(%a: tensor<1x?x10xf32>) outs(%c: tensor<5x?x10xf32>) broadcast = [0] -> tensor<5x?x10xf32>

    // CHECK: linalg.add
    %1 = hivm.hir.vadd ins(%b, %b: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: linalg.sub
    %2 = hivm.hir.vsub ins(%0, %1: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.mul
    %3 = hivm.hir.vmul ins(%1, %2: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.div
    %4 = hivm.hir.vdiv ins(%2, %3: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.max
    %5 = hivm.hir.vmax ins(%3, %4: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.min
    %6 = hivm.hir.vmin ins(%4, %5: tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.exp
    %7 = hivm.hir.vexp ins(%6: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.log
    %8 = hivm.hir.vln ins(%7: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.rsqrt
    %9 = hivm.hir.vrsqrt ins(%8: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.sqrt
    %10 = hivm.hir.vsqrt ins(%9: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.tanh
    %11 = hivm.hir.vtanh ins(%10: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.reciprocal
    %12 = hivm.hir.vrec ins(%11: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.select
    %13 = arith.constant true
    %14 = hivm.hir.vsel ins(%13, %c, %c: i1, tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%c: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.erf
    %15 = hivm.hir.verf ins(%14: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.copy
    %16 = hivm.hir.store ins(%15: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    // CHECK: linalg.transpose
    %17 = hivm.hir.vtranspose ins(%b: tensor<?x5x10xf32>) outs(%16: tensor<5x?x10xf32>) permutation = [1, 0, 2] -> tensor<5x?x10xf32>

    func.return %17: tensor<5x?x10xf32>
}

// -----

func.func @memref_direct_linalg_lowering(%a: memref<1x?x10xf32>, %b: memref<?x5x10xf32>, %c: memref<5x?x10xf32>, %d: memref<5x?x10xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.abs
    hivm.hir.vabs ins(%a: memref<1x?x10xf32>) outs(%c: memref<5x?x10xf32>) broadcast = [0]

    // CHECK: linalg.add
    hivm.hir.vadd ins(%b, %b: memref<?x5x10xf32>, memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) transpose = [1, 0, 2]

    // CHECK: linalg.sub
    hivm.hir.vsub ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.mul
    hivm.hir.vmul ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.div
    hivm.hir.vdiv ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.max
    hivm.hir.vmax ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.min
    hivm.hir.vmin ins(%c, %d: memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.exp
    hivm.hir.vexp ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.log
    hivm.hir.vln ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.rsqrt
    hivm.hir.vrsqrt ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.sqrt
    hivm.hir.vsqrt ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.tanh
    hivm.hir.vtanh ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.reciprocal
    hivm.hir.vrec ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.select
    %13 = arith.constant true
    hivm.hir.vsel ins(%13, %c, %c: i1, memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.erf
    hivm.hir.verf ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.copy
    hivm.hir.store ins(%c: memref<5x?x10xf32>) outs(%c: memref<5x?x10xf32>)

    // CHECK: linalg.transpose
    hivm.hir.vtranspose ins(%b: memref<?x5x10xf32>) outs(%c: memref<5x?x10xf32>) permutation = [1, 0, 2]

    func.return
}

// -----

func.func @elemwise_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    %0 = hivm.hir.vrelu ins(%a: tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    hivm.hir.vrelu ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
    hivm.hir.vnot ins(%b: memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    func.return %0: tensor<5x?x10xf32>
}

// -----

func.func @bitwise_like_lowering(%a: tensor<?x5x10xf32>, %aT: tensor<5x?x10xf32>, %b: memref<5x1x10xi32>, %bB: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.map {{.*}}
    // CHECK-2: arith.bitcast
    // CHECK:   arith.andi
    // CHECK:   arith.bitcast
    %0 = hivm.hir.vand ins(%a, %a: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%aT: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: linalg.map
    // CHECK-SAME:  arith.andi
    hivm.hir.vand ins(%b, %b: memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // CHECK: linalg.map {{.*}}
    // CHECK-2: arith.bitcast
    // CHECK:   arith.ori
    // CHECK:   arith.bitcast
    %1 = hivm.hir.vor ins(%a, %a: tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%0: tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

    // CHECK: linalg.map
    // CHECK-SAME:  arith.ori
    hivm.hir.vor ins(%b, %b: memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%bB: memref<5x?x10xi32>) broadcast = [1]

    // CHECK: linalg.map
    // CHECK-SAME:  arith.xori
    hivm.hir.vxor ins(%bB, %bB: memref<5x?x10xi32>, memref<5x?x10xi32>) outs(%bB: memref<5x?x10xi32>)

    func.return %1: tensor<5x?x10xf32>
}

// -----

func.func @cumulative_like_lowering(%a: tensor<5x?x10xf32>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: tensor<5x?x10xf32>, tensor<5x1x1xf32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: f32, %{{.*}}: f32, %[[out:.*]]: f32)
    // CHECK-NEXT:      %[[res:.*]] = arith.mulf
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    %0 = hivm.hir.vcumprod ins(%a: tensor<5x?x10xf32>) outs(%a: tensor<5x?x10xf32>) cum_dims = [0] reverse = false -> tensor<5x?x10xf32>

    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32)
    // CHECK-NEXT:      %[[res:.*]] = arith.muli
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    hivm.hir.vcumprod ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1] reverse = false

    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: tensor<5x?x10xf32>, tensor<5x1x1xf32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: f32, %{{.*}}: f32, %[[out:.*]]: f32)
    // CHECK-NEXT:      %[[res:.*]] = arith.addf
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    %1 = hivm.hir.vcumsum ins(%a: tensor<5x?x10xf32>) outs(%0: tensor<5x?x10xf32>) cum_dims = [0] reverse = false -> tensor<5x?x10xf32>

    // CHECK: linalg.generic
    // CHECK-SAME:  outs({{.*}}: memref<5x?x10xi32>, memref<5x?x1xi32>)
    // CHECK-NEXT:  ^bb0(%[[in:.*]]: i32, %{{.*}}: i32, %[[out:.*]]: i32)
    // CHECK-NEXT:      %[[res:.*]] = arith.addi
    // CHECK-DAG-SAME:      %[[in]]
    // CHECK-DAG-SAME:      %[[out]]
    // CHECK-NEXT:      linalg.yield %[[res]], %[[res]]
    hivm.hir.vcumsum ins(%b: memref<5x?x10xi32>) outs(%b: memref<5x?x10xi32>) cum_dims = [1] reverse = false

    func.return %0: tensor<5x?x10xf32>
}

// -----

// CHECK-DAG: #[[IdentityMap:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[NonIndexingMap:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[ZeroDim1Map:.*]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
// CHECK-DAG: #[[ZeroDim01Map:.*]] = affine_map<(d0, d1, d2) -> (0, 0, d2)>

// CHECK: @brc_lowering
func.func @brc_lowering(%a: tensor<1x1x10xf32>, %b: tensor<5x?x10xf32>, %c: memref<5x1x10xi32>, %d: memref<5x?x10xi32>) -> tensor<5x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    %c0_f = arith.constant 0.: f32

    // CHECK: linalg.generic
    // CHECK-SAME:  indexing_maps = [#[[ZeroDim01Map]], #[[IdentityMap]]]
    // CHECK-NEXT:  ^bb0(%[[in:[^:]*]]: f32
    // CHECK-NEXT:      linalg.yield %[[in]]
    %0 = hivm.hir.vbrc ins(%a: tensor<1x1x10xf32>) outs(%b: tensor<5x?x10xf32>) broadcast_dims = [0, 1] -> tensor<5x?x10xf32>

    // CHECK: linalg.generic
    // CHECK-SAME:  indexing_maps = [#[[NonIndexingMap]], #[[IdentityMap]]]
    // CHECK-NEXT:  ^bb0(%[[in:[^:]*]]: f32
    // CHECK-NEXT:      linalg.yield %[[in]]
    %1 = hivm.hir.vbrc ins(%c0_f: f32) outs(%0: tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

    %c0_i = arith.constant 0: i32

    // CHECK: linalg.generic
    // CHECK-SAME:  indexing_maps = [#[[ZeroDim1Map]], #[[IdentityMap]]]
    // CHECK-NEXT:  ^bb0(%[[in:[^:]*]]: i32
    // CHECK-NEXT:      linalg.yield %[[in]]
    hivm.hir.vbrc ins(%c: memref<5x1x10xi32>) outs(%d: memref<5x?x10xi32>) broadcast_dims = [1]

    // CHECK: linalg.generic
    // CHECK-SAME:  indexing_maps = [#[[NonIndexingMap]], #[[IdentityMap]]]
    // CHECK-NEXT:  ^bb0(%[[in:[^:]*]]: i32
    // CHECK-NEXT:      linalg.yield %[[in]]
    hivm.hir.vbrc ins(%c0_i: i32) outs(%d: memref<5x?x10xi32>)

    func.return %1: tensor<5x?x10xf32>
}

// -----

func.func @arange_lowering(%a: tensor<5x?x10xi64>, %b: memref<5x?x10xi32>) -> tensor<5x?x10xi64> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: %[[C0:.*]] = arith.constant 0
    %c0 = arith.constant 0: index
    // CHECK: %[[C1:.*]] = arith.constant 1
    %c1 = arith.constant 1: index
    // CHECK: %[[C2:.*]] = arith.constant 2
    %c2 = arith.constant 2: index
    // CHECK: %[[C3:.*]] = arith.constant 3
    %c3 = arith.constant 3: index

    // CHECK: hfusion.arange
    // CHECK-SAME:  strides[%[[C0]], %[[C3]], %[[C2]]]
    %0 = hivm.hir.varange offset[] strides[%c0, %c3, %c2] outs(%a: tensor<5x?x10xi64>) -> tensor<5x?x10xi64>

    // CHECK: hfusion.arange offset[%[[C3]]] strides[%[[C1]], %[[C1]], %[[C1]]]
    hivm.hir.varange offset[%c3] strides[%c1, %c1, %c1] outs(%b: memref<5x?x10xi32>)

    func.return %0: tensor<5x?x10xi64>
}

// -----

// CHECK-LABEL: @concat_lowering
// CHECK-SAME:      %[[a:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[b:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[c:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[d:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[e:[^:]*]]: {{[^,]*}}, 
// CHECK-SAME:      %[[f:[^:]*]]: {{[^,]*}}
func.func @concat_lowering(%a: tensor<5x?x10xf32>, %b: tensor<?x?x10xf32>, %c: tensor<?x?x10xf32>, %d: memref<5x?x10xi32>, %e: memref<?x?x10xi32>, %f: memref<?x?x10xi32>) -> tensor<?x?x10xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: tensor.concat
    %0 = hivm.hir.vconcat dim(0) ins(%a, %b: tensor<5x?x10xf32>, tensor<?x?x10xf32>) outs(%c: tensor<?x?x10xf32>) -> tensor<?x?x10xf32>

    // CHECK-DAG: %[[tensorD:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[tensorE:.*]] = bufferization.to_tensor %[[e]]
    // CHECK:   %[[concat:.*]] = tensor.concat dim(0)
    // CHECK-DAG-SAME:                      %[[tensorD]]
    // CHECK-DAG-SAME:                      %[[tensorE]]
    // CHECK:   bufferization.materialize_in_destination %[[concat]]
    // CHECK-SAME:                                          %[[f]]
    hivm.hir.vconcat dim(0) ins(%d, %e: memref<5x?x10xi32>, memref<?x?x10xi32>) outs(%f: memref<?x?x10xi32>)

    func.return %0: tensor<?x?x10xf32>
}

// -----

// CHECK-LABEL: @reduce_lowering
// CHECK-SAME:      %[[a:[^:]*]]
// CHECK-SAME:      %[[ai:[^:]*]]
// CHECK-SAME:      %[[id_t:[^:]*]]
// CHECK-SAME:      %[[b:[^:]*]]
// CHECK-SAME:      %[[bi:[^:]*]]
// CHECK-SAME:      %[[c:[^:]*]]
// CHECK-SAME:      %[[d:[^:]*]]
// CHECK-SAME:      %[[id_m:[^:]*]]
// CHECK-SAME:      %[[e:[^:]*]]
// CHECK-SAME:      %[[f:[^:]*]]
// CHECK-SAME:      %[[g:[^:]*]]
func.func @reduce_lowering(%a: tensor<5x?x10xf32>, %ai: tensor<5x?x10xi32>, %id_t: tensor<5x?x10xi32>, %b: tensor<5x1x10xf32>, %bi: tensor<5x1x10xi32>, %c: tensor<5x1x10xi32>, %d: memref<5x?x10xi32>, %id_m: memref<5x?x10xi32>, %e: memref<1x1x10xi32>, %f: memref<1x?x10xi32>, %g: memref<1x?x10xi32>) -> (tensor<5x1x10xf32>, tensor<5x1x10xi32>, tensor<5x1x10xi32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {

    // CHECK: %[[c0:.*]] = tensor.collapse_shape %[[b]]
    // CHECK: %[[r0:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[a]] :
    // CHECK-SAME:          outs(%[[c0]] :
    // CHECK:       arith.addf
    // CHECK: %[[o0:.*]] = tensor.expand_shape %[[r0]]
    %0 = hivm.hir.vreduce <sum> ins(%a: tensor<5x?x10xf32>) outs(%b: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK-DAG: %[[t0:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t1:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c1:.*]] = tensor.collapse_shape %[[t1]]
    // CHECK: %[[r1:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t0]] :
    // CHECK-SAME:      outs(%[[c1]] :
    // CHECK:       arith.addi
    // CHECK: %[[e1:.*]] = tensor.expand_shape %[[r1]]
    // CHECK: bufferization.materialize_in_destination %[[e1]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <sum> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    // CHECK: %[[c2:.*]] = tensor.collapse_shape %[[o0]]
    // CHECK: %[[r2:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[a]] :
    // CHECK-SAME:          outs(%[[c2]] :
    // CHECK:       arith.mulf
    // CHECK: %[[o1:.*]] = tensor.expand_shape %[[r2]]
    %1 = hivm.hir.vreduce <prod> ins(%a: tensor<5x?x10xf32>) outs(%0: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK-DAG: %[[t2:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t3:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c3:.*]] = tensor.collapse_shape %[[t3]]
    // CHECK: %[[r3:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t2]] :
    // CHECK-SAME:      outs(%[[c3]] :
    // CHECK:       arith.muli
    // CHECK: %[[e3:.*]] = tensor.expand_shape %[[r3]]
    // CHECK: bufferization.materialize_in_destination %[[e3]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <prod> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    // CHECK: %[[c4:.*]] = tensor.collapse_shape %[[o1]]
    // CHECK: %[[r4:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[a]] :
    // CHECK-SAME:          outs(%[[c4]] :
    // CHECK:       arith.maxnumf
    // CHECK: %[[o2:.*]] = tensor.expand_shape %[[r4]]
    %2 = hivm.hir.vreduce <any> ins(%a: tensor<5x?x10xf32>) outs(%1: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK-DAG: %[[t4:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t5:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c5:.*]] = tensor.collapse_shape %[[t5]]
    // CHECK: %[[r5:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t4]] :
    // CHECK-SAME:      outs(%[[c5]] :
    // CHECK:       arith.maxsi
    // CHECK: %[[e5:.*]] = tensor.expand_shape %[[r5]]
    // CHECK: bufferization.materialize_in_destination %[[e5]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <any> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    // CHECK: %[[c6:.*]] = tensor.collapse_shape %[[o2]]
    // CHECK: %[[r6:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[a]] :
    // CHECK-SAME:          outs(%[[c6]] :
    // CHECK:       arith.maxnumf
    // CHECK: %[[o3:.*]] = tensor.expand_shape %[[r6]]
    %3 = hivm.hir.vreduce <max> ins(%a: tensor<5x?x10xf32>) outs(%2: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK-DAG: %[[t6:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t7:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c7:.*]] = tensor.collapse_shape %[[t7]]
    // CHECK: %[[r7:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t6]] :
    // CHECK-SAME:      outs(%[[c7]] :
    // CHECK:       arith.maxsi
    // CHECK: %[[e7:.*]] = tensor.expand_shape %[[r7]]
    // CHECK: bufferization.materialize_in_destination %[[e7]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <max> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    // CHECK-DAG: %[[c8:.*]] = tensor.collapse_shape %[[o3]]
    // CHECK-DAG: %[[c9:.*]] = tensor.collapse_shape %[[c]]
    // CHECK: %[[r8:[^:]*]]:2 = hfusion.reduce_with_index {tie_break_left = true} <max>
    // CHECK-SAME:      ins(%[[a]] :
    // CHECK-SAME:      outs(%[[c8]], %[[c9]] :
    // CHECK-DAG: %[[o4:.*]] = tensor.expand_shape %[[r8]]#0
    // CHECK-DAG: %[[id1:.*]] = tensor.expand_shape %[[r8]]#1
    %4, %id1 = hivm.hir.vreduce <max_with_index_left> ins(%a: tensor<5x?x10xf32>) outs(%3, %c: tensor<5x1x10xf32>, tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xf32>, tensor<5x1x10xi32>

    // CHECK-DAG: %[[t8:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t9:.*]] = bufferization.to_tensor %[[f]]
    // CHECK-DAG: %[[t10:.*]] = bufferization.to_tensor %[[g]]
    // CHECK-DAG: %[[c10:.*]] = tensor.collapse_shape %[[t9]]
    // CHECK-DAG: %[[c11:.*]] = tensor.collapse_shape %[[t10]]
    // CHECK: %[[r9:[^:]*]]:2 = hfusion.reduce_with_index {tie_break_left = false} <max>
    // CHECK-SAME:      ins(%[[t8]] :
    // CHECK-SAME:      outs(%[[c10]], %[[c11]] :
    // CHECK-DAG: %[[e8:.*]] = tensor.expand_shape %[[r9]]#0
    // CHECK-DAG: %[[e9:.*]] = tensor.expand_shape %[[r9]]#1
    // CHECK-DAG: bufferization.materialize_in_destination %[[e8]]
    // CHECK-DAG-SAME:                                      %[[f]]
    // CHECK-DAG: bufferization.materialize_in_destination %[[e9]]
    // CHECK-DAG-SAME:                                      %[[g]]
    hivm.hir.vreduce <max_with_index_right> ins(%d: memref<5x?x10xi32>) outs(%f, %g: memref<1x?x10xi32>, memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: %[[c12:.*]] = tensor.collapse_shape %[[o4]]
    // CHECK: %[[r10:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[a]] :
    // CHECK-SAME:          outs(%[[c12]] :
    // CHECK:       arith.minnumf
    // CHECK: %[[o5:.*]] = tensor.expand_shape %[[r10]]
    %5 = hivm.hir.vreduce <all> ins(%a: tensor<5x?x10xf32>) outs(%4: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK-DAG: %[[t11:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t12:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c13:.*]] = tensor.collapse_shape %[[t12]]
    // CHECK: %[[r11:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t11]] :
    // CHECK-SAME:      outs(%[[c13]] :
    // CHECK:       arith.minsi
    // CHECK: %[[e11:.*]] = tensor.expand_shape %[[r11]]
    // CHECK: bufferization.materialize_in_destination %[[e11]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <all> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    // CHECK: %[[c14:.*]] = tensor.collapse_shape %[[o5]]
    // CHECK: %[[r12:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[a]] :
    // CHECK-SAME:          outs(%[[c14]] :
    // CHECK:       arith.minnumf
    // CHECK: %[[o6:.*]] = tensor.expand_shape %[[r12]]
    %6 = hivm.hir.vreduce <min> ins(%a: tensor<5x?x10xf32>) outs(%5: tensor<5x1x10xf32>) reduce_dims = [1] -> tensor<5x1x10xf32>

    // CHECK-DAG: %[[t13:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t14:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c15:.*]] = tensor.collapse_shape %[[t14]]
    // CHECK: %[[r13:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t13]] :
    // CHECK-SAME:      outs(%[[c15]] :
    // CHECK:       arith.minsi
    // CHECK: %[[e13:.*]] = tensor.expand_shape %[[r13]]
    // CHECK: bufferization.materialize_in_destination %[[e13]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <min> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    // CHECK-DAG: %[[c16:.*]] = tensor.collapse_shape %[[o6]]
    // CHECK-DAG: %[[c17:.*]] = tensor.collapse_shape %[[id1]]
    // CHECK: %[[r14:[^:]*]]:2 = hfusion.reduce_with_index {tie_break_left = false} <min>
    // CHECK-SAME:      ins(%[[a]], %[[id_t]] :
    // CHECK-SAME:      outs(%[[c16]], %[[c17]] :
    // CHECK-DAG: tensor.expand_shape %[[r14]]#0
    // CHECK-DAG: tensor.expand_shape %[[r14]]#1
    %7, %id2 = hivm.hir.vreduce <min_with_index_right> ins(%a: tensor<5x?x10xf32>) indices(%id_t: tensor<5x?x10xi32>) outs(%6, %id1: tensor<5x1x10xf32>, tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xf32>,  tensor<5x1x10xi32>

    // CHECK-DAG: %[[t15:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t16:.*]] = bufferization.to_tensor %[[id_m]]
    // CHECK-DAG: %[[t17:.*]] = bufferization.to_tensor %[[f]]
    // CHECK-DAG: %[[t18:.*]] = bufferization.to_tensor %[[g]]
    // CHECK-DAG: %[[c18:.*]] = tensor.collapse_shape %[[t17]]
    // CHECK-DAG: %[[c19:.*]] = tensor.collapse_shape %[[t18]]
    // CHECK: %[[r15:[^:]*]]:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
    // CHECK-SAME:      ins(%[[t15]], %[[t16]] :
    // CHECK-SAME:      outs(%[[c18]], %[[c19]] :
    // CHECK-DAG: %[[e14:.*]] = tensor.expand_shape %[[r15]]#0
    // CHECK-DAG: %[[e15:.*]] = tensor.expand_shape %[[r15]]#1
    // CHECK-DAG: bufferization.materialize_in_destination %[[e14]]
    // CHECK-DAG-SAME:                                      %[[f]]
    // CHECK-DAG: bufferization.materialize_in_destination %[[e15]]
    // CHECK-DAG-SAME:                                      %[[g]]
    hivm.hir.vreduce <min_with_index_left> ins(%d: memref<5x?x10xi32>) indices(%id_m: memref<5x?x10xi32>) outs(%f, %g: memref<1x?x10xi32>, memref<1x?x10xi32>) reduce_dims = [0]

    // CHECK: %[[c20:.*]] = tensor.collapse_shape %[[bi]]
    // CHECK: %[[r16:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[ai]] :
    // CHECK-SAME:          outs(%[[c20]] :
    // CHECK:       arith.xori
    // CHECK: %[[o8:.*]] = tensor.expand_shape %[[r16]]
    %8 = hivm.hir.vreduce <xori> ins(%ai: tensor<5x?x10xi32>) outs(%bi: tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xi32>

    // CHECK-DAG: %[[t19:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t20:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c21:.*]] = tensor.collapse_shape %[[t20]]
    // CHECK: %[[r17:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t19]] :
    // CHECK-SAME:      outs(%[[c21]] :
    // CHECK:       arith.xori
    // CHECK: %[[e17:.*]] = tensor.expand_shape %[[r17]]
    // CHECK: bufferization.materialize_in_destination %[[e17]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <xori> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    // CHECK: %[[c22:.*]] = tensor.collapse_shape %[[o8]]
    // CHECK: %[[r18:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[ai]] :
    // CHECK-SAME:          outs(%[[c22]] :
    // CHECK:       arith.ori
    // CHECK: %[[o9:.*]] = tensor.expand_shape %[[r18]]
    %9 = hivm.hir.vreduce <ori> ins(%ai: tensor<5x?x10xi32>) outs(%8: tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xi32>

    // CHECK-DAG: %[[t21:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t22:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c23:.*]] = tensor.collapse_shape %[[t22]]
    // CHECK: %[[r19:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t21]] :
    // CHECK-SAME:      outs(%[[c23]] :
    // CHECK:       arith.ori
    // CHECK: %[[e19:.*]] = tensor.expand_shape %[[r19]]
    // CHECK: bufferization.materialize_in_destination %[[e19]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <ori> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    // CHECK: %[[c24:.*]] = tensor.collapse_shape %[[o9]]
    // CHECK: %[[r20:.*]] = linalg.reduce
    // CHECK-SAME:          ins(%[[ai]] :
    // CHECK-SAME:          outs(%[[c24]] :
    // CHECK:       arith.andi
    // CHECK: tensor.expand_shape %[[r20]]
    %10 = hivm.hir.vreduce <andi> ins(%ai: tensor<5x?x10xi32>) outs(%9: tensor<5x1x10xi32>) reduce_dims = [1] -> tensor<5x1x10xi32>

    // CHECK-DAG: %[[t23:.*]] = bufferization.to_tensor %[[d]]
    // CHECK-DAG: %[[t24:.*]] = bufferization.to_tensor %[[e]]
    // CHECK: %[[c25:.*]] = tensor.collapse_shape %[[t24]]
    // CHECK: %[[r21:.*]] = linalg.reduce
    // CHECK-SAME:      ins(%[[t23]] :
    // CHECK-SAME:      outs(%[[c25]] :
    // CHECK:       arith.andi
    // CHECK: %[[e21:.*]] = tensor.expand_shape %[[r21]]
    // CHECK: bufferization.materialize_in_destination %[[e21]]
    // CHECK-SAME:                                      %[[e]]
    hivm.hir.vreduce <andi> ins(%d: memref<5x?x10xi32>) outs(%e: memref<1x1x10xi32>) reduce_dims = [0, 1]

    func.return %7, %id2, %10: tensor<5x1x10xf32>, tensor<5x1x10xi32>, tensor<5x1x10xi32>
}

// -----

// CHECK-LABEL: memref_load_lowering
// CHECK-SAME:              %[[src:[^:]*]]: {{[^,]*}}, %[[dst:[^:]*]]
func.func @memref_load_lowering(%src: memref<5x?x?xi32>, %dst: memref<5x?x?xi32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
    // CHECK: %[[c0:.*]] = arith.constant 0
    %c0 = arith.constant 0: i32
    %c2 = arith.constant 2: index
    %c5 = arith.constant 5: index

    %d2 = memref.dim %dst, %c2: memref<5x?x?xi32>
    // CHECK: %[[fifth:.*]] = arith.divui
    %fifth_d2 = arith.divui %d2, %c5: index
    // CHECK: %[[two_fifth:.*]] = arith.addi %[[fifth]], %[[fifth]]
    %two_fifth_d2 = arith.addi %fifth_d2, %fifth_d2: index

    // CHECK: linalg.copy
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>)

    // CHECK: linalg.copy
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadNull>

    // CHECK: %[[src_dim1:.*]] = memref.dim %[[src]]
    // CHECK: %[[dst_dim1:.*]] = memref.dim %[[dst]]
    // CHECK: %[[right_pad1:.*]] = arith.subi %[[dst_dim1]], %[[src_dim1]]
    // CHECK: %[[main_subview1:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[src_dim1]]]
    // CHECK: memref.copy %[[src]], %[[main_subview1]]
    // CHECK: %[[right_subview1:.*]] = memref.subview %[[dst]][0, 0, %[[src_dim1]]] [5, {{[^,]*}}, %[[right_pad1]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview1]]
    // CHECK-NEXT: ^bb0(%[[in1:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in1]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadValue> pad_value = %c0: i32

    // CHECK: %[[src_dim2:.*]] = memref.dim %[[src]]
    // CHECK: %[[right_offset1:.*]] = arith.addi %[[src_dim2]], %[[fifth]]
    // CHECK: %[[dst_dim2:.*]] = memref.dim %[[dst]]
    // CHECK: %[[right_pad2:.*]] = arith.subi %[[dst_dim2]], %[[right_offset1]]
    // CHECK: %[[left_subview1:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview1]]
    // CHECK-NEXT:  ^bb0(%[[in2:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in2]]
    // CHECK: %[[main_subview2:.*]] = memref.subview %[[dst]][0, 0, %[[fifth]]] [5, {{[^,]*}}, %[[src_dim2]]]
    // CHECK: memref.copy %[[src]], %[[main_subview2]]
    // CHECK: %[[right_subview2:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset1]]] [5, {{[^,]*}}, %[[right_pad2]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview2]]
    // CHECK-NEXT: ^bb0(%[[in3:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in3]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadValue> pad_value = %c0: i32 left_padding_num = %fifth_d2: index

    // CHECK: %[[src_dim3:.*]] = memref.dim %[[src]]
    // CHECK: %[[total_src2:.*]] = arith.addi %[[src_dim3]], %[[fifth]]
    // CHECK: %[[dst_dim3:.*]] = memref.dim %[[dst]]
    // CHECK: %[[left_pad1:.*]] = arith.subi %[[dst_dim3]], %[[total_src2]]
    // CHECK: %[[right_offset2:.*]] = arith.addi %[[left_pad1]], %[[src_dim3]]
    // CHECK: %[[left_subview1:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[left_pad1]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview1]]
    // CHECK-NEXT:  ^bb0(%[[in4:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in4]]
    // CHECK: %[[main_subview3:.*]] = memref.subview %[[dst]][0, 0, %[[left_pad1]]] [5, {{[^,]*}}, %[[src_dim3]]]
    // CHECK: memref.copy %[[src]], %[[main_subview3]]
    // CHECK: %[[right_subview3:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset2]]] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview3]]
    // CHECK-NEXT: ^bb0(%[[in5:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in5]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadValue> pad_value = %c0: i32 right_padding_num = %fifth_d2: index

    // CHECK: %[[src_dim4:.*]] = memref.dim %[[src]]
    // CHECK: %[[right_offset3:.*]] = arith.addi %[[src_dim4]], %[[fifth]]
    // CHECK: %[[left_subview2:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview2]]
    // CHECK-NEXT:  ^bb0(%[[in6:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in6]]
    // CHECK: %[[main_subview4:.*]] = memref.subview %[[dst]][0, 0, %[[fifth]]] [5, {{[^,]*}}, %[[src_dim4]]]
    // CHECK: memref.copy %[[src]], %[[main_subview4]]
    // CHECK: %[[right_subview4:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset3]]] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview4]]
    // CHECK-NEXT: ^bb0(%[[in7:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in7]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadValue> pad_value = %c0: i32 left_padding_num = %fifth_d2: index right_padding_num = %two_fifth_d2: index

    // CHECK: %[[src_dim5:.*]] = memref.dim %[[src]]
    // CHECK: %[[dst_dim5:.*]] = memref.dim %[[dst]]
    // CHECK: %[[right_pad3:.*]] = arith.subi %[[dst_dim5]], %[[src_dim5]]
    // CHECK: %[[pad_subview1:.*]] = memref.subview %[[src]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[main_subview5:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[src_dim5]]]
    // CHECK: memref.copy %[[src]], %[[main_subview5]]
    // CHECK: %[[right_subview5:.*]] = memref.subview %[[dst]][0, 0, %[[src_dim5]]] [5, {{[^,]*}}, %[[right_pad3]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview1]]
    // CHECK-SAME: outs(%[[right_subview5]]
    // CHECK-NEXT: ^bb0(%[[in8:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in8]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadFirstElem>

    // CHECK: %[[src_dim6:.*]] = memref.dim %[[src]]
    // CHECK: %[[right_offset4:.*]] = arith.addi %[[src_dim6]], %[[two_fifth]]
    // CHECK: %[[dst_dim6:.*]] = memref.dim %[[dst]]
    // CHECK: %[[right_pad4:.*]] = arith.subi %[[dst_dim6]], %[[right_offset4]]
    // CHECK: %[[pad_subview2:.*]] = memref.subview %[[src]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview3:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview2]]
    // CHECK-SAME: outs(%[[left_subview3]]
    // CHECK-NEXT:  ^bb0(%[[in9:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in9]]
    // CHECK: %[[main_subview6:.*]] = memref.subview %[[dst]][0, 0, %[[two_fifth]]] [5, {{[^,]*}}, %[[src_dim6]]]
    // CHECK: memref.copy %[[src]], %[[main_subview6]]
    // CHECK: %[[right_subview6:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset4]]] [5, {{[^,]*}}, %[[right_pad4]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview2]]
    // CHECK-SAME: outs(%[[right_subview6]]
    // CHECK-NEXT: ^bb0(%[[in10:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in10]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadFirstElem> left_padding_num = %two_fifth_d2: index

    // CHECK: %[[src_dim7:.*]] = memref.dim %[[src]]
    // CHECK: %[[total_src3:.*]] = arith.addi %[[src_dim7]], %[[two_fifth]]
    // CHECK: %[[dst_dim7:.*]] = memref.dim %[[dst]]
    // CHECK: %[[left_pad2:.*]] = arith.subi %[[dst_dim7]], %[[total_src3]]
    // CHECK: %[[right_offset5:.*]] = arith.addi %[[left_pad2]], %[[src_dim7]]
    // CHECK: %[[pad_subview3:.*]] = memref.subview %[[src]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview4:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[left_pad2]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview3]]
    // CHECK-SAME: outs(%[[left_subview4]]
    // CHECK-NEXT:  ^bb0(%[[in11:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in11]]
    // CHECK: %[[main_subview7:.*]] = memref.subview %[[dst]][0, 0, %[[left_pad2]]] [5, {{[^,]*}}, %[[src_dim7]]]
    // CHECK: memref.copy %[[src]], %[[main_subview7]]
    // CHECK: %[[right_subview7:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset5]]] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview3]]
    // CHECK-SAME: outs(%[[right_subview7]]
    // CHECK-NEXT: ^bb0(%[[in12:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in12]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadFirstElem> right_padding_num = %two_fifth_d2: index

    // CHECK: %[[src_dim8:.*]] = memref.dim %[[src]]
    // CHECK: %[[right_offset6:.*]] = arith.addi %[[src_dim8]], %[[two_fifth]]
    // CHECK: %[[pad_subview4:.*]] = memref.subview %[[src]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview5:.*]] = memref.subview %[[dst]][0, 0, 0] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview4]]
    // CHECK-SAME: outs(%[[left_subview5]]
    // CHECK-NEXT:  ^bb0(%[[in13:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in13]]
    // CHECK: %[[main_subview8:.*]] = memref.subview %[[dst]][0, 0, %[[two_fifth]]] [5, {{[^,]*}}, %[[src_dim8]]]
    // CHECK: memref.copy %[[src]], %[[main_subview8]]
    // CHECK: %[[right_subview8:.*]] = memref.subview %[[dst]][0, 0, %[[right_offset6]]] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview4]]
    // CHECK-SAME: outs(%[[right_subview8]]
    // CHECK-NEXT: ^bb0(%[[in14:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in14]]
    hivm.hir.load ins(%src: memref<5x?x?xi32>) outs(%dst: memref<5x?x?xi32>) pad_mode = <PadFirstElem> left_padding_num = %two_fifth_d2: index right_padding_num = %fifth_d2: index

    func.return
}

// -----

// CHECK-LABEL: tensor_load_lowering
// CHECK-SAME:              %[[tensor_src:[^:]*]]: {{[^,]*}}, %[[tensor_dst:[^:]*]]
func.func @tensor_load_lowering(%src: tensor<5x?x?xf32>, %dst: tensor<5x?x?xf32>) -> tensor<5x?x?xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
    // CHECK: %[[c0:.*]] = arith.constant 0
    %c0 = arith.constant 0.: f32
    %c2 = arith.constant 2: index
    %c5 = arith.constant 5: index

    %d2 = tensor.dim %dst, %c2: tensor<5x?x?xf32>
    // CHECK: %[[fifth:.*]] = arith.divui
    %fifth_d2 = arith.divui %d2, %c5: index
    // CHECK: %[[two_fifth:.*]] = arith.addi %[[fifth]], %[[fifth]]
    %two_fifth_d2 = arith.addi %fifth_d2, %fifth_d2: index

    // CHECK: linalg.copy
    %0 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%dst: tensor<5x?x?xf32>) -> tensor<5x?x?xf32>

    // CHECK: %[[tensor_dst1:.*]] = linalg.copy
    %1 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%0: tensor<5x?x?xf32>) pad_mode = <PadNull> -> tensor<5x?x?xf32>

    // CHECK: %[[src1:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst1:.*]] = bufferization.to_memref %[[tensor_dst1]]
    // CHECK: %[[dst1:.*]] = bufferization.clone %[[original_dst1]]
    // CHECK: %[[src_dim1:.*]] = memref.dim %[[src1]]
    // CHECK: %[[dst_dim1:.*]] = memref.dim %[[dst1]]
    // CHECK: %[[right_pad1:.*]] = arith.subi %[[dst_dim1]], %[[src_dim1]]
    // CHECK: %[[main_subview1:.*]] = memref.subview %[[dst1]][0, 0, 0] [5, {{[^,]*}}, %[[src_dim1]]]
    // CHECK: memref.copy %[[src1]], %[[main_subview1]]
    // CHECK: %[[right_subview1:.*]] = memref.subview %[[dst1]][0, 0, %[[src_dim1]]] [5, {{[^,]*}}, %[[right_pad1]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview1]]
    // CHECK-NEXT: ^bb0(%[[in1:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in1]]
    // CHECK: %[[tensor_dst2:.*]] = bufferization.to_tensor %[[dst1]] restrict
    %2 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%1: tensor<5x?x?xf32>) pad_mode = <PadValue> pad_value = %c0: f32 -> tensor<5x?x?xf32>

    // CHECK: %[[src2:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst2:.*]] = bufferization.to_memref %[[tensor_dst2]]
    // CHECK: %[[dst2:.*]] = bufferization.clone %[[original_dst2]]
    // CHECK: %[[src_dim2:.*]] = memref.dim %[[src2]]
    // CHECK: %[[right_offset1:.*]] = arith.addi %[[src_dim2]], %[[fifth]]
    // CHECK: %[[dst_dim2:.*]] = memref.dim %[[dst2]]
    // CHECK: %[[right_pad2:.*]] = arith.subi %[[dst_dim2]], %[[right_offset1]]
    // CHECK: %[[left_subview1:.*]] = memref.subview %[[dst2]][0, 0, 0] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview1]]
    // CHECK-NEXT:  ^bb0(%[[in2:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in2]]
    // CHECK: %[[main_subview2:.*]] = memref.subview %[[dst2]][0, 0, %[[fifth]]] [5, {{[^,]*}}, %[[src_dim2]]]
    // CHECK: memref.copy %[[src2]], %[[main_subview2]]
    // CHECK: %[[right_subview2:.*]] = memref.subview %[[dst2]][0, 0, %[[right_offset1]]] [5, {{[^,]*}}, %[[right_pad2]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview2]]
    // CHECK-NEXT: ^bb0(%[[in3:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in3]]
    // CHECK: %[[tensor_dst3:.*]] = bufferization.to_tensor %[[dst2]] restrict
    %3 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%2: tensor<5x?x?xf32>) pad_mode = <PadValue> pad_value = %c0: f32 left_padding_num = %fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src3:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst3:.*]] = bufferization.to_memref %[[tensor_dst3]]
    // CHECK: %[[dst3:.*]] = bufferization.clone %[[original_dst3]]
    // CHECK: %[[src_dim3:.*]] = memref.dim %[[src3]]
    // CHECK: %[[total_src2:.*]] = arith.addi %[[src_dim3]], %[[fifth]]
    // CHECK: %[[dst_dim3:.*]] = memref.dim %[[dst3]]
    // CHECK: %[[left_pad1:.*]] = arith.subi %[[dst_dim3]], %[[total_src2]]
    // CHECK: %[[right_offset2:.*]] = arith.addi %[[left_pad1]], %[[src_dim3]]
    // CHECK: %[[left_subview1:.*]] = memref.subview %[[dst3]][0, 0, 0] [5, {{[^,]*}}, %[[left_pad1]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview1]]
    // CHECK-NEXT:  ^bb0(%[[in4:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in4]]
    // CHECK: %[[main_subview3:.*]] = memref.subview %[[dst3]][0, 0, %[[left_pad1]]] [5, {{[^,]*}}, %[[src_dim3]]]
    // CHECK: memref.copy %[[src3]], %[[main_subview3]]
    // CHECK: %[[right_subview3:.*]] = memref.subview %[[dst3]][0, 0, %[[right_offset2]]] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview3]]
    // CHECK-NEXT: ^bb0(%[[in5:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in5]]
    // CHECK: %[[tensor_dst4:.*]] = bufferization.to_tensor %[[dst3]] restrict
    %4 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%3: tensor<5x?x?xf32>) pad_mode = <PadValue> pad_value = %c0: f32 right_padding_num = %fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src4:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst4:.*]] = bufferization.to_memref %[[tensor_dst4]]
    // CHECK: %[[dst4:.*]] = bufferization.clone %[[original_dst4]]
    // CHECK: %[[src_dim4:.*]] = memref.dim %[[src4]]
    // CHECK: %[[right_offset3:.*]] = arith.addi %[[src_dim4]], %[[fifth]]
    // CHECK: %[[left_subview2:.*]] = memref.subview %[[dst4]][0, 0, 0] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[left_subview2]]
    // CHECK-NEXT:  ^bb0(%[[in6:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in6]]
    // CHECK: %[[main_subview4:.*]] = memref.subview %[[dst4]][0, 0, %[[fifth]]] [5, {{[^,]*}}, %[[src_dim4]]]
    // CHECK: memref.copy %[[src4]], %[[main_subview4]]
    // CHECK: %[[right_subview4:.*]] = memref.subview %[[dst4]][0, 0, %[[right_offset3]]] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[c0]]
    // CHECK-SAME: outs(%[[right_subview4]]
    // CHECK-NEXT: ^bb0(%[[in7:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in7]]
    // CHECK: %[[tensor_dst5:.*]] = bufferization.to_tensor %[[dst4]] restrict
    %5 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%4: tensor<5x?x?xf32>) pad_mode = <PadValue> pad_value = %c0: f32 left_padding_num = %fifth_d2: index right_padding_num = %two_fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src5:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst5:.*]] = bufferization.to_memref %[[tensor_dst5]]
    // CHECK: %[[dst5:.*]] = bufferization.clone %[[original_dst5]]
    // CHECK: %[[src_dim5:.*]] = memref.dim %[[src5]]
    // CHECK: %[[dst_dim5:.*]] = memref.dim %[[dst5]]
    // CHECK: %[[right_pad3:.*]] = arith.subi %[[dst_dim5]], %[[src_dim5]]
    // CHECK: %[[pad_subview1:.*]] = memref.subview %[[src5]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[main_subview5:.*]] = memref.subview %[[dst5]][0, 0, 0] [5, {{[^,]*}}, %[[src_dim5]]]
    // CHECK: memref.copy %[[src5]], %[[main_subview5]]
    // CHECK: %[[right_subview5:.*]] = memref.subview %[[dst5]][0, 0, %[[src_dim5]]] [5, {{[^,]*}}, %[[right_pad3]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview1]]
    // CHECK-SAME: outs(%[[right_subview5]]
    // CHECK-NEXT: ^bb0(%[[in8:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in8]]
    // CHECK: %[[tensor_dst6:.*]] = bufferization.to_tensor %[[dst5]] restrict
    %6 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%5: tensor<5x?x?xf32>) pad_mode = <PadFirstElem> -> tensor<5x?x?xf32>

    // CHECK: %[[src6:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst6:.*]] = bufferization.to_memref %[[tensor_dst6]]
    // CHECK: %[[dst6:.*]] = bufferization.clone %[[original_dst6]]
    // CHECK: %[[src_dim6:.*]] = memref.dim %[[src6]]
    // CHECK: %[[right_offset4:.*]] = arith.addi %[[src_dim6]], %[[two_fifth]]
    // CHECK: %[[dst_dim6:.*]] = memref.dim %[[dst6]]
    // CHECK: %[[right_pad4:.*]] = arith.subi %[[dst_dim6]], %[[right_offset4]]
    // CHECK: %[[pad_subview2:.*]] = memref.subview %[[src6]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview3:.*]] = memref.subview %[[dst6]][0, 0, 0] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview2]]
    // CHECK-SAME: outs(%[[left_subview3]]
    // CHECK-NEXT:  ^bb0(%[[in9:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in9]]
    // CHECK: %[[main_subview6:.*]] = memref.subview %[[dst6]][0, 0, %[[two_fifth]]] [5, {{[^,]*}}, %[[src_dim6]]]
    // CHECK: memref.copy %[[src6]], %[[main_subview6]]
    // CHECK: %[[right_subview6:.*]] = memref.subview %[[dst6]][0, 0, %[[right_offset4]]] [5, {{[^,]*}}, %[[right_pad4]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview2]]
    // CHECK-SAME: outs(%[[right_subview6]]
    // CHECK-NEXT: ^bb0(%[[in10:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in10]]
    // CHECK: %[[tensor_dst7:.*]] = bufferization.to_tensor %[[dst6]] restrict
    %7 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%6: tensor<5x?x?xf32>) pad_mode = <PadFirstElem> left_padding_num = %two_fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src7:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst7:.*]] = bufferization.to_memref %[[tensor_dst7]]
    // CHECK: %[[dst7:.*]] = bufferization.clone %[[original_dst7]]
    // CHECK: %[[src_dim7:.*]] = memref.dim %[[src7]]
    // CHECK: %[[total_src3:.*]] = arith.addi %[[src_dim7]], %[[two_fifth]]
    // CHECK: %[[dst_dim7:.*]] = memref.dim %[[dst7]]
    // CHECK: %[[left_pad2:.*]] = arith.subi %[[dst_dim7]], %[[total_src3]]
    // CHECK: %[[right_offset5:.*]] = arith.addi %[[left_pad2]], %[[src_dim7]]
    // CHECK: %[[pad_subview3:.*]] = memref.subview %[[src7]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview4:.*]] = memref.subview %[[dst7]][0, 0, 0] [5, {{[^,]*}}, %[[left_pad2]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview3]]
    // CHECK-SAME: outs(%[[left_subview4]]
    // CHECK-NEXT:  ^bb0(%[[in11:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in11]]
    // CHECK: %[[main_subview7:.*]] = memref.subview %[[dst7]][0, 0, %[[left_pad2]]] [5, {{[^,]*}}, %[[src_dim7]]]
    // CHECK: memref.copy %[[src7]], %[[main_subview7]]
    // CHECK: %[[right_subview7:.*]] = memref.subview %[[dst7]][0, 0, %[[right_offset5]]] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview3]]
    // CHECK-SAME: outs(%[[right_subview7]]
    // CHECK-NEXT: ^bb0(%[[in12:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in12]]
    // CHECK: %[[tensor_dst8:.*]] = bufferization.to_tensor %[[dst7]] restrict
    %8 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%7: tensor<5x?x?xf32>) pad_mode = <PadFirstElem> right_padding_num = %two_fifth_d2: index -> tensor<5x?x?xf32>

    // CHECK: %[[src8:.*]] = bufferization.to_memref %[[tensor_src]] read_only
    // CHECK: %[[original_dst8:.*]] = bufferization.to_memref %[[tensor_dst8]]
    // CHECK: %[[dst8:.*]] = bufferization.clone %[[original_dst8]]
    // CHECK: %[[src_dim8:.*]] = memref.dim %[[src8]]
    // CHECK: %[[right_offset6:.*]] = arith.addi %[[src_dim8]], %[[two_fifth]]
    // CHECK: %[[pad_subview4:.*]] = memref.subview %[[src8]][0, 0, 0] [5, {{[^,]*}}, 1]
    // CHECK: %[[left_subview5:.*]] = memref.subview %[[dst8]][0, 0, 0] [5, {{[^,]*}}, %[[two_fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview4]]
    // CHECK-SAME: outs(%[[left_subview5]]
    // CHECK-NEXT:  ^bb0(%[[in13:[^:]*]]
    // CHECK-NEXT:    linalg.yield %[[in13]]
    // CHECK: %[[main_subview8:.*]] = memref.subview %[[dst8]][0, 0, %[[two_fifth]]] [5, {{[^,]*}}, %[[src_dim8]]]
    // CHECK: memref.copy %[[src8]], %[[main_subview8]]
    // CHECK: %[[right_subview8:.*]] = memref.subview %[[dst8]][0, 0, %[[right_offset6]]] [5, {{[^,]*}}, %[[fifth]]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[pad_subview4]]
    // CHECK-SAME: outs(%[[right_subview8]]
    // CHECK-NEXT: ^bb0(%[[in14:[^:]*]]
    // CHECK-NEXT:  linalg.yield %[[in14]]
    // CHECK: %[[tensor_dst9:.*]] = bufferization.to_tensor %[[dst8]] restrict
    %9 = hivm.hir.load ins(%src: tensor<5x?x?xf32>) outs(%8: tensor<5x?x?xf32>) pad_mode = <PadFirstElem> left_padding_num = %two_fifth_d2: index right_padding_num = %fifth_d2: index -> tensor<5x?x?xf32>

    func.return %9: tensor<5x?x?xf32>
}
