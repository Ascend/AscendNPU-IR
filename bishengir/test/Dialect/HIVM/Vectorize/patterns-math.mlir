// RUN: bishengir-opt %s --hivm-vectorize --verify-each -split-input-file \
// RUN:   | FileCheck %s

// Unary math matrix. The lifter maps each HIVM math op to its scalar math
// counterpart, then vectorization fuses the chain into one vector function.

module {
  func.func @unary(%a: tensor<64xf32>, %init: tensor<64xf32>)
      -> tensor<64xf32> {
    %0 = hivm.hir.vexp ins(%a : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %1 = hivm.hir.vln ins(%0 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %2 = hivm.hir.vlog2 ins(%1 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %3 = hivm.hir.vlog10 ins(%2 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %4 = hivm.hir.vlog1p ins(%3 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %5 = hivm.hir.vexpm1 ins(%4 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %6 = hivm.hir.vexp2 ins(%5 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %7 = hivm.hir.vtanh ins(%6 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %8 = hivm.hir.vsin ins(%7 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %9 = hivm.hir.vcos ins(%8 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %10 = hivm.hir.vatan ins(%9 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %11 = hivm.hir.vtan ins(%10 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %12 = hivm.hir.verf ins(%11 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %13 = hivm.hir.vrec ins(%12 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    %14 = hivm.hir.vilogb ins(%13 : tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %14 : tensor<64xf32>
  }
}

// CHECK-LABEL: func.func private @unary_fused_0
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: math.exp {{.*}} : vector<64xf32>
// CHECK: math.log {{.*}} : vector<64xf32>
// CHECK: math.log2 {{.*}} : vector<64xf32>
// CHECK: math.log10 {{.*}} : vector<64xf32>
// CHECK: math.log1p {{.*}} : vector<64xf32>
// CHECK: math.expm1 {{.*}} : vector<64xf32>
// CHECK: math.exp2 {{.*}} : vector<64xf32>
// CHECK: math.tanh {{.*}} : vector<64xf32>
// CHECK: math.sin {{.*}} : vector<64xf32>
// CHECK: math.cos {{.*}} : vector<64xf32>
// CHECK: math.atan {{.*}} : vector<64xf32>
// CHECK: math.tan {{.*}} : vector<64xf32>
// CHECK: math.erf {{.*}} : vector<64xf32>
// CHECK: arith.divf {{.*}} : vector<64xf32>
// CHECK: math.absf
// CHECK: math.floor
// CHECK-LABEL: func.func @unary(
// CHECK: call @unary_fused_0

// -----

// Predicate-producing math lowers to comparisons.

module {
  func.func @predicates(%a: tensor<64xf32>, %inf: tensor<64xi1>,
                        %nan: tensor<64xi1>) -> (tensor<64xi1>, tensor<64xi1>) {
    %0 = hivm.hir.visinf ins(%a : tensor<64xf32>)
        outs(%inf : tensor<64xi1>) -> tensor<64xi1>
    %1 = hivm.hir.visnan ins(%a : tensor<64xf32>)
        outs(%nan : tensor<64xi1>) -> tensor<64xi1>
    return %0, %1 : tensor<64xi1>, tensor<64xi1>
  }
}

// CHECK-LABEL: func.func private @predicates_fused_0
// CHECK: arith.cmpf oeq, {{.*}}, %{{.*}} : vector<64xf32>
// CHECK: arith.cmpf uno, {{.*}}, {{.*}} : vector<64xf32>

// -----

module {
  func.func @ldexp(%a: tensor<64xf32>, %b: tensor<64xf32>,
                   %init: tensor<64xf32>) -> tensor<64xf32> {
    %0 = hivm.hir.vldexp ins(%a, %b : tensor<64xf32>, tensor<64xf32>)
        outs(%init : tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}

// CHECK-LABEL: func.func private @ldexp_fused_0
// CHECK: arith.mulf {{.*}} : vector<64xf32>

// -----

// Arange with dynamic offset/strides plus reciprocal. Non-64 extents use
// masked rows.

module {
  func.func @range(%offset: index, %s0: index, %s1: index,
                   %init: tensor<3x67xf32>) -> tensor<3x67xf32> {
    %0 = hivm.hir.varange offset[%offset] strides[%s0, %s1]
        outs(%init : tensor<3x67xf32>) -> tensor<3x67xf32>
    %1 = hivm.hir.vrec ins(%0 : tensor<3x67xf32>)
        outs(%init : tensor<3x67xf32>) -> tensor<3x67xf32>
    return %1 : tensor<3x67xf32>
  }
}

// CHECK-LABEL: func.func private @range_fused_0
// CHECK: vector.create_mask {{.*}} : vector<1x64xi1>
// CHECK: vector.broadcast
// CHECK: arith.divf {{.*}} : vector<64xf32>
