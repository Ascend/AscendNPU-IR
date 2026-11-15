// RUN: bishengir-opt %s --hivm-vectorize --verify-each -split-input-file \
// RUN:   | FileCheck %s

// Reduction matrix: native vector reductions, scalarized non-native kinds,
// multi-dimensional rows, and indexed maxima.

module {
  func.func @sum64(%a: tensor<64xf32>, %init: tensor<1xf32>)
      -> tensor<1xf32> {
    %0 = hivm.hir.vreduce<sum> ins(%a : tensor<64xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// CHECK-LABEL: func.func private @sum64_fused_0
// CHECK: vector.reduction <add>, {{.*}} : vector<64xf32> into f32
// CHECK-LABEL: func.func @sum64(
// CHECK: call @sum64_fused_0

// -----

// A width that is not a multiple of the 64-lane row is scalarized.

module {
  func.func @sum67(%a: tensor<67xf32>, %init: tensor<1xf32>)
      -> tensor<1xf32> {
    %0 = hivm.hir.vreduce<sum> ins(%a : tensor<67xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// CHECK-LABEL: func.func private @sum67_fused_0
// CHECK: scf.for {{.*}} = %c0 to %c67 step %c1
// CHECK: arith.addf
// CHECK-NOT: vector.reduction <add>

// -----

module {
  func.func @prod(%a: tensor<64xf32>, %init: tensor<1xf32>) -> tensor<1xf32> {
    %0 = hivm.hir.vreduce<prod> ins(%a : tensor<64xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// CHECK-LABEL: func.func private @prod_fused_0
// CHECK: scf.for
// CHECK: arith.mulf
// CHECK-NOT: vector.reduction <mul>

// -----

module {
  func.func @max(%a: tensor<64xf32>, %init: tensor<1xf32>) -> tensor<1xf32> {
    %0 = hivm.hir.vreduce<max> ins(%a : tensor<64xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// CHECK-LABEL: func.func private @max_fused_0
// CHECK: vector.reduction <maximumf>, {{.*}} : vector<64xf32> into f32

// -----

module {
  func.func @min(%a: tensor<64xf32>, %init: tensor<1xf32>) -> tensor<1xf32> {
    %0 = hivm.hir.vreduce<min> ins(%a : tensor<64xf32>)
        outs(%init : tensor<1xf32>) reduce_dims = [0] -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// CHECK-LABEL: func.func private @min_fused_0
// CHECK: vector.reduction <minimumf>, {{.*}} : vector<64xf32> into f32

// -----

module {
  func.func @and(%a: tensor<64xi1>, %init: tensor<1xi1>) -> tensor<1xi1> {
    %0 = hivm.hir.vreduce<andi> ins(%a : tensor<64xi1>)
        outs(%init : tensor<1xi1>) reduce_dims = [0] -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}

// CHECK-LABEL: func.func private @and_fused_0
// CHECK: scf.for
// CHECK: arith.andi

// -----

// Reducing the leading dimension of a 2-D value keeps the row loop and the
// column loop.

module {
  func.func @rowsum(%a: tensor<4x67xf32>, %init: tensor<1x67xf32>)
      -> tensor<1x67xf32> {
    %0 = hivm.hir.vreduce<sum> ins(%a : tensor<4x67xf32>)
        outs(%init : tensor<1x67xf32>) reduce_dims = [0] -> tensor<1x67xf32>
    return %0 : tensor<1x67xf32>
  }
}

// CHECK-LABEL: func.func private @rowsum_fused_0
// CHECK: scf.for {{.*}} = %c0 to %c67 step %c64
// CHECK: scf.for {{.*}} = %c0 to %c4 step %c1

// -----

// Max-with-index produces the value and the integer index.

module {
  func.func @indexed(%a: tensor<67xf32>, %values: tensor<1xf32>,
                     %indices: tensor<1xi32>)
      -> (tensor<1xf32>, tensor<1xi32>) {
    %r:2 = hivm.hir.vreduce {already_initialize_init} <max_with_index_left>
        ins(%a : tensor<67xf32>)
        outs(%values, %indices : tensor<1xf32>, tensor<1xi32>)
        reduce_dims = [0] -> tensor<1xf32>, tensor<1xi32>
    return %r#0, %r#1 : tensor<1xf32>, tensor<1xi32>
  }
}

// CHECK-LABEL: func.func private @indexed_fused_0
// CHECK: arith.maximumf
// CHECK: arith.andi
// CHECK-LABEL: func.func @indexed(
// CHECK: call @indexed_fused_0
