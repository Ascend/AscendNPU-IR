// RUN: bishengir-opt %s -hfusion-auto-schedule -split-input-file | FileCheck %s

module {
  func.func @test_fully_dynamic(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim_0 = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf16>
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf16>
    %dim_2 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf16>
    %dim_3 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf16>
    %0 = tensor.empty(%dim_0, %dim_1, %dim_2, %dim_3) : tensor<?x?x?x?xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg0, %arg1 : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) outs(%0 : tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
    return %1 : tensor<?x?x?x?xf16>
  }
}

// CHECK: func.func @test_fully_dynamic_3
// CHECK: tensor.extract_slice {{.*}}[{{.*}}, {{.*}}, {{.*}}, {{.*}}] [1, 1, 1, {{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf16> to tensor<1x1x1x?xf16>

// CHECK: func.func @test_fully_dynamic_2
// CHECK: tensor.extract_slice {{.*}}[{{.*}}, {{.*}}, {{.*}}, {{.*}}] [1, 1, {{.*}}, {{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf16> to tensor<1x1x?x?xf16>

// CHECK: func.func @test_fully_dynamic_1
// CHECK: tensor.extract_slice {{.*}}[{{.*}}, {{.*}}, {{.*}}, {{.*}}] [1, {{.*}}, {{.*}}, {{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf16> to tensor<1x?x?x?xf16>

// CHECK: func.func @test_fully_dynamic_0
// CHECK: tensor.extract_slice {{.*}}[{{.*}}, {{.*}}, {{.*}}, {{.*}}] [{{.*}}, {{.*}}, {{.*}}, {{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf16> to tensor<?x?x?x?xf16>

// -----

module {
  func.func @test_fully_dynamic(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<ANY_PB>} {
    %1 = linalg.transpose ins(%arg0: tensor<?x?x?x?xf16>) outs(%arg1 : tensor<?x?x?x?xf16>) permutation = [3, 1, 2, 0]
    return %1 : tensor<?x?x?x?xf16>
  }
}


// CHECK: func.func @test_fully_dynamic_3
// CHECK: tensor.extract_slice {{.*}}[{{.*}}, {{.*}}, {{.*}}, {{.*}}] [{{.*}}, 1, 1, {{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf16> to tensor<?x1x1x?xf16>

// CHECK: func.func @test_fully_dynamic_2
// CHECK: tensor.extract_slice {{.*}}[{{.*}}, {{.*}}, {{.*}}, {{.*}}] [{{.*}}, 1, {{.*}}, {{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf16> to tensor<?x1x?x?xf16>

// CHECK: func.func @test_fully_dynamic_1
// CHECK: tensor.extract_slice {{.*}}[{{.*}}, {{.*}}, {{.*}}, {{.*}}] [{{.*}}, {{.*}}, {{.*}}, {{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf16> to tensor<?x?x?x?xf16>

// CHECK: func.func @test_fully_dynamic_0
// CHECK: tensor.extract_slice {{.*}}[{{.*}}, {{.*}}, {{.*}}, {{.*}}] [{{.*}}, {{.*}}, {{.*}}, {{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf16> to tensor<?x?x?x?xf16>
