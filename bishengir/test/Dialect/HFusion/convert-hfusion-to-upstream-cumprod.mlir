// REQUIRES: execution-engine
// RUN: bishengir-opt --execution-engine-convert-hfusion-to-upstream %s --split-input-file | FileCheck %s
// RUN: bishengir-opt --lower-for-cpu-runner-pipeline %s --split-input-file

// CHECK-LABEL: func.func @hfusion_cumprod_f32
func.func @hfusion_cumprod_f32(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
  // CHECK-DAG: %[[CST:[^ ]+]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[EMPTY:[^ ]+]] = tensor.empty() : tensor<2x4xf32>
  // CHECK: %[[OUTER_RES:[^ ]+]] = scf.for %[[IV0:[^ ]+]] = {{.*}} iter_args(%[[T0:[^ ]+]] = %[[EMPTY]])
  // CHECK:   %[[INNER_RES:[^ ]+]]:2 = scf.for %[[IV1:[^ ]+]] = {{.*}} iter_args(%[[ACC:[^ ]+]] = %[[CST]], %[[T1:[^ ]+]] = %[[T0]])
  // CHECK:     scf.yield {{.*}}, {{.*}} : f32, tensor<2x4xf32>
  // CHECK:   }
  // CHECK:   scf.yield %[[INNER_RES]]#1 : tensor<2x4xf32>
  %0 = hfusion.cumprod %arg0 : tensor<2x4xf32> cum_dims = [1] reverse = false -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @hfusion_cumprod_3d_f32
func.func @hfusion_cumprod_3d_f32(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  %0 = hfusion.cumprod %arg0 : tensor<2x3x4xf32> cum_dims = [1] reverse = false -> tensor<2x3x4xf32>
  func.return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: func.func @hfusion_cumprod_reverse_f32
func.func @hfusion_cumprod_reverse_f32(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
  // CHECK-DAG: %[[C3:[^ ]+]] = arith.constant 3 : index
  // CHECK: scf.for %[[IV_OUT:[^ ]+]] = {{.*}}
  // CHECK:   %[[INNER:[^ ]+]]:2 = scf.for %[[IV_IN:[^ ]+]] = {{.*}}
  // CHECK:     %[[REVERSE_IDX:[^ ]+]] = arith.subi %[[C3]], %[[IV_IN]]
  %0 = hfusion.cumprod %arg0 : tensor<2x4xf32> cum_dims = [1] reverse = true -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @hfusion_cumprod_i32
func.func @hfusion_cumprod_i32(%arg0: tensor<?x3xi32>) -> tensor<?x3xi32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
  // CHECK-DAG: %[[C1_I32:[^ ]+]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C0:[^ ]+]] = arith.constant 0 : index
  // CHECK: %[[D0:[^ ]+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: scf.for {{.*}} to %[[D0]]
  %0 = hfusion.cumprod %arg0 : tensor<?x3xi32> cum_dims = [0] reverse = false -> tensor<?x3xi32>
  func.return %0 : tensor<?x3xi32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_cumprod_cpu_lowering
// CHECK-SAME: (%[[ARG0:[^ ]+]]: tensor<1x2x3x4xi32>)
func.func @test_hfusion_cumprod_cpu_lowering(%arg0: tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
  // CHECK-DAG: %[[C1_I32:[^ ]+]] = arith.constant 1 : i32
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       %[[INNER_RES:[^ ]+]]:2 = scf.for
  // CHECK:       scf.yield %[[INNER_RES]]#1
  %0 = hfusion.cumprod %arg0 : tensor<1x2x3x4xi32> cum_dims = [2] reverse = false -> tensor<1x2x3x4xi32>
  return %0 : tensor<1x2x3x4xi32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_cumprod_reverse_lowering
func.func @test_hfusion_cumprod_reverse_lowering(%arg0: tensor<3xf32>) -> tensor<3xf32> attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<host_entry>} {
  // CHECK-DAG: %[[C2:[^ ]+]] = arith.constant 2 : index
  // CHECK: scf.for %[[IV_REV:[^ ]+]] = 
  // CHECK:   %[[REVERSE_IDX:[^ ]+]] = arith.subi %[[C2]], %[[IV_REV]]
  %0 = hfusion.cumprod %arg0 : tensor<3xf32> cum_dims = [0] reverse = true -> tensor<3xf32>
  return %0 : tensor<3xf32>
}