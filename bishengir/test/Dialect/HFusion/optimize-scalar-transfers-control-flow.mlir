// RUN: bishengir-opt %s --hfusion-optimize-scalar-transfers -verify-diagnostics | FileCheck %s

// Unsupported SCF operations make the pass skip the whole function.
// CHECK-LABEL: func.func @unsupported_scf
// CHECK: %[[EXTRACT:.*]] = tensor.extract %{{.*}}[] : tensor<f32>
// CHECK: %[[ADD:.*]] = arith.addf %[[EXTRACT]], %{{.*}} : f32
// CHECK: %[[INSERT:.*]] = tensor.insert %[[ADD]] into %{{.*}}[%{{.*}}] : tensor<1xf32>
// CHECK: scf.execute_region
// CHECK: return %[[INSERT]] : tensor<1xf32>
func.func @unsupported_scf(%arg: tensor<f32>) -> tensor<1xf32> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<1xf32>
  %extracted = tensor.extract %arg[] : tensor<f32>
  %add = arith.addf %extracted, %cst : f32
  %inserted = tensor.insert %add into %empty[%c0] : tensor<1xf32>
  // expected-warning @+1 {{unexpected control flow at the early hfusion pipeline}}
  scf.execute_region {
    scf.yield
  }
  return %inserted : tensor<1xf32>
}

// Unsupported CF operations also make the pass skip the whole function.
// CHECK-LABEL: func.func @unsupported_cf
// CHECK: %[[CF_EXTRACT:.*]] = tensor.extract %{{.*}}[] : tensor<f32>
// CHECK: %[[CF_ADD:.*]] = arith.addf %[[CF_EXTRACT]], %{{.*}} : f32
// CHECK: %[[CF_INSERT:.*]] = tensor.insert %[[CF_ADD]] into %{{.*}}[%{{.*}}] : tensor<1xf32>
// CHECK: cf.br ^bb1
// CHECK: ^bb1:
// CHECK: return %[[CF_INSERT]] : tensor<1xf32>
func.func @unsupported_cf(%arg: tensor<f32>) -> tensor<1xf32> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 2.0 : f32
  %empty = tensor.empty() : tensor<1xf32>
  %extracted = tensor.extract %arg[] : tensor<f32>
  %add = arith.addf %extracted, %cst : f32
  %inserted = tensor.insert %add into %empty[%c0] : tensor<1xf32>
  // expected-warning @+1 {{unexpected control flow at the early hfusion pipeline}}
  cf.br ^bb1

^bb1:
  return %inserted : tensor<1xf32>
}

// The preflight walk also detects unsupported operations in nested regions.
// CHECK-LABEL: func.func @nested_unsupported_scf
// CHECK: %[[NESTED_EXTRACT:.*]] = tensor.extract %{{.*}}[] : tensor<f32>
// CHECK: %[[NESTED_ADD:.*]] = arith.addf %[[NESTED_EXTRACT]], %[[NESTED_EXTRACT]] : f32
// CHECK: %[[NESTED_INSERT:.*]] = tensor.insert %[[NESTED_ADD]] into %{{.*}}[%{{.*}}] : tensor<1xf32>
// CHECK: scf.if
// CHECK: scf.execute_region
// CHECK: return %[[NESTED_INSERT]] : tensor<1xf32>
func.func @nested_unsupported_scf(%arg: tensor<f32>, %cond: i1) -> tensor<1xf32> attributes {mix_mode = "aiv"} {
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<1xf32>
  %extracted = tensor.extract %arg[] : tensor<f32>
  %add = arith.addf %extracted, %extracted : f32
  %inserted = tensor.insert %add into %empty[%c0] : tensor<1xf32>
  scf.if %cond {
    // expected-warning @+1 {{unexpected control flow at the early hfusion pipeline}}
    scf.execute_region {
      scf.yield
    }
  }
  return %inserted : tensor<1xf32>
}
