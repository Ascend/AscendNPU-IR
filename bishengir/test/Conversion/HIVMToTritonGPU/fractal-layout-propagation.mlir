// RUN: bishengir-opt -convert-hivm-to-tritongpu %s -split-input-file | FileCheck %s

// Test: A UB memref argument with hivm.fractal_layout = "zN" should propagate
// the attribute to the first tt.ptr<6> arg after the 1-to-5 expansion.

// The UB memref<8xf32, #ub> {hivm.fractal_layout = "zN"} expands to 5 args:
//   arg0: !tt.ptr<f32, 6> {hivm.fractal_layout = "zN"}  <-- attribute here
//   arg1: !tt.ptr<f32, 6>
//   arg2: index
//   arg3: index
//   arg4: index

// CHECK-LABEL: tt.func @fractal_zN_store
// CHECK-SAME:  %[[A0:.*]]: !tt.ptr<f32, 6> {hivm.fractal_layout = "zN"}
// CHECK-SAME:  !tt.ptr<f32, 6>
// CHECK-SAME:  index
// CHECK-SAME:  index
// CHECK-SAME:  index
// CHECK-SAME:  !tt.ptr<f32>
// CHECK-SAME:  !tt.ptr<f32>
// CHECK-SAME:  index
// CHECK-SAME:  index
// CHECK-SAME:  index
module {
  func.func @fractal_zN_store(
      %arg0: memref<8xf32, #hivm.address_space<ub>> {hivm.fractal_layout = "zN"},
      %arg1: memref<?xf32, #hivm.address_space<gm>> {hivm.memory_effect = #hivm.memory_effect<read>}
  ) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    %cst = arith.constant dense<1.0> : tensor<8xf32>
    hivm.hir.local_store ins(%arg0 : memref<8xf32, #hivm.address_space<ub>>, %cst : tensor<8xf32>)
    return
  }
}

// -----
// Test: fractal nZ attribute propagation.

// CHECK-LABEL: tt.func @fractal_nZ_load
// CHECK-SAME:  %[[A0:.*]]: !tt.ptr<f16, 6> {hivm.fractal_layout = "nZ"}
// CHECK-SAME:  !tt.ptr<f16, 6>
// CHECK-SAME:  index
// CHECK-SAME:  index
// CHECK-SAME:  index
module {
  func.func @fractal_nZ_load(
      %arg0: memref<8xf16, #hivm.address_space<ub>> {hivm.fractal_layout = "nZ"}
  ) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    return
  }
}

// -----
// Test: UB memref WITHOUT fractal attribute should NOT have fractal_layout on the ptr.

// CHECK-LABEL: tt.func @no_fractal
// CHECK-SAME:  %[[A0:.*]]: !tt.ptr<f32, 6>,
// CHECK-NOT:   hivm.fractal_layout
module {
  func.func @no_fractal(
      %arg0: memref<8xf32, #hivm.address_space<ub>>
  ) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>, no_inline, outline} {
    return
  }
}
