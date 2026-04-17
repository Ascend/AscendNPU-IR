// RUN: bishengir-opt -convert-shared-ptr-to-memdesc %s -split-input-file | FileCheck %s

// Test 1: Store without fractal → local_store with swizzled_shared encoding.

// CHECK: #[[$SHARED:.*]] = #ttg.swizzled_shared
// CHECK-LABEL: tt.func @store_no_fractal
// CHECK-SAME:  (%[[MD:.*]]: !ttg.memdesc<8xf32, #[[$SHARED]], #smem, mutable>)
// CHECK:         ttg.local_store %{{.*}}, %[[MD]]
// CHECK-NOT:     tt.store
// CHECK:         tt.return
module {
  tt.func @store_no_fractal(%arg0: !tt.ptr<f32, 6>) {
    %cst = arith.constant dense<1.0> : tensor<8xf32>
    %0 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x!tt.ptr<f32, 6>>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.addptr %0, %1 : tensor<8x!tt.ptr<f32, 6>>, tensor<8xi32>
    tt.store %2, %cst : tensor<8x!tt.ptr<f32, 6>>
    tt.return
  }
}

// -----
// Test 2: Load without fractal → local_load with swizzled_shared encoding.

// CHECK: #[[$SHARED:.*]] = #ttg.swizzled_shared
// CHECK-LABEL: tt.func @load_no_fractal
// CHECK-SAME:  (%[[MD:.*]]: !ttg.memdesc<8xf32, #[[$SHARED]], #smem, mutable>)
// CHECK:         ttg.local_load %[[MD]]
// CHECK-NOT:     tt.load
// CHECK:         tt.return
module {
  tt.func @load_no_fractal(%arg0: !tt.ptr<f32, 6>) {
    %0 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x!tt.ptr<f32, 6>>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.addptr %0, %1 : tensor<8x!tt.ptr<f32, 6>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<f32, 6>>
    tt.return
  }
}

// -----
// Test 3: Store with fractal zN → local_store with fractal_shared encoding.

// CHECK: #[[$FRACTAL:.*]] = #ttgext.fractal_shared<{fractalM0 = 16, fractalN0 = 16, layoutType = "zN"
// CHECK-LABEL: tt.func @store_fractal_zN
// CHECK-SAME:  (%[[MD:.*]]: !ttg.memdesc<8xf32, #[[$FRACTAL]], #smem, mutable>
// CHECK:         ttg.local_store %{{.*}}, %[[MD]]
// CHECK-NOT:     tt.store
// CHECK:         tt.return
module {
  tt.func @store_fractal_zN(%arg0: !tt.ptr<f32, 6> {hivm.fractal_layout = "zN"}) {
    %cst = arith.constant dense<2.0> : tensor<8xf32>
    %0 = tt.splat %arg0 : !tt.ptr<f32, 6> -> tensor<8x!tt.ptr<f32, 6>>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.addptr %0, %1 : tensor<8x!tt.ptr<f32, 6>>, tensor<8xi32>
    tt.store %2, %cst : tensor<8x!tt.ptr<f32, 6>>
    tt.return
  }
}

// -----
// Test 4: Load with fractal nZ → local_load with fractal_shared encoding.

// CHECK: #[[$FRACTAL:.*]] = #ttgext.fractal_shared<{fractalM0 = 16, fractalN0 = 16, layoutType = "nZ"
// CHECK-LABEL: tt.func @load_fractal_nZ
// CHECK-SAME:  (%[[MD:.*]]: !ttg.memdesc<8xf16, #[[$FRACTAL]], #smem, mutable>
// CHECK:         ttg.local_load %[[MD]]
// CHECK-NOT:     tt.load
// CHECK:         tt.return
module {
  tt.func @load_fractal_nZ(%arg0: !tt.ptr<f16, 6> {hivm.fractal_layout = "nZ"}) {
    %0 = tt.splat %arg0 : !tt.ptr<f16, 6> -> tensor<8x!tt.ptr<f16, 6>>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.addptr %0, %1 : tensor<8x!tt.ptr<f16, 6>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<f16, 6>>
    tt.return
  }
}

// -----
// Test 5: Mixed args — only ptr<6> args are converted, global ptr args are untouched.

// CHECK: #[[$SHARED:.*]] = #ttg.swizzled_shared
// CHECK-LABEL: tt.func @mixed_args
// CHECK-SAME:  (%[[GM:.*]]: !tt.ptr<f32>, %[[MD:.*]]: !ttg.memdesc<8xf32, #[[$SHARED]], #smem, mutable>)
// CHECK:         ttg.local_store %{{.*}}, %[[MD]]
// CHECK:         tt.return
module {
  tt.func @mixed_args(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32, 6>) {
    %cst = arith.constant dense<3.0> : tensor<8xf32>
    %0 = tt.splat %arg1 : !tt.ptr<f32, 6> -> tensor<8x!tt.ptr<f32, 6>>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.addptr %0, %1 : tensor<8x!tt.ptr<f32, 6>>, tensor<8xi32>
    tt.store %2, %cst : tensor<8x!tt.ptr<f32, 6>>
    tt.return
  }
}
