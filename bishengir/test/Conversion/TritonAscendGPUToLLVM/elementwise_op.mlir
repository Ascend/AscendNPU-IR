// RUN: bishengir-opt --convert-triton-ascend-gpu-to-llvm %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.enable-bishengir-simt-optimization" = 900101 : i32, "ttg.num-warps" = 8 : i32} {
  // Unary mappings.

  // CHECK-LABEL: @test_tanhf
  // CHECK: ascend_dpx.tanh {{.*}} : (f32) -> f32
  tt.func public @test_tanhf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_tanhf"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_tanhDh
  // CHECK: ascend_dpx.tanh {{.*}} : (f16) -> f16
  tt.func public @test_tanhDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_tanhDh"} : (tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_tanf
  // CHECK: ascend_dpx.tan {{.*}} : (f32) -> f32
  tt.func public @test_tanf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_tanf"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_tanDh
  // CHECK: ascend_dpx.tan {{.*}} : (f16) -> f16
  tt.func public @test_tanDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_tanDh"} : (tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_atanf
  // CHECK: ascend_dpx.atan {{.*}} : (f32) -> f32
  tt.func public @test_atanf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_atanf"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_atanDh
  // CHECK: ascend_dpx.atan {{.*}} : (f16) -> f16
  tt.func public @test_atanDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_atanDh"} : (tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_recipf
  // CHECK: ascend_dpx.recip {{.*}} : (f32) -> f32
  tt.func public @test_recipf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_recipf"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_recipDh
  // CHECK: ascend_dpx.recip {{.*}} : (f16) -> f16
  tt.func public @test_recipDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_recipDh"} : (tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_log1pf
  // CHECK: ascend_dpx.log1p {{.*}} : (f32) -> f32
  tt.func public @test_log1pf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_log1pf"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_log1pDh
  // CHECK: ascend_dpx.log1p {{.*}} : (f16) -> f16
  tt.func public @test_log1pDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_log1pDh"} : (tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_ilogbf
  // CHECK: ascend_dpx.ilogb {{.*}} : (f32) -> f32
  tt.func public @test_ilogbf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_ilogbf"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_ilogbDh
  // CHECK: ascend_dpx.ilogb {{.*}} : (f16) -> f16
  tt.func public @test_ilogbDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_ilogbDh"} : (tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_reluf
  // CHECK: ascend_dpx.relu {{.*}} : (f32) -> f32
  tt.func public @test_reluf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_reluf"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_reluDh
  // CHECK: ascend_dpx.relu {{.*}} : (f16) -> f16
  tt.func public @test_reluDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_reluDh"} : (tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_roundf
  // CHECK: ascend_dpx.round {{.*}} : (f32) -> f32
  tt.func public @test_roundf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_roundf"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_rint
  // CHECK: ascend_dpx.rint {{.*}} : (f32) -> f32
  tt.func public @test_rint(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_rint"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_float_as_int_fp32
  // CHECK: ascend_dpx.float_as_int {{.*}} : (f32) -> i32
  tt.func public @test_float_as_int_fp32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_float_as_int_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // Number-check mappings.

  // CHECK-LABEL: @test_isnan
  // CHECK: ascend_dpx.isnan {{.*}} : (f32) -> i1
  tt.func public @test_isnan(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_isnan"} : (tensor<1xf32, #blocked>) -> tensor<1xi1, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<1x!tt.ptr<i1>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i1>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_isinf
  // CHECK: ascend_dpx.isinf {{.*}} : (f32) -> i1
  tt.func public @test_isinf(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_isinf"} : (tensor<1xf32, #blocked>) -> tensor<1xi1, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<1x!tt.ptr<i1>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i1>, #blocked>
    tt.return
  }

  // Binary mappings.

  // CHECK-LABEL: @test_powf
  // CHECK: ascend_dpx.pow {{.*}} : (f16, f16) -> f16
  tt.func public @test_powf(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f16>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_powf"} : (tensor<1xf16, #blocked>, tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_powDh
  // CHECK: ascend_dpx.pow {{.*}} : (f16, f16) -> f16
  tt.func public @test_powDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f16>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_powDh"} : (tensor<1xf16, #blocked>, tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_powDb
  // CHECK: ascend_dpx.pow {{.*}} : (bf16, bf16) -> bf16
  tt.func public @test_powDb(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<bf16>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<bf16>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_powDb"} : (tensor<1xbf16, #blocked>, tensor<1xbf16, #blocked>) -> tensor<1xbf16, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_powi
  // CHECK: ascend_dpx.pow {{.*}} : (i32, i32) -> i32
  tt.func public @test_powi(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<i32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_powi"} : (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_ldexpf
  // CHECK: ascend_dpx.ldexp {{.*}} : (f32, f32) -> f32
  tt.func public @test_ldexpf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_ldexpf"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_ldexpDh
  // CHECK: ascend_dpx.ldexp {{.*}} : (f16, f16) -> f16
  tt.func public @test_ldexpDh(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f16>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f16>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_ldexpDh"} : (tensor<1xf16, #blocked>, tensor<1xf16, #blocked>) -> tensor<1xf16, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // Other elementwise patterns.

  // CHECK-LABEL: @test_addptr_scalar
  // CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  tt.func public @test_addptr_scalar(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.addptr %arg1, %c1_i32 : !tt.ptr<f32>, i32
    %1 = tt.load %0 : !tt.ptr<f32>
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }

  // CHECK-LABEL: @test_addptr_tensor
  // CHECK: llvm.getelementptr {{.*}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  tt.func public @test_addptr_tensor(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %c1 = arith.constant dense<1> : tensor<1xi32, #blocked>
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.addptr %0, %c1 : tensor<1x!tt.ptr<f32>, #blocked>, tensor<1xi32, #blocked>
    %2 = tt.load %1 : tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_select_tensor_cond
  // CHECK: llvm.select {{.*}} : i1, f32
  tt.func public @test_select_tensor_cond(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %zero = arith.constant dense<0> : tensor<1xi32, #blocked>
    %cond = arith.cmpi sgt, %1, %zero : tensor<1xi32, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %lhs = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %rhs = tt.load %3 : tensor<1x!tt.ptr<f32>, #blocked>
    %sel = arith.select %cond, %lhs, %rhs : tensor<1xi1, #blocked>, tensor<1xf32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %4, %sel : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_select_scalar_cond
  // CHECK: llvm.select {{.*}} : i1, f32
  tt.func public @test_select_scalar_cond(%arg0: !tt.ptr<f32>, %arg1: i1, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %lhs = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %rhs = tt.load %1 : tensor<1x!tt.ptr<f32>, #blocked>
    %sel = arith.select %arg1, %lhs, %rhs : tensor<1xf32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %2, %sel : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_absi
  // CHECK: "llvm.intr.abs"({{.*}}) {{.*}} : (i32) -> i32
  tt.func public @test_absi(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %src = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %abs = math.absi %src : tensor<1xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %1, %abs : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_absf
  // CHECK: llvm.intr.fabs(
  tt.func public @test_absf(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %src = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %abs = math.absf %src : tensor<1xf32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %1, %abs : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_map_elementwise
  // CHECK: llvm.add {{.*}} : i32
  tt.func public @test_map_elementwise(%arg0: !tt.ptr<i32>, %arg1: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg2: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %lhs = arith.constant dense<2> : tensor<1xi32, #blocked>
    %rhs = arith.constant dense<3> : tensor<1xi32, #blocked>
    %mapped = "tt.map_elementwise"(%lhs, %rhs) <{pack = 1 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.map_elementwise.return %sum : i32
    }) : (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %0 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %0, %mapped : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_sitofp
  // CHECK: llvm.sitofp {{.*}} : i32 to f32
  tt.func public @test_sitofp(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %src = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %cvt = arith.sitofp %src : tensor<1xi32, #blocked> to tensor<1xf32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %1, %cvt : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_trunc_fp32
  // CHECK: ascend_dpx.trunc {{.*}} : (f32) -> f32
  tt.func public @test_trunc_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_trunc_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_nearbyint_fp32
  // CHECK: ascend_dpx.nearbyint {{.*}} : (f32) -> f32
  tt.func public @test_nearbyint_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_nearbyint_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_log10_fp32
  // CHECK: ascend_dpx.log10 {{.*}} : (f32) -> f32
  tt.func public @test_log10_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_log10_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_asin_fp32
  // CHECK: ascend_dpx.asin {{.*}} : (f32) -> f32
  tt.func public @test_asin_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_asin_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_acos_fp32
  // CHECK: ascend_dpx.acos {{.*}} : (f32) -> f32
  tt.func public @test_acos_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_acos_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_sinh_fp32
  // CHECK: ascend_dpx.sinh {{.*}} : (f32) -> f32
  tt.func public @test_sinh_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_sinh_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_cosh_fp32
  // CHECK: ascend_dpx.cosh {{.*}} : (f32) -> f32
  tt.func public @test_cosh_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_cosh_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_asinh_fp32
  // CHECK: ascend_dpx.asinh {{.*}} : (f32) -> f32
  tt.func public @test_asinh_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_asinh_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_acosh_fp32
  // CHECK: ascend_dpx.acosh {{.*}} : (f32) -> f32
  tt.func public @test_acosh_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_acosh_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_atanh_fp32
  // CHECK: ascend_dpx.atanh {{.*}} : (f32) -> f32
  tt.func public @test_atanh_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_atanh_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_expm1_fp32
  // CHECK: ascend_dpx.expm1 {{.*}} : (f32) -> f32
  tt.func public @test_expm1_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_expm1_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_cyl_bessel_i0_fp32
  // CHECK: ascend_dpx.cyl_bessel_i0 {{.*}} : (f32) -> f32
  tt.func public @test_cyl_bessel_i0_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_cyl_bessel_i0_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_erfinv_fp32
  // CHECK: ascend_dpx.erfinv {{.*}} : (f32) -> f32
  tt.func public @test_erfinv_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_erfinv_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_lgamma_fp32
  // CHECK: ascend_dpx.lgamma {{.*}} : (f32) -> f32
  tt.func public @test_lgamma_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_lgamma_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_signbit_fp32
  // CHECK: ascend_dpx.signbit {{.*}} : (f32) -> i32
  tt.func public @test_signbit_fp32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_signbit_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_clz_i32
  // CHECK: ascend_dpx.clz {{.*}} : (i32) -> i32
  tt.func public @test_clz_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_clz_i32"} : (tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_popc_i32
  // CHECK: ascend_dpx.popc {{.*}} : (i32) -> i32
  tt.func public @test_popc_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_popc_i32"} : (tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_ffs_i32
  // CHECK: ascend_dpx.ffs {{.*}} : (i32) -> i32
  tt.func public @test_ffs_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_ffs_i32"} : (tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_abs_fp32
  // CHECK: ascend_dpx.abs {{.*}} : (f32) -> f32
  tt.func public @test_abs_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_abs_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_abs_i32
  // CHECK: ascend_dpx.abs {{.*}} : (i32) -> i32
  tt.func public @test_abs_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_abs_i32"} : (tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_saturate_fp32
  // CHECK: ascend_dpx.saturatef {{.*}} : (f32) -> f32
  tt.func public @test_saturate_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_saturate_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_exp10_fp32
  // CHECK: ascend_dpx.exp10 {{.*}} : (f32) -> f32
  tt.func public @test_exp10_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_exp10_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_rcp_rn_fp32
  // CHECK: ascend_dpx.rcp_rn {{.*}} : (f32) -> f32
  tt.func public @test_rcp_rn_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_rcp_rn_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_rcp_rz_fp32
  // CHECK: ascend_dpx.rcp_rz {{.*}} : (f32) -> f32
  tt.func public @test_rcp_rz_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_rcp_rz_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_rcp_rd_fp32
  // CHECK: ascend_dpx.rcp_rd {{.*}} : (f32) -> f32
  tt.func public @test_rcp_rd_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_rcp_rd_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_rcp_ru_fp32
  // CHECK: ascend_dpx.rcp_ru {{.*}} : (f32) -> f32
  tt.func public @test_rcp_ru_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_rcp_ru_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_rsqrt_rn_fp32
  // CHECK: ascend_dpx.rsqrt_rn {{.*}} : (f32) -> f32
  tt.func public @test_rsqrt_rn_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_rsqrt_rn_fp32"} : (tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_brev_i32
  // CHECK: ascend_dpx.brev {{.*}} : (i32) -> i32
  tt.func public @test_brev_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg3: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.extern_elementwise %1 {libname = "", libpath = "", pure = true, symbol = "__hmf_brev_i32"} : (tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_copysign_fp32
  // CHECK: ascend_dpx.copysign {{.*}} : (f32, f32) -> f32
  tt.func public @test_copysign_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_copysign_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_atan2_fp32
  // CHECK: ascend_dpx.atan2 {{.*}} : (f32, f32) -> f32
  tt.func public @test_atan2_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_atan2_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_nextafter_fp32
  // CHECK: ascend_dpx.nextafter {{.*}} : (f32, f32) -> f32
  tt.func public @test_nextafter_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_nextafter_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_hypot_fp32
  // CHECK: ascend_dpx.hypot {{.*}} : (f32, f32) -> f32
  tt.func public @test_hypot_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_hypot_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_mulhi_i32
  // CHECK: ascend_dpx.mulhi {{.*}} : (i32, i32) -> i32
  tt.func public @test_mulhi_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<i32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_mulhi_i32"} : (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_mul24_i32
  // CHECK: ascend_dpx.mul24 {{.*}} : (i32, i32) -> i32
  tt.func public @test_mul24_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<i32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_mul24_i32"} : (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_hadd_i32
  // CHECK: ascend_dpx.hadd {{.*}} : (i32, i32) -> i32
  tt.func public @test_hadd_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<i32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_hadd_i32"} : (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_rhadd_i32
  // CHECK: ascend_dpx.rhadd {{.*}} : (i32, i32) -> i32
  tt.func public @test_rhadd_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<i32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_rhadd_i32"} : (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_fdim_fp32
  // CHECK: ascend_dpx.fdim {{.*}} : (f32, f32) -> f32
  tt.func public @test_fdim_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_fdim_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_fast_dividef_fp32
  // CHECK: ascend_dpx.fast_dividef {{.*}} : (f32, f32) -> f32
  tt.func public @test_fast_dividef_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_fast_dividef_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_div_rz_fp32
  // CHECK: ascend_dpx.div_rz {{.*}} : (f32, f32) -> f32
  tt.func public @test_div_rz_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_div_rz_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_div_rd_fp32
  // CHECK: ascend_dpx.div_rd {{.*}} : (f32, f32) -> f32
  tt.func public @test_div_rd_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_div_rd_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_div_ru_fp32
  // CHECK: ascend_dpx.div_ru {{.*}} : (f32, f32) -> f32
  tt.func public @test_div_ru_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_div_ru_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_fmod_fp32
  // CHECK: ascend_dpx.fmod {{.*}} : (f32, f32) -> f32
  tt.func public @test_fmod_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_fmod_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_remainder_fp32
  // CHECK: ascend_dpx.remainder {{.*}} : (f32, f32) -> f32
  tt.func public @test_remainder_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_remainder_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_add_rn_fp32
  // CHECK: ascend_dpx.add_rn {{.*}} : (f32, f32) -> f32
  tt.func public @test_add_rn_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_add_rn_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_add_rz_fp32
  // CHECK: ascend_dpx.add_rz {{.*}} : (f32, f32) -> f32
  tt.func public @test_add_rz_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_add_rz_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_add_rd_fp32
  // CHECK: ascend_dpx.add_rd {{.*}} : (f32, f32) -> f32
  tt.func public @test_add_rd_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_add_rd_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_add_ru_fp32
  // CHECK: ascend_dpx.add_ru {{.*}} : (f32, f32) -> f32
  tt.func public @test_add_ru_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_add_ru_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_mul_rn_fp32
  // CHECK: ascend_dpx.mul_rn {{.*}} : (f32, f32) -> f32
  tt.func public @test_mul_rn_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_mul_rn_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_mul_rz_fp32
  // CHECK: ascend_dpx.mul_rz {{.*}} : (f32, f32) -> f32
  tt.func public @test_mul_rz_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_mul_rz_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_mul_rd_fp32
  // CHECK: ascend_dpx.mul_rd {{.*}} : (f32, f32) -> f32
  tt.func public @test_mul_rd_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_mul_rd_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_mul_ru_fp32
  // CHECK: ascend_dpx.mul_ru {{.*}} : (f32, f32) -> f32
  tt.func public @test_mul_ru_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg4: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.extern_elementwise %1, %3 {libname = "", libpath = "", pure = true, symbol = "__hmf_mul_ru_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %5, %4 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_byte_perm_i32
  // CHECK: ascend_dpx.byte_perm {{.*}} : (i32, i32, i32) -> i32
  tt.func public @test_byte_perm_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<i32>, #blocked>
    %4 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %5 = tt.load %4 : tensor<1x!tt.ptr<i32>, #blocked>
    %6 = tt.extern_elementwise %1, %3, %5 {libname = "", libpath = "", pure = true, symbol = "__hmf_byte_perm_i32"} : (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %7, %6 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_sad_i32
  // CHECK: ascend_dpx.sad {{.*}} : (i32, i32, i32) -> i32
  tt.func public @test_sad_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<i32>, #blocked>
    %4 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    %5 = tt.load %4 : tensor<1x!tt.ptr<i32>, #blocked>
    %6 = tt.extern_elementwise %1, %3, %5 {libname = "", libpath = "", pure = true, symbol = "__hmf_sad_i32"} : (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>, tensor<1xi32, #blocked>) -> tensor<1xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
    tt.store %7, %6 : tensor<1x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_fma_rn_fp32
  // CHECK: ascend_dpx.fma_rn {{.*}} : (f32, f32, f32) -> f32
  tt.func public @test_fma_rn_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %5 = tt.load %4 : tensor<1x!tt.ptr<f32>, #blocked>
    %6 = tt.extern_elementwise %1, %3, %5 {libname = "", libpath = "", pure = true, symbol = "__hmf_fma_rn_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %7, %6 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_fma_rz_fp32
  // CHECK: ascend_dpx.fma_rz {{.*}} : (f32, f32, f32) -> f32
  tt.func public @test_fma_rz_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %5 = tt.load %4 : tensor<1x!tt.ptr<f32>, #blocked>
    %6 = tt.extern_elementwise %1, %3, %5 {libname = "", libpath = "", pure = true, symbol = "__hmf_fma_rz_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %7, %6 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_fma_rd_fp32
  // CHECK: ascend_dpx.fma_rd {{.*}} : (f32, f32, f32) -> f32
  tt.func public @test_fma_rd_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %5 = tt.load %4 : tensor<1x!tt.ptr<f32>, #blocked>
    %6 = tt.extern_elementwise %1, %3, %5 {libname = "", libpath = "", pure = true, symbol = "__hmf_fma_rd_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %7, %6 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_fma_ru_fp32
  // CHECK: ascend_dpx.fma_ru {{.*}} : (f32, f32, f32) -> f32
  tt.func public @test_fma_ru_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %5 = tt.load %4 : tensor<1x!tt.ptr<f32>, #blocked>
    %6 = tt.extern_elementwise %1, %3, %5 {libname = "", libpath = "", pure = true, symbol = "__hmf_fma_ru_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %7, %6 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: @test_fma_fp32
  // CHECK: ascend_dpx.fma {{.*}} : (f32, f32, f32) -> f32
  tt.func public @test_fma_fp32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}, %arg5: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}, %arg6: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) attributes {noinline = false} {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %1 = tt.load %0 : tensor<1x!tt.ptr<f32>, #blocked>
    %2 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    %5 = tt.load %4 : tensor<1x!tt.ptr<f32>, #blocked>
    %6 = tt.extern_elementwise %1, %3, %5 {libname = "", libpath = "", pure = true, symbol = "__hmf_fma_fp32"} : (tensor<1xf32, #blocked>, tensor<1xf32, #blocked>, tensor<1xf32, #blocked>) -> tensor<1xf32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #blocked>
    tt.store %7, %6 : tensor<1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
