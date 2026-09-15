// RUN: bishengir-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK: [[$DPX:#.*]] = #ttg.blocked<{sizePerThread = [4],
// CHECK-LABEL: tt.func @dpx_f16
// CHECK: tt.load {{.*}} : tensor<256x!tt.ptr<f16>, [[$DPX]]>
// CHECK: tt.store {{.*}} : tensor<256x!tt.ptr<f16>, [[$DPX]]>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @dpx_f16(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %src : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    %value = tt.load %ptr : tensor<256x!tt.ptr<f16>, #blocked>
    %out = tt.splat %dst : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %out_ptr = tt.addptr %out, %offsets : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %out_ptr, %value : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK: [[$GENERIC:#.*]] = #ttg.blocked<{sizePerThread = [8],
// CHECK-LABEL: tt.func @generic_f16
// CHECK: tt.load {{.*}} : tensor<256x!tt.ptr<f16>, [[$GENERIC]]>
// CHECK: tt.store {{.*}} : tensor<256x!tt.ptr<f16>, [[$GENERIC]]>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @generic_f16(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %src : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    %value = tt.load %ptr : tensor<256x!tt.ptr<f16>, #blocked>
    %out = tt.splat %dst : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %out_ptr = tt.addptr %out, %offsets : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %out_ptr, %value : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK: [[$DPX32:#.*]] = #ttg.blocked<{sizePerThread = [2],
// CHECK-LABEL: tt.func @dpx_f32
// CHECK: tt.load {{.*}} : tensor<256x!tt.ptr<f32>, [[$DPX32]]>
// CHECK: tt.store {{.*}} : tensor<256x!tt.ptr<f32>, [[$DPX32]]>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @dpx_f32(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %src : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %value = tt.load %ptr : tensor<256x!tt.ptr<f32>, #blocked>
    %out = tt.splat %dst : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %out_ptr = tt.addptr %out, %offsets : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %out_ptr, %value : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// CHECK-DAG: [[$MIX16:#.*]] = #ttg.blocked<{sizePerThread = [4],
// CHECK-DAG: [[$MIX32:#.*]] = #ttg.blocked<{sizePerThread = [2],
// CHECK-LABEL: tt.func @masked_f16_to_f32
// CHECK: [[MASK16:%.*]] = ttg.convert_layout {{.*}} : tensor<256xi1, {{#[^>]*}}> -> tensor<256xi1, [[$MIX16]]>
// CHECK: tt.load {{.*}}, [[MASK16]], {{.*}} : tensor<256x!tt.ptr<f16>, [[$MIX16]]>
// CHECK: arith.extf {{.*}} : tensor<256xf16, {{#[^>]*}}> to tensor<256xf32, {{#[^>]*}}>
// CHECK: [[MASK32:%.*]] = ttg.convert_layout {{.*}} : tensor<256xi1, {{#[^>]*}}> -> tensor<256xi1, [[$MIX32]]>
// CHECK: tt.store {{.*}}, {{.*}}, [[MASK32]] : tensor<256x!tt.ptr<f32>, [[$MIX32]]>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @masked_f16_to_f32(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %limit = arith.constant dense<240> : tensor<256xi32, #blocked>
    %mask = arith.cmpi slt, %offsets, %limit : tensor<256xi32, #blocked>
    %zero = arith.constant dense<0.0> : tensor<256xf16, #blocked>
    %base = tt.splat %src : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    %value = tt.load %ptr, %mask, %zero : tensor<256x!tt.ptr<f16>, #blocked>
    %wide = arith.extf %value : tensor<256xf16, #blocked> to tensor<256xf32, #blocked>
    %out = tt.splat %dst : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %out_ptr = tt.addptr %out, %offsets : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %out_ptr, %wide, %mask : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
