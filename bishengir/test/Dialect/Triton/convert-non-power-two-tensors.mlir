// RUN: bishengir-opt -split-input-file -convert-non-power-two-tensors %s | FileCheck %s --check-prefix=CHECK
// RUN: bishengir-opt -split-input-file -convert-non-power-two-tensors %s | FileCheck %s --check-prefix=GLOBAL

// GLOBAL-NOT: tensor<{{.*}}x3x{{.*}}>
// GLOBAL-NOT: tensor<3x{{.*}}>
// GLOBAL-NOT: tensor<{{.*}}x5x{{.*}}>
// GLOBAL-NOT: tensor<5x{{.*}}>
// GLOBAL-NOT: tensor<{{.*}}x9x{{.*}}>
// GLOBAL-NOT: tensor<9x{{.*}}>
// GLOBAL-NOT: tensor<{{.*}}x6x{{.*}}>
// GLOBAL-NOT: tensor<6x{{.*}}>

// CHECK-LABEL: @loadStore1DUnmasked
// CHECK: tt.load %{{.*}}, %{{.*}}
// CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}}
tt.func @loadStore1DUnmasked(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %5, %3 : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @loadStore1DMasked
// CHECK-NOT: arith.constant dense<0.0{{.*}}>
// CHECK: %[[ORIG_MASK:.*]] = arith.constant dense<false> 
// CHECK: %[[PADDING:.*]] = arith.constant dense<false>
// CHECK: %[[MASK:.*]] = arith.select %{{.*}}, %[[ORIG_MASK]], %[[PADDING]]
// CHECK: tt.load %{{.*}}, %[[MASK]]
// CHECK: tt.store %{{.*}}, %{{.*}}, %[[MASK]]
tt.func @loadStore1DMasked(%ptr1: !tt.ptr<bf16>, %ptr2: !tt.ptr<bf16>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<bf16> -> tensor<3x!tt.ptr<bf16>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<bf16>>, tensor<3xi32>
  %3 = arith.constant dense<false> : tensor<3xi1>
  %4 = tt.load %2, %3 : tensor<3x!tt.ptr<bf16>>

  %5 = tt.splat %ptr2 : !tt.ptr<bf16> -> tensor<3x!tt.ptr<bf16>>
  %6 = tt.addptr %5, %0 : tensor<3x!tt.ptr<bf16>>, tensor<3xi32>
  tt.store %6, %4, %3 : tensor<3x!tt.ptr<bf16>>
  tt.return
}

// -----

// CHECK-LABEL: @loadStore1DMaskedWithOther
// CHECK-NOT: arith.constant dense<0.0{{.*}}>
// CHECK: %[[ORIG_MASK:.*]] = arith.constant dense<false> 
// CHECK: %[[MASK_PADDING:.*]] = arith.constant dense<false>
// CHECK: %[[MASK:.*]] = arith.select %{{.*}}, %[[ORIG_MASK]], %[[MASK_PADDING]]
// CHECK: %[[PADDING:.*]] = arith.constant dense<1.0{{.*}}> 
// CHECK: tt.load %{{.*}}, %[[MASK]], %[[PADDING]]
// CHECK: tt.store %{{.*}}, %{{.*}}, %[[MASK]]
tt.func @loadStore1DMaskedWithOther(%ptr1: !tt.ptr<bf16>, %ptr2: !tt.ptr<bf16>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<bf16> -> tensor<3x!tt.ptr<bf16>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<bf16>>, tensor<3xi32>
  %3 = arith.constant dense<false> : tensor<3xi1>
  %4 = arith.constant dense<1.0> : tensor<3xbf16>
  %5 = tt.load %2, %3, %4 : tensor<3x!tt.ptr<bf16>>

  %6 = tt.splat %ptr2 : !tt.ptr<bf16> -> tensor<3x!tt.ptr<bf16>>
  %7 = tt.addptr %6, %0 : tensor<3x!tt.ptr<bf16>>, tensor<3xi32>
  tt.store %7, %5, %3 : tensor<3x!tt.ptr<bf16>>
  tt.return
}

// -----

// CHECK-LABEL: @loadStore2DUnmasked
// CHECK: tt.load %{{.*}}, %{{.*}}
// CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}}
tt.func @loadStore2DUnmasked(%ptr1: !tt.ptr<f16>, %ptr2: !tt.ptr<f16>) {
  %0 = tt.make_range {end = 6: i32, start = 0: i32} : tensor<6xi32>
  %1 = arith.constant dense<6> : tensor<6xi32>
  %2 = arith.muli %0, %1 : tensor<6xi32>

  %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<6xi32> -> tensor<6x1xi32>
  %4 = tt.broadcast %3 : tensor<6x1xi32> -> tensor<6x6xi32>
  %5 = tt.expand_dims %2 {axis = 0 : i32} : tensor<6xi32> -> tensor<1x6xi32>
  %6 = tt.broadcast %5 : tensor<1x6xi32> -> tensor<6x6xi32>

  %7 = arith.addi %4, %6 : tensor<6x6xi32>
  %8 = tt.splat %ptr1 : !tt.ptr<f16> -> tensor<6x6x!tt.ptr<f16>>
  %9 = tt.addptr %8, %7 : tensor<6x6x!tt.ptr<f16>>, tensor<6x6xi32>

  %10 = tt.load %9 : tensor<6x6x!tt.ptr<f16>>

  %11 = tt.splat %ptr2 : !tt.ptr<f16> -> tensor<6x6x!tt.ptr<f16>>
  %12 = tt.addptr %11, %7 : tensor<6x6x!tt.ptr<f16>>, tensor<6x6xi32>
  tt.store %12, %10 : tensor<6x6x!tt.ptr<f16>>
  tt.return
}

// -----

// CHECK-LABEL: @elementwiseOp
tt.func @elementwiseOp(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %ptr3: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %5 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %7 = tt.load %6 : tensor<3x!tt.ptr<f32>>

  %addf = arith.addf %3, %7 : tensor<3xf32>
  %mulf = arith.mulf %3, %7 : tensor<3xf32>
  %subf = arith.subf %3, %7 : tensor<3xf32>
  %divf = arith.divf %3, %7 : tensor<3xf32>
  %remf = arith.remf %3, %7 : tensor<3xf32>
  %maxf = arith.maximumf %3, %7 : tensor<3xf32>
  %minf = arith.minimumf %3, %7 : tensor<3xf32>

  %acc0 = arith.addf %addf, %mulf : tensor<3xf32>
  %acc1 = arith.addf %acc0, %subf : tensor<3xf32>
  %acc2 = arith.addf %acc1, %divf : tensor<3xf32>
  %acc3 = arith.addf %acc2, %remf : tensor<3xf32>
  %acc4 = arith.addf %acc3, %maxf : tensor<3xf32>
  %res = arith.addf %acc4, %minf : tensor<3xf32>

  %9 = tt.splat %ptr3 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %10 = tt.addptr %9, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %10, %res : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @reduce1D_1Tensor_Identity
// CHECK: arith.constant dense<1.0{{.*}}>
// CHECK: arith.select
// CHECK-NOT: "tt.reduce"(%{{.*}}, %{{.*}})
// CHECK: "tt.reduce"(%{{.*}})
// CHECK-NOT: arith.select
tt.func @reduce1D_1Tensor_Identity(%ptr1: !tt.ptr<f32>) -> f32 {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %res = arith.mulf %arg0, %arg1 : f32
      "tt.reduce.return"(%res) : (f32) -> ()
    }) : (tensor<3xf32>) -> f32

  tt.return %4 : f32
}

// -----

// CHECK-LABEL: @reduce1D_1Tensor_NAN_Identity
// CHECK-NOT: arith.constant dense<0x7FC00000>
// CHECK: arith.constant dense<0x7F800000>
// CHECK-NOT: arith.constant dense<0x7FC00000>
// CHECK: arith.select
// CHECK-NOT: "tt.reduce"(%{{.*}}, %{{.*}})
// CHECK: "tt.reduce"(%{{.*}})
// CHECK-NOT: arith.select
tt.func @reduce1D_1Tensor_NAN_Identity(%ptr1: !tt.ptr<f32>) -> f32 {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %res = arith.minnumf %arg0, %arg1 : f32
      "tt.reduce.return"(%res) : (f32) -> ()
    }) : (tensor<3xf32>) -> f32

  tt.return %4 : f32
}

// -----

// CHECK-LABEL: @reduce1D_1Tensor_NoIdentity
// CHECK-NOT: arith.constant dense<1.0{{.*}}>
// CHECK: tt.load
// CHECK-NOT: arith.select
// CHECK-NOT: "tt.reduce"(%{{.*}})
// CHECK-NOT: arith.constant dense<0.0{{.*}}>
// CHECK-NOT: arith.select
// CHECK: "tt.reduce"(%{{.*}}, %{{.*}})
// CHECK: arith.select
tt.func @reduce1D_1Tensor_NoIdentity(%ptr1: !tt.ptr<f32>) -> f32 {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %tmp = arith.mulf %arg0, %arg1 : f32
      %cst2 = arith.constant 2.0 : f32
      %res = arith.mulf %tmp, %cst2 : f32
      "tt.reduce.return"(%res) : (f32) -> ()
    }) : (tensor<3xf32>) -> f32

  tt.return %4 : f32
}

// -----

// CHECK-LABEL: @reduce1D_2Tensors_Mixed
// CHECK: arith.constant dense<1.0{{.*}}>
// CHECK: arith.select
// CHECK-NOT: "tt.reduce"(%{{.*}}, %{{.*}})
// CHECK: "tt.reduce"(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK: arith.select
// CHECK: arith.select
// CHECK-NOT: arith.select
tt.func @reduce1D_2Tensors_Mixed(%ptr1: !tt.ptr<f32>) -> f32 {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %first, %second = "tt.reduce"(%3, %3) <{axis = 0 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
      %res0 = arith.mulf %arg0, %arg2 : f32

      %tmp = arith.mulf %arg1, %arg3 : f32
      %cst2 = arith.constant 2.0 : f32
      %res1 = arith.mulf %tmp, %cst2 : f32
      "tt.reduce.return"(%res0, %res1) : (f32, f32) -> ()
    }) : (tensor<3xf32>, tensor<3xf32>) -> (f32, f32)
  
  %4 = arith.addf %first, %second : f32

  tt.return %4 : f32
}

// -----

// CHECK-LABEL: @reduce1D_2Tensors_Identities
// CHECK: arith.constant dense<0xFF800000>
// CHECK: tt.load
// CHECK: arith.constant dense<false>
// CHECK-NOT: "tt.reduce"(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK: "tt.reduce"(%{{.*}}, %{{.*}})
// CHECK-NOT: arith.select
// CHECK: tt.reduce.return
tt.func @reduce1D_2Tensors_Identities(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<i1>) -> f32 {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = tt.splat %ptr2 : !tt.ptr<i1> -> tensor<3x!tt.ptr<i1>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<i1>>, tensor<3xi32>
  %6 = tt.load %5 : tensor<3x!tt.ptr<i1>>

  %first, %second = "tt.reduce"(%3, %6) <{axis = 0 : i32}> ({
    ^bb0(%arg0: f32, %arg1: i1, %arg2: f32, %arg3: i1):
      %res0 = arith.maximumf %arg0, %arg2 : f32
      %res1 = arith.xori %arg1, %arg3 : i1
      "tt.reduce.return"(%res0, %res1) : (f32, i1) -> ()
    }) : (tensor<3xf32>, tensor<3xi1>) -> (f32, i1)

  %other = arith.constant 0.0 : f32
  %7 = arith.select %second, %first, %other : f32

  tt.return %7 : f32
}

// -----

// CHECK-LABEL: @reduce2D
// CHECK: %[[LOAD:.*]] = tt.load {{.*}} : !tt.ptr<tensor<8x4xf32>>
// CHECK: %[[PADDING:.*]] = arith.constant dense<1.0{{.*}}>
// CHECK: %[[SRC:.*]] = arith.select {{.*}}, %[[LOAD]], %[[PADDING]]
// CHECK: "tt.reduce"(%[[SRC]])
tt.func @reduce2D(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = arith.constant 64 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.constant 0 : i32
  %3 = tt.make_tensor_ptr %ptr1, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<8x3xf32>>
  %4 = tt.load %3 : !tt.ptr<tensor<8x3xf32>>

  %res = "tt.reduce"(%4) <{axis = 1 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %tmp = arith.mulf %arg0, %arg1 : f32
      "tt.reduce.return"(%tmp) : (f32) -> ()
    }) : (tensor<8x3xf32>) -> tensor<8xf32>
  
  %5 = tt.make_tensor_ptr %ptr2, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<8xf32>>
  tt.store %5, %res : !tt.ptr<tensor<8xf32>>
  tt.return
}

// -----

// CHECK-LABEL: @reduce2D_PowerTwoAxis
// CHECK-NOT: arith.constant dense<1.0{{.*}}>
tt.func @reduce2D_PowerTwoAxis(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %2 = tt.broadcast %1 : tensor<1x3xi32> -> tensor<2x3xi32>
  %3 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %4 = arith.constant dense<3> : tensor<2xi32>
  %5 = arith.muli %3, %4 : tensor<2xi32>
  %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x3xi32>
  %8 = arith.addi %2, %7 : tensor<2x3xi32>

  %9 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %11 = tt.load %10 : tensor<2x3x!tt.ptr<f32>>

  %res = "tt.reduce"(%11) <{axis = 0 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %res0 = arith.mulf %arg0, %arg1 : f32
      "tt.reduce.return"(%res0) : (f32) -> ()
    }) : (tensor<2x3xf32>) -> tensor<3xf32>
  
  %12 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %13 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %14 = tt.addptr %13, %12 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %14, %res : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scan1D_1Tensor_NoPad
// CHECK-NOT: arith.constant dense<0x7F800000>
// CHECK-NOT: "tt.scan"(%{{.*}}, %{{.*}})
// CHECK: "tt.scan"(%{{.*}})
// CHECK-NOT: "arith.select"
// CHECK: tt.scan.return
tt.func @scan1D_1Tensor_NoPad(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %res = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %res0 = arith.minnumf %arg0, %arg1 : f32
      "tt.scan.return"(%res0) : (f32) -> ()
    }) : (tensor<3xf32>) -> tensor<3xf32>
  
  %4 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %5 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %6, %res : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scan1D_1Tensor_Identity
// CHECK: arith.constant dense<0xFF800000>
// CHECK: arith.select
// CHECK-NOT: "tt.scan"(%{{.*}}, %{{.*}})
// CHECK: "tt.scan"(%{{.*}})
// CHECK-NOT: "arith.select"
// CHECK: tt.scan.return
tt.func @scan1D_1Tensor_Identity(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %res = "tt.scan"(%3) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %res0 = arith.maximumf %arg0, %arg1 : f32
      "tt.scan.return"(%res0) : (f32) -> ()
    }) : (tensor<3xf32>) -> tensor<3xf32>
  
  %4 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %5 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %6, %res : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scan1D_1Tensor_NoIdentity
// CHECK-NOT: "tt.scan"(%{{.*}})
// CHECK: "tt.scan"(%{{.*}}, %{{.*}})
// CHECK: arith.select
// CHECK: arith.select
// CHECK: tt.scan.return
tt.func @scan1D_1Tensor_NoIdentity(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %res = "tt.scan"(%3) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %tmp = arith.mulf %arg0, %arg1 : f32
      %cst1 = arith.constant 1.0 : f32
      %res0 = arith.addf %tmp, %cst1 : f32
      "tt.scan.return"(%res0) : (f32) -> ()
    }) : (tensor<3xf32>) -> tensor<3xf32>
  
  %4 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %5 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %6, %res : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scan1D_2Tensors_NoPad
// CHECK-NOT: arith.constant dense<0xFF800000>
// CHECK-NOT: "tt.scan"(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK: "tt.scan"(%{{.*}}, %{{.*}})
// CHECK-NOT: arith.select
// CHECK: tt.scan.return
tt.func @scan1D_2Tensors_NoPad(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %ptr3: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %5 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %7 = tt.load %6 : tensor<3x!tt.ptr<f32>>

  %res0, %res1 = "tt.scan"(%3, %7) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
      %cst_2 = arith.constant 2.0 : f32
      %tmp = arith.divf %arg0, %cst_2 : f32
      %scanRes0 = arith.addf %tmp, %arg2 : f32
      %scanRes1 = arith.maximumf %arg1, %arg3 : f32
      "tt.scan.return"(%scanRes0, %scanRes1) : (f32, f32) -> ()
    }) : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
  
  %res = arith.addf %res0, %res1 : tensor<3xf32>

  %8 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %9 = tt.splat %ptr3 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %10, %res : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scan1D_2Tensors_Identities
// CHECK: arith.constant dense<0>
// CHECK: arith.constant dense<false>
// CHECK-NOT: "tt.scan"(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK: "tt.scan"(%{{.*}}, %{{.*}})
// CHECK-NOT: arith.select
// CHECK: tt.scan.return
tt.func @scan1D_2Tensors_Identities(%ptr1: !tt.ptr<i32>, %ptr2: !tt.ptr<i1>, %ptr3: !tt.ptr<i32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<i32>>

  %4 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %5 = tt.splat %ptr2 : !tt.ptr<i1> -> tensor<3x!tt.ptr<i1>>
  %6 = tt.addptr %5, %4 : tensor<3x!tt.ptr<i1>>, tensor<3xi32>
  %7 = tt.load %6 : tensor<3x!tt.ptr<i1>>

  %res0, %res1 = "tt.scan"(%3, %7) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%arg0: i32, %arg1: i1, %arg2: i32, %arg3: i1):
      %scanRes0 = arith.maxui %arg0, %arg2 : i32
      %scanRes1 = arith.ori %arg1, %arg3 : i1
      "tt.scan.return"(%scanRes0, %scanRes1) : (i32, i1) -> ()
    }) : (tensor<3xi32>, tensor<3xi1>) -> (tensor<3xi32>, tensor<3xi1>)

  %other = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %res = arith.select %res1, %res0, %other : tensor<3xi1>, tensor<3xi32>
  
  %8 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %9 = tt.splat %ptr3 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %10 = tt.addptr %9, %8 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  tt.store %10, %res : tensor<3x!tt.ptr<i32>>
  tt.return
}

// -----

// CHECK-LABEL: @scan1D_2Tensors_Mixed
// CHECK: arith.constant dense<1.0{{.*}}e+00>
// CHECK-NOT: "tt.scan"(%{{.*}}, %{{.*}})
// CHECK: "tt.scan"(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK: arith.select
// CHECK: arith.select
// CHECK: tt.scan.return
tt.func @scan1D_2Tensors_Mixed(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %ptr3: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %5 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %7 = tt.load %6 : tensor<3x!tt.ptr<f32>>

  %res0, %res1 = "tt.scan"(%3, %7) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
      %tmp = arith.subf %arg0, %arg3 : f32
      %scanRes0 = arith.addf %arg2, %tmp : f32
      %scanRes1 = arith.mulf %arg1, %arg3 : f32
      "tt.scan.return"(%scanRes0, %scanRes1) : (f32, f32) -> ()
    }) : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
  
  %res = arith.addf %res0, %res1 : tensor<3xf32>

  %8 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %9 = tt.splat %ptr3 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %10, %res : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scan2D_PowerTwoAxis
// CHECK-NOT: arith.constant dense<1.0{{.*}}>
// CHECK-NOT: "tt.scan"(%{{.*}}, %{{.*}})
// CHECK: "tt.scan"(%{{.*}})
tt.func @scan2D_PowerTwoAxis(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %2 = tt.broadcast %1 : tensor<1x3xi32> -> tensor<2x3xi32>
  %3 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %4 = arith.constant dense<3> : tensor<2xi32>
  %5 = arith.muli %3, %4 : tensor<2xi32>
  %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x3xi32>
  %8 = arith.addi %2, %7 : tensor<2x3xi32>

  %9 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %11 = tt.load %10 : tensor<2x3x!tt.ptr<f32>>

  %res = "tt.scan"(%11) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %res0 = arith.mulf %arg0, %arg1 : f32
      "tt.scan.return"(%res0) : (f32) -> ()
    }) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  
  %12 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %13 = tt.addptr %12, %8 : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  tt.store %13, %res : tensor<2x3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scan2D_Identity
// CHECK: %[[LOAD:.*]] = tt.load {{.*}} : !tt.ptr<tensor<8x4xf32>>
// CHECK: %[[PADDING:.*]] = arith.constant dense<1.0{{.*}}>
// CHECK: %[[SRC:.*]] = arith.select {{.*}}, %[[LOAD]], %[[PADDING]]
// CHECK: "tt.scan"(%[[SRC]])
tt.func @scan2D_Identity(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = arith.constant 64 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.constant 0 : i32
  %3 = tt.make_tensor_ptr %ptr1, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<8x3xf32>>
  %4 = tt.load %3 : !tt.ptr<tensor<8x3xf32>>

  %res = "tt.scan"(%4) <{axis = 1 : i32, reverse=true}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %tmp = arith.mulf %arg0, %arg1 : f32
      "tt.scan.return"(%tmp) : (f32) -> ()
    }) : (tensor<8x3xf32>) -> tensor<8x3xf32>
  
  %5 = tt.make_tensor_ptr %ptr2, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<8x3xf32>>
  tt.store %5, %res : !tt.ptr<tensor<8x3xf32>>
  tt.return
}

// -----

// CHECK-LABEL: @scfForOpInnerYield
tt.func @scfForOpInnerYield(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %lowerbound: i32, %upperbound: i32, %step: i32) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %forRes = scf.for %i = %lowerbound to %upperbound step %step iter_args(%acc = %3) -> (tensor<3xf32>) : i32 {
    %4 = arith.addf %acc, %acc : tensor<3xf32>
    scf.yield %4 : tensor<3xf32>
  }

  %4 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %5, %forRes : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scfForOpIterArgYield
tt.func @scfForOpIterArgYield(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %lowerbound: i32, %upperbound: i32, %step: i32) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %forRes = scf.for %i = %lowerbound to %upperbound step %step iter_args(%acc = %3) -> (tensor<3xf32>) : i32 {
    scf.yield %acc : tensor<3xf32>
  }

  %4 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %5, %forRes : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scfForOpOuterYield
tt.func @scfForOpOuterYield(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %lowerbound: i32, %upperbound: i32, %step: i32) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %forRes = scf.for %i = %lowerbound to %upperbound step %step iter_args(%acc = %3) -> (tensor<3xf32>) : i32 {
    scf.yield %3 : tensor<3xf32>
  }

  %4 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %5, %forRes : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @scfIfOpInnerYield
tt.func @scfIfOpInnerYield(%ptr1: !tt.ptr<bf16>, %ptr2: !tt.ptr<bf16>, %flag: i1) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<bf16> -> tensor<3x!tt.ptr<bf16>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<bf16>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<bf16>>

  %res = scf.if %flag -> (tensor<3xi1>) {
    %cst_true = arith.constant dense<true> : tensor<3xi1>
    scf.yield %cst_true : tensor<3xi1>
  } else {
    %cst_false = arith.constant dense<false> : tensor<3xi1>
    scf.yield %cst_false : tensor<3xi1>
  }

  %4 = tt.splat %ptr2 : !tt.ptr<bf16> -> tensor<3x!tt.ptr<bf16>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<bf16>>, tensor<3xi32>
  tt.store %5, %3, %res : tensor<3x!tt.ptr<bf16>>
  tt.return
}

// -----

// CHECK-LABEL: @scfIfOpOuterYield
tt.func @scfIfOpOuterYield(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %ptr3: !tt.ptr<f32>, %flag: i1) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %6 = tt.load %5 : tensor<3x!tt.ptr<f32>>

  %res = scf.if %flag -> (tensor<3xf32>) {
    scf.yield %3 : tensor<3xf32>
  } else {
    scf.yield %6 : tensor<3xf32>
  }

  %7 = tt.splat %ptr3 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %8 = tt.addptr %7, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %8, %res : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @castOpExtf
tt.func @castOpExtf(%ptr1: !tt.ptr<f16>, %ptr2: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f16> -> tensor<3x!tt.ptr<f16>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f16>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f16>>

  %4 = arith.extf %3 : tensor<3xf16> to tensor<3xf32>

  %5 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %6 = tt.addptr %5, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %6, %4 : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @tensorPtr
// CHECK: %[[CST_64:.*]] = arith.constant 64 : i64
// CHECK: %[[CST_1:.*]] = arith.constant 1 : i64
// CHECK: %[[CST_0:.*]] = arith.constant 0 : i32
// CHECK: %[[LOADPTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%[[CST_64]], %[[CST_64]]], [%[[CST_64]], %[[CST_1]]], [%[[CST_0]], %[[CST_0]]] {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
// CHECK: %[[LOADVAL:.*]] = tt.load %[[LOADPTR]] {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<4x8xf32>
// CHECK: %[[STOREPTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%[[CST_64]], %[[CST_64]]], [%[[CST_64]], %[[CST_1]]], [%[[CST_0]], %[[CST_0]]] {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
// CHECK: tt.store %[[STOREPTR]], %[[LOADVAL]] {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<4x8xf32>
tt.func @tensorPtr(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = arith.constant 64 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.constant 0 : i32
  %3 = tt.make_tensor_ptr %ptr0, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<3x8xf32>>
  %4 = tt.load %3 : !tt.ptr<tensor<3x8xf32>>
  %5 = tt.make_tensor_ptr %ptr1, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<3x8xf32>>
  tt.store %5, %4 : !tt.ptr<tensor<3x8xf32>>
  tt.return
}

// -----

// CHECK-LABEL: @advanceTensorPtr
// CHECK: %[[CST_64:.*]] = arith.constant 64 : i64
// CHECK: %[[CST_1:.*]] = arith.constant 1 : i64
// CHECK: %[[CST_64_i32:.*]] = arith.trunci %[[CST_64:.*]] : i64 to i32
// CHECK: %[[CST_0:.*]] = arith.constant 0 : i32
// CHECK: %[[INITIALPTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%[[CST_64]], %[[CST_64]]], [%[[CST_64]], %[[CST_1]]], [%[[CST_0]], %[[CST_0]]] {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
// CHECK: %[[ADVANCEPTR:.*]] = tt.advance %[[INITIALPTR]], [%[[CST_64_i32]], %[[CST_0]]] : <tensor<4x8xf32>
// CHECK: %[[LOADVAL:.*]] = tt.load %[[ADVANCEPTR]] {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<4x8xf32>
// CHECK: %[[STOREPTR:.*]] = tt.make_tensor_ptr %{{.*}}, [%[[CST_64]], %[[CST_64]]], [%[[CST_64]], %[[CST_1]]], [%[[CST_0]], %[[CST_0]]] {order = array<i32: 1, 0>} : <tensor<4x8xf32>>
// CHECK: tt.store %[[STOREPTR]], %[[LOADVAL]] {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<4x8xf32>
tt.func @advanceTensorPtr(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = arith.constant 64 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.trunci %0 : i64 to i32
  %3 = arith.constant 0 : i32
  %4 = tt.make_tensor_ptr %ptr0, [%0, %0], [%0, %1], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<3x8xf32>>
  %5 = tt.advance %4, [%2, %3] : !tt.ptr<tensor<3x8xf32>>
  %6 = tt.load %5 : !tt.ptr<tensor<3x8xf32>>
  %7 = tt.make_tensor_ptr %ptr1, [%0, %0], [%0, %1], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<3x8xf32>>
  tt.store %7, %6 : !tt.ptr<tensor<3x8xf32>>
  tt.return
}

// -----

// CHECK: %[[RANGE:.*]] = tt.make_range 
// CHECK: %[[SPLAT_PTR:.*]] = tt.splat
// CHECK: %[[PTR:.*]] = tt.addptr %[[SPLAT_PTR]], %[[RANGE]]
// CHECK: %[[SPLAT_VAL:.*]] = tt.splat
// CHECK: %[[VAL_PTR:.*]] = tt.addptr %[[SPLAT_VAL]], %[[RANGE]]
// CHECK: %[[VAL:.*]] = tt.load %[[VAL_PTR]]
// CHECK: %[[CST_TRUE:.*]] = arith.constant dense<true>
// CHECK: %[[CST_FALSE:.*]] = arith.constant dense<false>
// CHECK: %[[MASK:.*]] = arith.select %{{.*}}, %[[CST_TRUE]], %[[CST_FALSE]]
// CHECK: tt.atomic_rmw fadd, relaxed, gpu, %[[PTR]], %[[VAL]], %[[MASK]] : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
tt.func @AtomicRMWOp(%ptr: !tt.ptr<f32>, %val: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.splat %val : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %4 = tt.addptr %3, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %5 = tt.load %4 : tensor<3x!tt.ptr<f32>>
  %old = tt.atomic_rmw fadd, relaxed, gpu, %2, %5 : (tensor<3x!tt.ptr<f32>>, tensor<3xf32>) -> tensor<3xf32>
  tt.return
}

// -----

// CHECK: %[[RANGE:.*]] = tt.make_range 
// CHECK: %[[SPLAT_PTR:.*]] = tt.splat
// CHECK: %[[PTR:.*]] = tt.addptr %[[SPLAT_PTR]], %[[RANGE]]
// CHECK: %[[SPLAT_VAL:.*]] = tt.splat
// CHECK: %[[VAL_PTR:.*]] = tt.addptr %[[SPLAT_VAL]], %[[RANGE]]
// CHECK: %[[VAL:.*]] = tt.load %[[VAL_PTR]]
// CHECK: %[[SPLAT_MASK:.*]] = tt.splat
// CHECK: %[[MASK_PTR:.*]] = tt.addptr %[[SPLAT_MASK]], %[[RANGE]]
// CHECK: %[[OLD_MASK:.*]] = tt.load %[[MASK_PTR]]
// CHECK: %[[CST_FALSE:.*]] = arith.constant dense<false>
// CHECK: %[[MASK:.*]] = arith.select %{{.*}}, %[[OLD_MASK]], %[[CST_FALSE]]
// CHECK: tt.atomic_rmw min, acquire, sys, %[[PTR]], %[[VAL]], %[[MASK]] : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
tt.func @AtomicRMWOpMasked(%ptr: !tt.ptr<i32>, %val: !tt.ptr<i32>, %mask: !tt.ptr<i1>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %3 = tt.splat %val : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %4 = tt.addptr %3, %0 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %5 = tt.load %4 : tensor<3x!tt.ptr<i32>>
  %6 = tt.splat %mask : !tt.ptr<i1> -> tensor<3x!tt.ptr<i1>>
  %7 = tt.addptr %6, %0 : tensor<3x!tt.ptr<i1>>, tensor<3xi32>
  %8 = tt.load %7 : tensor<3x!tt.ptr<i1>>
  %old = tt.atomic_rmw min, acquire, sys, %2, %5, %8  : (tensor<3x!tt.ptr<i32>>, tensor<3xi32>, tensor<3xi1>) -> tensor<3xi32>
  tt.return
}

// -----

// CHECK-LABEL: @dotOp
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: %[[VAL1:.*]] = tt.expand_dims %[[LOAD]] {axis = 0 : i32}
// CHECK: %[[CST0_1:.*]] = arith.constant dense<0.0{{.*}}> : tensor<1x4xf32>
// CHECK: %[[SRC1:.*]] = arith.select %{{.*}}, %[[VAL1]], %[[CST0_1]]
// CHECK: %[[VAL2:.*]] = tt.expand_dims %[[LOAD]] {axis = 1 : i32}
// CHECK: %[[CST0_2:.*]] = arith.constant dense<0.0{{.*}}> : tensor<4x1xf32>
// CHECK: %[[SRC2:.*]] = arith.select %{{.*}}, %[[VAL2]], %[[CST0_2]]
// CHECK: tt.dot %[[SRC1]], %[[SRC2]], %{{.*}} : tensor<1x4xf32> * tensor<4x1xf32> -> tensor<1x1xf32>
tt.func @dotOp(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) -> f32 {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<3xf32> -> tensor<1x3xf32>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<3xf32> -> tensor<3x1xf32>
  %6 = arith.constant dense<0.0> : tensor<1x1xf32>
  %7 = tt.dot %4, %5, %6 : tensor<1x3xf32> * tensor<3x1xf32> -> tensor<1x1xf32> 
  %8 = tt.unsplat %7 : tensor<1x1xf32>
  tt.return %8 : f32
}

// -----
// CHECK-LABEL: @dotOpDiffElementTypes
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: %[[VAL1:.*]] = tt.expand_dims %[[LOAD]] {axis = 0 : i32}
// CHECK: %[[CST0_1:.*]] = arith.constant dense<0.0{{.*}}> : tensor<1x4xf16>
// CHECK: %[[SRC1:.*]] = arith.select %{{.*}}, %[[VAL1]], %[[CST0_1]]
// CHECK: %[[VAL2:.*]] = tt.expand_dims %[[LOAD]] {axis = 1 : i32}
// CHECK: %[[CST0_2:.*]] = arith.constant dense<0.0{{.*}}> : tensor<4x1xf16>
// CHECK: %[[SRC2:.*]] = arith.select %{{.*}}, %[[VAL2]], %[[CST0_2]]
// CHECK: tt.dot %[[SRC1]], %[[SRC2]], %{{.*}} : tensor<1x4xf16> * tensor<4x1xf16> -> tensor<1x1xf32>
tt.func @dotOpDiffElementTypes(%ptr1: !tt.ptr<f16>, %ptr2: !tt.ptr<f32>) -> f32 {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f16> -> tensor<3x!tt.ptr<f16>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f16>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f16>>
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<3xf16> -> tensor<1x3xf16>
  %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<3xf16> -> tensor<3x1xf16>
  %6 = arith.constant dense<0.0> : tensor<1x1xf32>
  %7 = tt.dot %4, %5, %6 : tensor<1x3xf16> * tensor<3x1xf16> -> tensor<1x1xf32> 
  %8 = tt.unsplat %7 : tensor<1x1xf32>
  tt.return %8 : f32
}

// -----

// CHECK-LABEL: @histogramNonPowTwoSource
// CHECK: tt.histogram {{.*}} : tensor<4xi32> -> tensor<4xi32>
tt.func @histogramNonPowTwoSource(%ptr1: !tt.ptr<i32>, %ptr2: !tt.ptr<i32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<i32>>

  %4 = tt.histogram %3 : tensor<3xi32> -> tensor<4xi32>

  %5 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %6 = tt.splat %ptr2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %7 = tt.addptr %6, %5 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  tt.store %7, %4 : tensor<4x!tt.ptr<i32>>
  tt.return
}

// -----

// CHECK-LABEL: @histogramNonPowTwoSourceMasked
// CHECK: %[[OLDMASK:.*]] = tt.load {{.*}} : tensor<4x!tt.ptr<i1>>
// CHECK: %[[MASK:.*]] = arith.andi %[[OLDMASK]], {{.*}} : tensor<4xi1>
// CHECK: tt.histogram {{.*}}, %[[MASK]] : tensor<4xi32> -> tensor<4xi32>
tt.func @histogramNonPowTwoSourceMasked(%ptr1: !tt.ptr<i32>, %ptr2: !tt.ptr<i1>, %ptr3: !tt.ptr<i32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<i32>>

  %4 = tt.splat %ptr2 : !tt.ptr<i1> -> tensor<3x!tt.ptr<i1>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<i1>>, tensor<3xi32>
  %6 = tt.load %5 : tensor<3x!tt.ptr<i1>>

  %7 = tt.histogram %3, %6 : tensor<3xi32> -> tensor<4xi32>

  %8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %9 = tt.splat %ptr3 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %10 = tt.addptr %9, %8 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  tt.store %10, %7 : tensor<4x!tt.ptr<i32>>
  tt.return
}

// -----

// CHECK-LABEL: @histogramNonPowTwoResult
// CHECK: %[[SOURCE:.*]] = tt.load {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.histogram %[[SOURCE]] : tensor<4xi32> -> tensor<4xi32>
tt.func @histogramNonPowTwoResult(%ptr1: !tt.ptr<i32>, %ptr2: !tt.ptr<i32>) {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %3 = tt.load %2 : tensor<4x!tt.ptr<i32>>

  %4 = tt.histogram %3 : tensor<4xi32> -> tensor<3xi32>

  %5 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %6 = tt.splat %ptr2 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %7 = tt.addptr %6, %5 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  tt.store %7, %4 : tensor<3x!tt.ptr<i32>>
  tt.return
}

// -----

// CHECK-LABEL: @histogramNonPowTwoSourceResult
// CHECK: tt.histogram {{.*}}, {{.*}} : tensor<4xi32> -> tensor<4xi32>
tt.func @histogramNonPowTwoSourceResult(%ptr1: !tt.ptr<i32>, %ptr2: !tt.ptr<i32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<i32>>

  %4 = tt.histogram %3 : tensor<3xi32> -> tensor<3xi32>

  %5 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %6 = tt.splat %ptr2 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %7 = tt.addptr %6, %5 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  tt.store %7, %4 : tensor<3x!tt.ptr<i32>>
  tt.return
}

// -----

// CHECK-LABEL: @histogramNonPowTwoSourceResultMasked
// CHECK: %[[SOURCE:.*]] = tt.load {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: %[[OLDMASK:.*]] = tt.load {{.*}} : tensor<4x!tt.ptr<i1>>
// CHECK: %[[MASK:.*]] = arith.andi %[[OLDMASK]], {{.*}} : tensor<4xi1>
// CHECK: tt.histogram %[[SOURCE]], %[[MASK]] : tensor<4xi32> -> tensor<4xi32>
tt.func @histogramNonPowTwoSourceResultMasked(%ptr1: !tt.ptr<i32>, %ptr2: !tt.ptr<i1>, %ptr3: !tt.ptr<i32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<i32>>

  %4 = tt.splat %ptr2 : !tt.ptr<i1> -> tensor<3x!tt.ptr<i1>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<i1>>, tensor<3xi32>
  %6 = tt.load %5 : tensor<3x!tt.ptr<i1>>

  %7 = tt.histogram %3, %6 : tensor<3xi32> -> tensor<3xi32>

  %8 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %9 = tt.splat %ptr3 : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %10 = tt.addptr %9, %8 : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  tt.store %10, %7 : tensor<3x!tt.ptr<i32>>
  tt.return
}

// -----

// CHECK-LABEL: @simpleReshapeDataLocs
// CHECK-NOT: tt.reshape {{.*}} : tensor<8xf32> -> tensor<8xf32>
// CHECK: tt.reshape {{.*}} : tensor<8xf32> -> tensor<4x2xf32>
tt.func @simpleReshapeDataLocs(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 6 : i32, start = 0 : i32} : tensor<6xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<6x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<6x!tt.ptr<f32>>, tensor<6xi32>
  %3 = tt.load %2 : tensor<6x!tt.ptr<f32>>
  %4 = tt.reshape %3 : tensor<6xf32> -> tensor<3x2xf32>

  %5 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %7 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<3x2xi32>
  %8 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %9 = arith.constant dense<2> : tensor<3xi32>
  %10 = arith.muli %8, %9 : tensor<3xi32>
  %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
  %12 = tt.broadcast %11 : tensor<3x1xi32> -> tensor<3x2xi32>  

  %13 = arith.addi %7, %12 : tensor<3x2xi32>
  %14 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x2x!tt.ptr<f32>>
  %15 = tt.addptr %14, %13 : tensor<3x2x!tt.ptr<f32>>, tensor<3x2xi32>

  tt.store %15, %4 : tensor<3x2x!tt.ptr<f32>>

  tt.return
}

// -----

// CHECK-LABEL: @simpleReshapeOnes
// CHECK: tt.reshape {{.*}} tensor<4x!tt.ptr<f32>> -> tensor<1x4x1x1x!tt.ptr<f32>>
// CHECK: tt.reshape {{.*}} tensor<4xi32> -> tensor<1x4x1x1xi32>
tt.func @simpleReshapeOnes(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.reshape %2 : tensor<3x!tt.ptr<f32>> -> tensor<1x3x1x1x!tt.ptr<f32>>
  %4 = tt.load %3 : tensor<1x3x1x1x!tt.ptr<f32>>
  
  %5 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %6 = tt.reshape %5 : tensor<3xi32> -> tensor<1x3x1x1xi32>
  %7 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<1x3x1x1x!tt.ptr<f32>>
  %8 = tt.addptr %7, %6 : tensor<1x3x1x1x!tt.ptr<f32>>, tensor<1x3x1x1xi32>
  tt.store %8, %4 : tensor<1x3x1x1x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @complexReshapeSameLogicalSize
// CHECK: tt.reshape {{.*}} : tensor<4x2x!tt.ptr<f32>> -> tensor<8x!tt.ptr<f32>>
// CHECK: %[[CST_0:.*]] = arith.constant 0 : i64
// CHECK: %[[NULLPTR:.*]] = tt.int_to_ptr %[[CST_0]] : i64 -> !tt.ptr<f32>
// CHECK: tt.splat %[[NULLPTR]] : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
tt.func @complexReshapeSameLogicalSize(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>
  %4 = tt.reshape %3 : tensor<3xf32> -> tensor<1x3xf32>
  %5 = tt.broadcast %4 : tensor<1x3xf32> -> tensor<2x3xf32>

  %6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %8 = tt.broadcast %7 : tensor<1x2xi32> -> tensor<3x2xi32>
  %9 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %10 = arith.constant dense<2> : tensor<3xi32>
  %11 = arith.muli %9, %10 : tensor<3xi32>
  %12 = tt.expand_dims %11 {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
  %13 = tt.broadcast %12 : tensor<3x1xi32> -> tensor<3x2xi32>  

  %14 = arith.addi %8, %13 : tensor<3x2xi32>
  %15 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x2x!tt.ptr<f32>>
  %16 = tt.addptr %15, %14 : tensor<3x2x!tt.ptr<f32>>, tensor<3x2xi32>
  %17 = tt.reshape %16 : tensor<3x2x!tt.ptr<f32>> -> tensor<2x3x!tt.ptr<f32>>
  tt.store %17, %5 : tensor<2x3x!tt.ptr<f32>>

  tt.return
}

// -----

// CHECK-LABEL: @complexReshapeLargerSource
// CHECK: tt.reshape {{.*}} : tensor<4x16xf32> -> tensor<64xf32>
// CHECK: arith.constant dense<0.0{{.*}}e+00> : tensor<32xf32>
tt.func @complexReshapeLargerSource(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 9 : i32, start = 0 : i32} : tensor<9xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<9xi32> -> tensor<1x9xi32>
  %2 = tt.broadcast %1 : tensor<1x9xi32> -> tensor<3x9xi32>
  %3 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %4 = arith.constant dense<9> : tensor<3xi32>
  %5 = arith.muli %3, %4 : tensor<3xi32>
  %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
  %7 = tt.broadcast %6 : tensor<3x1xi32> -> tensor<3x9xi32> 
  %8 = arith.addi %2, %7 : tensor<3x9xi32>
  %9 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<3x9x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<3x9x!tt.ptr<f32>>, tensor<3x9xi32>
  %11 = tt.load %10 : tensor<3x9x!tt.ptr<f32>> 
  %12 = tt.reshape %11 : tensor<3x9xf32> -> tensor<27xf32>

  %13 = tt.make_range {end = 27 : i32, start = 0 : i32} : tensor<27xi32>
  %14 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<27x!tt.ptr<f32>>
  %15 = tt.addptr %14, %13 : tensor<27x!tt.ptr<f32>>, tensor<27xi32>
  tt.store %15, %12 : tensor<27x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @complexReshapeLargerResult
// CHECK-NOT: tt.reshape {{.*}} : tensor<32xf32> -> tensor<32xf32>
tt.func @complexReshapeLargerResult(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 27 : i32, start = 0 : i32} : tensor<27xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<27x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<27x!tt.ptr<f32>>, tensor<27xi32>
  %3 = tt.load %2 : tensor<27x!tt.ptr<f32>>
  %4 = tt.reshape %3 : tensor<27xf32> -> tensor<3x9xf32>

  %7 = tt.make_range {end = 9 : i32, start = 0 : i32} : tensor<9xi32>
  %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<9xi32> -> tensor<1x9xi32>
  %9 = tt.broadcast %8 : tensor<1x9xi32> -> tensor<3x9xi32>
  %10 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %11 = arith.constant dense<9> : tensor<3xi32>
  %12 = arith.muli %10, %11 : tensor<3xi32>
  %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
  %14 = tt.broadcast %13 : tensor<3x1xi32> -> tensor<3x9xi32>  

  %15 = arith.addi %9, %14 : tensor<3x9xi32>
  %16 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x9x!tt.ptr<f32>>
  %17 = tt.addptr %16, %15 : tensor<3x9x!tt.ptr<f32>>, tensor<3x9xi32>
  tt.store %17, %4 : tensor<3x9x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @extractSlice
// CHECK: tensor.extract_slice {{.*}}[2] [8] [1]
tt.func @extractSlice(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 9 : i32, start = 0 : i32} : tensor<9xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<9x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<9x!tt.ptr<f32>>, tensor<9xi32>
  %3 = tt.load %2 : tensor<9x!tt.ptr<f32>>

  %4 = tensor.extract_slice %3[2] [5] [1] : tensor<9xf32> to tensor<5xf32>

  %5 = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
  %6 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<5x!tt.ptr<f32>>
  %7 = tt.addptr %6, %5 : tensor<5x!tt.ptr<f32>>, tensor<5xi32>
  tt.store %7, %4 : tensor<5x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @extractSliceSplit
// CHECK: %[[SOURCE:.*]] = tt.addptr {{.*}}, {{.*}} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[NULLPTR:.*]] = tt.int_to_ptr %[[ZERO]] : i64 -> !tt.ptr<f32>
// CHECK: %[[DST0:.*]] = tt.splat %[[NULLPTR]] : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
// CHECK: %[[EXT1:.*]] = tensor.extract_slice {{.*}}[10] [4] [1]
// CHECK: %[[DST1:.*]] = tensor.insert_slice %[[EXT1]] into %[[DST0]][0] [4] [1]
// CHECK: %[[EXT2:.*]] = tensor.extract_slice {{.*}}[14] [1] [1]
// CHECK: tensor.insert_slice %[[EXT2]] into %[[DST1]][4] [1] [1]
tt.func @extractSliceSplit(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 15 : i32, start = 0 : i32} : tensor<15xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<15x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<15x!tt.ptr<f32>>, tensor<15xi32>

  %3 = tensor.extract_slice %2[10] [5] [1] : tensor<15x!tt.ptr<f32>> to tensor<5x!tt.ptr<f32>>
  %4 = tt.load %3 : tensor<5x!tt.ptr<f32>>

  %5 = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
  %6 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<5x!tt.ptr<f32>>
  %7 = tt.addptr %6, %5 : tensor<5x!tt.ptr<f32>>, tensor<5xi32>
  tt.store %7, %4 : tensor<5x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @insertSlice
// CHECK: tensor.insert_slice {{.*}} into {{.*}}[0, 3] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<4x8xf32>
tt.func @insertSlice(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
  %2 = tt.broadcast %1 : tensor<1x4xi32> -> tensor<3x4xi32>
  %3 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %4 = arith.constant dense<4> : tensor<3xi32>
  %5 = arith.muli %3, %4 : tensor<3xi32>
  %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
  %7 = tt.broadcast %6 : tensor<3x1xi32> -> tensor<3x4xi32>
  %8 = arith.addi %2, %7 : tensor<3x4xi32>
  %9 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<3x4x!tt.ptr<f32>>
  %10 = tt.addptr %9, %8 : tensor<3x4x!tt.ptr<f32>>, tensor<3x4xi32>
  %11 = tt.load %10 : tensor<3x4x!tt.ptr<f32>>

  %12 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %14 = tt.broadcast %13 : tensor<1x8xi32> -> tensor<3x8xi32>
  %15 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %16 = arith.constant dense<3> : tensor<3xi32>
  %17 = arith.muli %15, %16 : tensor<3xi32>
  %18 = tt.expand_dims %17 {axis = 1 : i32} : tensor<3xi32> -> tensor<3x1xi32>
  %19 = tt.broadcast %18 : tensor<3x1xi32> -> tensor<3x8xi32>
  %20 = arith.addi %14, %19 : tensor<3x8xi32>
  %21 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x8x!tt.ptr<f32>>
  %22 = tt.addptr %21, %20 : tensor<3x8x!tt.ptr<f32>>, tensor<3x8xi32>
  %23 = tt.load %22 : tensor<3x8x!tt.ptr<f32>>

  %24 = tensor.insert_slice %11 into %23[0, 3] [3, 4] [1, 1] : tensor<3x4xf32> into tensor<3x8xf32>

  tt.store %22, %24 : tensor<3x8x!tt.ptr<f32>>

  tt.return
}

// -----

// CHECK-LABEL: @insertSliceSplit
// CHECK: %[[EXT1:.*]] = tensor.extract_slice {{.*}}[0] [4] [1]
// CHECK: %[[DST1:.*]] = tensor.insert_slice %[[EXT1]] into {{.*}}[2] [4] [1]
// CHECK: %[[EXT2:.*]] = tensor.extract_slice {{.*}}[4] [1] [1]
// CHECK: tensor.insert_slice %[[EXT2]] into %[[DST1]][6] [1] [1]
tt.func @insertSliceSplit(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<5x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<5x!tt.ptr<f32>>, tensor<5xi32>
  %3 = tt.load %2 : tensor<5x!tt.ptr<f32>>

  %4 = tt.make_range {end = 9 : i32, start = 0 : i32} : tensor<9xi32>
  %5 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<9x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<9x!tt.ptr<f32>>, tensor<9xi32>
  %7 = tt.load %6 : tensor<9x!tt.ptr<f32>>

  %8 = tensor.insert_slice %3 into %7[2] [5] [1] : tensor<5xf32> into tensor<9xf32>
  tt.store %6, %8 : tensor<9x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @insertSliceSplitBounds
// CHECK: %[[EXT1:.*]] = tensor.extract_slice {{.*}}[0] [4] [1]
// CHECK: %[[DST1:.*]] = tensor.insert_slice %[[EXT1]] into {{.*}}[10] [4] [1]
// CHECK: %[[EXT2:.*]] = tensor.extract_slice {{.*}}[4] [1] [1]
// CHECK: tensor.insert_slice %[[EXT2]] into %[[DST1]][14] [1] [1]
tt.func @insertSliceSplitBounds(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<5x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<5x!tt.ptr<f32>>, tensor<5xi32>
  %3 = tt.load %2 : tensor<5x!tt.ptr<f32>>

  %4 = tt.make_range {end = 15 : i32, start = 0 : i32} : tensor<15xi32>
  %5 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<15x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<15x!tt.ptr<f32>>, tensor<15xi32>
  %7 = tt.load %6 : tensor<15x!tt.ptr<f32>>

  %8 = tensor.insert_slice %3 into %7[10] [5] [1] : tensor<5xf32> into tensor<15xf32>
  tt.store %6, %8 : tensor<15x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @insertSliceNoSplit
// CHECK-NOT: tensor.extract_slice
tt.func @insertSliceNoSplit(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %0 = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
  %1 = tt.splat %ptr0 : !tt.ptr<f32> -> tensor<5x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<5x!tt.ptr<f32>>, tensor<5xi32>
  %3 = tt.load %2 : tensor<5x!tt.ptr<f32>>

  %4 = tt.make_range {end = 9 : i32, start = 0 : i32} : tensor<9xi32>
  %5 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<9x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<9x!tt.ptr<f32>>, tensor<9xi32>
  %7 = tt.load %6 : tensor<9x!tt.ptr<f32>>

  %8 = tensor.insert_slice %3 into %7[4] [5] [1] : tensor<5xf32> into tensor<9xf32>
  tt.store %6, %8 : tensor<9x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @extract
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: tensor.extract %[[LOAD]][%{{.*}}] : tensor<4xf32>
tt.func @extract(%ptr1: !tt.ptr<f32>, %index: index) -> f32 {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>
  %4 = tensor.extract %3[%index] : tensor<3xf32>
  tt.return %4 : f32
}

// -----

// CHECK-LABEL: @insert
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: tensor.insert %{{.*}} into %[[LOAD]][%{{.*}}] : tensor<4xf32>
tt.func @insert(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %index: index, %val: f32) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>
  %4 = tensor.insert %val into %3[%index] : tensor<3xf32>

  %5 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %6 = tt.addptr %5, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  tt.store %6, %4 : tensor<3x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @transpose
// CHECK: tt.trans {{.*}} {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
tt.func @transpose(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>) {
  %0 = arith.constant 64 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.constant 0 : i32
  %3 = tt.make_tensor_ptr %ptr1, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<3x8xf32>>
  %4 = tt.load %3 : !tt.ptr<tensor<3x8xf32>>

  %5 = tt.trans %4 {order = array<i32: 1, 0>} : tensor<3x8xf32> -> tensor<8x3xf32>
  %6 = tt.make_tensor_ptr %ptr1, [%0, %0], [%0, %1], [%2, %2] {order = array<i32: 1, 0>} : !tt.ptr<tensor<8x3xf32>>
  tt.store %6, %5 : !tt.ptr<tensor<8x3xf32>>
  tt.return
}

// -----

// CHECK-LABEL: @nestedAdvance
// CHECK: %[[CST3:.*]] = arith.constant 3 : i32
// CHECK: %[[CST0:.*]] = arith.constant 0 : i32
// CHECK: %[[PTR:.*]] = tt.make_tensor_ptr {{.*}} : <tensor<4x8xi1>
// CHECK: scf.for {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}, %[[MASKPTR:.*]] = %[[PTR]]) -> (tensor<4x8xf32>, !tt.ptr<tensor<4x8xi1>>) : i32
// CHECK: %[[ADVANCE:.*]] = tt.advance %[[MASKPTR]], [%[[CST3]], %[[CST0]]] : <tensor<4x8xi1>>
// CHECK: scf.yield {{.*}}, %[[ADVANCE]]
tt.func @nestedAdvance(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<i1>, %ptr3: !tt.ptr<f32>, %lowerbound: i32, %upperbound: i32, %step: i32) {
  %0 = arith.constant 64 : i64
  %1 = arith.constant 1 : i64
  %2 = arith.constant 3 : i32
  %3 = arith.constant 0 : i32
  %4 = tt.make_tensor_ptr %ptr1, [%0, %0], [%0, %1], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<3x8xf32>>
  %5 = tt.make_tensor_ptr %ptr2, [%0, %0], [%0, %1], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<3x8xi1>>

  %init = tt.load %4 : !tt.ptr<tensor<3x8xf32>>
  %other = arith.constant dense<0.0> : tensor<3x8xf32>
  %forRes0, %forRes1 = scf.for %i = %lowerbound to %upperbound step %step iter_args(%acc = %init, %maskptr = %5) -> (tensor<3x8xf32>, !tt.ptr<tensor<3x8xi1>>) : i32 {
    %mask = tt.load %maskptr : !tt.ptr<tensor<3x8xi1>>
    %select = arith.select %mask, %acc, %other : tensor<3x8xi1>, tensor<3x8xf32>
    %newptr = tt.advance %maskptr, [%2, %3] : !tt.ptr<tensor<3x8xi1>>
    scf.yield %select, %newptr : tensor<3x8xf32>, !tt.ptr<tensor<3x8xi1>>
  }

  %6 = tt.make_tensor_ptr %ptr3, [%0, %0], [%0, %1], [%3, %3] {order = array<i32: 1, 0>} : !tt.ptr<tensor<3x8xf32>>
  tt.store %6, %forRes0 : !tt.ptr<tensor<3x8xf32>>
  tt.return
}

// -----

// CHECK-LABEL: @nestedStore
// CHECK: scf.for {{.*}} iter_args(%[[ACCMASK:.*]] = %{{.*}})
// CHECK: %[[OTHER:.*]] = arith.constant dense<false> : tensor<4xi1>
// CHECK: %[[MASK:.*]] = arith.select {{.*}}, %[[ACCMASK]], %[[OTHER]]
// CHECK: tt.store {{.*}}, {{.*}}, %[[MASK]]
tt.func @nestedStore(%ptr1: !tt.ptr<f32>, %ptr2: !tt.ptr<f32>, %lowerbound: i32, %upperbound: i32, %step: i32) {
  %0 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %1 = tt.splat %ptr1 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>
  %3 = tt.load %2 : tensor<3x!tt.ptr<f32>>

  %4 = tt.splat %ptr2 : !tt.ptr<f32> -> tensor<3x!tt.ptr<f32>>
  %5 = tt.addptr %4, %0 : tensor<3x!tt.ptr<f32>>, tensor<3xi32>

  %6 = arith.constant dense<true> : tensor<3xi1>

  scf.for %i = %lowerbound to %upperbound step %step iter_args(%acc = %6) -> tensor<3xi1> : i32 {
    tt.store %5, %3, %acc : tensor<3x!tt.ptr<f32>>
    scf.yield %acc : tensor<3xi1>
  }
  tt.return
}

// -----

// CHECK-LABEL: @power_of_two_ops
// CHECK:       %[[LOAD:.*]] = tt.load
// CHECK:       %[[TMP_RESHAPE:.*]] = tt.reshape %[[LOAD]] : tensor<1x16xf32> -> tensor<4x4xf32>
// CHECK:       %[[EXTRACT:.*]] = tensor.extract_slice %[[TMP_RESHAPE]][0, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
// CHECK:       %[[BROADCAST:.*]] = tt.broadcast %[[EXTRACT]] : tensor<1x4xf32> -> tensor<4x4xf32>
// CHECK:       %[[FLATTEN:.*]] = tt.reshape %[[BROADCAST]] : tensor<4x4xf32> -> tensor<16xf32>
// CHECK:       %[[ARITH:.*]] = arith.addf %[[FLATTEN]], %{{.*}} : tensor<16xf32>
// CHECK:       %[[CAST:.*]] = arith.fptosi %[[ARITH]] : tensor<16xf32> to tensor<16xi32>
// CHECK:       %[[RESHAPE:.*]] = tt.reshape %[[CAST]] : tensor<16xi32> -> tensor<4x4xi32>
// CHECK:       %[[REDUCE:.*]] = "tt.reduce"(%[[RESHAPE]]) <{axis = 0 : i32}>
// CHECK:       %[[INSERT:.*]] = tensor.insert_slice %[[REDUCE]] into %{{.*}}[2] [4] [1] : tensor<4xi32> into tensor<8xi32>
// CHECK:       tt.store %{{.*}}, %[[INSERT]] : tensor<8x!tt.ptr<i32>>
tt.func @power_of_two_ops(%ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<i32>) {
  // Base constants for SSA values
  %c0 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i32
  %c4 = arith.constant 4 : i32
  %true = arith.constant true
  %f0 = arith.constant 0.000000e+00 : f32
  %f2 = arith.constant 2.000000e+00 : f32

  // Power of two offsets and pointer arithmetic
  %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %reshape_offsets = tt.reshape %offsets : tensor<16xi32> -> tensor<1x16xi32>
  %splat_ptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
  %input_ptrs = tt.addptr %splat_ptr, %reshape_offsets : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>

  // Load 2D power of two tensor using SSA-bound splats
  %mask = tt.splat %true : i1 -> tensor<1x16xi1>
  %pass_thru = tt.splat %f0 : f32 -> tensor<1x16xf32>
  %loaded = tt.load %input_ptrs, %mask, %pass_thru : tensor<1x16x!tt.ptr<f32>>

  %tmpReshape = tt.reshape %loaded : tensor<1x16xf32> -> tensor<4x4xf32>

  // Extract slice on only one dimension (extracting the 16-element row)
  %extracted = tensor.extract_slice %tmpReshape [0, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>

  %extractedBroadcasted = tt.broadcast %extracted : tensor<1x4xf32> -> tensor<4x4xf32>
  %flattened = tt.reshape %extractedBroadcasted : tensor<4x4xf32> -> tensor<16xf32>

  // Arith elementwise operation
  %cst = tt.splat %f2 : f32 -> tensor<16xf32>
  %added = arith.addf %flattened, %cst : tensor<16xf32>

  // Cast operation (Float to Signed Integer)
  %casted = arith.fptosi %added : tensor<16xf32> to tensor<16xi32>

  // Reshape operation (1D 16 -> 2D 4x4)
  %reshaped = tt.reshape %casted : tensor<16xi32> -> tensor<4x4xi32>

  // Triton reduction operation along a power of two dimension
  %reduced = "tt.reduce"(%reshaped) <{axis = 0: i32}> ({
  ^bb0(%arg0: i32, %arg1: i32):
    %sum = arith.addi %arg0, %arg1 : i32
    tt.reduce.return %sum : i32
  }) : (tensor<4x4xi32>) -> tensor<4xi32>

  // Insert slice on only one dimension back into a 2D canvas
  %init_val = tt.splat %c0 : i32 -> tensor<8xi32>
  %inserted = tensor.insert_slice %reduced into %init_val [2] [4] [1] : tensor<4xi32> into tensor<8xi32>

  // Store output back to memory
  %out_offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %splat_out_ptr = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
  %output_ptrs = tt.addptr %splat_out_ptr, %out_offsets : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
  
  tt.store %output_ptrs, %inserted : tensor<8x!tt.ptr<i32>>

  tt.return
}
