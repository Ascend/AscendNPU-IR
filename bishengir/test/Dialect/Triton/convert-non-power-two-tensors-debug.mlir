// RUN: bishengir-opt -split-input-file -convert-non-power-two-tensors %s | FileCheck %s

// A forward scan normally ignores the padded tail. Overflow assertions in its
// combine region make that tail observable, so its inputs need the identity.
// CHECK-LABEL: tt.func @forwardScanWithOverflowCheck
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : tensor<64xi32>
// CHECK: %[[PADDED:.*]] = arith.select %{{.*}}, %[[LOAD]], %[[ZERO]]
// CHECK: "tt.scan"(%[[PADDED]])
// CHECK: tt.assert %{{.*}}, "overflow" : i1
tt.func @forwardScanWithOverflowCheck(%ptr: !tt.ptr<i32>) {
  %r = tt.make_range {end = 33 : i32, start = 0 : i32} : tensor<33xi32>
  %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<33x!tt.ptr<i32>>
  %p = tt.addptr %base, %r : tensor<33x!tt.ptr<i32>>, tensor<33xi32>
  %v = tt.load %p : tensor<33x!tt.ptr<i32>>
  %result = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %b: i32):
    %a64 = arith.extsi %a : i32 to i64
    %b64 = arith.extsi %b : i32 to i64
    %wide = arith.addi %a64, %b64 : i64
    %max = arith.constant 2147483647 : i64
    %min = arith.constant -2147483648 : i64
    %upper = arith.cmpi sle, %wide, %max : i64
    %lower = arith.cmpi sge, %wide, %min : i64
    %ok = arith.andi %upper, %lower : i1
    tt.assert %ok, "overflow" : i1
    %sum = arith.addi %a, %b : i32
    tt.scan.return %sum : i32
  }) : (tensor<33xi32>) -> tensor<33xi32>
  tt.store %p, %result : tensor<33x!tt.ptr<i32>>
  tt.return
}

// -----

// Even when the reduction axis already has power-of-two size, the expanded
// parallel axis must not introduce undefined inputs to the overflow checks.
// CHECK-LABEL: tt.func @reduceWithPaddedParallelAxis
// CHECK: %[[VALUES:.*]] = tt.load
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : tensor<4x4xi32>
// CHECK: %[[PADDED:.*]] = arith.select %{{.*}}, %[[VALUES]], %[[ZERO]]
// CHECK: "tt.reduce"(%[[PADDED]])
// CHECK: tt.assert %{{.*}}, "overflow" : i1
tt.func @reduceWithPaddedParallelAxis(%ptr: !tt.ptr<i32>) {
  %r = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %p = tt.addptr %base, %r : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %col = tt.expand_dims %p {axis = 1 : i32} : tensor<3x!tt.ptr<i32>> -> tensor<3x1x!tt.ptr<i32>>
  %ptrs = tt.broadcast %col : tensor<3x1x!tt.ptr<i32>> -> tensor<3x4x!tt.ptr<i32>>
  %values = tt.load %ptrs : tensor<3x4x!tt.ptr<i32>>
  %result = "tt.reduce"(%values) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %a64 = arith.extsi %a : i32 to i64
    %b64 = arith.extsi %b : i32 to i64
    %wide = arith.addi %a64, %b64 : i64
    %max = arith.constant 2147483647 : i64
    %min = arith.constant -2147483648 : i64
    %upper = arith.cmpi sle, %wide, %max : i64
    %lower = arith.cmpi sge, %wide, %min : i64
    %ok = arith.andi %upper, %lower : i1
    tt.assert %ok, "overflow" : i1
    %sum = arith.addi %a, %b : i32
    tt.reduce.return %sum : i32
  }) : (tensor<3x4xi32>) -> tensor<3xi32>
  tt.store %p, %result : tensor<3x!tt.ptr<i32>>
  tt.return
}

// -----

// Forward scans also need neutral padding along a parallel dimension when
// their scan axis is already a power of two.
// CHECK-LABEL: tt.func @scanWithPaddedParallelAxis
// CHECK: %[[VALUES:.*]] = tt.load
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : tensor<4x4xi32>
// CHECK: %[[PADDED:.*]] = arith.select %{{.*}}, %[[VALUES]], %[[ZERO]]
// CHECK: "tt.scan"(%[[PADDED]])
// CHECK: tt.assert %{{.*}}, "overflow" : i1
tt.func @scanWithPaddedParallelAxis(%ptr: !tt.ptr<i32>) {
  %r = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<3x!tt.ptr<i32>>
  %p = tt.addptr %base, %r : tensor<3x!tt.ptr<i32>>, tensor<3xi32>
  %col = tt.expand_dims %p {axis = 1 : i32} : tensor<3x!tt.ptr<i32>> -> tensor<3x1x!tt.ptr<i32>>
  %ptrs = tt.broadcast %col : tensor<3x1x!tt.ptr<i32>> -> tensor<3x4x!tt.ptr<i32>>
  %values = tt.load %ptrs : tensor<3x4x!tt.ptr<i32>>
  %result = "tt.scan"(%values) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%a: i32, %b: i32):
    %a64 = arith.extsi %a : i32 to i64
    %b64 = arith.extsi %b : i32 to i64
    %wide = arith.addi %a64, %b64 : i64
    %max = arith.constant 2147483647 : i64
    %min = arith.constant -2147483648 : i64
    %upper = arith.cmpi sle, %wide, %max : i64
    %lower = arith.cmpi sge, %wide, %min : i64
    %ok = arith.andi %upper, %lower : i1
    tt.assert %ok, "overflow" : i1
    %sum = arith.addi %a, %b : i32
    tt.scan.return %sum : i32
  }) : (tensor<3x4xi32>) -> tensor<3x4xi32>
  tt.store %ptrs, %result : tensor<3x4x!tt.ptr<i32>>
  tt.return
}
