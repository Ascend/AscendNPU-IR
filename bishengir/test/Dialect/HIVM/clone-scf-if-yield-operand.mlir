// RUN: bishengir-opt %s -hivm-clone-scf-if-yield-operand -split-input-file | FileCheck %s

// -----

func.func @test_clone_same_yield_operands(%arg0: i32, 
                                          %arg1 : tensor<256xf16>,
                                          %arg2 : tensor<256xf16>,
                                          %arg3 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %1:3 = scf.if %0 -> (tensor<256xf16>, tensor<256xf16>, tensor<256xf16>) {
    %2 = tensor.empty() : tensor<256xf16>
    %3 = tensor.empty() : tensor<256xf16>
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.copy ins(%arg1 : tensor<256xf16>) outs(%2 : tensor<256xf16>) -> tensor<256xf16>
    %6 = hivm.hir.copy ins(%arg2 : tensor<256xf16>) outs(%3 : tensor<256xf16>) -> tensor<256xf16>
    %7 = hivm.hir.copy ins(%arg3 : tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %5, %6, %7 : tensor<256xf16>, tensor<256xf16>, tensor<256xf16>
  } else {
    %2 = tensor.empty() : tensor<256xf16>
    // CHECK: } else {
    // CHECK: %[[ARG_1:.*]] = hivm.hir.copy ins(%arg1 : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
    %3 = hivm.hir.copy ins(%arg1 : tensor<256xf16>) outs(%2 : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: %[[ARG_2:.*]] = hivm.hir.copy ins(%[[ARG_1]] : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: %[[ARG_3:.*]] = hivm.hir.copy ins(%[[ARG_1]] : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: scf.yield %[[ARG_1]], %[[ARG_2]], %[[ARG_3]] : tensor<256xf16>, tensor<256xf16>, tensor<256xf16>
    scf.yield %3, %3, %3 : tensor<256xf16>, tensor<256xf16>, tensor<256xf16>
  }
  return %1#0 : tensor<256xf16>
}

// -----

func.func @test_clone_use_after_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: %[[ARG_1:.*]] = hivm.hir.vbrc ins({{.*}} : f16) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: %[[ARG_2:.*]] = hivm.hir.copy ins(%[[ARG_1]] : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: scf.yield %[[ARG_2]] : tensor<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  hfusion.print " %1 " {hex = false} %1 : tensor<256xf16>
  return %3 : tensor<256xf16>
}

// -----

func.func @test_clone_use_after_write_in_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: %[[ARG_1:.*]] = hivm.hir.vbrc ins({{.*}} : f16) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    hfusion.print " %1 " {hex = false} %1 : tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: %[[ARG_2:.*]] = hivm.hir.copy ins(%[[ARG_1]] : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: scf.yield %[[ARG_2]] : tensor<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %3 : tensor<256xf16>
}

// -----

func.func @test_not_clone_in_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: %[[ARG_1:.*]] = hivm.hir.vbrc ins({{.*}} : f16) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK-DAG: scf.yield %[[ARG_1]] : tensor<256xf16>
    hfusion.print " %1 " {hex = false} %1 : tensor<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %3 : tensor<256xf16>
}

// -----

func.func @test_not_clone_before_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: %[[ARG_1:.*]] = hivm.hir.vbrc ins({{.*}} : f16) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  hfusion.print " %1 " {hex = false} %1 : tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK-DAG: scf.yield %[[ARG_1]] : tensor<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %3 : tensor<256xf16>
}

// -----

func.func @test_clone_double_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: %[[ARG_1:.*]] = hivm.hir.vbrc ins({{.*}} : f16) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %5 = tensor.empty() : tensor<256xf16>
    %6 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    %5 = tensor.empty() : tensor<256xf16>
    %6 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
    hfusion.print " %1 " {hex = false} %1 : tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  }
  %4 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %3 : tensor<256xf16>
  } else {
    // CHECK: %[[ARG_2:.*]] = hivm.hir.copy ins(%[[ARG_1]] : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: scf.yield %[[ARG_2]] : tensor<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %4 : tensor<256xf16>
}
// -----

func.func @test_clone_yield_operands_alias_by_for_operands(%arg0: i32, %arg1 : tensor<256xf16>,
                                                              %arg2 : tensor<256xf16>) -> (tensor<256xf16>) {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %6 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<256xf16>) {
    %7 = arith.cmpi eq, %arg3, %c1 : index
    // CHECK: %[[ARG_0:.*]] = scf.if {{.*}} -> (tensor<256xf16>) {
    %8 = scf.if %7 -> tensor<256xf16> {
      %11 = tensor.empty() : tensor<256xf16>
      %12 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%11 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %12 : tensor<256xf16>
    } else {
      // CHECK: %[[ARG_1:.*]] = hivm.hir.copy ins(%arg4 : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
      // CHECK: scf.yield %[[ARG_1]] : tensor<256xf16>
      scf.yield %arg4 : tensor<256xf16>
    }
    hfusion.print " %arg4 " {hex = false} %arg4 : tensor<256xf16>
    // CHECK: %[[ARG_2:.*]] = hivm.hir.copy ins(%[[ARG_0]] : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: scf.yield %[[ARG_2]] : tensor<256xf16>
    scf.yield %8 : tensor<256xf16>
  }
  return %6 : tensor<256xf16>
}
 