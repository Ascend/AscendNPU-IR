// RUN: bishengir-opt %s -hivm-clone-scf-if-yield-operand -split-input-file | FileCheck %s

// -----
module {
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
}

// -----
module {
  func.func @test_clone_use_after_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                        %arg2 : tensor<256xf16>) -> tensor<256xf16> {
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<256xf16>
    // CHECK: %[[ARG_1:.*]] = hivm.hir.vbrc ins({{.*}} : f16) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
    %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
    %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
    %3 = scf.if %2 -> tensor<256xf16> {
      %6 = tensor.empty() : tensor<256xf16>
      %7 = hivm.hir.copy ins(%arg1 : tensor<256xf16>) outs(%6 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %7 : tensor<256xf16>
    } else {
      // CHECK: %[[ARG_2:.*]] = hivm.hir.copy ins(%[[ARG_1]] : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
      // CHECK: scf.yield %[[ARG_2]] : tensor<256xf16>
      scf.yield %1 : tensor<256xf16>
    }
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%1, %3 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    return %5 : tensor<256xf16>
  }
}

// -----
module {
  func.func @test_clone_yield_operands_alias_by_for_operands(%arg0: i32, %arg1 : tensor<256xf16>,
                                                                %arg2 : tensor<256xf16>) -> (tensor<256xf16>, tensor<256xf16>, tensor<256xf16>) {
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<256xf16>
    %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
    %2 = tensor.empty() : tensor<256xf16>
    %3 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%2 : tensor<256xf16>) -> tensor<256xf16>
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    %6:3 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1, %arg5 = %3, %arg6 = %5) -> (tensor<256xf16>, tensor<256xf16>, tensor<256xf16>) {
      %7 = arith.cmpi eq, %arg3, %c1 : index
      // CHECK: %[[ARG_0:.*]]:2 = scf.if {{.*}} -> (tensor<256xf16>, tensor<256xf16>) {
      %8:2 = scf.if %7 -> (tensor<256xf16>, tensor<256xf16>) {
        %11 = tensor.empty() : tensor<256xf16>
        %12 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%11 : tensor<256xf16>) -> tensor<256xf16>
        %13 = tensor.empty() : tensor<256xf16>
        %14 = hivm.hir.vadd ins(%arg1, %12 : tensor<256xf16>, tensor<256xf16>) outs(%13 : tensor<256xf16>) -> tensor<256xf16>
        scf.yield %12, %14: tensor<256xf16>, tensor<256xf16>
      } else {
        // CHECK: %[[ARG_1:.*]] = hivm.hir.copy ins(%arg4 : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
        // CHECK: scf.yield %[[ARG_1]], %arg5 : tensor<256xf16>, tensor<256xf16>
        scf.yield %arg4, %arg5 : tensor<256xf16>, tensor<256xf16>
      }
      %9 = tensor.empty() : tensor<256xf16>
      %10 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%9 : tensor<256xf16>) -> tensor<256xf16>
      // CHECK: %[[ARG_2:.*]] = hivm.hir.copy ins(%[[ARG_0]]#0 : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
      // CHECK: scf.yield %[[ARG_2]], %[[ARG_0]]#1, {{.*}} : tensor<256xf16>, tensor<256xf16>
      scf.yield %8#0, %8#1, %10 : tensor<256xf16>, tensor<256xf16>, tensor<256xf16>
    }
    return %6#0, %6#1, %6#2 : tensor<256xf16>, tensor<256xf16>, tensor<256xf16>
  }
}
 