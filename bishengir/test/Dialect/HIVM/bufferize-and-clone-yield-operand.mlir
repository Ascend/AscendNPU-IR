// RUN: bishengir-opt %s -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries analysis-heuristic=top-down function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -split-input-file | FileCheck %s

// -----

func.func @test_clone_same_yield_operands(%arg0: i32, 
                                          %arg1 : tensor<256xf16>,
                                          %arg2 : tensor<256xf16>,
                                          %arg3 : tensor<256xf16>) -> (tensor<256xf16>, tensor<256xf16>, tensor<256xf16>) {
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
    // CHECK: hivm.hir.copy ins(%arg1 : memref<256xf16>) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
    %3 = hivm.hir.copy ins(%arg1 : tensor<256xf16>) outs(%2 : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_2:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_1]], %[[ALLOC_2]], %[[ALLOC_0]] : memref<256xf16>, memref<256xf16>, memref<256xf16>
    scf.yield %3, %3, %3 : tensor<256xf16>, tensor<256xf16>, tensor<256xf16>
  }
  return %1#0, %1#1, %1#2 : tensor<256xf16>, tensor<256xf16>, tensor<256xf16>
}

// -----

func.func @test_clone_use_after_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
  return %3 : tensor<256xf16>
}

// -----

func.func @test_clone_use_after_write_in_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
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
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: scf.yield %[[ALLOC_0]] : memref<256xf16>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
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
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: scf.yield %[[ALLOC_0]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %3 : tensor<256xf16>
}

// -----

func.func @test_clone_double_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %6 = tensor.empty() : tensor<256xf16>
    %7 = hivm.hir.vadd ins(%arg1, %arg1 : tensor<256xf16>, tensor<256xf16>) outs(%6 : tensor<256xf16>) -> tensor<256xf16>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
    scf.yield %7 : tensor<256xf16>
  } else {
    // CHECK: } else {
    %6 = tensor.empty() : tensor<256xf16>
    %7 = hivm.hir.vadd ins(%arg2, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%6 : tensor<256xf16>) -> tensor<256xf16>
    scf.yield %7 : tensor<256xf16>
  }
  %4 = arith.cmpi eq, %arg0, %c2_i32 : i32
  %5 = scf.if %4 -> tensor<256xf16> {
    scf.yield %3 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %5 : tensor<256xf16>
}

// -----

func.func @test_clone_double_SCFIf_v1(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %6 = tensor.empty() : tensor<256xf16>
    %7 = hivm.hir.vadd ins(%arg1, %arg1 : tensor<256xf16>, tensor<256xf16>) outs(%6 : tensor<256xf16>) -> tensor<256xf16>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
    scf.yield %7 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  %4 = arith.cmpi eq, %arg0, %c2_i32 : i32
  %5 = scf.if %4 -> tensor<256xf16> {
    scf.yield %3 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_2:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_2]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %5 : tensor<256xf16>
}

// -----

func.func @test_clone_double_SCFIf_v2(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %5 = scf.if %4 -> tensor<256xf16> {
      %6 = tensor.empty() : tensor<256xf16>
      %7 = hivm.hir.vadd ins(%arg1, %arg1 : tensor<256xf16>, tensor<256xf16>) outs(%6 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %7 : tensor<256xf16>
    } else {
      // CHECK: } else {
      // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
      // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
      scf.yield %1 : tensor<256xf16>
    }
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_2:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_2]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
  return %3 : tensor<256xf16>
}

// -----

func.func @test_not_clone_double_SCFIf(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = arith.cmpi eq, %arg0, %c2_i32 : i32
    %5 = scf.if %4 -> tensor<256xf16> {
      %6 = tensor.empty() : tensor<256xf16>
      %7 = hivm.hir.vadd ins(%arg1, %arg1 : tensor<256xf16>, tensor<256xf16>) outs(%6 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %7 : tensor<256xf16>
    } else {
      // CHECK: } else {
      // CHECK: scf.yield %[[ALLOC_0]] : memref<256xf16>
      scf.yield %1 : tensor<256xf16>
    }
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: scf.yield %[[ALLOC_0]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %3 : tensor<256xf16>
}

// -----

func.func @test_clone_yield_operands_in_for(%arg0: i32, %arg1 : tensor<256xf16>,
                                                              %arg2 : tensor<256xf16>) -> (tensor<256xf16>) {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<256xf16>) {
    %3 = tensor.empty() : tensor<256xf16>
    // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<256xf16>, memref<256xf16>) outs(%[[ALLOC:.*]] : memref<256xf16>)
    %4 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%3 : tensor<256xf16>) -> tensor<256xf16>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg4 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg4 : tensor<256xf16>
    // CHECK: memref.copy %[[ALLOC]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
    scf.yield %4 : tensor<256xf16>
  }
  return %2 : tensor<256xf16>
}// -----

func.func @test_not_clone_yield_operands_in_for(%arg0: i32, %arg1 : tensor<256xf16>,
                                                              %arg2 : tensor<256xf16>) -> (tensor<256xf16>) {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  // CHECK: %{{.*}} = scf.for {{.*}} iter_args(%[[ARG_0:.*]] = %{{.*}}) -> (memref<256xf16>) {
  %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<256xf16>) {
    %3 = tensor.empty() : tensor<256xf16>
    // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<256xf16>, memref<256xf16>) outs(%[[ALLOC:.*]] : memref<256xf16>)
    %4 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%3 : tensor<256xf16>) -> tensor<256xf16>
    // CHECK: scf.yield %[[ALLOC]] : memref<256xf16>
    scf.yield %4 : tensor<256xf16>
  }
  return %2 : tensor<256xf16>
}// -----

func.func @test_clone_yield_operands_in_for_and_if(%arg0: i32, %arg1 : tensor<256xf16>,
                                                              %arg2 : tensor<256xf16>) -> (tensor<256xf16>) {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  // CHECK: %{{.*}} = scf.for {{.*}} iter_args(%[[ARG_0:.*]] = %{{.*}}) -> (memref<256xf16>) {
  %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<256xf16>) {
    %3 = arith.cmpi eq, %arg3, %c1 : index
    // CHECK: %[[SCF_IF_RESULT:.*]] = scf.if {{.*}} -> (memref<256xf16>) {
    %4 = scf.if %3 -> tensor<256xf16> {
      %5 = tensor.empty() : tensor<256xf16>
      %6 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
      hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg4 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg4 : tensor<256xf16>
      scf.yield %6 : tensor<256xf16>
    } else {
      // CHECK: } else {
      // CHECK: memref.copy %[[ARG_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
      // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
      scf.yield %arg4 : tensor<256xf16>
    }
    // CHECK: memref.copy %[[SCF_IF_RESULT]], %[[ALLOC_2:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_2]] : memref<256xf16>
    scf.yield %4 : tensor<256xf16>
  }
  return %2 : tensor<256xf16>
}

// -----

func.func @test_not_clone_yield_operands_in_for_and_if(%arg0: i32, %arg1 : tensor<256xf16>,
                                                              %arg2 : tensor<256xf16>) -> (tensor<256xf16>) {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  // CHECK: %{{.*}} = scf.for {{.*}} iter_args(%[[ARG_0:.*]] = %{{.*}}) -> (memref<256xf16>) {
  %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<256xf16>) {
    %3 = arith.cmpi eq, %arg3, %c1 : index
    // CHECK: %[[SCF_IF_RESULT:.*]] = scf.if {{.*}} -> (memref<256xf16>) {
    %4 = scf.if %3 -> tensor<256xf16> {
      %5 = tensor.empty() : tensor<256xf16>
      %6 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %6 : tensor<256xf16>
    } else {
      // CHECK: } else {
      // CHECK: scf.yield %[[ARG_0]] : memref<256xf16>
      hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg4 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg4 : tensor<256xf16>
      scf.yield %arg4 : tensor<256xf16>
    }
    // CHECK: scf.yield %[[SCF_IF_RESULT]] : memref<256xf16>
    scf.yield %4 : tensor<256xf16>
  }
  return %2 : tensor<256xf16>
}

// -----

func.func @test_clone_yield_operands_in_for_out_if(%arg0: i32, %arg1 : tensor<256xf16>,
                                                              %arg2 : tensor<256xf16>) -> (tensor<256xf16>) {
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst_0 : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  // CHECK: %{{.*}} = scf.for {{.*}} iter_args(%[[ARG_0:.*]] = %{{.*}}) -> (memref<256xf16>) {
  %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<256xf16>) {
    %3 = arith.cmpi eq, %arg3, %c1 : index
    // CHECK: %[[SCF_IF_RESULT:.*]] = scf.if {{.*}} -> (memref<256xf16>) {
    %4 = scf.if %3 -> tensor<256xf16> {
      %5 = tensor.empty() : tensor<256xf16>
      %6 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %6 : tensor<256xf16>
    } else {
      // CHECK: } else {
      // CHECK: memref.copy %[[ARG_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
      // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
      scf.yield %arg4 : tensor<256xf16>
    }
    // CHECK: memref.copy %[[SCF_IF_RESULT]], %[[ALLOC_2:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_2]] : memref<256xf16>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg4 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg4 : tensor<256xf16>
    scf.yield %4 : tensor<256xf16>
  }
  return %2 : tensor<256xf16>
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
  %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<256xf16>) {
    %3 = arith.cmpi eq, %arg3, %c1 : index
    // CHECK: %[[SCF_IF_RESULT:.*]] = scf.if {{.*}}
    %4 = scf.if %3 -> tensor<256xf16> {
      %5 = tensor.empty() : tensor<256xf16>
      %6 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
      scf.yield %6 : tensor<256xf16>
    } else {
      %5 = tensor.empty() : tensor<256xf16>
      // CHECK: } else {
      // CHECK: scf.yield
      %6 = hivm.hir.vadd ins(%arg4, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%5 : tensor<256xf16>) -> tensor<256xf16>
      hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg4 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg4 : tensor<256xf16>
      scf.yield %6 : tensor<256xf16>
    }
    // CHECK: memref.copy %[[SCF_IF_RESULT]], %[[ALLOC_1:.*]] : memref
    // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg4 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg4 : tensor<256xf16>
    scf.yield %4 : tensor<256xf16>
  }
  return %2 : tensor<256xf16>
}

// -----

func.func @test_clone_if_yield_operands_defined_out_of_forOp(%arg0: i32, %arg1: tensor<256xf16>, %arg2: memref<256xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f16) outs(%[[ALLOC_0:.*]] : memref<256xf16>)
  %1 = hivm.hir.vbrc ins(%cst : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %c0_i32) -> (i32)  : i32 {
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.addi %3, %c1 : index
    %5 = arith.cmpi slt, %3, %c0 : index
    %6 = arith.cmpi sge, %3, %c4 : index
    %7 = arith.ori %5, %6 : i1
    %8 = scf.if %7 -> (tensor<256xf16>) {
      // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_1:.*]] : memref<256xf16> to memref<256xf16>
      // CHECK: scf.yield %[[ALLOC_1]] : memref<256xf16>
      scf.yield %1 : tensor<256xf16>
    } else {
      %extracted_slice = tensor.extract_slice %arg1[0] [%4] [1] : tensor<256xf16> to tensor<?xf16>
      %inserted_slice = tensor.insert_slice %extracted_slice into %1[0] [%4] [1] : tensor<?xf16> into tensor<256xf16>
      // CHECK: } else {
      scf.yield %inserted_slice : tensor<256xf16>
    }
    bufferization.materialize_in_destination %8 in writable %arg2 : (tensor<256xf16>, memref<256xf16>) -> ()
    %9 = arith.addi %arg4, %c1_i32 : i32
    scf.yield %9 : i32
  }
  return
}

// -----

func.func @test_clone_use_after_write_in_SCFIf_for_buffer(%arg0: i32, %arg1 : tensor<256xf16>,
                                      %arg2 : tensor<256xf16>) -> tensor<256xf16> {
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant 0.000000e+00 : f16
  %alloc = memref.alloc() : memref<256xf16, #hivm.address_space<ub>>
  annotation.mark %alloc {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<256xf16, #hivm.address_space<ub>>
  %0 = memref.memory_space_cast %alloc : memref<256xf16, #hivm.address_space<ub>> to memref<256xf16>
  %1 = bufferization.to_tensor %0 restrict writable : memref<256xf16>
  // CHECK: %[[CASTED_ALLOC:.*]] = memref.memory_space_cast {{.*}} : memref<256xf16, #hivm.address_space<ub>> to memref<256xf16>
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
  %2 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %3 = scf.if %2 -> tensor<256xf16> {
    %4 = tensor.empty() : tensor<256xf16>
    %5 = hivm.hir.vadd ins(%arg1, %arg2 : tensor<256xf16>, tensor<256xf16>) outs(%4 : tensor<256xf16>) -> tensor<256xf16>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %1 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %1 : tensor<256xf16>
    scf.yield %5 : tensor<256xf16>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[CASTED_ALLOC]], %[[ALLOC_0:.*]] : memref<256xf16> to memref<256xf16>
    // CHECK: scf.yield %[[ALLOC_0]] : memref<256xf16>
    scf.yield %1 : tensor<256xf16>
  }
  return %3 : tensor<256xf16>
}

// -----
	 
func.func @test_clone_trace_insertSliceOp(%arg0: i32, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC_0:.*]] : memref<16x16xf32>)
  %1 = hivm.hir.vbrc ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %2 = tensor.empty() : tensor<16x16xf32>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC_1:.*]] : memref<16x16xf32>)
  %3 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %4 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %5 = scf.if %4 -> (tensor<16x16xf32>) {
    %extracted_slice = tensor.extract_slice %1[0, 0] [16, %c1] [1, 1] : tensor<16x16xf32> to tensor<16x?xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice into %3[0, 0] [16, %c1] [1, 1] : tensor<16x?xf32> into tensor<16x16xf32>
    scf.yield %inserted_slice : tensor<16x16xf32>
  } else {
    // CHECK: } else {
    // CHECK: memref.copy %[[ALLOC_0]], %[[ALLOC_2:.*]] : memref<16x16xf32> to memref<16x16xf32>
    // CHECK: scf.yield %[[ALLOC_2]] : memref<16x16xf32>
    scf.yield %1 : tensor<16x16xf32>
  }
  return %5 : tensor<16x16xf32>
}

// -----

func.func @test_not_clone_trace_insertSliceOp(%arg0: i32, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC_0:.*]] : memref<16x16xf32>)
  %1 = hivm.hir.vbrc ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %2 = tensor.empty() : tensor<16x16xf32>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC_1:.*]] : memref<16x16xf32>)
  %3 = hivm.hir.vbrc ins(%cst_0 : f32) outs(%2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %4 = arith.cmpi eq, %arg0, %c1_i32 : i32
  %5 = scf.if %4 -> (tensor<16x16xf32>) {
    %extracted_slice = tensor.extract_slice %3[0, 0] [16, %c1] [1, 1] : tensor<16x16xf32> to tensor<16x?xf32>
    %inserted_slice = tensor.insert_slice %extracted_slice into %1[0, 0] [16, %c1] [1, 1] : tensor<16x?xf32> into tensor<16x16xf32>
    scf.yield %inserted_slice : tensor<16x16xf32>
  } else {
    // CHECK: } else {
    // CHECK: scf.yield %[[ALLOC_0]] : memref<16x16xf32>
    scf.yield %1 : tensor<16x16xf32>
  }
  return %5 : tensor<16x16xf32>
}

// -----

module {

  func.func @init_vf(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %mask = vector.constant_mask [16] : vector<64xi1>
    %ret = vector.transfer_write %cst, %arg0[%c0], %mask {in_bounds = [true]} : vector<64xf32>, tensor<16xf32>
    return %ret : tensor<16xf32>
  }

  func.func @test_vf(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    %extracted_slice = tensor.extract_slice %arg0[%c0] [16] [1] : tensor<16xf32> to tensor<16xf32>
    %1 = vector.transfer_write %cst, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<16xf32>
    %2 = vector.transfer_read %extracted_slice[%c0], %cst_1, %0 {in_bounds = [true]} : tensor<16xf32>, vector<64xf32>
    %3 = vector.transfer_write %2, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<16xf32>
    return %3 : tensor<16xf32>
  }

  func.func @test_conflict_in_vf(%arg0: tensor<16xf32>, %arg1: memref<16xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    
    %0 = tensor.empty() : tensor<16xf32>
    %1 = func.call @init_vf(%0) {hivm.vector_function, no_inline} : (tensor<16xf32>) -> tensor<16xf32>
    %2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %1) -> (tensor<16xf32>) : i32 {
      %3 = tensor.empty() : tensor<16xf32>
      // CHECK: %[[CALL_RESULT:.*]] = func.call @test_vf
      %4 = func.call @test_vf(%arg3, %3) {hivm.vector_function, no_inline} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
      // CHECK: memref.copy %[[CALL_RESULT]], %[[ALLOC:.*]] : memref<16xf32> to memref<16xf32>
      // CHECK: scf.yield %[[ALLOC]] : memref<16xf32>
      scf.yield %4 : tensor<16xf32>
    }
    return
  }
}

// -----

module {
  func.func @init_vf(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %0 = vector.constant_mask [16] : vector<64xi1>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %1 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %arg0) -> (tensor<16x16xf32>) {
      %extracted_slice_0 = tensor.extract_slice %arg3[%arg2, %c0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<16xf32>
      %3 = vector.transfer_write %cst, %extracted_slice_0[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<16xf32>
      %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2, %c0] [1, 16] [1, 1] : tensor<16xf32> into tensor<16x16xf32>
      scf.yield %inserted_slice : tensor<16x16xf32>
    }
    return %1 : tensor<16x16xf32>
  }

  func.func @test_vf(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    %1 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %arg1) -> (tensor<16x16xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg2, %c0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<16xf32>
      %extracted_slice_0 = tensor.extract_slice %arg3[%arg2, %c0] [1, 16] [1, 1] : tensor<16x16xf32> to tensor<16xf32>
      %2 = vector.transfer_read %extracted_slice[%c0], %cst_1, %0 {in_bounds = [true]} : tensor<16xf32>, vector<64xf32>
      %3 = vector.transfer_write %2, %extracted_slice_0[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<16xf32>
      %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2, %c0] [1, 16] [1, 1] : tensor<16xf32> into tensor<16x16xf32>
      scf.yield %inserted_slice : tensor<16x16xf32>
    }
    return %1 : tensor<16x16xf32>
  }

  func.func @test_not_conflict_in_vf(%arg0: tensor<16x16xf32>, %arg1: memref<16x16xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = func.call @init_vf(%0) {hivm.vector_function, no_inline} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %1) -> (tensor<16x16xf32>) : i32 {
      %3 = tensor.empty() : tensor<16x16xf32>
      // CHECK: %[[CALL_RESULT:.*]] = func.call @test_vf
      %4 = func.call @test_vf(%arg3, %3) {hivm.vector_function, no_inline} : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
      // CHECK: scf.yield %[[CALL_RESULT]] : memref<16x16xf32>
      scf.yield %4 : tensor<16x16xf32>
    }
    return
  }
}

// -----

module {

  func.func @init_vf(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %mask = vector.constant_mask [16] : vector<64xi1>
    %ret = vector.transfer_write %cst, %arg0[%c0], %mask {in_bounds = [true]} : vector<64xf32>, tensor<16xf32>
    return %ret : tensor<16xf32>
  }

  func.func @test_vf(%arg0: tensor<16xf32>) -> tensor<16xf32> attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [16] : vector<64xi1>
    %1 = vector.transfer_read %arg0[%c0], %cst_1, %0 {in_bounds = [true]} : tensor<16xf32>, vector<64xf32>
    %2 = arith.addf %1, %cst : vector<64xf32>
    %3 = vector.transfer_write %2, %arg0[%c0], %0 {in_bounds = [true]} : vector<64xf32>, tensor<16xf32>
    return %3 : tensor<16xf32>
  }

  func.func @test_conflict_in_vf(%arg0: tensor<16xf32>, %arg1: memref<16xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tensor.empty() : tensor<16xf32>
    // CHECK: %[[CALL_RESULT:.*]] = call @init_vf
    %1 = func.call @init_vf(%0) {hivm.vector_function, no_inline} : (tensor<16xf32>) -> tensor<16xf32>
    scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 : i32 {
      // CHECK: memref.copy %[[CALL_RESULT]], %[[ALLOC:.*]] : memref<16xf32> to memref<16xf32>
      %2 = func.call @test_vf(%1) {hivm.vector_function, no_inline} : (tensor<16xf32>) -> tensor<16xf32>
      hivm.hir.store ins(%2 : tensor<16xf32>) outs(%arg1 : memref<16xf32>)
    }
    return
  }
}

// -----

func.func @test_not_copy_in_scf_whileOp(%arg0: tensor<16xf32>,
                                        %arg1: tensor<16xf32>,
                                        %arg2: i32) -> tensor<16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c100_i32 = arith.constant 100 : i32
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hivm.hir.load ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: %{{.*}} = scf.while (%[[BEFORE_ARG:.*]] = %{{.*}}) : (memref<16xf32>) -> memref<16xf32> {
  %2 = scf.while (%arg3 = %1) : (tensor<16xf32>) -> tensor<16xf32> {
    %3 = arith.cmpi eq, %arg2, %c100_i32 : i32
    // CHECK: scf.condition(%{{.*}}) %[[BEFORE_ARG]] : memref<16xf32>
    scf.condition(%3) %arg3 : tensor<16xf32>
  } do {
  ^bb0(%arg3: tensor<16xf32>):
    %3 = tensor.empty() : tensor<16xf32>
    %4 = hivm.hir.load ins(%arg1 : tensor<16xf32>) outs(%3 : tensor<16xf32>) -> tensor<16xf32>
    %5 = tensor.empty() : tensor<16xf32>
    // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<16xf32>, memref<16xf32>) outs(%[[ALLOC:.*]] : memref<16xf32>)
    %6 = hivm.hir.vadd ins(%arg3, %4 : tensor<16xf32>, tensor<16xf32>) outs(%5 : tensor<16xf32>) -> tensor<16xf32>
    // CHECK: scf.yield %[[ALLOC]] : memref<16xf32>
    scf.yield %6 : tensor<16xf32>
  }
  return %2 : tensor<16xf32>
}

// -----

func.func @test_copy_in_scf_whileOp(%arg0: tensor<16xf32>,
                                    %arg1: tensor<16xf32>,
                                    %arg2: i32) -> tensor<16xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c100_i32 = arith.constant 100 : i32
  %0 = tensor.empty() : tensor<16xf32>
  %1 = hivm.hir.load ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  // CHECK: %{{.*}} = scf.while (%[[BEFORE_ARG:.*]] = %{{.*}}) : (memref<16xf32>) -> memref<16xf32> {
  %2 = scf.while (%arg3 = %1) : (tensor<16xf32>) -> tensor<16xf32> {
    %3 = arith.cmpi eq, %arg2, %c100_i32 : i32
    // CHECK: scf.condition(%{{.*}}) %[[BEFORE_ARG]] : memref<16xf32>
    scf.condition(%3) %arg3 : tensor<16xf32>
  } do {
  ^bb0(%arg3: tensor<16xf32>):
    %3 = tensor.empty() : tensor<16xf32>
    %4 = hivm.hir.load ins(%arg1 : tensor<16xf32>) outs(%3 : tensor<16xf32>) -> tensor<16xf32>
    %5 = tensor.empty() : tensor<16xf32>
    // CHECK: hivm.hir.vadd ins(%{{.*}}, %{{.*}} : memref<16xf32>, memref<16xf32>) outs(%[[ALLOC:.*]] : memref<16xf32>)
    %6 = hivm.hir.vadd ins(%arg3, %4 : tensor<16xf32>, tensor<16xf32>) outs(%5 : tensor<16xf32>) -> tensor<16xf32>
    hivm.hir.debug {debugtype = "print", hex = false, prefix = " %arg3 : ", tcoretype = #hivm.tcore_type<CUBE_OR_VECTOR>} %arg3 : tensor<16xf32>
    // CHECK: memref.copy %[[ALLOC]], %[[ALLOC_1:.*]] : memref<16xf32> to memref<16xf32>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<16xf32>
    scf.yield %6 : tensor<16xf32>
  }
  return %2 : tensor<16xf32>
}