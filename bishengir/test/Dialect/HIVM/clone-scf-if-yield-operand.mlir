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

// -----

func.func @test_clone_if_yield_operands_defined_out_of_forOp(%arg0: i32, %arg1: tensor<256xf16>, %arg2: memref<256xf16>) {
  %cst = arith.constant 0.000000e+00 : f16
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<256xf16>
  // CHECK: %[[ARG_1:.*]] = hivm.hir.vbrc ins({{.*}} : f16) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
  %1 = hivm.hir.vbrc ins(%cst : f16) outs(%0 : tensor<256xf16>) -> tensor<256xf16>
  %2 = scf.for %arg3 = %c0_i32 to %c0_i32 step %c1_i32 iter_args(%arg4 = %c0_i32) -> (i32)  : i32 {
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.addi %3, %c1 : index
    %5 = arith.cmpi slt, %3, %c0 : index
    %6 = arith.cmpi sge, %3, %c4 : index
    %7 = arith.ori %5, %6 : i1
    %8 = scf.if %7 -> (tensor<256xf16>) {
      // CHECK: %[[ARG_2:.*]] = hivm.hir.copy ins(%[[ARG_1]] : tensor<256xf16>) outs({{.*}} : tensor<256xf16>) -> tensor<256xf16>
      // CHECK: scf.yield %[[ARG_2]] : tensor<256xf16>
      scf.yield %1 : tensor<256xf16>
    } else {
      %extracted_slice = tensor.extract_slice %arg1[0] [%4] [1] : tensor<256xf16> to tensor<?xf16>
      %inserted_slice = tensor.insert_slice %extracted_slice into %1[0] [%4] [1] : tensor<?xf16> into tensor<256xf16>
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
   // CHECK: %[[ARG_1:.*]] = bufferization.to_tensor {{.*}} restrict writable : memref<256xf16>
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 1
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