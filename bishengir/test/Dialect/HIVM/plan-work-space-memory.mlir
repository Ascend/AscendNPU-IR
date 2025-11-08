// RUN: bishengir-opt %s -hivm-plan-memory="mem-plan-mode=global-work-space-plan" -split-input-file -verify-diagnostics | FileCheck %s

// -----
module {
  func.func @test_mem_global_workspace_plan(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                                            %arg1: memref<128x16xf32> ,
                                            %arg2: memref<16x128xf32> ,
                                            %arg7: tensor<128x128xf32>)  -> tensor<128x128xf32>{
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16 = arith.constant 16 : index
    %c65536 = arith.constant 65536 : index
    %76 = bufferization.to_tensor %arg1 restrict writable : memref<128x16xf32>
    %77 = bufferization.to_tensor %arg2 restrict writable : memref<16x128xf32>

    %21 = tensor.empty() : tensor<128x16xf32>
    %22 = tensor.empty() : tensor<16x128xf32>
    %3 = hivm.hir.load ins(%76 : tensor<128x16xf32>) outs(%21 : tensor<128x16xf32>) -> tensor<128x16xf32>
    %5 = hivm.hir.load ins(%77 : tensor<16x128xf32>) outs(%22 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %6 = tensor.empty() : tensor<128x128xf32>
    %7 = hivm.hir.mmadL1 {b_transpose} ins(%3, %5, %true, %c128, %c16, %c128 :
                          tensor<128x16xf32>, tensor<16x128xf32>, i1, index, index, index)
                          outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    // CHECK: memref_ext.alloc_workspace() from %arg0 offset = [%c0] : from memref<?xi8> to memref<65536xi8>
    %reinterpret_cast_3 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<65536xi8>
    %view_3 = memref.view %reinterpret_cast_3[%c0][] : memref<65536xi8> to memref<128x128xf32>
    %8 = bufferization.to_tensor %view_3 restrict writable : memref<128x128xf32>
    %9 = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<NO_QUANT>,
                           pre_relu = #hivm.fixpipe_pre_relu_mode<NO_RELU>}
                           ins(%7 : tensor<128x128xf32>) outs(%8 : tensor<128x128xf32>) -> tensor<128x128xf32>

    %44:1 = scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg21 = %9) -> (tensor<128x128xf32>)  : i32 {
    // CHECK: memref_ext.alloc_workspace() from %arg0 offset = [%c0] : from memref<?xi8> to memref<65536xi8>
    %10 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<65536xi8>
    %view_4 = memref.view %10[%c0][] : memref<65536xi8> to memref<128x128xf32>
    %11 = bufferization.to_tensor %view_4 restrict writable : memref<128x128xf32>
    %23 = tensor.empty() : tensor<128x128xf32>
    %13 = hivm.hir.store ins(%23 : tensor<128x128xf32>) outs(%11 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.yield %13 : tensor<128x128xf32>
    }
    %24 = tensor.empty() : tensor<128x128xf32>
    %25 = hivm.hir.load ins(%44#0 : tensor<128x128xf32>) outs(%24 : tensor<128x128xf32>) -> tensor<128x128xf32>

    return %25 : tensor<128x128xf32>
  }
}

// -----
module {
  func.func @test_mem_global_workspace_without_load(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
                                                    %arg1: memref<1xi32>) {
    // expected-error@+1 {{'memref_ext.alloc_workspace' op error: read before first write}}
    %0 = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<1xi32>
    hivm.hir.store ins(%0 : memref<1xi32>) outs(%arg1 : memref<1xi32>)
    return
  }
}
