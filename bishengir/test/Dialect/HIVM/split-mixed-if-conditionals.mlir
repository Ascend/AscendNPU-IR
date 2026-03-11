// RUN: bishengir-opt %s -hivm-split-mixed-if-conditionals -split-input-file -verify-diagnostics | FileCheck %s
 
// -----
// CHECK-LABEL: @split_cube_if_vector_else
func.func @split_cube_if_vector_else(%arg0: i1, %arg1: i1, %arg2: index, %arg3: tensor<32x32xf32>, %arg4: tensor<32x32xf32>, %arg5: tensor<32x32xf16>, %arg6: tensor<32x32xf16>, %arg7: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<MIX>} {
    %c32 = arith.constant 32 : index
    // CHECK:        %[[c32:.*]] = arith.constant 32 : index
    // CHECK:        %[[VAL_0:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_1:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          %[[VAL_7:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins({{.*}}, {{.*}}, %[[arg1:.*]], %[[c32]], %[[c32]], %[[c32]] : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%[[VAL_0:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_2:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>)  {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_3:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          %[[VAL_7:.*]] = hivm.hir.vadd ins(%[[VAL_2:.*]], %[[VAL_2:.*]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_2:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_4:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %[[VAL_1:.*]] : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %[[VAL_3:.*]] : tensor<32x32xf32>
    // CHECK:        }
    %0 = scf.if %arg0 -> (tensor<32x32xf32>) {
    %3 = tensor.empty() : tensor<32x32xf16>
    %4 = hivm.hir.copy ins(%arg5 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %5 = hivm.hir.copy ins(%arg6 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%4, %5, %arg1, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %7 : tensor<32x32xf32>
  } else {
    %3 = tensor.empty() : tensor<32x32xf32>
    %4 = hivm.hir.copy ins(%arg3 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %5 = hivm.hir.copy ins(%arg4 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = hivm.hir.vadd ins(%4, %5 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %6 : tensor<32x32xf32>
  }
  %1 = tensor.empty() : tensor<32x32xf32>
  %2 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
 
// -----
// CHECK-LABEL: @split_mix_if_cube_else
func.func @split_mix_if_cube_else(%arg0: i1, %arg1: i1, %arg2: index, %arg3: tensor<32x32xf32>, %arg4: tensor<32x32xf32>, %arg5: tensor<32x32xf16>, %arg6: tensor<32x32xf16>, %arg7: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry} {
    %c32 = arith.constant 32 : index
    // CHECK:        %[[c32:.*]] = arith.constant 32 : index
    // CHECK:        %[[VAL_0:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_1:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          %[[VAL_9:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins({{.*}}, {{.*}}, %[[arg1:.*]], %[[c32]], %[[c32]], %[[c32]] : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%[[VAL_0:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_2:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>)  {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_3:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          %[[VAL_9:.*]] = hivm.hir.vadd ins(%[[VAL_1:.*]], %[[VAL_2:.*]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_0:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_4:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_5:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          %[[VAL_9:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins({{.*}}, {{.*}}, %[[arg1:.*]], %[[c32]], %[[c32]], %[[c32]] : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%[[VAL_4:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_6:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %[[VAL_3:.*]] : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %[[VAL_5:.*]] : tensor<32x32xf32>
    // CHECK:        }
    %0 = scf.if %arg0 -> (tensor<32x32xf32>) {
    %3 = tensor.empty() : tensor<32x32xf16>
    %4 = hivm.hir.copy ins(%arg5 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %5 = hivm.hir.copy ins(%arg6 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%4, %5, %arg1, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %alloc = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%7 : tensor<32x32xf32>) outs(%alloc : memref<32x32xf32, #hivm.address_space<ub>>)
    %8 = bufferization.to_tensor %memspacecast restrict writable : memref<32x32xf32>
    %9 = hivm.hir.copy ins(%arg4 : tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %10 = hivm.hir.vadd ins(%8, %9 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %10 : tensor<32x32xf32>
  } else {
    %3 = tensor.empty() : tensor<32x32xf16>
    %4 = hivm.hir.copy ins(%arg5 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %5 = hivm.hir.copy ins(%arg6 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%4, %5, %arg1, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %7 : tensor<32x32xf32>
  }
  %1 = tensor.empty() : tensor<32x32xf32>
  %2 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0 : tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
 
// -----
// CHECK-LABEL: @no_split_empty_if_mix_else
func.func @no_split_empty_if_mix_else(%arg0: i1, %arg1: i1, %arg2: index, %arg3: tensor<32x32xf32>, %arg4: tensor<32x32xf32>, %arg5: tensor<32x32xf16>, %arg6: tensor<32x32xf16>, %arg7: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hacc.entry} {
    %c32 = arith.constant 32 : index
    // CHECK:        %[[c32:.*]] = arith.constant 32 : index
    // CHECK:        %[[VAL_0:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_1:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_2:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          %[[VAL_6:.*]] = hivm.hir.mmadL1 {already_set_real_mkn} ins({{.*}}, {{.*}}, %[[arg1:.*]], %[[c32]], %[[c32]], %[[c32]] : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%[[VAL_1:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_3:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>, tensor<32x32xf32>)  {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_4:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          %[[VAL_6:.*]] = hivm.hir.vadd ins(%[[VAL_2:.*]], %[[VAL_3:.*]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_3:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_5:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %[[VAL_0:.*]] : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %[[VAL_4:.*]] : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        return %[[VAL_5:.*]] : tensor<32x32xf32>
    %0 = scf.if %arg0 -> (tensor<32x32xf32>) {
        %1 = tensor.empty() : tensor<32x32xf32>
        scf.yield %1 : tensor<32x32xf32>
    } else {
        %1 = tensor.empty() : tensor<32x32xf16>
        %2 = hivm.hir.copy ins(%arg5 : tensor<32x32xf16>) outs(%1 : tensor<32x32xf16>) -> tensor<32x32xf16>
        %3 = tensor.empty() : tensor<32x32xf16>
        %4 = hivm.hir.copy ins(%arg6 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
        %5 = tensor.empty() : tensor<32x32xf32>
        %6 = hivm.hir.mmadL1 {already_set_real_mkn} ins(%2, %4, %arg1, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %7 = tensor.empty() : tensor<32x32xf32>
        %8 = hivm.hir.copy ins(%arg4 : tensor<32x32xf32>) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %9 = tensor.empty() : tensor<32x32xf32>
        %10 = hivm.hir.vadd ins(%6, %8 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.yield %10 : tensor<32x32xf32>
    }
    return %0 : tensor<32x32xf32>
}

// -----
// CHECK-LABEL: @split_mix_if_mix_else
func.func @split_mix_if_mix_else(%arg0: i1, %arg1: i1, %arg2: index, %arg3: tensor<32x32xf32>, %arg4: tensor<32x32xf32>, %arg5: tensor<32x32xf16>, %arg6: tensor<32x32xf16>, %arg7: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) attributes {hacc.entry} {
    %c32 = arith.constant 32 : index
    // CHECK:        %[[c32:.*]] = arith.constant 32 : index
    // CHECK:        %[[VAL_0:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_1:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          %[[VAL_12:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins({{.*}}, {{.*}}, %[[arg1:.*]], %[[c32]], %[[c32]], %[[c32]] : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%[[VAL_0:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_2:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_3:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          %[[VAL_12:.*]] = hivm.hir.vadd ins(%[[VAL_1:.*]], %[[VAL_2:.*]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_0:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_4:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf16>, tensor<32x32xf16>, tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_5:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          %[[VAL_12:.*]] = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins({{.*}}, {{.*}}, %[[arg1:.*]], %[[c32]], %[[c32]], %[[c32]] : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%[[VAL_4:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_6:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_7:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          %[[VAL_12:.*]] = hivm.hir.vadd ins(%[[VAL_5:.*]], %[[VAL_6:.*]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_4:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:          scf.yield %{{.*}} : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_8:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %[[VAL_3:.*]] : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %[[VAL_7:.*]] : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_9:.*]] = scf.if %[[arg0:.*]] -> (tensor<32x32xf32>) {
    // CHECK:          scf.yield %[[VAL_1:.*]] : tensor<32x32xf32>
    // CHECK:        } else {
    // CHECK:          scf.yield %[[VAL_5:.*]] : tensor<32x32xf32>
    // CHECK:        }
    // CHECK:        %[[VAL_10:.*]] = tensor.empty() : tensor<32x32xf32>
    // CHECK:        %[[VAL_11:.*]] = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%[[VAL_9:.*]] : tensor<32x32xf32>) outs(%[[VAL_10:.*]] : tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK:        return %[[VAL_8:.*]], %[[VAL_11:.*]] : tensor<32x32xf32>, tensor<32x32xf32>
    %0:2 = scf.if %arg0 -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %3 = tensor.empty() : tensor<32x32xf16>
    %4 = hivm.hir.copy ins(%arg5 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %5 = hivm.hir.copy ins(%arg6 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%4, %5, %arg1, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %8 = hivm.hir.copy ins(%arg4 : tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %9 = hivm.hir.vadd ins(%7, %8 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %9, %7 : tensor<32x32xf32>, tensor<32x32xf32>
  } else {
    %3 = tensor.empty() : tensor<32x32xf16>
    %4 = hivm.hir.copy ins(%arg5 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %5 = hivm.hir.copy ins(%arg6 : tensor<32x32xf16>) outs(%3 : tensor<32x32xf16>) -> tensor<32x32xf16>
    %6 = tensor.empty() : tensor<32x32xf32>
    %7 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_already_inserted = true} ins(%4, %5, %arg1, %c32, %c32, %c32 : tensor<32x32xf16>, tensor<32x32xf16>, i1, index, index, index) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %8 = hivm.hir.copy ins(%arg4 : tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %9 = hivm.hir.vadd ins(%7, %8 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    scf.yield %9, %7 : tensor<32x32xf32>, tensor<32x32xf32>
  }
  %1 = tensor.empty() : tensor<32x32xf32>
  %2 = hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%0#1 : tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0#0, %2 : tensor<32x32xf32>, tensor<32x32xf32>
}
