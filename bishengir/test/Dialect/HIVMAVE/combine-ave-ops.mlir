// RUN: bishengir-opt -combine-ave-ops="enable-fused-multiply-add=true" %s --split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @subexp_to_vexp
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @subexp_to_vexp(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %1 = ave.hir.vload <NORM> %arg1[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK-NOT: ave.hir.vsub
    // CHECK-NOT: ave.hir.vexp
    // CHECK: ave.hir.vexpdif
    %3 = ave.hir.vsub %0, %1, %2 : vector<64xf32>, vector<64xi1>
    %4 = ave.hir.pge <ALL> : vector<64xi1>
    %5 = ave.hir.vexp %3, %4 : vector<64xf32>, vector<64xi1>
    %6 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg2[%c0], %6, %5 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @muladd_to_mula_lhs
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @muladd_to_mula_lhs(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %1 = ave.hir.vload <NORM> %arg1[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK-NOT: ave.hir.vmul
    // CHECK-NOT: ave.hir.vadd
    // CHECK: ave.hir.vmula
    %3 = ave.hir.vmul %0, %1, %2 : vector<64xf32>, vector<64xi1>
    %4 = ave.hir.vload <NORM> %arg2[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.vadd %3, %4, %5 : vector<64xf32>, vector<64xi1>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg3[%c0], %7, %6 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @muladd_to_mula_rhs
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @muladd_to_mula_rhs(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>) attributes {element_alignment_bit_width = 32 : i32, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %0 = ave.hir.vload <NORM> %arg0[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %1 = ave.hir.vload <NORM> %arg1[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %2 = ave.hir.pge <ALL> : vector<64xi1>
    // CHECK-NOT: ave.hir.vmul
    // CHECK-NOT: ave.hir.vadd
    // CHECK: ave.hir.vmula
    %3 = ave.hir.vmul %0, %1, %2 : vector<64xf32>, vector<64xi1>
    %4 = ave.hir.vload <NORM> %arg2[%c0] : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %5 = ave.hir.pge <ALL> : vector<64xi1>
    %6 = ave.hir.vadd %4, %3, %5 : vector<64xf32>, vector<64xi1>
    %7 = ave.hir.pge <ALL> : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg3[%c0], %7, %6 : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
}
