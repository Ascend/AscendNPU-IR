// RUN: bishengir-opt -convert-arith-to-hivmave -convert-vector-to-hivmave -ave-normalize-ops -convert-hivmave-to-ave-intrin %s | FileCheck %s

func.func @triton_kernel_outlined_vf_0(%arg0: memref<256xi1, #hivm.address_space<ub>>, %arg1: memref<256xi1, #hivm.address_space<ub>>) attributes {hivm.vector_function} {
  %cst = arith.constant dense<1> : vector<256xi1>
  %c0_i1 = arith.constant 0 : i1
  %c0 = arith.constant 0 : index
  %0 = vector.constant_mask [256] : vector<256xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.plds.b8
  %1 = vector.transfer_read %arg0[%c0], %c0_i1, %0 {in_bounds = [true]} : memref<256xi1, #hivm.address_space<ub>>, vector<256xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.pnot.z
  %2 = arith.xori %1, %cst : vector<256xi1>
  // CHECK: hivm_regbaseintrins.intr.hivm.psts.b8
  vector.transfer_write %2, %arg1[%c0], %0 {in_bounds = [true]} : vector<256xi1>, memref<256xi1, #hivm.address_space<ub>>
  return
}
