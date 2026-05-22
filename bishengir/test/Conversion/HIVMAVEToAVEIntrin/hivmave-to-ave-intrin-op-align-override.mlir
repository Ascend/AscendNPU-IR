// RUN: bishengir-opt -convert-hivmave-to-ave-intrin %s | FileCheck %s

// Test that when a function has element_alignment_bit_width=32 but individual
// load/store/pge ops have element_alignment_bit_width=16 (matching the element
// width), the op's own alignment takes priority over the parent function's.
// This ensures NORM loads are not incorrectly converted to UNPK_B16 and
// NORM_B16 stores are not incorrectly converted to PK_B32.

// CHECK-LABEL: func.func @test_op_align_overrides_parent_align
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>} {
  func.func @test_op_align_overrides_parent_align(
      %arg0: memref<128xf16, #hivm.address_space<ub>>,
      %arg1: memref<128xf16, #hivm.address_space<ub>>,
      %arg2: memref<128xf16, #hivm.address_space<ub>>)
      attributes {element_alignment_bit_width = 32 : i32,
                  hivm.func_core_type = #hivm.func_core_type<AIV>,
                  hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    // Load with op's own alignment=16 (matches f16 elemWidth=16).
    // Should use dist=0 (NORM), NOT dist=14 (UNPK_B16).
    // CHECK: llvm.mlir.constant(0 : i32)
    // CHECK: llvm.mlir.constant(0 : i32)
    // CHECK: %[[LDIST0:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%{{.*}}, %{{.*}}, %[[LDIST0]], %{{.*}})
    %res = ave.hir.vload <NORM> %arg0[%c0] {element_alignment_bit_width = 16 : i32} : memref<128xf16, #hivm.address_space<ub>> into vector<128xf16>
    // CHECK: llvm.mlir.constant(0 : i32)
    // CHECK: llvm.mlir.constant(0 : i32)
    // CHECK: %[[LDIST1:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%{{.*}}, %{{.*}}, %[[LDIST1]], %{{.*}})
    %res_1 = ave.hir.vload <NORM> %arg1[%c0] {element_alignment_bit_width = 16 : i32} : memref<128xf16, #hivm.address_space<ub>> into vector<128xf16>
    // Pge for vmul: should be pge.b16, NOT pge.b32.
    // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
    %0 = ave.hir.pge <ALL> {element_alignment_bit_width = -1 : i32} : vector<128xi1>
    %1 = ave.hir.vmul %res, %res_1, %0 {element_alignment_bit_width = 16 : i32} : vector<128xf16>, vector<128xi1>
    // Pge for store: should be pge.b16, NOT pge.b32.
    // CHECK: "hivm_regbaseintrins.intr.hivm.pge.b16"
    %2 = ave.hir.pge <ALL> {element_alignment_bit_width = -1 : i32} : vector<128xi1>
    // Store with op's own alignment=16 (matches f16 elemWidth=16).
    // Should use dist=1 (NORM_B16), NOT dist=7 (PK_B32).
    // CHECK: %[[SDIST:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: "hivm_regbaseintrins.intr.hivm.vstsx1.v128f16"(%{{.*}}, %{{.*}}, %{{.*}}, %[[SDIST]], %{{.*}}, %{{.*}})
    ave.hir.masked_store <NORM_B16> %arg2[%c0], %2, %1 {element_alignment_bit_width = 16 : i32} : memref<128xf16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
    return
  }
}
