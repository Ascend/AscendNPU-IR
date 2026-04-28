// RUN: bishengir-opt %s -split-input-file -convert-hivmave-to-ave-intrin | FileCheck %s

#map = affine_map<(d0) -> (-d0 + 514, 64)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec< #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: test_unalign_load_store
  func.func @test_unalign_load_store(%arg0: memref<3x514xi32, #hivm.address_space<ub>>) -> () {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c514 = arith.constant 514 : index
    %c64 = arith.constant 64 : index
    scf.for %arg3 = %c0 to %c514 step %c64 {
      %2 = affine.min #map(%arg3)
      scf.for %arg2 = %c0 to %c3 step %c1 {
        %subview_3 = memref.subview %arg0[%arg2, %arg3] [1, %2] [1, 1] : memref<3x514xi32, #hivm.address_space<ub>> to memref<1x?xi32, strided<[514, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_4 = memref.subview %subview_3[0, 0] [1, %2] [1, 1] : memref<1x?xi32, strided<[514, 1], offset: ?>, #hivm.address_space<ub>> to memref<?xi32, #map1, #hivm.address_space<ub>>
        // CHECK: "hivm_regbaseintrins.intr.hivm.vldas"
        // CHECK: "hivm_regbaseintrins.intr.hivm.vldus.post.s32"
        %10 = ave.hir.vload <NORM> %subview_4[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<?xi32, #map1, #hivm.address_space<ub>> into vector<64xi32>
        %5 = ave.hir.pge <ALL> : vector<1xi1>
        // CHECK: "hivm_regbaseintrins.intr.hivm.init.vector.align.data"
        // CHECK: "hivm_regbaseintrins.intr.hivm.vstus.post.s32"
        // CHECK: "hivm_regbaseintrins.intr.hivm.vstas"
        ave.hir.masked_store <NORM_B32> %subview_4[%c0], %5, %10 {ave.unaligned_ub_access = #ave.unaligned_ub_access} : memref<?xi32, #map1, #hivm.address_space<ub>>, vector<1xi1>, vector<64xi32>
      }
    }
    return
  }
}
