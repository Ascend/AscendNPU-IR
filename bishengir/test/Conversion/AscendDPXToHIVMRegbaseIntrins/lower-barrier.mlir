// RUN: bishengir-opt %s -convert-ascend-dpx-to-hivmregbaseintrins | FileCheck %s

// CHECK-LABEL:   llvm.func @test_multi_barrier_kernel(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !llvm.ptr<1>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: !llvm.ptr<6> {hivm.shared_memory}) attributes {hivm_regbaseintrins.kernel} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x i32 : (i32) -> !llvm.ptr
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_4]] : i32, !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = hivm_regbaseintrins.thread_id_x
// CHECK:           %[[VAL_8:.*]] = llvm.urem %[[VAL_7]], %[[VAL_6]]  : i32
// CHECK:           %[[VAL_9:.*]] = llvm.icmp "eq" %[[VAL_8]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_11:.*]] = llvm.udiv %[[VAL_7]], %[[VAL_6]]  : i32
// CHECK:           %[[VAL_12:.*]] = llvm.urem %[[VAL_11]], %[[VAL_10]]  : i32
// CHECK:           %[[VAL_13:.*]] = llvm.udiv %[[VAL_11]], %[[VAL_10]]  : i32
// CHECK:           %[[VAL_14:.*]] = llvm.icmp "eq" %[[VAL_13]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_15:.*]] = llvm.and %[[VAL_14]], %[[VAL_9]]  : i1
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_17:.*]] = llvm.mul %[[VAL_12]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(12 : i32) : i32
// CHECK:           %[[VAL_19:.*]] = llvm.add %[[VAL_18]], %[[VAL_17]] : i32
// CHECK:           %[[VAL_20:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_19]]] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i8
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_22:.*]] = llvm.add %[[VAL_21]], %[[VAL_17]] : i32
// CHECK:           %[[VAL_23:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_22]]] : (!llvm.ptr<6>, i32) -> !llvm.ptr<6>, i8
// CHECK:           scf.if %[[VAL_15]] {
// CHECK:             llvm.store %[[VAL_2]], %[[VAL_20]] : i32, !llvm.ptr<6>
// CHECK:             llvm.store %[[VAL_2]], %[[VAL_23]] : i32, !llvm.ptr<6>
// CHECK:           }
// CHECK:           hivm_regbaseintrins.sync_threads
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           scf.if %[[VAL_9]] {
// CHECK:             %[[VAL_25:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i32
// CHECK:             %[[VAL_26:.*]] = llvm.add %[[VAL_25]], %[[VAL_3]] : i32
// CHECK:             llvm.store %[[VAL_26]], %[[VAL_4]] : i32, !llvm.ptr
// CHECK:             %[[VAL_27:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_28:.*]] = llvm.call_intrinsic "llvm.hivm.atom.ADD.S.s32"(%[[VAL_20]], %[[VAL_3]], %[[VAL_27]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:             %[[VAL_29:.*]] = llvm.sub %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_30:.*]] = llvm.icmp "eq" %[[VAL_28]], %[[VAL_29]] : i32
// CHECK:             scf.if %[[VAL_30]] {
// CHECK:               %[[VAL_31:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:               %[[VAL_32:.*]] = llvm.call_intrinsic "llvm.hivm.atom.EXCH.S.s32"(%[[VAL_20]], %[[VAL_2]], %[[VAL_31]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:               %[[VAL_33:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:               %[[VAL_34:.*]] = llvm.call_intrinsic "llvm.hivm.atom.ADD.S.s32"(%[[VAL_23]], %[[VAL_3]], %[[VAL_33]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:             } else {
// CHECK:               scf.while : () -> () {
// CHECK:                 %[[VAL_35:.*]] = llvm.load volatile %[[VAL_23]] : !llvm.ptr<6> -> i32
// CHECK:                 %[[VAL_36:.*]] = llvm.icmp "ne" %[[VAL_35]], %[[VAL_26]] : i32
// CHECK:                 scf.condition(%[[VAL_36]])
// CHECK:               } do {
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = llvm.add %[[VAL_24]], %[[VAL_24]] : i32
// CHECK:           scf.if %[[VAL_9]] {
// CHECK:             %[[VAL_38:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i32
// CHECK:             %[[VAL_39:.*]] = llvm.add %[[VAL_38]], %[[VAL_3]] : i32
// CHECK:             llvm.store %[[VAL_39]], %[[VAL_4]] : i32, !llvm.ptr
// CHECK:             %[[VAL_40:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_41:.*]] = llvm.call_intrinsic "llvm.hivm.atom.ADD.S.s32"(%[[VAL_20]], %[[VAL_3]], %[[VAL_40]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:             %[[VAL_42:.*]] = llvm.sub %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_43:.*]] = llvm.icmp "eq" %[[VAL_41]], %[[VAL_42]] : i32
// CHECK:             scf.if %[[VAL_43]] {
// CHECK:               %[[VAL_44:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:               %[[VAL_45:.*]] = llvm.call_intrinsic "llvm.hivm.atom.EXCH.S.s32"(%[[VAL_20]], %[[VAL_2]], %[[VAL_44]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:               %[[VAL_46:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:               %[[VAL_47:.*]] = llvm.call_intrinsic "llvm.hivm.atom.ADD.S.s32"(%[[VAL_23]], %[[VAL_3]], %[[VAL_46]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:             } else {
// CHECK:               scf.while : () -> () {
// CHECK:                 %[[VAL_48:.*]] = llvm.load volatile %[[VAL_23]] : !llvm.ptr<6> -> i32
// CHECK:                 %[[VAL_49:.*]] = llvm.icmp "ne" %[[VAL_48]], %[[VAL_39]] : i32
// CHECK:                 scf.condition(%[[VAL_49]])
// CHECK:               } do {
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_50:.*]] = llvm.add %[[VAL_37]], %[[VAL_24]] : i32
// CHECK:           scf.if %[[VAL_9]] {
// CHECK:             %[[VAL_51:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i32
// CHECK:             %[[VAL_52:.*]] = llvm.add %[[VAL_51]], %[[VAL_3]] : i32
// CHECK:             llvm.store %[[VAL_52]], %[[VAL_4]] : i32, !llvm.ptr
// CHECK:             %[[VAL_53:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:             %[[VAL_54:.*]] = llvm.call_intrinsic "llvm.hivm.atom.ADD.S.s32"(%[[VAL_20]], %[[VAL_3]], %[[VAL_53]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:             %[[VAL_55:.*]] = llvm.sub %[[VAL_5]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_56:.*]] = llvm.icmp "eq" %[[VAL_54]], %[[VAL_55]] : i32
// CHECK:             scf.if %[[VAL_56]] {
// CHECK:               %[[VAL_57:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:               %[[VAL_58:.*]] = llvm.call_intrinsic "llvm.hivm.atom.EXCH.S.s32"(%[[VAL_20]], %[[VAL_2]], %[[VAL_57]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:               %[[VAL_59:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:               %[[VAL_60:.*]] = llvm.call_intrinsic "llvm.hivm.atom.ADD.S.s32"(%[[VAL_23]], %[[VAL_3]], %[[VAL_59]]) : (!llvm.ptr<6>, i32, i32) -> i32
// CHECK:             } else {
// CHECK:               scf.while : () -> () {
// CHECK:                 %[[VAL_61:.*]] = llvm.load volatile %[[VAL_23]] : !llvm.ptr<6> -> i32
// CHECK:                 %[[VAL_62:.*]] = llvm.icmp "ne" %[[VAL_61]], %[[VAL_52]] : i32
// CHECK:                 scf.condition(%[[VAL_62]])
// CHECK:               } do {
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>,
  "ttg.num-warps" = 4 : i32,
  "ttg.super-block-factor" = 2 : ui32,
  ttg.shared = 12 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.super-block-barrier" = true
} {
  llvm.func @test_multi_barrier_kernel(
    %arg0: !llvm.ptr<1>,
    %arg1: !llvm.ptr<6> {hivm.shared_memory}
  ) attributes {hivm_regbaseintrins.kernel} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    ascend_dpx.sync_threads
    %1 = llvm.add %0, %0 : i32
    ascend_dpx.sync_threads
    %2 = llvm.add %1, %0 : i32
    ascend_dpx.sync_threads
    llvm.return
  }
}