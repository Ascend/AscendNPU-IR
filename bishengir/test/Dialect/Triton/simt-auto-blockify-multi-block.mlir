// RUN: bishengir-opt %s -simt-auto-blockify | FileCheck %s
// RUN: bishengir-opt %s -simt-auto-blockify="superblock-factor=1" | FileCheck %s
// RUN: bishengir-opt %s -simt-auto-blockify="superblock-factor=2" | FileCheck %s --check-prefix=SUPERBLOCK

// CHECK-LABEL: tt.func public @blockify_existing_grid(
// CHECK-SAME: %[[GRID_X:arg[0-9]+]]: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}
// CHECK-SAME: %[[GRID_Y:arg[0-9]+]]: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}
// CHECK-SAME: %[[GRID_Z:arg[0-9]+]]: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}

// CHECK: %[[YZ:.*]] = arith.muli %[[GRID_Y]], %[[GRID_Z]] : i32
// CHECK: %[[LOGICAL:.*]] = arith.muli %[[GRID_X]], %[[YZ]] : i32
// CHECK: %[[HW_IDX:.*]] = gpu.linear_block_id
// CHECK: %[[BLOCK_IDX:.*]] = arith.index_cast %[[HW_IDX]] : index to i32
// CHECK: %[[PHYSICAL:.*]] = arith.constant 64 : i32
// CHECK: %[[CHUNK:.*]] = arith.ceildivui %[[LOGICAL]], %[[PHYSICAL]] : i32
// CHECK: %[[START:.*]] = arith.muli %[[BLOCK_IDX]], %[[CHUNK]] : i32
// CHECK: %[[END:.*]] = arith.addi %[[START]], %[[CHUNK]] : i32
// CHECK: %[[UPPER:.*]] = arith.minui %[[END]], %[[LOGICAL]] : i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-NEXT: scf.for %[[IV:.*]] = %[[START]] to %[[UPPER]] step %[[ONE]] : i32 {
// CHECK-NEXT:  %[[DIV_X:.*]] = arith.divui %[[IV]], %[[GRID_X]] : i32
// CHECK-NEXT:  %[[PID_X:.*]] = arith.remui %[[IV]], %[[GRID_X]] : i32
// CHECK-NEXT:  %[[PID_Y:.*]] = arith.remui %[[DIV_X]], %[[GRID_Y]] : i32
// CHECK-NEXT:  %[[PID_Z:.*]] = arith.divui %[[DIV_X]], %[[GRID_Y]] : i32
// CHECK-NEXT:  scf.execute_region {
// CHECK-NEXT:    %[[FIVE:.*]] = arith.constant 5 : i32
// CHECK-NEXT:    %[[COND:.*]] = arith.cmpi sgt, %[[PID_X]], %[[FIVE]] : i32
// CHECK-NEXT:    cf.cond_br %[[COND]], ^bb1, ^bb2
// CHECK-NEXT:    ^bb1: // pred: ^bb0
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    ^bb2: // pred: ^bb0
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT: }
// CHECK-NEXT: tt.return
// CHECK-NOT: tt.get_program_id

// SUPERBLOCK-LABEL: tt.func public @blockify_existing_grid(
// SUPERBLOCK-SAME: %[[GRID_X:arg[0-9]+]]: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32}
// SUPERBLOCK-SAME: %[[GRID_Y:arg[0-9]+]]: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32}
// SUPERBLOCK-SAME: %[[GRID_Z:arg[0-9]+]]: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}

// SUPERBLOCK: %[[UPPER:.*]] = arith.minui
// SUPERBLOCK: %[[SBF:.*]] = arith.constant 2 : i32
// SUPERBLOCK: %[[WARP:.*]] = arith.constant 32 : i32
// SUPERBLOCK: scf.for %[[IV:.*]] = {{.*}} to %[[UPPER]] step %[[SBF]] : i32 {
// SUPERBLOCK-NEXT:  %[[TID:.*]] = ascend_dpx.thread_id_x
// SUPERBLOCK-NEXT:  %[[WID:.*]] = arith.divui %[[TID]], %[[WARP]] : i32
// SUPERBLOCK-NEXT:  %[[REM:.*]] = arith.remui %[[WID]], %[[SBF]] : i32
// SUPERBLOCK-NEXT:  %[[LINEAR:.*]] = arith.addi %[[IV]], %[[REM]] : i32
// SUPERBLOCK-NEXT:  %[[COND:.*]] = arith.cmpi slt, %[[LINEAR]], %[[UPPER]] : i32
// SUPERBLOCK-NEXT:  scf.if %[[COND]] {
// SUPERBLOCK-NEXT:    %[[DIV_X:.*]] = arith.divui %[[LINEAR]], %[[GRID_X]] : i32
// SUPERBLOCK-NEXT:    %[[PID_X:.*]] = arith.remui %[[LINEAR]], %[[GRID_X]] : i32
// SUPERBLOCK-NEXT:    %[[PID_Y:.*]] = arith.remui %[[DIV_X]], %[[GRID_Y]] : i32
// SUPERBLOCK-NEXT:    %[[PID_Z:.*]] = arith.divui %[[DIV_X]], %[[GRID_Y]] : i32
// SUPERBLOCK-NEXT:    scf.execute_region {
// SUPERBLOCK-NEXT:      %[[FIVE:.*]] = arith.constant 5 : i32
// SUPERBLOCK-NEXT:      %[[CMP:.*]] = arith.cmpi sgt, %[[PID_X]], %[[FIVE]] : i32
// SUPERBLOCK-NEXT:      cf.cond_br %[[CMP]], ^bb1, ^bb2
// SUPERBLOCK-NEXT:      ^bb1: // pred: ^bb0
// SUPERBLOCK-NEXT:        scf.yield
// SUPERBLOCK-NEXT:      ^bb2: // pred: ^bb0
// SUPERBLOCK-NEXT:        scf.yield
// SUPERBLOCK-NEXT:    }
// SUPERBLOCK:      }
// SUPERBLOCK:    }
// SUPERBLOCK-NEXT: tt.return
// SUPERBLOCK-NOT: tt.get_program_id


module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  tt.func public @blockify_existing_grid(
      %arg0: i32 {tt.divisibility = 16 : i32},
      %arg1: i32 {gpu.block = #gpu.block<x>, tt.divisibility = 1 : i32},
      %arg2: i32 {gpu.block = #gpu.block<y>, tt.divisibility = 1 : i32},
      %arg3: i32 {gpu.block = #gpu.block<z>, tt.divisibility = 1 : i32}) {
    %c5_i32 = arith.constant 5 : i32
    %pid_x = tt.get_program_id x : i32
    %cmp = arith.cmpi sgt, %pid_x, %c5_i32 : i32
    cf.cond_br %cmp, ^bb1, ^bb2
  ^bb1:
    tt.return
  ^bb2:
    tt.return
  }
}
