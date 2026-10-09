// Non-registry functions with existing preload-local marks must be handled
// after marking, with either automatic marking mode and no workspace slots.
// RUN: bishengir-opt %s -hivm-mark-multi-buffer="enable-auto=false enable-preload=false set-workspace-multibuffer=0" | FileCheck %s
// RUN: bishengir-opt %s -hivm-mark-multi-buffer="enable-auto=false enable-preload=true set-workspace-multibuffer=0" | FileCheck %s
// RUN: bishengir-opt %s -hivm-mark-multi-buffer="enable-auto=true enable-preload=false set-workspace-multibuffer=0" | FileCheck %s
// RUN: bishengir-opt %s -hivm-mark-multi-buffer="enable-auto=true enable-preload=true set-workspace-multibuffer=0" | FileCheck %s
// RUN: sed 's/hivm.multi_buffer = 1/hivm.multi_buffer = 2/' %s | bishengir-opt -hivm-mark-multi-buffer="enable-auto=false enable-preload=false set-workspace-multibuffer=0" | FileCheck %s --check-prefix=ROTATE
// RUN: sed 's/hivm.multi_buffer = 1/hivm.multi_buffer = 2/' %s | bishengir-opt -hivm-mark-multi-buffer="enable-auto=false enable-preload=true set-workspace-multibuffer=0" | FileCheck %s --check-prefix=ROTATE
// RUN: sed 's/hivm.multi_buffer = 1/hivm.multi_buffer = 2/' %s | bishengir-opt -hivm-mark-multi-buffer="enable-auto=true enable-preload=false set-workspace-multibuffer=0" | FileCheck %s --check-prefix=ROTATE
// RUN: sed 's/hivm.multi_buffer = 1/hivm.multi_buffer = 2/' %s | bishengir-opt -hivm-mark-multi-buffer="enable-auto=true enable-preload=true set-workspace-multibuffer=0" | FileCheck %s --check-prefix=ROTATE


func.func @vf_consume(
    %arg0: memref<32x128xf32, #hivm.address_space<ub>>)
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>,
                hivm.vector_function, no_inline} {
  return
}

// A multi_buffer=1 + preload_local=1 alloc used by one scope is
// sunk into that scope and its mark is deleted. multi_buffer>1 stays
// at loop scope so slots can still rotate.
// ROTATE-LABEL: func.func @sink_single_slot_preload_alloc
// ROTATE: %[[BUF:.*]] = memref.alloc
// ROTATE: annotation.mark %[[BUF]] {{.*}}multi_buffer = 2{{.*}}preload_local_buffer
// ROTATE: scope.scope
// CHECK-LABEL: func.func @sink_single_slot_preload_alloc
// CHECK-NOT:     memref.alloc
// CHECK:         scope.scope
// CHECK:           %[[BUF:.*]] = memref.alloc
// CHECK-NOT:       annotation.mark %[[BUF]]
// CHECK:           call @vf_consume(%[[BUF]])
func.func @sink_single_slot_preload_alloc()
    attributes {hivm.func_core_type = #hivm.func_core_type<AIV>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %buf = memref.alloc() : memref<32x128xf32, #hivm.address_space<ub>>
    annotation.mark %buf {hivm.multi_buffer = 1 : i32,
                          hivm.preload_local_buffer = 1 : i32} :
        memref<32x128xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @vf_consume(%buf) {hivm.vector_function, no_inline} :
          (memref<32x128xf32, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.preload_num = 0 : i32, no_inline}
  }
  return
}
