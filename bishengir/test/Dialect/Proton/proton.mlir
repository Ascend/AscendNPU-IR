// default option, opt
// RUN: bishengir-opt %s \
// RUN:   -convert-proton-to-protongpu \
// RUN:   -cse \
// RUN:   -allocate-proton-shared-memory \
// RUN:   -convert-triton-ascend-gpu-to-llvm \
// RUN:   -allocate-proton-global-scratch-buffer \
// RUN:   -convert-proton-ascend-gpu-to-llvm \
// RUN:   -mlir-print-ir-after-all \
// RUN:   2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-NO-SAMPLING

// default option, compile
// RUN: bishengir-compile \
// RUN:   %s \
// RUN:   --target=Ascend910_9589 \
// RUN:   --enable-triton-ir-compile \
// RUN:   --use-dpx \
// RUN:   --pure-simt \
// RUN:   --num-warps=8 \
// RUN:   --threads-per-warp=32 \
// RUN:   --enable-bishengir-simt-optimization=100 \
// RUN:   --mlir-print-ir-after-all \
// RUN:   2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-NO-SAMPLING --dump-input=fail

// non-default option
// RUN: bishengir-compile \
// RUN:   %s \
// RUN:   --target=Ascend910_9589 \
// RUN:   --enable-triton-ir-compile \
// RUN:   --use-dpx \
// RUN:   --pure-simt \
// RUN:   --num-warps=8 \
// RUN:   --threads-per-warp=32 \
// RUN:   --enable-bishengir-simt-optimization=100 \
// RUN:   --proton-max-shared-mem=1024 \
// RUN:   --proton-profile-scratch-size=1024 \
// RUN:   --proton-clk-ext=true \
// RUN:   --mlir-print-ir-after-all \
// RUN:   2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-NO-SAMPLING

// selective sampling
// RUN: bishengir-compile \
// RUN:   %s \
// RUN:   --target=Ascend910_9589 \
// RUN:   --enable-triton-ir-compile \
// RUN:   --use-dpx \
// RUN:   --pure-simt \
// RUN:   --num-warps=8 \
// RUN:   --threads-per-warp=32 \
// RUN:   --enable-bishengir-simt-optimization=100 \
// RUN:   --proton-sampling-strategy=selective \
// RUN:   --proton-sampling-options=0,2 \
// RUN:   --mlir-print-ir-after-all \
// RUN:   2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-SELECTIVE


module attributes { "ttg.num-warps" = 8, "ttg.enable-bishengir-simt-optimization" = 100 : i32 } {
  tt.func public @foo() {
    proton.record start "kernel"
    proton.record end "kernel"
    tt.return
  }
}

// ==========
// CHECK: IR Dump After ConvertProtonToProtonGPU

// CHECK: IR Dump After CSE
// CHECK: %[[PROFILE_MEM:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = [[PROFILE_MEM_BYTES:.*]] : i32} : !tt.ptr<i32>
// CHECK: %[[BUFFER:.*]] = ttg.local_alloc : () -> !ttg.memdesc<[[SEGMENT_WORDS:.*]]xi32, #shared, #smem, mutable>
// CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUFFER]] : !ttg.memdesc<[[SEGMENT_WORDS]]xi32, #shared, #smem, mutable> -> <[[SEGMENT_BYTES:.*]], #smem, [[granularity:.*]][[selectIdVec:(,.*)?]]>

// CHECK: %[[COUNTER_START:.*]] = proton_gpu.read_counter : i[[CLOCK_TY:.*]]
// CHECK: proton_gpu.circular_store start %[[SEGMENT]], %[[COUNTER_START]] {scopeId = 0 : i32} : !proton_gpu.segment<[[SEGMENT_BYTES]], #smem, [[granularity]][[selectIdVec]]>, i[[CLOCK_TY]]

// CHECK: %[[COUNTER_END:.*]] = proton_gpu.read_counter : i[[CLOCK_TY]]
// CHECK: proton_gpu.circular_store end %[[SEGMENT]], %[[COUNTER_END]] {scopeId = 0 : i32} : !proton_gpu.segment<[[SEGMENT_BYTES]], #smem, [[granularity]][[selectIdVec]]>, i[[CLOCK_TY]]

// CHECK: gpu.barrier
// CHECK: proton_gpu.finalize %[[SEGMENT]], %[[PROFILE_MEM]] : !proton_gpu.segment<[[SEGMENT_BYTES]], #smem, [[granularity]][[selectIdVec]]>, !tt.ptr<i32>

// ==========
// CHECK: IR Dump After AllocateProtonSharedMemoryPass
// CHECK: module
// CHECK-SAME: ttg.shared = [[SEGMENT_BYTES]]

// ==========
// CHECK: IR Dump After ConvertTritonAscendGPUToLLVM
// CHECK: @foo
// CHECK-SAME: %[[GLOBAL_SCRATCH:.*]]: !llvm.ptr<1>,
// CHECK-SAME: %[[PROFILE_SCRATCH:.*]]: !llvm.ptr<1>

// converted from ttg.local_alloc
// CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[SMEM_BASE:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
// CHECK: %[[BUFFER:.*]] = llvm.getelementptr %[[SMEM_BASE]]{{\[}}%[[OFFSET]]{{\]}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
// CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.mlir.undef : !llvm.struct<(ptr<3>, i32)>
// CHECK: llvm.insertvalue %[[BUFFER]], %{{.*}}[0] : !llvm.struct<(ptr<3>, i32)> 
// CHECK: %[[SMEM_OBJ:.*]] = llvm.insertvalue %[[CST_0]], %{{.*}}[1] : !llvm.struct<(ptr<3>, i32)> 
// CHECK: %[[SMEM_OBJ:.*]] = builtin.unrealized_conversion_cast {{.*}} : !llvm.struct<(ptr<3>, i32)> to !ttg.memdesc<[[SEGMENT_WORDS]]xi32, #shared, #smem, mutable>

// CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[SMEM_OBJ]] : !ttg.memdesc<[[SEGMENT_WORDS]]xi32, #shared, #smem, mutable> -> <[[SEGMENT_BYTES]], #smem, [[granularity]][[selectIdVec]]>

// ==========
// CHECK: IR Dump After AllocateProtonGlobalScratchBufferPass
// CHECK: module
// CHECK-SAME: ttg.profile_scratch_memory_size = [[PROFILE_MEM_BYTES]]

// ==========
// CHECK: IR Dump After ReconcileUnrealizedCasts

// ==========
// CHECK: IR Dump After ConvertProtonAscendGPUToLLVM
// CHECK: @foo
// CHECK-SAME: %[[GLOBAL_SCRATCH:.*]]: !llvm.ptr<1>,
// CHECK-SAME: %[[PROFILE_SCRATCH:.*]]: !llvm.ptr<1>

// converted from proton_gpu.global_scratch_alloc
// CHECK: %[[ALLOC_OFFSET:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: ascend_dpx.block_idx
// CHECK: llvm.sext
// CHECK: %[[LINEAR_ID:.*]] = llvm.trunc
// CHECK: llvm.mlir.constant([[PROFILE_MEM_BYTES]] : i32) : i32
// CHECK: llvm.mul %[[LINEAR_ID]], {{.*}}
// CHECK: %[[GMEM_OFFSET:.*]] = llvm.add %[[ALLOC_OFFSET]],
// CHECK: %[[SEGMENT:.*]] = llvm.getelementptr %[[PROFILE_SCRATCH]]{{\[}}%[[GMEM_OFFSET]]{{\]}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8

// converted from proton_gpu.segment_alloc
// CHECK: ascend_dpx.thread_id_x
// CHECK: llvm.sext
// CHECK: %[[CUR_THREAD_ID:.*]] = llvm.trunc

// CHECK: %[[THREADS_PER_WARP:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK: %[[CUR_WARP_ID:.*]] = llvm.udiv %[[CUR_THREAD_ID]], %[[THREADS_PER_WARP]]  : i32

// allWarpSegmentAlloc
// CHECK-NO-SAMPLING: %[[SEGMENT_WORDS:.*]] = llvm.mlir.constant({{.*}} : i32) : i32
// CHECK-NO-SAMPLING: %[[segmentBase:.*]] = llvm.mul %[[CUR_WARP_ID]], %[[SEGMENT_WORDS]] : i32

// defaultSegmentAlloc
// CHECK-SELECTIVE: llvm.icmp "eq"
// CHECK-SELECTIVE: llvm.select
// CHECK-SELECTIVE: llvm.icmp "eq"
// CHECK-SELECTIVE: %[[segmentBase:.*]] = llvm.select

// CHECK: %[[INDEX_PTR:.*]] = llvm.alloca {{.*}} x i32 : (i32) -> !llvm.ptr<1>
// CHECK: %[[CST_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.store %[[CST_0]], %[[INDEX_PTR]] : i32, !llvm.ptr<1>

// CHECK: llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, ptr<1>)>
// CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<3>, i32, ptr<1>)>
// CHECK: llvm.insertvalue %[[segmentBase]], %{{.*}}[1] : !llvm.struct<(ptr<3>, i32, ptr<1>)>
// CHECK: %[[SEGMENT_OBJ:.*]] = llvm.insertvalue %[[INDEX_PTR]], %{{.*}}[2] : !llvm.struct<(ptr<3>, i32, ptr<1>)>

// converted from proton_gpu.read_counter
// CHECK: %[[CLOCK:.*]] = ascend_dpx.clock[[CLOCK_TY]]

// converted from proton_gpu.circular_store
// CHECK: %[[INDEX_PTR:.*]] = llvm.extractvalue %[[SEGMENT_OBJ]][0] : !llvm.struct<(ptr<3>, i32, ptr<1>)>
// CHECK: %[[BUFFER_BASE:.*]] = llvm.extractvalue %[[SEGMENT_OBJ]][1] : !llvm.struct<(ptr<3>, i32, ptr<1>)>
// CHECK: %[[SEGMENTBASE:.*]] = llvm.extractvalue %[[SEGMENT_OBJ]][2] : !llvm.struct<(ptr<3>, i32, ptr<1>)>

// Update the index
// CHECK: llvm.load {{.*}} : !llvm.ptr<1> -> i32
// CHECK: llvm.add
// CHECK: llvm.store {{.*}} : i32, !llvm.ptr<1>

// Compute the actual base offset (with urem as circular buffer).
// CHECK: llvm.urem
// CHECK: llvm.add

// Store the counter into buffer.
// CHECK: %[[PTR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32

// // Constructing the tag and clock (8 byte)
// CHECK: llvm.insertelement {{.*}} : vector<2xi32>
// CHECK: %[[RECORD:.*]] = llvm.insertelement {{.*}} : vector<2xi32>
// CHECK: ascend_dpx.store %[[PTR]], %[[RECORD]], {{.*}} : <3>, vector<2xi32>

// CHECK: ascend_dpx.sync_threads

// converted from proton_gpu.finalize
// CHECK: %[[INDEX_PTR:.*]] = llvm.extractvalue %[[SEGMENT_OBJ]][0] : !llvm.struct<(ptr<3>, i32, ptr<1>)>
// CHECK: %[[BUFFER_BASE:.*]] = llvm.extractvalue %[[SEGMENT_OBJ]][1] : !llvm.struct<(ptr<3>, i32, ptr<1>)>
// CHECK: %[[SEGMENTBASE:.*]] = llvm.extractvalue %[[SEGMENT_OBJ]][2] : !llvm.struct<(ptr<3>, i32, ptr<1>)>

// CHECK: ascend_dpx.thread_id_x
// CHECK: llvm.sext
// CHECK: %[[THREAD_ID:.*]] = llvm.trunc
// CHECK: %[[THREADS_PER_WARP:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK: %[[WARP_ID:.*]] = llvm.udiv %[[THREAD_ID]], %[[THREADS_PER_WARP]] : i32
// CHECK: %[[IS_FIRST_THREAD:.*]] = llvm.icmp "eq" %[[THREAD_ID]], {{.*}}

// CHECK: llvm.cond_br %[[IS_FIRST_THREAD:.*]], ^bb1, ^bb3
// CHECK: ^bb1:  // pred: ^bb0

// CHECK: llvm.br ^bb2(%{{.*}} : i32)
// CHECK: ^bb2(%[[INDEX:.*]]: i32):  // 2 preds: ^bb1, ^bb2

// CHECK: %[[CMP:.*]] = llvm.icmp "slt" %[[INDEX]], {{.*}} : i32
// CHECK: %[[STEP:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[NEW_INDEX:.*]] = llvm.add %[[INDEX]], %[[STEP]] : i32
// CHECK: llvm.cond_br %[[CMP]], ^bb2(%[[NEW_INDEX]] : i32), ^bb3

// CHECK: ^bb3:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT: llvm.return