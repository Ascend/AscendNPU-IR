// RUN: bishengir-opt --hivm-inline-fixpipe --debug-only=hivm-inline-fixpipe %s

func.func @_fwd(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "mix", parallel_mode = "simd"} {
  %c2 = arith.constant 2 : index
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0_i8 = arith.constant 0 : i8
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg9, %arg8 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg7 : i32
  scf.for %arg10 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
    %7 = arith.muli %arg10, %c4_i32 : i32
    %8 = arith.addi %7, %6 : i32
    %9 = arith.divsi %8, %c4_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%10], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %11 = memref.load %reinterpret_cast[%c0] : memref<1xi8, strided<[1], offset: ?>>
    %12 = arith.cmpi ne, %11, %c0_i8 : i8
    scf.if %12 {
      %13 = arith.index_cast %7 : i32 to index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [2, 2], strides: [2, 1] : memref<?xbf16> to memref<2x2xbf16, strided<[2, 1], offset: ?>>
      %alloc = memref.alloc() : memref<2x2xbf16>
      hivm.hir.load ins(%reinterpret_cast_0 : memref<2x2xbf16, strided<[2, 1], offset: ?>>) outs(%alloc : memref<2x2xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %14 = bufferization.to_tensor %alloc restrict writable : memref<2x2xbf16>
      %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%13], sizes: [2, 2], strides: [2, 1] : memref<?xbf16> to memref<2x2xbf16, strided<[2, 1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<2x2xbf16>
      hivm.hir.load ins(%reinterpret_cast_1 : memref<2x2xbf16, strided<[2, 1], offset: ?>>) outs(%alloc_2 : memref<2x2xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      %15 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x2xbf16>
      %16 = tensor.empty() : tensor<2x2xf32>
      %17 = hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%14, %15, %true, %c2, %c2, %c2 : tensor<2x2xbf16>, tensor<2x2xbf16>, i1, index, index, index) outs(%16 : tensor<2x2xf32>) -> tensor<2x2xf32>
      %18 = tensor.empty() : tensor<2x2xbf16>
      %19 = hivm.hir.fixpipe {enable_nz2nd, pre_quant = #hivm.fixpipe_pre_quant_mode<F322BF16>} ins(%17 : tensor<2x2xf32>) outs(%18 : tensor<2x2xbf16>) -> tensor<2x2xbf16>
      hivm.hir.debug {debugtype = "print", hex = false, prefix = " do: ", tcoretype = #hivm.tcore_type<CUBE>} %19 : tensor<2x2xbf16>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%13], sizes: [2, 2], strides: [2, 1] : memref<?xbf16> to memref<2x2xbf16, strided<[2, 1], offset: ?>>
      hivm.hir.store ins(%19 : tensor<2x2xbf16>) outs(%reinterpret_cast_3 : memref<2x2xbf16, strided<[2, 1], offset: ?>>)
    }
  }
  return
}

// CHECK-NOT: hivm.store

// Match both fixpipes via their distinct outputs
// CHECK-DAG: [[T_ALLOC:%[0-9]+]] = tensor.empty() : tensor<2x2xbf16>
// CHECK-DAG: [[FP_DBG:%[0-9]+]] = "hivm.hir.fixpipe"({{.*}}) outs([[T_ALLOC]] : tensor<2x2xbf16>) -> tensor<2x2xbf16>
// CHECK-DAG: [[REINTERP:%[0-9]+]] = memref.reinterpret_cast {{.*}} : memref<?xbf16> to memref<2x2xbf16, strided<[2, 1], offset: ?>>
// CHECK-DAG: "hivm.hir.fixpipe"({{.*}}) outs([[REINTERP]] : memref<2x2xbf16, strided<[2, 1], offset: ?>>)

// Ensure debug uses the tensor-path fixpipe
// CHECK: "hivm.hir.debug"([[FP_DBG]]) : (tensor<2x2xbf16>) -> ()