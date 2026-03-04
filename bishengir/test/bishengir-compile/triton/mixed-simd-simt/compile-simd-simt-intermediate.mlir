// RUN: bishengir-compile -enable-lir-compile=true -enable-hfusion-compile=true --target=Ascend910_9589 -enable-triton-kernel-compile -enable-simd-simt-mix-compile -pure-simt -enable-triton-ir-compile=true %s

module @top_level {
    module @main attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
        func.func @add_kernel(%arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mix_mode = "aiv", parallel_mode = "simd"} {
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1]>>
            %0 = bufferization.to_tensor %reinterpret_cast restrict writable : memref<1024xf32, strided<[1]>>
            %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1]>>
            %1 = bufferization.to_tensor %reinterpret_cast_0 restrict writable : memref<1024xf32, strided<[1]>>
            %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1]>>
            %out = bufferization.to_tensor %reinterpret_cast_1 restrict writable : memref<1024xf32, strided<[1]>>

            %3 = func.call @simt_func(%0, %1, %out) {hivm.vector_function,hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline}: (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
            return
        }

        func.func @simt_func(%arg0 :tensor<1024xf32> {memory.effect = #hivm.memory.effect<read>}, %arg1:tensor<1024xf32>{memory.effect = #hivm.memory.effect<read>}, %arg2:tensor<1024xf32>{memory.effect = #hivm.memory.effect<write>}) -> tensor<1024xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, no_inline,simt_wrapper,hivm.vector_function} {
            return %arg2:tensor<1024xf32>
        }
    }
    
    module @simt_mod attributes{dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hacc.simt_module}{
        tt.func public @simt_func(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
            %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
            %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> 
            %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> 
            %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
            %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> 
            %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> 
            %6 = tt.load %5 : tensor<1024x!tt.ptr<f32>> 
            %7 = arith.addf %3, %6 : tensor<1024xf32> 
            %8 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> 
            %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> 
            tt.store %9, %7 : tensor<1024x!tt.ptr<f32>>
            tt.return 
        }
    }
}