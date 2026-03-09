# Programming model

This section introduces the AscendNPU IR programming model and its core concepts. We build up to a dynamic, high-performance GEMM (matrix multiply) that uses the main features of AscendNPU IR.

AscendNPU IR’s tile-based programming model preserves high-level semantics so the compiler can apply pipeline parallelism, memory management, and other optimizations. We introduce ways to represent tensor computation by gradually adjusting examples to use Tile IR and to obtain performance portability.

## Heterogeneous programming model

```
┌──────────────────────────────────────┐
│            Host                       │
│  • CPU and host memory                │
│  • Sequential execution               │
│  • Program flow control               │
└──────────────────────┬───────────────┘
                       │ NPU API bridge
┌──────────────────────┴───────────────┐
│            Device                    │
│  • NPU and device memory             │
│  • Massively parallel execution      │
│  • Data-parallel compute focus       │
└──────────────────────────────────────┘
```

AscendNPU IR supports Host/Device heterogeneous programming: Host code runs on the CPU, Device kernels on the device and can run concurrently.

The `hacc.function_kind` attribute indicates Host vs Device. For Device kernels, `hacc.entry` marks the entry. `hacc.launch` on the host launches a Device kernel with a logical block dimension (e.g. `block_dim`); the logical count can exceed the physical core count and the NPU schedules execution.

```
module {
  func.func @device_kernel()
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    hivm.hir.debug {debugtype = "print", prefix = "hello world"}
    return
  }

  func.func @host_func(%block_dim: i32)
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    hacc.launch [%block_dim] @device_kernel() : () -> ()
    return
  }
}
```

## Kernel programming model

```
module attributes {hacc.target = #hacc.target<"Ascend910B3">} {
  func.func @add_kernel(%arg0: memref<?xf16>, %arg1: memref<?xf16>, %arg2: memref<?xf16>)
                        attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, memref.memref_as_ptr} {
    hivm.hir.set_mask_norm
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16], strides: [1] : memref<?xf16> to memref<16xf16, strided<[1]>>
    %alloc = memref.alloc() : memref<16xf16>
    hivm.hir.load ins(%reinterpret_cast : memref<16xf16, strided<[1]>>) outs(%alloc : memref<16xf16>)
    %2 = bufferization.to_tensor %alloc restrict writable : memref<16xf16>

    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [16], strides: [1] : memref<?xf16> to memref<16xf16, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<16xf16>
    hivm.hir.load ins(%reinterpret_cast_0 : memref<16xf16, strided<[1]>>) outs(%alloc_1 : memref<16xf16>)
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<16xf16>

    %4 = tensor.empty() : tensor<16xf16>
    %5 = hivm.hir.vadd ins(%2, %3 : tensor<16xf16>, tensor<16xf16>) outs(%4 : tensor<16xf16>) -> tensor<16xf16>

    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [16], strides: [1] : memref<?xf16> to memref<16xf16, strided<[1]>>
    hivm.hir.store ins(%5 : tensor<16xf16>) outs(%reinterpret_cast_2 : memref<16xf16, strided<[1]>>)
    return
  }
}
```

```
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>, ...>>} {
  ...
}
```
