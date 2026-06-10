module {
  func.func @ub_bubble_up() -> tensor<8x16xf32> attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, mix_mode = "mix"} {
    %alloc_0 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    annotation.mark %alloc_0 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<16x16xf32, #hivm.address_space<ub>>
    %memspacecast = memref.memory_space_cast %alloc_0 : memref<16x16xf32, #hivm.address_space<ub>> to memref<16x16xf32>
    %to_tensor = bufferization.to_tensor %memspacecast restrict writable : memref<16x16xf32>
    %slice = tensor.extract_slice %to_tensor[0, 0] [8, 16] [1, 1] {to_be_bubbled_slice} : tensor<16x16xf32> to tensor<8x16xf32>
    return %slice : tensor<8x16xf32>
  }
}
