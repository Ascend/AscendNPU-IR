// RUN: bishengir-opt --hoist-simt-scalar-calls-to-simd -split-input-file -verify-diagnostics %s

// -----

// A wrapper position that maps to a memref argument. A divisor is a scalar, so
// landing on a memref means the replayed expansion disagrees with the one
// FuncToTriton used.
module {
  module {
    func.func private @wrong_type_scope_0(memref<64xf32>, i32, memref<20xi8> {hivm.shared_memory}, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
    func.func @wrong_type(%buf: memref<64xf32>, %d: i32, %gx: i32, %gy: i32, %gz: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
      %alloc = memref.alloc() : memref<20xi8>
      // expected-error @below {{wrapper position 0 maps to argument 0, a 'memref<64xf32>', not a scalar}}
      call @wrong_type_scope_0(%buf, %d, %alloc, %gx, %gy, %gz) : (memref<64xf32>, i32, memref<20xi8>, i32, i32, i32) -> ()
      return
    }
  }
  module attributes {hacc.simt_module} {
    llvm.func @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32
    llvm.func @wrong_type_scope_0(%arg0: i32, %arg1: !llvm.ptr<6> {hivm.shared_memory}, %arg2: i32 {gpu.block = #gpu.block<x>}, %arg3: i32 {gpu.block = #gpu.block<y>}, %arg4: i32 {gpu.block = #gpu.block<z>}) attributes {hacc.entry} {
      %0 = llvm.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg0) : (i32) -> i32
      %1 = llvm.getelementptr %arg1[0] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      llvm.store %0, %1 {use_shmem_offset = 0 : i32} : i32, !llvm.ptr<6>
      llvm.return
    }
  }
}

// -----

// A wrapper position that maps to a scalar of the wrong width. The call
// operand is what the library call would take, so it must be i32.
module {
  module {
    func.func private @not_i32_scope_0(i64, memref<20xi8> {hivm.shared_memory}, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
    func.func @not_i32(%d: i64, %gx: i32, %gy: i32, %gz: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
      %alloc = memref.alloc() : memref<20xi8>
      // expected-error @below {{wrapper position 0 maps to argument 0, a 'i64' operand, not i32}}
      call @not_i32_scope_0(%d, %alloc, %gx, %gy, %gz) : (i64, memref<20xi8>, i32, i32, i32) -> ()
      return
    }
  }
  module attributes {hacc.simt_module} {
    llvm.func @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32
    llvm.func @not_i32_scope_0(%arg0: i32, %arg1: !llvm.ptr<6> {hivm.shared_memory}, %arg2: i32 {gpu.block = #gpu.block<x>}, %arg3: i32 {gpu.block = #gpu.block<y>}, %arg4: i32 {gpu.block = #gpu.block<z>}) attributes {hacc.entry} {
      %0 = llvm.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%arg0) : (i32) -> i32
      %1 = llvm.getelementptr %arg1[0] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      llvm.store %0, %1 {use_shmem_offset = 0 : i32} : i32, !llvm.ptr<6>
      llvm.return
    }
  }
}

// -----

// An operand that is neither a wrapper argument, a constant, nor an earlier
// scalar call result. Nothing in the caller corresponds to it.
module {
  module {
    func.func private @computed_scope_0(i32, memref<20xi8> {hivm.shared_memory}, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
    func.func @computed(%d: i32, %gx: i32, %gy: i32, %gz: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
      %alloc = memref.alloc() : memref<20xi8>
      call @computed_scope_0(%d, %alloc, %gx, %gy, %gz) : (i32, memref<20xi8>, i32, i32, i32) -> ()
      return
    }
  }
  module attributes {hacc.simt_module} {
    llvm.func @_mlir_ciface_simt_div_magic_shift_uint32_t(i32) -> i32
    llvm.func @computed_scope_0(%arg0: i32, %arg1: !llvm.ptr<6> {hivm.shared_memory}, %arg2: i32 {gpu.block = #gpu.block<x>}, %arg3: i32 {gpu.block = #gpu.block<y>}, %arg4: i32 {gpu.block = #gpu.block<z>}) attributes {hacc.entry} {
      %0 = llvm.add %arg0, %arg0 : i32
      %1 = llvm.call @_mlir_ciface_simt_div_magic_shift_uint32_t(%0) : (i32) -> i32
      %2 = llvm.getelementptr %arg1[0] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      // expected-error @below {{neither a wrapper argument, a constant, nor an earlier scalar call result}}
      llvm.store %1, %2 {use_shmem_offset = 0 : i32} : i32, !llvm.ptr<6>
      llvm.return
    }
  }
}

// -----

// A stamped store whose value no call produces: there is no callee to rebuild.
module {
  module {
    func.func private @no_call_scope_0(i32, memref<20xi8> {hivm.shared_memory}, i32, i32, i32) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMT>}
    func.func @no_call(%d: i32, %gx: i32, %gy: i32, %gz: i32) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
      %alloc = memref.alloc() : memref<20xi8>
      call @no_call_scope_0(%d, %alloc, %gx, %gy, %gz) : (i32, memref<20xi8>, i32, i32, i32) -> ()
      return
    }
  }
  module attributes {hacc.simt_module} {
    llvm.func @no_call_scope_0(%arg0: i32, %arg1: !llvm.ptr<6> {hivm.shared_memory}, %arg2: i32 {gpu.block = #gpu.block<x>}, %arg3: i32 {gpu.block = #gpu.block<y>}, %arg4: i32 {gpu.block = #gpu.block<z>}) attributes {hacc.entry} {
      %0 = llvm.getelementptr %arg1[0] : (!llvm.ptr<6>) -> !llvm.ptr<6>, i8
      // expected-error @below {{stamped store's value is not produced by an llvm.call}}
      llvm.store %arg0, %0 {use_shmem_offset = 0 : i32} : i32, !llvm.ptr<6>
      llvm.return
    }
  }
}
