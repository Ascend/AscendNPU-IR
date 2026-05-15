//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def DecomposeFRem : Pass<"decompose-frem", "ModuleOp"> {
  let summary = "Decompose llvm.frem to sub(mul(trunc(div)))";
  let constructor = "bishengir::triton::createDecomposeFRemPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "hivm::HIVMDialect", "hivm_regbaseintrins::HIVMRegbaseIntrinsDialect"];
}


#ifdef BISHENGIR_ENABLE_TRITON_COMPILE

def SetBishengirSimtOptAttr
    : Pass<"set-bishengir-simt-opt-attr", "mlir::ModuleOp"> {
  let summary = "Set BishengIR SIMT optimization module attribute";
  let description = [{
    Sets the module attribute controlling BishengIR SIMT optimization.
  }];
  let constructor = "bishengir::triton::createSetBishengirSimtOptAttrPass()";

  let options = [
    Option<"enableBishengirSimtOptimization", "enable-bishengir-simt-optimization",
           "int32_t", /*default*/"0",
           "enable optimization for simt">,
  ];
}

def AdaptTritonIRKernel
   : Pass<"adapt-triton-ir-kernel", "mlir::triton::FuncOp"> {
  let constructor = "bishengir::triton::createAdaptTritonIRKernelPass()";
  let dependentDialects = [
    "gpu::GPUDialect",
  ];
}

def SIMTAutoBlockify
   : Pass<"simt-auto-blockify", "mlir::triton::FuncOp"> {
  let summary = "Blockify Triton SIMT kernels when logical grid is larger than the physical core count";
  let description = [{
    This pass wraps the original Triton kernel body with an outer `scf.for`
    loop so one hardware wave of physical cores can process multiple logical
    programs in batches.
  }];
  let constructor = "bishengir::triton::createSIMTAutoBlockifyPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "scf::SCFDialect",
    "mlir::triton::TritonDialect",
  ];
}

def EnableAscendDPXMMA
   : Pass<"enable-ascend-dpx-mma", "mlir::triton::FuncOp"> {
  let constructor = "bishengir::triton::createEnableAscendDPXMMAPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];
}

def OptimizeLoads
   : Pass<"optimize-loads", "mlir::ModuleOp"> {
  let constructor = "bishengir::triton::createOptimizeLoadsPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
  ];
}

def LoopRestructureArangeOptimization
   : Pass<"loop-restructure-arange-optimization", "mlir::ModuleOp"> {
  let constructor = "bishengir::triton::createLoopRestructureArangeOptimizationPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
  ];
}

def DecomposeReduction
   : Pass<"decompose-reduction", "mlir::ModuleOp"> {
  let constructor = "bishengir::triton::createDecomposeReductionPass()";
  let dependentDialects = [
      "mlir::gpu::GPUDialect",
    ];
}

def LegalizeF16ForTriton
   : Pass<"legalize-f16-for-triton", "mlir::triton::FuncOp"> {
  let constructor = "bishengir::triton::createLegalizeF16ForTritonPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
  ];
}

def OptimizeLayouts
   : Pass<"optimize-layouts", "mlir::triton::FuncOp"> {
  let constructor = "bishengir::triton::createOptimizeLayoutsPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
  ];
}

def DumpFractalLayout : Pass<"dump-fractal-layout", "mlir::ModuleOp"> {
  let summary = "Print a fractal shared memory layout for a given shape and offset";
  let description = [{
    Computes the fractal linear layout (zN or nZ) for the given tensor shape
    and fractal block sizes, then prints the logical coordinate (dim0, dim1)
    that corresponds to a probed element offset.  Useful for verifying
    fractalSharedToLinearLayout in LIT / FileCheck tests.

    Example:
      bishengir-opt --dump-fractal-layout="shape=64,64 layout-type=zN probe-offset=1536" %s
    Output:
      fractal-layout: offset=1536 -> dim0=32 dim1=16
  }];
  let constructor = "bishengir::triton::createDumpFractalLayoutPass()";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "bishengir::triton_ext::TritonExtDialect"
  ];
  let options = [
    Option<"shapeStr", "shape", "std::string", "\"64,64\"",
           "Tensor shape as comma-separated integers, e.g. '64,64'">,
    Option<"fractalM0", "fractal-m0", "int64_t", /*default*/"16",
           "Fractal block height (M dimension)">,
    Option<"fractalN0", "fractal-n0", "int64_t", /*default*/"16",
           "Fractal block width (N dimension)">,
    Option<"layoutTypeStr", "layout-type", "std::string", "\"zN\"",
           "Fractal layout type: 'zN' or 'nZ'">,
    Option<"probeOffset", "probe-offset", "int64_t", /*default*/"0",
           "Element offset to probe (not byte offset)">,
  ];
}

def FixFusedCat: Pass<"fix-fused-cat", "mlir::ModuleOp"> {
  let summary = "fix fused cat kernels";
  let constructor = "bishengir::triton::createFixFusedCatPass()";
  let description = [{
    This pass fixes some fused cat kernels generated by Huawei inductor
    which uses tensor.extract_slice/insert_slice
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                            "mlir::tensor::TensorDialect",
                            "mlir::triton::TritonDialect"];
}

def RewriteSliceOpToTriton
    : Pass<"rewrite-slice-op-to-triton", "mlir::ModuleOp"> {
  let summary = "rewrite tensor.extract_slice/insert_slice into tt.trans/reshape/split/join";
  let constructor = "bishengir::triton::createRewriteSliceOpToTritonPass()";
  let description = [{
    Replace `tensor.extract_slice` and `tensor.insert_slice` with sequences
    of Triton dialect ops (`tt.trans`, `tt.reshape`, `tt.split`, `tt.join`).
    Only the pattern of indexing a single contiguous block along *one*
    axis `a` (any position) of an N-D tensor is handled; every other axis
    must be passed through in full:

      offsets[a] = r       offsets[i!=a] = 0
      sizes[a]   = S       sizes[i!=a]   = D_i
      strides    = [1, 1, ..., 1]
      large : tensor<D0 x ... x D_a (= N) x ... x D_{R-1} x T>
      small : tensor<D0 x ... x        S  x ... x D_{R-1} x T>

    For `extract_slice`, large is the source and small is the result.
    For `insert_slice`, large is the dest and small is the source.

    Constraints:
      * every dim of the large tensor is a power of two,
      * S is a power of two with 1 <= S <= N,
      * r is static, in [0, N - S], and a multiple of S.

    Any slice op that does not match this shape (including multi-axis
    slicing or non-aligned offsets) causes the pass to fail.
  }];

  let dependentDialects = ["mlir::tensor::TensorDialect",
                            "mlir::triton::TritonDialect"];
}

def PopulateSharedMemoryOffsetToDPX : Pass<"populate-shared-memory-offset-to-dpx", "mlir::triton::FuncOp"> {
  let summary = "Convert tt.call_scalar to ascend_dpx.call_scalar with allocation offset";
  let constructor = "bishengir::triton::createPopulateSharedMemoryOffsetToDPXPass()";
  let dependentDialects = ["mlir::ascend_dpx::AscendDPXDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def SIMTFastDiv : Pass<"simt-fast-div", "mlir::triton::FuncOp"> {
  let summary = "SIMT fast division optimization";
  let description = [{
    Optimizes division operations for SIMT execution.
  }];
  let constructor = "bishengir::triton::createSIMTFastDivPass()";
  let dependentDialects = ["mlir::ascend_dpx::AscendDPXDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect"];
}

def ConvertSharedPtrToMemDesc
    : Pass<"convert-shared-ptr-to-memdesc", "mlir::ModuleOp"> {
  let summary = "Convert tt.load/tt.store with ptr<6> to ttg.local_load/local_store with memdesc";
  let description = [{
    Runs after ConvertTritonToTritonGPU.  Finds tt.load / tt.store operations
    whose pointer tensors use address-space 6 (shared memory / UB) and rewrites
    them to ttg.local_load / ttg.local_store with ttg.memdesc types.  The
    corresponding function arguments are changed from !tt.ptr<T, 6> to
    !ttg.memdesc<...>.
  }];
  let constructor = "bishengir::triton::createConvertSharedPtrToMemDescPass()";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::TritonDialect",
    "bishengir::triton_ext::TritonExtDialect"
  ];
}

def FlattenMemDescArgs
    : Pass<"flatten-memdesc-args", "mlir::ModuleOp"> {
  let summary = "Replace memdesc struct function args with bare pointers";
  let description = [{
    After ConvertTritonAscendGPUToLLVM, memdesc function arguments become
    llvm.struct<(ptr<AS>, i32, ...)>.  This pass replaces each such argument
    with the bare pointer, rewriting extractvalue %arg[0] to use the pointer
    directly and erroring on extractvalue %arg[N>0].
  }];
  let constructor = "bishengir::triton::createFlattenMemDescArgsPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
  ];
}

def GetTritonMetadata :
    Pass<"get-triton-metadata", "mlir::ModuleOp"> {
  let summary = "Get triton metadata";

  let description = [{
    Get triton metadata, e.g, ttg.total-num-warps, ttg.shared.
    Print to either stdout or file.
  }];

  let constructor = "bishengir::triton::createGetTritonMetadataPass()";

  let options = [Option<"tritonMetadataOutput",
                        "triton-metadata-output", "std::string", "",
                        "File to dump triton metadata. -- means stdout">];
}

def AdaptGPUKernel : Pass<"adapt-gpu-kernel", "ModuleOp"> {
  let summary = "Generate HIVM SIMT wrappers and attributes for a GPU/SIMT kernel";
  let constructor = "bishengir::triton::createAdaptGPUKernelPass()";
  let dependentDialects = [
    "hacc::HACCDialect", "ascend_dpx::AscendDPXDialect", "hivm_regbaseintrins::HIVMRegbaseIntrinsDialect", "LLVM::LLVMDialect", "hivm::HIVMDialect"
  ];
}

def TritonRemap : Pass<"triton-remap", "mlir::ModuleOp"> {
  let summary = "Converts triton-generated LLVM IR into NPU-compatible LLVM IR.";
  let constructor = "bishengir::triton::createTritonRemapPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect", "mlir::hivm::HIVMDialect", "mlir::hivm_regbaseintrins::HIVMRegbaseIntrinsDialect",
    "mlir::hacc::HACCDialect", "mlir::NVVM::NVVMDialect", "mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"gridDimX", "dim-x", "int", "1", "Grid dimension for x axis">,
    Option<"gridDimY", "dim-y", "int", "1", "Grid dimension for y axis">,
    Option<"gridDimZ", "dim-z", "int", "1", "Grid dimension for z axis">,
    Option<"useIfElse", "use-if-else", "bool", "true", "Replace inline asm load/stores with scf and llvm.load/store">,
    Option<"useGridFlag", "use-grid-flag", "bool", "false", "Override the argument dim x,y,z with the dim given by flag">,
    Option<"blockified", "enabled-blockify", "bool", "false", "Indicates that blockify is enabled">,
    Option<"isSimdSimtMixCompile", "is-simd-simt-mix-compile", "bool", "false",
           "Whether the SIMT kernel is lowered from the SIMD-SIMT mixed pipeline">
  ];
}
#endif

#endif // BISHENGIR_DIALECT_TRITON_TRANSFORMS_PASSES_TD
