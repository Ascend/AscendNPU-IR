//===- HIVMPasses.td - Conversion pass definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_PASSES_H
#define BISHENGIR_CONVERSION_PASSES_H

include "mlir/Pass/PassBase.td"
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"

//===----------------------------------------------------------------------===//
// ArithToAffine
//===----------------------------------------------------------------------===//

def ConvertArithToAffine : Pass<"convert-arith-to-affine"> {
  let summary = "Convert arith ops to affine ops";
  let constructor = "mlir::createArithToAffineConversionPass()";
  let dependentDialects = ["arith::ArithDialect", "affine::AffineDialect"];
}

//===----------------------------------------------------------------------===//
// ArithToHFusion
//===----------------------------------------------------------------------===//

def ConvertArithToHFusion : Pass<"convert-arith-to-hfusion"> {
  let summary = "Convert arith ops to HFusion ops";
  let constructor = "mlir::createArithToHFusionConversionPass()";
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect", "hfusion::HFusionDialect"];
}

//===----------------------------------------------------------------------===//
// MathToHFusion
//===----------------------------------------------------------------------===//

def ConvertMathToHFusion : Pass<"convert-math-to-hfusion"> {
  let summary = "Convert math ops to HFusion ops";
  let constructor = "mlir::createMathToHFusionConversionPass()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToHFusion
//===----------------------------------------------------------------------===//

def ConvertLinalgToHFusion : Pass<"convert-linalg-to-hfusion"> {
  let summary = "Convert linalg ops to HFusion ops";
  let constructor = "mlir::createLinalgToHFusionConversionPass()";
  let dependentDialects = [
    "arith::ArithDialect", 
    "linalg::LinalgDialect", 
    "hfusion::HFusionDialect", 
    "bufferization::BufferizationDialect"];
}

//===----------------------------------------------------------------------===//
// GPUToHFusion
//===----------------------------------------------------------------------===//

def ConvertGPUToHFusion : Pass<"convert-gpu-to-hfusion"> {
  let summary = "Convert GPU ops to HFusion ops";
  let constructor = "mlir::createGPUToHFusionConversionPass()";
  let dependentDialects = ["gpu::GPUDialect", "hfusion::HFusionDialect"];
}

//===----------------------------------------------------------------------===//
// ArithToHIVMLLVM
//===----------------------------------------------------------------------===//

def ArithToHIVMLLVMConversionPass: Pass<"convert-arith-to-hivm-llvm"> {
  let summary = "Partial convert arith to llvm for double buffer in backend";
  let description = [{
    This pass convert arith::select to llvm::select with pointer as operand as
    opposed to memref descriptor in the ArithToLLVM conversion pass. This is the
    first step to help the auto-sync in backend to reconginze double buffer. This
    pass is very machine specific so it is not stored in conversion pass folder.
  }];
  let constructor = "mlir::createArithToHIVMLLVMConversionPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "hivm::HIVMDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// HFusionToHIVM
//===----------------------------------------------------------------------===//

def ConvertHFusionToHIVM : Pass<"convert-hfusion-to-hivm"> {
  let summary = "Convert HFusion ops to HIVM ops";
  let constructor = "mlir::createHFusionToHIVMConversionPass()";
  let dependentDialects = ["linalg::LinalgDialect", "hivm::HIVMDialect"];
  let options = [
      Option<"mmMapMode", "mm-map-mode", "mlir::hfusion::MmMapMode", "mlir::hfusion::MmMapMode::CoreOp",
             "matmul map mode, default is CoreOp",
             [{::llvm::cl::values(
        clEnumValN(mlir::hfusion::MmMapMode::CoreOp, "core_op",
                   "covnert to gm buffer based matmul"),
        clEnumValN(mlir::hfusion::MmMapMode::MacroInstr, "macro_instr",
                   "covnert to local buffer based matmul")
    )}]>
  ];
}

//===----------------------------------------------------------------------===//
// HIVMToTritonGPU
//===----------------------------------------------------------------------===//
 
def ConvertHIVMToTritonGPU : Pass<"convert-hivm-to-tritongpu"> {
  let summary = "Convert hivm ops to triton ops";
  let constructor = "mlir::createHIVMToTritonGPUConversionPass()";
  let dependentDialects = ["mlir::triton::TritonDialect", "memref::MemRefDialect", "scf::SCFDialect"];
  let options = [
    Option<"allowReturnValue", "allow-return-value", "bool",
           /*default=*/"false",
           "When allow return value, the FuncOpPattern will not work."
             "This option is only for testing purposes, because in formal"
             "scenarios, the simt vf does not support return values.">,
  ];
}

//===----------------------------------------------------------------------===//
// TensorToHFusion
//===----------------------------------------------------------------------===//

def ConvertTensorToHFusion : Pass<"convert-tensor-to-hfusion"> {
  let summary = "Convert tensor ops to Linalg/HFusion ops";
  let constructor = "mlir::createTensorToHFusionConversionPass()";
  let dependentDialects = ["tensor::TensorDialect", "hfusion::HFusionDialect",
                          "linalg::LinalgDialect"];
}

//===----------------------------------------------------------------------===//
// TensorToHIVM
//===----------------------------------------------------------------------===//

def ConvertTensorToHIVM : Pass<"convert-tensor-to-hivm"> {
  let summary = "Convert tensor ops to HIVM ops";
  let constructor = "mlir::createTensorToHIVMConversionPass()";
  let dependentDialects = ["tensor::TensorDialect", "hivm::HIVMDialect"];
}


def LowerMesh : Pass<"lower-mesh"> {
  let summary = "Lowers mesh dialect collective operations to appropriate "
                "calls to LCCL/HCCL.";
  let description = [{
    Lowers Mesh dialect collectives to calls to external helper functions
    that calls LCCL/HCCL.
  }];
  let constructor = "mlir::createMeshLoweringPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "bufferization::BufferizationDialect",
    "hivm::HIVMDialect",
    "tensor::TensorDialect",
    "memref::MemRefDialect",
  ];
  let options = [Option<"target", "target", "bishengir::mesh::TargetLib",
                        /*default=*/"bishengir::mesh::TargetLib::HCCL",
                        "Target to lower to, options are hccl or lccl.",
                        [{::llvm::cl::values(
          clEnumValN(bishengir::mesh::TargetLib::HCCL, "hccl",
                     "invoke HCCL for collectives"),
          clEnumValN(bishengir::mesh::TargetLib::LCCL, "lccl",
                     "invoke HCCL for collectives"))}]>];
}


//===----------------------------------------------------------------------===//
// HFusionToVector
//===----------------------------------------------------------------------===//
def ConvertHFusionToVector : Pass<"convert-hfusion-to-vector", "func::FuncOp"> {
  let summary = "Convert hfusion operations to vector regbased ops and add vector::transfer_read/write";
  let description = [{
    This pass converts hfusion operations to vector regbased ops and add
    vector::transfer_read/write to load data into registers as these ops cant be vectorize in normal ways.
  }];
  let constructor = "mlir::createHFusionToVectorConversionPass()";
  let dependentDialects = [
    "vector::VectorDialect",
    "hivm::HIVMDialect",
    "hfusion::HFusionDialect",
  ];
}

def LowerMemRefExt : Pass<"lower-memref-ext", "ModuleOp"> {
  let summary = "Lower memref ext dialect to memref dialect";
  let constructor = "mlir::createMemrefExtLoweringPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "hivm::HIVMDialect",
    "memref::MemRefDialect",
    "bishengir::memref_ext::MemRefExtDialect",
  ];
}

def ConvertTritonAscendGPUToLLVM : Pass<"convert-triton-ascend-gpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert TritonAscendGPU to LLVM";
  let description = [{
    Convert TritonAscendGPU operations to LLVM dialect.
  }];
  let constructor = "mlir::triton::createConvertTritonAscendGPUToLLVMPass()";
  
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::ascend_dpx::AscendDPXDialect"
  ];
}

def ConvertProtonAscendGPUToLLVM :
    Pass<"convert-proton-ascend-gpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert ProtonGPU to LLVM";
  let description = [{
    Convert ProtonGPU to LLVM using Ascend-specific lowering patterns.
  }];
  let constructor = "bishengir::triton::proton::gpu::createConvertProtonAscendGPUToLLVMPass()";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::ascend_dpx::AscendDPXDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::proton::ProtonDialect",
    "mlir::triton::proton::gpu::ProtonGPUDialect"
  ];
}

def AllocateProtonAscendGlobalScratchBufferPass :
    Pass<"allocate-proton-ascend-global-scratch-buffer", "mlir::ModuleOp"> {
  let summary = "Update metadata for proton global scratch buffer allocation (Ascend)";
  let description = [{
    Ascend-specific version of the proton global scratch buffer allocation pass.
    This pass updates the amount of global memory used by proton intra kernel
    profiling, using NVVM kernel attributes for function identification.
  }];
  let constructor = "bishengir::triton::proton::gpu::createAllocateProtonAscendGlobalScratchBufferPass()";

  let dependentDialects = [
    "mlir::triton::proton::ProtonDialect",
    "mlir::triton::proton::gpu::ProtonGPUDialect"
  ];
}


#ifdef BISHENGIR_ENABLE_TORCH_CONVERSIONS
//===----------------------------------------------------------------------===//
// Torch conversions
//===----------------------------------------------------------------------===//

def ConvertTorchToHFusion : Pass<"convert-torch-to-hfusion", "func::FuncOp"> {
  let summary = "Convert recognized Torch ops to Linalg or HFusion named ops";
  let description = [{
    Convert ATen ops to linalg or hfusion named ops.
  }];
  let constructor = "mlir::createConvertTorchToHFusionPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "math::MathDialect",
    "func::FuncDialect",
    "tensor::TensorDialect",
    "arith::ArithDialect",
    "hfusion::HFusionDialect"
  ];
  let options = [
    Option<"ensureNoImplicitBroadcast", "ensure-no-implicit-broadcast",
           "bool", /*default=*/"false",
           "Whether to ensure that there is no implicit broadcast semantics."
           "If there is a dynamic to dynamic dim broadcast, raise a runtime error."
    >,
  ];
}

def ConvertTorchToSymbol : Pass<"convert-torch-to-symbol", "func::FuncOp"> {
  let summary = "Convert Torch symbolic operations to Symbol dialect";
  let description = [{
    Convert Torch dialect symbolic operations (like torch.symbolic_int) to
    corresponding operations in the Symbol dialect. This pass handles the
    lowering of torch symbolic operations that represent dynamic dimensions
    and symbolic values to a more generic symbol representation.

    The pass specifically targets operations like:
    - torch.symbolic_int -> symbol.symbolic_int

    This conversion enables further optimizations and lowering passes that
    work with the Symbol dialect.
  }];
  let constructor = "mlir::createConvertTorchToSymbolPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "tensor::TensorDialect",
    "symbol::SymbolDialect",
    "arith::ArithDialect"
  ];
}
#endif

#endif // BISHENGIR_CONVERSION_PASSES_H
