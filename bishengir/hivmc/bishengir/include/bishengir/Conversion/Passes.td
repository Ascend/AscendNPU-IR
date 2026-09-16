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
// HIVMToLLVM
//===----------------------------------------------------------------------===//

def ConvertHIVMToLLVM : Pass<"convert-hivm-to-llvm"> {
  let summary = "Convert HIVM dialect to LLVM dialect";
  let constructor = "mlir::createConvertHIVMToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "scf::SCFDialect", "mlir::hivm_regbaseintrins::HIVMRegbaseIntrinsDialect"];

  let options = [
      Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
             /*default=*/"false",
             "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
             "element types">,
      Option<"onDemandBarePtrCallConv", "on-demand-bare-ptr-memref-call-conv", "bool",
             /*default=*/"false",
             "Support fine-grained control over which FuncOp's MemRef argument "
             "will be lowered to bare pointers">
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


//===----------------------------------------------------------------------===//
// ConvertMemRefToBarePtr
//===----------------------------------------------------------------------===//

def ConvertMemRefToBarePtr : Pass<"convert-memref-to-bare-ptr"> {
  let summary = "Convert memref to base ptr";
  let constructor = "mlir::createConvertMemRefToBarePtrPass()";
  let dependentDialects = ["memref::MemRefDialect", "LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// TritonGlobalKernelArgsToLLVM
//===----------------------------------------------------------------------===//

def TritonGlobalKernelArgsToLLVM : Pass<"triton-globalkernelargs-to-llvm"> {
  let summary = "Handle the globalkernel args in the llvm dialect";
  let constructor = "mlir::createTritonGlobalKernelArgsToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// AppendUsePrintDebugData
//===----------------------------------------------------------------------===//

def AppendUsePrintDebugData : Pass<"append-use-print-debug-data"> {
  let summary = "Append args to the global kernel in the llvm dialect";
  let constructor = "mlir::createAppendUsePrintDebugDataPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// AppendSanitizerData
//===----------------------------------------------------------------------===//

  def AppendSanitizerData : Pass<"append-sanitizer-data"> {
    let summary = "Append args to the global kernel in the llvm dialect so "
                  "mssanitizer could use it";
    let constructor = "mlir::createAppendSanitizerDataPass()";
    let dependentDialects = ["LLVM::LLVMDialect"];
  }
//===----------------------------------------------------------------------===//
// HACCToLLVM
//===----------------------------------------------------------------------===//
def ConvertHACCToLLVM : Pass<"convert-hacc-to-llvm", "ModuleOp"> {
  let summary = "Convert HACC dialect to LLVM dialect";
  let constructor = "mlir::createConvertHACCToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "scf::SCFDialect"];
  let options = [
      Option<"tempDeviceLLVMFilePath", "device-llvm-file-path", "std::string",
             /*default=*/[{""}], "Pass in the path to device.so">
  ];
}

//===----------------------------------------------------------------------===//
// LowerMesh
//===----------------------------------------------------------------------===//
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
