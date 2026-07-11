//===- HIVMPasses.td - Conversion pass definition file -----*- tablegen -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect", "hfusion::HFusionDialect", "bufferization::BufferizationDialect"];
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
    )}]>,
      Option<"isEnableUbufSaving", "is-enable-ubuf-saving", "bool", /*default*/"false",
             "enable low ubuf usage mode ">,
      Option<"isDisableSizeAlignForCast", "is-disable-size-align-for-cast",
             "bool", /*default*/"false",
             "disable VCast alloc size alignment and use new temp buffer scheme (s32/s16->s8)">
  ];
}


//===----------------------------------------------------------------------===//
// HIVMToStandard
//===----------------------------------------------------------------------===//

def ConvertHIVMToStandard : Pass<"convert-hivm-to-std", "ModuleOp"> {
  let summary = "Convert the operations from the hivm dialect into the "
                "Standard dialect";
  let constructor = "mlir::createConvertHIVMToStandardPass()";
  let dependentDialects =
      ["func::FuncDialect", "memref::MemRefDialect", "scf::SCFDialect", "LLVM::LLVMDialect"];
  let options = [
        Option<"isOpsAligned", "is-ops-aligned", "bool", /*default*/"false",
               "Regard All Ops already dims aligned except last axis of "
               "brc and reduce">
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
// LowerMemRefExt
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// VectorToHIVMAVE
//===----------------------------------------------------------------------===//

def ConvertVectorToHIVMAVE : Pass<"convert-vector-to-hivmave", "ModuleOp"> {
  let summary = "Convert vector operations to ave dialect";
  let description = [{
    This pass converts vector operations to ave regbased ops.
  }];
  let constructor = "mlir::createVectorToHIVMAVEConversionPass()";
  let dependentDialects = [
    "hivmave::AVEDialect",
    "vector::VectorDialect"
  ];
  let options = [Option<"enableTritonCompile", "enable-triton-compile",
                        "bool", "false", "Enable Triton compile.">
  ];
}

//===----------------------------------------------------------------------===//
// HIVMAVEToStandard
//===----------------------------------------------------------------------===//

def ConvertHIVMAVEToStandard : Pass<"convert-hivmave-to-std", "ModuleOp"> {
  let summary = "Convert the operations from the hivmave dialect into the Standard dialect";
  let constructor = "mlir::createConvertHIVMAVEToStandardPass()";
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect", "scf::SCFDialect", "LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// ArithToHIVMAVE
//===----------------------------------------------------------------------===//

def ConvertArithToHIVMAVE : Pass<"convert-arith-to-hivmave", "func::FuncOp"> {
  let summary = "Convert arith operations to ave regbased ops";
  let description = [{
    This pass converts arith operations to ave regbased ops.
  }];
  let constructor = "mlir::createArithToHIVMAVEConversionPass()";
  let dependentDialects = [
    "hivmave::AVEDialect",
    "arith::ArithDialect",
    "hivm::HIVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// HIVMAVEToAVEIntrin
//===----------------------------------------------------------------------===//

def ConvertHIVMAVEToAVEIntrin : Pass<"convert-hivmave-to-ave-intrin", "ModuleOp"> {
  let summary = "Convert the operations from hivmave ops into ave intrins op";
  let constructor = "mlir::createConvertHIVMAVEToAVEIntrinPass()";
  let dependentDialects =
      ["LLVM::LLVMDialect", "scf::SCFDialect", "hivmave::AVEDialect",
       "hivm_regbaseintrins::HIVMRegbaseIntrinsDialect"];
  let options = [
      Option<"useUnsignedCmpFor", "use-unsigned-cmp-for", "bool", /*default*/"true",
             "use 'scf.for' with 'unsignedCmp'">,
      Option<"enableI16IndVar", "enable-i16-indvar", "bool", /*default*/"false",
             "enable 'i16' induction variables">
  ];
}

#endif // BISHENGIR_CONVERSION_PASSES_H
