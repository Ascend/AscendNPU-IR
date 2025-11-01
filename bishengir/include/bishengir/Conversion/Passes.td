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
             "enable low ubuf usage mode ">
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
