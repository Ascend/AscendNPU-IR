//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_PASSES_H
#define BISHENGIR_CONVERSION_PASSES_H

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/ArithToHFusion/ArithToHFusion.h"
#include "bishengir/Conversion/ArithToHIVMLLVM/ArithToHIVMLLVM.h"
#include "bishengir/Conversion/GPUToHFusion/GPUToHFusion.h"
#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVMPass.h"
#include "bishengir/Conversion/HFusionToVector/HFusionToVector.h"
#include "bishengir/Conversion/HIVMToTritonGPU/HIVMToTritonGPU.h"
#include "bishengir/Conversion/LinalgToHFusion/LinalgToHFusion.h"
#include "bishengir/Conversion/LowerMemRefExt/LowerMemRefExt.h"
#include "bishengir/Conversion/LowerMesh/LowerMesh.h"
#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"
#include "bishengir/Conversion/ProtonAscendGPUToLLVM/ProtonAscendGPUToLLVM.h"
#include "bishengir/Conversion/TensorToHFusion/TensorToHFusion.h"
#include "bishengir/Conversion/TensorToHIVM/TensorToHIVM.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/TritonAscendGPUToLLVM.h"
#include "mlir/Pass/Pass.h"

#if BISHENGIR_ENABLE_TORCH_CONVERSIONS
#include "bishengir/Conversion/TorchToHFusion/TorchToHFusion.h"
#include "bishengir/Conversion/TorchToSymbol/TorchToSymbol.h"
#endif

namespace bishengir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Conversion/Passes.h.inc"

} // namespace bishengir

#endif // BISHENGIR_CONVERSION_PASSES_H
