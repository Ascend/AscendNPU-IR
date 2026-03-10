//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "bishengir/Config/bishengir-config.h"
#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/ArithToHFusion/ArithToHFusion.h"
#include "bishengir/Conversion/GPUToHFusion/GPUToHFusion.h"
#include "bishengir/Conversion/HFusionToHIVM/HFusionToHIVMPass.h"
#include "bishengir/Conversion/HIVMToStandard/HIVMToStandard.h"
#include "bishengir/Conversion/LinalgToHFusion/LinalgToHFusion.h"
#include "bishengir/Conversion/LowerMemRefExt/LowerMemRefExt.h"
#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"
#include "bishengir/Conversion/TensorToHFusion/TensorToHFusion.h"
#include "bishengir/Conversion/TensorToHIVM/TensorToHIVM.h"
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
