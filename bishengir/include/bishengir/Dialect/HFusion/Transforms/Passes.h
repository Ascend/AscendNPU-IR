//===- Passes.h - HFusion dialect pass entrypoints --------------*- C++ -*-===//
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
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_PASSES_H

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
} // namespace func

namespace hfusion {
namespace opfusion {
class FusibleHelper;
class FusibleBlock;
using FusibleBlocks = SmallVector<FusibleBlock, 8>;
} // namespace opfusion

} // namespace hfusion
} // namespace mlir

namespace mlir {

#define GEN_PASS_DECL
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

namespace hfusion {

class OutlineFuncOptions : public HFusionOpFusionOptions {
public:
  OutlineFuncOptions(const HFusionOpFusionOptions &options, func::FuncOp funcOp)
      : HFusionOpFusionOptions(options),
        shouldKeepFuncSignature(!options.enableMultiKernel &&
                                hacc::utils::isHost(funcOp)) {}

  // if shouldKeepFuncSignature is true:
  // 1. the outlined func should have equal function signature as original
  // function, which is important when the host need to make sure the outliner
  // will not break the calling conventions example of case that it handles:
  // original host kernel:
  // func kernel(a, b) -> (c, c, c)
  //
  // kernel launch:
  // call kernel(a, b, c0, c1, c2)
  //
  // outlined kernel:
  // func outlined_kernel(a, b) -> (c)
  // number of args mismatch because duplicate outputs will be returned only one
  // and it causes core dump
  //
  // 2. to avoid the outliner to optimize multiple same output values as a
  // single output
  const bool shouldKeepFuncSignature;
};

/// Given a funcOp, try to outline fusible ops into functions with options and
/// return the outlined functions by vector
opfusion::FusibleBlocks
getFusibleBlocks(func::FuncOp func, opfusion::FusibleHelper &fusibleHelper);

/// Given a funcOp, try to outline fusible ops into functions with options and
/// return the outlined functions by vector
LogicalResult outlineFusedFuncs(func::FuncOp entryFunc,
                                const HFusionOpFusionOptions &options,
                                SmallVector<func::FuncOp> &outlinedFuncs);

LogicalResult outlineSingleFusedFuncs(func::FuncOp entryFunc,
                                      const HFusionOpFusionOptions &options,
                                      SmallVector<func::FuncOp> &outlinedFuncs);

/// Create a pass to fuse operations into outlined functions.
std::unique_ptr<mlir::Pass>
createHFusionOpFusionPass(const HFusionOpFusionOptions &options = {});

/// Create a pass to auto schedule fused kernels.
std::unique_ptr<Pass>
createHFusionAutoSchedulePass(const AutoScheduleOptions &options = {});

/// Create a pass to execute auto schedule sequence for the target kernel.
std::unique_ptr<Pass>
createAutoScheduleInterpreterPass(const std::string &kernelName,
                                  transform::TransformOptions options = {});

/// Create a pass to erase auto schedule sequence for the target kernel.
std::unique_ptr<Pass>
createEraseAutoSchedulePass(const std::string &kernelName);

/// Create a pass to add ffts base address to func param and annotation
std::unique_ptr<Pass>
createAddFFTSAddrPass(const AddFFTSAddrOptions &options = {});

/// Create a pass to lianlg generic ops to named ops
std::unique_ptr<Pass> createConvertGenericToNamedOpPass();

/// Create a pass to flatten linalg and hfusion ops.
std::unique_ptr<Pass>
createFlattenOpsPass(const FlattenOpsOptions &options = {});

/// Create a pass to move output tensor results' tied init operand to function
/// parameters.
std::unique_ptr<Pass>
createTensorResToOutParamsPass(const TensorResToOutParamsOptions &options = {});

/// Create a pass to outline single linalg op.
std::unique_ptr<Pass>
createOutlineSingleOpPass(const OutlineSingleOpOptions &options = {});

/// Create a pass to simplify operations.
std::unique_ptr<Pass> createSimplifyOpsPass();

/// Create a pass to normalize operations.
std::unique_ptr<Pass> createHFusionNormalizeOpsPass();

/// Create a pass to normalize slice operations, including
/// extract_slice/insert_slice.
std::unique_ptr<Pass>
createHFusionNormalizeSliceOpsPass(bool skipAlignedSlice = false);

/// Create a pass to inline broadcast-like op
std::unique_ptr<Pass> createHFusionInlineBrcPass();

/// Create a pass to pack tiling data.
std::unique_ptr<Pass>
createPackTilingDataPass(const PackTilingDataOptions &options = {});

/// Create a pass to constantize tiling data.
std::unique_ptr<Pass> createConstantizeTilingDataPass();

/// Create a pass to label the triton entry kernel
std::unique_ptr<Pass>
createAdaptTritonKernelPass(std::string hivmcVersion = "");

/// Create a pass to infer func fusion kind
std::unique_ptr<Pass> createInferFuncFusionKind();

// Create a pass to generate out tensor's shape function
std::unique_ptr<Pass> createInferOutShapesPass();

/// Create a pass to legalize bf16 type
std::unique_ptr<Pass> createLegalizeBF16Pass();

/// Create a pass to legalize bool
std::unique_ptr<Pass> createLegalizeBoolPass();

/// create a pass to reorder hfusion ops by bfs
std::unique_ptr<Pass> createReorderOpsByBFS();

/// Create a pass to downgrade FP64 constants to FP32
std::unique_ptr<Pass> createDowngradeFP64CstOpPass();

// create compose and decompose multi reduce opt pass
std::unique_ptr<Pass>
createComposeMultiReduce(const ComposeMultiReduceOptions &options = {});

std::unique_ptr<Pass> createDecomposeMulti();

/// create a pass to cache io arguments
std::unique_ptr<Pass> createCacheIO();

/// create a pass to cache io for arguments that returns directly
std::unique_ptr<Pass> createCacheIOForReturnArg();

/// create a pass to recache io
std::unique_ptr<Pass> createReCacheIO();

/// Create a pass to convert hoist tensor empty to func parameters and merge
/// into one parameter.
std::unique_ptr<Pass> createHoistTensorEmptyPass();

/// Create a pass to create wrappers for certain host related functions
std::unique_ptr<Pass>
createWrapHostFuncPass(const WrapHostFuncOptions &options = {});

/// Create passes to fold and unfold symbolic dim for easier fusion
std::unique_ptr<Pass> createFoldSymbolicDimPass();
std::unique_ptr<Pass> createUnfoldSymbolicDimPass();
std::unique_ptr<Pass> createDropSymbolsPass();

/// Create a pass to decompose ops that implemented AggregatedOpInterface.
std::unique_ptr<Pass> createDecomposePass(const DecomposeOptions &options = {});

/// Create a pass to eliminate duplicate functions.
std::unique_ptr<Pass> createEliminateDuplicateFuncsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"

/// Register a pass to execute auto schedule sequence for the target kernel.
void registerAutoScheduleInterpreterPass();

/// Register a pass to erase auto schedule sequence for the target kernel.
void registerEraseAutoSchedulePass();

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_PASSES_H
