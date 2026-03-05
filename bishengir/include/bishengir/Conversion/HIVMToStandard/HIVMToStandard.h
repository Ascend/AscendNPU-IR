//===- HIVMToStandard.h - Convert HIVM dialect to Standard dialect --------===//
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

#ifndef BISHENGIR_CONVERSION_HIVMTOSTANDARD_HIVMTOSTANDARD_H_
#define BISHENGIR_CONVERSION_HIVMTOSTANDARD_HIVMTOSTANDARD_H_

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

#define GEN_PASS_DECL_CONVERTHIVMTOSTANDARD
#include "bishengir/Conversion/Passes.h.inc"

namespace hivm {
/// Populate the given list with patterns that convert from HIVM to Standard.
void populateHIVMToStandardConversionPatterns(RewritePatternSet &patterns,
                                              bool isOpsAligned = false);
} // namespace hivm

/// Create a pass to convert HIVM operations to the Standard dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertHIVMToStandardPass(
    const ConvertHIVMToStandardOptions &options = {});

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HIVMTOSTANDARD_HIVMTOSTANDARD_H_
