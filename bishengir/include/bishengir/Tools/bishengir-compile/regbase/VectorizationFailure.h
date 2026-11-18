//===- VectorizationFailure.h - Vectorization retry diagnostics -*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_VECTORIZATIONFAILURE_H
#define BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_VECTORIZATIONFAILURE_H

#include "llvm/ADT/StringRef.h"

namespace bishengir {
namespace regbase {

/// True when \p message is a failure from the HFusion vectorization pipeline
/// (autovectorize, outline, verifier, TreeReduce, VFFusion, manual vectorize).
/// The in-pass AutoVectorizeV2 warning that still says "retrying" is not a
/// failure: that attempt recovered.
inline bool isVectorizationPipelineFailure(llvm::StringRef message) {
  // The in-pass warning that still says "retrying" means AutoVectorizeV2
  // recovered. Only the terminal failure should rerun compilation.
  if (message.contains("AutoVectorizeV2 failed") &&
      !message.contains("retrying"))
    return true;
  // TODO: replace me to a links to strtable which will be also used by error
  // reporter
  static constexpr llvm::StringLiteral kMessages[] = {
      "Failed to vectorize the function.",
      "outside outlined vector region",
      "unexpected vector operation outside vector function",
      "Failed to outline vector function",
      "fail to preprocess",
      "direct-register-ra requires the only-marked safety contract",
      "Failed to compute dynamic vector sizes",
      "Failed to compute max element bit width",
      "Empty shape: rank is zero",
      "Invalid dimension index",
      "Invalid shape dimension: must be positive",
      "Exceeds vector capacity",
  };
  for (llvm::StringRef pattern : kMessages)
    if (message.contains(pattern))
      return true;
  return false;
}

} // namespace regbase
} // namespace bishengir

#endif // BISHENGIR_TOOLS_BISHENGIR_COMPILE_REGBASE_VECTORIZATIONFAILURE_H
