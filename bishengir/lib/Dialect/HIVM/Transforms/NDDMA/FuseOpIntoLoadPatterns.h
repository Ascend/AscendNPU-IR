//===- FuseOpIntoLoadPatterns.h - internal load patterns ----------===//
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

#ifndef BISHENGIR_LIB_DIALECT_HIVM_TRANSFORMS_NDDMA_FUSEOPINTOLOADPATTERNS_H
#define BISHENGIR_LIB_DIALECT_HIVM_TRANSFORMS_NDDMA_FUSEOPINTOLOADPATTERNS_H

namespace mlir {
class RewritePatternSet;
namespace hivm::detail {
// Register independent load-fusion patterns. The pass owns architecture gating
// and applies the complete set with the MLIR pattern driver.
void populateBroadcastLoadPatterns(RewritePatternSet &patterns);
} // namespace hivm::detail
} // namespace mlir

#endif
