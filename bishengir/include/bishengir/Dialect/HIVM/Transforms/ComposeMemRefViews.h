//===- ComposeMemRefViews.h -------------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_COMPOSEMEMREFVIEWS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_COMPOSEMEMREFVIEWS_H

namespace mlir {
class MLIRContext;
class RewritePatternSet;

namespace hivm {

void populateComposeMemRefViewPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_COMPOSEMEMREFVIEWS_H
