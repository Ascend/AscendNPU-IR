//===- RegisterEverything.cpp - API to register all dialects/passes -------===//
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

#include "bishengir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/Pass/PassManager.h"

#include "bishengir-c/Dialect/Annotation.h"
#include "bishengir-c/Dialect/HFusion.h"
#include "bishengir-c/Dialect/HIVM.h"

PYBIND11_MODULE(_bishengirRegisterEverything, m) {
  m.doc() =
      "BiShengIR All Upstream Dialects, Extensions and Passes Registration";

  // register dialects of bishengir i.e. hivm, hfusion. annotation
  m.def("register_dialects",
        [](MlirContext context) { bishengirRegisterAllDialects(context); });

  m.def("register_translations",
        [](MlirContext context) { bishengirRegisterAllTranslations(context); });

  // Register all passes on load.
  bishengirRegisterAllPasses();
}
