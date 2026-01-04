//===-- Passes.td - HACC dialect pass definition file ---------*-tablegen-*-==//
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

#ifndef BISHENGIR_DIALECT_HACC_PASSES
#define BISHENGIR_DIALECT_HACC_PASSES

include "mlir/Pass/PassBase.td"

def HACCRenameFunction : Pass<"hacc-rename-func"> {
  let summary = "Rename function based on attribute.";
  let description = [{
    This pass renames a function according to the `hacc.rename_func` attribute
    and updates all the references within the module.

    For example:
    ```mlir
    func.func @bar() attributes {hacc.rename_func = #hacc.rename_func<@foo>} {
      return
    }

    func.func @caller() {
      func.call @bar() : () -> ()
      return
    }
    ```

    Will be transformed into:
    ```mlir
    func.func @foo() {
      return
    }

    func.func @caller() {
      func.call @foo() : () -> ()
      return
    }
    ```

    Restrictions:
      * There must be no existing function whose name is the same as the target
        function name
  }];
  let constructor = "mlir::hacc::createRenameFuncPass()";
}

def AppendTargetDeviceSpec : Pass<"hacc-append-device-spec", "ModuleOp"> {
  let constructor = "mlir::hacc::createAppendDeviceSpecPass()";
  let options = [
     Option<
      "target", "target", "::mlir::hacc::TargetDevice",
      "::mlir::hacc::TargetDevice::Unknown",
      "Device target name",
      [{
        ::llvm::cl::values(
#define TO_STRING(x) #x
#define REGISTER_TARGET(TARGET)                                                \
  clEnumValN(::mlir::hacc::TargetDevice::TARGET, TO_STRING(TARGET),            \
             TO_STRING(TARGET))
            // Ascend910B series
            REGISTER_TARGET(Ascend910B1), REGISTER_TARGET(Ascend910B2),
            REGISTER_TARGET(Ascend910B3), REGISTER_TARGET(Ascend910B4),
            clEnumValN(mlir::hacc::TargetDevice::Ascend910B4_1,
                           "Ascend910B4-1", "Ascend910B4-1"),
            REGISTER_TARGET(Ascend910B2C),
            // Ascend910_93 series
            REGISTER_TARGET(Ascend910_9362), REGISTER_TARGET(Ascend910_9372),
            REGISTER_TARGET(Ascend910_9381), REGISTER_TARGET(Ascend910_9382),
            REGISTER_TARGET(Ascend910_9391), REGISTER_TARGET(Ascend910_9392),
            // Ascend910_95 series
            REGISTER_TARGET(Ascend910_950z), REGISTER_TARGET(Ascend910_9579),
            REGISTER_TARGET(Ascend910_957b), REGISTER_TARGET(Ascend910_957d),
            REGISTER_TARGET(Ascend910_9581), REGISTER_TARGET(Ascend910_9589),
            REGISTER_TARGET(Ascend910_958a), REGISTER_TARGET(Ascend910_958b),
            REGISTER_TARGET(Ascend910_9599),
            REGISTER_TARGET(Unknown)
#undef REGISTER_TARGET
#undef TO_STRING
        )
      }]
    >,
  ];
  let dependentDialects = [
    "hacc::HACCDialect",
    "mlir::DLTIDialect",
  ];
}

#endif // BISHENGIR_DIALECT_HACC_PASSES
