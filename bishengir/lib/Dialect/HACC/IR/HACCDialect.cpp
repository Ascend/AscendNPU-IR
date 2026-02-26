//===- HACCDialect.cpp - Implementation of HACC dialect and types ---------===//
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

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/IR/HACCInterfaces.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "bishengir/Dialect/HACC/IR/HACCEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HACC/IR/HACCAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::hacc;

void mlir::hacc::HACCDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "bishengir/Dialect/HACC/IR/HACCTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "bishengir/Dialect/HACC/IR/HACCAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TargetDeviceSpecAttr
//===----------------------------------------------------------------------===//

LogicalResult
hacc::TargetDeviceSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   ArrayRef<DataLayoutEntryInterface> entries) {
  // Entries in a target device spec must be present in HACC DeviceSpecEnum
  DenseSet<StringAttr> ids;
  for (DataLayoutEntryInterface entry : entries) {
    if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
      return emitError()
             << "dlti.target_device_spec does not allow type as a key: "
             << type;
    }
    // Check that keys in a target device spec are unique.
    auto id = entry.getKey().get<StringAttr>();
    if (!ids.insert(id).second)
      return emitError() << "repeated layout entry key: " << id.getValue();

    auto maybeSpec = symbolizeEnum<DeviceSpec>(id.getValue());
    if (!maybeSpec.has_value())
      return emitError() << "invalid target device spec: " << id;
  }
  return success();
}

#ifdef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
::mlir::FailureOr<::mlir::Attribute> 
mlir::hacc::TargetDeviceSpecAttr::query(::mlir::DataLayoutEntryKey key) const {
  for (const auto& entry : getEntries()) {
    if (entry.getKey() == key)
      return entry.getValue();
  }
  return ::mlir::failure();
}
#endif

#include "bishengir/Dialect/HACC/IR/HACCBaseDialect.cpp.inc"
