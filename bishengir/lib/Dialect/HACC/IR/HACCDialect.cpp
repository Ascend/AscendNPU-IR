//===- HACCDialect.cpp - Implementation of HACC dialect and types ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "bishengir/Dialect/HACC/IR/HACCBaseDialect.cpp.inc"
