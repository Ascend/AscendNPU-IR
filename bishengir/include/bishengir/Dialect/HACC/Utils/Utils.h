//===- Utils.h - Utilities to support the HACC dialect -----------*- C++-*-===//
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

#ifndef BISHENGIR_DIALECT_HACC_UTILS_UTILS_H
#define BISHENGIR_DIALECT_HACC_UTILS_UTILS_H

#include "bishengir/Dialect/HACC/IR/HACC.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallSet.h"

#include <type_traits>

namespace mlir {
namespace hacc {
namespace utils {

//===----------------------------------------------------------------------===//
// Utility functions for HACCFunction
//===----------------------------------------------------------------------===//

template <typename SourceFuncTy, typename TargetFuncTy>
struct FuncTyMapping;

template <>
struct FuncTyMapping<func::FuncOp, hacc::HACCFunction>;

template <>
struct FuncTyMapping<LLVM::LLVMFuncOp, hacc::HACCFunction>;

template <>
struct FuncTyMapping<hacc::HACCFunction, func::FuncOp>;

template <>
struct FuncTyMapping<hacc::HACCFunction, LLVM::LLVMFuncOp>;

struct ExternalFuncInfo {
  StringRef funcName;
  StringRef srcPath;
};

/// Dynamic cast from `SourceFuncTy` to `TargetFuncTy`.
///
/// The only combinations supported are:
///   1) `func::FuncOp` to `hacc::HACCFunction`
///   2) `LLVM::LLVMFuncOp` to `hacc::HACCFunction`
///   2) `hacc::HACCFunction` to `func::FuncOp`
///   2) `hacc::HACCFunction` to `LLVM::LLVMFuncOp`
template <typename TargetFuncTy, typename SourceFuncTy,
          typename = std::enable_if_t<
              std::is_class<FuncTyMapping<SourceFuncTy, TargetFuncTy>>::value>>
TargetFuncTy dynCastFunc(SourceFuncTy op) {
  return dyn_cast<TargetFuncTy>(op.getOperation());
}

/// Return if the function is a host function.
bool isHost(Operation *func);

/// Return if the function is a device function.
bool isDevice(Operation *func);

/// Return if the function is a device entry function.
bool isDeviceEntry(Operation *func);

/// Return if the function inputs/outputs are strictly not alias.
bool hasNoIOAlias(Operation *func);

/// Return the host function type. If the function is not a host function,
/// return `std::nullopt`.
std::optional<HostFuncType> getHostFuncType(Operation *func);

/// Return if the function argument at `argIdx` has `hacc.arg_type` of
/// `argType`.
bool isKernelArg(Operation *func, unsigned argIdx, KernelArgType argType);

/// Return the function argument has `hacc.arg_type` of`argType`.
std::optional<BlockArgument> getBlockArgument(func::FuncOp funcOp,
                                              KernelArgType argType);

/// Return if the function argument at `argIdx` has `hacc.arg_type` of
/// `tiling_key` or `tiling_data`.
bool isTilingArg(Operation *func, unsigned argIdx);

/// Set the function to be a device function.
/// \note disallowed attributes will be automatically dropped.
void setDevice(Operation *func);

/// Set the function to be a device entry function.
/// \note: disallowed attributes will be automatically dropped.
void setDeviceEntry(Operation *func);

/// Set the function to be a host function.
/// \note disallowed attributes will be automatically dropped.
void setHost(Operation *func);

/// Set the host function function type.
void setHostFuncType(Operation *func, HostFuncType hostFuncType);

/// Set the function to be always inlined.
void setAlwaysInline(Operation *func);

/// Collect external functions
SmallVector<ExternalFuncInfo> collectExternalFuncs(ModuleOp mod);

//===----------------------------------------------------------------------===//
// Data Layout and Target Info
//===----------------------------------------------------------------------===//

/// Get NPU target specification from module.
std::optional<HACCTargetDeviceSpecInterface> getNPUTargetSpec(ModuleOp op);

/// Set NPU target specification to module.
void setNPUTargetSpec(ModuleOp op, HACCTargetDeviceSpecInterface spec);

/// Get integer value from the spec entry.
int64_t getIntegerSpecValue(DataLayoutEntryInterface entry);

} // namespace utils

/// Seperate modules containing host and device code
std::pair<ModuleOp, ModuleOp> separateHostDeviceModule(ModuleOp);

/// Filter funcs based on a callback
ModuleOp filterFuncsInModule(ModuleOp &op,
                             std::function<bool(Operation *op)> shouldInclude);

/// Check func satisfication based on callback
bool checkSatisfy(ModuleOp &op, std::function<bool(Operation *op)> satisfyRule);

/// Check if the current operation is a device or a host code
bool existHost(Operation *module);
bool existEntryHost(Operation *module);

bool isMixEntry(Operation *func);
bool notExportedAsDag(Operation *func);

/// Create a named attribute `hacc.arg_type = #hacc.arg_type<argType>`.
NamedAttribute createHACCKernelArgAttr(MLIRContext *ctx, KernelArgType argType);

/// Get the `hacc.input`/`hacc.output` index that the argument at `argIdx`
/// corresponds to from the function argument attribute.
///
/// Return `std::nullopt` if the argument is not an input/output argument.
std::optional<unsigned> getHACCInputIdx(func::FuncOp func, unsigned argIdx);
std::optional<unsigned> getHACCOuputIdx(func::FuncOp func, unsigned argIdx);

/// Construct a host function name for the device kernel based on the host
/// function type.
std::string constructHostFunctionName(const std::string &kernelName,
                                      HostFuncType type);

size_t countDeviceArgSizeInByte(ModuleOp modOp);

} // namespace hacc
} // namespace mlir

#endif // BISHENGIR_DIALECT_HACC_UTILS_UTILS_H
