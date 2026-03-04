//===- BiShengRunnerUtils.cpp - Utils for MLIR execution ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Targets/NPUTargetSpec.cpp.inc"

#include <random>

extern "C" llvm::ToolOutputFile *MLIR_RUNNERUTILS_EXPORT
getFileHandle(const char *filePath) {
  if (const auto parentPath = llvm::sys::path::parent_path(filePath);
      !parentPath.empty() &&
      llvm::sys::fs::create_directories(parentPath).value() != 0) {
    llvm_unreachable("Couldn't create directories!");
    return nullptr;
  }

  auto output = mlir::openOutputFile(filePath);
  if (!output) {
    llvm_unreachable("Couldn't open output file!");
    return nullptr;
  }
  output->keep();
  return output.release();
}

extern "C" void MLIR_RUNNERUTILS_EXPORT
closeFileHandle(llvm::ToolOutputFile *output) {
  if (output == nullptr)
    llvm_unreachable("Erasing non-existing pointer to a file");

  delete output;
}

namespace {

template <typename Distribution, typename T, typename... Args>
void getData(DynamicMemRefType<T> data, Args &&...args) {
  static std::mt19937 generator(1);
  Distribution distribution(std::forward<Args>(args)...);
  for (auto &element : data)
    element = distribution(generator);
}

// NOTE: DO NOT MODIFY
// bishengir-cpu-runner.py relies on the logic here
template <typename T>
void printData(llvm::raw_fd_ostream &out, DynamicMemRefType<T> data) {
  out << data.rank;

  for (int64_t i = 0; i < data.rank; i++)
    out << (i == 0 ? '\n' : ' ') << data.sizes[i];

  bool isFirst = true;
  for (const auto &element : data) {
    out << (isFirst ? '\n' : ' ');
    isFirst = false;

    if constexpr (std::is_integral_v<T>)
      out << element;
    else {
      std::stringstream ss;
      ss << std::scientific << std::setprecision(9);
      ss << element;
      out << ss.str();
    }
  }
  out << '\n';
}

// Expose API func getDefaultSimtDynamicSize so that external caller
// is able to get the default UB size in SIMT/SIMD mix mode.
extern "C" int MLIR_RUNNERUTILS_EXPORT
getDefaultSimtDynamicSize(const char *archChar) {
  if (!archChar) return 0;
  std::string arch(archChar);

  auto target = mlir::hacc::symbolizeTargetDeviceEnum(arch);
  auto maybeSpec = mlir::hacc::getTargetSpec(target);
  if (!maybeSpec.has_value()) {
    llvm_unreachable("getTargetSpec failed");
  }

  auto ubSpaceSize = maybeSpec.value()->UbSize;
  constexpr int numBitsInByte = 8;
  constexpr int numByteInKB = 1024;
  auto minimalDCacheSize = maybeSpec.value()->MinimalDCacheSize;
  
  return (ubSpaceSize - minimalDCacheSize) / numBitsInByte / numByteInKB;
}

} // namespace

#define GENERATE_FOR_TYPE(SUFFIX, T, Distribution, ...)                        \
  extern "C" void MLIR_RUNNERUTILS_EXPORT _mlir_ciface_getData##SUFFIX(        \
      UnrankedMemRefType<T> *data) {                                           \
    getData<Distribution>(DynamicMemRefType<T>(*data), __VA_ARGS__);           \
  }                                                                            \
  extern "C" void MLIR_RUNNERUTILS_EXPORT _mlir_ciface_printData##SUFFIX(      \
      llvm::ToolOutputFile *out, UnrankedMemRefType<T> *data) {                \
    printData(out->os(), DynamicMemRefType<T>(*data));                         \
  }

GENERATE_FOR_TYPE(BF16, bf16, std::uniform_real_distribution<float>, -100.,
                  100.)

GENERATE_FOR_TYPE(F16, f16, std::uniform_real_distribution<float>, -100., 100.)
GENERATE_FOR_TYPE(F32, float, std::uniform_real_distribution<float>, -1000.,
                  1000.)
GENERATE_FOR_TYPE(F64, double, std::uniform_real_distribution<double>, -10000.,
                  10000.)

GENERATE_FOR_TYPE(I1, bool, std::uniform_int_distribution<uint8_t>, 0, 1)
GENERATE_FOR_TYPE(I8, int8_t, std::uniform_int_distribution<int8_t>, -100, 100)
GENERATE_FOR_TYPE(I16, int16_t, std::uniform_int_distribution<int16_t>, -1000,
                  1000)
GENERATE_FOR_TYPE(I32, int32_t, std::uniform_int_distribution<int32_t>, -10000,
                  10000)
GENERATE_FOR_TYPE(I64, int64_t, std::uniform_int_distribution<int64_t>, -100000,
                  100000)
