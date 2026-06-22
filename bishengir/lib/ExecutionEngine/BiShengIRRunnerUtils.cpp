//===- BiShengRunnerUtils.cpp - Utils for MLIR execution ------------------===//
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

#include <random>

extern "C" llvm::ToolOutputFile *MLIR_RUNNERUTILS_EXPORT
getFileHandle(const char *filePath) {
  if (const auto parentPath = llvm::sys::path::parent_path(filePath);
      !parentPath.empty() &&
      llvm::sys::fs::create_directories(parentPath).value() != 0) {
    llvm::report_fatal_error("Couldn't create directories!");
    return nullptr;
  }

  auto output = mlir::openOutputFile(filePath);
  if (!output) {
    llvm::report_fatal_error("Couldn't open output file!");
    return nullptr;
  }
  output->keep();
  return output.release();
}

extern "C" void MLIR_RUNNERUTILS_EXPORT
closeFileHandle(llvm::ToolOutputFile *output) {
  if (output == nullptr)
    llvm::report_fatal_error("Erasing non-existing pointer to a file");

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
