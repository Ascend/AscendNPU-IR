//===- CPURunnerMetadata.h - CPU Runner metadata definition -----*- C++ -*-===//
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

#ifndef BISHENGIR_PASS_CPURUNNERMETADATA_H
#define BISHENGIR_PASS_CPURUNNERMETADATA_H

#include "bishengir/Config/bishengir-config.h"

#if (defined(MLIR_ENABLE_EXECUTION_ENGINE) && MLIR_ENABLE_EXECUTION_ENGINE) || \
    defined(BISHENGIR_ENABLE_EXECUTION_ENGINE)
#include "bishengir/ExecutionEngine/Passes.h"

namespace bishengir {

template <bool includePassInfo>
struct CPURunnerMetadata;

template <>
struct CPURunnerMetadata<false> {
  unsigned numOccurrences{0};
  mlir::execution_engine::CPURunnerPipelineOptions options;
};

template <>
struct CPURunnerMetadata<true> : public CPURunnerMetadata<false> {
  std::string passName;
  std::size_t passIndex = 1;
};

template <bool includePassInfo>
struct CPURunnerMetadataParser
    : public llvm::cl::parser<CPURunnerMetadata<includePassInfo>> {
  using parser_data_type = CPURunnerMetadata<includePassInfo>;

  explicit CPURunnerMetadataParser(llvm::cl::Option &o)
      : llvm::cl::parser<parser_data_type>(o) {}

  void printOptionInfo(const llvm::cl::Option &opt,
                       size_t globalWidth) const final;
  // Return true on error.
  static bool parse(llvm::cl::Option &opt, llvm::StringRef argName,
                    llvm::StringRef arg, parser_data_type &value);
};
} // namespace bishengir

#endif // MLIR_ENABLE_EXECUTION_ENGINE
  static bool parse(llvm::cl::Option &opt, llvm::StringRef argName,
                    llvm::StringRef arg, parser_data_type &value);
};

} // namespace bishengir

#endif


#endif // BISHENGIR_PASS_CPURUNNERMETADATA_H
