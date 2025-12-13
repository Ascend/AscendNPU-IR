//===- TestPasses.h -------------------------------------------------------===//
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
//============================================================================//

#ifndef TEST_TESTPASSES_H
#define TEST_TESTPASSES_H
namespace bishengir_test {
// Macro to generate function declarations and the calls
// When used with a pass name like "BiShengSegmenterPass",

// function declaration: void registerBiShengSegmenterPass()
#define DECLARE_PASS(name) void register##name()
// the function call: registerBiShengSegmenterPass()
#define REGISTER_PASS(name) register##name()

// X-Macro pattern: This macro takes another macro (X) as a parameter
// and applies it to each pass name in the list.
// This allows us to generate different code for the same list of passes
// by passing different macros as X.
#define PASS_LIST(X)                                                           \
  X(BiShengSegmenterPass);                                                     \
  X(InstructionMarkerPass);                                                    \
  X(TestAssignFusionKindAttrs);                                                \
  X(TestBufferUtilsPass);                                                      \
  X(TestCanFusePass);                                                          \
  X(TestDimensionAnalyzer);                                                    \
  X(TestFlattenInterface);                                                     \
  X(TestFunctionCallPass);                                                     \
  X(TestHIVMDimensionAnalyzer);                                                \
  X(TestHIVMTransformsPass);                                                   \
  X(ValidPropagatedReshapePass)

// Generate declarations
PASS_LIST(DECLARE_PASS);
#undef DECLARE_PASS

inline void registerAllTestPasses() {
  // Generate registration calls
  PASS_LIST(REGISTER_PASS);
}
#undef REGISTER_PASS
#undef PASS_LIST
} // namespace bishengir_test
#endif // TEST_TESTPASSES_H
