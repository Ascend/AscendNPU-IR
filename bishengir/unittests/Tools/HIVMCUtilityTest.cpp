//===- HIVMCUtilityTest.cpp - Device IR serialization tests
//----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Tools/hivmc/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

#include <thread>

using namespace bishengir;

namespace {

constexpr llvm::StringLiteral debugModule = R"(
@storage = global [16 x i8] zeroinitializer

define void @kernel_vf_debug() !dbg !4 {
entry:
  #dbg_value(<2 x float> <float 0.000000e+00, float undef>, !7, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !9)
  #dbg_value(i32 7, !8, !DIExpression(), !9)
  #dbg_declare(ptr getelementptr (i8, ptr @storage, i64 4), !7, !DIExpression(DW_OP_plus_uconst, 4), !9)
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "debug.py", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !5)
!4 = distinct !DISubprogram(name: "kernel_vf_debug", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!5 = !{}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DILocalVariable(name: "vector", scope: !4, file: !1, line: 1, type: !10)
!8 = !DILocalVariable(name: "scalar", scope: !4, file: !1, line: 1, type: !6)
!9 = !DILocation(line: 1, column: 1, scope: !4)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 64, flags: DIFlagVector, elements: !12)
!11 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!12 = !{!13}
!13 = !DISubrange(count: 2)
)";

constexpr llvm::StringLiteral mlirDebugModule = R"mlir(
#file = #llvm.di_file<"debug.py" in "/">
#cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C,
  file = #file, producer = "test", isOptimized = false, emissionKind = Full>
#sp = #llvm.di_subprogram<compileUnit = #cu, scope = #file,
  name = "kernel_vf_debug", file = #file, subprogramFlags = Definition>
#int = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int",
  sizeInBits = 32, encoding = DW_ATE_signed>
#float = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float",
  sizeInBits = 32, encoding = DW_ATE_float>
#vector_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #float,
  sizeInBits = 64, flags = Vector, elements = #llvm.di_subrange<count = 2>>
#scalar = #llvm.di_local_variable<scope = #sp, name = "scalar", file = #file,
  line = 1, type = #int>
#vector = #llvm.di_local_variable<scope = #sp, name = "vector", file = #file,
  line = 1, type = #vector_type>
#loc = loc(fused<#sp>["debug.py":1:1])
module {
  llvm.func @kernel_vf_debug(%arg: i32, %address: !llvm.ptr) {
    %vector = llvm.mlir.constant(dense<[0.0, 1.0]> : vector<2xf32>) : vector<2xf32>
    llvm.intr.dbg.value #vector #llvm.di_expression<[DW_OP_LLVM_fragment(0, 32)]> = %vector : vector<2xf32> loc(#loc)
    llvm.intr.dbg.value #scalar = %arg : i32 loc(#loc)
    llvm.intr.dbg.declare #scalar #llvm.di_expression<[DW_OP_plus_uconst(4)]> = %address : !llvm.ptr loc(#loc)
    llvm.return
  } loc(fused<#sp>["debug.py":1:1])
}
)mlir";

std::unique_ptr<llvm::Module> parseDebugModule(llvm::LLVMContext &context) {
  llvm::SMDiagnostic diagnostic;
  return llvm::parseAssemblyString(debugModule, diagnostic, context);
}

size_t countDebugRecords(const llvm::Module &module) {
  size_t count = 0;
  for (const auto &function : module)
    for (const auto &instruction : llvm::instructions(function))
      count += llvm::range_size(instruction.getDbgRecordRange());
  return count;
}

TEST(HIVMCUtilityTest, SerializeLegacyDebugIntrinsics) {
  llvm::LLVMContext context;
  auto module = parseDebugModule(context);
  ASSERT_NE(module, nullptr);
  module->setIsNewDbgInfoFormat(true);
  ASSERT_EQ(countDebugRecords(*module), 3u);
  module->setIsNewDbgInfoFormat(false);

  std::string output = moduleToIrString(*module);

  EXPECT_EQ(output.find("#dbg_"), std::string::npos);
  EXPECT_NE(output.find("call void @llvm.dbg.value(metadata <2 x float> <float "
                        "0.000000e+00, float undef>"),
            std::string::npos);
  EXPECT_NE(output.find("!DIExpression(DW_OP_LLVM_fragment, 0, 32)"),
            std::string::npos);
  EXPECT_NE(output.find("call void @llvm.dbg.value(metadata i32 7"),
            std::string::npos);
  EXPECT_NE(
      output.find("call void @llvm.dbg.declare(metadata ptr getelementptr"),
      std::string::npos);
  EXPECT_NE(output.find("!DIExpression(DW_OP_plus_uconst, 4)"),
            std::string::npos);
  EXPECT_NE(output.find("attrtype: 64"), std::string::npos);
  EXPECT_NE(output.find("functype: 256"), std::string::npos);
  EXPECT_FALSE(module->IsNewDbgInfoFormat);
  EXPECT_EQ(countDebugRecords(*module), 0u);
  EXPECT_EQ(output, moduleToIrString(*module));
}

TEST(HIVMCUtilityTest, TranslationProducesLegacyDebugIntrinsics) {
  mlir::MLIRContext mlirContext;
  mlirContext.loadDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerBuiltinDialectTranslation(mlirContext);
  mlir::registerLLVMDialectTranslation(mlirContext);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(mlirDebugModule, &mlirContext);
  ASSERT_TRUE(source);

  llvm::LLVMContext context;
  auto original = mlir::translateModuleToLLVMIR(*source, context);
  ASSERT_NE(original, nullptr);
  ASSERT_TRUE(original->IsNewDbgInfoFormat);
  ASSERT_EQ(countDebugRecords(*original), 3u);

  auto module = translateModuleToLegacyLLVMIR(*source, context);
  ASSERT_NE(module, nullptr);
  EXPECT_FALSE(module->IsNewDbgInfoFormat);
  EXPECT_EQ(countDebugRecords(*module), 0u);

  std::string output = moduleToIrString(*module);

  EXPECT_EQ(output.find("#dbg_"), std::string::npos);
  EXPECT_NE(output.find("call void @llvm.dbg.value"), std::string::npos);
  EXPECT_NE(output.find("call void @llvm.dbg.declare"), std::string::npos);
  EXPECT_NE(output.find("metadata <2 x float> <float 0.000000e+00, float "
                        "1.000000e+00>"),
            std::string::npos);
  EXPECT_NE(output.find("!DIExpression(DW_OP_LLVM_fragment, 0, 32)"),
            std::string::npos);
  EXPECT_NE(output.find("!DIExpression(DW_OP_plus_uconst, 4)"),
            std::string::npos);
  EXPECT_FALSE(module->IsNewDbgInfoFormat);
}

TEST(HIVMCUtilityTest, FileAndMemorySerializationMatch) {
  llvm::LLVMContext context;
  auto module = parseDebugModule(context);
  ASSERT_NE(module, nullptr);
  module->setIsNewDbgInfoFormat(false);
  std::string expected = moduleToIrString(*module);

  llvm::SmallString<128> path;
  int fd;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("hivmc-debug-records", "ll",
                                                  fd, path));
  llvm::sys::fs::closeFile(fd);
  llvm::SmallVector<IRModulePair> modules;
  modules.emplace_back(std::move(module), SubCoreTarget::AIV);
  auto files = saveToFiles(modules, path.str().str(), false);
  ASSERT_EQ(files.size(), 1u);
  ASSERT_NE(files[0].first, nullptr);
  auto contents = llvm::MemoryBuffer::getFile(files[0].first->getFilename());
  ASSERT_TRUE(contents);
  EXPECT_EQ((*contents)->getBuffer(), expected);
  EXPECT_FALSE(modules[0].first->IsNewDbgInfoFormat);
  EXPECT_EQ(countDebugRecords(*modules[0].first), 0u);
}

TEST(HIVMCUtilityTest, ParallelSerializationWithSharedContext) {
  mlir::MLIRContext mlirContext;
  mlirContext.loadDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerBuiltinDialectTranslation(mlirContext);
  mlir::registerLLVMDialectTranslation(mlirContext);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(mlirDebugModule, &mlirContext);
  ASSERT_TRUE(source);

  llvm::LLVMContext context;
  auto aic = translateModuleToLegacyLLVMIR(*source, context);
  auto aiv = translateModuleToLegacyLLVMIR(*source, context);
  ASSERT_NE(aic, nullptr);
  ASSERT_NE(aiv, nullptr);
  ASSERT_FALSE(aic->IsNewDbgInfoFormat);
  ASSERT_FALSE(aiv->IsNewDbgInfoFormat);
  const std::string expectedAic = moduleToIrString(*aic);
  const std::string expectedAiv = moduleToIrString(*aiv);

  std::string actualAic, actualAiv;
  std::thread aicWorker([&] { actualAic = moduleToIrString(*aic); });
  std::thread aivWorker([&] { actualAiv = moduleToIrString(*aiv); });
  aicWorker.join();
  aivWorker.join();

  EXPECT_EQ(actualAic, expectedAic);
  EXPECT_EQ(actualAiv, expectedAiv);
  EXPECT_FALSE(aic->IsNewDbgInfoFormat);
  EXPECT_FALSE(aiv->IsNewDbgInfoFormat);
  EXPECT_EQ(countDebugRecords(*aic), 0u);
  EXPECT_EQ(countDebugRecords(*aiv), 0u);
}

} // namespace
