// RUN: bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %s | FileCheck %s

include "bishengir/Target/NPU/NPUArchitectureFeatures.td"

// CHECK: enum class CubeVectorDataExchange {
// CHECK-NEXT: kGlobalMemoryOnly,
// CHECK-NEXT: kOnChipSupported,
// CHECK-NEXT: kUnknown,
// CHECK: case CubeVectorDataExchange::kGlobalMemoryOnly:
// CHECK-SAME: return "global-memory-only";
// CHECK: case CubeVectorDataExchange::kOnChipSupported:
// CHECK-SAME: return "on-chip-supported";
// CHECK: enum class CubeVectorMode {
// CHECK-NEXT: kCoupled,
// CHECK-NEXT: kSeparated,
// CHECK-NEXT: kUnknown,
// CHECK: case CubeVectorMode::kCoupled: return "coupled";
// CHECK: case CubeVectorMode::kSeparated: return "separated";
// CHECK: enum class ExecutionModel {
// CHECK-NEXT: kMemBased,
// CHECK-NEXT: kRegBased,
// CHECK-NEXT: kUnknown,
// CHECK: case ExecutionModel::kMemBased: return "mem-based";
// CHECK: case ExecutionModel::kRegBased: return "reg-based";
// CHECK: struct ArchitectureFeatureProfile {
// CHECK-NEXT: NPUArchitectureKind architecture;
// CHECK-NEXT: ::bishengir::npu::CubeVectorDataExchange cubeVectorDataExchange;
// CHECK-NEXT: ::bishengir::npu::CubeVectorMode cubeVectorMode;
// CHECK-NEXT: ::bishengir::npu::ExecutionModel executionModel;
// CHECK: kDavC220Profile{
// CHECK-NEXT: NPUArchitectureKind::kDavC220,
// CHECK-NEXT: CubeVectorDataExchange::kGlobalMemoryOnly,
// CHECK-NEXT: CubeVectorMode::kSeparated,
// CHECK-NEXT: ExecutionModel::kMemBased
// CHECK: kDavC310Profile{
// CHECK-NEXT: NPUArchitectureKind::kDavC310,
// CHECK-NEXT: CubeVectorDataExchange::kOnChipSupported,
// CHECK-NEXT: CubeVectorMode::kSeparated,
// CHECK-NEXT: ExecutionModel::kRegBased
// CHECK: kDavM300Profile{
// CHECK-NEXT: NPUArchitectureKind::kDavM300,
// CHECK-NEXT: CubeVectorDataExchange::kOnChipSupported,
// CHECK-NEXT: CubeVectorMode::kCoupled,
// CHECK-NEXT: ExecutionModel::kRegBased
// CHECK: inline CubeVectorDataExchange getCubeVectorDataExchange(
// CHECK-SAME: NPUArchitectureKind architecture)
// CHECK: return profile ? profile->cubeVectorDataExchange
// CHECK-SAME: : CubeVectorDataExchange::kUnknown;
// CHECK: inline CubeVectorMode getCubeVectorMode(
// CHECK-SAME: NPUArchitectureKind architecture)
// CHECK: return profile ? profile->cubeVectorMode : CubeVectorMode::kUnknown;
// CHECK: inline ExecutionModel getExecutionModel(
// CHECK-SAME: NPUArchitectureKind architecture)
// CHECK: return profile ? profile->executionModel : ExecutionModel::kUnknown;
