#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMPROTONOPS_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMPROTONOPS_H

#include <cstdint>

namespace mlir::hivm::proton {

// 12-bit DFX region_id layout:
//   bits [9:0]  - logical scope / region id
//   bit  10     - always 1, MindStudio parser requirement
//   bit  11     - 0 = start, 1 = end
inline constexpr uint32_t kRegionIdBits = 12;
inline constexpr uint32_t kMaxRegionId = (1U << kRegionIdBits) - 1;
inline constexpr uint32_t kLogicalScopeIdBits = 10;
inline constexpr uint32_t kMaxScopeCount = 1U << kLogicalScopeIdBits;
inline constexpr uint32_t kRequiredRegionIdMask = 1U << 10;
inline constexpr uint32_t kStartEndBit = 11;
inline constexpr uint32_t kStartEndMask = 1U << kStartEndBit;

constexpr uint32_t getStartRegionId(uint32_t logicalScopeId) {
  return logicalScopeId | kRequiredRegionIdMask;
}

constexpr uint32_t getEndRegionId(uint32_t logicalScopeId) {
  return logicalScopeId | kRequiredRegionIdMask | kStartEndMask;
}

} // namespace mlir::hivm::proton

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMPROTONOPS_H
