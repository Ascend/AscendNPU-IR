set(LLVM_TARGET_DEFINITIONS SyncCapabilityProfile.td)
bishengir_target_tablegen(SyncCapability.h.inc
  -gen-target-capability-decls)
bishengir_target_tablegen(SyncCapability.cpp.inc
  -gen-target-capability-defs)
add_public_tablegen_target(BiShengIRSyncCapabilityIncGen)
add_dependencies(BiShengIRSyncCapabilityIncGen
  BiShengIRNPUArchitectureIncGen)
add_dependencies(mlir-headers BiShengIRSyncCapabilityIncGen)
