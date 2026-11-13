set(LLVM_TARGET_DEFINITIONS MmadCapabilityProfile.td)
bishengir_target_tablegen(MmadCapability.h.inc
  -gen-target-capability-decls)
bishengir_target_tablegen(MmadCapability.cpp.inc
  -gen-target-capability-defs)
add_public_tablegen_target(BiShengIRMmadCapabilityIncGen)
add_dependencies(BiShengIRMmadCapabilityIncGen
  BiShengIRNPUArchitectureIncGen)
add_dependencies(mlir-headers BiShengIRMmadCapabilityIncGen)
