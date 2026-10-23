# Keep this catalogue local to the unit test: no production capability or
# dialect definitions are needed to compile the generated runtime.
set(LLVM_TARGET_DEFINITIONS CapabilityTest.td)
bishengir_target_tablegen(CapabilityTest.h.inc
  -gen-target-capability-decls)
bishengir_target_tablegen(CapabilityTest.cpp.inc
  -gen-target-capability-defs)
add_public_tablegen_target(BiShengIRTargetCapabilityTestIncGen)

set(LLVM_LINK_COMPONENTS Support)
add_bishengir_unittest(BiShengIRTargetCapabilityTests
  PARTIAL_SOURCES_INTENDED
  TargetCapabilityTest.cpp
)

target_include_directories(BiShengIRTargetCapabilityTests PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(BiShengIRTargetCapabilityTests
  BiShengIRNPUArchitectureIncGen
  BiShengIRTargetCapabilityTestIncGen
)
target_link_libraries(BiShengIRTargetCapabilityTests PRIVATE MLIRIR)

# A separate executable ensures a generic, profile-free catalogue compiles
# without architecture headers, definitions, or generated include dependencies.
set(LLVM_TARGET_DEFINITIONS SchemaOnlyCapabilityTest.td)
bishengir_target_tablegen(SchemaOnlyCapabilityTest.h.inc
  -gen-target-capability-decls)
bishengir_target_tablegen(SchemaOnlyCapabilityTest.cpp.inc
  -gen-target-capability-defs)
add_public_tablegen_target(BiShengIRTargetCapabilitySchemaOnlyTestIncGen)

add_bishengir_unittest(BiShengIRTargetCapabilitySchemaOnlyTests
  PARTIAL_SOURCES_INTENDED
  SchemaOnlyCapabilityTest.cpp
)
target_include_directories(BiShengIRTargetCapabilitySchemaOnlyTests PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(BiShengIRTargetCapabilitySchemaOnlyTests
  BiShengIRTargetCapabilitySchemaOnlyTestIncGen
)
target_link_libraries(BiShengIRTargetCapabilitySchemaOnlyTests PRIVATE MLIRIR)

# The type-free catalogue exercises empty and parameter-only schemas without
# type slots, type constraints, or architecture declarations.
set(LLVM_TARGET_DEFINITIONS TypeFreeCapabilityTest.td)
bishengir_target_tablegen(TypeFreeCapabilityTest.h.inc
  -gen-target-capability-decls)
bishengir_target_tablegen(TypeFreeCapabilityTest.cpp.inc
  -gen-target-capability-defs)
add_public_tablegen_target(BiShengIRTargetCapabilityTypeFreeTestIncGen)

add_bishengir_unittest(BiShengIRTargetCapabilityTypeFreeTests
  PARTIAL_SOURCES_INTENDED
  TypeFreeCapabilityTest.cpp
)
target_include_directories(BiShengIRTargetCapabilityTypeFreeTests PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(BiShengIRTargetCapabilityTypeFreeTests
  BiShengIRTargetCapabilityTypeFreeTestIncGen
)
target_link_libraries(BiShengIRTargetCapabilityTypeFreeTests PRIVATE MLIRIR)
