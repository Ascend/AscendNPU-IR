set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

# New bishengir-hfusion-ods-yaml-gen.
add_mlir_tool(bishengir-hfusion-ods-yaml-gen
  bishengir-hfusion-ods-yaml-gen.cpp
)
llvm_update_compile_flags(bishengir-hfusion-ods-yaml-gen)
target_link_libraries(bishengir-hfusion-ods-yaml-gen PRIVATE
  MLIRIR
  MLIRSupport
  MLIRParser
  )

setup_host_tool(bishengir-hfusion-ods-yaml-gen BISHENGIR_HFUSION_ODS_YAML_GEN BISHENGIR_HFUSION_ODS_YAML_GEN_EXE BISHENGIR_HFUSION_ODS_YAML_GEN_TARGET)

if(NOT ${BISHENGIR_HFUSION_ODS_YAML_GEN_EXE} STREQUAL "bishengir-hfusion-ods-yaml-gen")
  add_custom_target(bishengir-hfusion-ods-yaml-gen-host DEPENDS ${BISHENGIR_HFUSION_ODS_YAML_GEN_EXE})

  if(NOT LLVM_BUILD_UTILS)
    set_target_properties(bishengir-hfusion-ods-yaml-gen PROPERTIES EXCLUDE_FROM_ALL ON)
  endif()
endif()