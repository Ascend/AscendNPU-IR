set(LLVM_LINK_COMPONENTS
  CodeGenTypes
  Demangle
  Support
  TableGen
)

add_tablegen(bishengir-target-spec-tblgen BISHENGIR_TARGET_SPEC
  bishengir-target-spec-tblgen.cpp
  TargetSpecGen.cpp
)

target_link_libraries(bishengir-target-spec-tblgen
  PRIVATE
  MLIRTblgenLib
)

mlir_check_all_link_libraries(bishengir-target-spec-tblgen)
