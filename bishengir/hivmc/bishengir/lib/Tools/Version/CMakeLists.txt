set(vcs_version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSVersion.inc")

configure_file(
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Version/Version.inc.in
  ${BISHENGIR_INCLUDE_DIR}/bishengir/Version/Version.inc)

file(WRITE ${vcs_version_inc} "#define BISHENGIR_REPOSITORY \"\"\n#define BISHENGIR_REVISION \"\"\n#define LLVM_REPOSITORY \"\"\n#define LLVM_REVISION \"\"\n")

set_source_files_properties(${vcs_version_inc}
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_bishengir_library(BiShengIRVersionLib
  Version.cpp
  ${vcs_version_inc})
