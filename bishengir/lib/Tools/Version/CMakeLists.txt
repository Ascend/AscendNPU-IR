# if(BISHENGIR_VERSION_STRING)
#   set(BISHENGIR_FULL_VERSION_STRING ${BISHENGIR_VERSION_STRING})
# endif()

# Configure the VCSVersion.inc file.
set(vcs_version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSVersion.inc")
set(generate_vcs_version_script "${LLVM_CMAKE_DIR}/GenerateVersionFromVCS.cmake")

find_first_existing_vc_file("${BISHENGIR_SRC_DIR}" bishengir_vc)

if(bishengir_vc AND LLVM_APPEND_VC_REV)
  set(bishengir_src_dir ${BISHENGIR_SRC_DIR})
endif()

message("generate_vcs_version_script: " ${generate_vcs_version_script})
message("bishengir_src_dir: " ${bishengir_src_dir})
message("vcs_version_inc: " ${vcs_version_inc})
message("bishengir_vc: " ${bishengir_vc})
message("LLVM_MAIN_SRC_DIR: " ${LLVM_MAIN_SRC_DIR})

add_custom_command(OUTPUT "${vcs_version_inc}"
  DEPENDS "${bishengir_vc}" "${generate_vcs_version_script}"
  COMMAND ${CMAKE_COMMAND} "-DNAMES=BISHENGIR;LLVM"
                           "-DBISHENGIR_SOURCE_DIR=${bishengir_src_dir}"
                           "-DLLVM_SOURCE_DIR=${LLVM_MAIN_SRC_DIR}"
                           "-DHEADER_FILE=${vcs_version_inc}"
                           "-DLLVM_FORCE_VC_REVISION=${LLVM_FORCE_VC_REVISION}"
                           "-DLLVM_FORCE_VC_REPOSITORY=${LLVM_FORCE_VC_REPOSITORY}"
                           -P "${generate_vcs_version_script}"
  VERBATIM
)

set_source_files_properties("${vcs_version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

# Configure the Version.inc file.
set(version_inc "${BISHENGIR_BINARY_DIR}/bishengir/include/bishengir/Version/Version.inc")

configure_file(
  ${BISHENGIR_SRC_DIR}/include/bishengir/Version/Version.inc.in
  ${version_inc})

set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_bishengir_library(BiShengIRVersionLib 
  Version.cpp
  ${vcs_version_inc}
  ${version_inc})
