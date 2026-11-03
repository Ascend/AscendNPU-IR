# The regular configure command specifies the CANN bin directory. Use its
# installation root when the shell has not exported ASCEND_HOME_PATH.
set(BISHENGIR_SHMEM_CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
if(NOT BISHENGIR_SHMEM_CANN_ROOT)
    get_filename_component(BISHENGIR_SHMEM_CANN_ROOT
        "${BISHENG_COMPILER_PATH}/.." ABSOLUTE)
endif()

set(BISHENGIR_SHMEM_API_DEFINES)
option(BISHENGIR_SHMEM_RELAY_SUPPORT
    "Enable C310 SHMEM relay wrappers; requires a matching relay-enabled runtime" OFF)
set(BISHENGIR_SHMEM_RDMA_BACKEND "IN_DIE" CACHE STRING
    "SHMEM RoCE backend; must match the host runtime (IN_DIE, XSCALE, HNS_1825)")
set_property(CACHE BISHENGIR_SHMEM_RDMA_BACKEND PROPERTY STRINGS
    IN_DIE XSCALE HNS_1825)
if(BISHENGIR_SHMEM_RDMA_BACKEND STREQUAL "XSCALE" OR
   BISHENGIR_SHMEM_RDMA_BACKEND STREQUAL "HNS_1825")
    list(APPEND BISHENGIR_SHMEM_API_DEFINES
        "-DACLSHMEMI_RDMA_K_BACKEND_${BISHENGIR_SHMEM_RDMA_BACKEND}=1")
elseif(NOT BISHENGIR_SHMEM_RDMA_BACKEND STREQUAL "IN_DIE")
    message(FATAL_ERROR "Unknown SHMEM RoCE backend: ${BISHENGIR_SHMEM_RDMA_BACKEND}")
endif()

file(GLOB cpp_list CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach (cpp_with_path ${cpp_list})
    compile_shmem_to_bc(${cpp_with_path} "aic" "Ascend910B")
    compile_shmem_to_bc(${cpp_with_path} "aiv" "Ascend910B")
    compile_shmem_to_bc(${cpp_with_path} "mix_aiv" "Ascend910B")
    compile_shmem_to_bc(${cpp_with_path} "mix_aic" "Ascend910B")
    compile_shmem_to_bc(${cpp_with_path} "aic" "Ascend910D")
    compile_shmem_to_bc(${cpp_with_path} "aiv" "Ascend910D")
    compile_shmem_to_bc(${cpp_with_path} "mix_aiv" "Ascend910D")
    compile_shmem_to_bc(${cpp_with_path} "mix_aic" "Ascend910D")
    if (BISHENGIR_BUILD_TEMPLATE_ASCEND960)
        compile_shmem_to_bc(${cpp_with_path} "aic" "Ascend960")
        compile_shmem_to_bc(${cpp_with_path} "aiv" "Ascend960")
        compile_shmem_to_bc(${cpp_with_path} "mix_aiv" "Ascend960")
        compile_shmem_to_bc(${cpp_with_path} "mix_aic" "Ascend960")
    endif ()
endforeach()
