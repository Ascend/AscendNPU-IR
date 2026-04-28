option(ENABLE_CPU_TRACE_INTRINSIC "If enable debug, use cpu compile." OFF)
message("-- ENABLE_CPU_TRACE_INTRINSIC Value: ${ENABLE_CPU_TRACE_INTRINSIC}")

# Initialize all file lists to nil. Global properties are used instead of
# simple lists because these are modified in multiply nested scopes.
set_property(GLOBAL PROPERTY c220_aic_op_list)
set_property(GLOBAL PROPERTY c220_aiv_op_list)
set_property(GLOBAL PROPERTY c220_mix_aic_op_list)
set_property(GLOBAL PROPERTY c220_mix_aiv_op_list)
set_property(GLOBAL PROPERTY c310_aic_op_list)
set_property(GLOBAL PROPERTY c310_aiv_op_list)
set_property(GLOBAL PROPERTY c310_mix_aic_op_list)
set_property(GLOBAL PROPERTY c310_mix_aiv_op_list)
set_property(GLOBAL PROPERTY noarch_aic_op_list)
set_property(GLOBAL PROPERTY noarch_aiv_op_list)
set_property(GLOBAL PROPERTY all_host_op_list)
set_property(GLOBAL PROPERTY all_linked_bitcode_files)

#[[Compile single xx.cpp to xx.bc
 Core_type: "aic" "aiv" "host"
 Platform_type: "Ascend910B" "Ascend910D"]]
function(compile_single_cpp_to_bc cpp_with_path core_type platform_type)
    # Get name of xx.bc
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})

    set(bisheng_args "-O2")
    list(APPEND bisheng_args -x cce)
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args --cce-auto-sync=off)
    list(APPEND bisheng_args --cce-aicore-only)
    list(APPEND bisheng_args --cce-generic-addrspace=off)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)

    if ((${platform_type} STREQUAL "Ascend910D") AND
        (${cpp_name} STREQUAL "LocalMmad.cpp") OR
        (${cpp_name} STREQUAL "LocalMmadMx.cpp"))
        # list(APPEND bisheng_args "-I$ENV{ASCEND_HOME_PATH}/include/aclnn")
        # list(APPEND bisheng_args "-I$ENV{ASCEND_HOME_PATH}/include")
        # list(APPEND bisheng_args "-I$ENV{ASCEND_HOME_PATH}/include/experiment/runtime")
        # list(APPEND bisheng_args "-I$ENV{ASCEND_HOME_PATH}/include/experiment/msprof")
        # list(APPEND bisheng_args "-I$ENV{ASCEND_HOME_PATH}/compiler/tikcpp")
        # list(APPEND bisheng_args "-I$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw")
        # list(APPEND bisheng_args "-I$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl")
        # list(APPEND bisheng_args "-I$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface")
        list(APPEND bisheng_args "-I${CMAKE_SOURCE_DIR}/../../../bishengir/lib/Template/lib/RegBase/Cube/include")
    endif()

    if ((${core_type} STREQUAL "mix_aic") OR (${core_type} STREQUAL "mix_aiv"))
        list(APPEND bisheng_args -cce-enable-mix)
        list(APPEND bisheng_args -mllvm -enable-mix=true)
    endif()

    if ((${core_type} STREQUAL "aic") OR (${core_type} STREQUAL "mix_aic"))
        set(suffix ".bc")
        if(${platform_type} STREQUAL "Ascend910B")
          list(APPEND bisheng_args --cce-aicore-arch=dav-c220-cube)
          list(APPEND bisheng_args --cce-enable-print)
          list(APPEND bisheng_args --cce-enable-sanitizer)
          set(arch "c220")
        elseif(${platform_type} STREQUAL "Ascend910D")
          list(APPEND bisheng_args --cce-enable-print)
          list(APPEND bisheng_args --cce-aicore-arch=dav-c310-cube)
          set(arch "c310")
        else()
          message(FATAL_ERROR "Unknown platform type: ${platform_type}")
        endif()
        set(suffix ".${core_type}.${arch}${suffix}")
        string(REPLACE ".cpp" ${suffix} bc_name ${cpp_name})
        set_property(GLOBAL APPEND PROPERTY ${arch}_${core_type}_op_list "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    elseif ((${core_type} STREQUAL "aiv") OR (${core_type} STREQUAL "mix_aiv"))
        set(suffix ".bc")
        if(${platform_type} STREQUAL "Ascend910B")
          list(APPEND bisheng_args --cce-aicore-arch=dav-c220-vec)
          list(APPEND bisheng_args --cce-enable-print)
          list(APPEND bisheng_args --cce-enable-sanitizer)
          set(arch "c220")
        elseif(${platform_type} STREQUAL "Ascend910D")
          list(APPEND bisheng_args --cce-enable-print)
          list(APPEND bisheng_args --cce-aicore-arch=dav-c310-vec)
          set(arch "c310")
        else()
          message(FATAL_ERROR "Unknown platform type: ${platform_type}")
        endif()
        set(suffix ".${core_type}.${arch}${suffix}")
        string(REPLACE ".cpp" ${suffix} bc_name ${cpp_name})
        set_property(GLOBAL APPEND PROPERTY ${arch}_${core_type}_op_list "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})
        set_property(GLOBAL APPEND PROPERTY all_host_op_list "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    else ()
        message(FATAL_ERROR "Core type must be aic/mix_aic/aiv/mix_aiv or host.")
    endif ()

    if (${core_type} STREQUAL "host")
        message(STATUS "Running on host platform.")
    else ()
        if (NOT ${platform_type} STREQUAL "Ascend910B" AND NOT ${platform_type} STREQUAL "Ascend310P"
            AND NOT ${platform_type} STREQUAL "Ascend910D")
            message(FATAL_ERROR "Unsupported platform! Currently support Ascend910B, Ascend310P, and Ascend910D.")
        endif ()
    endif ()

    list(APPEND bisheng_args -std=c++17)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -o "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bc_name}
        COMMAND ${BISHENG_TOOL} ${bisheng_args}
        COMMAND_EXPAND_LISTS
        DEPENDS ${cpp_with_path}
    )
    message(STATUS "add_custom_command: "
        "OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bc_name} "
        "COMMAND ${BISHENG_TOOL} ${bisheng_args} "
        "COMMAND_EXPAND_LISTS "
        "DEPENDS ${cpp_with_path}"
    )
    set(local_deps ${local_deps} ${CMAKE_CURRENT_BINARY_DIR}/${bc_name} PARENT_SCOPE)
endfunction()

#[[Compile single xx.cpp to xx_debug.bc
 Core_type: "aic" "aiv" "host"
 Platform_type: "DEBUG"]]
function(compile_single_cpp_to_bc_debug cpp_with_path core_type platform_type)
    # Get name of xx_debug.bc
    string(REGEX MATCHALL "[0-9A-Za-z_]*.cpp" cpp_name ${cpp_with_path})
    set_source_files_properties(${cpp_with_path} PROPERTIES LANGUAGE c++)

    set(bisheng_args "-O2")
    list(APPEND bisheng_args ${cpp_with_path})
    list(APPEND bisheng_args -emit-llvm -c)
    list(APPEND bisheng_args -mllvm -disable-llvm-optzns)

    if (${core_type} STREQUAL "aic")
        string(REGEX REPLACE ".cpp" ".aic.bc" bc_name ${cpp_name})
        # FIXME: noarch_aic_op_list is currently unused.
        set_property(GLOBAL APPEND PROPERTY noarch_aic_op_list "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    elseif (${core_type} STREQUAL "aiv")
        string(REGEX REPLACE ".cpp" ".aiv.bc" bc_name ${cpp_name})
        # FIXME: noarch_aiv_op_list is currently unused.
        set_property(GLOBAL APPEND PROPERTY noarch_aiv_op_list "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    elseif (${core_type} STREQUAL "host")
        string(REGEX REPLACE ".cpp" ".host.bc" bc_name ${cpp_name})
        set_property(GLOBAL APPEND PROPERTY all_host_op_list "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")
    else ()
        message(FATAL_ERROR "Core type must be aic/aiv, host")
    endif ()

    list(APPEND bisheng_args -std=c++17)
    list(APPEND bisheng_args -I ${INCLUDE_PATH})
    list(APPEND bisheng_args -DENABLE_CPU_TRACE_INTRINSIC)
    list(APPEND bisheng_args -o "${CMAKE_CURRENT_BINARY_DIR}/${bc_name}")

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bc_name}
        COMMAND ${BISHENG_TOOL} ${bisheng_args}
        COMMAND_EXPAND_LISTS
        DEPENDS ${cpp_with_path}
    )
    message(STATUS "add_custom_command: "
        "OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bc_name} "
        "COMMAND ${BISHENG_TOOL} ${bisheng_args} "
        "COMMAND_EXPAND_LISTS "
        "DEPENDS ${cpp_with_path}"
    )
    set(local_deps ${local_deps} ${CMAKE_CURRENT_BINARY_DIR}/${bc_name} PARENT_SCOPE)
endfunction()

#[[Compile multi xx.cpp to multi xx.bc
 Core_type: "aic" "aiv"
 Platform_type: "Ascend910B"]]
function(compile_cpp_to_bc)
    set(local_deps "")
    set (core_type " ")
    set (platform_type " ")

    set(core_type ${ARGV0})
    if(NOT ${ARGV0} STREQUAL "host")
        set(platform_type ${ARGV1})
    endif ()

    file(GLOB cpp_list "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    foreach (cpp_with_path ${cpp_list})
        if (ENABLE_CPU_TRACE_INTRINSIC AND
            (NOT ${core_type} STREQUAL "mix_aic") AND
            (NOT ${core_type} STREQUAL "mix_aiv"))
            compile_single_cpp_to_bc_debug(${cpp_with_path} ${core_type} "DEBUG")
        else ()
            compile_single_cpp_to_bc(${cpp_with_path} ${core_type} ${platform_type})
        endif ()
    endforeach ()

    string(REPLACE ${CMAKE_BINARY_DIR} "" local_target_name ${CMAKE_CURRENT_BINARY_DIR})
    string(REGEX REPLACE "[\.\/ ]" "_" local_target_name
           "${local_target_name}_${core_type}_${platform_type}_bitcode")
    add_custom_target(${local_target_name} DEPENDS ${local_deps})
    add_dependencies(bishengir_template_bitcode ${local_target_name})
endfunction()

# Defining this macro helps make the calling functions short and simple, while
# allowing the macro to read and write the variables in the caller's scope.
macro(add_custom_llvm_link_command proplist outfile)
    get_property(tmp GLOBAL PROPERTY ${proplist})
    if(tmp)
        add_custom_command(
            OUTPUT ${outfile}
            COMMAND ${LLVM_LINK_TOOL} ${tmp}
                -o ${outfile}
            COMMAND_EXPAND_LISTS
            DEPENDS ${tmp}
        )
        message(STATUS "add_custom_command: "
            "OUTPUT ${outfile} "
            "COMMAND ${LLVM_LINK_TOOL} ${tmp} "
                "-o ${outfile} "
            "COMMAND_EXPAND_LISTS "
            "DEPENDS ${tmp} "
        )
        list(APPEND local_deps ${outfile})
        set_property(GLOBAL APPEND PROPERTY all_linked_bitcode_files
                     "${outfile}")
    endif()
endmacro()

function(link_one_to_meta_op arch)
    set(local_deps "")
    set(BITCODE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
    file(MAKE_DIRECTORY ${BITCODE_OUTPUT_DIR})
    message(STATUS "BITCODE_OUTPUT_DIR = ${BITCODE_OUTPUT_DIR}")
    set(output_aic_file_name "${BITCODE_OUTPUT_DIR}/meta_op.aic.${arch}.bc")
    set(output_aiv_file_name "${BITCODE_OUTPUT_DIR}/meta_op.aiv.${arch}.bc")
    set(output_mix_aic_file_name "${BITCODE_OUTPUT_DIR}/meta_op.mix.aic.${arch}.bc")
    set(output_mix_aiv_file_name "${BITCODE_OUTPUT_DIR}/meta_op.mix.aiv.${arch}.bc")

    add_custom_llvm_link_command(${arch}_aic_op_list ${output_aic_file_name})
    add_custom_llvm_link_command(${arch}_aiv_op_list ${output_aiv_file_name})
    add_custom_llvm_link_command(${arch}_mix_aic_op_list ${output_mix_aic_file_name})
    add_custom_llvm_link_command(${arch}_mix_aiv_op_list ${output_mix_aiv_file_name})

    set(local_target_name "link_meta_op_${arch}_bitcode")
    add_custom_target(${local_target_name} DEPENDS ${local_deps})
    add_dependencies(bishengir_template_bitcode ${local_target_name})
endfunction()

function(link_all_to_meta_op)
    link_one_to_meta_op("c220")
    link_one_to_meta_op("c310")

    set(local_deps "")
    set(BITCODE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
    message(STATUS "BITCODE_OUTPUT_DIR = ${BITCODE_OUTPUT_DIR}")
    file(MAKE_DIRECTORY ${BITCODE_OUTPUT_DIR})
    set(output_host_file_name "${BITCODE_OUTPUT_DIR}/host-a5.bc")
    message(STATUS "output_host_file_name = ${output_host_file_name}")
    add_custom_llvm_link_command(all_host_op_list ${output_host_file_name})

    set(local_target_name "link_all_to_meta_op_bitcode")
    add_custom_target(${local_target_name} DEPENDS ${local_deps})
    add_dependencies(bishengir_template_bitcode ${local_target_name})
endfunction()

if (BISHENGIR_BUILD_TEMPLATE STREQUAL "ON")
    message("-- BISHENGIR_BUILD_TEMPLATE Value: ${BISHENGIR_BUILD_TEMPLATE}")
    message("-- BISHENG_COMPILER_PATH Value: ${BISHENG_COMPILER_PATH}")

    set(INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(BISHENG_TOOL "${BISHENG_COMPILER_PATH}/ccec")
    set(LLVM_LINK_TOOL "${BISHENG_COMPILER_PATH}/llvm-link")
    if ((NOT EXISTS "${BISHENG_TOOL}") OR (NOT EXISTS "${LLVM_LINK_TOOL}"))
        message(FATAL_ERROR "Cannot find ccec or llvm-link in ${BISHENG_COMPILER_PATH}; "
                            "please check whether ${BISHENG_TOOL} or ${LLVM_LINK_TOOL} "
                            "exists or not!")
    endif ()

    add_custom_target(bishengir_template_bitcode ALL)
    add_subdirectory(lib)
    link_all_to_meta_op()
endif ()

option(ENABLE_SYCL "If enable sycl, use bishengcpp. Otherwise use bisheng." OFF)
message("-- ENABLE_SYCL Value: ${ENABLE_SYCL}")

if (ENABLE_SYCL)
    add_definitions(-DENABLE_SYCL)
endif ()
