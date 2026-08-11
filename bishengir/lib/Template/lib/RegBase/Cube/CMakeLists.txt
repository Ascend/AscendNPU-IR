set(A5_REGBASE_CUBE_SOURCES
    Copy1D.cpp
    Fixpipe.cpp
    GlobalMmad.cpp
    LocalMmad.cpp
    LocalMmadMx.cpp
    l12ub.cpp
    loadMXScale.cpp
    nd2nz.cpp
    set2d.cpp
)

compile_cpp_to_bc("aic" "Ascend910D" ${A5_REGBASE_CUBE_SOURCES})