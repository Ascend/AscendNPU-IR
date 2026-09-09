// SIMT re-emission of the c310 print chain (aiv only, -DCCE_PRINT_SIMT=1
// from this dir's CMakeLists). Arch gate inherited from ../Debug.cpp.
#define pipe_barrier(...)
#define trap() (*((__gm__ uint8_t *)-1) = 0)  // invalid memory access

#include "../Debug.cpp"
