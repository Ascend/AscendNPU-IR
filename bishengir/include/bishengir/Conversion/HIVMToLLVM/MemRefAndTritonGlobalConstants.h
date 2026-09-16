#ifndef BISHENGIR_MEMREF_AND_TRITON_GLOBAL_CONSTANTS_H
#define BISHENGIR_MEMREF_AND_TRITON_GLOBAL_CONSTANTS_H

// TritonGlobal
static constexpr unsigned kAlignedPtrPosInMemRefDescriptor = 0;
static constexpr unsigned kAllocatedPtrPosInMemRefDescriptor = 1;
static constexpr unsigned kOffsetPosInMemRefDescriptor = 2;
static constexpr unsigned kSizePosInMemRefDescriptor = 3;
static constexpr unsigned kStridePosInMemRefDescriptor = 4;
static constexpr unsigned kMemRefDescriptorArgsNum = 5;


// ConvertMemRefToBarePtr
static constexpr unsigned kMemRefAllocatedPtrPosInMemRefDescriptor = 0;
static constexpr unsigned kMemRefAlignedPtrPosInMemRefDescriptor = 1;
#endif
