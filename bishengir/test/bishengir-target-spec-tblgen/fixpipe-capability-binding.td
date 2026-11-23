// clang-format off
// RUN: bishengir-target-spec-tblgen -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include | FileCheck %s
// clang-format on

include "bishengir/Dialect/HIVM/IR/Capability/DMA/Fixpipe/FixpipeCapabilityBinding.td"

// CHECK: class FixpipeCapabilityRequestBuilder
// CHECK: from(::mlir::hivm::FixpipeOp op)
// create() lists every schema field. Dropping one is a compile error.
// CHECK: create(::mlir::Type srcElementType, ::mlir::hivm::AddressSpace srcAddressSpace, ::mlir::Type dstElementType, ::mlir::hivm::AddressSpace dstAddressSpace, ::mlir::hivm::FixpipeDMAMode dmaMode, ::mlir::hivm::FixpipePreQuantMode preQuantMode, ::mlir::hivm::FixpipePreReluMode preRelu, bool channelSplit, bool quantScalePresent)
// CHECK: dyn_cast<::mlir::ShapedType>
// CHECK: getAttr("dma_mode")
// CHECK: getAttr("channel_split")
// An omitted default is applied only when the attribute is absent.
// CHECK: if (!raw)
// A present attribute of the wrong type is not replaced by that default.
// CHECK: if (!typed)
// CHECK: return ::std::nullopt
