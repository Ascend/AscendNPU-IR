// REQUIRES: hivmc
// RUN: bishengir-compile %s --target=Ascend950PR_957c --enable-hfusion-compile=true --enable-triton-kernel-compile=true -o %t.o

// legalize-scalar must route both min/max through the existing tensor pipeline.
// The trailing 6 i32 arguments are required by the Triton kernel ABI.
module attributes {hacc.target = #hacc.target<"Ascend950PR_957c">} {
  func.func @kernel(%x: f16, %lower: f16, %upper: f16, %out: memref<1xf16>,
                    %num_x: i32, %num_y: i32, %num_z: i32,
                    %id_x: i32, %id_y: i32, %id_z: i32)
      attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %min = arith.minimumf %x, %upper : f16
    %clamp = arith.maximumf %lower, %min : f16
    %result = tensor.from_elements %clamp : tensor<1xf16>
    bufferization.materialize_in_destination %result in writable %out : (tensor<1xf16>, memref<1xf16>) -> ()
    return
  }
}
