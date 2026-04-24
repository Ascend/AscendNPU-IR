# AscendNPU IR Compiler

Python bindings for the bishengir-compile compiler.

## Installation

```bash
pip install ascendnpu-ir
```

## Usage

```python
import ascendnpuir

mlir_str = """
module {
  func.func @add(%arg0: memref<16xi16, #hivm.address_space<gm>>, %arg1: memref<16xi16, #hivm.address_space<gm>>, %arg2: memref<16xi16, #hivm.address_space<gm>>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %alloc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg0 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_0 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xi16, #hivm.address_space<gm>>) outs(%alloc_0 : memref<16xi16, #hivm.address_space<ub>>)
    %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xi16, #hivm.address_space<ub>>, memref<16xi16, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>) outs(%arg2 : memref<16xi16, #hivm.address_space<gm>>)
    return
  }
}
"""
output_path = "example.o"
options = [
    "-enable-hivm-compile=true",
]

# Compile a model
res = ascendnpuir.compile(
    mlir_str,
    output_path=output_path,
    option=options,
)
print(f"Compiled to: {output_path}")
```

## Building from Source

To build the wheel package from source, you need to first build the bishengir-compile binary:

```bash
# Build the compiler
cd build-tools
./build.sh

# Build the wheel package
./build_wheel.sh
```

The wheel package will be created in the `bishengir/python/wheel/dist` directory.

## Requirements

- Python 3.9 or higher
- bishengir-compile binary (built from source)

## License

Apache License 2.0
