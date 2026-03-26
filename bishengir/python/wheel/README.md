# AscendNPU IR Compiler

Python bindings for the bishengir-compile compiler.

## Installation

```bash
pip install ascendnpuir
```

## Usage

```python
import ascendnpuir

mlir_str = """
func.func @example() -> tensor<1x10x10xf32> {
  %0 = constant 0.0 : tensor<1x10x10xf32>
  return %0 : tensor<1x10x10xf32>
}
"""
output_path = "exmaple.o"
options = [
    "-enable-hfusion-compile=true",
    "-enable-triton-kernel-compile",
    "target=Ascend910_950z",
]

# Compile a model
res = ascendnpuir.compile(
    mlir_str,
    output_file=output_path,
    options=options,
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

- Python 3.8 or higher
- bishengir-compile binary (built from source)

## License

Apache License 2.0
