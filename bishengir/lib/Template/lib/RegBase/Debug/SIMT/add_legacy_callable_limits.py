"""Add legacy CCEC resource metadata to the A5 SIMT debug template."""

import argparse
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory


def add_callable_limits(ir, max_registers):
    header = re.compile(
        r"^define [^\n]*?\bsimt_callable\b[^@\n]*@(?P<symbol>[^\s(]+)\([^\n]* \{$",
        re.MULTILINE,
    )
    node_ids = [int(node) for node in re.findall(r"^!(\d+) = ", ir, re.MULTILINE)]
    annotation_id = max(node_ids, default=-1) + 1
    existing_limits = set(
        re.findall(
            r'^!(\d+) = (?:distinct )?!\{!"simt-max-(?:registers|threads)", i32 \d+\}$',
            ir,
            re.MULTILINE,
        )
    )
    candidates = set()
    added = set()

    def annotate(match):
        symbol = match["symbol"]
        if symbol.startswith("_mlir_ciface_assert_") and symbol.endswith("_simt"):
            candidates.add("assert")
        elif symbol.startswith("_mlir_ciface_print_") and symbol.endswith("_simt"):
            candidates.add("print")
        elif symbol.startswith("_ZN3cceL6printfI"):
            candidates.add("printf")
        else:
            return match[0]

        if existing_limits.intersection(re.findall(r" !annotation !(\d+)", match[0])):
            return match[0]
        added.add(symbol)
        return match[0][:-2] + f" !annotation !{annotation_id} {{"

    updated = header.sub(annotate, ir)
    if candidates != {"assert", "print", "printf"}:
        raise ValueError("Expected SIMT assert, print and cce::printf definitions in the debug template")
    if added:
        updated += f'!{annotation_id} = !{{!"simt-max-registers", i32 {max_registers}}}\n'
    print(f"Added legacy register metadata to {len(added)} SIMT debug functions")
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llvm_link", type=Path)
    parser.add_argument("max_registers", type=int)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    if not 0 < options.max_registers < 2**31:
        parser.error("max_registers must be a positive i32")

    with TemporaryDirectory(dir=options.output.parent) as scratch:
        workdir = Path(scratch)
        ir = workdir / "debug.ll"
        bitcode = workdir / "debug.bc"
        subprocess.run([options.llvm_link, "-S", options.input, "-o", ir], check=True)
        ir.write_text(add_callable_limits(ir.read_text(), options.max_registers))
        subprocess.run([options.llvm_link, ir, "-o", bitcode], check=True)
        bitcode.replace(options.output)


if __name__ == "__main__":
    main()
