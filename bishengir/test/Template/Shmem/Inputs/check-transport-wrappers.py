import pathlib
import re
import subprocess
import sys

lib_dir, compiler = sys.argv[1:3]
roce_backend = sys.argv[3] if len(sys.argv) > 3 else "IN_DIE"
relay_enabled = len(sys.argv) > 4 and sys.argv[4].upper() in ("ON", "TRUE", "1", "YES", "Y")
types = [
    "half", "float", "int8", "int16", "int32", "int64", "uint8",
    "uint16", "uint32", "uint64", "char", "bfloat16",
]
# C wrappers receive pointers to rank-one memref descriptors. Count, PE, relay
# PE, and event ID remain 32-bit scalar arguments.
operations = {
    "roce_get_nbi": ["ptr", "ptr", "i32", "i32"],
    "roce_put_nbi": ["ptr", "ptr", "i32", "i32"],
    "roce_qp_get_nbi": ["ptr", "ptr", "i32", "i32", "i32"],
    "roce_qp_put_nbi": ["ptr", "ptr", "i32", "i32", "i32"],
    "roce_qp_quiet": ["i32", "i32"],
    "udma_qp_get_nbi": ["ptr", "ptr", "i32", "i32", "i32"],
    "udma_qp_put_nbi": ["ptr", "ptr", "i32", "i32", "i32"],
    "udma_qp_put_signal_nbi": ["ptr", "ptr", "i32", "ptr", "i64", "i32", "i32"],
    "roce_quiet": ["i32"],
    "udma_get_nbi": ["ptr", "ptr", "i32", "i32"],
    "udma_put_nbi": ["ptr", "ptr", "i32", "i32"],
    "udma_putmem_signal_nbi": ["ptr", "ptr", "i32", "ptr", "i64", "i32"],
}
relay_operations = {
    "udma_relay_get_nbi": ["ptr", "ptr", "i32", "i32", "i32"],
    "udma_relay_put_nbi": ["ptr", "ptr", "i32", "i32", "i32"],
}
collectives = {
    "udma_quiet": ("void", ["i32"]),
    "udma_qp_quiet": ("void", ["i32", "i32"]),
    "roce_team_sync": ("i32", ["i32"]),
    "roce_barrier": ("i32", ["i32"]),
    "roce_sync_all": ("void", []),
    "roce_barrier_all": ("void", []),
}


def check(definitions, name, result, arguments):
    assert name in definitions, f"Missing definition: {name}"
    actual_result, params = definitions[name]
    assert actual_result == result, (name, actual_result, result)
    actual = params.split(",") if params else []
    assert len(actual) == len(arguments), (name, actual)
    for arg, expected in zip(actual, arguments):
        assert arg.strip().startswith(expected + " "), (name, arg, expected)


for arch in ("c220", "c310"):
    for core in ("aic", "aiv", "mix.aic", "mix.aiv"):
        bitcode = pathlib.Path(lib_dir) / f"meta_op.{core}.{arch}.bc"
        # CANN device bitcode can contain extensions that the community LLVM
        # reader cannot decode. Use its matching reader without running passes.
        ir = subprocess.check_output([
            compiler, "-cc1", "-disable-llvm-passes", "-triple",
            "hiipu64-hisilicon-cce", "-x", "ir", "-emit-llvm",
            str(bitcode), "-o", "-",
        ], text=True)
        definitions = {}
        for result, name, params in re.findall(
                r"^define .*?\b(void|i32|i64|float) @([^ (]+)\(([^\n]*)\)", ir, re.MULTILINE):
            # Mixed-core bitcode decorates symbols for the destination core.
            name = re.sub(r"\.(vector|cube)$", "", name)
            definitions[name] = (result, params)
        for op, args in operations.items():
            for ty in types:
                name = f"_mlir_ciface_aclshmemx_{op}_{ty}"
                if op.startswith("roce_qp_") and roce_backend == "IN_DIE":
                    assert name not in definitions, name
                else:
                    check(definitions, name, "void", args)
        for op, args in relay_operations.items():
            for ty in types:
                name = f"_mlir_ciface_aclshmemx_{op}_{ty}"
                if arch == "c310" and relay_enabled:
                    check(definitions, name, "void", args)
                else:
                    assert name not in definitions, name
        for op, (result, args) in collectives.items():
            check(definitions, f"_mlir_ciface_aclshmemx_{op}", result, args)
        if arch == "c310":
            for transport in ("roce", "udma"):
                for ty, scalar in (("int32", "i32"), ("uint32", "i32"),
                                   ("int64", "i64"), ("uint64", "i64")):
                    for op in ("fetch", "fetch_inc", "inc", "set", "swap",
                               "add", "and", "or", "xor", "fetch_add",
                               "fetch_and", "fetch_or", "fetch_xor", "compare_swap"):
                        operands = 0 if op in ("fetch", "fetch_inc", "inc") else 1
                        if op == "compare_swap":
                            operands = 2
                        result = scalar if (op.startswith("fetch") or
                                            op in ("swap", "compare_swap")) else "void"
                        check(definitions,
                              f"_mlir_ciface_aclshmemx_{transport}_atomic_{op}_{ty}",
                              result, ["ptr"] + [scalar] * operands + ["i32"])
            for op in ("fetch", "set", "swap"):
                name = f"_mlir_ciface_aclshmemx_roce_atomic_{op}_float"
                if roce_backend == "HNS_1825":
                    assert name not in definitions, name
                else:
                    check(definitions, name, "void" if op == "set" else "float",
                          ["ptr"] + ([] if op == "fetch" else ["float"]) + ["i32"])
        print(f"Verified transport wrapper ABI: {core}.{arch}")
