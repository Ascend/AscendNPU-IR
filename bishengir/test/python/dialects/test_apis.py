# RUN: python3 %s | FileCheck %s

from bishengir.ir import *
from bishengir.helper import *
from bishengir.dialects import arith, func, builtin, hfusion, linalg, hacc
from bishengir.passmanager import *
from bishengir.dialects.linalg.opdsl.lang import *
from bishengir._mlir_libs._bishengirRegisterEverything import register_dialects, register_translations

# Create a global instance of BishengIRHelper
BishengIRHelper.change_print_debug(True)


def run(f):
    BishengIRHelper.print("\nTEST:", f.__name__)
    f()
    return f


@run
def test_example():
    with Context() as ctx, Location.unknown() as loc:
        ctx.allow_unregistered_dialects = True
        register_dialects(ctx)
        register_translations(ctx)
        m = builtin.ModuleOp()

        # Define types
        f32 = F32Type.get()
        f64 = F64Type.get()
        i32 = IntegerType.get_signless(32)
        i16 = IntegerType.get_signless(16)
        tensor_type = RankedTensorType.get((2, 2), i16)

        # Your test code here
        with InsertionPoint(m.body):
            @func.FuncOp.from_py_func(tensor_type, tensor_type, tensor_type)
            def none_return(a, b, c):
                result = hfusion.ElemwiseBinaryOp.create(
                    result_type=tensor_type,
                    input_tensor1=a,
                    input_tensor2=b,
                    output_tensor=c,
                    fun="vor"
                )

                result_2 = hfusion.ElemwiseUnaryOp.create(
                    result_type=tensor_type,
                    input_tensor=c,
                    output_tensor=c,
                    fun="vnot"
                )

                return result, result_2

        # CHECK: test_example
        # CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>}
        # CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
        BishengIRHelper.print(m)


@run
def test_full_bisheng_compile():

    with Context() as ctx, Location.unknown() as loc:
        ctx.allow_unregistered_dialects = True
        register_dialects(ctx)
        register_translations(ctx)

        m = builtin.ModuleOp()
        with InsertionPoint(m.body):
            tensor_type = RankedTensorType.get(
                (ShapedType.get_dynamic_size(),), F32Type.get())

            @func.FuncOp.from_py_func(tensor_type, tensor_type, tensor_type)
            def host_elemwise(arg0, arg1, out):
                result = linalg.elemwise_binary(
                    arg0, arg1, outs=[out],
                    fun=BinaryFn.add
                )
                return result

            hacc.HACCHelper.add_function_kind_attribute(
                host_elemwise.func_op, "HOST")

        BishengIRHelper.print(m)

        pm = PassManager("any")
        pm.add("hacc-append-device-spec{target=Ascend910B1}")
        pm.add(
            "bishengir-compile{enable-hfusion-compile=true "
            "enable-hivm-compile=true "
            "enable-lir-compile=false "
            "o=/tmp/tmp}")
        pm.run(m.operation)
        BishengIRHelper.print(m)
