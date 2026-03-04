# RUN: python3 %s | FileCheck %s

from bishengir.ir import *
import bishengir.extras.types as Ty
from bishengir.helper import *
from bishengir.dialects import arith, func, hivm, memref, hivmave
from bishengir.passmanager import *
from bishengir.dialects.linalg.opdsl.lang import *

from bishengir._mlir_libs._bishengirRegisterEverything import register_dialects, register_translations


with Context() as ctx, Location.unknown():
    ctx.allow_unregistered_dialects = True
    register_dialects(ctx)
    register_translations(ctx)
    module = Module.create()
    UB = hivm.addressSpaceAttr.get(hivm.addressSpace.UB)
    inTy = Ty.memref(64, Ty.i32(),memory_space=UB)
    pipe_v = hivm.pipeAttr.get(hivm.pipe.v)
    pipe_all = hivm.pipeAttr.get(hivm.pipe.all)
    pipe_mte2 = hivm.pipeAttr.get(hivm.pipe.mte2)
    event_id0 = hivm.eventAttr.get(hivm.event.id_0)
    # CHECK: test_vf
    # CHECK: hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    # CHECK: hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    # CHECK: hivm.hir.pipe_barrier[<PIPE_ALL>]
    with InsertionPoint(module.body), Location.unknown():
        funcOp = func.FuncOp("test_vf", ([inTy,inTy], []))
        set_flag = hivm.SetFlagOp(set_pipe=pipe_v, wait_pipe=pipe_mte2, static_event_id=event_id0)
        wait_flag = hivm.WaitFlagOp(set_pipe=pipe_v, wait_pipe=pipe_mte2, static_event_id=event_id0)
        bar_all = hivm.PipeBarrierOp(pipe_all)
        block = funcOp.add_entry_block()
        arg0, arg1 = block.arguments

        with InsertionPoint(block):
            c0 = arith.ConstantOp(IndexType.get(), 0)
            vecTy,maskTy = Ty.vector(64, Ty.i32()), Ty.vector(256, Ty.bool())
            load_dist = hivmave.LoadDist.BRC_B32
            vlds = hivmave.VFLoadOp(vecTy,load_dist,arg0,[c0])
            store_dist = hivmave.StoreDist.NORM_B32

            mask = hivmave.VFPgeOp(maskTy, hivmave.PgePattern.ALL)
            
            vsts = hivmave.VFMaskedStoreOp(store_dist, arg1, [c0], mask,vlds)

            func.ReturnOp([])
        
    print(module)
