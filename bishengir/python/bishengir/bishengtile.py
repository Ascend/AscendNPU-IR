from bishengir.ir import *
import bishengir.extras.types as Ty
from bishengir.helper import *
from bishengir.dialects import arith, func, hivm, memref

class Ascend:
    @staticmethod
    def check_path(allowed_paths, target_path):
      if target_path not in allowed_paths:
        raise TypeError(
          f"Source memref of address space {target_path[0].__str__} cannot be copied to "
          f"destination memref of address space {target_path[1].__str__}"
        )

    @staticmethod
    def copy_in(src, dst):
      ub_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.UB)
      gm_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.GM)
      cbuf_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.L1)
      # (src, dst)
      target_path = (src.type.memory_space, dst.type.memory_space)
      allowed_memory_spaces_paths =  [(gm_attr, ub_attr), (gm_attr, cbuf_attr)]

      Ascend.check_path(allowed_memory_spaces_paths, target_path)

      return hivm.LoadOp(None, src, dst)

    @staticmethod
    def copy_out(src, dst):
      ub_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.UB)
      gm_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.GM)
      cbuf_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.L1)
      # (src, dst)
      target_path = (src.type.memory_space, dst.type.memory_space)
      allowed_memory_spaces_paths =  [(ub_attr, gm_attr)]

      Ascend.check_path(allowed_memory_spaces_paths, target_path)

      return hivm.StoreOp(None, src, dst)

    @staticmethod
    def copy(src, dst):
      ub_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.UB)
      gm_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.GM)
      cbuf_attr = hivm.addressSpaceAttr.get(hivm.addressSpace.L1)
      # (src, dst)
      target_path = (src.type.memory_space, dst.type.memory_space)
      allowed_memory_spaces_paths =  [(gm_attr, ub_attr), (ub_attr, gm_attr), 
                                      (gm_attr, cbuf_attr), (ub_attr, ub_attr)]

      Ascend.check_path(allowed_memory_spaces_paths, target_path)

      return hivm.CopyOp(None, src, dst)