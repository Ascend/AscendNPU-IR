/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
 
#ifndef CATLASS_EPILOGUE_FUSION_FUSION_HPP
#define CATLASS_EPILOGUE_FUSION_FUSION_HPP
 
#include "catlass/epilogue/fusion/operations.hpp"
#include "catlass/epilogue/fusion/visitor_impl_base.hpp"
#include "catlass/epilogue/fusion/visitor_impl.hpp"
#include "catlass/epilogue/fusion/visitor_acc_load.hpp"
#include "catlass/epilogue/fusion/visitor_aux_load.hpp"
#include "catlass/epilogue/fusion/visitor_compute.hpp"
#include "catlass/epilogue/fusion/visitor_aux_store.hpp"
#include "catlass/epilogue/fusion/visitor_cast.hpp"
#include "catlass/epilogue/fusion/visitor_row_broadcast.hpp"
#include "catlass/epilogue/fusion/tree_visitor.hpp"
#include "catlass/epilogue/fusion/topological_visitor.hpp"
 
#endif
 
 