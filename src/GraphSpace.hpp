/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of smoothG. For more information and source code
 * availability, see https://www.github.com/llnl/smoothG.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file

    @brief Contains DofHandler class.
 */

#ifndef __DOFHANDLER_HPP__
#define __DOFHANDLER_HPP__

#include "mfem.hpp"
#include "GraphTopology.hpp"

namespace smoothg
{

/**
   @brief Contains information about degrees of freedom to vertex/edge
*/
class DofHandler
{
public:
    /**
       @brief Create a mixed graph in parallel mode.

       @param graph_topology
    */
    DofHandler(const GraphTopology& graph_topology);



private:
    const GraphTopology& graph_topology_;

    mfem::SparseMatrix vertex_vdof_;
    mfem::SparseMatrix vertex_edof_;
    mfem::SparseMatrix edge_edof_;

    mfem::SparseMatrix Agg_vdof_;
    mfem::SparseMatrix Agg_edof_;
    mfem::SparseMatrix face_edof_;

}; // class DofHandler

} // namespace smoothg

#endif /* __DOFHANDLER_HPP__ */
