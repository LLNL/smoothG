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

#ifndef _REDISTRIBUTOR_HPP_
#define _REDISTRIBUTOR_HPP_

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "matred.hpp"

#include "GraphTopology.hpp"
#include "MixedMatrix.hpp"

namespace smoothg
{

unique_ptr<mfem::HypreParMatrix> Move(matred::ParMatrix& A);

void Mult(const mfem::HypreParMatrix& A, const mfem::Array<int>& x, mfem::Array<int>& Ax);

// From the parallel proc-to-proc connectivity table,
// get a copy of the global matrix as a serial matrix locally (via permutation),
// and then call METIS to "partition processors" in each processor locally
std::vector<int> RedistributeElements(
    const mfem::HypreParMatrix& elem_face, int& num_redist_procs);

/// A helper to redistribute AgglomeratedTopology, DofHandler, DeRhamSequence
class Redistributor
{
    using ParMatrix = matred::ParMatrix;

    unique_ptr<mfem::HypreParMatrix> redVertex_vertex_;
    unique_ptr<mfem::HypreParMatrix> redTrueEdge_trueEdge_;
    unique_ptr<mfem::HypreParMatrix> redEdge_trueEdge_;
    unique_ptr<mfem::HypreParMatrix> redVDOF_VDOF_;
    unique_ptr<mfem::HypreParMatrix> redTrueEDOF_trueEDOF_;
    unique_ptr<mfem::HypreParMatrix> redEDOF_trueEDOF_;

    unique_ptr<mfem::HypreParMatrix> BuildRedEntToTrueEnt(
        const mfem::HypreParMatrix& vert_trueEntity) const;

    unique_ptr<mfem::HypreParMatrix> BuildRedEntToRedTrueEnt(
        const mfem::HypreParMatrix& redEntity_trueEntity) const;

    unique_ptr<mfem::HypreParMatrix> BuildRedTrueEntToTrueEnt(
        const mfem::HypreParMatrix& redEntity_redTrueEntity,
        const mfem::HypreParMatrix& redEntity_trueEntity) const;

    unique_ptr<mfem::HypreParMatrix>
    BuildRepeatedEDofToTrueEDof(const GraphSpace& space) const;

    unique_ptr<mfem::HypreParMatrix>
    BuildRepeatedEDofRedistribution(const GraphSpace& space,
                                    const GraphSpace& redist_space) const;

    void Init(const GraphSpace& space, const std::vector<int>& vert_redist_procs);
public:

    /// Constructor for Redistributor
    /// A redistributed topology will be constructed and stored in the class
    /// @param graph graph in the original data distribution
    /// @param vert_redist_procs an array of size number of local vertices.
    /// vert_redist_procs[i] indicates which processor the i-th local vertex
    /// will be redistributed to. Other entities are redistributed accordingly.
    Redistributor(const GraphSpace& space, const std::vector<int>& vert_redist_procs);

    /// @param num_redist_procs number of processors to be redistributed to
    Redistributor(const GraphSpace& space, int& num_redist_procs);

    ///  Get redistribution matrix of true EDGE or VERTEX
    const mfem::HypreParMatrix& TrueEntityRedistribution(EntityType entity) const;

    ///  Get redistribution matrix of true EDOF or VDOF
    const mfem::HypreParMatrix& TrueDofRedistribution(DofType dof) const;

    /// Redistribute a graph
    Graph RedistributeGraph(const Graph& graph) const;

    /// Redistribute a graph space
    GraphSpace RedistributeSpace(const GraphSpace& space) const;

    /// Redistribute a mixed matrix
    MixedMatrix RedistributeMatrix(const MixedMatrix& system) const;
};

} // namespace parelag

#endif // _REDISTRIBUTOR_HPP_
