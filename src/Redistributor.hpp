/*
  Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef _REDISTRIBUTOR_HPP_
#define _REDISTRIBUTOR_HPP_

#include <memory>
#include <vector>

#include "mfem.hpp"
#include "matred.hpp"

//#include "topology/Topology.hpp"
//#include "structures/BooleanMatrix.hpp"
//#include "structures/SharingMap.hpp"
//#include "topology/TopologyTable.hpp"
//#include "utilities/elagError.hpp"
//#include "partitioning/MetisGraphPartitioner.hpp"
//#include "amge/DofHandler.hpp"
// #include "amge/DeRhamSequence.hpp"

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

    // Enumeration convention follows the ones in AgglomeratedTopology/DofHandler
    std::vector<unique_ptr<mfem::HypreParMatrix> > redTrueEntity_trueEntity;
    std::vector<unique_ptr<mfem::HypreParMatrix> > redEntity_trueEntity;
    std::vector<unique_ptr<mfem::HypreParMatrix> > redTrueDof_trueDof;
    std::vector<unique_ptr<mfem::HypreParMatrix> > redDof_trueDof;

    Graph redist_graph_;

    Graph Redistribute(const Graph& graph);

    GraphSpace Redistribute(const GraphSpace& dof);

    unique_ptr<mfem::HypreParMatrix> BuildRedEntToTrueEnt(
        const mfem::HypreParMatrix& elem_trueEntity) const;

    unique_ptr<mfem::HypreParMatrix> BuildRedEntToRedTrueEnt(
        const mfem::HypreParMatrix& redEntity_trueEntity) const;

    unique_ptr<mfem::HypreParMatrix> BuildRedTrueEntToTrueEnt(
        const mfem::HypreParMatrix& redEntity_redTrueEntity,
        const mfem::HypreParMatrix& redEntity_trueEntity) const;

    unique_ptr<mfem::HypreParMatrix>
    BuildRepeatedEDofToTrueEDof(const GraphSpace& dof);

    unique_ptr<mfem::HypreParMatrix>
    BuildRepeatedEDofRedistribution(const GraphSpace& dof,
                                    const GraphSpace& redist_dof);

    void Init(const Graph& graph, const std::vector<int>& elem_redist_procs);
public:

    /// Constructor for Redistributor
    /// A redistributed topology will be constructed and stored in the class
    /// @param graph graph in the original data distribution
    /// @param elem_redist_procs an array of size number of local elements.
    /// elem_redist_procs[i] indicates which processor the i-th local element
    /// will be redistributed to. Other entities are redistributed accordingly.
    Redistributor(const Graph& graph, const std::vector<int>& elem_redist_procs);

    /// @param num_redist_procs number of processors to be redistributed to
    Redistributor(const Graph& graph, int& num_redist_procs);

    const mfem::HypreParMatrix& TrueEntityRedistribution(int codim) const
    {
        return *(redTrueEntity_trueEntity[codim]);
    }

    const mfem::HypreParMatrix& TrueDofRedistribution(int jform) const
    {
        return *(redTrueDof_trueDof[jform]);
    }

    Graph& GetRedistributedGraph() { return redist_graph_; }

    MixedMatrix Redistribute(const MixedMatrix& seq);
};

} // namespace parelag

#endif // _REDISTRIBUTOR_HPP_
