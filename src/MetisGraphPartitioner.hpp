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

    @brief Defines object for partitioning graphs with Metis.
*/

#ifndef __METISGRAPHPARTITIONER_HPP__
#define __METISGRAPHPARTITIONER_HPP__

#include <functional>
#include <numeric>
#include <metis.h>
#include "mfem.hpp"
#include "MatrixUtilities.hpp"

namespace smoothg
{

/**
   @brief Wrap Metis in a C++ class for partitioning a graph.
*/
class MetisGraphPartitioner
{
public:

    //! Flags
    enum class PartType
    {
        KWAY,
        RECURSIVE,
        TOTALCOMMUNICATION
    };

    //! Constructor: initialize default options for the partitioner.
    explicit MetisGraphPartitioner(PartType _part_time = PartType::KWAY);

    //! Destructor
    virtual ~MetisGraphPartitioner() { };

    //! Set flags
    void setFlags(PartType type)
    {
        part_type_ = type;
    }

    //! Set non-default options for metis
    void setOption(const int i, const int val)
    {
        options_[i] = val;
    }

    //! Allow some imbalance in the size of the partitions
    void setUnbalanceTol(double utol)
    {
        unbalance_tol_ = utol;
    }

    //! Partition a graph with weighted edges in num_partitions
    /*!
     * @param wtable a num_vertexes by num_vertexes wtable representing the connectivity of the graph.
     *               table has an entry (i,j) if there is an edge between vertex i and vertex j.
     *               The weight of the edge is the value of the matrix
     *
     * @param num_partitions number of partitions in which we want to divide the graph
     *
     * @param partitioning vector of size num_vertexes. partitioning[v] = p if vertex v belongs to partition p. (OUT).
     */
    void doPartition(const mfem::SparseMatrix& wtable,
                     int& num_partitions,
                     mfem::Array<int>& partitioning,
                     bool use_edge_weight = false);

    // TODO: move this function to MatrixUtilities?
    static mfem::SparseMatrix getAdjacency(const mfem::SparseMatrix& wtable);

    /**
       Isolate some critical vertices of the graph into their own partition, so
       that these vertices are not coarsened but remain on coarser levels.

       (this does a deep copy of the indices)
    */
    void SetPreIsolateVertices(int index);
    void SetPreIsolateVertices(const mfem::Array<int>& indices);
    void SetPostIsolateVertices(int index);
    void SetPostIsolateVertices(const mfem::Array<int>& indices);

private:
    int options_[METIS_NOPTIONS];
    int flags_;

    PartType part_type_;
    real_t unbalance_tol_;
    mfem::Array<int> pre_isolated_vertices_;
    mfem::Array<int> post_isolated_vertices_;

    void removeEmptyParts(mfem::Array<int>& partitioning,
                          int& num_partitions) const;

    int connectedComponents(mfem::Array<int>& partitioning,
                            const mfem::SparseMatrix& conn);

    void IsolatePreProcess(const mfem::SparseMatrix& wtable,
                           mfem::SparseMatrix& sub_table,
                           mfem::Array<int>& sub_to_global);

    void IsolatePostProcess(const mfem::SparseMatrix& wtable,
                            int& num_partitions,
                            mfem::Array<int>& partitioning);

    std::function<int(int*, int*, int*, int*, int*, int*,
                      int*, int*, float*, float*, int*, int*, int*)> CallMetis;
};

void Partition(const mfem::SparseMatrix& w_table, mfem::Array<int>& partitioning,
               int num_parts, bool use_edge_weight = false);

void PartitionAAT(const mfem::SparseMatrix& vertex_edge,
                  mfem::Array<int>& partitioning, int coarsening_factor);

} // namespace smoothg

#endif
