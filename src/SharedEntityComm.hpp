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
*/

#ifndef __SHAREDENTITYCOMM_HPP__
#define __SHAREDENTITYCOMM_HPP__

#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"

#include "Utilities.hpp"

namespace smoothg
{

template <typename T>
class SharedEntityComm
{
    public:
        SharedEntityComm(const ParMatrix& entity_true_entity);

    private:
        const ParMatrix& entity_true_entity_;

        MPI_Comm comm_;
        MPI_Comm myid_;

        int num_entities_;

        std::vector<int> entity_master_;
        std::vector<int> entity_slave_id_;

        int num_master_comms_;
        int num_slave_comms_;

        linalgcpp::SparseMatrix<int> entity_proc_;
};

template <typename T>
SharedEntityComm<T>::SharedEntityComm(const ParMatrix& entity_true_entity)
    : entity_true_entity_(entity_true_entity),
      comm_(entity_true_entity_.GetComm()),
      num_entities_(entity_true_entity_.Rows()),
      entity_master_(num_entities_),
      entity_slave_id_(num_entities_),
      num_master_comms_(0),
      num_slave_comms_(0)
{
    MPI_Comm_rank(comm_, &myid_);

    auto comm_pkg = entity_true_entity_.MakeCommPkg();

    auto& send_starts = comm_pkg.send_map_starts_;
    auto& recv_starts = comm_pkg.recv_vec_starts_;

    std::vector<std::pair<int, int> > true_entity_proc;

    int num_sends = comm_pkg.num_sends_;

    for (int send = 0; send < num_sends; ++send)
    {
        int proc = comm_pkg.send_procs_[send];

        for (int j = send_starts[send]; j < send_starts[send + 1]; ++j)
        {
            int true_entity = comm_pkg.send_map_elmts_[j];

            true_entity_proc.push_back(std::make_pair(true_entity, proc));
        }
    }

    const auto& ete_diag = entity_true_entity_.GetDiag();
    const auto& ete_diag_indptr = ete_diag.GetIndptr();
    const auto& ete_diag_indices = ete_diag.GetIndices();

    const auto& ete_offd = entity_true_entity_.GetOffd();
    const auto& ete_offd_indptr = ete_offd.GetIndptr();
    const auto& ete_offd_indices = ete_offd.GetIndices();

    linalgcpp::CooMatrix<int> entity_proc;

    for (int entity = 0; entity < num_entities_; ++entity)
    {
        int diag_size = ete_diag.RowSize(entity);
        int offd_size = ete_offd.RowSize(entity);

        entity_master_[entity] = myid_;
        entity_proc.Add(entity, myid_, 1);

        if (offd_size == 0)
        {
            assert(diag_size == 1);

            int true_entity = ete_diag_indices[ete_diag_indptr[entity]];

            for (auto& true_entity_proc_i : true_entity_proc)
            {
                if (true_entity_proc_i.first == true_entity)
                {
                    entity_proc.Add(entity, true_entity_proc_i.second, 1);
                }
            }
        }
        else
        {
            assert(diag_size == 0 && offd_size == 1);

            int shared_entity = ete_offd_indices[ete_offd_indptr[entity]];

            for (int recv = 0; recv < comm_pkg.num_recvs_; ++recv)
            {
                int proc = comm_pkg.recv_procs_[recv];

                for (int k = recv_starts[recv]; k < recv_starts[recv + 1]; ++k)
                {
                    if (k == shared_entity)
                    {
                        entity_proc.Add(entity, proc, 1);

                        if (proc < entity_master_[entity])
                        {
                            entity_master_[entity] = proc;
                        }
                    }
                }
            }
        }
    }

    entity_proc_ = entity_proc.ToSparse();

    for (int i = 0; i < num_entities_; ++i)
    {
        if (entity_master_[i] == myid_)
        {
            num_master_comms_ += entity_proc_.RowSize(i) - 1; // -1 for myself
            entity_slave_id_[i] = -1;
        }
        else
        {
            entity_slave_id_[i] = num_slave_comms_++;
        }
    }

    //SetSizeSpecifier();
}


} // namespace smoothg


#endif // __SHAREDENTITYCOMM_HPP__
