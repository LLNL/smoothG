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

        void ReduceSend(int entity, T mat);

        bool IsOwnedByMe(int entity) const;

        int GetTrueEntity(int entity) const;

        std::vector<std::vector<T>> Collect();

        void Broadcast(std::vector<T>& mats);

    private:
        void MakeEntityProc();

        void SetSizeSpecifier();
        std::vector<int> PackSendSize(const T& mat) const;

        void SendData(const T& mat, int recipient, int tag, MPI_Request& request) const;
        T ReceiveData(const std::vector<int>& sizes, int sender, int tag, MPI_Request& request) const;

        void ReducePrepare();

        void BroadcastSizes(std::vector<T>& mats);
        void BroadcastData(std::vector<T>& mats);

        const ParMatrix& entity_true_entity_;
        SparseMatrix entity_diag_T_;
        SparseMatrix entity_offd_T_;

        parlinalgcpp::ParCommPkg comm_pkg_;

        MPI_Comm comm_;
        MPI_Comm myid_;

        int num_entities_;
        int size_specifier_;
        int send_counter_;

        std::vector<int> entity_master_;
        std::vector<int> entity_slave_id_;

        int num_master_comms_;
        int num_slave_comms_;

        linalgcpp::SparseMatrix<int> entity_proc_;

        std::vector<T> send_buffer_;
        std::vector<std::vector<T>> recv_buffer_;

        std::vector<std::vector<int>> send_headers_;
        std::vector<std::vector<int>> recv_headers_;

        std::vector<MPI_Request> header_requests_;
        std::vector<MPI_Request> data_requests_;

        bool preparing_to_reduce_;

        enum { ENTITY_HEADER_TAG, ENTITY_MESSAGE_TAG, };
};

template <typename T>
SharedEntityComm<T>::SharedEntityComm(const ParMatrix& entity_true_entity)
    : entity_true_entity_(entity_true_entity),
      entity_diag_T_(entity_true_entity_.GetDiag().Transpose()),
      entity_offd_T_(entity_true_entity_.GetOffd().Transpose()),
      comm_pkg_(entity_true_entity_.MakeCommPkg()),
      comm_(entity_true_entity_.GetComm()),
      myid_(entity_true_entity_.GetMyId()),
      num_entities_(entity_true_entity_.Rows()),
      size_specifier_(0),
      send_counter_(0),
      entity_master_(num_entities_),
      entity_slave_id_(num_entities_),
      num_master_comms_(0),
      num_slave_comms_(0),
      recv_buffer_(num_entities_),
      preparing_to_reduce_(false)
{
    MakeEntityProc();

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

    send_buffer_.resize(num_slave_comms_);

    SetSizeSpecifier();
}

template <typename T>
void SharedEntityComm<T>::MakeEntityProc()
{
    auto& send_starts = comm_pkg_.send_map_starts_;
    auto& recv_starts = comm_pkg_.recv_vec_starts_;

    std::vector<std::pair<int, int> > true_entity_proc;

    int num_sends = comm_pkg_.num_sends_;

    for (int send = 0; send < num_sends; ++send)
    {
        int proc = comm_pkg_.send_procs_[send];

        for (int j = send_starts[send]; j < send_starts[send + 1]; ++j)
        {
            int true_entity = comm_pkg_.send_map_elmts_[j];

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

            for (int recv = 0; recv < comm_pkg_.num_recvs_; ++recv)
            {
                int proc = comm_pkg_.recv_procs_[recv];

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
}

template <typename T>
void SharedEntityComm<T>::ReducePrepare()
{
    const int header_length = size_specifier_ + 1;

    send_headers_.resize(num_slave_comms_);
    recv_headers_.resize(num_master_comms_, std::vector<int>(header_length));

    header_requests_.resize(num_master_comms_ + num_slave_comms_);
    data_requests_.resize(num_master_comms_ + num_slave_comms_);

    int header_recv_counter = 0;

    for (int i = 0; i < num_entities_; ++i)
    {
        if (entity_master_[i] == myid_)
        {
            int neighbor_row_size = entity_proc_.RowSize(i);
            recv_buffer_[i].resize(neighbor_row_size);

            std::vector<int> neighbor_row = entity_proc_.GetIndices(i);

            for (auto neighbor : neighbor_row)
            {
                if (neighbor != myid_)
                {
                    MPI_Irecv(
                        recv_headers_[header_recv_counter].data(),
                        header_length, MPI_INT, neighbor,
                        ENTITY_HEADER_TAG, comm_,
                        &header_requests_[header_recv_counter]);
                    header_recv_counter++;
                }
            }
        }
    }

    assert(header_recv_counter == num_master_comms_);

    preparing_to_reduce_ = true;
}

template <class T>
int SharedEntityComm<T>::GetTrueEntity(int entity) const
{
    if (entity_master_[entity] == myid_)
    {
        const auto& ete_diag = entity_true_entity_.GetDiag();
        const auto& ete_diag_indptr = ete_diag.GetIndptr();
        const auto& ete_diag_indices = ete_diag.GetIndices();
        const auto& ete_col_starts = entity_true_entity_.GetColStarts();

        return ete_col_starts[0] + ete_diag_indices[ete_diag_indptr[entity]];
    }
    else
    {
        const auto& ete_offd = entity_true_entity_.GetOffd();
        const auto& ete_offd_indptr = ete_offd.GetIndptr();
        const auto& ete_offd_indices = ete_offd.GetIndices();
        const auto& ete_colmap = entity_true_entity_.GetColMap();

        return ete_colmap[ete_offd_indices[ete_offd_indptr[entity]]];
    }
}

template <class T>
bool SharedEntityComm<T>::IsOwnedByMe(int entity) const
{
    assert(entity >= 0);
    assert(entity < static_cast<int>(entity_master_.size()));

    return (entity_master_[entity] == myid_);
}

template <typename T>
void SharedEntityComm<T>::ReduceSend(int entity, T mat)
{
    if (!preparing_to_reduce_)
    {
        ReducePrepare();
    }

    int owner = entity_master_[entity];

    if (owner == myid_)
    {
        swap(recv_buffer_[entity][0], mat);
    }
    else
    {
        int true_entity = GetTrueEntity(entity);
        int send_id = entity_slave_id_[entity];

        assert(send_id >= 0);

        send_headers_[send_id] = PackSendSize(mat);
        send_headers_[send_id].push_back(true_entity);

        const int header_length = send_headers_[send_id].size();

        MPI_Isend(send_headers_[send_id].data(), header_length,
                  MPI_INT, owner, ENTITY_HEADER_TAG, comm_,
                  &header_requests_[num_master_comms_ + send_id]);

        swap(send_buffer_[send_id], mat);
        SendData(send_buffer_[send_id], owner, ENTITY_MESSAGE_TAG,
                 data_requests_[send_id]);
        send_counter_++;
    }
}

template <typename T>
std::vector<std::vector<T>> SharedEntityComm<T>::Collect()
{
    assert(send_counter_ == num_slave_comms_);

    std::vector<MPI_Status> header_statuses(num_slave_comms_ + num_master_comms_);
    MPI_Waitall(num_slave_comms_ + num_master_comms_, header_requests_.data(), header_statuses.data());

    header_requests_.clear();

    const auto& diag_T_indices = entity_diag_T_.GetIndices();
    const auto& ete_col_starts = entity_true_entity_.GetColStarts();

    std::vector<int> received_entities(num_entities_, 0);
    int data_receive_counter = 0;

    for (int i = 0; i < num_entities_; ++i)
    {
        int owner = entity_master_[i];

        if (owner != myid_)
        {
            continue;
        }

        std::vector<int> neighbors = entity_proc_.GetIndices(i);

        for (auto neighbor : neighbors)
        {
            if (neighbor == myid_)
            {
                continue;
            }

            auto& header = recv_headers_[data_receive_counter];

            int true_entity = header.back();
            int entity = diag_T_indices[true_entity - ete_col_starts[0]];

            int row = entity;
            int column = 1 + received_entities[entity];

            recv_buffer_[row][column] = ReceiveData(header, neighbor, ENTITY_MESSAGE_TAG,
                    data_requests_[num_slave_comms_ + data_receive_counter]);

            received_entities[entity]++;
            data_receive_counter++;
        }
    }

    std::vector<MPI_Status> data_statuses(num_slave_comms_ + num_master_comms_);
    MPI_Waitall(num_slave_comms_ + num_master_comms_,
            data_requests_.data(), data_statuses.data());

    preparing_to_reduce_ = false;

    return std::move(recv_buffer_);
}

template <class T>
void SharedEntityComm<T>::Broadcast(std::vector<T>& mats)
{
    assert(!preparing_to_reduce_);

    BroadcastSizes(mats);
    BroadcastData(mats);
}

template <class T>
void SharedEntityComm<T>::BroadcastSizes(std::vector<T>& mats)
{
    send_headers_.resize(num_master_comms_);
    recv_headers_.resize(num_slave_comms_, std::vector<int>(size_specifier_));

    header_requests_.resize(num_master_comms_ + num_slave_comms_);

    const auto& offd_T_indices = entity_offd_T_.GetIndices();
    const auto& entity_offd = entity_true_entity_.GetOffd();
    int num_recv = entity_offd.Cols();
    assert(num_recv == num_slave_comms_);

    for (int i = 0; i < num_recv; ++i)
    {
        int entity = offd_T_indices[i];
        int owner = entity_master_[entity];

        MPI_Irecv(recv_headers_[i].data(), size_specifier_, MPI_INT,
                  owner, ENTITY_HEADER_TAG, comm_,
                  &header_requests_[i]);
    }

    const auto& diag_T_indices = entity_diag_T_.GetIndices();
    int num_send = entity_true_entity_.Cols();
    int send_counter = 0;

    for (int i = 0; i < num_send; ++i)
    {
        int entity = diag_T_indices[i];
        std::vector<int> neighbor_row = entity_proc_.GetIndices(entity);

        for (auto neighbor : neighbor_row)
        {
            if (neighbor != myid_)
            {
                send_headers_[send_counter] = PackSendSize(mats[entity]);

                MPI_Isend(send_headers_[send_counter].data(), size_specifier_,
                        MPI_INT, neighbor, ENTITY_HEADER_TAG, comm_,
                        &header_requests_[num_slave_comms_ + send_counter]);
                send_counter++;
            }
        }
    }

    assert(send_counter == num_master_comms_);

    std::vector<MPI_Status> header_statuses(num_slave_comms_ + num_master_comms_);
    MPI_Waitall(num_slave_comms_ + num_master_comms_,
                header_requests_.data(), header_statuses.data());
}

template <class T>
void SharedEntityComm<T>::BroadcastData(std::vector<T>& mats)
{
    data_requests_.resize(num_master_comms_ + num_slave_comms_);

    const auto& offd_T_indices = entity_offd_T_.GetIndices();
    const auto& entity_offd = entity_true_entity_.GetOffd();
    int num_recv = entity_offd.Cols();
    assert(num_recv == num_slave_comms_);

    for (int i = 0; i < num_recv; ++i)
    {
        int entity = offd_T_indices[i];
        int owner = entity_master_[entity];

        mats[entity] = ReceiveData(recv_headers_[i], owner,
                                   ENTITY_MESSAGE_TAG,
                                   data_requests_[i]);
    }

    const auto& diag_T_indices = entity_diag_T_.GetIndices();
    int num_send = entity_true_entity_.Cols();
    int send_counter = 0;

    for (int i = 0; i < num_send; ++i)
    {
        int entity = diag_T_indices[i];
        std::vector<int> neighbor_row = entity_proc_.GetIndices(entity);

        for (auto neighbor : neighbor_row)
        {
            if (neighbor != myid_)
            {
                SendData(mats[entity], neighbor, ENTITY_MESSAGE_TAG,
                        data_requests_[num_slave_comms_ + send_counter]);
                send_counter++;
            }
        }
    }

    assert(send_counter == num_master_comms_);

    std::vector<MPI_Status> data_statuses(num_slave_comms_ + num_master_comms_);
    MPI_Waitall(num_slave_comms_ + num_master_comms_,
                data_requests_.data(), data_statuses.data());
}


} // namespace smoothg


#endif // __SHAREDENTITYCOMM_HPP__
