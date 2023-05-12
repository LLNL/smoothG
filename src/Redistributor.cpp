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

#include <numeric>

#include "MetisGraphPartitioner.hpp"
#include "Redistributor.hpp"

namespace smoothg
{

using std::unique_ptr;

unique_ptr<mfem::HypreParMatrix> Move(matred::ParMatrix& A)
{
    auto out = make_unique<mfem::HypreParMatrix>(A);
    A.SetOwnerShip(false);
    out->CopyRowStarts();
    out->CopyColStarts();
    return out;
}

void Mult(const mfem::HypreParMatrix& A, const mfem::Array<int>& x, mfem::Array<int>& Ax)
{
    assert(A.NumRows() == Ax.Size() && A.NumCols() == x.Size());

    mfem::Vector x_vec(x.Size());
    for (int i = 0; i < x.Size(); ++i)
    {
        x_vec[i] = x[i];
    }

    mfem::Vector Ax_vec(Ax.Size());
    A.Mult(x_vec, Ax_vec);

    for (int i = 0; i < Ax.Size(); ++i)
    {
        Ax[i] = Ax_vec[i];
    }
}

std::vector<int> RedistributeElements(
    const mfem::HypreParMatrix& elem_face, int& num_redist_procs)
{
    MPI_Comm comm = elem_face.GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);

    mfem::Array<int> proc_starts, perm_rowstarts;
    int num_procs_loc = elem_face.NumRows() > 0 ? 1 : 0;
    GenerateOffsets(comm, num_procs_loc, proc_starts);

    int num_procs = proc_starts.Last();
    GenerateOffsets(comm, num_procs, perm_rowstarts);

    mfem::SparseMatrix proc_elem(num_procs_loc, elem_face.NumRows());
    if (num_procs_loc == 1)
    {
        for (int j = 0 ; j < proc_elem.NumCols(); ++j)
        {
            proc_elem.Set(0, j, 1.0);
        }
    }
    proc_elem.Finalize();

    unique_ptr<mfem::HypreParMatrix> proc_face(
        elem_face.LeftDiagMult(proc_elem, proc_starts));
    unique_ptr<mfem::HypreParMatrix> face_proc(proc_face->Transpose());
    unique_ptr<mfem::HypreParMatrix> proc_proc(
        mfem::ParMult(proc_face.get(), face_proc.get()));

    mfem::Array<HYPRE_Int> proc_colmap(num_procs - num_procs_loc);

    mfem::SparseMatrix perm_diag(num_procs, num_procs_loc);
    mfem::SparseMatrix perm_offd(num_procs, num_procs - num_procs_loc);
    int offd_proc_count = 0;
    for (int i = 0 ; i < num_procs; ++i)
    {
        if (i == myid)
        {
            perm_diag.Set(i, 0, 1.0);
        }
        else
        {
            perm_offd.Set(i, offd_proc_count, 1.0);
            proc_colmap[offd_proc_count++] = i;
        }
    }
    perm_diag.Finalize();
    perm_offd.Finalize();

    int num_perm_rows = perm_rowstarts.Last();
    mfem::HypreParMatrix permute(comm, num_perm_rows, num_procs, perm_rowstarts,
                                 proc_starts, &perm_diag, &perm_offd, proc_colmap);

    unique_ptr<mfem::HypreParMatrix> permuteT(permute.Transpose());
    unique_ptr<mfem::HypreParMatrix> permProc_permProc(
        Mult(permute, *proc_proc, *permuteT));

    mfem::SparseMatrix globProc_globProc;
    permProc_permProc->GetDiag(globProc_globProc);

    std::vector<int> out(elem_face.NumRows());
    if (elem_face.NumRows() > 0)
    {
        mfem::Array<int> partition;
        MetisGraphPartitioner partitioner;
        partitioner.setDefaultFlags(globProc_globProc.NumRows() / num_redist_procs);
        partitioner.doPartition(globProc_globProc, num_redist_procs, partition);

        assert(myid < partition.Size());
        std::fill_n(out.begin(), elem_face.NumRows(), partition[myid]);
    }
    return out;
}

Redistributor::Redistributor(
    const GraphSpace& space, const std::vector<int>& elem_redist_procs)
{
    Init(space, elem_redist_procs);
}

Redistributor::Redistributor(const GraphSpace& space, int& num_redist_procs)
{
    auto& graph = space.GetGraph();
    auto elem_redist_procs = RedistributeElements(graph.VertexToTrueEdge(), num_redist_procs);
    Init(space, elem_redist_procs);
}

void Redistributor::Init(
    const GraphSpace& space, const std::vector<int>& vert_redist_procs)
{
    const Graph& graph = space.GetGraph();

    auto vert_redProc = matred::EntityToProcessor(graph.GetComm(), vert_redist_procs);
    auto redProc_vert = matred::Transpose(vert_redProc);
    auto redVert_vert = matred::BuildRedistributedEntityToTrueEntity(redProc_vert);
    redVertex_vertex_ = Move(redVert_vert); // vert are always "true"

    redEdge_trueEdge_ = BuildRedEntToTrueEnt(graph.VertexToTrueEdge());
    auto redEdge_redTrueEdge = BuildRedEntToRedTrueEnt(*redEdge_trueEdge_);
    redTrueEdge_trueEdge_ =
        BuildRedTrueEntToTrueEnt(*redEdge_redTrueEdge, *redEdge_trueEdge_);

    // build dofs redistribution relation
    mfem::SparseMatrix vert_vdof(space.VertexToVDof());
    unique_ptr<mfem::HypreParMatrix> vert_truevdof(
        ToParMatrix(graph.GetComm(), vert_vdof));
    redVDOF_VDOF_ = BuildRedEntToTrueEnt(*vert_truevdof);

    auto vert_trueedof = ParMult(space.VertexToEDof(), space.EDofToTrueEDof(),
                                 graph.VertexStarts());
    redEDOF_trueEDOF_ = BuildRedEntToTrueEnt(*vert_trueedof);

    auto redD_redTD = BuildRedEntToRedTrueEnt(*redEDOF_trueEDOF_);
    redTrueEDOF_trueEDOF_ = BuildRedTrueEntToTrueEnt(*redD_redTD, *redEDOF_trueEDOF_);
}

unique_ptr<mfem::HypreParMatrix>
Redistributor::BuildRedEntToTrueEnt(const mfem::HypreParMatrix& vert_tE) const
{
    unique_ptr<mfem::HypreParMatrix> redVert_tE(
        mfem::ParMult(redVertex_vertex_.get(), &vert_tE));
    DropSmallEntries(*redVert_tE, 1e-6);

    matred::ParMatrix redVert_trueEntity(*redVert_tE, false);
    auto out = matred::BuildRedistributedEntityToTrueEntity(redVert_trueEntity);
    return Move(out);
}

unique_ptr<mfem::HypreParMatrix>
Redistributor::BuildRedEntToRedTrueEnt(const mfem::HypreParMatrix& redE_tE) const
{
    matred::ParMatrix redE_tE_ref(redE_tE, false);
    auto tE_redE = matred::Transpose(redE_tE_ref);
    auto redE_tE_redE = matred::Mult(redE_tE_ref, tE_redE);
    auto out = matred::BuildEntityToTrueEntity(redE_tE_redE);
    return Move(out);
}

unique_ptr<mfem::HypreParMatrix>
Redistributor::BuildRedTrueEntToTrueEnt(const mfem::HypreParMatrix& redE_redTE,
                                        const mfem::HypreParMatrix& redE_tE) const
{
    unique_ptr<mfem::HypreParMatrix> redTE_redE(redE_redTE.Transpose());
    unique_ptr<mfem::HypreParMatrix> out(mfem::ParMult(redTE_redE.get(), &redE_tE));
    DropSmallEntries(*out, 1e-6);
    out->CopyRowStarts();
    out->CopyColStarts();
    *out = 1.0;
    return out;
}

unique_ptr<mfem::HypreParMatrix>
Redistributor::BuildRepeatedEDofToTrueEDof(const GraphSpace& space) const
{
    auto& dof_truedof = space.EDofToTrueEDof();
    auto comm = dof_truedof.GetComm();

    // Build "repeated dof" to dof relation
    auto& vert_dof = space.VertexToEDof();
    int nnz = vert_dof.NumNonZeroElems();
    int* I = new int[nnz + 1];
    int* J = new int[nnz];
    double* data = new double[nnz];
    std::iota(I, I + nnz + 1, 0);
    std::copy_n(vert_dof.GetJ(), nnz, J);
    std::fill_n(data, nnz, 1.0);
    mfem::SparseMatrix rdof_dof(I, J, data, nnz, vert_dof.NumCols());

    // Build "repeated dof" to true dof relation
    mfem::Array<int> rdof_starts;
    int num_rdofs = rdof_dof.NumRows();
    GenerateOffsets(comm, num_rdofs, rdof_starts);
    auto out = ParMult(rdof_dof, dof_truedof, rdof_starts);
    out->CopyRowStarts();
    return out;
}

// Build vert to "repeated dof" relation
mfem::SparseMatrix BuildVertRDof(const mfem::SparseMatrix& vert_dof)
{
    int nnz = vert_dof.NumNonZeroElems();
    int* I = new int[vert_dof.NumRows()+1];
    int* J = new int[nnz];
    double* data = new double[nnz];
    std::copy_n(vert_dof.GetI(), vert_dof.NumRows()+1, I);
    std::iota(J, J + nnz, 0);
    std::fill_n(data, nnz, 1.0);
    return mfem::SparseMatrix(I, J, data, vert_dof.NumRows(), nnz);
}

unique_ptr<mfem::HypreParMatrix>
Redistributor::BuildRepeatedEDofRedistribution(const GraphSpace& space,
                                               const GraphSpace& redist_space) const
{
    auto comm = redVertex_vertex_->GetComm();

    auto vert_rdof = BuildVertRDof(space.VertexToEDof());
    unique_ptr<mfem::HypreParMatrix> TE_RD(ToParMatrix(comm, vert_rdof));

    auto redvert_redrdof = BuildVertRDof(redist_space.VertexToEDof());
    unique_ptr<mfem::HypreParMatrix> redTE_redRD(ToParMatrix(comm, redvert_redrdof));

    unique_ptr<mfem::HypreParMatrix> redRD_redTE(redTE_redRD->Transpose());
    auto redRD_TE_RD = Mult(*redRD_redTE, *redVertex_vertex_, *TE_RD);

    // Find intersection of redRDof_TrueElem_RDof and redRDof_TrueDof_RDof
    auto RD_TD = BuildRepeatedEDofToTrueEDof(space);
    unique_ptr<mfem::HypreParMatrix> TD_RD(RD_TD->Transpose());
    auto redRD_redTD = BuildRepeatedEDofToTrueEDof(redist_space);
    auto redRD_TD_RD = Mult(*redRD_redTD, *redTrueEDOF_trueEDOF_, *TD_RD);

    mfem::SparseMatrix redRD_TE_RD_diag, redRD_TE_RD_offd;
    mfem::SparseMatrix redRD_TD_RD_diag, redRD_TD_RD_offd;
    HYPRE_BigInt* redRD_TE_RD_colmap, *redRD_TD_RD_colmap;

    redRD_TE_RD->GetDiag(redRD_TE_RD_diag);
    redRD_TE_RD->GetOffd(redRD_TE_RD_offd, redRD_TE_RD_colmap);
    redRD_TD_RD->GetDiag(redRD_TD_RD_diag);
    redRD_TD_RD->GetOffd(redRD_TD_RD_offd, redRD_TD_RD_colmap);

    HYPRE_BigInt* out_colmap = new HYPRE_BigInt[redRD_TE_RD_offd.NumCols()];
    std::copy_n(redRD_TE_RD_colmap, redRD_TE_RD_offd.NumCols(), out_colmap);

    auto out_diag = new mfem::SparseMatrix(redRD_TE_RD_diag.NumRows(),
                                           redRD_TE_RD_diag.NumCols());
    auto out_offd = new mfem::SparseMatrix(redRD_TE_RD_offd.NumRows(),
                                           redRD_TE_RD_offd.NumCols());
    for (int i = 0; i < redRD_TE_RD->NumRows(); ++i)
    {
        if (redRD_TE_RD_diag.RowSize(i) > 0)
        {
            mfem::Array<int> RDs;
            for (int j = 0; j < redRD_TE_RD_diag.RowSize(i); ++j)
            {
                int RD = redRD_TE_RD_diag.GetRowColumns(i)[j];
                GetTableRow(redRD_TD_RD_diag, i, RDs);
                if (RDs.Find(RD) != -1)
                {
                    out_diag->Add(i, RD, 1.0);
                    break;
                }
            }
            assert(out_diag->RowSize(i) == 1);
        }
        else
        {
            assert(redRD_TE_RD_offd.RowSize(i) > 0);
            for (int j = 0; j < redRD_TE_RD_offd.RowSize(i); ++j)
            {
                int RD = redRD_TE_RD_offd.GetRowColumns(i)[j];
                HYPRE_Int RD_global = redRD_TE_RD_colmap[RD];
                mfem::Array<int> RDs(redRD_TD_RD_offd.RowSize(i));
                for (int k = 0; k < RDs.Size(); ++k)
                {
                    RDs[k] = redRD_TD_RD_colmap[redRD_TD_RD_offd.GetRowColumns(i)[k]];
                }
                if (RDs.Find(RD_global) != -1)
                {
                    out_offd->Add(i, RD, 1.0);
                    break;
                }
            }
            assert(out_offd->RowSize(i) == 1);
        }
    }
    out_diag->Finalize();
    out_offd->Finalize();

    auto redRD_RD = make_unique<mfem::HypreParMatrix>(
                        comm, redTE_redRD->N(), TE_RD->N(), redTE_redRD->ColPart(),
                        TE_RD->ColPart(), out_diag, out_offd, out_colmap);
    redRD_RD->CopyRowStarts();
    redRD_RD->CopyColStarts();
    redRD_RD->SetOwnerFlags(3, 3, 1);
    return redRD_RD;
}

Graph Redistributor::RedistributeGraph(const Graph& graph) const
{
    // Redistribute other remaining data
    unique_ptr<mfem::HypreParMatrix> TE_redE(redEdge_trueEdge_->Transpose());
    unique_ptr<mfem::HypreParMatrix> redV_redE(
        Mult(*redVertex_vertex_, graph.VertexToTrueEdge(), *TE_redE) );

    int myid;
    MPI_Comm_rank(graph.GetComm(), &myid);

    mfem::Array<int> attr_starts(2), attr_map;
    attr_starts[0] = myid > 0 ? graph.EdgeToBdrAtt().NumCols() : 0;
    attr_starts[1] = graph.EdgeToBdrAtt().NumCols();

    auto& e_starts = const_cast<mfem::Array<int>&>(graph.EdgeStarts());

    if (myid > 0)
    {
        attr_map.SetSize(graph.EdgeToBdrAtt().NumCols());
        iota(attr_map.GetData(), attr_map.GetData() + attr_starts[1], 0);
    }

    mfem::SparseMatrix empty(graph.NumEdges(), 0);
    empty.Finalize();

    auto& e_tE = graph.EdgeToTrueEdge();
    unique_ptr<mfem::HypreParMatrix> tE_e(e_tE.Transpose());

    mfem::SparseMatrix edge_bdratt(graph.EdgeToBdrAtt());
    mfem::HypreParMatrix e_bdrattr(graph.GetComm(), e_tE.M(), attr_starts[1], e_starts,
                                   attr_starts, myid ? &empty : &edge_bdratt,
                                   myid ? &edge_bdratt : &empty, attr_map);

    unique_ptr<mfem::HypreParMatrix> tE_bdrattr(mfem::ParMult(tE_e.get(), &e_bdrattr));
    unique_ptr<mfem::HypreParMatrix> redE_bdrattr_tmp(
        mfem::ParMult(redEdge_trueEdge_.get(), tE_bdrattr.get()));

    HYPRE_Int* trash_map;
    mfem::SparseMatrix redE_bdrattr;
    if (myid == 0)
    {
        redE_bdrattr_tmp->GetDiag(redE_bdrattr);
    }
    else
    {
        redE_bdrattr_tmp->GetOffd(redE_bdrattr, trash_map);
    }

    auto redE_redTE = BuildRedEntToRedTrueEnt(*redEdge_trueEdge_);
    return Graph(GetDiag(*redV_redE), *redE_redTE, mfem::Vector(), &redE_bdrattr);
}

GraphSpace Redistributor::RedistributeSpace(const GraphSpace& space) const
{
    const Graph& graph = space.GetGraph();
    Graph redist_graph = RedistributeGraph(graph);

    // redistribute vdofs
    unique_ptr<mfem::HypreParMatrix> vdof_redvdof(redVDOF_VDOF_->Transpose());
    auto vert_redvdof = ParMult(space.VertexToVDof(), *vdof_redvdof, graph.VertexStarts());
    unique_ptr<mfem::HypreParMatrix> redvert_redvdof(
        ParMult(redVertex_vertex_.get(), vert_redvdof.get()));

    // redistribute edofs
    auto& edof_trueedof = space.EDofToTrueEDof();
    unique_ptr<mfem::HypreParMatrix> trueedof_rededof(redEDOF_trueEDOF_->Transpose());

    auto edge_trueedof = ParMult(space.EdgeToEDof(), edof_trueedof, graph.EdgeStarts());
    unique_ptr<mfem::HypreParMatrix> trueedge_edge(graph.EdgeToTrueEdge().Transpose());
    unique_ptr<mfem::HypreParMatrix> trueedge_trueedof(
        ParMult(trueedge_edge.get(), edge_trueedof.get()));
    *trueedge_trueedof = 1.0;

    unique_ptr<mfem::HypreParMatrix> rededge_rededof(
        Mult(*redEdge_trueEdge_, *trueedge_trueedof, *trueedof_rededof));

    auto rededge_rededof_diag = GetDiag(*rededge_rededof);
    return GraphSpace(std::move(redist_graph), rededge_rededof_diag, GetDiag(*redvert_redvdof));
}

MixedMatrix Redistributor::RedistributeMatrix(const MixedMatrix& system) const
{
    const GraphSpace& space = system.GetGraphSpace();
    const Graph& graph = space.GetGraph();
    GraphSpace redist_space = RedistributeSpace(space);

    unique_ptr<mfem::HypreParMatrix> trueD(system.MakeParallelD(system.GetD()));
    unique_ptr<mfem::HypreParMatrix> tD_redD(redEDOF_trueEDOF_->Transpose());
    auto redTrueD = Mult(*redVDOF_VDOF_, *trueD, *tD_redD);
    mfem::SparseMatrix redD = GetDiag(*redTrueD);

    auto vert_rdof = BuildVertRDof(space.VertexToEDof());
    mfem::Array<int> rdofs;
    mfem::SparseMatrix M(vert_rdof.NumCols(), vert_rdof.NumCols());
    for (int vert = 0; vert < vert_rdof.NumRows(); ++vert)
    {
        GetTableRow(vert_rdof, vert, rdofs);
        M.SetSubMatrix(rdofs, rdofs, system.GetLocalMs()[vert]);
    }
    M.Finalize();

    unique_ptr<mfem::HypreParMatrix> trueM(ToParMatrix(graph.GetComm(), M));
    auto redRD_RD = BuildRepeatedEDofRedistribution(space, redist_space);
    unique_ptr<mfem::HypreParMatrix> RD_redRD(redRD_RD->Transpose());
    unique_ptr<mfem::HypreParMatrix> red_trueM(Mult(*redRD_RD, *trueM, *RD_redRD));
    mfem::SparseMatrix redM = GetDiag(*red_trueM);

    auto redvert_redrdof = BuildVertRDof(redist_space.VertexToEDof());
    std::vector<mfem::DenseMatrix> red_M_vert(redvert_redrdof.NumRows());
    for (int vert = 0; vert < redvert_redrdof.NumRows(); ++vert)
    {
        GetTableRow(redvert_redrdof, vert, rdofs);
        red_M_vert[vert].SetSize(rdofs.Size(), rdofs.Size());
        redM.GetSubMatrix(rdofs, rdofs, red_M_vert[vert]);
    }

    auto red_const_rep = Mult(*redVDOF_VDOF_, system.GetConstantRep());
    auto red_vert_sizes = Mult(*redVertex_vertex_, system.GetVertexSizes());

    unique_ptr<mfem::HypreParMatrix> VDOF_redVDOF(redVDOF_VDOF_->Transpose());
    auto redW = SparseIdentity(0);

    int W_loc_size = system.GetW().NumRows();
    int W_glo_size;
    MPI_Allreduce(&W_loc_size, &W_glo_size, 1, MPI_INT, MPI_SUM, graph.GetComm());
    if (W_glo_size > 0)
    {
        auto redW_tmp = ParMult(system.GetW(), *VDOF_redVDOF, space.VDofStarts());
        unique_ptr<mfem::HypreParMatrix> redTrueW(
            ParMult(redVDOF_VDOF_.get(), redW_tmp.get()) );
        auto redTrueW_diag = GetDiag(*redTrueW);
        redW.Swap(redTrueW_diag);
    }

    auto tmp = ParMult(system.GetPWConstProj(), *VDOF_redVDOF, graph.VertexStarts());
    unique_ptr<mfem::HypreParMatrix> redTrueP_pwc(
        ParMult(redVertex_vertex_.get(), tmp.get()));
    mfem::SparseMatrix redP_pwc(GetDiag(*redTrueP_pwc));

    return MixedMatrix(std::move(redist_space), std::move(red_M_vert),
                       std::move(redD), std::move(redW), std::move(red_const_rep),
                       std::move(red_vert_sizes), std::move(redP_pwc));
}


const mfem::HypreParMatrix&
Redistributor::TrueEntityRedistribution(EntityType entity) const
{
    if (entity == VERTEX)
    {
        return *redVertex_vertex_;
    }
    else
    {
        return *redTrueEdge_trueEdge_;
    }
}

const mfem::HypreParMatrix&
Redistributor::TrueDofRedistribution(DofType dof) const
{
    if (dof == VDOF)
    {
        return *redVDOF_VDOF_;
    }
    else
    {
        return *redTrueEDOF_trueEDOF_;
    }
}

} // namespace parelag
