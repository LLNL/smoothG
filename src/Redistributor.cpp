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
    const GraphSpace& space, const std::vector<int>& elem_redist_procs)
{
    redTrueEntity_trueEntity.resize(2);
    redEntity_trueEntity.resize(2);
    redTrueDof_trueDof.resize(2);
    redDof_trueDof.resize(2);

    const Graph& graph = space.GetGraph();

    auto elem_redProc = matred::EntityToProcessor(graph.GetComm(), elem_redist_procs);
    auto redProc_elem = matred::Transpose(elem_redProc);
    auto redElem_elem = matred::BuildRedistributedEntityToTrueEntity(redProc_elem);
    redEntity_trueEntity[0] = Move(redElem_elem); // elems are always "true"
    redTrueEntity_trueEntity[0] = Copy(*redEntity_trueEntity[0]);

    redEntity_trueEntity[1] = BuildRedEntToTrueEnt(graph.VertexToTrueEdge());
    auto redE_redTE = BuildRedEntToRedTrueEnt(*redEntity_trueEntity[1]);
    redTrueEntity_trueEntity[1] =
        BuildRedTrueEntToTrueEnt(*redE_redTE, *redEntity_trueEntity[1]);

    // build dofs redistribution relation
    mfem::SparseMatrix vert_vdof(space.VertexToVDof());
    unique_ptr<mfem::HypreParMatrix> vert_truevdof(
        ToParMatrix(graph.GetComm(), vert_vdof));
    redDof_trueDof[0] = BuildRedEntToTrueEnt(*vert_truevdof);
    redTrueDof_trueDof[0] = Copy(*redDof_trueDof[0]);

    auto& vert_edof = space.VertexToEDof();
    auto& edof_trueedof = space.EDofToTrueEDof();
    auto vert_trueedof = ParMult(vert_edof, edof_trueedof, graph.VertexStarts());
    redDof_trueDof[1] = BuildRedEntToTrueEnt(*vert_trueedof);

    auto redD_redTD = BuildRedEntToRedTrueEnt(*redDof_trueDof[1]);
    redTrueDof_trueDof[1] = BuildRedTrueEntToTrueEnt(*redD_redTD, *redDof_trueDof[1]);
}

unique_ptr<mfem::HypreParMatrix>
Redistributor::BuildRedEntToTrueEnt(const mfem::HypreParMatrix& elem_tE) const
{
    unique_ptr<mfem::HypreParMatrix> redElem_tE(
        mfem::ParMult(redTrueEntity_trueEntity[0].get(), &elem_tE));
    DropSmallEntries(*redElem_tE, 1e-6);

    matred::ParMatrix redElem_trueEntity(*redElem_tE, false);
    auto out = matred::BuildRedistributedEntityToTrueEntity(redElem_trueEntity);
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
Redistributor::BuildRepeatedEDofToTrueEDof(const GraphSpace& dof) const
{
    auto& dof_truedof = dof.EDofToTrueEDof();
    auto comm = dof_truedof.GetComm();

    // Build "repeated dof" to dof relation
    auto& vert_dof = dof.VertexToEDof();
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
Redistributor::BuildRepeatedEDofRedistribution(const GraphSpace& dof,
                                               const GraphSpace& redist_dof) const
{
    int codim = 0;
    int jform = 1;
    auto& redTE_TE = *redTrueEntity_trueEntity[codim];
    auto comm = redTE_TE.GetComm();

    auto vert_rdof = BuildVertRDof(dof.VertexToEDof());
    unique_ptr<mfem::HypreParMatrix> TE_RD(ToParMatrix(comm, vert_rdof));

    auto redvert_redrdof = BuildVertRDof(redist_dof.VertexToEDof());
    unique_ptr<mfem::HypreParMatrix> redTE_redRD(ToParMatrix(comm, redvert_redrdof));

    unique_ptr<mfem::HypreParMatrix> redRD_redTE(redTE_redRD->Transpose());
    auto redRD_TE_RD = Mult(*redRD_redTE, redTE_TE, *TE_RD);

    // Find intersection of redRDof_TrueElem_RDof and redRDof_TrueDof_RDof
    auto RD_TD = BuildRepeatedEDofToTrueEDof(dof);
    unique_ptr<mfem::HypreParMatrix> TD_RD(RD_TD->Transpose());
    auto redRD_redTD = BuildRepeatedEDofToTrueEDof(redist_dof);
    auto redRD_TD_RD = Mult(*redRD_redTD, *redTrueDof_trueDof[jform], *TD_RD);

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
    auto& trueB = graph.VertexToTrueEdge();
    unique_ptr<mfem::HypreParMatrix> TE_redE(redEntity_trueEntity[1]->Transpose());
    unique_ptr<mfem::HypreParMatrix> redB(
        Mult(*redTrueEntity_trueEntity[0], trueB, *TE_redE) );

    auto& e_tE = graph.EdgeToTrueEdge();
    unique_ptr<mfem::HypreParMatrix> tE_e(e_tE.Transpose());

    int myid;
    MPI_Comm_rank(graph.GetComm(), &myid);

    mfem::Array<int> e_starts(3), attr_starts(3), attr_map;
    e_starts[0] = e_tE.RowPart()[0];
    e_starts[1] = e_tE.RowPart()[1];
    e_starts[2] = e_tE.M();
    attr_starts[0] = myid > 0 ? graph.EdgeToBdrAtt().NumCols() : 0;
    attr_starts[1] = graph.EdgeToBdrAtt().NumCols();
    attr_starts[2] = graph.EdgeToBdrAtt().NumCols();

    if (myid > 0)
    {
        attr_map.SetSize(graph.EdgeToBdrAtt().NumCols());
        iota(attr_map.GetData(), attr_map.GetData() + attr_starts[2], 0);
    }

    mfem::SparseMatrix empty(graph.EdgeToBdrAtt().NumRows(), 0);
    empty.Finalize();

    mfem::SparseMatrix edge_bdratt(graph.EdgeToBdrAtt());
    mfem::HypreParMatrix e_bdrattr(graph.GetComm(), e_tE.M(), attr_starts[2], e_starts,
                                   attr_starts, myid ? &empty : &edge_bdratt,
                                   myid ? &edge_bdratt : &empty, attr_map);

    unique_ptr<mfem::HypreParMatrix> tE_bdrattr(mfem::ParMult(tE_e.get(), &e_bdrattr));
    unique_ptr<mfem::HypreParMatrix> redE_bdrattr(
        mfem::ParMult(redEntity_trueEntity[1].get(), tE_bdrattr.get()));

    HYPRE_Int* trash_map;
    mfem::SparseMatrix redE_bdrattr_local;
    if (myid == 0)
    {
        redE_bdrattr->GetDiag(redE_bdrattr_local);
    }
    else
    {
        redE_bdrattr->GetOffd(redE_bdrattr_local, trash_map);
    }

    auto redE_redTE = BuildRedEntToRedTrueEnt(*redEntity_trueEntity[1]);
    return Graph(GetDiag(*redB), *redE_redTE, mfem::Vector(), &redE_bdrattr_local);
}

GraphSpace Redistributor::RedistributeSpace(const GraphSpace& dof) const
{
    const Graph& graph = dof.GetGraph();
    Graph redist_graph = RedistributeGraph(graph);

    // redistribute vdofs
    std::unique_ptr<mfem::HypreParMatrix> truevdof_redvdof(redDof_trueDof[0]->Transpose());
    std::unique_ptr<mfem::HypreParMatrix> redvert_redvdof;
    {
        auto vert_redvdof = ParMult(dof.VertexToVDof(), *truevdof_redvdof, graph.VertexStarts());
        redvert_redvdof.reset(ParMult(redEntity_trueEntity[0].get(), vert_redvdof.get()));
    }
    // redistribute edofs
    std::unique_ptr<mfem::HypreParMatrix> rededge_rededof;
    {
        auto& edof_trueedof = dof.EDofToTrueEDof();
        unique_ptr<mfem::HypreParMatrix> trueedof_rededof(redDof_trueDof[1]->Transpose());

        auto edge_trueedof = ParMult(dof.EdgeToEDof(), edof_trueedof, graph.EdgeStarts());
        unique_ptr<mfem::HypreParMatrix> trueedge_edge(graph.EdgeToTrueEdge().Transpose());
        unique_ptr<mfem::HypreParMatrix> trueedge_trueedof(
            ParMult(trueedge_edge.get(), edge_trueedof.get()));
        *trueedge_trueedof = 1.0;//@todo probably need something like IgnoreNonLocalRange

        rededge_rededof.reset(
            Mult(*redEntity_trueEntity[1], *trueedge_trueedof, *trueedof_rededof));
    }

    auto rededge_rededof_diag = GetDiag(*rededge_rededof);

    GraphSpace out(std::move(redist_graph), rededge_rededof_diag, GetDiag(*redvert_redvdof));

    return out;
}

MixedMatrix Redistributor::RedistributeMatrix(const MixedMatrix& sequence) const
{
    const GraphSpace& space = sequence.GetGraphSpace();
    auto redist_space = RedistributeSpace(space);

    mfem::SparseMatrix redD;
    {
        auto& redD_tD = redDof_trueDof[1];
        {
            unique_ptr<mfem::HypreParMatrix> trueD(sequence.MakeParallelD(sequence.GetD()));
            unique_ptr<mfem::HypreParMatrix> tD_redD(redD_tD->Transpose());
            auto redTrueD = Mult(*redDof_trueDof[0], *trueD, *tD_redD);
            mfem::SparseMatrix redD_tmp = GetDiag(*redTrueD);
            redD.Swap(redD_tmp);
        }
    }

    std::vector<mfem::DenseMatrix> red_M_vert(redist_space.GetGraph().NumVertices());
    {
        {
            auto vert_rdof = BuildVertRDof(space.VertexToEDof());
            mfem::Array<int> rdofs;
            mfem::SparseMatrix M(vert_rdof.NumCols(), vert_rdof.NumCols());
            for (int vert = 0; vert < vert_rdof.NumRows(); ++vert)
            {
                GetTableRow(vert_rdof, vert, rdofs);
                M.SetSubMatrix(rdofs, rdofs, sequence.GetLocalMs()[vert]);
            }
            M.Finalize();

            unique_ptr<mfem::HypreParMatrix> trueM(ToParMatrix(space.GetGraph().GetComm(), M));

            unique_ptr<mfem::HypreParMatrix> red_trueM;
            {
                auto redRD_RD = BuildRepeatedEDofRedistribution(space, redist_space);
                unique_ptr<mfem::HypreParMatrix> RD_redRD(redRD_RD->Transpose());
                red_trueM.reset(Mult(*redRD_RD, *trueM, *RD_redRD));
            }

            mfem::SparseMatrix redM = GetDiag(*red_trueM);

            auto redvert_redrdof = BuildVertRDof(redist_space.VertexToEDof());
            for (int vert = 0; vert < redvert_redrdof.NumRows(); ++vert)
            {
                GetTableRow(redvert_redrdof, vert, rdofs);
                red_M_vert[vert].SetSize(rdofs.Size(), rdofs.Size());
                redM.GetSubMatrix(rdofs, rdofs, red_M_vert[vert]);
            }
        }
    }

    auto& redTD_tD = TrueDofRedistribution(0);
    auto red_const_rep = Mult(redTD_tD, sequence.GetConstantRep());

    auto& redTE_tE = TrueEntityRedistribution(0);
    auto red_vert_sizes = Mult(redTE_tE, sequence.GetVertexSizes());

    unique_ptr<mfem::HypreParMatrix> tD_redTD(redTD_tD.Transpose());
    auto redW = SparseIdentity(0);
    if (sequence.GetW().NumRows() > 0)
    {
        auto redW_tmp = ParMult(sequence.GetW(), *tD_redTD, space.VDofStarts());
        unique_ptr<mfem::HypreParMatrix> redTrueW(ParMult(&redTD_tD, redW_tmp.get()));
        auto redTrueW_diag = GetDiag(*redTrueW);
        redW.Swap(redTrueW_diag);
    }

    auto redP_pwc_tmp = ParMult(sequence.GetPWConstProj(), *tD_redTD, space.GetGraph().VertexStarts());
    unique_ptr<mfem::HypreParMatrix> redTrueP_pwc(ParMult(&redTE_tE, redP_pwc_tmp.get()));
    mfem::SparseMatrix redP_pwc(GetDiag(*redTrueP_pwc));

    MixedMatrix out(std::move(redist_space), std::move(red_M_vert), std::move(redD), std::move(redW),
                    std::move(red_const_rep), std::move(red_vert_sizes), std::move(redP_pwc));

return out;
}

} // namespace parelag
