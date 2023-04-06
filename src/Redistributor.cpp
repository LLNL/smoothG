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

#include <numeric>

#include "MetisGraphPartitioner.hpp"
#include "Redistributor.hpp"
// #include "utilities/MemoryUtils.hpp"
// #include "linalg/utilities/ParELAG_MatrixUtils.hpp"
// #include "utilities/mpiUtils.hpp"
// #include "hypreExtension/hypreExtension.hpp"

namespace smoothg
{
// using namespace mfem;
// using std::shared_ptr;
// using std::make_shared;
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
    const Graph& graph, const std::vector<int>& elem_redist_procs)
    : redTrueEntity_trueEntity(2),
      redEntity_trueEntity(2),
      redTrueDof_trueDof(2),
      redDof_trueDof(2)
{
    Init(graph, elem_redist_procs);
}

Redistributor::Redistributor(const Graph& graph, int& num_redist_procs)
    : redTrueEntity_trueEntity(2),
      redEntity_trueEntity(2),
      redTrueDof_trueDof(2),
      redDof_trueDof(2)
{
    auto elem_redist_procs = RedistributeElements(graph.VertexToTrueEdge(), num_redist_procs);
    Init(graph, elem_redist_procs);
}

void Redistributor::Init(
    const Graph& graph, const std::vector<int>& elem_redist_procs)
{
    auto elem_redProc = matred::EntityToProcessor(graph.GetComm(), elem_redist_procs);
    auto redProc_elem = matred::Transpose(elem_redProc);
    auto redElem_elem = matred::BuildRedistributedEntityToTrueEntity(redProc_elem);
    redTrueEntity_trueEntity[0] = Move(redElem_elem); // elems are always "true"

    auto& elem_trueEntity = graph.VertexToTrueEdge();
    redEntity_trueEntity[1] = BuildRedEntToTrueEnt(elem_trueEntity);

    redist_graph_ = Redistribute(graph);
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
Redistributor::BuildRepeatedEDofToTrueEDof(const GraphSpace& dof)
{
    auto& dof_truedof = dof.EDofToTrueEDof();
    auto comm = dof_truedof.GetComm();

    // Build "repeated dof" to dof relation
    auto& vert_dof = dof.VertexToEDof();
    int nnz = vert_dof.NumNonZeroElems();
    int* I = new int[nnz+1];
    int* J = new int[nnz];
    double* data = new double[nnz];
    std::iota(I, I+nnz+1, 0);
    std::copy_n(vert_dof.GetJ(), nnz, J);
    std::fill_n(data, nnz, 1);
    mfem::SparseMatrix rdof_dof(I, J, data, nnz, vert_dof.NumCols());
    
    // Build "repeated dof" to true dof relation
    mfem::Array<int> rdof_starts;
    int num_rdofs = rdof_dof.NumRows();
    GenerateOffsets(comm, num_rdofs, rdof_starts);
    unique_ptr<mfem::HypreParMatrix> rdof_truedof(
             dof_truedof.LeftDiagMult(rdof_dof, rdof_starts));
    rdof_truedof->CopyRowStarts();
    return rdof_truedof;
}

// Build vert to "repeated dof" relation
mfem::SparseMatrix BuildVertRDof(const mfem::SparseMatrix& vert_dof)
{
    int nnz = vert_dof.NumNonZeroElems();
    int* I = new int[nnz+1];
    int* J = new int[nnz];
    double* data = new double[nnz];
    std::copy_n(vert_dof.GetI(), vert_dof.NumRows(), I);
    std::iota(J, I+nnz, 0);
    std::fill_n(data, nnz, 1);
    return mfem::SparseMatrix(I, J, data, vert_dof.NumRows(), nnz);
}

unique_ptr<mfem::HypreParMatrix>
Redistributor::BuildRepeatedEDofRedistribution(const GraphSpace& dof,
                                               const GraphSpace& redist_dof)
{
    int codim = 0;
    int jform = 1;
//    auto type = static_cast<GraphTopology::Entity>(codim);
    auto& redTE_TE = *redTrueEntity_trueEntity[codim];
    auto comm = redTE_TE.GetComm();

//    auto E_RD = ToParMatrix(comm, dof.GetEntityRDofTable(type));
//    auto E_TE = topo.EntityTrueEntity(codim).get_entity_trueEntity();

    auto vert_rdof = BuildVertRDof(dof.VertexToEDof());
    unique_ptr<mfem::HypreParMatrix> E_RD(ToParMatrix(comm, vert_rdof));

    auto& E_TE = dof.GetGraph().EdgeToTrueEdge();
    unique_ptr<mfem::HypreParMatrix> TE_E(E_TE.Transpose());
    unique_ptr<mfem::HypreParMatrix> TE_RD(mfem::ParMult(TE_E.get(), E_RD.get()));


    // auto redE_redTE = redist_topo->EntityTrueEntity(codim).get_entity_trueEntity();
    auto redvert_redrdof = BuildVertRDof(redist_dof.VertexToEDof());
    auto redE_redRD = ToParMatrix(comm, redvert_redrdof);
    auto& redE_redTE = redist_dof.GetGraph().EdgeToTrueEdge();

    unique_ptr<mfem::HypreParMatrix> redRD_redE(redE_redRD->Transpose());
    auto redRD_redTE = mfem::ParMult(redRD_redE.get(), &redE_redTE);
    auto redRD_TE_RD = Mult(*redRD_redTE, redTE_TE, *TE_RD);

    // Find intersection of redRDof_TrueElem_RDof and redRDof_TrueDof_RDof
    auto RD_TD = BuildRepeatedEDofToTrueEDof(dof);
    unique_ptr<mfem::HypreParMatrix> TD_RD(RD_TD->Transpose());
    auto redRD_redTD = BuildRepeatedEDofToTrueEDof(redist_dof);
    auto redRD_TD_RD = RAP(*redRD_redTD, *redTrueDof_trueDof[jform], *TD_RD);

    mfem::SparseMatrix redRD_TE_RD_diag, redRD_TE_RD_offd;
    mfem::SparseMatrix redRD_TD_RD_diag, redRD_TD_RD_offd;
    HYPRE_BigInt *redRD_TE_RD_colmap, *redRD_TD_RD_colmap;

    redRD_TE_RD->GetDiag(redRD_TE_RD_diag);
    redRD_TE_RD->GetOffd(redRD_TE_RD_offd, redRD_TE_RD_colmap);
    redRD_TD_RD->GetDiag(redRD_TD_RD_diag);
    redRD_TD_RD->GetOffd(redRD_TD_RD_offd, redRD_TD_RD_colmap);

    HYPRE_BigInt * out_colmap = new HYPRE_BigInt[redRD_TE_RD_offd.NumCols()];
    std::copy_n(redRD_TE_RD_colmap, redRD_TE_RD_offd.NumCols(), out_colmap);

    auto out_diag = new mfem::SparseMatrix(redRD_TE_RD_diag.NumRows(),
                                        redRD_TE_RD_diag.NumCols());
    auto out_offd = new mfem::SparseMatrix(redRD_TE_RD_offd.NumRows(),
                                        redRD_TE_RD_offd.NumCols());
    for (int i = 0; i < redRD_TE_RD->NumRows(); ++i)
    {
        if (redRD_TE_RD_diag.RowSize(i) > 0)
        {
            for (int j = 0; j < redRD_TE_RD_diag.RowSize(i); ++j)
            {
                int RD = redRD_TE_RD_diag.GetRowColumns(i)[j];
                mfem::Array<int> RDs(redRD_TD_RD_diag.GetRowColumns(i),
                                    redRD_TD_RD_diag.RowSize(i));
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
                comm, redE_redRD->N(), E_RD->N(), redE_redRD->ColPart(),
                E_RD->ColPart(), out_diag, out_offd, out_colmap);
    redRD_RD->CopyRowStarts();
    redRD_RD->CopyColStarts();
    redRD_RD->SetOwnerFlags(3, 3, 1);
    return redRD_RD;
}

Graph Redistributor::Redistribute(const Graph& graph)
{
    // Redistribute the current graph to another set of processors
    // auto out = make_unique<Graph>(graph.GetComm());

    // const int num_redElem = TrueEntityRedistribution(0).NumRows();
    // out->EntityTrueEntity(0).SetUp(num_redElem);

    //    auto redElem_redTrueElem = out->EntityTrueEntity(0).get_entity_trueEntity();
    //    redEntity_trueEntity[0] = Mult(*redElem_redTrueElem, *redTrueEntity_trueEntity[0]);
    redEntity_trueEntity[0] = Copy(*redTrueEntity_trueEntity[0]);
    DropSmallEntries(*redEntity_trueEntity[0], 1e-6);

    auto redE_redTE = BuildRedEntToRedTrueEnt(*redEntity_trueEntity[1]);

    redTrueEntity_trueEntity[1] =
        BuildRedTrueEntToTrueEnt(*redE_redTE, *redEntity_trueEntity[1]);
    //    out->entityTrueEntity[1]->SetUp(std::move(redE_redTE));

    // Redistribute other remaining data
    //    auto& trueB = const_cast<GraphTopology&>(topo).TrueB(0);
    auto trueB = graph.VertexToTrueEdge();
    unique_ptr<mfem::HypreParMatrix> TE_redE(redEntity_trueEntity[1]->Transpose());
    auto redB = Mult(*redTrueEntity_trueEntity[0], trueB, *TE_redE);

    // mfem::SparseMatrix redB_diag = GetDiag(*redB);
    //    mfem::SparseMatrix redB_diag_copy(redB_diag);

    //    out->B_[0] = make_unique<TopologyTable>(redB_diag_copy);

    //    auto e_tE = topo.EntityTrueEntity(1).get_entity_trueEntity();
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

    // mfem::SparseMatrix redE_bdrattr_local(redE_bdrattr_local_ref);
    //    out->facet_bdrAttribute = make_unique<TopologyTable>(redE_bdrattr_local);

    //    out->element_attribute.SetSize(num_redElem);
    //    Mult(*redTrueEntity_trueEntity[0], topo.element_attribute, out->element_attribute);

    //    out->Weights_[0]->SetSize(num_redElem);
    //    redTrueEntity_trueEntity[0]->Mult(topo.Weight(0), out->Weight(0));

    //    auto trueFacetWeight = topo.TrueWeight(1);
    //    out->Weights_[1]->SetSize(out->B_[0]->NumCols());
    //    redEntity_trueEntity[1]->Mult(*trueFacetWeight, *(out->Weights_[1]));
    
    return Graph(GetDiag(*redB), *redE_redTE, mfem::Vector(), &redE_bdrattr_local);
}

GraphSpace Redistributor::Redistribute(const GraphSpace& dof)
{
    // auto& nonconst_dof = const_cast<GraphSpace&>(dof);
    // auto dof_alg = dynamic_cast<GraphSpaceALG*>(&nonconst_dof);
    // PARELAG_ASSERT(dof_alg); // only GraphSpaceALG can be redistributed currently

    // const int dim = 2;//redist_graph->Dimensions();
    // const int max_codim = 1; //dof.GetMaxCodimensionBaseForDof();
    const int jform = 1; //dim - max_codim;
    // auto out = make_unique<GraphSpace>(std::move(*redist_graph));

    // auto elem = GraphTopology::ELEMENT;
    // smoothg has all the dofs (vdof and edof in GraphSpace), so need to do at the same time unlike parelag

    // redistribute vdofs
    mfem::SparseMatrix vert_vdof(dof.VertexToVDof());
    std::unique_ptr<mfem::HypreParMatrix> vert_truevdof(
        ToParMatrix(dof.GetGraph().GetComm(), vert_vdof));

    redDof_trueDof[0] = BuildRedEntToTrueEnt(*vert_truevdof);
    auto redvdof_redtruevdof = BuildRedEntToRedTrueEnt(*redDof_trueDof[0]);
    redTrueDof_trueDof[0] =
        BuildRedTrueEntToTrueEnt(*redvdof_redtruevdof, *redDof_trueDof[0]);

    std::unique_ptr<mfem::HypreParMatrix> truevdof_redvdof(redDof_trueDof[0]->Transpose());
    std::unique_ptr<mfem::HypreParMatrix> redvert_redvdof;
    {
        // std::unique_ptr<mfem::HypreParMatrix> redvert_redvdof(
        // Mult(*redEntity_trueEntity[0], *vert_truevdof, *truevdof_redvdof));
        auto vert_redvdof = ParMult(dof.VertexToVDof(), *truevdof_redvdof, dof.GetGraph().VertexStarts());
        redvert_redvdof.reset(ParMult(redEntity_trueEntity[0].get(), vert_redvdof.get()));
        // out->vertex_vdof_ = GetDiag(*redvert_redvdof);
    }


    // redistribute edofs
    auto& vert_edof = dof.VertexToEDof();
    auto& edof_trueedof = dof.EDofToTrueEDof();
    auto vert_trueedof = ParMult(vert_edof, edof_trueedof, dof.VDofStarts());

    redDof_trueDof[jform] = BuildRedEntToTrueEnt(*vert_trueedof);
    // auto redDof_redTrueDof = BuildRedEntToRedTrueEnt(*redDof_trueDof[jform]);
    // redTrueDof_trueDof[jform] = // TODO: prabably need to take case of rdof ordering
    //   BuildRedTrueEntToTrueEnt(*redDof_redTrueDof, *redDof_trueDof[jform]);
    // out->edof_trueedof_ = BuildRedEntToRedTrueEnt(*redDof_trueDof[jform]);
    // redTrueDof_trueDof[jform] = // TODO: prabably need to take care of rdof ordering
    //     BuildRedTrueEntToTrueEnt(*out->edof_trueedof_, *redDof_trueDof[jform]);

    std::unique_ptr<mfem::HypreParMatrix> trueedof_rededof(redDof_trueDof[jform]->Transpose());
    // out->dofTrueDof.SetUp(std::move(redDof_redTrueDof));

    // int myid;
    // MPI_Comm_rank(redist_graph->GetComm(), &myid);

    // for (int i = 0; i < max_codim+1; ++i)
    // {
        //    std::unique_ptr<mfem::HypreParMatrix> tE_tD;
        //    if (i > 0)
        std::unique_ptr<mfem::HypreParMatrix> rededge_rededof;
        {
            //   auto codim = static_cast<GraphTopology::Entity>(i);
            auto& edge_edof = dof.EdgeToEDof();
            //   tE_tD = IgnoreNonLocalRange(dof.GetEntityTrueEntity(i), ent_dof, dof_trueDof);
            auto edge_trueedof = ParMult(edge_edof, edof_trueedof, dof.GetGraph().EdgeStarts());
            auto trueedge_trueedof = ParMult(&dof.GetGraph().EdgeToTrueEdge(), edge_trueedof.get());
            *trueedge_trueedof = 1.0;//@todo probably need something like IgnoreNonLocalRange

            rededge_rededof.reset(
                Mult(*redEntity_trueEntity[1], *trueedge_trueedof, *trueedof_rededof));
            // out->edge_edof_ = GetDiag(*rededge_rededof);

        }
        //    else
        // {
            // std::unique_ptr<mfem::HypreParMatrix> redvert_rededof(
                // Mult(*redEntity_trueEntity[0], *vert_trueedof, *trueedof_rededof));
            // out->vertex_edof_ = GetDiag(*redvert_rededof);
        // }

        //    mfem::SparseMatrix redEnt_redDof_diag;
        //    redEnt_redDof->GetDiag(redEnt_redDof_diag);
        //    out->entity_dof[i].reset(new mfem::SparseMatrix(redEnt_redDof_diag));
        //    out->finalized[i] = true;
    // }


    GraphSpace out(std::move(redist_graph_), GetDiag(*rededge_rededof), GetDiag(*redvert_redvdof));
    redTrueDof_trueDof[jform] = // TODO: prabably need to take care of rdof ordering
        BuildRedTrueEntToTrueEnt(out.EDofToTrueEDof(), *redDof_trueDof[jform]);

    return out;
}

MixedMatrix Redistributor::Redistribute(const MixedMatrix& sequence)
{
    // const int dim = 2;
    const int num_forms = 2;

    // auto redist_seq = std::make_shared<MixedMatrix>(redist_topo, num_forms);
    // redist_seq->Targets_.resize(num_forms);



    const GraphSpace& space = sequence.GetGraphSpace();
    auto redist_space = Redistribute(space);
    
    mfem::SparseMatrix redD;
    for (int codim = 0; codim < num_forms; ++codim)
    {
        const int jform = 1;//num_forms - codim - 1;
        auto& redD_tD = redDof_trueDof[jform];

        // if (jform != (num_forms - 1))
        {
            // auto trueD = sequence.ComputeTrueD(jform);
            unique_ptr<mfem::HypreParMatrix> trueD(sequence.MakeParallelD(sequence.GetD()));
            unique_ptr<mfem::HypreParMatrix> tD_redD(redD_tD->Transpose());
            auto redTrueD = RAP(*redDof_trueDof[0], *trueD, *tD_redD);
            redTrueD->GetDiag(redD);
            // redist_seq->D_[jform].reset(new SerialCSRMatrix(redD_diag));
        }

        // redistribution of M is taken out of this look

        // const int true_size = dof_handler.GetDofTrueDof().GetTrueLocalSize();
        // const int true_size = space.EDofToTrueEDof().NumCols();
        
        // auto& Targets = *(sequence.Targets_[jform]);
        // MultiVector trueTargets(Targets.NumberOfVectors(), true_size);
        // trueTargets = 0.0;
        // dof_handler.GetDofTrueDof().IgnoreNonLocal(Targets, trueTargets);

        // const int redist_size = redist_seq->Dof_[jform]->GetNDofs();
        // redist_seq->Targets_[jform].reset(
            // new MultiVector(trueTargets.NumberOfVectors(), redist_size));
        // Mult(*redD_tD, trueTargets, *(redist_seq->Targets_[jform]));
    }

    std::vector<mfem::DenseMatrix> red_M_vert(redist_space.GetGraph().NumVertices());
    for (int codim = 0; codim < num_forms; ++codim)
    {
        // const int jform = num_forms - codim - 1;
        // if (jform < sequence.jformStart_) { break; }

        // auto type = static_cast<GraphTopology::Entity>(codim);

        // for (int j = sequence.jformStart_; j <= jform; ++j)
        {
            // mfem::SparseMatrix M(*const_cast<MixedMatrix&>(sequence).GetM(type, j));
            // Build block diagonal "mass" matrix
            auto vert_rdof = BuildVertRDof(space.VertexToEDof());
            mfem::Array<int> rdofs;
            mfem::SparseMatrix M(vert_rdof.NumCols(), vert_rdof.NumCols());
            for (int vert = 0; vert < vert_rdof.NumRows(); ++vert)
            {
                GetTableRow(vert_rdof, vert, rdofs); 
                M.SetSubMatrix(rdofs, rdofs, sequence.GetLocalMs()[vert]);
            }
            M.Finalize();
            
            unique_ptr<mfem::HypreParMatrix> trueM(ToParMatrix(redist_space.GetGraph().GetComm(), M));

            // M only exists for the vert-edof pair (codim=0, jform=1) in smoothG 
            unique_ptr<mfem::HypreParMatrix> red_trueM;
            // if (codim == 0) // redistribution of RDofs when codim=0 follows that of elements
            {
                // auto topo = const_cast<MixedMatrix&>(sequence).GetTopology();
                auto redRD_RD = BuildRepeatedEDofRedistribution(space, redist_space);
                unique_ptr<mfem::HypreParMatrix> RD_redRD(redRD_RD->Transpose());
                red_trueM.reset(Mult(*redRD_RD, *trueM, *RD_redRD));
            }
            // if (codim == 1) // codim-1 RDofs when jform=dim-2 are identified with true dofs
            // {
            //     auto RD_TD = BuildRepeatedEDofToTrueEDof(*sequence.Dof_[j]);
            //     auto redRD_redTD = BuildRepeatedEDofToTrueEDof(*redist_seq->Dof_[j]);
            //     unique_ptr<mfem::HypreParMatrix> redTD_redRD(redRD_redTD->Transpose());
            //     unique_ptr<mfem::HypreParMatrix> tD_redTD(redTrueDof_trueDof[j]->Transpose());

            //     auto trueM = IgnoreNonLocalRange(*RD_TD, *pM, *RD_TD);

            //     unique_ptr<mfem::HypreParMatrix> red_trueM(
            //         mfem::RAP(tD_redTD.get(), trueM.get(), tD_redTD.get()));
            //     redM.reset(mfem::RAP(red_trueM.get(), redTD_redRD.get()));
            // }
            // else
            // {
                // PARELAG_TEST_FOR_EXCEPTION(
                    // true, std::runtime_error,
                    // "redistribution of M when codim > 1 is not implemented.");
            // }

            mfem::SparseMatrix redM = GetDiag(*red_trueM);

            auto redvert_redrdof = BuildVertRDof(redist_space.VertexToEDof());
            for (int vert = 0; vert < redvert_redrdof.NumRows(); ++vert)
            {
                GetTableRow(redvert_redrdof, vert, rdofs);
                red_M_vert[vert].SetSize(rdofs.Size(), rdofs.Size());
                redM.GetSubMatrix(rdofs, rdofs, red_M_vert[vert]);
            }

            // const int idx = (dim - j) * (num_forms - j) / 2 + codim;
            // redist_seq->M_[idx].reset(new mfem::SparseMatrix(redM_diag));
        }
    }

    auto redTD_tD = TrueDofRedistribution(0);
    auto red_const_rep = Mult(redTD_tD, sequence.GetConstantRep());
    auto red_vert_sizes = Mult(redTD_tD, sequence.GetVertexSizes());

    unique_ptr<mfem::HypreParMatrix> tD_redTD(redTD_tD.Transpose());
    auto redW_tmp = ParMult(sequence.GetW(), *tD_redTD, space.VDofStarts());
    unique_ptr<mfem::HypreParMatrix> redTrueW(ParMult(&redTD_tD, redW_tmp.get()));
    auto redW = GetDiag(*redTrueW);

    auto redTE_tE = TrueEntityRedistribution(0);
    auto redP_pwc_tmp = ParMult(sequence.GetW(), *tD_redTD, space.VDofStarts());
    unique_ptr<mfem::HypreParMatrix> redTrueP_pwc(ParMult(&redTE_tE, redP_pwc_tmp.get()));
    auto redP_pwc = GetDiag(*redTrueP_pwc);

    return MixedMatrix(std::move(redist_space), std::move(red_M_vert), std::move(redD), std::move(redW), 
                       std::move(red_const_rep), std::move(red_vert_sizes), std::move(redP_pwc));
}

} // namespace parelag
