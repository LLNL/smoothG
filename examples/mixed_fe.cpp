//                       MFEM Example 5 - Parallel Version
//
// Compile with: make ex5p
//
// Sample runs:  mpirun -np 4 ex5p -m ../data/square-disc.mesh
//               mpirun -np 4 ex5p -m ../data/star.mesh
//               mpirun -np 4 ex5p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex5p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex5p -m ../data/escher.mesh
//               mpirun -np 4 ex5p -m ../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockMatrix class, as
//               well as the collective saving of several grid functions in a
//               VisIt (visit.llnl.gov) visualization format.
//
//               We recommend viewing example 5 before viewing this miniapp.

#include "mfem.hpp"
#include "../src/mixed_fe_solvers.hpp"
#include "../src/smoothG.hpp"
#include "pde.hpp"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <memory>

using namespace std;
using namespace mfem;
using namespace smoothg;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);

//     Assemble the finite element matrices for the Darcy problem
//
//                            D = [ M  B^T ]
//                                [ B   0  ]
//     where:
//
//     M = \int_\Omega u_h \cdot v_h d\Omega   u_h, v_h \in R_h
//     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
class FEDarcyProblem
{
    OperatorPtr M_;
    OperatorPtr B_;
    Vector rhs_;
    Vector ess_data_;
    ParGridFunction u_;
    ParGridFunction p_;
    ParMesh mesh_;
    VectorFunctionCoefficient ucoeff_;
    FunctionCoefficient pcoeff_;
    DFSDataCollector collector_;
    const IntegrationRule *irs_[Geometry::NumGeom];
    unique_ptr<MixedMatrix> mixed_system_;
public:
    FEDarcyProblem(Mesh* mesh, int num_refines, int order,
                 Array<int>& ess_bdr, DFSParameters param); // TODO: better design

    HypreParMatrix& GetM() { return *M_.As<HypreParMatrix>(); }
    HypreParMatrix& GetB() { return *B_.As<HypreParMatrix>(); }
    const Vector& GetRHS() { return rhs_; }
    const Vector& GetBC() { return ess_data_; }
    const DFSDataCollector& GetDFSDataCollector() const { return collector_; }
    const MixedMatrix& GetMixedMatrix() const { return *mixed_system_; }

    void ShowError(const Vector& sol, bool verbose);
};

FEDarcyProblem::FEDarcyProblem(Mesh* mesh, int num_refines, int order,
                               Array<int>& ess_bdr, DFSParameters dfs_param)
    : mesh_(MPI_COMM_WORLD, *mesh), ucoeff_(mesh_.Dimension(), uFun_ex),
      pcoeff_(pFun_ex), collector_(order, num_refines, &mesh_, ess_bdr, dfs_param)
{
    delete mesh;
    for (int l = 0; l < num_refines; l++)
    {
        mesh_.UniformRefinement();
        collector_.CollectData(&mesh_);
    }

    // Define the coefficients, analytical solution, and rhs of the PDE.
    VectorFunctionCoefficient fcoeff(mesh_.Dimension(), fFun);
    FunctionCoefficient fnatcoeff(f_natural);
    FunctionCoefficient gcoeff(gFun);

    u_.SetSpace(collector_.hdiv_fes_.get());
    p_.SetSpace(collector_.l2_fes_.get());
    u_ = 0.0;
    u_.ProjectBdrCoefficientNormal(ucoeff_, ess_bdr);

    ParLinearForm fform(collector_.hdiv_fes_.get());
    fform.AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
    fform.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
    fform.Assemble();

    ParLinearForm gform(collector_.l2_fes_.get());
    gform.AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
    gform.Assemble();

    ParBilinearForm mVarf(collector_.hdiv_fes_.get());
    ParMixedBilinearForm bVarf(&(*collector_.hdiv_fes_), &(*collector_.l2_fes_));

    mVarf.AddDomainIntegrator(new VectorFEMassIntegrator);
    mVarf.ComputeElementMatrices();
    mVarf.Assemble();
    mVarf.EliminateEssentialBC(ess_bdr, u_, fform);
    mVarf.Finalize();
    M_.Reset(mVarf.ParallelAssemble());

    bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf.Assemble();
    bVarf.SpMat() *= -1.0;
    bVarf.Finalize();
    mfem::SparseMatrix D_tmp(bVarf.SpMat());
    bVarf.EliminateTrialDofs(ess_bdr, u_, gform);
    B_.Reset(bVarf.ParallelAssemble());

//    if (order == 0)
    {
        RT_FECollection RT0_fec(0, mesh_.Dimension());
        ParFiniteElementSpace RT0_fes(&mesh_, &RT0_fec);

        auto edge_bdratt = GenerateBoundaryAttributeTable(&mesh_);
        auto vertex_edge = TableToMatrix(mesh_.ElementToFaceTable());
        auto& edge_trueedge = *(RT0_fes.Dof_TrueDof_Matrix()); //TODO: for higher order this doesn't work!
        Graph graph(vertex_edge, edge_trueedge, mfem::Vector(), &edge_bdratt);
        GraphSpace graph_space(std::move(graph), *collector_.hdiv_fes_, *collector_.l2_fes_);

        std::vector<mfem::DenseMatrix> M_el(graph_space.GetGraph().NumVertices());
        mfem::Array<int> vdofs, reordered_edges, original_edges;
        for (int i = 0; i < mesh_.GetNE(); ++i)
        {
            mVarf.ComputeElementMatrix(i, M_el[i]);

            DenseMatrix sign_fix(M_el[i].NumRows());
            collector_.hdiv_fes_.get()->GetElementVDofs(i, vdofs);
            for (int j = 0; j < sign_fix.NumRows(); ++j)
            {
                sign_fix(j, j) = vdofs[j] < 0 ? -1.0 : 1.0;
            }

            mfem::DenseMatrix help(sign_fix);
            mfem::Mult(M_el[i], sign_fix, help);
            mfem::Mult(sign_fix, help, M_el[i]);

//            GetTableRow(graph.VertexToEdge(), i, reordered_edges);
//            GetTableRow(vertex_edge, i, original_edges);
//            mfem::DenseMatrix local_reorder_map(reordered_edges.Size());
//            for (int j = 0; j < reordered_edges.Size(); ++j)
//            {
//                int reordered_edge = reordered_edges[j];
//                int original_edge = graph.EdgeReorderMap().GetRowColumns(reordered_edge)[0];
//                int original_local = original_edges.Find(original_edge);
//                assert(original_local != -1);
//                local_reorder_map(original_local, j) = 1;
//            }

//            mfem::Mult(M_el[i], local_reorder_map, help);
//            local_reorder_map.Transpose();
//            mfem::Mult(local_reorder_map, help, M_el[i]);
        }cout<<"yoyoy2\n";
        auto mbuilder = make_unique<ElementMBuilder>(std::move(M_el), graph_space.VertexToEDof());

//        auto edge_reoder_mapT = smoothg::Transpose(graph.EdgeReorderMap());
        mfem::SparseMatrix D = D_tmp;//smoothg::Mult(D_tmp, edge_reoder_mapT);

        mfem::SparseMatrix W;
        mfem::Vector const_rep(graph_space.VertexToVDof().NumCols());
        const_rep = 1.0 / std::sqrt(collector_.l2_fes_->GlobalTrueVSize());

        mfem::Vector vertex_sizes(graph_space.GetGraph().NumVertices());
        vertex_sizes = 1.0;

        mfem::SparseMatrix P_pwc = SparseIdentity(const_rep.Size());

//        mfem::Vector edge_bc(graph_space.VertexToEDof().NumCols());
//        edge_bc = u_;
cout<<"yoyoy3\n";
        mixed_system_.reset(new MixedMatrix(std::move(graph_space), std::move(mbuilder),
                                            std::move(D), std::move(W), std::move(const_rep),
                                            std::move(vertex_sizes), std::move(P_pwc)));
        mixed_system_->SetEssDofs(ess_bdr);
        mixed_system_->BuildM();cout<<"yoyoy4\n";
    }

    rhs_.SetSize(M_->NumRows()+B_->NumRows());
    rhs_ = 0.0;
    Vector block0_view(rhs_.GetData(), M_->NumRows());
    Vector block1_view(rhs_.GetData()+M_->NumRows(), B_->NumRows());
    fform.ParallelAssemble(block0_view);
    gform.ParallelAssemble(block1_view);

    ess_data_.SetSize(M_->NumRows()+B_->NumRows());
    ess_data_ = 0.0;
    Vector ess_data_block0(ess_data_.GetData(), M_->NumRows());
    u_.ParallelProject(ess_data_block0);

    int order_quad = max(2, 2*order+1);
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs_[i] = &(IntRules.Get(i, order_quad));
    }
}

void FEDarcyProblem::ShowError(const Vector &sol, bool verbose)
{
    u_.Distribute(Vector(sol.GetData(), M_->NumRows()));
    p_.Distribute(Vector(sol.GetData()+M_->NumRows(), B_->NumRows()));

    double err_u  = u_.ComputeL2Error(ucoeff_, irs_);
    double norm_u = ComputeGlobalLpNorm(2, ucoeff_, mesh_, irs_);
    double err_p  = p_.ComputeL2Error(pcoeff_, irs_);
    double norm_p = ComputeGlobalLpNorm(2, pcoeff_, mesh_, irs_);

    {
        int num_procs, myid;
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream u_sock(vishost, visport);
       u_sock << "parallel " << num_procs << " " << myid << "\n";
       u_sock.precision(8);
       u_sock << "solution\n" << mesh_ << p_ << "window_title 'Primal variable S'"
              << endl;

       MPI_Barrier(mesh_.GetComm());
       socketstream sigma_sock(vishost, visport);
       sigma_sock << "parallel " << num_procs << " " << myid << "\n";
       sigma_sock.precision(8);
       sigma_sock << "solution\n" << mesh_ << u_ << "window_title 'Flux sigma'"
                  << endl;
    }

    if (!verbose) return;
    cout << "\n|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
    cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
}

int main(int argc, char *argv[])
{
    StopWatch chrono;
    auto ResetTimer = [&chrono]() { chrono.Clear(); chrono.Start(); };

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    bool verbose = (myid == 0);

    // 2. Parse command-line options.
    int order = 0;
    int num_refines = 2;
    bool use_tet_mesh = false;
    bool show_error = false;
    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&num_refines, "-r", "--ref",
                   "Number of parallel refinement steps.");
    args.AddOption(&use_tet_mesh, "-tet", "--tet-mesh", "-hex", "--hex-mesh",
                   "Use a tetrahedral or hexahedral mesh (on unit cube).");
    args.AddOption(&show_error, "-se", "--show-error", "-no-se", "--no-show-error",
                   "Show or not show approximation error.");
    args.Parse();
    if (!args.Good())
    {
        if (verbose) args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }
    if (verbose) args.PrintOptions(cout);

    auto elem_type = use_tet_mesh ? Element::TETRAHEDRON : Element::HEXAHEDRON;
    Mesh* mesh = new Mesh(2, 2, 2, elem_type, true);
    for (int i = 0; i < (int)(log(num_procs)/log(8)); ++i)
         mesh->UniformRefinement();

    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
//    ess_bdr[1] = 1;

    IterSolveParameters param;
    DFSParameters dfs_param;
    dfs_param.MG_type = order > 0 && use_tet_mesh ? AlgebraicMG : GeometricMG;
    dfs_param.B_has_nullity_one = (ess_bdr.Sum() == ess_bdr.Size());
    if (order > 0 && use_tet_mesh) dfs_param.ml_particular = false;

    string line = "\n*******************************************************\n";
    {
        ResetTimer();
        FEDarcyProblem darcy(mesh, num_refines, order, ess_bdr, dfs_param);
        HypreParMatrix& M = darcy.GetM();
        HypreParMatrix& B = darcy.GetB();
        const DFSDataCollector& collector = darcy.GetDFSDataCollector();

        if (verbose)
        {
            cout << line << "dim(R) = " << M.M() << ", dim(W) = " << B.M() << ", ";
            cout << "dim(N) = " << collector.hcurl_fes_->GlobalTrueVSize() << "\n";
            cout << "System assembled in " << chrono.RealTime() << "s.\n";
        }


        MinresBlockSolver minres(darcy.GetMixedMatrix(), &ess_bdr);
        const Vector& rhs = darcy.GetRHS();
        Vector sol = darcy.GetBC();
        Array<int> offsets(3);
        offsets = 0; offsets[1] = M.NumRows(); offsets[2] = offsets[1]+B.NumRows();
        BlockVector blk_sol(sol.GetData(), offsets);
        BlockVector blk_rhs(rhs.GetData(), offsets);

        ResetTimer();
        minres.Mult(blk_rhs, blk_sol);

        if (verbose) cout << "  solve time: " << chrono.RealTime() << "s.\n";
        if (verbose) cout << "  iteration count: "
                          << minres.GetNumIterations() <<"\n";
        if (show_error) darcy.ShowError(sol, verbose);

        std::map<const DarcySolver*, double> setup_time;
        std::map<const DarcySolver*, std::string> solver_to_name;

        ResetTimer();
        DivFreeSolver dfs(M, B, collector.hcurl_fes_.get(), collector.GetData());
        setup_time[&dfs] = chrono.RealTime();
//        solver_to_name[&dfs] = "Divergence free";

        ResetTimer();
        BDPMinresSolver bdp(M, B, param);
        setup_time[&bdp] = chrono.RealTime();
//        solver_to_name[&bdp] = "Block-diagonal-preconditioned MINRES";

        unique_ptr<HybridSolver> hybrid;
//        if (order == 0)
        {
            ResetTimer();
            hybrid.reset(new HybridSolver(darcy.GetMixedMatrix(), &ess_bdr, 0));
            setup_time[hybrid.get()] = chrono.RealTime();
            solver_to_name[hybrid.get()] = "Hybridization";
        }

        for (const auto& solver_pair : solver_to_name)
        {
            auto& solver = solver_pair.first;
            auto& name = solver_pair.second;

            if (verbose) cout << line << name << " solver:\n";
            if (verbose) cout << "  setup time: " << setup_time[solver] << "s.\n";

            const Vector& rhs = darcy.GetRHS();
            Vector sol = darcy.GetBC();
            ResetTimer();
            solver->Mult(rhs, sol);

            if (verbose) cout << "  solve time: " << chrono.RealTime() << "s.\n";
            if (verbose) cout << "  iteration count: "
                              << solver->GetNumIterations() <<"\n";
            if (show_error) darcy.ShowError(sol, verbose);
        }
    }
    if (verbose) cout << line << "\n";

    MPI_Finalize();
    return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
    }

    u(0) = - exp(xi)*sin(yi)*cos(zi);
    u(1) = - exp(xi)*cos(yi)*cos(zi);

    if (x.Size() == 3)
    {
        u(2) = exp(xi)*sin(yi)*sin(zi);
    }
}

// Change if needed
double pFun_ex(const Vector & x)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);

    if (x.Size() == 3)
    {
        zi = x(2);
    }

    return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
    f = 0.0;
}

double gFun(const Vector & x)
{
    if (x.Size() == 3)
    {
        return -pFun_ex(x);
    }
    else
    {
        return 0;
    }
}

double f_natural(const Vector & x)
{
    return (-pFun_ex(x));
}
