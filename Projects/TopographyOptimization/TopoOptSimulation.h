#ifndef TOPO_OPT_SIMULATION_H
#define TOPO_OPT_SIMULATION_H

#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/Timer.h>
#include <Ziran/CS/Util/DataDir.h>
#include <Ziran/Physics/ConstitutiveModel/NeoHookean.h>
#include <Ziran/Physics/ConstitutiveModel/LinearCorotated.h>
#include <Ziran/Physics/ConstitutiveModel/CorotatedIsotropic.h>
#include <Partio.h>
#include <openvdb/openvdb.h>
#include "TopoSpacialHash.h"
#include "getRSS.hpp"
#include "HexFEMSolver.h"
#include "StaticGrid.h"
#include "BoundaryCondition.h"

namespace ZIRAN {

template <class T, int dim>
class TopographyOptimization {
public:
    typedef Vector<T, dim> TV;
    typedef Vector<T, 3> TV3;
    typedef Vector<T, 2> TV2;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, dim, dim> TM;
    typedef Matrix<T, dim, Eigen::Dynamic> TVStack;
    typedef Matrix<T, Eigen::Dynamic, 1> VectorXT;
    typedef Matrix<bool, Eigen::Dynamic, 1> VectorXb;
    typedef Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXT;
    typedef Matrix<T, dim + 1, Eigen::Dynamic> Variables;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using Optimization = TopographyOptimization<T, dim>;
    using Hessian = Eigen::Matrix<T, dim * dim, dim * dim>;
    using Solver = HexFEMSolver<Optimization, BoundaryCondition<T, dim, Optimization>, T, dim>;
    using Scalar = T;

    using LinearModel = LinearCorotated<T, dim>;
    using NonLinearModel = NeoHookean<T, dim>;
    
    StdVector<TM> F;
    StdVector<TM> F_moved;
    StaticGrid<T, dim> grid;

    T dx = -1;
    T vol = -1;
    T m;
    TV gravity = TV::Zero();
    

    int num_nodes;
    int num_cells;
    int num_particles;
    const int ppc = 1 << dim; // quadratures per cell

    std::shared_ptr<LogWorker> logger;
    DataDir output_dir;

    VectorXi quadrature_flag;

    StdVector<TV> gauss_quadrature_offset;

    T linear_tol = 1e-5, newton_tol = 1e-2;
    bool use_iterative_solver = true;

    bool nonlinear = false;
    bool invertable = true;
    bool debug = false;
    bool difftest_implicit = false;
    
    bool quasi_static = true;
    
    T thickness = 0;

    TopographyOptimization()
    {
        openvdb::initialize();
        
        if constexpr (dim == 2)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                {
                    TV offset = (1./std::sqrt(3.) * TV(i, j)).array() + (0.5 - 0.5 / std::sqrt(3.));
                    gauss_quadrature_offset.push_back(offset);
                }
        else
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < 2; ++k)
                    {
                        TV offset = (1./std::sqrt(3.) * TV(i, j, k)).array() + (0.5 - 0.5 / std::sqrt(3.));
                        gauss_quadrature_offset.push_back(offset);
                    }
        if constexpr (std::is_same<NonLinearModel, NeoHookean<T, dim>>::value) invertable = false;
        
        gravity[1] = -9.81 * 1.0;
    }

    ~TopographyOptimization()
    {
        if (logger) {
            logger->printTimings();
            logger->finishWriting();
        }
    }

    void setGravity(TV gravity_in) { gravity = gravity_in; }

    void initializeLog(std::string output_path)
    {
        output_dir.path = output_path;
        output_dir.createPath();
        logger = LogWorker::initializeLogging();
        logger->openLogFile(output_dir.absolutePath("log.txt"), false);
    }
    
    
    VectorXT getGridCenterLocationDeformed(Eigen::Ref<const TVStack> u) 
    { 
        VectorXT nodal_locations(num_nodes * dim);
        iterateGrid([&](const IV& node, auto& grid_state, auto&) {
            if (grid_state.grid_idx < 0) return;
            TV grid_coord = node.template cast<T>() * dx;
            for (int d = 0; d < dim; d++)
                nodal_locations[grid_state.grid_idx*dim + d] = grid_coord[d];
        });
        

        VectorXT center_location(num_cells * dim);
        center_location.setZero();
        iterateCell([&](const IV& cell, const auto& cell_state, auto& grid) {
            if (cell_state.cell_idx < 0) return; 
            if (!cell_state.active_cell) return;           
            grid.iterateCellKernel(cell, [&](const IV& grid, const auto& g) {
                for (int d = 0; d < dim; d++)
                    center_location[cell_state.cell_idx * dim + d] += T(1) / ppc * (nodal_locations[g.grid_idx * dim + d] + u.col(g.grid_idx)[d]);
                // center_location.segment(cell_state.cell_idx, dim).array() += T(1) / ppc * (nodal_locations.segment(g.grid_idx, dim).array());// + u.col(g.grid_idx));
            });
        });
        return center_location; 
    }

     // only valid cells
    template <typename OP>
    void iterateCell(const OP& f) {
        
        grid.iterateGrid([&](auto& base_node, auto& cell_state){
            if (cell_state.cell_idx < 0) return;
            f(base_node, cell_state, grid);
        });
        
    }

    template <typename OP>
    void iterateCellSerial(const OP& f) {
        
        grid.iterateGridSerial([&](auto& base_node, auto& cell_state){
            if (cell_state.cell_idx < 0) return;
            f(base_node, cell_state, grid);
        });
        
    }

    template <typename OP>
    void iterateGrid(const OP& f) {
        
        grid.iterateGrid([&](auto& node, auto& grid_state){
            if (grid_state.grid_idx < 0) return;
            f(node, grid_state, grid);
        });
        
    }

    template <typename OP>
    void iterateGridSerial(const OP& f) {
        
        grid.iterateGridSerial([&](auto& node, auto& grid_state){
            if (grid_state.grid_idx < 0) return;
            f(node, grid_state, grid);
        });
        
    }

    template <typename OP>
    void coloredParEachCell(const OP& f)
    {
        using SparseMask = typename StaticGrid<T, dim>::SparseMask;
        
        StdVector<IV> indices;
        std::vector<std::pair<int, int>> groups[1 << dim];
        std::vector<std::pair<uint64_t, IV>> zorder_idx;
        zorder_idx.reserve(num_cells);

        grid.iterateGridSerial([&](auto& basenode, auto& cell_state) {
            if (cell_state.cell_idx < 0) return;
            uint64_t offset = grid.linearOffset(basenode);
            zorder_idx.push_back(std::make_pair(offset >> SparseMask::data_bits, basenode));
        });

        /* Sort nodes by linear offsets */
        std::sort(zorder_idx.begin(), zorder_idx.end(), [](const std::pair<uint64_t, IV>&a, const std::pair<uint64_t, IV>&b) {return a.first < b.first;});

        /* After sorting, each block is a continous segment*/
        indices.reserve(num_cells);
        int last_index = 0;
        for (int i = 0; i < (int) zorder_idx.size(); ++i) {
            indices.push_back(zorder_idx[i].second);
            if (i == (int) zorder_idx.size() - 1 || (zorder_idx[i].first >> SparseMask::block_bits) != (zorder_idx[i + 1].first >> SparseMask::block_bits)) {
                groups[(zorder_idx[i].first >> SparseMask::block_bits) & ((1 << dim) - 1)].push_back(std::make_pair(last_index, i));
                last_index = i + 1;
            }
        }
    
        for (int color = 0; color < (1 << dim); ++color) {
            tbb::parallel_for(0, (int)groups[color].size(), [&](int group_idx){
                auto& range = groups[color][group_idx];
                for (int idx = range.first; idx <= range.second; ++idx) 
                    f(indices[idx], grid[indices[idx]], grid);
            });
        }
        
    }

    void solveDisplacementField(BoundaryCondition<T, dim, Optimization> &bc, Eigen::Ref<TVStack> u)
    {
        ZIRAN_TIMER();
        quasi_static = true;
        
        ZIRAN_INFO("solve system");
        Solver solver(*this, bc, u, 1);
        TVStack du = TVStack::Zero(dim, num_nodes);
        updateDeformationGradient(u);
        
        solver.implicitUpdate(du, linear_tol, newton_tol);
        u += du;
    }

    void computeInnerForce(Eigen::Ref<TVStack> u, Eigen::Ref<TVStack> f)
    {
        updateDeformationGradient(u);
        
        f.resize(dim, u.cols());
        f.setZero();
        coloredParEachCell([&](const IV& cell, const auto& cell_state, auto& grid) {
            if (!cell_state.active_cell) return;
            T vol = this->vol;
            for (int qp_loop = 0; qp_loop < ppc; qp_loop++) {
                
                TM firstPiola = this->firstPiola(F[ppc * cell_state.cell_idx + qp_loop], cell_state.E, cell_state.nu);
                grid.iterateKernel(
                    cell, qp_loop, dx, [&](const int& dof, const IV& node, const T& w, const TV& dw) {
                        if (dof < 0) return;
                        TV vPFTw = vol * firstPiola * dw;
                        f.col(dof) -= vPFTw;
                    });
            }
        });
    }

    void exportHessian(BoundaryCondition<T, dim, Optimization> &bc, Eigen::Ref<TVStack> u)
    {
        updateDeformationGradient(u);
        Solver solver(*this, bc, u);
    }


    T computeCompliance(Eigen::Ref<const TVStack> u)
    {
        VectorXT psis(ppc * num_cells);
        psis.setZero();
        updateDeformationGradient(u);
        iterateCell([&](const IV& cell, const auto& cell_state, auto&) {
            T vol = this->vol;
            for (int qp_loop = 0; qp_loop < ppc; qp_loop++) {
                psis(cell_state.cell_idx * ppc + qp_loop) = vol * psi(F[cell_state.cell_idx * ppc + qp_loop], cell_state.E, cell_state.nu);
            }
        });
        return psis.sum();
    }


public:
    

    T psi(TM& F, T E, T nu)
    {
        if (nonlinear) {
            typename NonLinearModel::Scratch s;
            NonLinearModel model(E, nu);
            model.project = false;
            model.updateScratch(F, s);
            return model.psi(s);
        }
        else {
            typename LinearModel::Scratch s;
            LinearModel model(E, nu);
            model.updateScratch(F, s);
            return model.psi(s);
        }
    }

    TM firstPiola(TM& F, T E, T nu)
    {
        TM P = TM::Zero();
        if (nonlinear) {
            typename NonLinearModel::Scratch s;
            NonLinearModel model(E, nu);
            model.project = false;
            model.updateScratch(F, s);
            model.firstPiola(s, P);
        }
        else {
            typename LinearModel::Scratch s;
            LinearModel model(E, nu);
            model.updateScratch(F, s);
            model.firstPiola(s, P);
        }
        return P;
    }

    Hessian firstPiolaDerivative(TM& F, T E, T nu, bool project = false)
    {
        Hessian dPdF = Hessian::Zero();
        if (nonlinear) {
            typename NonLinearModel::Scratch s;
            NonLinearModel model(E, nu);
            model.project = project;
            model.updateScratch(F, s);
            model.firstPiolaDerivative(s, dPdF);
        }
        else {
            typename LinearModel::Scratch s;
            LinearModel model(E, nu);
            model.updateScratch(TM::Identity(), s);
            model.firstPiolaDerivative(s, dPdF);
        }
        return dPdF;
    }

    void updateMovedDeformationGradient(Eigen::Ref<const TVStack> du)
    {
        F_moved = StdVector<TM>(ppc * num_cells, TM::Identity());
        iterateCell([&](const IV& cell, const auto& cell_state, auto& grid) {
            if (!cell_state.active_cell) return;
            for (int qp_loop = 0; qp_loop < ppc; qp_loop++) {
                TM Fn = TM::Zero();
                grid.iterateKernel(
                    cell, qp_loop, dx, [&](const auto& dof, const IV& node, const T& w, const TV& dw) {
                        if (dof < 0) return;
                        Fn += du.col(dof) * dw.transpose();
                    });
                F_moved[ppc * cell_state.cell_idx + qp_loop] = F[ppc * cell_state.cell_idx + qp_loop] + Fn;
                // if (!invertable) { ZIRAN_ASSERT(F_moved[ppc * cell_state.cell_idx + qp_loop].determinant() > 0); }
            }
        });
    }
    
    void updateDeformationGradient(Eigen::Ref<const TVStack> u)
    {
        F.resize(ppc * num_cells);
        tbb::parallel_for(0, ppc * num_cells, [&](int i) {
            F[i].setIdentity();
        });

        iterateCell([&](const IV& cell, const auto& cell_state, auto& grid) {
            if (!cell_state.active_cell) return;
            for (int qp_loop = 0; qp_loop < ppc; qp_loop++) {
                TM Fn = TM::Zero();
                grid.iterateKernel(
                    cell, qp_loop, dx, [&](const auto& dof, const IV& node, const T& w, const TV& dw) {
                        if (dof < 0) return;
                        Fn += u.col(dof) * dw.transpose();
                    });
                F[ppc * cell_state.cell_idx + qp_loop] = TM::Identity() + Fn;
            }
        });

        F_moved=F;
    }

    void fixGridNeighborCells(TV grid_coord)
    {
        IV node = (grid_coord / dx).array().round().template cast<int>();
        
        grid.iterateNodeCell(node, [&](const IV& neighbor, auto& neighbor_state) {
            neighbor_state.fixed_density = true;
        });
        
    }

    void reinitializeGrid()
    {
        
        iterateGrid([&](const IV& grid, auto& grid_state, auto&) {
            grid_state.grid_idx = -1;
        });
        iterateCellSerial([&](const IV& cell, auto& cell_state, auto& grid) {
            
            cell_state.active_cell = true;
        });

        num_nodes = 0;
        iterateCellSerial([&](const IV& cell, auto& cell_state, auto& grid) {
            
            if (!cell_state.active_cell) return;
            
            grid.iterateCellKernel(cell, [&](const IV& node, auto& grid_state) {
                if (grid_state.grid_idx == -1) {
                    grid_state.grid_idx = num_nodes++; }
            });
        });
        ZIRAN_INFO("number of nodes: ", num_nodes);
    }

public:

    void getMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, const TVStack& u)
    {
        if constexpr (dim == 3) 
        {
            std::vector<std::vector<int>> face(
                {std::vector<int>({2, 3, 7, 6}), 
                std::vector<int>({0, 4, 5, 1}),
                std::vector<int>({4, 6, 7, 5}),
                std::vector<int>({0, 1, 3, 2}),
                std::vector<int>({1, 5, 7, 3}),
                std::vector<int>({0, 2, 6, 4}),
                });

            std::vector<IV> shift({IV(0, 1, 0), 
                                IV(0, -1, 0),
                                IV(1, 0, 0),
                                IV(-1, 0, 0),
                                IV(0, 0, 1),
                                IV(0, 0, -1)});
            
            int n_vertex = 0;

            auto& coarse_grid = grid;

            auto boundary_cell = [&](IV base_node) -> bool {
                // return true;
                int n_neighbors = 0;
                coarse_grid.iterateAxisAlignedOneLaynerNeighbor(base_node, [&](const IV& neighbor, auto& neighbor_state) {
                    if (base_node == neighbor) return;
                    if (!neighbor_state.active_cell) return;
                    n_neighbors += 1;        
                });
                return n_neighbors != 2 * dim;
            };

            // set visiting flag
            std::vector<int> visited(num_nodes, -1);
            
            std::vector<TV> vertex_list;
            std::vector<Vector<int, 4>> face_list;
            std::vector<int> rho_id_list;
            coarse_grid.iterateGridSerial([&](const IV& node, auto& grid_state) {
                if (grid_state.cell_idx < 0) return;
                if (!grid_state.active_cell) return;
                if (!boundary_cell(node)) return;
                // six faces
                for (int i = 0; i < 6; ++i) {
                    IV index = shift[i] + node;
                    if (!coarse_grid.allocated(index) || !coarse_grid[index].active_cell) {
                        // nothing on top of shift direction -> boundary face
                        Vector<int, 4> vertex_loop;
                        vertex_loop.setConstant(-1);
                        // four vertices
                        for (int j = 0; j < 4; ++j) {
                            IV grid_index = node + coarse_grid.offset_vector[face[i][j]];
                            TV material_coord = grid_index.template cast<T>() ;
                            IV nearest_node = material_coord.array().round().template cast<int>();
                            IV base_node = nearest_node;
                            coarse_grid.iterateNodeCell(nearest_node, [&](const IV& cell, const auto& cell_state) {
                                if (!cell_state.active_cell) return;
                                if ((material_coord - cell.template cast<T>()).minCoeff() >= 0 && (material_coord - cell.template cast<T>()).maxCoeff() <= 1)
                                    base_node = cell;
                            });
                            TV grid_u = u.col(coarse_grid[grid_index].grid_idx);//TV::Zero();
                            TV X = grid_index.template cast<T>().transpose();
                            
                        
                            if (visited[coarse_grid[grid_index].grid_idx] == -1) {
                                visited[coarse_grid[grid_index].grid_idx] = n_vertex++;
                                TV vtx_pos = grid_u.transpose() + grid_index.template cast<T>().transpose() * dx;
                                vertex_list.push_back(vtx_pos);
                                // f << "v " << grid_u.transpose() + grid_index.template cast<T>().transpose() * dx  <<std::endl;
                            }
                            vertex_loop(j) = visited[coarse_grid[grid_index].grid_idx];
                        }
                        // f << "f " << vertex_loop.transpose().array() + 1 << std::endl;
                        face_list.push_back(vertex_loop);
                        rho_id_list.push_back(grid_state.cell_idx);
                    }
                }
            });

            V.resize(vertex_list.size(), 3);
            F.resize(face_list.size() * 2, 3);
            C.resize(face_list.size() * 2, 3);

            tbb::parallel_for(0, (int)vertex_list.size(), [&](int i)
            {
                V.row(i) = vertex_list[i];
            });

            tbb::parallel_for(0, (int)face_list.size(), [&](int i)
            {
                F.row(i * 2) = face_list[i].template segment<3>(0);
                F.row(i * 2 + 1) = Eigen::Vector3i(face_list[i][0], face_list[i][2], face_list[i][3]);
                C.row(i * 2) = Eigen::Vector3d(0, 0.3, 1);
                C.row(i * 2 + 1) = Eigen::Vector3d(0, 0.3, 1);
            });
        }
    }
    
    void getBBox(IV& min_corner, IV& max_corner)
    {
        min_corner = grid.min_corner;
        max_corner = grid.max_corner;
    }

public: // ShapeInitializer.cpp
    void initializeDesignPad(T dx, TV min_corner, TV max_corner);
    
    void addLambdaShape(std::function<bool(const TV&)> &cell_helper, T E, T nu, T density, bool fixed_density);
    void addBox(TV min_corner, TV max_corner, T E, T nu, T density, bool fixed_density);
    void addSphere(TV center, T radius, T E, T nu, T density, bool fixed_density);
    void finalizeDesignDomain();

};
} // namespace ZIRAN

#endif