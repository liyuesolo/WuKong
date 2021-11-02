#include "TopographyOptimization.h"
#include "autodiff/Distance.h"

#include <algorithm>
#include <Eigen/PardisoSupport>

template<class T, int dim, class Solver>
void TopographyOptimization<T, dim, Solver>::addBeads(BeadType bead_type,
    const TV& min_corner, const TV& max_corner, const TV& dx)
{
    IV scene_range = solver.scene_range;

    if constexpr (dim == 3)
    {
        if (bead_type == BeadRib)
        {
            if (solver.name == "HexFEM")
            {
                int node_cnt = 0;
                int cnt_bead = 0;

                T sign = 1.0;

                for (T x = min_corner[0]; x < max_corner[0] + 0.1 * dx[0]; x += dx[0])
                {
                    for (T y = min_corner[1]; y < max_corner[1] + 0.1 * dx[1]; y += dx[1])
                    {
                        for (T z = min_corner[2]; z < max_corner[2] + 0.1 * dx[2]; z += dx[2])
                        {
                            if (sign > 0)
                                undeformed[node_cnt * dim + 1] += 2.0 * sign * dx[1];
                            cnt_bead ++;
                            if (cnt_bead % 10 == 0)
                            {
                                sign *= -1;
                            }
                            node_cnt++; 
                        }
                        cnt_bead = 0;
                    }
                }
            }
        }
        else if (bead_type == VRib)
        {
            if (solver.name == "HexFEM")
            {
                int node_cnt = 0;
                int cnt_bead = 0;

                T sign = 1.0;
                // int n_v_rib = std::min(5, (max_corner[2] - min_corner[2]) / 4 / dx[2]);
                
                // std::cout << n_v_rib << std::endl;

                Vector<T, 5> offset;
                offset << 0.0, dx[1], 2.0 * dx[1], dx[1], 0;
                int offset_cnt = 0;
                bool add_bead = false;

                for (T x = min_corner[0]; x < max_corner[0] + 0.1 * dx[0]; x += dx[0])
                {
                    for (T y = min_corner[1]; y < max_corner[1] + 0.1 * dx[1]; y += dx[1])
                    {
                        for (T z = min_corner[2]; z < max_corner[2] + 0.1 * dx[2]; z += dx[2])
                        {
                            if (cnt_bead % 8 == 0)
                            {
                                add_bead = true;
                            }
                            if (add_bead)
                            {
                                undeformed[node_cnt * dim + 1] += offset[offset_cnt];
                                offset_cnt++;
                            }
                            if (offset_cnt == 5)
                            {
                                add_bead = false;
                                offset_cnt = 0;
                            }
                            cnt_bead ++;
                            
                            node_cnt++; 
                        }
                        cnt_bead = 0;
                        offset_cnt = 0;
                        add_bead = false;
                    }
                }
            }
            else if (solver.name == "ShellFEM")
            {
                for (int i = 0; i < scene_range[0]; i++)
                {
                    for (int j = 1; j < scene_range[2] - 1; j += 3)
                    {
                        undeformed[i * scene_range[2] * dim + j * dim + 1] += 0.01;
                    }    
                }
                
            }
        }
        else if (bead_type == DiagonalBead)
        {
            if (solver.name == "HexFEM")
            {
                int node_cnt = 0;
                
                for (T x = min_corner[0]; x < max_corner[0] + 0.1 * dx[0]; x += dx[0])
                {
                    for (T y = min_corner[1]; y < max_corner[1] + 0.1 * dx[1]; y += dx[1])
                    {
                        for (T z = min_corner[2]; z < max_corner[2] + 0.1 * dx[2]; z += dx[2])
                        {
                            if (std::abs(x - z) < 0.05 || std::abs(max_corner[0] - x - z) < 0.05)
                                undeformed[node_cnt * dim + 1] += 2.0 * dx[1];
                            node_cnt++; 
                        }
                    }
                }
            }
            else if (solver.name == "ShellFEM")
            {
                for (int i = 0; i < scene_range[0]; i++)
                {
                    for (int j = 0; j < scene_range[2]; j++)
                    {
                        if (i == j || (i - 1) == j || (i + 1 == j))
                        {
                            undeformed[i * scene_range[2] * dim + j * dim + 1] = 0.02;
                            undeformed[(scene_range[0] - 1 - i) * scene_range[2] * dim + j * dim + 1] = 0.02;
                        }
                    }
                    
                }
                
            }
        }
        else if (bead_type == FourParts)
        {
            if (solver.name == "HexFEM")
            {
                Vector<T, 5> offset;
                offset << 0.0, dx[1], 2.0 * dx[1], dx[1], 0;
                int offset_cnt = 0;
                bool add_bead = false;
                int cnt_bead = 0;

                IV scene_range = solver.scene_range;
                IV half_range = scene_range / 2;
                for (int i = 0; i < scene_range[0] / 2; i++)
                {
                    for (int j = 0; j < scene_range[2] / 2; j++)
                    {
                        if (cnt_bead % 8 == 0)
                        {
                            add_bead = true;
                        }
                        if (add_bead)
                        {
                            int idx = solver.globalOffset(IV(i, 0, j));
                            undeformed[idx * dim + 1] += offset[offset_cnt];
                            idx = solver.globalOffset(IV(i, 1, j));
                            undeformed[idx * dim + 1] += offset[offset_cnt];

                            idx = solver.globalOffset(IV(half_range[0] + i, 0, half_range[2] + j));
                            undeformed[idx * dim + 1] += offset[offset_cnt];
                            idx = solver.globalOffset(IV(half_range[0] + i, 1, half_range[2] + j));
                            undeformed[idx * dim + 1] += offset[offset_cnt];

                            idx = solver.globalOffset(IV(half_range[0] + j, 0, i));
                            undeformed[idx * dim + 1] += offset[offset_cnt];
                            idx = solver.globalOffset(IV(half_range[0] + j, 1, i));
                            undeformed[idx * dim + 1] += offset[offset_cnt];


                            idx = solver.globalOffset(IV(j, 0, half_range[2] + i));
                            undeformed[idx * dim + 1] += offset[offset_cnt];
                            idx = solver.globalOffset(IV(j, 1, half_range[2] + i));
                            undeformed[idx * dim + 1] += offset[offset_cnt];

                            offset_cnt++;
                        }
                        if (offset_cnt == 5)
                        {
                            add_bead = false;
                            offset_cnt = 0;
                        }
                        cnt_bead ++;
                    }
                    cnt_bead = 0;
                    offset_cnt = 0;
                    add_bead = false;
                }
            }
            else if (solver.name == "ShellFEM")
            {
                IV half_range = scene_range / 2;
                for (int i = 0; i < half_range[0]; i++)
                {
                    for (int j = 0; j < half_range[2]; j += 3)
                    {
                        undeformed[i * scene_range[2] * dim + j * dim + 1] += 0.02;
                        undeformed[(i + half_range[0]) * scene_range[2] * dim + (j + half_range[2]) * dim + 1] += 0.02;

                        undeformed[(j + half_range[0]) * scene_range[2] * dim + i * dim + 1] += 0.02;
                        undeformed[j * scene_range[2] * dim + (i + half_range[2]) * dim + 1] += 0.02;
                    }    
                }
            }
        }
        else if (bead_type == Circle)
        {
            if (solver.name == "HexFEM")
            {
                int node_cnt = 0;
                TV center = 0.5 * (min_corner + max_corner);
                for (T x = min_corner[0]; x < max_corner[0] + 0.1 * dx[0]; x += dx[0])
                {
                    for (T y = min_corner[1]; y < max_corner[1] + 0.1 * dx[1]; y += dx[1])
                    {
                        for (T z = min_corner[2]; z < max_corner[2] + 0.1 * dx[2]; z += dx[2])
                        {
                            // std::cout << TV(x, y, z).transpose() << " " << center.transpose() << std::endl;
                            if ((TV(x, y, z) - center).norm() < 0.15)
                            {
                                undeformed[node_cnt * dim + 1] += 2.0 * dx[1];
                            }
                            node_cnt++; 
                        }
                    }
                }
            }
        }
        else if (bead_type == CurveBD)
        {
            IV scene_range = solver.scene_range;
            if (solver.name == "ShellFEM")
            {
                int n = 1;
                for (int j = 0; j < n; j++)
                {
                    for (int i = 0; i < scene_range[2]; i++)
                    {
                        undeformed[ j * scene_range[2] * dim + i * dim + 1] -= (n - j) * dx[0];
                        undeformed[(scene_range[0] - 1 - j) * scene_range[2] * dim + i * dim + 1] -= (n - j) * dx[0];
                    }
                    for (int i = 1; i < scene_range[0] - 1; i++)
                    {
                        // undeformed[i * dim + 1] -= dx[0];
                        undeformed[i * scene_range[2] * dim + 1] -= dx[0];
                        undeformed[i * scene_range[2] * dim + 1 + (scene_range[2] - 1) * dim] -= dx[0];
                    }
                }
            }
            
        }
        else if (bead_type == Levelset)
        {
            addBeadLevelset(1);
            VectorXT rest_shape = solver.undeformed;
            tbb::parallel_for(0, solver.num_nodes, [&](int i){
                for (auto level_set : bead_levelsets)
                {
                    T height;
                    TV x = undeformed.template segment<dim>(i * dim);
                    bool inside = level_set(x, height);
                    if (inside)
                        undeformed[i * dim + 1] = rest_shape[i * dim + 1] + height;
                }
            });
        }
        else if (bead_type == None)
        {

        }
    }
    deformed = undeformed;
    solver.updateRestshape();
}

template<class T, int dim, class Solver>
void TopographyOptimization<T, dim, Solver>::addBeadLevelset(int type)
{
    //single circle
    if (type == 0)
    {
        T r = 0.4 * (solver.max_corner[0] - solver.min_corner[0]);
        TV center = 0.5 * (solver.max_corner + solver.min_corner);
        T dx = solver.dx;

        auto circleBead = [r, dx, center](const TV& x, T& height)
        {
            T dis = std::abs((x - center).norm() - r);
            if (dis < dx * 4.0)
            {
                height = dx;
                return true;
            }
            return false;  
        };
        bead_levelsets.push_back(circleBead);
    }
    // box exact https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
    else if (type == 1)
    {
        T rx = 0.4 * (solver.max_corner[0] - solver.min_corner[0]);
        T rz = 0.4 * (solver.max_corner[0] - solver.min_corner[0]);

        TV center = 0.5 * (solver.max_corner + solver.min_corner);
        center[1] = 0.0;
        // T dx = solver.dx;
        T dx = 0.01;
        auto boxBead = [rx, rz, dx, center](const TV& x, T& height)
        {
            TV p = x - center;
            p[1] = 0.0;
            TV d = p.array().abs() - TV(rx, 0, rz).array();
            TV first_term = d.array().max(TV::Zero().array());
            T dis = std::abs(first_term.norm() + std::min(std::max(d[0], d[2]), 0.0));

            if (dis < dx * 4.0)
            {
                height = dx;
                return true;
            }
            return false;  
        };
        bead_levelsets.push_back(boxBead);
    }
    // regular hexagon https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
    else if (type == 2)
    {
        T r = 0.4 * (solver.max_corner[0] - solver.min_corner[0]);
        TV center = 0.5 * (solver.max_corner + solver.min_corner);
        // T dx = solver.dx;
        T dx = 0.01;

        auto hexagonBead = [r, center, dx](const TV&x, T& height)
        {
            TV k(-0.866025404,0.5,0.577350269);
            TV p3d = ((x - center).array().abs());
            TV2 p(p3d[0], p3d[2]);
            TV2 kxy = TV2(k[0], k[1]);
            
            p -= 2.0 * std::min(p.dot(kxy), 0.0) * kxy;
            p -= TV2(std::clamp(p[0], -k[2]*r, k[2]*r), r);
            T dis = p.norm();

            if (dis < dx * 4.0)
            {
                height = dx;
                return true;
            }
            return false;  
        };
        bead_levelsets.push_back(hexagonBead);
    }
    // star 5 https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
    else if (type == 3)
    {
        T r = 0.4 * (solver.max_corner[0] - solver.min_corner[0]);
        TV center = 0.5 * (solver.max_corner + solver.min_corner);
        T dx = solver.dx;

        auto starBead = [r, center, dx](const TV&x, T& height)
        {
            TV2 rf(0.5, 0.5);
            TV p3d = x - center;
            TV2 p(p3d[0], p3d[2]);
            TV2 k1(0.809016994375, -0.587785252292);
            TV2 k2(-k1[0],k1[1]);
            p[0] = std::abs(p[0]);

            p -= 2.0*std::max(k1.dot(p),0.0)*k1;
            p -= 2.0*std::max(k2.dot(p),0.0)*k2;
            p[0] = std::abs(p[0]);
            p[1] -= r;
            
            TV2 ba = rf.array() * TV2(-k1[1],k1[0]).array() - TV2(0,1).array();

            T h = std::clamp( p.dot(ba)/ba.dot(ba), 0.0, r );

            T dis = (p-ba*h).norm();

            if (dis < dx * 4.0)
            {
                height = dx;
                return true;
            }
            return false;  
        };
        bead_levelsets.push_back(starBead);
    }
    else if (type == 4)
    {
        T r = 0.09 * (solver.max_corner[0] - solver.min_corner[0]);
        T dx = solver.dx;
        int n_row = 4, n_col = 4;
        T x_span = solver.max_corner[0] - solver.min_corner[0];
        T z_span = solver.max_corner[2] - solver.min_corner[2];

        T offset = 0.05 * (solver.max_corner[0] - solver.min_corner[0]);

        for (int i = 0; i < n_row; i++)
        {
            T offset_left;
            if (i == 0)
                offset_left = offset + r;
            else
                offset_left = offset + r +  i * (offset + 2 * r);
            for (int j = 0; j < n_col; j++)
            {
                T offset_bottom;
                if (j == 0)
                    offset_bottom = offset + r;
                else
                    offset_bottom = offset + r +  j * (offset + 2 * r);
                TV center = solver.min_corner + TV(offset_left, 0, offset_bottom);
                auto circleBead = [r, dx, center](const TV& x, T& height)
                {
                    T dis = std::abs((x - center).norm() - r);
                    if (dis < dx * 1.0)
                    {
                        height = dx;
                        return true;
                    }
                    return false;  
                };
                bead_levelsets.push_back(circleBead);
            }   
        }
    }
}

template<class T, int dim, class Solver>
void TopographyOptimization<T, dim, Solver>::initializeScene(BeadType type)
{
    // T dx = 0.003; //3mm
    // T dx = 0.01;
    // T dx = 0.05;
    T dx = 0.5;
    solver.dx = dx;

    // T dy = 0.5 * dx;
    T dy = dx;

    TV grid_spacing;
    if constexpr (dim == 3)
    {
        grid_spacing = TV(dx, dy, dx);
        TV min_corner = TV::Zero();
        TV max_corner = TV(0.5, dy, 0.5);
        
        
        TV middle_point = 0.5 * (min_corner + max_corner);

        solver.min_corner = min_corner;
        solver.max_corner = max_corner;
        
        std::vector<TV> nodal_position;
        

        if (solver.name == "HexFEM")
        {   
            for (T x = min_corner[0]; x < max_corner[0] + 0.1 * dx; x += dx)
            {
                for (T y = min_corner[1]; y < max_corner[1] + 0.1 * dy; y += dy)
                {
                    for (T z = min_corner[2]; z < max_corner[2] + 0.1 * dx; z += dx)
                    {
                        
                        TV vtx = TV(x, y, z);
                        // if (std::abs(x - (0.25 * std::cos(40.0 * M_PI * z)  + 0.25)) < 0.2)
                        //     vtx = TV(x, y + 0.5 * dy, z);
                        nodal_position.push_back(vtx);
                    }
                    
                }
            }
        }
        else if (solver.name == "ShellFEM")
        {
            
            for (T x = min_corner[0]; x < max_corner[0] + 0.1 * dx; x += dx)
                for (T z = min_corner[2]; z < max_corner[2] + 0.1 * dx; z += dx)
                {
                    TV vtx = TV(x, min_corner[1], z);
                    nodal_position.push_back(vtx );
                }
            
        }
        

        solver.createSceneFromNodes(min_corner, max_corner, dx, nodal_position);
        // solver.loadFromMesh("/home/yueli/Documents/ETH/WuKong/Projects/TopographyOptimization/Data/coarse_grid.obj");
        // solver.loadFromMesh("/home/yueli/Documents/ETH/WuKong/Projects/TopographyOptimization/Data/plane_tri.obj");
        std::cout << "Total area before adding beads: " << solver.computeTotalVolume() << std::endl;
        addBeads(type, min_corner, max_corner, grid_spacing);
        std::cout << "Total area after adding beads: " << solver.computeTotalVolume() << std::endl;
        // solver.fixAxisEnd(0);
        auto forceFunc = [&, dx](const TV& pos, TV& force)->bool
        {
            force = TV(0, -9.8 * std::pow(dx, dim), 0.0) / 8.0 * 8.0;
            // force = TV(9.8 * std::pow(dx, dim), 0, 0);
            // force *= 1000.0;
            // force = TV(0, -9.8, 0.0) * 0.01;
            // force *= 1e-5;
            return true;

            if (pos[0]>max_corner[0]-1e-4 && pos[1] < min_corner[1] + 1e-4)
            {
                force = TV(0, -1e-4, 0);
                return true;
            }
            force = TV::Zero();
            return false;
        };



        auto displaceFunc = [&, dx](const TV& pos, TV& delta)->bool
        {
            if (pos[0]>max_corner[0]-1e-4)// && pos[1] < min_corner[1] + 1e-4)
            {
                // delta = TV(-0.02, -0.05, 0);
                delta = TV(-0.1 * dx,  -0.3 * dx, 0);
                
                return true;
            }
            return false;
        };


        // solver.addDirichletLambda(displaceFunc);
        // solver.addNeumannLambda(forceFunc, solver.f);
        // std::cout << "done" << std::endl;
    }
}

template<class T, int dim, class Solver>
void TopographyOptimization<T, dim, Solver>::forward()
{
    timer.start();
    // solver.verbose = true;
    // solver.newton_tol = 1e-4;
    
    solver.max_newton_iter = 1000;
    solver.staticSolve();
    std::cout << "forward time: " <<  timer.elapsed_sec() << "s" << std::endl;
}

template<class T, int dim, class Solver>
void TopographyOptimization<T, dim, Solver>::inverseRestShape()
{
    solver.max_newton_iter = 1000;

    using Line = Vector<T, dim * 2>;

    Line line;
    line << 0.25, 0, 0.05, 0.25, 0, 0.2;

    VectorXT rest_shape = solver.undeformed;
    
    int n_design = line.rows();

    T bead_height = solver.dx; 
    T bead_width = solver.dx * 2.0;

    std::cout << "bead height " << bead_height << " bead width " << bead_width << std::endl; 

    auto updateDesignParameters = [&](const VectorXT& new_parameters)
    {
        solver.u.setZero();
        solver.undeformed = rest_shape;

        line = new_parameters;
        
        // line[1] = 0; line[4] = 0;
        // line[0] = std::max(line[0], solver.min_corner[0]);
        // line[0] = std::min(line[0], solver.max_corner[0]);

        // line[3] = std::max(line[3], solver.min_corner[0]);
        // line[3] = std::min(line[3], solver.max_corner[0]);

        // line[2] = std::max(line[2], solver.min_corner[2]);
        // line[2] = std::min(line[2], solver.max_corner[2]);

        // line[5] = std::max(line[5], solver.min_corner[2]);
        // line[5] = std::min(line[5], solver.max_corner[2]);
        
        // std::cout << "from " << line.template segment<dim>(0) << " " << line.template segment<dim>(dim) << std::endl;

        TV from = line.template segment<dim>(0);
        TV to = line.template segment<dim>(dim);

        // if ((from - to).norm() > 1e-4)        
        {
            tbb::parallel_for(0, solver.num_nodes, [&](int i)
            {
                TV X = undeformed.template segment<dim>(i * dim);
                TV from = line.template segment<dim>(0);
                TV to = line.template segment<dim>(dim);
                T dis = pointToLineDistanceXZ(from, to, X);

                if (std::abs(dis) > bead_width)
                    return;
                
                undeformed[i * dim + 1] = rest_shape[i * dim + 1] + bead_height * (1.0 - std::abs(dis) / bead_width);
            });
        }
        solver.deformed = solver.undeformed;
    };

    auto objective = [&](const VectorXT& param)
    {
        updateDesignParameters(param);
        solver.staticSolve();
        T psi = solver.computeElasticPotential(solver.u);
        return psi;
    };

    auto gradient = [&](const VectorXT& param)
    {
        updateDesignParameters(param);

        // dPsidp = dPsidu dudp

        // de/du = 0
        // d2e/dudp + d2e/du2 dudp = 0
        // df/dp = df/dX dXdp
        // 

        VectorXT dedp = VectorXT::Zero(deformed.rows());

        StiffnessMatrix H(deformed.rows(), deformed.rows());
        StiffnessMatrix dfdX(deformed.rows(), deformed.rows());

        solver.staticSolve();
        solver.buildSystemMatrix(solver.u, H);
        
        solver.computedfdX(solver.u, dfdX);
        
        VectorXT dedu(deformed.rows());
        solver.computeInternalForce(solver.u, dedu);
        dedu *= -1.0;

        Eigen::PardisoLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, 
            typename StiffnessMatrix::StorageIndex>> linear_solver;

        linear_solver.analyzePattern(H);
        linear_solver.factorize(H);
        if (linear_solver.info() == Eigen::NumericalIssue)
            std::cout << "Forward Hessian indefinite!!" << std::endl;

        Eigen::MatrixXd dXdp(deformed.rows(), n_design);
        dXdp.setZero();

        for (int i = 0; i < solver.num_nodes; i++)
        {
            TV X = undeformed.template segment<dim>(i * dim);
            TV from = line.template segment<dim>(0);
            TV to = line.template segment<dim>(dim);
            T dis = pointToLineDistanceXZ(from, to, X);

            if (std::abs(dis) > bead_width)
                continue;

            Vector<T, 6> dddp;
            pointToLineDistanceXZGradient(from, to, X, dddp);
            
            if (dis > 0)
                dddp *= -bead_height / bead_width;
            else
                dddp *= bead_height / bead_width;

            for (int j = 0; j < n_design; j++)
            {
                dXdp(i * dim + 1, j) += dddp[j];
            }
            // std::cout << "|dXdp|: " << dXdp.norm() << std::endl;
        }
    
        
        for (int i = 0; i < n_design; i++)
        {
            VectorXT dfdpi = -dfdX * dXdp.col(i);
            for (auto data : solver.dirichlet_data)
                dfdpi[data.first] = 0;
            // std::cout << "|dfdpi| : " << dfdpi.norm() << std::endl;
            VectorXT dudp_i = linear_solver.solve(dfdpi);

            // std::cout << "|dudp_i| : " << dudp_i.norm() << std::endl;
            dedp[i] += dedu.dot(dudp_i);
            // std::cout << dedp[i] << std::endl;

        }
        
        return dedp;
    };

    auto diffTestdXdp = [&]()
    {
        Eigen::MatrixXd dXdp(deformed.rows(), n_design);
        dXdp.setZero();
        updateDesignParameters(line);

        VectorXT rest = solver.undeformed;

        for (int i = 0; i < solver.num_nodes; i++)
        {
            TV X = undeformed.template segment<dim>(i * dim);
            TV from = line.template segment<dim>(0);
            TV to = line.template segment<dim>(dim);
            T dis = pointToLineDistanceXZ(from, to, X);

            if (std::abs(dis) > bead_width)
                continue;

            Vector<T, 6> dddp;
            pointToLineDistanceXZGradient(from, to, X, dddp);
            // std::cout << "node " << i << " " << dis << " dddp: " <<  dddp.transpose() << std::endl;
            
            if (dis > 0)
                dddp *= -bead_height / bead_width;
            else
                dddp *= bead_height / bead_width;
            
            // std::cout << "\tnode " << i << " " << dis << " dddp: " <<  dddp.transpose() << std::endl;

            for (int j = 0; j < n_design; j++)
            {
                dXdp(i * dim + 1, j) += dddp[j];
            }
            // std::cout << "|dXdp|: " << dXdp.norm() << std::endl;
        }

        T epsilon = 1e-7;

        for (int i = 0; i < n_design; i++)
        {
            line[i] += epsilon;
            updateDesignParameters(line);
            for (int j = 0; j < solver.num_nodes; j++)
            {
                int d = 1;
                {
                    T fd = (solver.undeformed[j * dim + d] - rest[j * dim + d]) / epsilon;
                    T analytical = dXdp(j * dim + d, i);
                    if (std::abs(fd) > 1e-8 && std::abs(analytical) > 1e-8)
                    {
                        TV x = rest_shape.template segment<dim>(j * dim);

                        std::cout << "node " << j << " at " << x.transpose()  << " | rest " << rest[j * dim + d] << " deformed " << solver.undeformed[j * dim + d] << std::endl;
                        std::cout << "FD " << fd << " dXdp: " << analytical << std::endl;
                    }
                }
            }
            line[i] -= epsilon;
        }
    };

    auto derivativeTest = [&]()
    {
        
        VectorXT dOdp = gradient(line);

        T epsilon = 1e-6;
        T E0 = objective(line);
        for (int i = 0; i < n_design; i++)
        {
            line[i] += epsilon;
            T E1 = objective(line);
            std::cout << (E1 - E0) / epsilon << " " << dOdp[i] << std::endl;
            std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
            line[i] -= epsilon;
        }
        
    };
    
    // passed
    // diffTestdXdp(); 
    
    // derivativeTest();
    // return;

    if (om == GD)
    {
        int gd_step_max = 2;
        int step = 0;
        while (true)
        {
            step++;
            if (step > gd_step_max)
                break;
            VectorXT dOdp = gradient(line);
            T g_norm = dOdp.norm();
            std::cout << "|g| " << g_norm << std::endl;
            if (dOdp.norm() < 1e-6)
                break;
            T E0 = objective(line);
            std::cout << "obj: " << E0 << std::endl;
            // std::cout << "from " << line.template segment<dim>(0) << " " << line.template segment<dim>(dim) << std::endl;
            T alpha = 1.0;
            int ls_cnt = 0;
            while (true)
            {
                VectorXT p_ls = line + alpha * dOdp;
                // std::cout << "from " << p_ls.template segment<dim>(0) << " " << p_ls.template segment<dim>(dim) << std::endl;
                T E1 = objective(p_ls);
                std::cout << "\t ls obj: " << E1 << std::endl;
                if (E1 < E0 || ls_cnt > 15)
                {
                    line = p_ls;
                    break;
                }
                alpha *= 0.5;
                ls_cnt ++;
            }
        }
    }
    


}
template class TopographyOptimization<double, 3, FEMSolver<double, 3>>;
template class TopographyOptimization<double, 3, ShellFEMSolver<double, 3>>;