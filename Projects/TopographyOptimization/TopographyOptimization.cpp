#include "TopographyOptimization.h"


template<class T, int dim, class Solver>
void TopographyOptimization<T, dim, Solver>::addBeads(BeadType bead_type,
    const TV& min_corner, const TV& max_corner, const TV& dx)
{
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
                for (int i = 0; i < scene_range[2]; i++)
                {
                    undeformed[i * dim + 1] -= dx[0];
                    undeformed[(scene_range[0] - 1) * scene_range[2] * dim + i * dim + 1] -= dx[0];
                }
                for (int i = 1; i < scene_range[0] - 1; i++)
                {
                    // undeformed[i * dim + 1] -= dx[0];
                    undeformed[i * scene_range[2] * dim + i * dim + 1] -= dx[0];
                }
            }
            
        }
    }
    deformed = undeformed;
}

template<class T, int dim, class Solver>
void TopographyOptimization<T, dim, Solver>::initializeScene(int type)
{
    // T dx = 0.003; //3mm
    T dx = 0.01;
    // T dx = 0.05;
    // T dx = 0.5;
    
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

        addBeads(CurveBD, min_corner, max_corner, grid_spacing);
        
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
    using Line = Vector<T, dim * 2>;
    

    VectorXT p = solver.undeformed;
    int n_design = p.rows();

    auto updateDesignParameters = [&](const VectorXT& new_parameters)
    {
        solver.undeformed = new_parameters;
        solver.deformed = solver.undeformed;
    };

    auto objective = [&](const VectorXT& param)
    {
        updateDesignParameters(param);
        solver.staticSolve();
        T psi = solver.computeTotalEnergy(solver.u);
        return psi;
    };

    auto gradient = [&](const VectorXT& param)
    {
        return param;
    };

    // if (om == GD)
    // {
    //     while (true)
    //     {
    //         VectorXT dOdp = gradient;
    //         if (dOdp.norm() < 1e-6)
    //             break;
    //         T E0 = objective(p);
    //         T alpha = 1.0;
    //         int ls_cnt = 0;
    //         while (true)
    //         {
    //             VectorXT p_ls = p + alpha * dOdp;
    //             T E1 = objective(p_ls);
    //             if (E1 < E0 || ls_cnt > 15)
    //             {
    //                 p = p_ls;
    //                 break;
    //             }
    //             alpha *= 0.5;
    //             ls_cnt ++;
    //         }
    //     }
    // }
    


}
template class TopographyOptimization<double, 3, FEMSolver<double, 3>>;
template class TopographyOptimization<double, 3, ShellFEMSolver<double, 3>>;