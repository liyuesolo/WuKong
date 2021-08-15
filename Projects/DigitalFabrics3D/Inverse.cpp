#include "EoLRodSim.h"

#include "LBFGS.h"

using namespace LBFGSpp;

template<class T, int dim>
class LBFGSWrapper
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using TV = Vector<T, dim>;
    using Offset = Vector<int, dim + 1>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    
private:
    int n;
    EoLRodSim<T, dim>& sim;
    
    int n0 = 29, n1 = 30;
    TV x0, x1;
    TV x0_target = TV(0.0377535,   0.0789261,  0);
    TV x1_target = TV(0.0479829,    0.0789278, 0);

    VectorXT computedOdx()
    {
        sim.Rods[5]->x(n0, x0);
        sim.Rods[6]->x(n1, x1);
        VectorXT dOdx = VectorXT::Zero(sim.deformed_states.rows());
        Offset offset0, offset1;
        sim.Rods[5]->getEntry(n0, offset0);
        sim.Rods[6]->getEntry(n1, offset1);
        dOdx.template segment<dim>(offset0[0]) = x0 - x0_target;
        dOdx.template segment<dim>(offset1[0]) = x1 - x1_target;
        return sim.W.transpose() * dOdx;
    }

    void updateDesignParameter(const VectorXT& new_p)
    {
        int cnt = 0;
        for (auto& crossing : sim.rod_crossings)
        {
            if (crossing->is_fixed)
            {
                cnt += crossing->rods_involved.size() * 2;
                continue;
            }
            for (int i = 0; i < crossing->rods_involved.size(); i++)
            {
                // if (i == 0)
                //     crossing->sliding_ranges[i].setZero();
                // else
                // {
                //     crossing->sliding_ranges[i] = new_p.template segment<2>(cnt);
                // }
                crossing->sliding_ranges[i] = new_p.template segment<2>(cnt);
                crossing->sliding_ranges[i][0] = std::max(0.0, crossing->sliding_ranges[i][0]);
                crossing->sliding_ranges[i][1] = std::max(0.0, crossing->sliding_ranges[i][1]);
                // std::cout << crossing->sliding_ranges[i].transpose() << std::endl;
                cnt += 2;
            }
        }
    }
    
public:
    LBFGSWrapper(int n_, EoLRodSim<T, dim>& _sim) : n(n_), sim(_sim) {}
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
    {
        updateDesignParameter(x);

        sim.resetScene();
        VectorXT dq(sim.W.cols()); dq.setZero();
        sim.forward(dq);
        sim.Rods[5]->x(n0, x0);
        sim.Rods[6]->x(n1, x1);

        int nx = sim.deformed_states.rows();

        StiffnessMatrix H;
        sim.buildSystemDoFMatrix(dq, H);
        VectorXT dOdx = computedOdx();
        // std::cout << "|dOdx| " << dOdx.norm() << std::endl;
        Eigen::SimplicialLLT<StiffnessMatrix> solver;
        solver.compute(H);

        VectorXT lambda = solver.solve(dOdx);
        // std::cout << "|lambda| " << lambda.norm() << std::endl;
        VectorXT dOdp(n); dOdp.setZero();

        VectorXT dfdp_full(n * nx);
        
        sim.parallelContactdfdp(dfdp_full);
        dfdp_full *= -1.0;
        //de/dx = -f
        
        // std::cout << "|dfdp_full| " << dfdp_full.norm() << std::endl;
        
        for (int i = 0; i < n; i++)
        {
            VectorXT dfdp = sim.W.transpose() * dfdp_full.segment(i * nx, nx);
            
            grad[i] = -lambda.dot(dfdp);
        }
        T obj = 0.5 * ((x0 - x0_target).dot((x0 - x0_target)) + (x1 - x1_target).dot((x1 - x1_target))); 
        std::cout << "E " << obj << std::endl;
        return obj;
    }
};


template<class T, int dim>
void EoLRodSim<T, dim>::resetScene()
{ 
    deformed_states = rest_states; 
    for (auto& rod : Rods)
    {
        rod->reference_twist.setZero();
        rod->reference_angles.setZero();
    }
    for (auto& crossing : rod_crossings)
    {
        crossing->omega.setZero();
        crossing->rotation_accumulated.setIdentity();
    }

    for (auto& rod : Rods)
    {
        rod->setupBishopFrame();
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::inverse()
{
    if constexpr (dim == 3)
    {
        
        int n_design = 0;
        std::vector<bool> flag;
        for (auto& crossing : rod_crossings)
        {
            for (int rod_idx : crossing->rods_involved)
            {
                n_design += 2;
                if (crossing->is_fixed)
                {
                    flag.push_back(false); flag.push_back(false);
                }
                else
                {
                    flag.push_back(true); flag.push_back(true);
                }
            }
            
        }
        std::cout << "#design parameters: " << n_design << std::endl;
        VectorXT p(n_design);
        p.setZero();
        
        int loop_cnt = 0;
        for (auto& crossing : rod_crossings)
        {
            for (int i = 0; i < crossing->rods_involved.size(); i++)
            {
                p[loop_cnt++] = crossing->sliding_ranges[i][0];
                p[loop_cnt++] = crossing->sliding_ranges[i][1];
                
            }
            
        }
        // std::cout << p.transpose() << std::endl;
        
        int n0 = 29, n1 = 30;
        TV x0, x1;
        TV x0_target = TV(0.0377535,   0.0789261,  0);
        TV x1_target = TV(0.0479829,    0.0789278, 0);

        

            
        auto objective = [&]()
        {
            resetScene();
            VectorXT dq(W.cols()); dq.setZero();
            forward(dq);
            Rods[5]->x(n0, x0);
            Rods[6]->x(n1, x1);
            return 0.5 * ((x0 - x0_target).dot((x0 - x0_target)) + (x1 - x1_target).dot((x1 - x1_target)));
        };

        auto computedOdx = [&]()
        {
            
            Rods[5]->x(n0, x0);
            Rods[6]->x(n1, x1);
            VectorXT dOdx = VectorXT::Zero(deformed_states.rows());
            Offset offset0, offset1;
            Rods[5]->getEntry(n0, offset0);
            Rods[6]->getEntry(n1, offset1);
            dOdx.template segment<dim>(offset0[0]) = x0 - x0_target;
            dOdx.template segment<dim>(offset1[0]) = x1 - x1_target;
            return W.transpose() * dOdx;
        };

        

        auto gradient = [&]()
        {
            resetScene();

            VectorXT dq(W.cols()); dq.setZero();
            forward(dq);
            
            int nx = deformed_states.rows();

            StiffnessMatrix H;
            buildSystemDoFMatrix(dq, H);
            VectorXT dOdx = computedOdx();
            // std::cout << "|dOdx| " << dOdx.norm() << std::endl;
            Eigen::SimplicialLLT<StiffnessMatrix> solver;
            solver.compute(H);

            VectorXT lambda = solver.solve(dOdx);
            // std::cout << "|lambda| " << lambda.norm() << std::endl;
            VectorXT dOdp(n_design); dOdp.setZero();

            VectorXT dfdp_full(n_design * nx);
            
            parallelContactdfdp(dfdp_full);
            dfdp_full *= -1.0;
            //de/dx = -f
            
            // std::cout << "|dfdp_full| " << dfdp_full.norm() << std::endl;
            
            for (int i = 0; i < n_design; i++)
            {
                VectorXT dfdp = W.transpose() * dfdp_full.segment(i * nx, nx);
                
                dOdp[i] += -lambda.dot(dfdp);
            }
            // std::getchar();
            return dOdp;

        };

        auto updateDesignParameter = [&](const VectorXT& new_p)
        {
            int cnt = 0;
            for (auto& crossing : rod_crossings)
            {
                if (crossing->is_fixed)
                {
                    cnt += crossing->rods_involved.size() * 2;
                    continue;
                }
                for (int i = 0; i < crossing->rods_involved.size(); i++)
                {
                    if (i == 0)
                        crossing->sliding_ranges[i].setZero();
                    else
                    {
                        crossing->sliding_ranges[i] = new_p.template segment<2>(cnt);
                    }
                    // crossing->sliding_ranges[i] = new_p.template segment<2>(cnt);
                    crossing->sliding_ranges[i][0] = std::max(0.0, crossing->sliding_ranges[i][0]);
                    crossing->sliding_ranges[i][1] = std::max(0.0, crossing->sliding_ranges[i][1]);
                    // std::cout << crossing->sliding_ranges[i].transpose() << std::endl;
                    cnt += 2;
                }
            }
        };

        updateDesignParameter(p);

        auto diffTest = [&]()
        {
            T epsilon = 1e-8;
            VectorXT g = gradient();
            
            T E0 = objective();
            for (int i = 0; i < n_design; i++)
            {
                if (!flag[i])
                    continue;
                // std::cout << i << std::endl;
                p[i] += epsilon;
                updateDesignParameter(p);
                T E1 = objective();
                
                std::cout << "FD " << (E1 - E0) / epsilon << " symbolic " << g[i] << std::endl;
                // // std::getchar();
                p[i] -= epsilon;
                // std::cout << E1 << std::endl;
            }
        };

        // diffTest();

        LBFGSParam<T> param;
        param.epsilon = 1e-4;
        param.max_iterations = 100;

        LBFGSSolver<T, LineSearchBracketing> solver(param);
        LBFGSWrapper<T, dim> fun(n_design, *this);
        
        
        T fx;
        VectorXT x = VectorXT::Constant(n_design, 0.0);
        int niter = solver.minimize(fun, x, fx);

        std::cout << niter << " iterations" << std::endl;
        std::cout << "x = \n" << x.transpose() << std::endl;
        std::cout << "f(x) = " << fx << std::endl;

        // for (int iter = 0; iter < 1000; iter++)
        // {
        //     VectorXT g = gradient();
        //     std::cout << "|g| " << g.norm() << std::endl;
            
        //     if (g.norm() < 1e-6)
        //     {
        //         break;
        //     }
        //     T E0 = objective();
        //     std::cout << "E " << E0 << std::endl;
        //     T alpha = 1.0;
        //     while (true)
        //     {
        //         VectorXT p_ls = p - alpha * g;
        //         updateDesignParameter(p_ls);
        //         // std::getchar();
        //         T E1 = objective();
        //         // std::getchar();
        //         if (E1 < E0)
        //         {
        //             p = p_ls;
        //             break;
        //         }
        //         else
        //         {
        //             alpha *= 0.5;
        //         }
        //     }
            

        // }
        std::cout << p.transpose() << std::endl;
    }
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;