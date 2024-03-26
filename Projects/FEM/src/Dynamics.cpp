#include "../include/FEMSolver.h"
#include <stdexcept>

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

template <int dim>
void FEMSolver<dim>::constructMassMatrix(std::vector<Entry>& entries, bool project_PD, bool add_hessian)
{
    std::vector<Entry> entries_local;

    if constexpr(dim ==2)
    {
        iterateElementSerial([&](const EleNodes& x_deformed, 
            const EleNodes& x_undeformed, const EleIdx& shabi, int tet_idx)
        {
        Matrix<T, 6, 6> element_mass_matrix;
        VectorXT m(36);
        VectorXT p(6);
        p << x_undeformed(0), x_undeformed(1), x_undeformed(2), x_undeformed(3),
        x_undeformed(4), x_undeformed(5);
        T t1 = (p[1] - p[5]) * p[2] + (-p[1] + p[3]) * p[4] + p[0] * (-p[3] + p[5]);
        t1 = sqrt(pow(t1, 0.2e1));
        T t2 = t1 / 0.12e2;
        t1 = t1 / 0.24e2;
        m[0] = t2;
        m[1] = 0;
        m[2] = t1;
        m[3] = 0;
        m[4] = t1;
        m[5] = 0;
        m[6] = 0;
        m[7] = t2;
        m[8] = 0;
        m[9] = t1;
        m[10] = 0;
        m[11] = t1;
        m[12] = t1;
        m[13] = 0;
        m[14] = t2;
        m[15] = 0;
        m[16] = t1;
        m[17] = 0;
        m[18] = 0;
        m[19] = t1;
        m[20] = 0;
        m[21] = t2;
        m[22] = 0;
        m[23] = t1;
        m[24] = t1;
        m[25] = 0;
        m[26] = t1;
        m[27] = 0;
        m[28] = t2;
        m[29] = 0;
        m[30] = 0;
        m[31] = t1;
        m[32] = 0;
        m[33] = t1;
        m[34] = 0;
        m[35] = t2;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                if(add_hessian)
                    element_mass_matrix(i, j) = rou*m[i * 6 + j]/pow(h,2);
                else
                    element_mass_matrix(i, j) = rou*m[i * 6 + j];
            }
        }
            
                
        if (project_block_PD)
            projectBlockPD<6>(element_mass_matrix);

        addHessianEntry<6>(entries_local, shabi, element_mass_matrix);
        });
    }

    if constexpr(dim == 3)
    {
        iterateElementSerial([&](const EleNodes& x_deformed, 
            const EleNodes& x_undeformed, const EleIdx& shabi, int tet_idx)
        {
        Matrix<T, 12, 12> element_mass_matrix;
        VectorXT m(144);
        VectorXT p(12);
        p << x_undeformed(0), x_undeformed(1), x_undeformed(2), x_undeformed(3),
        x_undeformed(4), x_undeformed(5),x_undeformed(6), x_undeformed(7), x_undeformed(8), x_undeformed(9),
        x_undeformed(10), x_undeformed(11);

        T t1 = -p[5] + p[8];
        T t2 = p[2] - p[8];
        T t3 = -p[2] + p[5];
        t1 = fabs((p[1] * p[5] - p[2] * p[4]) * p[6] + (-p[0] * p[5] + p[2] * p[3]) * p[7] + (p[0] * p[4] - p[1] * p[3]) * p[8] + (t1 * p[1] + t2 * p[4] + t3 * p[7]) * p[9] + (-t1 * p[0] - t2 * p[3] - t3 * p[6]) * p[10] + ((p[1] - p[7]) * p[3] + (-p[1] + p[4]) * p[6] + p[0] * (-p[4] + p[7])) * p[11]);
        t2 = t1 / 0.60e2;
        t1 = t1 / 0.120e3;

        //std::cout<<t1<<" "<<t2<<" "<<t3<<std::endl;
        m[0] = t2;
        m[1] = 0;
        m[2] = 0;
        m[3] = t1;
        m[4] = 0;
        m[5] = 0;
        m[6] = t1;
        m[7] = 0;
        m[8] = 0;
        m[9] = t1;
        m[10] = 0;
        m[11] = 0;
        m[12] = 0;
        m[13] = t2;
        m[14] = 0;
        m[15] = 0;
        m[16] = t1;
        m[17] = 0;
        m[18] = 0;
        m[19] = t1;
        m[20] = 0;
        m[21] = 0;
        m[22] = t1;
        m[23] = 0;
        m[24] = 0;
        m[25] = 0;
        m[26] = t2;
        m[27] = 0;
        m[28] = 0;
        m[29] = t1;
        m[30] = 0;
        m[31] = 0;
        m[32] = t1;
        m[33] = 0;
        m[34] = 0;
        m[35] = t1;
        m[36] = t1;
        m[37] = 0;
        m[38] = 0;
        m[39] = t2;
        m[40] = 0;
        m[41] = 0;
        m[42] = t1;
        m[43] = 0;
        m[44] = 0;
        m[45] = t1;
        m[46] = 0;
        m[47] = 0;
        m[48] = 0;
        m[49] = t1;
        m[50] = 0;
        m[51] = 0;
        m[52] = t2;
        m[53] = 0;
        m[54] = 0;
        m[55] = t1;
        m[56] = 0;
        m[57] = 0;
        m[58] = t1;
        m[59] = 0;
        m[60] = 0;
        m[61] = 0;
        m[62] = t1;
        m[63] = 0;
        m[64] = 0;
        m[65] = t2;
        m[66] = 0;
        m[67] = 0;
        m[68] = t1;
        m[69] = 0;
        m[70] = 0;
        m[71] = t1;
        m[72] = t1;
        m[73] = 0;
        m[74] = 0;
        m[75] = t1;
        m[76] = 0;
        m[77] = 0;
        m[78] = t2;
        m[79] = 0;
        m[80] = 0;
        m[81] = t1;
        m[82] = 0;
        m[83] = 0;
        m[84] = 0;
        m[85] = t1;
        m[86] = 0;
        m[87] = 0;
        m[88] = t1;
        m[89] = 0;
        m[90] = 0;
        m[91] = t2;
        m[92] = 0;
        m[93] = 0;
        m[94] = t1;
        m[95] = 0;
        m[96] = 0;
        m[97] = 0;
        m[98] = t1;
        m[99] = 0;
        m[100] = 0;
        m[101] = t1;
        m[102] = 0;
        m[103] = 0;
        m[104] = t2;
        m[105] = 0;
        m[106] = 0;
        m[107] = t1;
        m[108] = t1;
        m[109] = 0;
        m[110] = 0;
        m[111] = t1;
        m[112] = 0;
        m[113] = 0;
        m[114] = t1;
        m[115] = 0;
        m[116] = 0;
        m[117] = t2;
        m[118] = 0;
        m[119] = 0;
        m[120] = 0;
        m[121] = t1;
        m[122] = 0;
        m[123] = 0;
        m[124] = t1;
        m[125] = 0;
        m[126] = 0;
        m[127] = t1;
        m[128] = 0;
        m[129] = 0;
        m[130] = t2;
        m[131] = 0;
        m[132] = 0;
        m[133] = 0;
        m[134] = t1;
        m[135] = 0;
        m[136] = 0;
        m[137] = t1;
        m[138] = 0;
        m[139] = 0;
        m[140] = t1;
        m[141] = 0;
        m[142] = 0;
        m[143] = t2;

        for (int i = 0; i < 12; i++)
        {
            for (int j = 0; j < 12; j++)
            {
                if(add_hessian)
                    element_mass_matrix(i, j) = rou*m[i * 12 + j]/pow(h,2);
                else
                    element_mass_matrix(i, j) = rou*m[i * 12 + j];
            }
        }

        // Eigen::LLT<Eigen::MatrixXd> lltOfA(element_mass_matrix); // compute the Cholesky decomposition of A
        // if(lltOfA.info() == Eigen::NumericalIssue)
        // {
        //     throw std::runtime_error("Possibly non semi-positive definitie matrix!");
        // }    
            
                
        if (project_block_PD)
            projectBlockPD<12>(element_mass_matrix);

        //addHessianEntry<12>(entries_local, shabi, element_mass_matrix);

        });
    }

    int used_nodes = num_nodes;
    if(USE_NEW_FORMULATION) used_nodes+=additional_dof;
    for(int i=0; i<dim*num_nodes; ++i)
    {
        if(add_hessian)
            entries_local.push_back(Entry(i,i,rou/pow(h,2)));
        else
            entries_local.push_back(Entry(i,i,rou));
    }
    

    if(!add_hessian)
    {
        M.resize(dim*used_nodes,dim*used_nodes);
        M.setFromTriplets(entries_local.begin(), entries_local.end());
    }else
        entries.insert(entries.end(), entries_local.begin(), entries_local.end());
    
       
}

template <int dim>
VectorXT FEMSolver<dim>::get_a(VectorXT& x)
{
    VectorXT a = (x-x_prev)/(h*h)-v_prev/h;
    
    if(USE_NEW_FORMULATION)
    {
        for(int i=0; i<dim*additional_dof; ++i)
        {
            a(i+dim*num_nodes) = 0;
        }
    }

    return a;
}

template <int dim>
bool FEMSolver<dim>::dynamicSolve()
{
    int num_step = 0;

    //Initialization
    std::vector<Entry> entries;
    constructMassMatrix(entries);
    x_prev = undeformed;
    v_prev = x_prev;
    v_prev.setZero();

    for(int i=0; i<simulation_time/h; ++i)
    {
        staticSolve();

        v_prev = (deformed-x_prev)/h;
        x_prev = deformed;
        num_step++;
    }

    


}

template <int dim>
void FEMSolver<dim>::addUnilateralQubicPenaltyEnergy(T w, T& energy)
{
    std::cout<<xz.transpose()<<std::endl;
    VectorXT energies(num_nodes); energies.setZero();
    tbb::parallel_for(8220, num_nodes, [&](int i){
        TV xi = deformed.segment<dim>(i * dim);
        VectorXT ds(4);
        ds.setZero();
        T d = xi(0)-xz(0);
        // ds(1) = std::min(-xi[0]+xz(1),0.);
        // ds(2) = std::min(xi[2]-xz(2),0.);
        // ds(3) = std::min(-xi[2]+xz(3),0.);
        // T d = xi[1] - y_bar;
        // if(d > 0)
        //     energies[i] += w * std::pow(d, 3);
        if(d>0)
            energies[i] += w * std::pow(d, 3);
    });
    energy += energies.sum();
}
template <int dim>
void FEMSolver<dim>::addUnilateralQubicPenaltyForceEntries(T w, VectorXT& residuals)
{   
    // if(xz.size() == 0){
    //     xz.resize(4);
    //     xz.setZero();
    // }
    VectorXT residual_bk = residuals;
    tbb::parallel_for(8220, num_nodes, [&](int i){
        TV xi = deformed.segment<dim>(i * dim);
        VectorXT ds(4);
        ds.setZero();
        //std::cout<<xz(0)<<std::endl;
        ds(0) = std::min(xz[0]-xi(0),0.);
        T d = xi(0)-xz(0);
        // ds(1) = std::min(-xi[0]+xz(1),0.);
        // ds(2) = std::min(xi[2]-xz(2),0.);
        // ds(3) = std::min(-xi[2]+xz(3),0.);
        // T d = xi[1] - y_bar;
        // if(d > 0)
        //     residuals[i * dim + 1] -= w * 3.0 * std::pow(d, 2);
        if(d>0)
            residuals[i * dim + 0] -= w * 3.0 * std::pow(d, 2.);
        // residuals[i * dim + 0] -= -w * 3.0 * std::pow(ds(1), 2.);
        // residuals[i * dim + 2] -= w * 3.0 * std::pow(ds(2), 2.);
        // residuals[i * dim + 2] -= -w * 3.0 * std::pow(ds(3), 2.);
    });

    std::cout<<"Penalty Force: "<<(residuals-residual_bk).norm()<<std::endl;
}
template <int dim>
void FEMSolver<dim>::addUnilateralQubicPenaltyHessianEntries(T w, std::vector<Entry>& entries)
{
    for (int i = 8220; i < num_nodes; i++)
    {
        TV xi = deformed.segment<dim>(i * dim);
        VectorXT ds(4);
        ds.setZero();
        ds(0) = std::min(xz[0]-xi(0),0.);
        T d = xi(0)-xz(0);
        // ds(1) = std::min(-xi[0]+xz(1),0.);
        // ds(2) = std::min(xi[2]-xz(2),0.);
        // ds(3) = std::min(-xi[2]+xz(3),0.);
        // T d = xi[1] - y_bar;
        // if(d > 0)
        //     entries.push_back(Entry(i * dim + 1, i * dim + 1, w * 6.0 * d));
        if(d>0)
            entries.push_back(Entry(i * dim + 0, i * dim + 0, w * 6.0 * d));
        // entries.push_back(Entry(i * dim + 0, i * dim + 0, w * 6.0 * ds(1)));
        // entries.push_back(Entry(i * dim + 2, i * dim + 2, w * 6.0 * ds(2)));
        // entries.push_back(Entry(i * dim + 2, i * dim + 2, w * 6.0 * ds(3)));
    }
}

template class FEMSolver<2>;
template class FEMSolver<3>;