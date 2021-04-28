#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addPBCK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{
    iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
        TV xi = q_temp.col(node_i).template segment<dim>(0);
        TV xj = q_temp.col(node_j).template segment<dim>(0);

        T dij = (xj - xi).dot(strain_dir);
        TM Hessian = strain_dir * strain_dir.transpose();
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                entry_K.push_back(Eigen::Triplet<T>(node_i * dof + i, node_i * dof + j, kr * Hessian(i, j)));
                entry_K.push_back(Eigen::Triplet<T>(node_i * dof + i, node_j * dof + j, -kr * Hessian(i, j)));
                entry_K.push_back(Eigen::Triplet<T>(node_j * dof + i, node_i * dof + j, -kr * Hessian(i, j)));
                entry_K.push_back(Eigen::Triplet<T>(node_j * dof + i, node_j * dof + j, kr * Hessian(i, j)));
            }
        }
    });
    if constexpr (dim == 2)
    {
        std::vector<TV> data;
        std::vector<int> nodes(4);
        buildMapleRotationPenaltyData(q_temp, data, nodes);
        T J[8][8];
        memset(J, 0, sizeof(J));
        #include "Maple/RotationPenaltyJ.mcg"
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < dim; k++)
                    for (int l = 0; l < dim; l++)
                        entry_K.push_back(Eigen::Triplet<T>(nodes[i]*dof + k, nodes[j] * dof + l, -J[i*dim + k][j*dim+l]));

        // iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
        //     TV xi = q_temp.col(node_i).template segment<dim>(0);
        //     TV xj = q_temp.col(node_j).template segment<dim>(0);

        //     Vector<T, 12> dedx;
        //     nodes.push_back(node_i); nodes.push_back(node_j);
        //     data.push_back(xi); data.push_back(xj);

        //     T J[12][12];
        //     memset(J, 0, sizeof(J));
        //     #include "Maple/UniAxialStrainJ.mcg"
        //     for (int i = 0; i < 6; i++)
        //         for (int j = 0; j < 6; j++)
        //             for (int k = 0; k < dim; k++)
        //                 for (int l = 0; l < dim; l++)
        //                     entry_K.push_back(Eigen::Triplet<T>(nodes[i]*dof + k, nodes[j] * dof + l, -kr * J[i*dim + k][j*dim+l]));
        //     data.pop_back(); data.pop_back();
        //     nodes.pop_back(); nodes.pop_back();
        // });
        // return;
        iteratePBCReferencePairs([&](int yarn_type, int node_i, int node_j){
            int ref_i = pbc_ref_unique[yarn_type](0);
            int ref_j = pbc_ref_unique[yarn_type](1);

            TVDOF qi = q_temp.col(node_i);
            TVDOF qj = q_temp.col(node_j);
            TVDOF qi_ref = q_temp.col(ref_i);
            TVDOF qj_ref = q_temp.col(ref_j);

            if (ref_i == node_i && ref_j == node_j)
                return;

            std::vector<int> nodes = {node_i, node_j, ref_i, ref_j};
            std::vector<T> sign_J = {-1, 1, 1, -1};
            std::vector<T> sign_F = {1, -1, -1, 1};

            for(int k = 0; k < 4; k++)
                for(int l = 0; l < 4; l++)
                    for(int i = 0; i < dim + 2; i++)
                        for(int j = 0; j < dim + 2; j++)
                            entry_K.push_back(Eigen::Triplet<T>(nodes[k]*dof + i, nodes[l] * dof + j, -k_pbc *sign_F[k]*sign_J[l]));

        });
    }
    
}


template<class T, int dim>
void EoLRodSim<T, dim>::addPBCForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    
    iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
        TV xi = q_temp.col(node_i).template segment<dim>(0);
        TV xj = q_temp.col(node_j).template segment<dim>(0);

        T dij = (xj - xi).dot(strain_dir);
        
        residual.col(node_i).template segment<dim>(0) += kr * strain_dir * (dij - Dij);
        residual.col(node_j).template segment<dim>(0) += -kr * strain_dir * (dij - Dij);
    });

    std::vector<TV> data;
    std::vector<int> nodes(4);
    buildMapleRotationPenaltyData(q_temp, data, nodes);
    Vector<T, 8> dedx;
    #include "Maple/RotationPenaltyF.mcg"
    for (int i = 0; i < 4; i++)
        residual.col(nodes[i]).template segment<dim>(0) += dedx.template segment<dim>(i*dim);

    // iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
    //     TV xi = q_temp.col(node_i).template segment<dim>(0);
    //     TV xj = q_temp.col(node_j).template segment<dim>(0);

    //     Vector<T, 12> dedx;
    //     dedx.setZero();
    //     nodes.push_back(node_i);
    //     nodes.push_back(node_j);
    //     data.push_back(xi); data.push_back(xj);
    //     #include "Maple/UniAxialStrainF.mcg"
    //     for (int i = 0; i < 6; i++)
    //         residual.col(nodes[i]).template segment<dim>(0) += kr * dedx.template segment<dim>(i*dim);
    //     data.pop_back(); data.pop_back();
    //     nodes.pop_back(); nodes.pop_back();
    // });
    // return;
    iteratePBCReferencePairs([&](int yarn_type, int node_i, int node_j){
        int ref_i = pbc_ref_unique[yarn_type](0);
        int ref_j = pbc_ref_unique[yarn_type](1);

        TVDOF qi = q_temp.col(node_i);
        TVDOF qj = q_temp.col(node_j);
        TVDOF qi_ref = q_temp.col(ref_i);
        TVDOF qj_ref = q_temp.col(ref_j);

        if (ref_i == node_i && ref_j == node_j)
            return;
            
        TVDOF pair_dis_vec = qj - qi - (qj_ref - qi_ref);
        
        residual.col(node_i) += k_pbc *pair_dis_vec;
        residual.col(node_j) += -k_pbc *pair_dis_vec;

        residual.col(ref_i) += -k_pbc  *pair_dis_vec;
        residual.col(ref_j) += k_pbc *pair_dis_vec;
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::buildMapleRotationPenaltyData(Eigen::Ref<const DOFStack> q_temp, 
    std::vector<TV>& data, std::vector<int>& nodes)
{
    IV2 ref0 = pbc_ref_unique[0];
    IV2 ref1 = pbc_ref_unique[1];

    data.resize(8);
    data[0] = q_temp.col(ref0[0]).template segment<dim>(0);
    data[1] = q_temp.col(ref0[1]).template segment<dim  >(0);
    data[2] = q_temp.col(ref1[0]).template segment<dim>(0);
    data[3] = q_temp.col(ref1[1]).template segment<dim>(0);
    data[4] = q0.col(ref0[0]).template segment<dim>(0);
    data[5] = q0.col(ref0[1]).template segment<dim>(0);
    data[6] = q0.col(ref1[0]).template segment<dim>(0);
    data[7] = q0.col(ref1[1]).template segment<dim>(0);

    nodes[0] = ref0[0];
    nodes[1] = ref0[1];
    nodes[2] = ref1[0];
    nodes[3] = ref1[1];
}


template<class T, int dim>
T EoLRodSim<T, dim>::addPBCEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    T energy_pbc = 0.0;
    iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
        TV xi = q_temp.col(node_i).template segment<dim>(0);
        TV xj = q_temp.col(node_j).template segment<dim>(0);
        T dij = (xj - xi).dot(strain_dir);
        energy_pbc += 0.5 * kr * (dij - Dij) * (dij - Dij);
    });
    std::vector<TV> data;
    std::vector<int> nodes(4);
    buildMapleRotationPenaltyData(q_temp, data, nodes);
    T V[1];
    #include "Maple/RotationPenaltyV.mcg"
    energy_pbc += V[0];

    // iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
    //     TV xi = q_temp.col(node_i).template segment<dim>(0);
    //     TV xj = q_temp.col(node_j).template segment<dim>(0);
    //     data.push_back(xi); data.push_back(xj);
    //     #include "Maple/UniAxialStrainV.mcg"
    //     energy_pbc+= kr * V[0];
    //     data.pop_back(); data.pop_back();
    // });

    
// return energy_pbc;
    iteratePBCReferencePairs([&](int yarn_type, int node_i, int node_j){
        
        int ref_i = pbc_ref_unique[yarn_type](0);
        int ref_j = pbc_ref_unique[yarn_type](1);

        TVDOF qi = q_temp.col(node_i);
        TVDOF qj = q_temp.col(node_j);
        TVDOF qi_ref = q_temp.col(ref_i);
        TVDOF qj_ref = q_temp.col(ref_j);

        if (ref_i == node_i && ref_j == node_j)
            return;

        TVDOF pair_dis_vec = qj - qi - (qj_ref - qi_ref);
        energy_pbc += 0.5  *k_pbc * pair_dis_vec.dot(pair_dis_vec);
    });

    return energy_pbc;
}



template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;