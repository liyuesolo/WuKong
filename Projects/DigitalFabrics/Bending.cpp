#include "EoLRodSim.h"
using std::abs;
template<class T, int dim>
void EoLRodSim<T, dim>::toMapleNodesVector(std::vector<Vector<T, dim + 1>>& x, Eigen::Ref<const DOFStack> q_temp,
    std::vector<int>& nodes, int yarn_type)
{
    int cnt = 0;
    for (int node : nodes)
    {
        Vector<T, dim + 1> xu;
        xu.template segment<dim>(0) = q_temp.col(node).template segment<dim>(0);
        xu[dim] = q_temp(dim + yarn_type, node);
        x[cnt++] = xu;
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::entryHelperBending(Eigen::Ref<const DOFStack> q_temp, 
    std::vector<Eigen::Triplet<T>>& entry_K, int n0, int n1, int n2, int uv_offset)
{
    std::vector<Vector<T, dim + 1>> x(3);
    std::vector<int> nodes = {n0, n1, n2};
    toMapleNodesVector(x, q_temp, nodes, uv_offset);
    T J[9][9];
    memset(J, 0, sizeof(J));
    #include "Maple/YarnBendJ.mcg"
    for(int k = 0; k < nodes.size(); k++)
        for(int l = 0; l < nodes.size(); l++)
            for(int i = 0; i < dim + 1; i++)
                for (int j = 0; j < dim + 1; j++)
                    {
                        int u_offset = i == dim ? dim + uv_offset : i;
                        int v_offset = j == dim ? dim + uv_offset : j;
                        entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + u_offset, nodes[l] * dof + v_offset, -J[k*(dim + 1) + i][l * (dim + 1) + j]));
                    }
    
}


template<class T, int dim>
void EoLRodSim<T, dim>::addBendingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{

    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
                entryHelperBending(q_temp, entry_K, middle, right, left, 0);
        if (top != -1 && bottom != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
                entryHelperBending(q_temp, entry_K, middle, top, bottom, 1);
    });
    if (!subdivide)
        iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
            int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
            std::vector<Vector<T, dim + 1>> x(nodes.size());
            toMapleNodesVector(x, q_temp, nodes, yarn_type);
            T J[15][15];
            memset(J, 0, sizeof(J));
            #include "Maple/YarnBendPBCJ.mcg"
            for(int k = 0; k < nodes.size(); k++)
                for(int l = 0; l < nodes.size(); l++)
                    for(int i = 0; i < dim + 1; i++)
                        for (int j = 0; j < dim + 1; j++)
                            {
                                int u_offset = i == dim ? dim + yarn_type : i;
                                int v_offset = j == dim ? dim + yarn_type : j;
        
                                entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + u_offset, nodes[l] * dof + v_offset, -J[k*(dim + 1) + i][l * (dim + 1) + j]));
                            }
        });    
    else
        iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
            std::vector<Vector<T, dim + 1>> x(nodes.size());
            toMapleNodesVector(x, q_temp, nodes, yarn_type);
            T J[12][12];
            memset(J, 0, sizeof(J));
            #include "Maple/YarnBendPBCSDJ.mcg"
            int cnt = 0;
            for(int k = 0; k < nodes.size(); k++)
                for(int l = 0; l < nodes.size(); l++)
                    for(int i = 0; i < dim + 1; i++)
                        for (int j = 0; j < dim + 1; j++)
                            {
                                int u_offset = i == dim ? dim + yarn_type : i;
                                int v_offset = j == dim ? dim + yarn_type : j;
        
                                entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + u_offset, nodes[l] * dof + v_offset, -J[k*(dim + 1) + i][l * (dim + 1) + j]));
                            }
        });
}

template<class T, int dim>
void EoLRodSim<T, dim>::addBendingForceSingleDirection(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual,
        int n0, int n1, int n2, int uv_offset)
{
    // if(n0 == 4)
        // cout4Nodes(n0, n1, n2, uv_offset);
    std::vector<Vector<T, dim + 1>> x(3);
    std::vector<int> nodes = {n0, n1, n2};
    toMapleNodesVector(x, q_temp, nodes, uv_offset);
    Vector<T, 9> F;
    F.setZero();
    #include "Maple/YarnBendF.mcg"
    int cnt = 0;
    for (int node : nodes)
    {
        residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*(dim+1));
        residual(dim + uv_offset, node) += F[cnt*(dim+1)+dim];
        cnt++;
    } 
} 

template<class T, int dim>
void EoLRodSim<T, dim>::addBendingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{

    DOFStack residual_cp = residual;
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
                addBendingForceSingleDirection(q_temp, residual, middle, right, left, 0);
        if (top != -1 && bottom != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
                addBendingForceSingleDirection(q_temp, residual, middle, top, bottom, 1);
    });   
    if (!subdivide)
        iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
            int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
            std::vector<Vector<T, dim + 1>> x(nodes.size());
            toMapleNodesVector(x, q_temp, nodes, yarn_type);
            Vector<T, 15> F;
            F.setZero();
            #include "Maple/YarnBendPBCF.mcg"
            int cnt = 0;
            for (int node : nodes)
            {
                residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*(dim+1));
                residual(dim + yarn_type, node) += F[cnt*(dim+1)+dim];
                cnt++;
            }
        });
    else
        iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
            std::vector<Vector<T, dim + 1>> x(nodes.size());
            toMapleNodesVector(x, q_temp, nodes, yarn_type);
            Vector<T, 12> F;
            F.setZero();
            #include "Maple/YarnBendPBCSDF.mcg"
            int cnt = 0;
            for (int node : nodes)
            {
                residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*(dim+1));
                residual(dim + yarn_type, node) += F[cnt*(dim+1)+dim];
                cnt++;
            }
        });
    // std::cout << "bending force " << (residual - residual_cp).transpose() << std::endl;
    if (print_force_mag)
        std::cout << "bending force " << (residual - residual_cp).norm() << std::endl;
    // std::exit(0);
}

template<class T, int dim>
T EoLRodSim<T, dim>::bendingEnergySingleDirection(Eigen::Ref<const DOFStack> q_temp, int n0, int n1, int n2, int uv_offset)
{
    // cout4Nodes(n0, n1, n2, uv_offset);
    // std::cout << "q2" <<  q_temp.col(n2).transpose() <<std::endl;
    // std::cout << "q0" <<  q_temp.col(n0).transpose() <<std::endl;
    // std::cout << "q1" <<  q_temp.col(n1).transpose() <<std::endl;
    std::vector<Vector<T, dim + 1>> x(3);
    std::vector<int> nodes = {n0, n1, n2};
    toMapleNodesVector(x, q_temp, nodes, uv_offset);
    T V[1];
    #include "Maple/YarnBendV.mcg"
    // std::cout << V[0] << std::endl;
    // std::getchar();
    return V[0];
}

template<class T, int dim>
T EoLRodSim<T, dim>::addBendingEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    T energy = 0.0;
    
    VectorXT crossing_energy(n_nodes);
    crossing_energy.setZero();
     
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
            if((!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left]) )
                crossing_energy[middle] += bendingEnergySingleDirection(q_temp, middle, right, left, 0);
        if (top != -1 && bottom != -1)
            if((!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom]) )
                crossing_energy[middle] += bendingEnergySingleDirection(q_temp, middle, top, bottom, 1);
    });
    energy += crossing_energy.sum();
    
    if (!subdivide)
        iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
            int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
            std::vector<Vector<T, dim + 1>> x(nodes.size());
            toMapleNodesVector(x, q_temp, nodes, yarn_type);
            T V[1];
            #include "Maple/YarnBendPBCV.mcg"
            energy += V[0];
        });
    else
        iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
            // std::cout << nodes[0] << " " << nodes[1] << " "<< nodes[2] << " "<< nodes[3] << " "<<yarn_type <<std::endl;
            std::vector<Vector<T, dim + 1>> x(nodes.size());
            toMapleNodesVector(x, q_temp, nodes, yarn_type);
            T V[1];
            #include "Maple/YarnBendPBCSDV.mcg"
            energy += V[0];
            // std::cout << V[0] << std::endl;
        });
    return energy;
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;