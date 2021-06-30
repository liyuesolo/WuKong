#include "EoLRodSim.h"
using std::abs;
template<class T, int dim>
void EoLRodSim<T, dim>::addBendingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{

    auto entryHelperBending = [&](int n0, int n1, int n2, int uv_offset)
    {
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        std::vector<int> nodes = {n0, n1, n2};        
        getMaterialPositions(q_temp, nodes, X, uv_offset, dXdu, d2Xdu2, true, true);

        std::vector<TV> x(6);
        convertxXforMaple(x, X, q_temp, nodes);
        
        T J[12][12];
        memset(J, 0, sizeof(J));
        #include "Maple/YarnBendDiscreteRestCurvatureJ.mcg"
        for(int k = 0; k < nodes.size(); k++)
            for(int l = 0; l < nodes.size(); l++)
                for(int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        {
                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + i, nodes[l] * dof + j, -J[k*dim + i][l * dim + j]));

                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + i, nodes[l] * dof + dim + uv_offset, -J[k*dim + i][3 * dim + l * dim + j] * dXdu[l][j]));

                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + dim + uv_offset, nodes[l] * dof + j, -J[3 * dim + k * dim + i][l * dim + j] * dXdu[k][i]));

                            
                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + dim + uv_offset, 
                                                                nodes[l] * dof + dim + uv_offset, 
                                                                -J[3 * dim + k*dim + i][3 * dim + l * dim + j] * dXdu[l][j] * dXdu[k][i]));

                        }
        
        
    };

    auto addHessianEntry = [&](int n0, int n1, int n2, int uv_offset)
    {
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        std::vector<int> nodes = {n0, n1, n2};        
        getMaterialPositions(q_temp, nodes, X, uv_offset, dXdu, d2Xdu2, true, true);

        std::vector<TV> x(6);
        convertxXforMaple(x, X, q_temp, nodes);
        
        Vector<T, 12> F;
        F.setZero();

        #include "Maple/YarnBendDiscreteRestCurvatureF.mcg"

        for(int d = 0; d < dim; d++)
        {
            entry_K.push_back(Eigen::Triplet<T>(nodes[0] * dof + dim + uv_offset, 
                                nodes[0] * dof + dim + uv_offset, -F[0*dim + 3*dim + d] * d2Xdu2[0][d]));
            entry_K.push_back(Eigen::Triplet<T>(nodes[1] * dof + dim + uv_offset, 
                                nodes[1] * dof + dim + uv_offset, -F[1*dim + 3*dim + d] * d2Xdu2[1][d]));
            entry_K.push_back(Eigen::Triplet<T>(nodes[2] * dof + dim + uv_offset, 
                                nodes[2] * dof + dim + uv_offset, -F[2*dim + 3*dim + d] * d2Xdu2[2][d]));
        }
        
    };

    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
                {
                    entryHelperBending(middle, right, left, 0);
                    addHessianEntry(middle, right, left, 0);
                }
        if (top != -1 && bottom != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
                {
                    entryHelperBending(middle, top, bottom, 1);
                    addHessianEntry(middle, top, bottom, 1);
                }
    });

    if(!add_pbc_bending)
        return;

    iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
        
        std::vector<TV> x(nodes.size());
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        getMaterialPositions(q_temp, nodes, X, yarn_type, dXdu, d2Xdu2, false, false);
        convertxXforMaple(x, X, q_temp, nodes);
        
        T J[16][16];
        memset(J, 0, sizeof(J));
        #include "Maple/YarnBendDiscreteRestCurvaturePBCJ.mcg"
        for(int k = 0; k < nodes.size(); k++)
            for(int l = 0; l < nodes.size(); l++)
                for(int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        {
                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + i, nodes[l] * dof + j, -J[k*dim + i][l * dim + j]));

                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + i, nodes[l] * dof + dim + yarn_type, -J[k*dim + i][4 * dim + l * dim + j] * dXdu[l][j]));

                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + dim + yarn_type, nodes[l] * dof + j, -J[4 * dim + k * dim + i][l * dim + j] * dXdu[k][i]));

                            
                            entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + dim + yarn_type, 
                                                                nodes[l] * dof + dim + yarn_type, 
                                                                -J[4 * dim + k*dim + i][4 * dim + l * dim + j] * dXdu[l][j] * dXdu[k][i]));

                        }
    });

    iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
        
        std::vector<TV> x(nodes.size());
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        getMaterialPositions(q_temp, nodes, X, yarn_type, dXdu, d2Xdu2, false, false);
        convertxXforMaple(x, X, q_temp, nodes);
        Vector<T, 16> F;
        #include "Maple/YarnBendDiscreteRestCurvaturePBCF.mcg"
        
        for(int d = 0; d < dim; d++)
        {
            entry_K.push_back(Eigen::Triplet<T>(nodes[0] * dof + dim + yarn_type, 
                                nodes[0] * dof + dim + yarn_type, -F[0*dim + 4*dim + d] * d2Xdu2[0][d]));
            entry_K.push_back(Eigen::Triplet<T>(nodes[1] * dof + dim + yarn_type, 
                                nodes[1] * dof + dim + yarn_type, -F[1*dim + 4*dim + d] * d2Xdu2[1][d]));
            entry_K.push_back(Eigen::Triplet<T>(nodes[2] * dof + dim + yarn_type, 
                                nodes[2] * dof + dim + yarn_type, -F[2*dim + 4*dim + d] * d2Xdu2[2][d]));
            entry_K.push_back(Eigen::Triplet<T>(nodes[3] * dof + dim + yarn_type, 
                                nodes[3] * dof + dim + yarn_type, -F[3*dim + 4*dim + d] * d2Xdu2[3][d]));
        }
    });

}

// template<class T, int dim>
// void EoLRodSim<T, dim>::addBendingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
// {

//     auto entryHelperBending = [&](int n0, int n1, int n2, int uv_offset)
//     {
//         T kappa0 = curvature_functions[uv_offset]->value(q_temp(dim + uv_offset, n0));
//         T dkappa0du;
//         curvature_functions[uv_offset]->gradient(q_temp(dim + uv_offset, n0), dkappa0du);
//         T d2kappa0du2;
//         curvature_functions[uv_offset]->hessian(q_temp(dim + uv_offset, n0), d2kappa0du2);

//         std::vector<Vector<T, dim + 1>> x(3);
//         std::vector<int> nodes = {n0, n1, n2};
//         toMapleNodesVector(x, q_temp, nodes, uv_offset);
//         T J[10][10];
//         memset(J, 0, sizeof(J));
//         #include "Maple/YarnBendRestCurvatureJ.mcg"
//         for(int k = 0; k < nodes.size(); k++)
//             for(int l = 0; l < nodes.size(); l++)
//                 for(int i = 0; i < dim + 1; i++)
//                     for (int j = 0; j < dim + 1; j++)
//                         {
//                             int u_offset = i == dim ? dim + uv_offset : i;
//                             int v_offset = j == dim ? dim + uv_offset : j;
//                             entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + u_offset, nodes[l] * dof + v_offset, -J[k*(dim + 1) + i][l * (dim + 1) + j]));
//                         }
        
//         for (int i = 0; i < 9; i++)
//         {
//             int node_i = std::floor(T(i)/(dim + 1));
//             int dof_i = i % (dim + 1);
//             int offset = dof_i == dim ? dim + uv_offset : dof_i;

//             entry_K.push_back(Entry(nodes[node_i] * dof + offset, nodes[0] * dof + dim + uv_offset, -J[i][9] * -dkappa0du));
//             entry_K.push_back(Entry(nodes[0] * dof + dim + uv_offset, nodes[node_i] * dof + offset, -J[9][i] * -dkappa0du));
//         }

//         entry_K.push_back(Entry(nodes[0] * dof + dim + uv_offset, nodes[0] * dof + dim + uv_offset, -J[9][9] * d2kappa0du2));
//     };

//     iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
//         if (left != -1 && right != -1)
//             if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
//                 entryHelperBending(middle, right, left, 0);
//         if (top != -1 && bottom != -1)
//             if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
//                 entryHelperBending(middle, top, bottom, 1);
//     });

//     if(!add_pbc_bending)
//         return;
        
//     if (!subdivide)
//         iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
//             int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
//             std::vector<Vector<T, dim + 1>> x(nodes.size());
//             toMapleNodesVector(x, q_temp, nodes, yarn_type);
//             T J[15][15];
//             memset(J, 0, sizeof(J));
//             #include "Maple/YarnBendPBCJ.mcg"
//             for(int k = 0; k < nodes.size(); k++)
//                 for(int l = 0; l < nodes.size(); l++)
//                     for(int i = 0; i < dim + 1; i++)
//                         for (int j = 0; j < dim + 1; j++)
//                             {
//                                 int u_offset = i == dim ? dim + yarn_type : i;
//                                 int v_offset = j == dim ? dim + yarn_type : j;
        
//                                 entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + u_offset, nodes[l] * dof + v_offset, -J[k*(dim + 1) + i][l * (dim + 1) + j]));
//                             }
//         });    
//     else
//         iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
//             T kappa0 = curvature_functions[yarn_type]->value(q_temp(dim + yarn_type, nodes[0]));
//             T kappa1 = curvature_functions[yarn_type]->value(q_temp(dim + yarn_type, nodes[3]));

//             T dkappa0du, dkappa1du;
//             curvature_functions[yarn_type]->gradient(q_temp(dim + yarn_type, nodes[0]), dkappa0du);
//             curvature_functions[yarn_type]->gradient(q_temp(dim + yarn_type, nodes[3]), dkappa1du);
            
//             T d2kappa0du2, d2kappa1du2;

//             curvature_functions[yarn_type]->hessian(q_temp(dim + yarn_type, nodes[0]), d2kappa0du2);
//             curvature_functions[yarn_type]->hessian(q_temp(dim + yarn_type, nodes[3]), d2kappa1du2);

//             std::vector<Vector<T, dim + 1>> x(nodes.size());
//             toMapleNodesVector(x, q_temp, nodes, yarn_type);
//             T J[14][14];
//             memset(J, 0, sizeof(J));
//             #include "Maple/YarnBendPBCRestCurvatureJ.mcg"
//             int cnt = 0;
//             for(int k = 0; k < nodes.size(); k++)
//                 for(int l = 0; l < nodes.size(); l++)
//                     for(int i = 0; i < dim + 1; i++)
//                         for (int j = 0; j < dim + 1; j++)
//                             {
//                                 int u_offset = i == dim ? dim + yarn_type : i;
//                                 int v_offset = j == dim ? dim + yarn_type : j;
        
//                                 entry_K.push_back(Eigen::Triplet<T>(nodes[k] * dof + u_offset, nodes[l] * dof + v_offset, -J[k*(dim + 1) + i][l * (dim + 1) + j]));
//                             }

            
//             for (int i = 0; i < 12; i++)
//             {
//                 int node_i = std::floor(T(i)/(dim + 1));
//                 int dof_i = i % (dim + 1);
//                 int offset = dof_i == dim ? dim + yarn_type : dof_i;

//                 entry_K.push_back(Entry(nodes[node_i] * dof + offset, nodes[0] * dof + dim + yarn_type, -J[i][12] * -dkappa0du));
//                 entry_K.push_back(Entry(nodes[node_i] * dof + offset, nodes[3] * dof + dim + yarn_type, -J[i][13] * -dkappa1du));

//                 entry_K.push_back(Entry(nodes[0] * dof + dim + yarn_type, nodes[node_i] * dof + offset, -J[12][i] * -dkappa0du));
//                 entry_K.push_back(Entry(nodes[3] * dof + dim + yarn_type, nodes[node_i] * dof + offset, -J[13][i] * -dkappa1du));
//             }

//             entry_K.push_back(Entry(nodes[0] * dof + dim + yarn_type, nodes[0] * dof + dim + yarn_type, -J[12][12] * d2kappa0du2));
//             entry_K.push_back(Entry(nodes[3] * dof + dim + yarn_type, nodes[3] * dof + dim + yarn_type, -J[13][13] * d2kappa1du2));
//             entry_K.push_back(Entry(nodes[0] * dof + dim + yarn_type, nodes[3] * dof + dim + yarn_type, -J[12][13] * -dkappa0du * -dkappa1du));
//             entry_K.push_back(Entry(nodes[3] * dof + dim + yarn_type, nodes[0] * dof + dim + yarn_type, -J[13][12] * -dkappa0du * -dkappa1du));
//         });
// }

// template<class T, int dim>
// void EoLRodSim<T, dim>::addBendingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
// {
//     auto addBendingForceSingleDirection = [&](int n0, int n1, int n2, int uv_offset)
//     {
//         // if(n0 == 4)
//             // cout4Nodes(n0, n1, n2, uv_offset);
//         T kappa0 = curvature_functions[uv_offset]->value(q_temp(dim + uv_offset, n0));
//         T dkappa0du;
//         curvature_functions[uv_offset]->gradient(q_temp(dim + uv_offset, n0), dkappa0du);
        
//         std::vector<Vector<T, dim + 1>> x(3);
//         std::vector<int> nodes = {n0, n1, n2};
//         toMapleNodesVector(x, q_temp, nodes, uv_offset);
//         Vector<T, 10> F;
//         F.setZero();
//         #include "Maple/YarnBendRestCurvatureF.mcg"
//         // std::cout << "bending crossing force local " << F.transpose() << std::endl;
//         // for (int node : nodes)
//         //     std::cout << node << " " << q_temp.col(node).transpose() << " uv " << uv_offset << std::endl;
        
//         int cnt = 0;
//         for (int node : nodes)
//         {
//             residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*(dim+1));
//             residual(dim + uv_offset, node) += F[cnt*(dim+1)+dim];
//             cnt++;
//         } 
//         residual(dim + uv_offset, n0) += F(9) * -dkappa0du;
//     };
    
//     DOFStack residual_cp = residual;
//     iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
//         if (left != -1 && right != -1)
//             if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
//                 addBendingForceSingleDirection(middle, right, left, 0);
//         if (top != -1 && bottom != -1)
//             if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
//                 addBendingForceSingleDirection(middle, top, bottom, 1);
//     });  
//     if(!add_pbc_bending)
//     {
//         if (print_force_mag)
//             std::cout << "bending force " << (residual - residual_cp).norm() << std::endl;
//         return;
//     }
//     // std::cout << "bending crossing force " << (residual - residual_cp).norm() << std::endl; 
//     if (!subdivide)
//         iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
//             int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
//             std::vector<Vector<T, dim + 1>> x(nodes.size());
//             toMapleNodesVector(x, q_temp, nodes, yarn_type);
//             Vector<T, 15> F;
//             F.setZero();
//             #include "Maple/YarnBendPBCF.mcg"
//             int cnt = 0;
//             for (int node : nodes)
//             {
//                 residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*(dim+1));
//                 residual(dim + yarn_type, node) += F[cnt*(dim+1)+dim];
//                 cnt++;
//             }
//         });
//     else
//         iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
//             // for (int node : nodes)
//             //     std::cout << node << " ";
//             // std::cout << std::endl;
//             T kappa0 = curvature_functions[yarn_type]->value(q_temp(dim + yarn_type, nodes[0]));
//             T kappa1 = curvature_functions[yarn_type]->value(q_temp(dim + yarn_type, nodes[3]));
//             T dkappa0du, dkappa1du;
//             curvature_functions[yarn_type]->gradient(q_temp(dim + yarn_type, nodes[0]), dkappa0du);
//             curvature_functions[yarn_type]->gradient(q_temp(dim + yarn_type, nodes[3]), dkappa1du);
//             // std::cout << dkappa0du << " " << dkappa1du << std::endl;
//             std::vector<Vector<T, dim + 1>> x(nodes.size());
//             toMapleNodesVector(x, q_temp, nodes, yarn_type);
//             Vector<T, 14> F;
//             F.setZero();
//             #include "Maple/YarnBendPBCRestCurvatureF.mcg"
//             // std::cout << "bending force " << F.transpose() << std::endl;
//             // for (int node : nodes)
//             //     std::cout << node << " ";
//             // std::cout << std::endl;
//             int cnt = 0;
//             for (int node : nodes)
//             {
//                 residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*(dim+1));
//                 residual(dim + yarn_type, node) += F[cnt*(dim+1)+dim];
//                 cnt++;
//             }
//             // std::cout << "bending pbc force " << (residual - residual_cp).norm() << std::endl;
//             residual(dim + yarn_type, nodes[0]) += F(12) * -dkappa0du;
//             residual(dim + yarn_type, nodes[3]) += F(13) * -dkappa1du;
//             // std::cout << "something" << std::endl;
//         });
//     // std::cout << "bending force " << (residual - residual_cp).transpose() << std::endl;
//     if (print_force_mag)
//         std::cout << "bending force " << (residual - residual_cp).norm() << std::endl;
//     // std::getchar();
//     // std::exit(0);
// }


template<class T, int dim>
void EoLRodSim<T, dim>::addBendingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    auto addBendingForceSingleDirection = [&](int n0, int n1, int n2, int uv_offset)
    {
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        std::vector<int> nodes = {n0, n1, n2};        
        getMaterialPositions(q_temp, nodes, X, uv_offset, dXdu, d2Xdu2, true, false);

        std::vector<TV> x(6);
        convertxXforMaple(x, X, q_temp, nodes);

        Vector<T, 12> F;
        F.setZero();

        #include "Maple/YarnBendDiscreteRestCurvatureF.mcg"
        // std::cout << "bending crossing force local " << F.transpose() << std::endl;
        // for (int node : nodes)
        //     std::cout << node << " " << q_temp.col(node).transpose() << " uv " << uv_offset << std::endl;
        
        int cnt = 0;
        for (int node : nodes)
        {
            residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*dim);
            for(int d = 0; d < dim; d++)
            {
                residual(dim + uv_offset, node) += F[cnt*dim + 3*dim + d] * dXdu[cnt][d];
            }
            cnt++;
        } 
        
    };
    
    DOFStack residual_cp = residual;
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left])
                addBendingForceSingleDirection(middle, right, left, 0);
        if (top != -1 && bottom != -1)
            if(!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom])
                addBendingForceSingleDirection(middle, top, bottom, 1);
    });  
    if(!add_pbc_bending)
    {
        if (print_force_mag)
            std::cout << "bending force " << (residual - residual_cp).norm() << std::endl;
        return;
    }

    iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
        
        std::vector<TV> x(nodes.size());
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        getMaterialPositions(q_temp, nodes, X, yarn_type, dXdu, d2Xdu2, false, false);
        convertxXforMaple(x, X, q_temp, nodes);
        Vector<T, 16> F;
        #include "Maple/YarnBendDiscreteRestCurvaturePBCF.mcg"
        
        int cnt = 0;
        for (int node : nodes)
        {
            residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*dim);
            for(int d = 0; d < dim; d++)
            {
                residual(dim + yarn_type, node) += F[cnt*dim + 4*dim + d] * dXdu[cnt][d];
            }
            cnt++;
        }
    });
    
    if (print_force_mag)
        std::cout << "bending force " << (residual - residual_cp).norm() << std::endl;
    
}



// template<class T, int dim>
// T EoLRodSim<T, dim>::addBendingEnergy(Eigen::Ref<const DOFStack> q_temp)
// {

//     auto get_material_positions = [&](int n0, int n1, int n2)
//     {
//         T X0 = curvature_functions[0]->value(q_temp(dim + 0, n0));
//         T X1 = curvature_functions[0]->value(q_temp(dim + 0, n1));
//         T X2 = curvature_functions[0]->value(q_temp(dim + 0, n2));

//         T Y0 = curvature_functions[1]->value(q_temp(dim + 1, n0));
//         T Y1 = curvature_functions[1]->value(q_temp(dim + 1, n1));
//         T Y2 = curvature_functions[1]->value(q_temp(dim + 1, n2));

//         return { TV2(X0, Y0), TV2(X1, Y1), TV2(X2, Y2) };

//     };

//     auto bendingEnergySingleDirection = [&](int n0, int n1, int n2, int uv_offset)
//     {
        
//         auto X = get_material_positions(n0, n1, n2);

//         T kappa0 = curvature_functions[uv_offset]->value(q_temp(dim + uv_offset, n0));
//         // if (uv_offset == 1)
//             // std::cout << "curvature "<< kappa0 << std::endl;
//         std::vector<Vector<T, dim + 1>> x(3);
//         std::vector<int> nodes = {n0, n1, n2};
//         toMapleNodesVector(x, q_temp, nodes, uv_offset);

//         // x0, x1, x2, u1, u2
//         // X0, X1, X2, u0, u1, u2

//         T V[1];
//         #include "Maple/YarnBendRestCurvatureV.mcg"
//         // std::cout << V[0] << std::endl;
//         // std::getchar();
        
//         // if (uv_offset == 0)
//         // {
//         //     std::cout << "node " << n0 << " " << V[0] << std::endl;
//         //     std::cout << "theta " << t27 << " discrete " << 0.4E1/t1*t27 << " analytical " << kappa0 << std::endl;
//         //     std::cout << "u left: " << q_temp(dim + uv_offset, n2) << " u middle: " << q_temp(dim + uv_offset, n0) << " u right: " << q_temp(dim + uv_offset, n1) << std::endl;            
//         //     std::cout << "u left: " << q0(dim + uv_offset, n2) << " u middle: " << q0(dim + uv_offset, n0) << " u right: " << q0(dim + uv_offset, n1) << std::endl;
//         // }
//         // std::getchar();
//         return V[0];
//     };

//     T energy = 0.0;
    
//     VectorXT crossing_energy(n_nodes);
//     crossing_energy.setZero();
     
//     iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
//         if (left != -1 && right != -1)
//             if((!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left]) )
//                 crossing_energy[middle] += bendingEnergySingleDirection(middle, right, left, 0);
//         if (top != -1 && bottom != -1)
//             if((!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom]) )
//                 crossing_energy[middle] += bendingEnergySingleDirection(middle, top, bottom, 1);
//     });
//     energy += crossing_energy.sum();
    
//     if(!add_pbc_bending)
//         return energy;
//     if (!subdivide)
//         iteratePBCBendingPairs([&](std::vector<int> nodes, int pair_id){
//             int yarn_type = pbc_ref[pair_id].first == WARP ? 0 : 1;
//             std::vector<Vector<T, dim + 1>> x(nodes.size());
//             toMapleNodesVector(x, q_temp, nodes, yarn_type);
//             T V[1];
//             #include "Maple/YarnBendPBCV.mcg"
//             energy += V[0];
//         });
//     else
//         iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
//             // std::cout << nodes[0] << " " << nodes[1] << " "<< nodes[2] << " "<< nodes[3] << " "<<yarn_type <<std::endl;
//             T kappa0 = curvature_functions[yarn_type]->value(q_temp(dim + yarn_type, nodes[0]));
//             T kappa1 = curvature_functions[yarn_type]->value(q_temp(dim + yarn_type, nodes[3]));
//             // std::cout << kappa0 << " " << kappa1 << std::endl;
//             // std::getchar();
//             std::vector<Vector<T, dim + 1>> x(nodes.size());
//             toMapleNodesVector(x, q_temp, nodes, yarn_type);
//             T V[1];
//             #include "Maple/YarnBendPBCRestCurvatureV.mcg"
//             energy += V[0];
//             // std::cout << V[0] << std::endl;
//         });
//     return energy;
// }


template<class T, int dim>
T EoLRodSim<T, dim>::addBendingEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    auto bendingEnergySingleDirection = [&](int n0, int n1, int n2, int uv_offset)
    {
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        std::vector<int> nodes = {n0, n1, n2};        
        getMaterialPositions(q_temp, nodes, X, uv_offset, dXdu, d2Xdu2, false, false);

        std::vector<TV> x(6);
        convertxXforMaple(x, X, q_temp, nodes);
        
        T V[1];
        #include "Maple/YarnBendDiscreteRestCurvatureV.mcg"
        return V[0];
    };

    T energy = 0.0;
    
    VectorXT crossing_energy(n_nodes);
    crossing_energy.setZero();
     
    iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        if (left != -1 && right != -1)
            if((!is_end_nodes[middle] && !is_end_nodes[right] && !is_end_nodes[left]) )
                crossing_energy[middle] += bendingEnergySingleDirection(middle, right, left, 0);
        if (top != -1 && bottom != -1)
            if((!is_end_nodes[middle] && !is_end_nodes[top] && !is_end_nodes[bottom]) )
                crossing_energy[middle] += bendingEnergySingleDirection(middle, top, bottom, 1);
    });
    energy += crossing_energy.sum();
    
    if(!add_pbc_bending)
        return energy;
    

    iteratePBCBoundaryPairs([&](std::vector<int> nodes, int yarn_type){
        
        std::vector<TV> x(nodes.size());
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        getMaterialPositions(q_temp, nodes, X, yarn_type, dXdu, d2Xdu2, false, false);
        convertxXforMaple(x, X, q_temp, nodes);

        T V[1];
        #include "Maple/YarnBendDiscreteRestCurvaturePBCV.mcg"
        energy += V[0];
        // std::cout << V[0] << std::endl;
    });

    return energy;
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;