#include "../include/FEMSolver.h"

double stiffness_val_virtual = 1e8;

const std::vector<std::pair<double, double>> IntegrationPoints1 = {
    std::pair<double, double>(2.0, 0.0)};

const std::vector<std::pair<double, double>> IntegrationPoints3 = {
    std::pair<double, double>(5.0 / 9.0, -sqrt(3.0 / 5.0)),
    std::pair<double, double>(8.0 / 9.0, 0.0),
    std::pair<double, double>(5.0 / 9.0, +sqrt(3.0 / 5.0))};

const std::vector<std::pair<double, double>> IntegrationPoints4 = {
    std::pair<double, double>((18.0 - sqrt(30.0)) / 36.0, +sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt(6.0 / 5.0))),
    std::pair<double, double>((18.0 + sqrt(30.0)) / 36.0, +sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt(6.0 / 5.0))),
    std::pair<double, double>((18.0 + sqrt(30.0)) / 36.0, -sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt(6.0 / 5.0))),
    std::pair<double, double>((18.0 - sqrt(30.0)) / 36.0, -sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt(6.0 / 5.0)))};

const std::vector<std::pair<double, double>> IntegrationPoints5 = {
    std::pair<double, double>((322.0 - 13 * sqrt(70.0)) / 900.0, +sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0)) / 3.0),
    std::pair<double, double>((322.0 + 13 * sqrt(70.0)) / 900.0, +sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)) / 3.0),
    std::pair<double, double>(128.0 / 225.0, 0),
    std::pair<double, double>((322.0 + 13 * sqrt(70.0)) / 900.0, -sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)) / 3.0),
    std::pair<double, double>((322.0 - 13 * sqrt(70.0)) / 900.0, -sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0)) / 3.0)};

struct virtual_info_transform
{
    int i1;
    int i2;
    double scale1;
    double scale2;
};

template <typename DerivedLocalGrad, typename DerivedGrad>
void local_gradient_to_global_gradient_virtual(
    const Eigen::MatrixBase<DerivedLocalGrad> &local_grad,
    const std::vector<virtual_info_transform> &ids,
    int dim,
    Eigen::PlainObjectBase<DerivedGrad> &grad)
{
    for (int i = 0; i < ids.size(); i++)
    {

        //grad.segment(dim * ids[i].i1, dim);
        //std::cout<< local_grad.size()<<std::endl;
        grad.segment(dim * ids[i].i1, dim) += ids[i].scale1 * local_grad.segment(dim * i, dim);
        if (ids[i].i2 != -1)
            grad.segment(dim * ids[i].i2, dim) += ids[i].scale2 * local_grad.segment(dim * i, dim);
    }
}

template <typename Derived>
void local_hessian_to_global_triplets_virtual(
    const Eigen::MatrixBase<Derived> &local_hessian,
    const std::vector<virtual_info_transform> &ids,
    int dim,
    std::vector<Eigen::Triplet<double>> &triplets)
{
    for (int i = 0; i < ids.size(); i++)
    {
        for (int j = 0; j < ids.size(); j++)
        {
            for (int k = 0; k < dim; k++)
            {
                for (int l = 0; l < dim; l++)
                {
                    triplets.emplace_back(
                        dim * ids[i].i1 + k, dim * ids[j].i1 + l,
                        ids[i].scale1 * ids[j].scale1 * local_hessian(dim * i + k, dim * j + l));

                    if (ids[j].i2 != -1)
                    {
                        triplets.emplace_back(
                            dim * ids[i].i1 + k, dim * ids[j].i2 + l,
                            ids[i].scale1 * ids[j].scale2 * local_hessian(dim * i + k, dim * j + l));
                    }

                    if (ids[i].i2 != -1)
                    {
                        triplets.emplace_back(
                            dim * ids[i].i2 + k, dim * ids[j].i1 + l,
                            ids[i].scale2 * ids[j].scale1 * local_hessian(dim * i + k, dim * j + l));
                    }

                    if (ids[i].i2 != -1 && ids[j].i2 != -1)
                    {
                        triplets.emplace_back(
                            dim * ids[i].i2 + k, dim * ids[j].i2 + l,
                            ids[i].scale2 * ids[j].scale2 * local_hessian(dim * i + k, dim * j + l));
                    }
                }
            }
        }
    }
}

template <int dim>
void FEMSolver<dim>::GenerateVirtualPoints()
{
    std::cout << "Generate Gauss Points" << std::endl;
    virtual_slave_nodes.clear();
    if (slave_nodes.size() > 1)
    {
        for (int i = 0; i < slave_nodes.size() - 1; ++i)
        {
            int i1 = slave_nodes[i];
            int i2 = slave_nodes[i + 1];
            if (USE_NEW_FORMULATION)
            {
                i1 += num_nodes;
                i2 += num_nodes;
            }
            for (int j = 0; j < IntegrationPoints3.size(); ++j)
            {
                double w = IntegrationPoints3[j].first;
                double pos = IntegrationPoints3[j].second;

                virtual_slave_nodes.push_back({i1, i2, pos, w});
            }
        }
    }
    if (IMLS_BOTH || BILATERAL)
    {
        virtual_master_nodes.clear();
        if (master_nodes.size() > 1)
        {
            for (int i = 0; i < master_nodes.size() - 1; ++i)
            {
                int i1 = master_nodes[i];
                int i2 = master_nodes[i + 1];

                if (USE_NEW_FORMULATION)
                {
                    i1 += num_nodes;
                    i2 += num_nodes;
                }
                for (int j = 0; j < IntegrationPoints3.size(); ++j)
                {
                    double w = IntegrationPoints3[j].first;
                    double pos = IntegrationPoints3[j].second;

                    virtual_master_nodes.push_back({i1, i2, pos, w});
                }
            }
        }
    }
}

double compute_potential_virtual(double d_hat, double d, int i)
{
    if (d >= 0)
        return 0;
    else
        return -stiffness_val_virtual * d * d * d;
}

Eigen::VectorXd compute_potential_gradient_virtual(double d_hat, double d, const Eigen::VectorXd &dist_grad)
{
    double barrier_gradient;
    if (d >= 0)
        barrier_gradient = 0;
    else
        barrier_gradient = -3 * stiffness_val_virtual * d * d;

    return dist_grad * barrier_gradient;
}

Eigen::MatrixXd compute_potential_hessian_virtual(double d_hat, double d, const Eigen::VectorXd &dist_grad, const Eigen::MatrixXd &dist_hess)
{
    Eigen::MatrixXd hess;

    double barrier_hess, barrier_gradient;
    if (d >= 0)
        barrier_gradient = 0;
    else
        barrier_gradient = -3 * stiffness_val_virtual * d * d;

    if (d >= 0)
        barrier_hess = 0;
    else
        barrier_hess = -6 * stiffness_val_virtual * d;
    hess = barrier_hess * dist_grad * dist_grad.transpose() + barrier_gradient * dist_hess;

    return hess;
}

double compute_potential_vts_Sqaured_Norm(double d_hat, double d, int i)
{
    return 0.5 * d * d;
}

Eigen::VectorXd compute_potential_vts_gradient_Sqaured_Norm(double d_hat, double d, const Eigen::VectorXd &dist_grad)
{
    double barrier_gradient = d;
    return dist_grad * barrier_gradient;
}

Eigen::MatrixXd compute_potential_vts_hessian_Sqaured_Norm(double d_hat, double d, const Eigen::VectorXd &dist_grad, const Eigen::MatrixXd &dist_hess)
{
    Eigen::MatrixXd hess;
    double barrier_hess = 1;
    double barrier_gradient = d;
    hess = barrier_hess * dist_grad * dist_grad.transpose() + barrier_gradient * dist_hess;

    return hess;
}

template <int dim>
double FEMSolver<dim>::compute_vts_potential2D(Eigen::MatrixXd &ipc_vertices_deformed, bool eval_same_side)
{
    std::unordered_map<int,int> map_boundary_virtual_current = map_boundary_virtual;
    std::vector<point_segment_pair> boundarys = boundary_info;
    if(USE_NEW_FORMULATION && eval_same_side)
    {
        map_boundary_virtual_current = map_boundary_virtual_same_side;
        boundarys = boundary_info_same_side;
    }
    tbb::enumerable_thread_specific<double> storage(0);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), boundarys.size()),
        [&](const tbb::blocked_range<size_t> &r)
        {
            auto &local_potential = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                if (map_boundary_virtual_current.find(i) != map_boundary_virtual_current.end())
                {
                    if (USE_NEW_FORMULATION && eval_same_side)
                        local_potential += IMLS_param * virtual_slave_nodes[map_boundary_virtual_current[i]].w * 0.5 *
                                           boundarys[i].scale * compute_potential_vts_Sqaured_Norm(barrier_distance, boundarys[i].dist, boundarys[i].slave_index);
                    else
                        local_potential += virtual_slave_nodes[map_boundary_virtual_current[i]].w * 0.5 *
                                           boundarys[i].scale * compute_potential_virtual(barrier_distance, boundarys[i].dist, boundarys[i].slave_index);
                }
                else
                {
                    if (USE_NEW_FORMULATION && eval_same_side)
                        local_potential += IMLS_param *
                                           boundarys[i].scale * compute_potential_vts_Sqaured_Norm(barrier_distance, boundarys[i].dist, boundarys[i].slave_index);
                    else
                        local_potential +=
                            boundarys[i].scale * compute_potential_virtual(barrier_distance, boundarys[i].dist, boundarys[i].slave_index);
                }
            }
        });

    double potential = 0;
    for (const auto &local_potential : storage)
    {
        potential += local_potential;
    }
    // std::cout<<"potential: "<<potential<<std::endl;
    return potential;
}

template <int dim>
Eigen::VectorXd FEMSolver<dim>::compute_vts_potential_gradient2D(Eigen::MatrixXd &ipc_vertices_deformed, bool eval_same_side)
{
    int num_nodes_all = num_nodes;
    if (USE_NEW_FORMULATION)
        num_nodes_all *= 2;
    tbb::enumerable_thread_specific<Eigen::VectorXd> storage(
        Eigen::VectorXd::Zero(num_nodes_all * dim));
    std::unordered_map<int,int> map_boundary_virtual_current = map_boundary_virtual;
    std::vector<point_segment_pair> boundarys = boundary_info;
    if(USE_NEW_FORMULATION && eval_same_side)
    {
        map_boundary_virtual_current = map_boundary_virtual_same_side;
        boundarys = boundary_info_same_side;
    }
        
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), boundarys.size()),
        [&](const tbb::blocked_range<size_t> &r)
        {
            auto &local_grad = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                std::vector<virtual_info_transform> vertices;

                if (map_boundary_virtual_current.find(i) != map_boundary_virtual_current.end())
                {
                    VirtualNodeInfo info = virtual_slave_nodes[map_boundary_virtual_current[i]];
                    if (i >= virtual_slave_nodes.size())
                        info = virtual_master_nodes[map_boundary_virtual_current[i]];
                    if (USE_IMLS)
                    {

                        vertices.push_back({info.left_index, info.right_index, (1 - info.eta) / 2.0, (1 + info.eta) / 2.0});
                        for (int j = 0; j < boundarys[i].results.size(); ++j)
                        {
                            vertices.push_back({boundarys[i].results[j], -1, 1., 0.});
                        }
                    }
                    else
                    {
                        vertices.push_back({info.left_index, info.right_index, (1 - info.eta) / 2.0, (1 + info.eta) / 2.0});
                        vertices.push_back({boundarys[i].master_index_1, -1, 1., 0.});
                        vertices.push_back({boundarys[i].master_index_2, -1, 1., 0.});
                    }
                }
                else
                {
                    if (USE_IMLS)
                    {
                        vertices.push_back({boundarys[i].slave_index, -1, 1., 0.});
                        for (int j = 0; j < boundarys[i].results.size(); ++j)
                        {
                            vertices.push_back({boundarys[i].results[j], -1, 1., 0.});
                        }
                    }
                    else
                    {
                        vertices.push_back({boundarys[i].slave_index, -1, 1., 0.});
                        vertices.push_back({boundarys[i].master_index_1, -1, 1., 0.});
                        vertices.push_back({boundarys[i].master_index_2, -1, 1., 0.});
                    }
                }

                if (map_boundary_virtual_current.find(i) != map_boundary_virtual_current.end())
                {
                    if (USE_NEW_FORMULATION && eval_same_side)
                    {
                        //std::cout<<boundarys[i].dist_grad.size()<<std::endl;
                        local_gradient_to_global_gradient_virtual(
                            IMLS_param * virtual_slave_nodes[map_boundary_virtual_current[i]].w * 0.5 * boundarys[i].scale * compute_potential_vts_gradient_Sqaured_Norm(barrier_distance, boundarys[i].dist, boundarys[i].dist_grad),
                            vertices, dim, local_grad);
                    }
                    else
                    {
                        local_gradient_to_global_gradient_virtual(
                            virtual_slave_nodes[map_boundary_virtual_current[i]].w * 0.5 * boundarys[i].scale * compute_potential_gradient_virtual(barrier_distance, boundarys[i].dist, boundarys[i].dist_grad),
                            vertices, dim, local_grad);
                    }
                }
                else
                {
                    if (USE_NEW_FORMULATION && eval_same_side)
                    {
                        local_gradient_to_global_gradient_virtual(
                            IMLS_param * boundarys[i].scale * compute_potential_vts_gradient_Sqaured_Norm(barrier_distance, boundarys[i].dist, boundarys[i].dist_grad),
                            vertices, dim, local_grad);
                    }
                    else
                    {
                        local_gradient_to_global_gradient_virtual(
                            boundarys[i].scale * compute_potential_gradient_virtual(barrier_distance, boundarys[i].dist, boundarys[i].dist_grad),
                            vertices, dim, local_grad);
                    }
                }
            }
        });

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_nodes_all * dim);
    for (const auto &local_grad : storage)
    {
        grad += local_grad;
    }
    return grad;
}

template <int dim>
Eigen::SparseMatrix<double> FEMSolver<dim>::compute_vts_potential_hessian2D(Eigen::MatrixXd &ipc_vertices_deformed, bool eval_same_side)
{
    int num_nodes_all = num_nodes;
    if (USE_NEW_FORMULATION)
        num_nodes_all *= 2;
    tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>>
        storage;

    std::unordered_map<int,int> map_boundary_virtual_current = map_boundary_virtual;
    std::vector<point_segment_pair> boundarys = boundary_info;
    if(USE_NEW_FORMULATION && eval_same_side)
    {
        map_boundary_virtual_current = map_boundary_virtual_same_side;
        boundarys = boundary_info_same_side;
    }

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), boundarys.size()),
        [&](const tbb::blocked_range<size_t> &r)
        {
            auto &local_hess_triplets = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++)
            {
                std::vector<virtual_info_transform> vertices;

                if (map_boundary_virtual_current.find(i) != map_boundary_virtual_current.end())
                {
                    VirtualNodeInfo info = virtual_slave_nodes[map_boundary_virtual_current[i]];
                    if (i >= virtual_slave_nodes.size())
                        info = virtual_master_nodes[map_boundary_virtual_current[i]];
                    if (USE_IMLS)
                    {

                        vertices.push_back({info.left_index, info.right_index, (1 - info.eta) / 2.0, (1 + info.eta) / 2.0});
                        for (int j = 0; j < boundarys[i].results.size(); ++j)
                        {
                            vertices.push_back({boundarys[i].results[j], -1, 1., 0.});
                        }
                    }
                    else
                    {
                        vertices.push_back({info.left_index, info.right_index, (1 - info.eta) / 2.0, (1 + info.eta) / 2.0});
                        vertices.push_back({boundary_info[i].master_index_1, -1, 1., 0.});
                        vertices.push_back({boundary_info[i].master_index_2, -1, 1., 0.});
                    }
                }
                else
                {
                    if (USE_IMLS)
                    {
                        vertices.push_back({boundarys[i].slave_index, -1, 1., 0.});
                        for (int j = 0; j < boundarys[i].results.size(); ++j)
                        {
                            vertices.push_back({boundarys[i].results[j], -1, 1., 0.});
                        }
                    }
                    else
                    {
                        vertices.push_back({boundary_info[i].slave_index, -1, 1., 0.});
                        vertices.push_back({boundary_info[i].master_index_1, -1, 1., 0.});
                        vertices.push_back({boundary_info[i].master_index_2, -1, 1., 0.});
                    }
                }

                if (map_boundary_virtual_current.find(i) != map_boundary_virtual_current.end())
                {
                    if (USE_NEW_FORMULATION && eval_same_side)
                        local_hessian_to_global_triplets_virtual(
                            IMLS_param * virtual_slave_nodes[map_boundary_virtual_current[i]].w * 0.5 * boundarys[i].scale * compute_potential_vts_hessian_Sqaured_Norm(barrier_distance, boundarys[i].dist, boundarys[i].dist_grad, boundarys[i].dist_hess),
                            vertices, dim,
                            local_hess_triplets);
                    else
                        local_hessian_to_global_triplets_virtual(
                            virtual_slave_nodes[map_boundary_virtual_current[i]].w * 0.5 * boundarys[i].scale * compute_potential_hessian_virtual(barrier_distance, boundarys[i].dist, boundarys[i].dist_grad, boundarys[i].dist_hess),
                            vertices, dim,
                            local_hess_triplets);
                }
                else
                {
                    if (USE_NEW_FORMULATION && eval_same_side)
                        local_hessian_to_global_triplets_virtual(
                            IMLS_param * boundarys[i].scale * compute_potential_vts_hessian_Sqaured_Norm(barrier_distance, boundarys[i].dist, boundarys[i].dist_grad, boundarys[i].dist_hess),
                            vertices, dim,
                            local_hess_triplets);
                    else
                        local_hessian_to_global_triplets_virtual(
                            boundarys[i].scale * compute_potential_hessian_virtual(barrier_distance, boundarys[i].dist, boundarys[i].dist_grad, boundarys[i].dist_hess),
                            vertices, dim,
                            local_hess_triplets);
                }
            }
        });

    Eigen::SparseMatrix<double> hess(num_nodes_all * dim, num_nodes_all * dim);
    for (const auto &local_hess_triplets : storage)
    {
        Eigen::SparseMatrix<double> local_hess(num_nodes_all * dim, num_nodes_all * dim);
        local_hess.setFromTriplets(
            local_hess_triplets.begin(), local_hess_triplets.end());
        hess += local_hess;
    }

    return hess;
}

template class FEMSolver<2>;
template class FEMSolver<3>;