#include "../include/FEMSolver.h"
//#include <igl/mosek/mosek_quadprog.h>

int max_num_iter = 200;
// 0 -- parabola y = x^2
int TEST_FUNCTION = 0;

template<int dim>
void FEMSolver<dim>::CalculateProjectionIMLS()
{
    // First build the IMLS surface for the slave side
    projectedPts.setZero(slave_nodes.size(),dim);
    bool IMLS_MEM = IMLS_BOTH;
    if(IMLS_MEM)
    {
        projectedPts.setZero(slave_nodes.size()+master_nodes.size(),dim);
    }
    IMLS_BOTH = true;
    if(use_multiple_pairs)
    {
        int num_pairs = multiple_slave_nodes.size();
        int index = 0;
        for(int i=0; i<num_pairs; ++i)
        {
            samplePointsFromSTL(i);
            buildConstraintsPointSet();
            for(int j=0; j<multiple_slave_nodes[i].size(); ++j)
            {
                Eigen::VectorXd proj = SinglePointProjection(deformed.segment<dim>(dim*multiple_slave_nodes[i][j]));
                projectedPts.row(index) = proj;
                index++;
            }
        }

        if(IMLS_MEM)
        {
            int num_pairs = multiple_master_nodes.size();
            for(int i=0; i<num_pairs; ++i)
            {
                samplePointsFromSTL(i);
                buildConstraintsPointSet();
                for(int j=0; j<multiple_master_nodes[i].size(); ++j)
                {
                    Eigen::VectorXd proj = SinglePointProjection(deformed.segment<dim>(dim*multiple_master_nodes[i][j]),0);
                    projectedPts.row(index) = proj;
                    index++;
                }
            }
        }
    }
    else
    {
        samplePointsFromSTL();
        buildConstraintsPointSet();
        
        projectedPts.setZero(slave_nodes.size(), dim);
        if(IMLS_BOTH) projectedPts.setZero(slave_nodes.size()+master_nodes.size(), dim);

        Eigen::MatrixXd ps(slave_nodes.size(),dim);
        
        for(int i=0; i<slave_nodes.size(); ++i)
        {
            ps.row(i) = deformed.segment<dim>(dim*slave_nodes[i]);
        }
        projectedPts.block(0,0,slave_nodes.size(),dim) = SinglePointProjection(ps,false,1);

        if(IMLS_MEM)
        {
            Eigen::MatrixXd ps(master_nodes.size(),dim);
        
            for(int i=0; i<master_nodes.size(); ++i)
            {
                ps.row(i) = deformed.segment<dim>(dim*master_nodes[i]);
            }
            projectedPts.block(slave_nodes.size(),0,master_nodes.size(),dim) = SinglePointProjection(ps,false,1);
        }
    }

    int num_all_ptr = deformed.size()/dim + projectedPts.rows();
    deformed_all.setZero(dim*num_all_ptr);

    for(int i=0; i<deformed.size(); ++i)
    {
        deformed_all(i) = deformed(i);
    }
    for(int i=0; i<projectedPts.rows(); ++i)
    {
        deformed_all.segment<dim>(deformed.size()+dim*i) = projectedPts.row(i);
    }

    undeformed = deformed_all;
    deformed = deformed_all;
    u.setZero(num_all_ptr * dim);
    f.setZero(num_all_ptr * dim);

    IMLS_BOTH = IMLS_MEM;
}

template<int dim>
Eigen::MatrixXd FEMSolver<dim>::SinglePointProjection(Eigen::MatrixXd ps, bool test, int index)
{
    // Find the projection points using SQP
    double g_norm = 1e10;
    double tol_g = 1e-5;
    double initial_gradient_norm = 1e10;
    double E0;
    Eigen::VectorXd dp;
    int num_pts = ps.rows();

    Eigen::VectorXd lc(num_pts);
    Eigen::VectorXd uc(num_pts);
    Eigen::VectorXd lx(dim*num_pts);
    Eigen::VectorXd ux(dim*num_pts);

    Eigen::VectorXd projs_vec(dim*num_pts);

    // set lower and upper bounds for each dof (in our case -inf to inf)
    for(int i=0; i<dim*num_pts; ++i)
    {
        lx(i) = -1000;
        ux(i) = 1000;
    }

    // Lagrangian Multipliers
    std::vector<Eigen::VectorXd> lamdas(4);
    lamdas[0] = Eigen::VectorXd::Random(num_pts);
    lamdas[1] = Eigen::VectorXd::Random(num_pts);
    lamdas[2] = Eigen::VectorXd::Random(dim*num_pts);
    lamdas[3] = Eigen::VectorXd::Random(dim*num_pts);
    
    // Results
    Eigen::MatrixXd projs = ps;
    for(int i=0; i<num_pts; ++i)
    {
        projs(i,0) -= 0.08;
        projs(i,1) += 0.08;
    }
    //projs.setR();

    // for(int step = 0; step < max_num_iter; step++)
    // {
    //     dp.setZero(dim*num_pts);

    //     StiffnessMatrix Q = getQ(ps-projs,test).sparseView();
    //     StiffnessMatrix A = getgradphi(projs,test,index).sparseView();
    //     Eigen::VectorXd c = getc(ps-projs,test);
    //     double cf = getcf(ps-projs,test);

    //     Eigen::MatrixXd gradphi = getgradphi(projs,test,index);
    //     Eigen::VectorXd phi = getphi(projs,test,index);
    //     for(int j=0; j<num_pts; ++j)
    //     {
    //         lc(j) = -phi(j);
    //         uc(j) = -phi(j);
    //     }

    //     igl::mosek::MosekData mosek_data;
        
    //     // std::cout<<"Q: "<<std::endl;
    //     // std::cout<<Q<<std::endl; 
    //     // std::cout<<"c: "<<std::endl;
    //     // std::cout<<c<<std::endl;
    //     // std::cout<<"A: "<<std::endl;
    //     // std::cout<<A<<std::endl;
    //     // std::cout<<"lc: "<<std::endl;
    //     // std::cout<<lc<<std::endl;
    //     // std::cout<<"uc: "<<std::endl;
    //     // std::cout<<uc<<std::endl;

    //     bool solve_success = igl::mosek::mosek_quadprog(Q, c, cf, A, lc, uc, lx, ux, mosek_data, dp, lamdas);
    //     // Reshape result
    //     for(int i=0; i<num_pts; ++i)
    //     {
    //         projs_vec.segment<dim>(dim*i) = projs.row(i);
    //     }

    //     // calculate the g_norm
    //     g_norm = getKKTgrad(lamdas,projs,ps,test).norm();
    //     E0 = 0.5*(projs-ps).squaredNorm() - lamdas[0].dot(getphi(projs,test)) + lamdas[1].dot(getphi(projs,test)) - lamdas[2].dot(projs_vec) + lamdas[3].dot(projs_vec);

    //     if(step == 0)
    //     {
    //         initial_gradient_norm = g_norm;
    //     }

    //     std::cout << "[" << "SQP" << "] iter " << step << " |g| " << g_norm 
    //         << " |g_init| " << initial_gradient_norm << " tol " << tol_g << " obj: " << E0 << " dp_norm: "<<dp.norm()<<std::endl;

    //     if (g_norm < tol_g )
    //         break;

    //     if(dp.norm() < 1e-8)
    //         break;

    //     // Without Linesearch
    //     double alpha = 0.5;

    //     for(int i=0; i<num_pts; ++i)
    //         projs.row(i) += alpha* dp.segment<dim>(dim*i);
        
    //     std::cout<<"current pts: "<<projs<<std::endl;
    // }

    // Check if projection point is on IMLS surface
    std::cout<<"Original Points: "<<std::endl;
    std::cout<<ps<<std::endl;
    std::cout<<"Original phi values: "<<getphi(ps,test,index).transpose()<<std::endl;
    std::cout<<"Projected Points: "<<std::endl;
    std::cout<<projs<<std::endl;
    std::cout<<"Projection phi values: "<<getphi(projs,test,index).transpose()<<std::endl;
    return projs;
}

template<int dim>
Eigen::VectorXd FEMSolver<dim>::getKKTgrad(std::vector<Eigen::VectorXd>& lambdas, Eigen::MatrixXd ps, Eigen::MatrixXd qs, bool test)
{
    assert(ps.rows() == qs.rows());

    int dims = ps.rows();
    Eigen::MatrixXd gradphi = getgradphi(ps,test);
    Eigen::VectorXd gradO(dim*dims);
    for(int i=0; i<dims; ++i)
    {
        for(int j=0; j<dim; ++j)
            gradO(dim*i+j) = (ps(i,j)-qs(i,j));
    }

    // double n1 = gradO.norm();
    // double n2 = (gradphi.transpose()*lambdas[0]).norm();
    // double n3 = (gradphi.transpose()*lambdas[1]).norm();
    // double n4 = (lambdas[2]).norm();
    // double n5 = (lambdas[3]).norm();

    // Eigen::VectorXd res(1);
    // res<<n1+n2+n3+n4+n5;

    return gradO - gradphi.transpose()*lambdas[0] + gradphi.transpose()*lambdas[1] - lambdas[2] + lambdas[3];
    //return res;
}

template<int dim>
Eigen::MatrixXd FEMSolver<dim>::getQ(Eigen::MatrixXd qs, bool test)
{
    int dims = qs.rows();
    Eigen::MatrixXd Q(dim*dims,dim*dims);
    Q.setZero();
    for(int i=0; i<dim*dims; ++i)
        Q(i,i) = 1;
    return Q;
}

template<int dim>
Eigen::VectorXd FEMSolver<dim>::getc(Eigen::MatrixXd qs, bool test)
{
    int dims = qs.rows();
    Eigen::VectorXd c(dim*dims);
    c.setZero();
    for(int i=0; i<dims; ++i)
    {
        for(int j=0; j<dim; ++j)
            c(dim*i+j) = -qs(i,j);
    }
    return c;
}

template<int dim>
double FEMSolver<dim>::getcf(Eigen::MatrixXd qs, bool test)
{
    int dims = qs.rows();
    double cf = 0;
    for(int i=0; i<dims; ++i)
    {
        cf+= qs.row(i).dot(qs.row(i));
    }
    return 0.5*cf;
}

template<int dim>
Eigen::MatrixXd FEMSolver<dim>::getgradphi(Eigen::MatrixXd p0s, bool test, int index)
{
    // P0 is the current estimation of p
    int dims = p0s.rows();
    Eigen::MatrixXd A(dims,dim*dims);
    A.setZero();
    if(!test)
    {
        Eigen::VectorXd fs;
        std::vector<Eigen::VectorXd> dfsdps;

        evaluateImplicitPotentialKRGradient(p0s,dfsdps,fs,true,index);
        for(int i=0; i<dims; ++i)
        {
            for(int j=0; j<dim; ++j)
                A(i,dim*i+j) = dfsdps[i](j);
        }
    }
    else
    {
        if(TEST_FUNCTION == 0)
        {
            for(int i=0; i<dims; ++i)
            {   
                double x0,x1,x2;
                int num_root = gsl_poly_solve_cubic(0,-(p0s(i,1)-0.5),-0.5*p0s(i,0),&x0,&x1,&x2);
                //std::cout<<"Find "<<num_root<<" roots."<<std::endl;

                // Eigen::VectorXd par_hpar_p(2);
                // par_hpar_p(0) = -(x0-p0s(i,0));
                // par_hpar_p(1) = -(x0*x0-p0s(i,1));

                // Eigen::VectorXd par_hpar_x(2);
                // par_hpar_x = (x0-p0s(i,0)) + 2*(x0*x0-p0s(i,1))*x0;

                //std::cout<<"result x: "<<x0<<" value: "<< (x0-p0s(i,0)) + 2*(x0*x0-p0s(i,1))*x0<<std::endl;


                A(i,2*i) = -2*(x0-p0s(i,0));
                A(i,2*i+1) = -2*(x0*x0-p0s(i,1));
            }
        }
    }
    return A;
}

template<int dim>
Eigen::VectorXd FEMSolver<dim>::getphi(Eigen::MatrixXd p0s, bool test, int index)
{
    int dims = p0s.rows();
    Eigen::VectorXd b(dims);
    if(!test)
    {
        b.setZero();
        Eigen::VectorXd fs;
        std::vector<Eigen::VectorXd> dfsdps;

        evaluateImplicitPotentialKRGradient(p0s,dfsdps,fs,true,index);
        for(int i=0; i<dims; ++i)
        {
            b(i) =fs(i);
        }
    }
    else
    {
        if(TEST_FUNCTION == 0)
        {
            for(int i=0; i<dims; ++i)
            {   
                double x0,x1,x2;
                int num_root = gsl_poly_solve_cubic(0,-(p0s(i,1)-0.5),-0.5*p0s(i,0),&x0,&x1,&x2);
                //std::cout<<"Find "<<num_root<<" roots."<<std::endl;

                b(i) = pow((p0s(i,0)-x0),2)+pow((p0s(i,1)-x0*x0),2);
                //std::cout<<"closest point: "<<x0<<" "<<" distance: "<<b(i)<<std::endl;
            }
            
        }
    }
    return b;
}

bool compareEigenVectorXd(std::pair<Eigen::VectorXd,int> v1, std::pair<Eigen::VectorXd,int> v2)
{
    return (v1.first(0) < v2.first(0));
}

template<int dim>
void FEMSolver<dim>::SortedProjectionPoints2D()
{
    // For 2D only! Sort the Projection Points from left to right
    // Ideally this is done in a parametric way
    assert(dim == 2);
    std::vector<std::pair<Eigen::VectorXd,int>> temp;
    extendedAoC.clear();
    for(int i=0; i<projectedPts.rows(); i++)
    {
        
        if(IMLS_BOTH && i>=slave_nodes.size())
        {
            temp.push_back({projectedPts.row(i),master_nodes[i-slave_nodes.size()]});
        }
        else
            temp.push_back({projectedPts.row(i),slave_nodes[i]});
    }
    std::sort(temp.begin(),temp.end(),compareEigenVectorXd);
    double prev = 0;
    double next = 0;
    for(int i=0; i<projectedPts.rows(); i++)
    {
        projectedPts.row(i) = temp[i].first;
        if(i>0)
        {
            next = (projectedPts.row(i)-projectedPts.row(i-1)).norm();
            extendedAoC[temp[i-1].second] = (prev+next)/2.0;
        }
        prev = next;
    }
    extendedAoC[temp.back().second] = prev/2.0;

    double val1 = extendedAoC[14];
    double val2 = extendedAoC[10];

    extendedAoC[14] = val1/2.0;
    extendedAoC[9] = val1/2.0;
    extendedAoC[10] = val2/2.0;
    extendedAoC[5] = val2/2.0;

    for(auto it = extendedAoC.begin(); it != extendedAoC.end(); ++it)
    {
        std::cout<<it->first<<" "<<it->second<<std::endl;
    }

}

template<int dim>
void FEMSolver<dim>::IMLSProjectionTest()
{
    // Find the projection of point (0,1) on parabola y = x^2
    Eigen::MatrixXd ps(1,2);
    ps(0,0) = 0.1; ps(0,1) = 0.1;

    // Eigen::MatrixXd QD(2,2);
    // QD.setZero();
    // QD(0,0) = 1; QD(1,1) = 1;

    // Eigen::VectorXd c(2);
    // c<<0,0;

    // Eigen::MatrixXd AD(1,2);
    // AD<<-0.0412268,0.170905;

    // Eigen::VectorXd lc(1);
    // lc<<-0.00772703;
    // Eigen::VectorXd uc(1);
    // uc<<-0.00772703;

    // Eigen::VectorXd lx(2);
    // lx<<-1000,-1000; 
    // Eigen::VectorXd ux(2);
    // ux<<1000,1000;

    // StiffnessMatrix Q = QD.sparseView();
    // StiffnessMatrix A = AD.sparseView();

    // std::vector<Eigen::VectorXd> lamdas;
    // igl::mosek::MosekData mosek_data;
    // Eigen::VectorXd x;
    // bool solve_success = igl::mosek::mosek_quadprog(Q, c, 0.0, A, lc, lc, lx, ux, mosek_data, x, lamdas);

    // std::cout<<x<<std::endl;

    // Eigen::MatrixXd proj = SinglePointProjection(ps,true);
    // std::cout<<proj<<std::endl;


}

template class FEMSolver<2>;
template class FEMSolver<3>;