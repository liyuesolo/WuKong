#include "../include/FEMSolver.h"
#include <Eigen/SparseCholesky>
#include <Eigen/CholmodSupport>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

using namespace Spectra;

template<int dim>
void FEMSolver<dim>::calculateR()
{
    // if(MODE != 3) std::cerr<<"Please use Mode 3"<<std::endl;

    // Eigen::MatrixXd left(left_boundaries.size(),2);

    // for(int j=0; j<left.rows(); ++j)
    // {
    //     left.row(j) = deformed.segment<2>(left_boundaries[j] * 2);
    // }

    // double x_left = 0.0;
    // for(int i=0; i<left.rows(); ++i)
    // {
    //     x_left += left(i,0);
    // }
    // x_left /= left.rows();

    // R = x_left;

    // // Derivatives
    // dRdx.setZero(1,num_nodes*2);

    // for(int i=0; i<left_boundaries.size(); ++i)
    // {
    //     dRdx(0, 2*left_boundaries[i]) = 1.0/double(left.rows());
    // }

    if(SLIDING_TEST)
    {
        int index = 2*(sliding_res+1);
        double x = deformed(dim*index);
        double y = deformed(dim*index+1);
        R = -atan2(y, x);

        dRdx.setZero(1,num_nodes*2);
        dRdx(0,2*index) = -sliding_stiffness*r2*sin(theta2);
        dRdx(0,2*index+1) = -sliding_stiffness*r2*cos(theta2);
    }
    else
    {
        // Hard code for R: it is the right boundary of the slave nodes
        R = deformed(2*16);

        // Derivatives of external force with respect to f
        dRdx.setZero(1,num_nodes*2);
        dRdx(0,2*17+1) = -0.25;
        dRdx(0,2*18+1) = -0.25;
    }

    

}

template<int dim>
void FEMSolver<dim>::calculatedfdp()
{
    dgdp.setZero(2*num_nodes,1);

    // Eigen::MatrixXd left(left_boundaries.size(),2);
    // for(int j=0; j<left.rows(); ++j)
    // {
    //     left.row(j) = deformed.segment<2>(left_boundaries[j] * 2);
    // }

    // double x_left = 0.0;
    // for(int i=0; i<left.rows(); ++i)
    // {
    //     x_left += left(i,0);
    // }
    // x_left /= left.rows();

    // Eigen::VectorXd dx_leftdx(num_nodes*2);
    // dx_leftdx.setZero();
    // for(int i=0; i<left.rows(); ++i)
    // {
    //     dx_leftdx(2*left_boundaries[i]) = 1.0/double(left.rows());
    // }

    // dgdp = x_left*dx_leftdx;

    if(SLIDING_TEST)
    {
        int index = 2*(sliding_res+1);
        double x = deformed(dim*index);
        double y = deformed(dim*index+1);
        dgdp(2*index,0) = y/(x*x+y*y);
        dgdp(2*index+1,0) = -x/(x*x+y*y);
    }
    else
        dgdp(2*16,0) = 1;

    std::cout<<"calculatedfdp done"<<std::endl;
}

template<int dim>
void FEMSolver<dim>::checkDerivative()
{
    double eps = 1e-5;

    ForwardSim();
    double r = R;
    double d = dRdp(0);

    theta2 += eps;
    ForwardSim();
    std::cout<<"Analytic and Numerical Derivatives: "<<d<<" "<<(R-r)/eps<<std::endl;
    theta2 -= eps;
}

template<int dim>
void FEMSolver<dim>::ResetSim()
{
    initializeElementData(V_all, F_all, F_all, F_all_Quad, F_all_Quad);

    if(!SLIDING_TEST)
    {
        if(USE_FROM_STL)
        {
            addNeumannBCFromSTL();
            addDirichletBCFromSTL();
        }
        else
        {
            addNeumannBC();
            addDirichletBC();
        }

        if(TEST)
        {
            use_pos_penalty = true;
            use_rotational_penalty = false;
        }
    }else
    {
        addDirichletBC();
    }
    
}

template<int dim>
void FEMSolver<dim>::ForwardSim()
{
    // Reset the syetem
    ResetSim();
    
    // Do forward simulation here
    staticSolve();

    // Calculate targeted value R
    calculateR();

    // Calculate all useful derivatives
    calculatedfdp();

    H.resize(2*num_nodes, 2*num_nodes);
    H.setZero();
    buildSystemMatrix(u, H);

    dRdp.resize(1,1);
    Eigen::SparseLU<StiffnessMatrix> solver;
    Eigen::MatrixXd dRdxT = dRdx.transpose();
    solver.compute(H.transpose());
    dRdp = (solver.solve(dRdxT)).transpose()*dgdp;

    // std::cout<<H.transpose()*solver.solve(dRdxT)<<std::endl;
    // std::cout<<dRdxT<<std::endl;

    // std::cout<<solver.solve(dRdxT)<<std::endl;
    // std::cout<<"_____________________________"<<std::endl;
    // std::cout<<dgdp<<std::endl;
    // std::cout<<"_____________________________"<<std::endl;

    //std::cout<<dRdp<<std::endl;
}

template<int dim>
void FEMSolver<dim>::InverseDesign(double& tar, double& opt_p)
{
    // tar:: target value; opt_p: optimized parameter

    double l1 = 10000;

    calculateR();
    double O_init = 0.5*l1*(R-tar)*(R-tar);

    std::cout<<"Initial Objective: "<<O_init<<std::endl;

    double max_iter = 100;
    double max_iter_2 = 10;
    int iter = 0;
    double O = O_init;
    double th = 1e-6;
    double O_prev = O_init;
    double theta_prev = theta2;
    double F_prev = FORCE;
    std::ofstream myfile;
    myfile.open ("example2.txt");
    std::ofstream myfile2;
    myfile2.open ("example2.csv");
    

    while(O_prev>th && iter < max_iter)
    {
        //FORCE = F_prev;
        theta2 = theta_prev;
        theta1 = 0.1;
        ForwardSim();

        
        double F_cur = F_prev;
        double theta_cur;
        int iter2 = 0;
        double grad = l1 * (R-tar) * dRdp(0);
        double alpha = 100.0;
        while(true)
        {
            // F_cur = F_prev - alpha * l1 * (R-tar) * dRdp(0);
            // FORCE = F_prev;
           
            theta_cur = theta_prev - alpha * grad;
            while(theta_cur < 0.05*M_PI || theta_cur>0.45*M_PI)
            {
                alpha/=2;
                theta_cur = theta_prev - alpha * grad;
            }
            myfile<<"theta cur: "<<theta_cur/M_PI<<" alpha: "<<alpha<<std::endl;
            
            theta2 = theta_cur;
            theta1 = 0.1;
            ForwardSim();

            double O_cur = 0.5* l1 * (R-tar) * (R-tar);
            myfile<<"theta2: "<<theta2/M_PI<<" R: "<<R/M_PI<<" O_prev: "<<O_prev<<" O_cur: "<<O_cur<<" Gradient: "<<alpha * l1 * dRdp.norm()<<std::endl;
            if(O_cur < O_prev || iter2 > max_iter_2)
            {
                std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
                std::cout<<"Iter: "<<iter<<" Line Search Iter: "<<iter2<<" Objective: "<<O_cur<<" gradient norm: "<<alpha * l1 * dRdp.norm()<<std::endl;
                std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
                myfile<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
                myfile<<"Iter: "<<iter<<" Line Search Iter: "<<iter2<<" Objective: "<<O_cur<<" gradient norm: "<<alpha * l1 * dRdp.norm()<<" theta2: "<<theta_cur/M_PI<<std::endl;
                myfile<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
                myfile2<<iter<<","<<O_cur<<std::endl;
                O_prev = O_cur;
                //F_prev = F_cur;
                theta_prev = theta_cur;
                break;
            }
            else
                alpha /= 5;
            iter2++;
        }
        iter++;
    }
    myfile.close();
    myfile2.close();
    std::cout<<"Initial Objevtive: "<<O_init<<" Final Objective: "<<O_prev<<" and Final Theta: "<<theta2<<std::endl;
}

template<int dim>
void FEMSolver<dim>::eigenAnalysis(StiffnessMatrix& K)
{
    Eigen::SparseMatrix<double> M(K.rows(), K.rows());
    for(int i=0; i<K.rows(); ++i)
    {
        for(int j=0; j<K.rows(); ++j)
        {
            M.insert(i,j) = K.coeff(i,j);
        }
    }
    Spectra::SparseGenMatProd<T> op(M);
    int n_eigen = 20;
    GenEigsSolver<SparseGenMatProd<T>> eigs(op, n_eigen, K.rows());

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(SortRule::SmallestMagn);

    // Retrieve results
    Eigen::VectorXcd evalues;
    if (eigs.info() == CompInfo::Successful)
    {
        Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
        Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
        std::cout << eigen_values << std::endl;
        std::ofstream out("eigen_vectors.txt");
        out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
        for (int i = 0; i < eigen_vectors.cols(); i++)
            out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
        out << std::endl;
        for (int i = 0; i < eigen_vectors.rows(); i++)
        {
            // for (int j = 0; j < eigen_vectors.cols(); j++)
            for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                out << eigen_vectors(i, j) << " ";
            out << std::endl;
        }       
        out << std::endl;
        out.close();
    }
}

template<int dim>
void FEMSolver<dim>::checkHessianPD(bool save_txt)
{
    int nmodes = 10;
    int n_dof_sim = deformed.rows();
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    buildSystemMatrix(u, d2edx2);
    bool use_Spectra = true;

    Eigen::SimplicialLLT<StiffnessMatrix> solver;
    solver.analyzePattern(d2edx2); 
    solver.factorize(d2edx2);
    bool indefinite = false;
    if (solver.info() == Eigen::NumericalIssue)
    {
        std::cout << "!!!indefinite matrix!!!" << std::endl;
        indefinite = true;
        // saveStates("indefinite_state.obj");
        // std::exit(0);
    }

    // if (use_Spectra)
    // {
    //     Eigen::SparseMatrix<double> M(d2edx2.rows(), d2edx2.rows());
    //     for(int i=0; i<d2edx2.rows(); ++i)
    //     {
    //         for(int j=0; j<d2edx2.rows(); ++j)
    //         {
    //             M.insert(i,j) = d2edx2.coeff(i,j);
    //         }
    //     }

    //     Spectra::SparseSymShiftSolve<double, Eigen::Upper> op(M);

    //     //0 cannot cannot be used as a shift
    //     T shift = indefinite ? -1e5 : -1e-5;
    //     Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double, Eigen::Upper>> eigs(op, nmodes, 2 * nmodes, shift);
    //     // Spectra::SparseGenMatProd<T> op(M);
    //     // GenEigsSolver<SparseGenMatProd<T>> eigs(op, nmodes, d2edx2.rows());
    //     eigs.init();

    //     int nconv = eigs.compute(SortRule::LargestMagn);

    //     if (eigs.info() == CompInfo::Successful)
    //     {
    //         Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
    //         Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
    //         std::cout << eigen_values.transpose() << std::endl;
    //         if (save_txt)
    //         {
    //             std::ofstream out("cell_eigen_vectors_2d.txt");
    //             out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
    //             for (int i = 0; i < eigen_vectors.cols(); i++)
    //                 out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
    //             out << std::endl;
    //             for (int i = 0; i < eigen_vectors.rows(); i++)
    //             {
    //                 // for (int j = 0; j < eigen_vectors.cols(); j++)
    //                 for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
    //                     out << eigen_vectors(i, j) << " ";
    //                 out << std::endl;
    //             }       
    //             out << std::endl;
    //             out.close();
    //         }
    //     }
    //     else
    //     {
    //         std::cout << "Eigen decomposition failed" << std::endl;
    //     }
    // }
}

template class FEMSolver<2>;
template class FEMSolver<3>;