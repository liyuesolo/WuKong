#include <fstream>
#include <igl/triangle/triangulate.h>

#include "ShellFEMSolver.h"
#include <Eigen/PardisoSupport>
#include <iomanip>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <igl/readOBJ.h>

#include "autodiff/CST3DShell.h"



template <typename Scalar, int size>
void makePD(Eigen::Matrix<Scalar, size, size>& symMtr)
{
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
	if (eigenSolver.eigenvalues()[0] >= 0.0) {
		return;
	}
	Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
	int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
	for (int i = 0; i < rows; i++) {
		if (D.diagonal()[i] < 0.0) {
			D.diagonal()[i] = 0.0;
		}
		else {
			break;
		}
	}
	symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

template<class T, int dim>
void ShellFEMSolver<T, dim>::computeEigenMode()
{
    int nmodes = 15;

    StiffnessMatrix K(deformed.rows(), deformed.rows());
    run_diff_test = true;
    buildSystemMatrix(u, K);
    
    bool use_Spectra = true;

    if (use_Spectra)
    {

        Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(K);

        //0 cannot cannot be used as a shift
        T shift = -0.1;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values << std::endl;
            std::ofstream out("bead_eigen_vectors.txt");
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
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
    else
    {
        Eigen::SelfAdjointEigenSolver<StiffnessMatrix> eigs(K);
        Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().block(0, 0, K.rows(), nmodes);
        
        Eigen::VectorXd eigen_values = eigs.eigenvalues().segment(0, nmodes);
        std::cout << eigen_values << std::endl;
        std::ofstream out("bead_eigen_vectors.txt");
        out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
        for (int i = 0; i < eigen_vectors.rows(); i++)
        {
            for (int j = 0; j < eigen_vectors.cols(); j++)
                out << eigen_vectors(i, j) << " ";
            out << std::endl;
        }       
        out << std::endl;
        out.close();
    }

}


// from Simon Duenser
template<class T, int dim>
void ShellFEMSolver<T, dim>::buildHingeStructure()
{
    struct Hinge
	{
		Hinge()
		{
			for (int i = 0; i < 2; i++)
			{
				edge[i] = -1;
				flaps[i] = -1;
				tris[i] = -1;
			}
		}
		int edge[2];
		int flaps[2];
		int tris[2];
	};
	
	std::vector<Hinge> hinges_temp;
	
	hinges_temp.clear();
	std::map<std::pair<int, int>, int> edge2index;
	for (int i = 0; i < faces.size() / 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int i1 = faces(3 * i + j);
			int i2 = faces(3 * i + (j + 1) % 3);
			int i1t = i1;
			int i2t = i2;
			bool swapped = false;
			if (i1t > i2t)
			{
				std::swap(i1t, i2t);
				swapped = true;
			}
			
			auto ei = std::make_pair(i1t, i2t);
			auto ite = edge2index.find(ei);
			if (ite == edge2index.end())
			{
				//insert new hinge
				edge2index[ei] = hinges_temp.size();
				hinges_temp.push_back(Hinge());
				Hinge& hinge = hinges_temp.back();
				hinge.edge[0] = i1t;
				hinge.edge[1] = i2t;
				int itmp = swapped ? 1 : 0;
				hinge.tris[itmp] = i;
				hinge.flaps[itmp] = faces(3 * i + (j + 2) % 3);
			}
			else
			{
				//hinge for this edge already exists, add missing information for this triangle
				Hinge& hinge = hinges_temp[ite->second];
				int itmp = swapped ? 1 : 0;
				hinge.tris[itmp] = i;
				hinge.flaps[itmp] = faces(3 * i + (j + 2) % 3);
			}
		}
	}
	//ordering for edges
	
	hinges.resize(hinges_temp.size(), Eigen::NoChange);
	int ii = 0;
	/*
      auto diff code takes
           x3
         /   \
        x2---x1
         \   /
           x0	

      hinge is 
           x2
         /   \
        x0---x1
         \   /
           x3	
    */
    for(Hinge & hinge : hinges_temp) {
		if ((hinge.tris[0] == -1) || (hinge.tris[1] == -1)) {
			continue; //skip boundary edges
		}
		hinges(ii, 2) = hinge.edge[0]; //x0
		hinges(ii, 1) = hinge.edge[1]; //x1
		hinges(ii, 3) = hinge.flaps[0]; //x2
		hinges(ii, 0) = hinge.flaps[1]; //x3
		++ii;
	}
	hinges.conservativeResize(ii, Eigen::NoChange);
}

template<class T, int dim>
void ShellFEMSolver<T, dim>::computeRestShape()
{
    Xinv.resize(nFaces());
    rest_area = VectorXT::Ones(nFaces());
    thickness = VectorXT::Ones(nFaces()) * 0.003;
    
    iterateFaceSerial([&](int i){
        TV E[2]; // edges
        FaceVtx vertices = getFaceVtxUndeformed(i);
        E[0] = vertices.row(1) - vertices.row(0);
        E[1] = vertices.row(2) - vertices.row(0);
        
        TV N = E[0].cross(E[1]);
        T m_A0 = N.norm()*0.5;
        rest_area[i] = m_A0;
        
        // // compute rest
        TV B2D[2];
        B2D[0] = E[0].normalized();
        TV n = B2D[0].cross(E[1]);
        B2D[1] = E[0].cross(n);
        B2D[1] = B2D[1].normalized();
        
        
        Eigen::Matrix2d MatE2D(2, 2);
        MatE2D(0, 0) = E[0].dot(B2D[0]);
        MatE2D(1, 0) = E[0].dot(B2D[1]);
        MatE2D(0, 1) = E[1].dot(B2D[0]);
        MatE2D(1, 1) = E[1].dot(B2D[1]);

        Eigen::Matrix2d Einv = MatE2D.inverse();
        Xinv[i] = Einv;
        // std::cout << Einv << std::endl;
        // std::cout << N.transpose() << std::endl;
        // std::getchar();
    });
}

template<class T, int dim>
T ShellFEMSolver<T, dim>::computeElasticPotential(const VectorXT& _u)
{
    T energy = 0.0;
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;
    if constexpr (dim == 3)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

            FaceIdx indices = faces.segment<3>(face_idx * 3);
            
            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            TV X0 = undeformed_vertices.row(0); 
            TV X1 = undeformed_vertices.row(1); 
            TV X2 = undeformed_vertices.row(2);

            T k_s = E * thickness[0] / (1.0 - nu * nu);

            energy += compute3DCSTShellEnergy(nu, k_s, x0, x1, x2, X0, X1, X2);
        });

        iterateHingeSerial([&](const HingeIdx& hinge_idx){
            if (!add_bending)
                return;
            HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
            HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);

            T k_bend = E * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));

            TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

            energy += computeDSBendingEnergy(k_bend, x0, x1, x2, x3, X0, X1, X2, X3);
        });
    }
    return energy;
}

template<class T, int dim>
T ShellFEMSolver<T, dim>::computeTotalEnergy(const VectorXT& _u)
{
    T energy = 0.0;
    if constexpr (dim == 3)
    {
        VectorXT projected = _u;
        if (!run_diff_test)
        {
            iterateDirichletDoF([&](int offset, T target)
            {
                projected[offset] = target;
            });
        }
        deformed = undeformed + projected;

        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

            FaceIdx indices = faces.segment<3>(face_idx * 3);
            // std::cout << "face# " << face_idx << " nodes: " << indices.transpose()<<std::endl;


            T m_Einv[2][2];
            m_Einv[0][0] = Xinv[face_idx](0, 0);
            m_Einv[0][1] = Xinv[face_idx](0, 1);
            m_Einv[1][0] = Xinv[face_idx](1, 0);
            m_Einv[1][1] = Xinv[face_idx](1, 1);

            T m_A0 = rest_area[face_idx];
            T m_h = thickness[face_idx];

            std::array<TV, 3> x;
            x[0] = vertices.row(0);
            x[1] = vertices.row(1);
            x[2] = vertices.row(2);
            
            
            // T Wdmy[1];
            // #include "autodiff/CST3D_StVK_W.mcg"
            
            // energy += Wdmy[0];
            
            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            TV X0 = undeformed_vertices.row(0); 
            TV X1 = undeformed_vertices.row(1); 
            TV X2 = undeformed_vertices.row(2);

            T k_s = E * m_h / (1.0 - nu * nu);

            energy += compute3DCSTShellEnergy(nu, k_s, x0, x1, x2, X0, X1, X2);

            if (gravitional_energy)
                energy += compute3DCSTGravitationalEnergy(density, m_h, gravity, x0, x1, x2, X0, X1, X2);
            

        });

        iterateHingeSerial([&](const HingeIdx& hinge_idx){
            if (!add_bending)
                return;
            HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
            HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);

            std::array<TV, 4> x, X;
            x[0] = deformed_vertices.row(0);
            x[1] = deformed_vertices.row(1);
            x[2] = deformed_vertices.row(2);
            x[3] = deformed_vertices.row(3);

            X[0] = undeformed_vertices.row(0);
            X[1] = undeformed_vertices.row(1);
            X[2] = undeformed_vertices.row(2);
            X[3] = undeformed_vertices.row(3);

            T k_bend = E * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));

            // T Wdmy[1];
            // #include "autodiff/DSBending_W.mcg"
            // // maple end
            // energy += Wdmy[0];

            TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

            energy += computeDSBendingEnergy(k_bend, x0, x1, x2, x3, X0, X1, X2, X3);

        });
        
        energy -= _u.dot(f);
        // std::cout << "u dot f" << _u.dot(f) << std::endl;
    }
    return energy;
}

template<class T, int dim>
void ShellFEMSolver<T, dim>::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

template<class T, int dim>
void ShellFEMSolver<T, dim>::computeInternalForce(const VectorXT& _u, VectorXT& dPsidu)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    dPsidu = _u;
    dPsidu.setZero();

    if constexpr (dim == 3)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
            FaceIdx indices = faces.segment<3>(face_idx * 3);

            T k_s = E * thickness[0] / (1.0 - nu * nu);
            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
            
            
            Vector<T, 9> dedx;
            compute3DCSTShellEnergyGradient(nu, k_s, x0, x1, x2, X0, X1, X2, dedx);
            dedx *= -1.0;

            for (int i = 0; i < 3; i++)
                for (int d = 0; d < dim; d++)
                    dPsidu[indices[i] * dim + d] += dedx[i * dim + d];
        });

        iterateHingeSerial([&](const HingeIdx& hinge_idx){
            if (!add_bending)
                return;

            HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
            HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);


            T k_bend = E * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));

            Vector<T, 12> dedx;
            
            TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

            computeDSBendingEnergyGradient(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, dedx);
            dedx *= -1.0;

            for (int i = 0; i < 4; i++)
                for (int d = 0; d < dim; d++)
                    dPsidu[hinge_idx[i] * dim + d] += dedx[i * dim + d];
        });                
    }
}

template<class T, int dim>
T ShellFEMSolver<T, dim>::computeResidual(const VectorXT& _u,
                                        VectorXT& residual)
{
    VectorXT projected = _u;

    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }

    deformed = undeformed + projected;

    residual = f;
    
    if constexpr (dim == 3)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);
            FaceIdx indices = faces.segment<3>(face_idx * 3);
            // std::cout << "face# " << face_idx << " nodes: " << indices.transpose()<<std::endl;
            // std::cout << "deformed" << std::endl;
            // std::cout << vertices << std::endl;
            // std::cout << "undeformed" << std::endl;
            // std::cout << undeformed_vertices << std::endl;
            T m_Einv[2][2];
            m_Einv[0][0] = Xinv[face_idx](0, 0);
            m_Einv[0][1] = Xinv[face_idx](0, 1);
            m_Einv[1][0] = Xinv[face_idx](1, 0);
            m_Einv[1][1] = Xinv[face_idx](1, 1);

            T m_A0 = rest_area[face_idx];
            T m_h = thickness[face_idx];

            std::array<TV, 3> x, X;
            x[0] = vertices.row(0);
            x[1] = vertices.row(1);
            x[2] = vertices.row(2);

            X[0] = undeformed_vertices.row(0);
            X[1] = undeformed_vertices.row(1);
            X[2] = undeformed_vertices.row(2);
            
            // Matrix<T, 3, 2> temp;

            // temp.col(0) = x[1] - x[0];
            // temp.col(1) = x[2] - x[0];

            // Matrix<T, 3, 2> F = temp * Xinv[face_idx];

            // std::cout << F << std::endl;
            // std::cout << (F.transpose() * F - TM2::Identity()).norm() << std::endl;
            // std::getchar();

            
            // Vector<T, 9> dedx_maple;
            
            // #include "autodiff/CST3D_StVK_f.mcg"
            // std::cout << "\t dedx: " << -dedx_maple.transpose() << std::endl;

            T k_s = E * m_h / (1.0 - nu * nu);
            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
            
            
            Vector<T, 9> dedx;
            compute3DCSTShellEnergyGradient(nu, k_s, x0, x1, x2, X0, X1, X2, dedx);
            // std::cout << dedx.transpose()<<std::endl;
            // std::getchar();
            dedx *= -1.0;
            
            if (gravitional_energy)
            {
                Vector<T, 9> graviational_force;
                compute3DCSTGravitationalEnergyGradient(density, m_h, gravity, x0, x1, x2, X0, X1, X2, graviational_force);
                graviational_force *= -1;
                for (int i = 0; i < 3; i++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        residual[indices[i] * dim + d] += graviational_force[i * dim + d];
                    }   
                }
            }
            
            
            for (int i = 0; i < 3; i++)
            {
                for (int d = 0; d < dim; d++)
                {
                    residual[indices[i] * dim + d] += dedx[i * dim + d];
                }   
            }
        });

        
        iterateHingeSerial([&](const HingeIdx& hinge_idx){
            if (!add_bending)
                return;
                    
            HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
            HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);

            std::array<TV, 4> x, X;
            x[0] = deformed_vertices.row(0);
            x[1] = deformed_vertices.row(1);
            x[2] = deformed_vertices.row(2);
            x[3] = deformed_vertices.row(3);

            X[0] = undeformed_vertices.row(0);
            X[1] = undeformed_vertices.row(1);
            X[2] = undeformed_vertices.row(2);
            X[3] = undeformed_vertices.row(3);

            T k_bend = E * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));

            Vector<T, 12> fx;

            // #include "autodiff/DSBending_fx.mcg"
            
            TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

            computeDSBendingEnergyGradient(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, fx);
            // std::cout << fx.norm() << std::endl;
            // std::cout << fx.transpose() << std::endl;
            // std::cout << computeDSBendingEnergy(k_bend, x0, x1, x2, x3, X0, X1, X2, X3) << std::endl;
            // std::cout << hinge_idx.transpose() << std::endl;
            // std::getchar();
            fx *= -1;

            for (int i = 0; i < 4; i++)
            {
                for (int d = 0; d < dim; d++)
                {
                    residual[hinge_idx[i] * dim + d] += fx[i * dim + d];
                }
            }
        });

    }
    // std::cout << residual.norm() << std::endl;
    // std::getchar();
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });
    
    return residual.norm();
}

template<class T, int dim>
void ShellFEMSolver<T, dim>::computedfdX(const VectorXT& _u, StiffnessMatrix& dfdX)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;
    
    std::vector<Entry> entries;
    if constexpr (dim == 3)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

            FaceIdx indices = faces.segment<3>(face_idx * 3);

            T m_h = thickness[0];

            T k_s = E * m_h / (1.0 - nu * nu);

            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);

            std::array<TV, 3> x, X;
            x[0] = vertices.row(0);
            x[1] = vertices.row(1);
            x[2] = vertices.row(2);

            X[0] = undeformed_vertices.row(0);
            X[1] = undeformed_vertices.row(1);
            X[2] = undeformed_vertices.row(2);

            Matrix<T, 9, 9> dfdX_element;
            compute3DCSTShellEnergydfdX(nu, k_s, x0, x1, x2, X0, X1, X2, dfdX_element);
            // std::cout << "dfdX element" << std::endl;
            // std::cout << dfdX_element << std::endl;

            // T epsilon = 1e-7;


            // // std::cout << "g0: " << g0.transpose() << std::endl;
            // for (int d = 0; d < dim; d++)
            // {
            //     X0[d] += epsilon;
            //     Vector<T, 9> g0;
            //     compute3DCSTShellEnergyGradient(nu, k_s, x0, x1, x2, X0, X1, X2, g0);
            //     g0 *= -1;
            //     X0[d] -= 2.0 * epsilon;
            //     Vector<T, 9> g1;
            //     compute3DCSTShellEnergyGradient(nu, k_s, x0, x1, x2, X0, X1, X2, g1);
            //     g1 *= -1;
            //     X0[d] += epsilon;
            //     VectorXT row_FD = (g1 - g0) / (epsilon) * 0.5;
            //     for(int i = 0; i < 9; i++)
            //     {
            //         if(dfdX_element.coeff(i, d) == 0 && row_FD(i) == 0)
            //             continue;
            //         std::cout << "dfdX(" << i << ", " << d << ") " << " FD: " <<  row_FD(i) << " symbolic: " << dfdX_element.coeff(i, d) << std::endl;
            //         std::getchar();
            //     }
            // }
            // Vector<T, 9> f0 = g0;
            // {
            //     VectorXT dx(9);
            //     dx.setRandom();
            //     dx *= 1.0 / dx.norm();
            //     dx *= 0.001;
            //     T previous = 0.0;
            //     for (int i = 0; i < 10; i++)
            //     {
            //         Vector<T, 9> f1;
            //         for (int d = 0; d < dim; d++)
            //         {
            //             X0[d] = undeformed_vertices.row(0)[d] + dx[d];
            //             X1[d] = undeformed_vertices.row(1)[d] + dx[dim + d];
            //             X2[d] = undeformed_vertices.row(2)[d] + dx[2 * dim + d];
            //         }
                    
            //         compute3DCSTShellEnergyGradient(nu, k_s, x0, x1, x2, X0, X1, X2, f1);
            //         f1 *= -1;
            //         T df_norm = (f0 + (dfdX_element * dx) - f1).norm();
            //         std::cout << "df_norm " << df_norm << std::endl;
            //         if (i > 0)
            //         {
            //             std::cout << (previous/df_norm) << std::endl;
            //         }
            //         previous = df_norm;
            //         dx *= 0.5;
            //     }
            // }
            std::exit(0);
            
            
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        for (int dd = 0; dd < dim; dd++)
                        {
                            entries.push_back(Entry(indices[i] * dim + d, indices[j] * dim + dd, dfdX_element(i * dim + d, j * dim + dd)));
                        }   
                    } 
                }
            }


            if (gravitional_energy)
            {
                Matrix<T, 9, 9> g_J;
                compute3DCSTGravitationalEnergydfdX(density, m_h, gravity, x0, x1, x2, X0, X1, X2, g_J);
                
                // T epsilon = 1e-7;

                // Vector<T, 9> g0;
                // compute3DCSTGravitationalEnergyGradient(density, m_h, gravity, x0, x1, x2, X0, X1, X2, g0);
                // g0 *= -1;
                
                // for (int d = 0; d < dim; d++)
                // {
                //     X0[d] += epsilon;
                //     Vector<T, 9> g1;
                //     compute3DCSTGravitationalEnergyGradient(density, m_h, gravity, x0, x1, x2, X0, X1, X2, g1);
                //     g1 *= -1;
                    
                //     X0[d] -= epsilon;
                //     VectorXT row_FD = (g1 - g0) / (epsilon);
                //     for(int i = 0; i < 9; i++)
                //     {
                //         if(g_J.coeff(i, d) == 0 && row_FD(i) == 0)
                //             continue;
                //         std::cout << "dfdX(" << i << ", " << d << ") " << " FD: " <<  row_FD(i) << " symbolic: " << g_J.coeff(i, d) << std::endl;
                //         std::getchar();
                //     }
                // }
                // std::exit(0);

                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        for (int d = 0; d < dim; d++)
                            for (int dd = 0; dd < dim; dd++)
                                entries.push_back(Entry(indices[i] * dim + d, indices[j] * dim + dd, g_J(i * dim + d, j * dim + dd)));
            }
        });

        iterateHingeSerial([&](const HingeIdx& hinge_idx){
            if (!add_bending)
                return;
            HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
            HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);

            T k_bend = E * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));
            
            Matrix<T, 12, 12> dfdX_element;
            TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

            computeDSBendingEnergydfdX(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, dfdX_element);

            // T epsilon = 1e-7;

            // Vector<T, 12> g0;
            // computeDSBendingEnergyGradient(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, g0);
            // g0 *= -1;
            
            // for (int d = 0; d < dim; d++)
            // {
            //     X1[d] += epsilon;
            //     Vector<T, 12> g1;
            //     computeDSBendingEnergyGradient(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, g1);
            //     g1 *= -1;
                
            //     X1[d] -= epsilon;
            //     VectorXT row_FD = (g1 - g0) / (epsilon);
            //     for(int i = 0; i < 12; i++)
            //     {
            //         if(dfdX_element.coeff(i, dim + d) == 0 && row_FD(i) == 0)
            //             continue;
            //         std::cout << "dfdX(" << i << ", " << d << ") " << " FD: " <<  row_FD(i) << " symbolic: " << dfdX_element.coeff(i, dim + d) << std::endl;
            //         std::getchar();
            //     }
            // }
            // std::exit(0);

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    for (int d = 0; d < dim; d++)
                        for (int dd = 0; dd < dim; dd++)
                            entries.push_back(Entry(hinge_idx[i] * dim + d, hinge_idx[j] * dim + dd, dfdX_element(i * dim + d, j * dim + dd)));
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

        });
    }

    dfdX.setFromTriplets(entries.begin(), entries.end());
    dfdX.makeCompressed();
}

template<class T, int dim>
void ShellFEMSolver<T, dim>::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;
    
    std::vector<Entry> entries;
    if constexpr (dim == 3)
    {
        iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

            FaceIdx indices = faces.segment<3>(face_idx * 3);
            T m_Einv[2][2];
            m_Einv[0][0] = Xinv[face_idx](0, 0);
            m_Einv[0][1] = Xinv[face_idx](0, 1);
            m_Einv[1][0] = Xinv[face_idx](1, 0);
            m_Einv[1][1] = Xinv[face_idx](1, 1);

            T m_A0 = rest_area[face_idx];
            T m_h = thickness[face_idx];

            std::array<TV, 3> x, X;
            x[0] = vertices.row(0);
            x[1] = vertices.row(1);
            x[2] = vertices.row(2);

            X[0] = undeformed_vertices.row(0);
            X[1] = undeformed_vertices.row(1);
            X[2] = undeformed_vertices.row(2);

            
            // T J[9][9];
    		// memset(J, 0, sizeof(J));

            // #include "autodiff/CST3D_StVK_J.mcg"

            // Matrix<T, 9, 9> hess = -Eigen::Map<Eigen::Matrix<double, 9, 9>>(J[0]);

            T k_s = E * m_h / (1.0 - nu * nu);

            TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);

            Matrix<T, 9, 9> hess;
            compute3DCSTShellEnergyHessian(nu, k_s, x0, x1, x2, X0, X1, X2, hess);
            // makePD<T, 9>(hess);
            
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        for (int dd = 0; dd < dim; dd++)
                        {
                            // std::cout << "J(" << i << ", " << j << ") " << J[i * dim + d][j * dim + dd] << std::endl;
                            // entries.push_back(Entry(indices[i] * dim + d, indices[j] * dim + dd, -J[i * dim + d][j * dim + dd]));
                            entries.push_back(Entry(indices[i] * dim + d, indices[j] * dim + dd, hess(i * dim + d, j * dim + dd)));
                        }   
                    } 
                }
            }


            if (gravitional_energy)
            {
                Matrix<T, 9, 9> g_J;
                compute3DCSTGravitationalEnergyHessian(density, m_h, gravity, x0, x1, x2, X0, X1, X2, g_J);

                // // std::cout << J << std::endl;

                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        for (int d = 0; d < dim; d++)
                            for (int dd = 0; dd < dim; dd++)
                                entries.push_back(Entry(indices[i] * dim + d, indices[j] * dim + dd, g_J(i * dim + d, j * dim + dd)));
            }
        });

        iterateHingeSerial([&](const HingeIdx& hinge_idx){
            if (!add_bending)
                return;
            HingeVtx deformed_vertices = getHingeVtxDeformed(hinge_idx);
            HingeVtx undeformed_vertices = getHingeVtxUndeformed(hinge_idx);

            std::array<TV, 4> x, X;
            x[0] = deformed_vertices.row(0);
            x[1] = deformed_vertices.row(1);
            x[2] = deformed_vertices.row(2);
            x[3] = deformed_vertices.row(3);

            X[0] = undeformed_vertices.row(0);
            X[1] = undeformed_vertices.row(1);
            X[2] = undeformed_vertices.row(2);
            X[3] = undeformed_vertices.row(3);

            T k_bend = E * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));

            // maple
            // T Jx[12][12];
    		// memset(Jx, 0, sizeof(Jx));

            // #include "autodiff/DSBending_Jx.mcg"
            

            // Matrix<T, 12, 12> hess = -Eigen::Map<Eigen::Matrix<double, 12, 12>>(Jx[0]);
            
            // makePD<T, 12>(hess);
            
            Matrix<T, 12, 12> hess;
            TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

            computeDSBendingEnergyHessian(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, hess);
            

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        for (int dd = 0; dd < dim; dd++)
                        {
                            // entries.push_back(Entry(hinge_idx[i] * dim + d, hinge_idx[j] * dim + dd, -Jx[i * dim + d][j * dim + dd]));
                            entries.push_back(Entry(hinge_idx[i] * dim + d, hinge_idx[j] * dim + dd, hess(i * dim + d, j * dim + dd)));
                        }   
                    } 
                }
            }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

        });
    }

    K.setFromTriplets(entries.begin(), entries.end());

    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();

}

template<class T, int dim>
bool ShellFEMSolver<T, dim>::linearSolve(StiffnessMatrix& K, 
    VectorXT& residual, VectorXT& du)
{
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;

    Eigen::PardisoLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
    
    T alpha = 10e-6;
    solver.analyzePattern(K);
    for (int i = 0; i < 50; i++)
    {
        // std::cout << i << std::endl;
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout << "indefinite" << std::endl;
            K = H + alpha * I;        
            alpha *= 10;
            continue;
        }
        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        // VectorXT d_vector = solver.vectorD();
        int num_negative_eigen_values = 0;

        // for (int i = 0; i < d_vector.size(); i++)
        // {
        //     if (d_vector[i] < 0)
        //     {
        //         num_negative_eigen_values++;
        //         break;
        //     }
        
        // }
        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (positive_definte && search_dir_correct_sign && solve_success)
            return true;
        else
        {
            K = H + alpha * I;        
            alpha *= 10;
        }
    }
    return false;
}

template<class T, int dim>
bool ShellFEMSolver<T, dim>::staticSolve()
{
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;

    iterateDirichletDoF([&](int offset, T target)
    {
        f[offset] = 0;
    });

    while (true)
    {
        VectorXT residual(deformed.rows());
        residual.setZero();

        residual_norm = computeResidual(u, residual);
        
        if (verbose)
            std::cout << "residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
        
        if (residual_norm < newton_tol)
            break;
        
        dq_norm = lineSearchNewton(u, residual, 15);
        
        if(cnt == max_newton_iter || dq_norm > 1e10)
            break;
        cnt++;
    }

    iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });

    deformed = undeformed + u;

    std::cout << "# of newton solve: " << cnt << " exited with |g|: " << residual_norm << "|dq|: " << dq_norm  << std::endl;
    // std::cout << u.norm() << std::endl;
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
    
}

template<class T, int dim>
T ShellFEMSolver<T, dim>::lineSearchNewton(VectorXT& _u, VectorXT& residual, int ls_max)
{
    VectorXT backup = _u;

    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    buildSystemMatrix(_u, K);
    
    
    bool success = linearSolve(K, residual, du);
    
    // std::cout << "dx: " <<  du.norm() << std::endl;
    
    if (!success)
        return 1e16;
    T norm = du.norm();
    
    T alpha = 1;
    T E0 = computeTotalEnergy(_u);
    // std::cout << "E0 " << E0 << std::endl;
    int cnt = 1;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        // std::cout << "ls# " << cnt << " E1 " << E1 << std::endl;
        if (E1 - E0 < 0 || cnt > ls_max)
        {
            _u = u_ls;
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }
    
    return norm;
}

template<class T, int dim>
void ShellFEMSolver<T, dim>::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    if constexpr (dim == 3)
    {
        deformed = undeformed + u;

        int n_vtx = deformed.size() / dim;
        V.resize(n_vtx, 3);
        tbb::parallel_for(0, n_vtx, [&](int i)
        {
            V.row(i) = deformed.template segment<dim>(i * dim).template cast<double>();
        });
        int n_faces = faces.size() / 3;
        F.resize(n_faces, 3);
        C.resize(n_faces, 3);
        tbb::parallel_for(0, (int)faces.size()/3, [&](int i)
        {
            F.row(i) = Eigen::Vector3i(faces[i * 3 + 0], 
                                                faces[i * 3 + 1],
                                                faces[i * 3 + 2]);
            
            C.row(i) = Eigen::Vector3d(0, 0.3, 1);
            
        });

        // std::cout << V << std::endl;
        // std::cout << F << std::endl;
    }
}


template<class T, int dim>
void ShellFEMSolver<T, dim>::createMeshFromNods(const TV& _min_corner, const TV& _max_corner, T dx)
{
    if constexpr (dim == 3)
    {
        Eigen::MatrixXd boundary_vertices;
        Eigen::MatrixXi boundary_edges;
        Eigen::MatrixXd points_inside_a_hole;

        // Triangulated interior
        Eigen::MatrixXd V2;
        Eigen::MatrixXi F2;

        boundary_vertices.resize(4, 2);
        boundary_edges.resize(4, 2);

        boundary_vertices(0, 0) = min_corner[0]; boundary_vertices(0, 1) = min_corner[2];
        boundary_vertices(1, 0) = max_corner[0]; boundary_vertices(1, 1) = min_corner[2];
        boundary_vertices(2, 0) = max_corner[0]; boundary_vertices(2, 1) = max_corner[2];
        boundary_vertices(3, 0) = min_corner[0]; boundary_vertices(3, 1) = max_corner[2];

        boundary_edges << 0,1, 1,2, 2,3, 3,0;
        T area = 0.5 * dx * dx;
        std::string cmd = "a" + std::to_string(area) + "q";
        igl::triangle::triangulate(boundary_vertices,
            boundary_edges,
            points_inside_a_hole, 
            cmd, 
            // "a0.005q",
            V2, F2);

        deformed.resize(V2.rows() * dim);
        deformed.setZero();

        tbb::parallel_for(0, (int)V2.rows(), [&](int i)
        {
            deformed[i * dim + 0] = V2(i, 0);
            deformed[i * dim + 2] = V2(i, 1);
        });

        undeformed = deformed;

        num_nodes = deformed.rows() / dim;

        f = VectorXT::Zero(deformed.rows());
        u = VectorXT::Zero(deformed.rows());

        faces.resize(F2.rows() * 3);
        
        tbb::parallel_for(0, (int)F2.rows(), [&](int i)
        {
            faces[i * dim + 0] = F2(i, 2);
            faces[i * dim + 1] = F2(i, 1);
            faces[i * dim + 2] = F2(i, 0);
        });

    }
    // E = 1e6;
    // density = 7.85e0;
    nu = 0.3;
    updateLameParameters();
    

    buildHingeStructure();
    computeRestShape();

    gravitional_energy = true;
    add_bending = true;
}

template<class T, int dim>
void ShellFEMSolver<T, dim>::createSceneFromNodes(const TV& _min_corner, const TV& _max_corner, T dx, 
        const std::vector<TV>& nodal_position)
{
    // createMeshFromNods(min_corner, _max_corner, dx);
    // return;
    if constexpr (dim == 3)
    {
        deformed.resize(nodal_position.size() * dim);
        tbb::parallel_for(0, (int)nodal_position.size(), [&](int i)
        {
            for (int d = 0; d < dim; d++)
                deformed[i * dim + d] = nodal_position[i][d];
        });

        // deformed.template segment<dim>(0 * dim) = TV(0.075000, 0, -0.075000);
        // deformed.template segment<dim>(1 * dim) = TV(-0.225000, 0, -0.075000);
        // deformed.template segment<dim>(2 * dim) = TV(0.075000, 0, 0.225000);
        // deformed.template segment<dim>(3 * dim) = TV(-0.225000, 0, 0.225000);

        undeformed = deformed;

        num_nodes = deformed.rows() / dim;

        f = VectorXT::Zero(deformed.rows());
        u = VectorXT::Zero(deformed.rows());

        int nx = std::floor((max_corner[0] - min_corner[0]) / dx) + 1;
        int nz = std::floor((max_corner[2] - min_corner[2]) / dx) + 1;
        

        scene_range = IV(nx, 1, nz);
        std::cout << "#nodes : " << scene_range.transpose() << std::endl;

        faces.resize((nx-1) * (nz-1) * 6); 
        faces.setZero();
        int cnt = 0;
        for (int i = 0; i < nx - 1; i++)
            for (int k = 0; k < nz - 1; k++)
            {
                IV idx;
                if (cnt % 2 == 0)
                {
                    idx[0] = globalOffset(IV(i, 0, k));
                    idx[1] = globalOffset(IV(i, 0, k+1));
                    idx[2] = globalOffset(IV(i+1, 0, k+1));
                    faces.template segment<3>(cnt*6) = idx;

                    idx[0] = globalOffset(IV(i+1, 0, k+1));
                    idx[1] = globalOffset(IV(i+1, 0, k));
                    idx[2] = globalOffset(IV(i, 0, k));
                    faces.template segment<3>(cnt*6 + 3) = idx;
                }
                else
                {
                    idx[0] = globalOffset(IV(i+1, 0, k));
                    idx[1] = globalOffset(IV(i, 0, k));
                    idx[2] = globalOffset(IV(i, 0, k+1));
                    faces.template segment<3>(cnt*6) = idx;

                    idx[0] = globalOffset(IV(i, 0, k+1));
                    idx[1] = globalOffset(IV(i+1, 0, k+1));
                    idx[2] = globalOffset(IV(i+1, 0, k));
                    faces.template segment<3>(cnt*6 + 3) = idx;
                }
                cnt+=1;
            }
        // faces.segment<3>(0 * 3) = IV(3, 1, 0);
        // faces.segment<3>(1 * 3) = IV(0, 2, 3);
    }
    // E = 1e6;
    // density = 7.85e0;
    nu = 0.3;
    updateLameParameters();
    

    buildHingeStructure();
    computeRestShape();

    gravitional_energy = true;
    add_bending = true;
}

template<class T, int dim>
void ShellFEMSolver<T, dim>::loadFromMesh(std::string filename)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    igl::readOBJ(filename, V, F);

    if constexpr (dim == 3)
    {
        deformed.resize(V.rows() * dim);
        tbb::parallel_for(0, (int)V.rows(), [&](int i)
        {
            for (int d = 0; d < dim; d++)
                deformed[i * dim + d] = V(i, d);
        });

        undeformed = deformed;

        num_nodes = deformed.rows() / dim;

        f = VectorXT::Zero(deformed.rows());
        u = VectorXT::Zero(deformed.rows());

        faces.resize(F.rows() * 3);
        
        tbb::parallel_for(0, (int)F.rows(), [&](int i)
        {
            for (int d = 0; d < dim; d++)
                faces[i * dim + d] = F(i, d);
        });
    }
    E = 1e6;
    density = 7.85e0;
    nu = 0.3;
    updateLameParameters();
    

    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < V.rows(); i++)
    {
        TV x = V.row(i);
        for (int d = 0; d < dim; d++)
        {
            max_corner[d] = std::max(max_corner[d], x[d]);
            min_corner[d] = std::min(min_corner[d], x[d]);
        }
    }
    
    buildHingeStructure();
    computeRestShape();

    gravitional_energy = true;
    add_bending = true;
}


template class ShellFEMSolver<double, 3>;