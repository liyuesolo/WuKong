#include "../include/FEMSolver.h"
#include "autodiff/CST3DShell.h"
using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

template <int dim>
void FEMSolver<dim>::buildHingeStructure()
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

template <int dim>
void FEMSolver<dim>::computeRestShape()
{
    Xinv.resize(nFaces());
    rest_area = VectorXT::Ones(nFaces());
    thickness = VectorXT::Ones(nFaces()) * 0.01;

    if constexpr (dim == 3)
    {
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
}

template <int dim>
void FEMSolver<dim>::addShellEnergy(T& energy)
{
    //std::cout<<"deformed: "<<deformed.transpose()<<std::endl;
    if constexpr (dim == 3)
    {
        if(add_stretching)
        {
            iterateFaceSerial([&](int face_idx)
        {
            FaceVtx vertices = getFaceVtxDeformed(face_idx);
            FaceVtx undeformed_vertices = getFaceVtxUndeformed(face_idx);

            FaceIdx indices = faces.segment<3>(face_idx * 3);
            //std::cout << "face# " << face_idx << " nodes: " << indices.transpose()<<std::endl;


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
            // std::cout << "Node# " << 0 << " nodes: " << x0.transpose()<<std::endl;
            // std::cout << "Node# " << 0 << " nodes: " << x1.transpose()<<std::endl;
            // std::cout << "Node# " << 0 << " nodes: " << x2.transpose()<<std::endl;
            // std::cout << "Node# " << 0 << " nodes: " << X0.transpose()<<std::endl;
            // std::cout << "Node# " << 0 << " nodes: " << X1.transpose()<<std::endl;
            // std::cout << "Node# " << 0 << " nodes: " << X2.transpose()<<std::endl;


            T k_s = E_2 * m_h / (1.0 - nu * nu);


            energy += compute3DCSTShellEnergy(nu, k_s, x0, x1, x2, X0, X1, X2);

            if (gravitional_energy)
                energy += compute3DCSTGravitationalEnergy(rou, m_h, gravity, x0, x1, x2, X0, X1, X2);
            

        });
        }
        

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

            T k_bend = E_2 * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));

            // T Wdmy[1];
            // #include "autodiff/DSBending_W.mcg"
            // // maple end
            // energy += Wdmy[0];

            TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
            TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

            energy += computeDSBendingEnergy(k_bend, x0, x1, x2, x3, X0, X1, X2, X3);
    
        });
    }
}

template <int dim>
void FEMSolver<dim>::addShellForceEntries(VectorXT& residual)
{
    if constexpr (dim == 3)
    {
        if(add_stretching)
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

            T k_s = E_2 * m_h / (1.0 - nu * nu);
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
                compute3DCSTGravitationalEnergyGradient(rou, m_h, gravity, x0, x1, x2, X0, X1, X2, graviational_force);
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
        }
        

        
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

            T k_bend = E_2 * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));

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
            fx *= -1.0;

            for (int i = 0; i < 4; i++)
            {
                for (int d = 0; d < dim; d++)
                {
                    residual[hinge_idx[i] * dim + d] += fx[i * dim + d];
                }
            }
        });
    }
}


template <int dim>
void FEMSolver<dim>::addShellHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    if constexpr (dim == 3)
    {
        if(add_stretching)
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

            // std::array<TV, 3> x, X;
            // x[0] = vertices.row(0);
            // x[1] = vertices.row(1);
            // x[2] = vertices.row(2);

            // X[0] = undeformed_vertices.row(0);
            // X[1] = undeformed_vertices.row(1);
            // X[2] = undeformed_vertices.row(2);

            
            // T J[9][9];
    		// memset(J, 0, sizeof(J));

            // #include "autodiff/CST3D_StVK_J.mcg"

            // Matrix<T, 9, 9> hess = -Eigen::Map<Eigen::Matrix<double, 9, 9>>(J[0]);

            T k_s = E_2 * m_h / (1.0 - nu * nu);

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
                compute3DCSTGravitationalEnergyHessian(rou, m_h, gravity, x0, x1, x2, X0, X1, X2, g_J);

                // // std::cout << J << std::endl;

                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        for (int d = 0; d < dim; d++)
                            for (int dd = 0; dd < dim; dd++)
                                entries.push_back(Entry(indices[i] * dim + d, indices[j] * dim + dd, g_J(i * dim + d, j * dim + dd)));
            }
        });
        }
        

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

            T k_bend = E_2 * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu, 2)));
    

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
}

template class FEMSolver<2>;
template class FEMSolver<3>;