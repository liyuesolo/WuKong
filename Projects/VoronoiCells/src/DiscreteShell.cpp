#include "../autodiff/CST3DShell.h"
#include "../include/IntrinsicSimulation.h"

void IntrinsicSimulation::buildHingeStructure()
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

void IntrinsicSimulation::computeRestShape()
{
    Xinv.resize(nFaces());
    shell_rest_area = VectorXT::Ones(nFaces());
    thickness = VectorXT::Ones(nFaces()) * 0.003;
    
    iterateFaceSerial([&](int i){
        TV E[2]; // edges
        FaceVtx vertices = getFaceVtxUndeformed(i);
        E[0] = vertices.row(1) - vertices.row(0);
        E[1] = vertices.row(2) - vertices.row(0);
        
        TV N = E[0].cross(E[1]);
        T m_A0 = N.norm()*0.5;
        shell_rest_area[i] = m_A0;
        
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

void IntrinsicSimulation::addShellEnergy(T& energy)
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

        T m_A0 = shell_rest_area[face_idx];
        T m_h = thickness[face_idx];

        std::array<TV, 3> x;
        x[0] = vertices.row(0);
        x[1] = vertices.row(1);
        x[2] = vertices.row(2);
        
        

        
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); 
        TV X1 = undeformed_vertices.row(1); 
        TV X2 = undeformed_vertices.row(2);

        T k_s = E_shell * m_h / (1.0 - nu_shell * nu_shell);

        energy += compute3DCSTShellEnergy(nu_shell, k_s, x0, x1, x2, X0, X1, X2);

    });

    iterateHingeSerial([&](const HingeIdx& hinge_idx){
        
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

        T k_bend = E_shell * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu_shell, 2)));

        TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

        energy += computeDSBendingEnergy(k_bend, x0, x1, x2, x3, X0, X1, X2, X3);

    });
}

void IntrinsicSimulation::addShellForceEntry(VectorXT& residual)
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

        T m_A0 = shell_rest_area[face_idx];
        T m_h = thickness[face_idx];

        std::array<TV, 3> x, X;
        x[0] = vertices.row(0);
        x[1] = vertices.row(1);
        x[2] = vertices.row(2);

        X[0] = undeformed_vertices.row(0);
        X[1] = undeformed_vertices.row(1);
        X[2] = undeformed_vertices.row(2);
        

        T k_s = E_shell * m_h / (1.0 - nu_shell * nu_shell);
        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);
        
        Vector<T, 9> dedx;
        compute3DCSTShellEnergyGradient(nu_shell, k_s, x0, x1, x2, X0, X1, X2, dedx);
        
        dedx *= -1.0;
        
        for (int i = 0; i < 3; i++)
        {
            for (int d = 0; d < 3; d++)
            {
                residual[indices[i] * 3 + d + shell_dof_start] += dedx[i * 3 + d];
            }   
        }
    });

    
    iterateHingeSerial([&](const HingeIdx& hinge_idx){
                
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

        T k_bend = E_shell * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu_shell, 2)));

        Vector<T, 12> fx;

        // #include "autodiff/DSBending_fx.mcg"
        
        TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

        computeDSBendingEnergyGradient(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, fx);
        fx *= -1;

        for (int i = 0; i < 4; i++)
        {
            for (int d = 0; d < 3; d++)
            {
                residual[hinge_idx[i] * 3 + d + shell_dof_start] += fx[i * 3 + d];
            }
        }
    });
}
void IntrinsicSimulation::addShellHessianEntries(std::vector<Entry>& entries)
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

        T m_A0 = shell_rest_area[face_idx];
        T m_h = thickness[face_idx];

        std::array<TV, 3> x, X;
        x[0] = vertices.row(0);
        x[1] = vertices.row(1);
        x[2] = vertices.row(2);

        X[0] = undeformed_vertices.row(0);
        X[1] = undeformed_vertices.row(1);
        X[2] = undeformed_vertices.row(2);

        T k_s = E_shell * m_h / (1.0 - nu_shell * nu_shell);

        TV x0 = vertices.row(0); TV x1 = vertices.row(1); TV x2 = vertices.row(2);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2);

        Matrix<T, 9, 9> hess;
        compute3DCSTShellEnergyHessian(nu_shell, k_s, x0, x1, x2, X0, X1, X2, hess);
        // makePD<T, 9>(hess);
        
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int d = 0; d < 3; d++)
                {
                    for (int dd = 0; dd < 3; dd++)
                    {
                        entries.push_back(Entry(indices[i] * 3 + d + shell_dof_start, 
                            indices[j] * 3 + dd + shell_dof_start, hess(i * 3 + d, j * 3 + dd)));
                    }   
                } 
            }
        }
    });

    iterateHingeSerial([&](const HingeIdx& hinge_idx){
        
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

        T k_bend = E_shell * std::pow(thickness[0], 3) / (24 * (1.0 - std::pow(nu_shell, 2)));
        
        Matrix<T, 12, 12> hess;
        TV x0 = deformed_vertices.row(0); TV x1 = deformed_vertices.row(1); TV x2 = deformed_vertices.row(2); TV x3 = deformed_vertices.row(3);
        TV X0 = undeformed_vertices.row(0); TV X1 = undeformed_vertices.row(1); TV X2 = undeformed_vertices.row(2); TV X3 = undeformed_vertices.row(3);

        computeDSBendingEnergyHessian(k_bend, x0, x1, x2, x3, X0, X1, X2, X3, hess);
        

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int d = 0; d < 3; d++)
                {
                    for (int dd = 0; dd < 3; dd++)
                    {
                        entries.push_back(Entry(hinge_idx[i] * 3 + d + shell_dof_start, 
                            hinge_idx[j] * 3 + dd  + shell_dof_start, hess(i * 3 + d, j * 3 + dd)));
                    }   
                } 
            }
        }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

    });
}