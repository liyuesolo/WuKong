#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"

template <int dim>
void FEMSolver<dim>::computeDeformationGradient(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, TM& F)
{
    //N(b) = [1]
    Matrix<T, 4, 3> dNdb;
        dNdb << -1.0, -1.0, -1.0, 
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;
    TM dXdb = x_undeformed.transpose() * dNdb;
    TM dxdb = x_deformed.transpose() * dNdb;
    F = dxdb * dXdb.inverse();
}

template <int dim>
T FEMSolver<dim>::computeVolume(const EleNodes& x_undeformed)
{
    TV a = x_undeformed.row(1) - x_undeformed.row(0);
	TV b = x_undeformed.row(2) - x_undeformed.row(0);
	TV c = x_undeformed.row(3) - x_undeformed.row(0);
	T volumeParallelepiped = a.cross(b).dot(c);
	T tetVolume = 1.0 / 6.0 * volumeParallelepiped;
	return tetVolume;
}

template <int dim>
T FEMSolver<dim>::computeNeoHookeanStrainEnergy(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed)
{
    T lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
	T mu = E / 2.0 / (1.0 + nu);

    T volume = computeVolume(x_undeformed);
    TM F;
    computeDeformationGradient(x_deformed, x_undeformed, F);
    TM U, VT;
    TV Sigma;
    polarSVD(F, U, Sigma, VT);

    TM right_cauchy = F.transpose() * F;
    T J = F.determinant();
    T lnJ = std::log(J);
    T I1 = right_cauchy.trace();

    T energy_density = 0.5 * mu * (I1 - 3.0 - 2.0 * lnJ) + lambda * 0.5 * (lnJ*lnJ);
    return energy_density * volume;
}

template <int dim>
void FEMSolver<dim>::computeNeoHookeanStrainEnergyGradient(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, Vector<T, 12>& gradient)
{
    gradient.setZero();

    T lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
	T mu = E / 2.0 / (1.0 + nu);
    
    TM F;
    computeDeformationGradient(x_deformed, x_undeformed, F);
    TM U, VT;
    TV Sigma;
    polarSVD(F, U, Sigma, VT);

    T J = F.determinant();
    T lnJ = std::log(J);

    T volume = computeVolume(x_undeformed);

    TM Piola = mu * (F - F.inverse().transpose()) + lambda * lnJ * F.inverse().transpose();

    Matrix<T, 4, 3> dNdb;
        dNdb << -1.0, -1.0, -1.0, 
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;

    TM dXdb = x_undeformed.transpose() * dNdb;

    Matrix<T, 4, 3> dNdX = dNdb * dXdb.inverse();
    
    for (int i = 0; i < 4; i++)
    {
        gradient.segment<3>(i * 3) += volume * Piola * dNdX.row(i).transpose();
    }
}

template <int dim>
void FEMSolver<dim>::polarSVD(TM& F, TM& U, TV& Sigma, TM& VT)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
	U = svd.matrixU();
	Sigma = svd.singularValues();
	Eigen::Matrix3d V = svd.matrixV();


	// if det(u) = -1 flip the sign of u3 and sigma3
	if (U.determinant() < 0)
	{
		for (int i = 0; i < 3; i++)
		{
			U(i, 2) *= -1.0;
		}
		Sigma[2] *= -1.0;
	}
	// if det(v) = -1 flip the sign of v3 and sigma3
	if (V.determinant() < 0)
	{
		for (int i = 0; i < 3; i++)
		{
			V(i, 2) *= -1.0;
		}
		Sigma[2] *= -1.0;
	}

	VT = V.transpose();

	F = U * Sigma.asDiagonal() * VT;
}


template <int dim>
void FEMSolver<dim>::computeNeoHookeanStrainEnergyHessian(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, Matrix<T, 12, 12>& hessian)
{
    hessian.setZero();

    T lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
	T mu = E / 2.0 / (1.0 + nu);
    
    TM F;
    computeDeformationGradient(x_deformed, x_undeformed, F);
    
    TM U, VT;
    TV Sigma;
    polarSVD(F, U, Sigma, VT);
    
    T J = F.determinant();
    T lnJ = std::log(J);

    T volume = computeVolume(x_undeformed);

    // https://dl.acm.org/doi/pdf/10.5555/2422356.2422361
    // section 2
    
    // these are the derivativs dPsi/dsigma_i
    T psi0, psi1, psi2, psi00, psi11, psi22, psi01, psi02, psi12;
    
    T m01, p01, m02, p02, m12, p12;


    T inv0 = T(1) / Sigma(0);
    T inv1 = T(1) / Sigma(1);
    T inv2 = T(1) / Sigma(2);

    psi0 = mu * (Sigma(0) - inv0) + lambda * inv0 * lnJ;
    psi1 = mu * (Sigma(1) - inv1) + lambda * inv1 * lnJ;
    psi2 = mu * (Sigma(2) - inv2) + lambda * inv2 * lnJ;

    T inv2_0 = T(1) / Sigma(0) / Sigma(0);
    T inv2_1 = T(1) / Sigma(1) / Sigma(1);
    T inv2_2 = T(1) / Sigma(2) / Sigma(2);

    psi00 = mu * (T(1) + inv2_0) - lambda * inv2_0 * (lnJ - T(1));
    psi11 = mu * (T(1) + inv2_1) - lambda * inv2_1 * (lnJ - T(1));
    psi22 = mu * (T(1) + inv2_2) - lambda * inv2_2 * (lnJ - T(1));
    psi01 = lambda / Sigma(0) / Sigma(1);
    psi12 = lambda / Sigma(1) / Sigma(2);
    psi02 = lambda / Sigma(0) / Sigma(2);

    // (psiA-psiB)/(SigmaA-SigmaB)
    T common = mu - lambda * lnJ;
    m01 = mu + common / Sigma(0) / Sigma(1);
    m02 = mu + common / Sigma(0) / Sigma(2);
    m12 = mu + common / Sigma(1) / Sigma(2);

    // (psiA+psiB)/(SigmaA+SigmaB)

    p01 = (psi0 + psi1) / std::max(Sigma(0) + Sigma(1), 1e-6);
    p02 = (psi0 + psi2) / std::max(Sigma(0) + Sigma(2), 1e-6);
    p12 = (psi1 + psi2) / std::max(Sigma(1) + Sigma(2), 1e-6);

    Matrix<T, 9, 9> dP_dF; dP_dF.setZero();
    //s11, s22, s33, s12, s21, s13, s31, s23, s32
    //This is the A block
    dP_dF(0, 0) = psi00;
    dP_dF(0, 4) = psi01;
    dP_dF(4, 0) = psi01;
    dP_dF(0, 8) = psi02;
    dP_dF(8, 0) = psi02;
    dP_dF(4, 4) = psi11;
    dP_dF(4, 8) = psi12;
    dP_dF(8, 4) = psi12;
    dP_dF(8, 8) = psi22;

    // these are the B B B blocks
    dP_dF(1, 1) = 0.5 * (m01 + p01);
    dP_dF(1, 3) = 0.5 * (m01 - p01);
    dP_dF(3, 1) = 0.5 * (m01 - p01);
    dP_dF(3, 3) = 0.5 * (m01 + p01);

    dP_dF(2, 2) = 0.5 * (m02 + p02);
    dP_dF(2, 6) = 0.5 * (m02 - p02);
    dP_dF(6, 2) = 0.5 * (m02 - p02);
    dP_dF(6, 6) = 0.5 * (m02 + p02);

    dP_dF(5, 5) = 0.5 * (m12 + p12);
    dP_dF(5, 7) = 0.5 * (m12 - p12);
    dP_dF(7, 5) = 0.5 * (m12 - p12);
    dP_dF(7, 7) = 0.5 * (m12 + p12);

    Matrix<T, 9, 9> dP_dF_UUVV; dP_dF_UUVV.setZero();

    //https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf 
    // equation 198 matrix contraction index notation.
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int r = 0; r < 3; r++)
            {
                for (int s = 0; s < 3; s++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            for (int m = 0; m < 3; m++)
                            {
                                for (int o = 0; o < 3; o++)
                                {
                                    dP_dF_UUVV(i * 3 + j, r * 3 + s) += dP_dF(k * 3 + l, m * 3 + o) *
                                        U(i, k) * U(r, m) * VT(o, s) * VT(l, j);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Matrix<T, 9, 9> dPdF_ad;
    // neoHookeandPdF(lambda, mu, F, dPdF_ad);

    // std::cout << "dPdF ad" << std::endl;
    // std::cout << dPdF_ad << std::endl;
    // std::cout << "dPdF " << std::endl;
    // std::cout << dP_dF_UUVV << std::endl;
    // std::cout << "========================" << std::endl;
    // std::getchar();
    
    Matrix<T, 4, 3> dNdb;
        dNdb << -1.0, -1.0, -1.0, 
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;

    TM dXdb = x_undeformed.transpose() * dNdb;

    Matrix<T, 4, 3> dNdX = dNdb * dXdb.inverse();
    TM F0 = TM::Identity();
    
    //https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf 
    // equation 72 
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int omega = 0; omega < 3; omega++)
            {
                for (int gamma = 0; gamma < 3; gamma++)
                {
                    for (int alpha = 0; alpha < 3; alpha++)
                    {
                        for (int tao = 0; tao < 3; tao++)
                        {
                            for (int beta = 0; beta < 3; beta++)
                            {
                                for (int sigma = 0; sigma < 3; sigma++)
                                {
                                    hessian(i * 3 + alpha, j * 3 + tao) += volume * dP_dF_UUVV(alpha * 3 + beta, tao * 3 + sigma) * 
                                        dNdX(j, omega) * dNdX(i, gamma) * F0(omega, sigma) * F0(gamma, beta); 
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <int dim>
void FEMSolver<dim>::addElastsicPotential(T& energy)
{
    VectorXT energies_neoHookean(num_ele);
    energies_neoHookean.setZero();
    iterateElementParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        T ei = computeNeoHookeanStrainEnergy(x_deformed, x_undeformed);
        energies_neoHookean[tet_idx] += ei;
    });
    energy += energies_neoHookean.sum();
}

template <int dim>
void FEMSolver<dim>::addElasticForceEntries(VectorXT& residual)
{
    iterateElementSerial([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        Vector<T, 12> dedx;
        computeNeoHookeanStrainEnergyGradient(x_deformed, x_undeformed, dedx);
        addForceEntry<12>(residual, indices, -dedx);
    });
}

template <int dim>
void FEMSolver<dim>::addElasticHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    iterateElementSerial([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        Matrix<T, 12, 12> hessian, hessian_ad;
        computeNeoHookeanStrainEnergyHessian(x_deformed, x_undeformed, hessian);

        if (project_block_PD)
            projectBlockPD<12>(hessian);
        
        addHessianEntry<12>(entries, indices, hessian);
    });
}

// template class FEMSolver<2>;
template class FEMSolver<3>;