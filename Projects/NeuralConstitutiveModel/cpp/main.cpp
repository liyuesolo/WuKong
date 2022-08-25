#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#include "VecMatDef.h"
#include "energy_autodiff.h"

std::pair<double, double> ComputeInvariantsFromLambda(const double* lambda)
{
	double I1, I2;

	double t1 = lambda[0] * lambda[0];
	double t2 = lambda[1] * lambda[1];
	double t3 = 0.1e1 / 0.2e1;
	double t4 = 0.1e1 / 0.4e1;
	I1 = t3 * (t2 + t1) - 0.1e1;
	I2 = t4 * (-t1 * (0.1e1 - t2) - t2 + 0.1e1);

	return std::make_pair(I1, I2);
}

std::pair<double, double> ComputeDerivativeStVKWrtInvariants(const double* p, double I1s, double I2s)
{
	double dWI1, dWI2;

	double t1 = 2 * p[1];
	dWI1 = I1s * (p[0] + t1);
	dWI2 = -t1;

	return std::make_pair(dWI1, dWI2);
}

void isotropicStVKInvariant()
{
    std::vector<double> material_parameters = {1., 1.}; //Have no idea what these parameters represent physically (random sh*t)

	std::pair<double, double> range_lambda1(0.5, 5.0);
	std::pair<double, double> range_lambda2(0.5, 5.0);

	size_t num_data_points = 400;

	std::ofstream file("data.txt"); //Format: lambda1, lambda2, I1, I2, dWI1, dWI2

	for(int i=0; i<num_data_points; ++i)
	{
		double lambda1 = range_lambda1.first + ((double)i/(double)num_data_points)*(range_lambda1.second - range_lambda1.first);

		//I don't account for lambda1 > lambda2 (probably good for learning?)
		for(int j=0; j<num_data_points; ++j)
		{
			double lambda2 = range_lambda2.first + ((double)j/(double)num_data_points)*(range_lambda2.second - range_lambda2.first);

			std::vector<double> lambda = {lambda1, lambda2};

			double I1, I2, dWI1, dWI2;

            std::pair<double, double> I1_I2 = ComputeInvariantsFromLambda(lambda.data());
            double energy = 0.5 * material_parameters[0] * I1_I2.first * I1_I2.first
                + material_parameters[1] * (I1_I2.first * I1_I2.first - 2.0 * I1_I2.second);

            std::pair<double, double> dWI1_dWI2 = ComputeDerivativeStVKWrtInvariants(material_parameters.data(), I1_I2.first, I1_I2.second);

			file << lambda1 << " " << lambda2 << " " << I1_I2.first << " " << I1_I2.second << " " << dWI1_dWI2.first << " " << dWI1_dWI2.second << " " << energy << std::endl;
		}
	}

	file.close();
}

void orthotropicStVK()
{
    using TV = Vector<T, 2>;
    Vector<T, 4> ExEy_nuxy_nuyx;
    ExEy_nuxy_nuyx << 50., 12.5, 0.4, 0.1;
    TV range_lambda1(0.5, 5.0);
	TV range_lambda2(0.5, 5.0);

    int num_data_points = 20;
    std::string base_dir = "/home/yueli/Documents/ETH/WuKong/Projects/NeuralConstitutiveModel/python/";
	std::ofstream file(base_dir + "orthotropic_stvk_train.txt"); 
    std::vector<T> thetas = {0, M_PI / 2.0};
    for (T theta : thetas)
    {
        for(int i=0; i<num_data_points; ++i)
        {
            double lambda1 = range_lambda1[0] + ((double)i/(double)num_data_points)*(range_lambda1[1] - range_lambda1[0]);

            for(int j=0; j<num_data_points; ++j)
            {
                double lambda2 = range_lambda2[0] + ((double)j/(double)num_data_points)*(range_lambda2[1] - range_lambda2[0]);

                T energy;
                compute2DOrthoStVkEnergy(lambda1, lambda2, theta, ExEy_nuxy_nuyx, energy);
                Vector<T, 3> dedx;
                compute2DOrthoStVkEnergyGradient(lambda1, lambda2, theta, ExEy_nuxy_nuyx, dedx);
                file << std::setprecision(12) << lambda1 << " " << lambda2 << " " << theta << " " << dedx[0] << " " << dedx[1] << " " << energy << std::endl;
            }
        }
    }
	file.close();
    file.open(base_dir + "orthotropic_stvk_test.txt");
    thetas = {M_PI / 6.0, M_PI / 4.0,  M_PI / 3.0};
    num_data_points = 50;
    for (T theta : thetas)
    {
        for(int i=0; i<num_data_points; ++i)
        {
            double lambda1 = range_lambda1[0] + ((double)i/(double)num_data_points)*(range_lambda1[1] - range_lambda1[0]);

            for(int j=0; j<num_data_points; ++j)
            {
                double lambda2 = range_lambda2[0] + ((double)j/(double)num_data_points)*(range_lambda2[1] - range_lambda2[0]);

                T energy;
                compute2DOrthoStVkEnergy(lambda1, lambda2, theta, ExEy_nuxy_nuyx, energy);
                Vector<T, 3> dedx;
                compute2DOrthoStVkEnergyGradient(lambda1, lambda2, theta, ExEy_nuxy_nuyx, dedx);
                file << std::setprecision(12) << lambda1 << " " << lambda2 << " " << theta << " " << dedx[0] << " " << dedx[1] << " " << energy << std::endl;
            }
        }
    }
    file.close();
}

void isotropicStVK()
{
    using TV = Vector<T, 2>;
    TV Enu(50, 0.4);
    
    TV range_lambda1(0.5, 5.0);
	TV range_lambda2(0.5, 5.0);
    TV range_theta(0.0, 2.0 * M_PI);

    int num_data_points = 50;
    int num_data_points_theta = 50;
    std::string base_dir = "/home/yueli/Documents/ETH/WuKong/Projects/NeuralConstitutiveModel/python/";
	std::ofstream file(base_dir + "isotropic_stvk_train.txt"); 
    // std::vector<T> thetas = {0, M_PI / 2.0};
    // for (T theta : thetas)
    for(int k = 0; k < num_data_points; k++ )
    {
        T theta = range_theta[0] + ((double)k/(double)num_data_points_theta)*(range_theta[1] - range_theta[0]);
        for(int i=0; i<num_data_points; ++i)
        {
            double lambda1 = range_lambda1[0] + ((double)i/(double)num_data_points)*(range_lambda1[1] - range_lambda1[0]);

            for(int j=0; j<num_data_points; ++j)
            {
                double lambda2 = range_lambda2[0] + ((double)j/(double)num_data_points)*(range_lambda2[1] - range_lambda2[0]);

                T energy;
                compute2DisoStVkEnergy(lambda1, lambda2, theta, Enu, energy);
                Vector<T, 3> dedx;
                compute2DisoStVkEnergyGradient(lambda1, lambda2, theta, Enu, dedx);
                file << std::setprecision(12) << lambda1 << " " << lambda2 << " " << theta << " " << dedx[0] << " " << dedx[1] << " " << energy << std::endl;
            }
        }
    }
	file.close();
    file.open(base_dir + "isotropic_stvk_test.txt");
    std::vector<T> thetas = {M_PI / 6.0, M_PI / 4.0,  M_PI / 3.0};
    num_data_points = 50;
    for (T theta : thetas)
    {
        for(int i=0; i<num_data_points; ++i)
        {
            double lambda1 = range_lambda1[0] + ((double)i/(double)num_data_points)*(range_lambda1[1] - range_lambda1[0]);

            for(int j=0; j<num_data_points; ++j)
            {
                double lambda2 = range_lambda2[0] + ((double)j/(double)num_data_points)*(range_lambda2[1] - range_lambda2[0]);

                T energy;
                compute2DisoStVkEnergy(lambda1, lambda2, theta, Enu, energy);
                Vector<T, 3> dedx;
                compute2DisoStVkEnergyGradient(lambda1, lambda2, theta, Enu, dedx);
                file << std::setprecision(12) << lambda1 << " " << lambda2 << " " << theta << " " << dedx[0] << " " << dedx[1] << " " << energy << std::endl;
            }
        }
    }
    file.close();
    T rest_energy;
    compute2DisoStVkEnergy(1, 1, 0, Enu, rest_energy);
    std::cout << "rest " << rest_energy << std::endl;
}

int main(int argc, char *argv[])
{
    // orthotropicStVK();
    isotropicStVK();
	return 0;
}