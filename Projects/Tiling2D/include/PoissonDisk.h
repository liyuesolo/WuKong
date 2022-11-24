#ifndef POISSON_DISK_H
#define POISSON_DISK_H

#include <iostream>
#include <vector>
#include <string>

#include "cyVector.h"
#include "cySampleElim.h"

#include "VecMatDef.h"

class PoissonDisk
{
public:
    using TV3 = Vector<T, 3>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
public:
    

public:
    void sample3DBox(const TV3& min_corner, const TV3& max_corner,
        int n_samples, VectorXT& samples)
    {
        cy::WeightedSampleElimination< cy::Vec3f, float, 3, int > wse;

        cy::Vec3f minimumBounds( min_corner[0], min_corner[1], min_corner[2]);
        cy::Vec3f maximumBounds( max_corner[0], max_corner[1], max_corner[2]);
        wse.SetBoundsMin( minimumBounds );
        wse.SetBoundsMax( maximumBounds );

        std::vector< cy::Vec3f > inputPoints(n_samples * 10);
        for ( size_t i=0; i<inputPoints.size(); i++ ) {
            inputPoints[i].x = minimumBounds[0] + (float) rand() / RAND_MAX * (maximumBounds[0] - minimumBounds[0]);
            inputPoints[i].y = minimumBounds[1] + (float) rand() / RAND_MAX * (maximumBounds[1] - minimumBounds[1]);
            inputPoints[i].z = minimumBounds[2] + (float) rand() / RAND_MAX * (maximumBounds[2] - minimumBounds[2]);
        }

        std::vector< cy::Vec3f > outputPoints(n_samples);

        float d_max = 2 * wse.GetMaxPoissonDiskRadius( 3, outputPoints.size() );
        wse.Eliminate( inputPoints.data(), inputPoints.size(), 
                    outputPoints.data(), outputPoints.size(),
                    true,
                    d_max );
        samples.resize(n_samples * 3);
        for (int i = 0; i < n_samples; i++)
        {
            samples[i * 3 + 0] = outputPoints[i].x;
            samples[i * 3 + 1] = outputPoints[i].y;
            samples[i * 3 + 2] = outputPoints[i].z;
            std::cout << samples.segment<3>(i * 3).transpose() << std::endl;
        }
    }

    template<int dim>
    void sampleNDBox(const Vector<T, dim>& min_corner, const Vector<T, dim>& max_corner,
        int n_samples, VectorXT& samples)
    {
        cy::WeightedSampleElimination< cy::Vec<T, dim>, T, dim, int > wse;

        cy::Vec<T, dim> minimumBounds, maximumBounds;
        for (int d = 0; d < dim; d++)
        {
            minimumBounds[d] = min_corner[d];
            maximumBounds[d] = max_corner[d];
        }

        wse.SetBoundsMin( minimumBounds );
        wse.SetBoundsMax( maximumBounds );


        std::vector< cy::Vec<T, dim> > inputPoints(n_samples * 10);
        for ( size_t i=0; i<inputPoints.size(); i++ ) 
        {
            for (int d = 0; d < dim; d++)
                inputPoints[i][d] = minimumBounds[d] + (T) rand() / RAND_MAX * (maximumBounds[d] - minimumBounds[d]);
        }

        std::vector< cy::Vec<T, dim> > outputPoints(n_samples);

        float d_max = 2 * wse.GetMaxPoissonDiskRadius( dim, outputPoints.size() );
        wse.Eliminate( inputPoints.data(), inputPoints.size(), 
                    outputPoints.data(), outputPoints.size(),
                    true,
                    d_max );
        samples.resize(n_samples * dim);
        for (int i = 0; i < n_samples; i++)
        {
            for (int d = 0; d < dim; d++)
                samples[i * dim + d] = outputPoints[i][d];    

            std::cout << samples.template segment<dim>(i * dim).transpose() << std::endl;
        }
    }
public:
    PoissonDisk() {}
    ~PoissonDisk() {}
};

#endif // !POISSON_DISK_H