#ifndef HYBRID_C2_CURVE_H
#define HYBRID_C2_CURVE_H


// reference
// view-source:http://www.cemyuksel.com/research/interpolating_splines/curves.html
// A Class of C2 Interpolating Splines ACM Transactions on Graphics, 39, 5, 2020

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "VecMatDef.h"

template<class T, int dim>
struct CurveData
{
    CurveData(Vector<T, dim> c, Vector<T, dim> a1, Vector<T, dim> a2, Vector<T, 2> bound)
    : center(c), axis1(a1), axis2(a2), limits(bound) {}
    Vector<T, dim> center, axis1, axis2;
    Vector<T, 2> limits;
};

template<class T, int dim>
class HybridC2Curve
{
public:
    using TV = Vector<T, dim>;

    int sub_div;

    std::vector<TV> data_points;
    std::vector<TV> points_on_curve;
    std::vector<CurveData<T, dim>*> curve_data;
 
public:
    HybridC2Curve(int _sub_div) : sub_div(_sub_div) {}
    HybridC2Curve() : sub_div(64) {}
    ~HybridC2Curve() 
    {
        for(auto data : curve_data)
        {
            delete data;
        }
    }

    void getLinearSegments(std::vector<TV>& points)
    {
    
        if (data_points.size() <= 2)
            points = data_points;
        else
        {
            generateC2Curves();
            points = points_on_curve;
        }
    }

    void constructIndividualCurves()
    {
        curve_data.clear();
        
        for(int i = 1; i < data_points.size() - 1; i++)
        {
            TV center, axis1, axis2; Vector<T, 3> limits;
            hybridInterpolation(i, center, axis1, axis2, limits);
            curve_data.push_back(new CurveData<T, dim>(
                                    center, axis1, axis2, 
                                    Vector<T, 2>(limits[0], limits[2])));
        }
    }

    void sampleCurves(std::vector<TV>& points)
    {
        points_on_curve.clear();
        constructIndividualCurves();
        
        // first half of the first spline, no interpolation
        for (int vtx = 0; vtx < sub_div / 2; vtx++)
        {
            T t = T(vtx) / sub_div;
            TV F0; F(t, 0, F0);
            points_on_curve.push_back(F0);
        }
        // interpolate between two splines
        for (int curve_idx = 0; curve_idx < curve_data.size()-1; curve_idx++)
        {
            for (int vtx = 0; vtx < sub_div / 2; vtx++)
            {
                T t = T(vtx) / sub_div * 2.0;
                T theta = t * M_PI * 0.5;
                TV F0, F1;
                // maps to 0.5 ~ 1 for the first spline
                // TV F0 = curve_func((theta + M_PI * 0.5)/M_PI, curve_idx);
                F((theta + M_PI * 0.5)/M_PI, curve_idx, F0);
                // maps to 0 ~ 0.5 for the second spline
                // TV F1 = curve_func(theta/M_PI, curve_idx+1);
                F(theta/M_PI, curve_idx+1, F1);
                // equation (2)
                TV Ci = std::cos(theta) * std::cos(theta) * F0 + std::sin(theta) * std::sin(theta) * F1;
                points_on_curve.push_back(Ci);
            }
        }
        // second half of the last spline, no interpolation
        for (int vtx = sub_div / 2; vtx < sub_div + 1; vtx++)
        {
            T t = T(vtx) / sub_div;
            TV F1; F(t, curve_data.size() - 1, F1);
            points_on_curve.push_back(F1);
        }

        points = points_on_curve;
    }
    

    void normalizeDataPoints()
    {
        TV min_point = TV::Ones() * 1e10, max_point = TV::Ones() * -1e10;
        for (auto pt : data_points)
        {
            for (int d = 0; d < dim; d++)
            {
                min_point[d] = std::min(min_point[d], pt[d]);
                max_point[d] = std::min(max_point[d], pt[d]);
            }
        }
        T longest_dis = -1000;
        for (int d = 0; d < dim; d++)
            longest_dis = std::max(longest_dis, max_point[d] - min_point[d]);
        for (TV& pt : data_points)
        {
            pt = (pt - min_point) / longest_dis;
        }
            // pt /= longest_dis;   
    }

    void getPosOnCurve(int curve_idx, T t, TV& pos, bool interpolate)
    {
        if (interpolate)
        {
            T theta = t * M_PI * 0.5;
            TV F0, F1;
            F((theta + M_PI * 0.5)/M_PI, curve_idx, F0);
            F(theta/M_PI, curve_idx+1, F1);
            pos = std::cos(theta) * std::cos(theta) * F0 + std::sin(theta) * std::sin(theta) * F1;
        }
        else
        {
            F(t, curve_idx, pos);
        }
    }

    // adapted from line 1143 curvePos
    void F(T t, int idx, TV& pos)
    {
        T tt = curve_data[idx]->limits[0] + t * (curve_data[idx]->limits[1] - curve_data[idx]->limits[0]);
        pos = curve_data[idx]->center + 
                curve_data[idx]->axis1 * std::cos(tt) +
                curve_data[idx]->axis2 * std::sin(tt);
    }

    void generateC2Curves()
    {
        curve_data.clear();
        points_on_curve.clear();
        
        
        for(int i = 1; i < data_points.size() - 1; i++)
        {
            TV center, axis1, axis2; Vector<T, 3> limits;
            hybridInterpolation(i, center, axis1, axis2, limits);
            curve_data.push_back(new CurveData<T, dim>(
                                    center, axis1, axis2, 
                                    Vector<T, 2>(limits[0], limits[2])));
        }
        
        // first half of the first spline, no interpolation
        for (int vtx = 0; vtx < sub_div / 2; vtx++)
        {
            T t = T(vtx) / sub_div;
            TV F0; F(t, 0, F0);
            points_on_curve.push_back(F0);
        }
        // interpolate between two splines
        for (int curve_idx = 0; curve_idx < curve_data.size()-1; curve_idx++)
        {
            for (int vtx = 0; vtx < sub_div / 2; vtx++)
            {
                T t = T(vtx) / sub_div * 2.0;
                T theta = t * M_PI * 0.5;
                TV F0, F1;
                // maps to 0.5 ~ 1 for the first spline
                // TV F0 = curve_func((theta + M_PI * 0.5)/M_PI, curve_idx);
                F((theta + M_PI * 0.5)/M_PI, curve_idx, F0);
                // maps to 0 ~ 0.5 for the second spline
                // TV F1 = curve_func(theta/M_PI, curve_idx+1);
                F(theta/M_PI, curve_idx+1, F1);
                // equation (2)
                TV Ci = std::cos(theta) * std::cos(theta) * F0 + std::sin(theta) * std::sin(theta) * F1;
                points_on_curve.push_back(Ci);
            }
        }
        // second half of the last spline, no interpolation
        for (int vtx = sub_div / 2; vtx < sub_div + 1; vtx++)
        {
            T t = T(vtx) / sub_div;
            TV F1; F(t, curve_data.size() - 1, F1);
            points_on_curve.push_back(F1);
        }
    }
    
    

private:

    // adapted from source code
    void circularInterpolation(int i, TV& center, TV& axis1, TV& axis2, Vector<T, 3>& limits)
    {
        axis1.setZero(); axis2.setZero(); center.setZero(); limits.setZero();

        int j = (i - 1 + data_points.size()) % data_points.size();
        int k = (i + 1) % data_points.size();
        TV vec1 = data_points[i] - data_points[j];
        TV mid1 = data_points[j] + 0.5 * vec1;
        TV vec2 = data_points[k] - data_points[i];
        TV mid2 = data_points[i] + 0.5 * vec2;
        TV dir1, dir2;
        dir1[0] = -vec1[1]; dir1[1] = vec1[0];
        dir2[0] = -vec2[1]; dir2[1] = vec2[0];
        
        T det = dir1[0] * dir2[1] - dir1[1] * dir2[0];

        if (std::abs(det) < 0.001)
        {
            if(vec1[0] * vec1[0] + vec1[1] * vec1[1] >= 0 || data_points.size() <=2 )
            {
                T small_angle = 0.01;
                T s = std::sin(small_angle);
                T l1 = vec1.norm(), l2 = vec2.norm();
                center = data_points[i];
                axis1 = TV::Zero();
                axis2 = vec2 / s;
                limits = Vector<T, 3>(-small_angle * l1 / l2, 0, small_angle);
            }
            else
                det = 0.001;
        }
        T s = ( dir2[1] * (mid2[0] - mid1[0]) + dir2[0] * (mid1[1] - mid2[1]) ) / det;
        center = mid1 + dir1 * s;
        axis1  = data_points[i] - center;
        axis2[0] = -axis1[1]; axis2[1] = axis1[0];
        T len2   = axis1[0]*axis1[0] + axis1[1]*axis1[1];
        TV toPt2  = data_points[k] - center;
        T limit2 = std::atan2( axis2.dot(toPt2), axis1.dot(toPt2) );
        TV toPt1  = data_points[j] - center;
        T limit1 = std::atan2( axis2.dot(toPt1), axis1.dot(toPt1) );

        if ( limit1 * limit2 > 0 ) 
        {
            if ( std::abs(limit1)<std::abs(limit2) ) limit2 += limit2 > 0 ? -T(2) * M_PI : T(2) * M_PI;
            if ( std::abs(limit1)>std::abs(limit2) ) limit1 += limit1 > 0 ? -T(2) * M_PI : T(2) * M_PI;
        }

        limits = Vector<T, 3>(limit1, 0, limit2);
    }

    // adapted from source code
    void ellipticalInterpolation(int i, TV& center, TV& axis1, TV& axis2, Vector<T, 3>& limits)
    {
        axis1.setZero(); axis2.setZero(); center.setZero(); limits.setZero();
        int numIter = 16;
        int j = (i - 1 + data_points.size()) % data_points.size();
        int k = (i + 1) % data_points.size();
        TV vec1 = data_points[j] - data_points[i];
        TV vec2 = data_points[k] - data_points[i];

        if ( data_points.size() <= 2 ) 
        {
            T small_angle = 0.01;
            T s = std::sin(small_angle);
            center = data_points[i];
            axis1 = TV::Zero();
            axis2 = vec2 / s;
            limits = Vector<T, 3>(-small_angle, 0, small_angle);
        }

        T len1 = std::sqrt( vec1[0]*vec1[0] + vec1[1]*vec1[1] );
        T len2 = std::sqrt( vec2[0]*vec2[0] + vec2[1]*vec2[1] );
        T cosa = (vec1[0]*vec2[0] + vec1[1]*vec2[1]) / (len1*len2);
        T maxA = std::acos(cosa);
        T ang  = maxA * 0.5;
        T incA = maxA * 0.25;
        T l1 = len1;
        T l2 = len2;
        if ( len1 < len2 ) { l1=len2; l2=len1; }
        T a, b, c, d;
        for ( int iter=0; iter<numIter; iter++ ) 
        {
            T theta = ang * 0.5;
            a = l1 * std::sin(theta);
            b = l1 * std::cos(theta);
            T beta = maxA - theta;
            c = l2 * std::sin(beta);
            d = l2 * std::cos(beta);
            T v = (1.0-d/b)*(1.0-d/b)+(c*c)/(a*a);	// ellipse equation
            ang += ( v > 1 ) ? incA : -incA;
            incA *= 0.5;
        }

        TV vec, pt2;
        T len;
        if ( len1 < len2 ) 
        {
            vec = vec2;
            len = len2;
            pt2 = data_points[k];
        } 
        else 
        {
            vec = vec1;
            len = len1;
            pt2 = data_points[j];
        }
        TV dir  = vec / len;
        TV perp = TV::Zero();
        perp[0] = -dir[1]; perp[1] = dir[0];
        T cross = vec1[0] * vec2[1] - vec1[1] * vec2[0];
        if ( (len1<len2 && cross>0) || (len1>=len2 && cross<0) ) 
        {
            perp[0] = dir[1]; 
            perp[1] = -dir[0];
        }
            
        T v = b*b/len;
        T h = b*a/len;
        axis1  = -dir * v - perp * h;
        center = data_points[i] - axis1;
        axis2 = pt2 - center;
        T beta   = std::asin(std::min(c/a, T(1)));
        if (len1<len2)
        {
            limits = Vector<T, 3>(-beta, 0, M_PI * 0.5);
        }
        else
        {
            axis2 *= -1;
            limits = Vector<T, 3>(-M_PI * 0.5, 0, beta);
        }
    }

    // adapted from source code
    void hybridInterpolation(int i, TV& center, TV& axis1, TV& axis2, Vector<T, 3>& limits)
    {
        circularInterpolation(i, center, axis1, axis2, limits);
        T lim0 = limits[0];
        T lim2 = limits[2];
        if (lim2 < lim0)
        {
            T tmp = lim0;
            lim0 = lim2;
            lim2 = tmp;
        }
        if (lim0 < -M_PI * 0.5 || lim2 > M_PI * 0.5)
            ellipticalInterpolation(i, center, axis1, axis2, limits);
    }

};



#endif