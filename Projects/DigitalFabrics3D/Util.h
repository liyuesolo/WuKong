#ifndef UTIL_H
#define UTIL_H

#include "VecMatDef.h"

template<class T>
inline T signedAngle(const Vector<T, 3>& u, const Vector<T, 3>& v, const Vector<T, 3>& n) {
  Vector<T, 3> w = u.cross(v);
  T angle = std::atan2(w.norm(), u.dot(v));
  if (n.dot(w) < 0) return -angle;
  return angle;
}

template<class T, int dim>
inline bool colinear(Vector<T, dim> a, Vector<T, dim> b)
{
    if((a-b).norm()<1e-2)
		return true;
	if((a-b).norm()>1.99)
		return true;
	return false;
}

template <class T>
inline void rotateAxisAngle(Vector<T, 3>& v,
                            const Vector<T, 3>& z,
                            const T theta) 
{
  
  if (theta == 0) return;

  T c = cos(theta);
  T s = sin(theta);

  v = c * v + s * z.cross(v) + z.dot(v) * (1.0 - c) * z;
}

template<class T>
Vector<T, 3> parallelTransport(const Vector<T, 3>& u, const Vector<T, 3>& t0, const Vector<T, 3>& t1) 
{
  
    Vector<T, 3> b = t0.cross(t1);


    if(b.norm() < std::numeric_limits<T>::epsilon())
        return u;

    b.normalize();

    Vector<T, 3> n0 = t0.cross(b).normalized();
    Vector<T, 3> n1 = t1.cross(b).normalized();

    return u.dot(t0.normalized()) * t1.normalized() + u.dot(n0) * n1 +
            u.dot(b) * b;
}

template<class T>
Vector<T, 3> parallelTransportOrthonormalVector(const Vector<T, 3>& u, const Vector<T, 3>& t0, const Vector<T, 3>& t1) {
  
    Vector<T, 3> b = t0.cross(t1);


    if(b.norm() < std::numeric_limits<T>::epsilon())
        return u;

    b.normalize();

    Vector<T, 3> n0 = t0.cross(b);
    Vector<T, 3> n1 = t1.cross(b);

    return u.dot(n0) * n1 + u.dot(b) * b;
}

#endif