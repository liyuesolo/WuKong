#ifndef UTIL_H
#define UTIL_H

#include "VecMatDef.h"

#include <iostream>

struct Capsule
{
    using TV = Vector<T, 3>;
    TV from, to;
    double radius;
    Capsule(const TV& _from, const TV& _to, T _radius) : 
        from(_from), to(_to), radius(_radius) {}
};

Vector<double,  3> closestPointOnLineSegment(const Vector<double,  3>& A, 
    const Vector<double,  3>& B, Vector<double, 3>& Point)
{
  Vector<T,  3> AB = B - A;
  double t = (Point - A).dot(AB) / AB.dot(AB);
  return A + std::min(std::max(t, 0.0), 1.0) * AB;
}


bool capsuleCapsuleIntersect3D(const Capsule& c0, const Capsule& c1)
{
    using TV = Vector<T, 3>;
    // capsule A:
    TV a_Normal = (c0.from - c0.to).normalized();
    TV a_LineEndOffset = a_Normal * c0.radius; 
    TV a_A = c0.to + a_LineEndOffset; 
    TV a_B = c0.from - a_LineEndOffset;
    
    // capsule B:
    TV b_Normal = (c1.from - c1.to).normalized();
    TV b_LineEndOffset = b_Normal * c1.radius; 
    TV b_A = c1.to + b_LineEndOffset; 
    TV b_B = c1.from - b_LineEndOffset;
    
    // vectors between line endpoints:
    TV v0 = b_A - a_A; 
    TV v1 = b_B - a_A; 
    TV v2 = b_A - a_B; 
    TV v3 = b_B - a_B;
    
    // squared distances:
    double d0 = v0.dot(v0), d1 = v1.dot(v1), d2 = v2.dot(v2), d3 = v3.dot(v3);
    
    // select best potential endpoint on capsule A:
    TV bestA;
    if (d2 < d0 || d2 < d1 || d3 < d0 || d3 < d1)
        bestA = a_B;
    else
        bestA = a_A;
    
    
    // select point on capsule B line segment nearest to best potential endpoint on A capsule:
    TV bestB = closestPointOnLineSegment(b_A, b_B, bestA);
    
    // now do the same for capsule A segment:
    bestA = closestPointOnLineSegment(a_A, a_B, bestB);

    TV penetration_normal = bestA - bestB;
    double len = penetration_normal.norm();
    penetration_normal /= len;  // normalize
    double penetration_depth = c0.radius + c1.radius - len;
    std::cout << len << " " << penetration_depth << std::endl;

    return penetration_depth > 0;
}

//https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

bool lineSegementsIntersect3D(const Vector<double, 3>& p, const Vector<double, 3>& p2, 
    const Vector<double, 3>& q, const Vector<double, 3>& q2, 
    Vector<double, 3>& intersection, bool considerCollinearOverlapAsIntersect = true)
{
  using TV = Vector<double, 3>;

  intersection = TV::Zero();

  TV r = p2 - p;
  TV s = q2 - q;
  TV qmp = q - p;

  TV rxs = r.cross(s);
  TV qpxr = qmp.cross(r);

  if (rxs.norm() < 1e-6 && qpxr.norm() < 1e-6)
  {
      // 1. If either  0 <= (q - p) * r <= r * r or 0 <= (p - q) * s <= * s
      // then the two lines are overlapping,
      if (considerCollinearOverlapAsIntersect)
          if ((0 <= (q - p).dot(r) && (q - p).dot(r) <= r.dot(r)) || (0 <= (p - q).dot(s) && (p - q).dot(s) <= s.dot(s)))
              return true;

      // 2. If neither 0 <= (q - p) * r = r * r nor 0 <= (p - q) * s <= s * s
      // then the two lines are collinear but disjoint.
      // No need to implement this expression, as it follows from the expression above.
      return false;
  }

  // 3. If r x s = 0 and (q - p) x r != 0, then the two lines are parallel and non-intersecting.
  if (rxs.norm() < 1e-6 && qpxr.norm() > 1e-6)
      return false;

  // t = (q - p) x s / (r x s)
  T t = qmp.cross(s).norm() / rxs.norm();

  // u = (q - p) x r / (r x s)

  T u = qmp.cross(r).norm() / rxs.norm();

  // 4. If r x s != 0 and 0 <= t <= 1 and 0 <= u <= 1
  // the two line segments meet at the point p + t r = q + u s.
  if (rxs.norm() > 1e-6 && (0 <= t && t <= 1) && (0 <= u && u <= 1))
  {
      // We can calculate the intersection point using either t or u.
      intersection = p + t*r;
      // An intersection was found.
      return true;
  }

  // 5. Otherwise, the two line segments are not parallel but do not intersect.
  return false;
}

#endif