#ifndef CS2D_POT_H
#define CS2D_POT_H
#include <array>
#include <cmath>
#include <iostream>

using std::pow, std::sqrt, std::abs;

namespace cs2d::pot {
	template <int order>
		void line_length(const double* a, const double* b, double* y) noexcept;

	template <>
		inline void line_length<0>(const double* a, const double* b, double* y) noexcept {
			const double abx = b[0]-a[0], aby = b[1]-a[1];
			*y = sqrt(pow(abx, 2) + pow(aby, 2));
		}
	template <>
		void line_length<1>(const double* a, const double* b, double* y) noexcept;

	template <>
		void line_length<2>(const double* a, const double* b, double* y) noexcept;

	template <int order>
		void triangle_area_signed(const double *a, const double *b, const double *c, double* y) noexcept;
	template <>
		inline void triangle_area_signed<0>(const double* a, const double* b, const double* c, double* y) noexcept {
			const double abx = b[0] - a[0], aby = b[1] - a[1];
			const double acx = c[0] - a[0], acy = c[1] - a[1];
			*y = abx * acy - aby * acx;
		}
	template <>
		void triangle_area_signed<1>(const double *a, const double *b, const double *c, double* y) noexcept;

	template <>
		void triangle_area_signed<2>(const double *a, const double *b, const double *c, double* y) noexcept;

	namespace collision {
		inline double vertex_line_distance(const double* p, const double* a, const double* b) {
			const double abx = b[0]-a[0], aby = b[1]-a[1];
			const double apx = p[0]-a[0], apy = p[1]-a[1];
			const double l_ab = sqrt(pow(abx, 2) + pow(aby, 2));
			const double area = apx*aby - apy*abx;
			return area / l_ab;
		}

		enum VertexEdgeMode {A, B, Line, Null};
		inline VertexEdgeMode vertex_edge_mode (const double* p, const double* a, const double* b, const double d0){
			const double dist_perp = abs(vertex_line_distance(p, a, b));
			if (dist_perp >= d0)
				return Null;

			const double abx = b[0]-a[0], aby = b[1]-a[1];
			const double apx = p[0]-a[0], apy = p[1]-a[1];
			if (apx*abx + apy*aby < 0) {
				if (sqrt(apx*apx+apy*apy) < d0)
					return A;
				return Null;
			}
			const double bpx = p[0]-b[0], bpy = p[1]-b[1];
			if (-bpx*abx - bpy*aby < 0) {
				if (sqrt(bpx*bpx+bpy*bpy) < d0)
					return B;
				return Null;
			}
			return Line;
		}

		template <int order>
			void vertex_line_potential(const double *p, const double *a, const double *b, const double *d0, double* y);
		template <>
			inline void vertex_line_potential<0>(const double *p, const double *a, const double *b, const double *d0, double* y) {
				const double abx = b[0]-a[0], aby = b[1]-a[1];
				const double apx = p[0]-a[0], apy = p[1]-a[1];
				const double l_ab = sqrt(pow(abx, 2) + pow(aby, 2));
				const double area = apx*aby - apy*abx;
				const double dist = abs(area / l_ab);
				const double potential = -pow(dist-*d0, 2) * log(dist/ *d0);
				*y = potential;
			}
		template <>
			void vertex_line_potential<1>(const double *p, const double *a, const double *b, const double *d0, double* y);
		template <>
			void vertex_line_potential<2>(const double *p, const double *a, const double *b, const double *d0, double* y);

		template <int order>
			void vertex_vertex_potential_a(const double *p, const double *a, const double *b, const double* d0, double* y);

		template <>
			inline void vertex_vertex_potential_a<0>(const double *p, const double *a, const double *, const double* d0, double* y) {
				const double dist = sqrt(pow(a[0]-p[0], 2) + pow(a[1]-p[1], 2));
				const double potential = -pow(dist-*d0, 2) * log(dist/ *d0);
				*y = potential;
			}
		template <>
			void vertex_vertex_potential_a<1>(const double *p, const double *a, const double *b, const double* d0, double* y);
		template <>
			void vertex_vertex_potential_a<2>(const double *p, const double *a, const double *b, const double* d0, double* y);

		template <int order>
			void vertex_vertex_potential_b(const double *p, const double *a, const double *b, const double* d0, double* y);
		template <>
			inline void vertex_vertex_potential_b<0>(const double *p, const double *, const double *b, const double* d0, double* y) {
				const double dist = sqrt(pow(b[0]-p[0], 2) + pow(b[1]-p[1], 2));
				const double potential = -pow(dist-*d0, 2) * log(dist/ *d0);
				*y = potential;
			}
		template <>
			void vertex_vertex_potential_b<1>(const double *p, const double *a, const double *b, const double* d0, double* y);
		template <>
			void vertex_vertex_potential_b<2>(const double *p, const double *a, const double *b, const double* d0, double* y);


		template <int order>
			VertexEdgeMode vertex_edge_potential(const double *p, const double *a, const double *b, const double d0, double* y);

		template <>
			inline VertexEdgeMode vertex_edge_potential<0>(const double *p, const double *a, const double *b, const double d0, double* y) {
				const VertexEdgeMode mode = vertex_edge_mode(p, a, b, d0);
				switch(mode) {
					case Null:
						*y = 0;
						break;
					case Line:
						vertex_line_potential<0>(p, a, b, &d0, y);
						if (*y < 0) {
							std::cout << "NEGATIVE POTENTIAL(LINE)!!!";
						}
						break;
					case A:
						vertex_vertex_potential_a<0>(p, a, b, &d0, y);
						if (*y < 0) {
							std::cout << "NEGATIVE POTENTIAL(A)!!!";
						}
						break;
					case B:
						vertex_vertex_potential_b<0>(p, a, b, &d0, y);
						if (*y < 0) {
							std::cout << "NEGATIVE POTENTIAL(B)!!!";
						}
						break;
				}
				return mode;
			}

		template <>
			inline VertexEdgeMode vertex_edge_potential<1>(const double *p, const double *a, const double *b, const double d0, double* y) {
				const VertexEdgeMode mode = vertex_edge_mode(p, a, b, d0);
				switch(mode) {
					case Null:
						for (int i = 0; i < 6; ++i)
							y[i] = 0;
						break;
					case Line:
						vertex_line_potential<1>(p, a, b, &d0, y);
						break;
					case A:
						vertex_vertex_potential_a<1>(p, a, b, &d0, y);
						break;
					case B:
						vertex_vertex_potential_b<1>(p, a, b, &d0, y);
						break;
				}
				return mode;
			}

		template <>
			inline VertexEdgeMode vertex_edge_potential<2>(const double *p, const double *a, const double *b, const double d0, double* y) {
				const VertexEdgeMode mode = vertex_edge_mode(p, a, b, d0);
				switch(mode) {
					case Null:
						for (int i = 0; i < 6*6; ++i)
							y[i] = 0;
						break;
					case Line:
						vertex_line_potential<2>(p, a, b, &d0, y);
						break;
					case A:
						vertex_vertex_potential_a<2>(p, a, b, &d0, y);
						break;
					case B:
						vertex_vertex_potential_b<2>(p, a, b, &d0, y);
						break;
				}
				return mode;
			}
	}

	namespace adhesion {
		template <typename T>
		T d_mid(const T* a, const T* b, const T* c, const T* d) {
			const T dx = 0.5*(*a + *b - *c - *d);
			const T dy = 0.5*(*(a+1) + *(b+1) - *(c+1) - *(d+1));
			return sqrt((dx*dx) + (dy*dy));
		}

		template <typename T>
		T l_mid(const T* a, const T* b, const T* c, const T* d) {
			const T dx = 0.5*(*a + *c - *b - *d);
			const T dy = 0.5*(*(a+1) + *(c+1) - *(b+1) - *(d+1));
			return sqrt((dx*dx) + (dy*dy));
		}

		template <typename T>
		T f(const T d, const T d0) {
			const T d_sq = d*d, d0_sq = d0*d0;
			const T v = 3*d_sq*d_sq -8*d0*d*d_sq + 6*d0_sq*d_sq - d0_sq*d0_sq; 
			return v/12;
		}

		inline bool active(const double* a, const double* b, const double* c, const double* d, const double d0) {
			const double dm = d_mid(a, b, c, d);
			if (dm > d0) return false;
			const double x = (*b-*a)*(*d-*c);
			const double y = (*(b+1)-*(a+1))*(*(d+1)-*(c+1));
			if (x+y <= 0) return false;
			return true;
		}

		template <typename T>
		T limit_l(const T l, const T logi_k) {
			return 2 /(1+exp(-logi_k * l))-1;
		}

		template <int order>
			void potential(const double* a, const double* b, const double* c, const double* d, const double d0, const double logi_k, double *y);
		template <>
			inline void potential<0>(const double* a, const double* b, const double* c, const double* d, const double d0, const double logi_k, double *y) {
				const double dm = d_mid(a, b, c, d);
				const double lm = l_mid(a, b, c, d);
				*y = f(dm, d0) * limit_l(lm, logi_k);
			}
		template <>
			void potential<1>(const double* a, const double* b, const double* c, const double* d, const double d0, const double logi_k, double *y);
		template <>
			void potential<2>(const double* a, const double* b, const double* c, const double* d, const double d0, const double logi_k, double *y);
	}
}


namespace cs2d::pen {
	// penalties
	template <int order>
		inline double square(const double x, const double x0);
	template <>
		inline double square<0>(const double x, const double x0) {
			return std::pow(x-x0, 2);
		}
	template <>
		inline double square<1>(const double x, const double x0) {
			return 2 * (x-x0);
		}
	template <>
		inline double square<2>(const double, const double) {
			return 2;
		}
}
#endif
