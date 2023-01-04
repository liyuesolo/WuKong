// Collision detection
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>

using Eigen::Dynamic, Eigen::Matrix, Eigen::Matrix2d, Eigen::RowVector2d, Eigen::Vector2d, Eigen::VectorXd;
using Eigen::Array, Eigen::Array2d, Eigen::ArrayXd;

using VertexMatrix2d = Matrix<double, Dynamic, 2>;

// for a given edge and a set of vertices, check if a collision occurs during interpolation and compute the time of collision
// returns the time of the first-occurring collision in the interval t in [0...1) and 1 if none occurs

inline bool verifyCollision(
		const Vector2d a0, const Vector2d b0, const Vector2d a1, const Vector2d b1,
		const Vector2d c0, const Vector2d c1, double t) {
	//std::cout << "Verifying collision at time t=" << t << std::endl;
	const Vector2d at = a0 * (1-t) + a1 * t;
	const Vector2d bt = b0 * (1-t) + b1 * t;
	const Vector2d ct = c0 * (1-t) + c1 * t;
	//std::cout << "at=" << at.transpose() << std::endl;
	//std::cout << "bt=" << bt.transpose() << std::endl;
	//std::cout << "ct=" << ct.transpose() << std::endl;
	if ((ct-at).dot(bt-at) > 0 && (ct-bt).dot(at-bt) > 0) {
		//std::cout << "Collision could be verified." << std::endl;
		return true;
	}
	return false;
}

inline bool verifyCollisionConservative(
		const Vector2d a0, const Vector2d b0, const Vector2d a1, const Vector2d b1,
		const Vector2d c0, const Vector2d c1, double t, double eta) {
	//FIXME: modify to account for eta
	//std::cout << "Verifying collision at time t=" << t << std::endl;
	const Vector2d at = a0 * (1-t) + a1 * t;
	const Vector2d bt = b0 * (1-t) + b1 * t;
	const Vector2d ct = c0 * (1-t) + c1 * t;
	//std::cout << "at=" << at.transpose() << std::endl;
	//std::cout << "bt=" << bt.transpose() << std::endl;
	//std::cout << "ct=" << ct.transpose() << std::endl;
	const double thresh = (bt-at).norm()*eta;
	if ((ct-at).dot(bt-at) > -thresh && (ct-bt).dot(at-bt) > -thresh) {
		//std::cout << "Collision could be verified." << std::endl;
		return true;
	}
	return false;
}

inline double collisionTime(
		const RowVector2d a0, const RowVector2d b0, const RowVector2d a1, const RowVector2d b1,
		const VertexMatrix2d& c0, const VertexMatrix2d& c1
		) {

	// a is always (0, 0) and b is (1, 0), with the lines being transformed to fit this frame of reference
	// a collision occurs if the line from the start vertex to the end vertex intersects the interval (0, 1)
	// we detect this by

	using VertArray = Array<double, Dynamic, 2>;

	const VertArray m = (c1-c0).rowwise()+(a0-a1);
	const VertArray n = c0.rowwise()-a0;
	const Array2d   o = b1-b0 + a0-a1;
	const Array2d   p = b0-a0;

	const ArrayXd A = m.col(0)*o[1] - m.col(1)*o[0];
	const ArrayXd B = m.col(0)*p[1] + n.col(0)*o[1] - m.col(1)*p[0] - n.col(1)*o[0];
	const ArrayXd C = n.col(0)*p[1] - n.col(1)*p[0];

	const ArrayXd disc = B*B-4*A*C;

	/*
	std::cout << "c1-c0:" << std::endl <<c1-c0 << std::endl;
	std::cout << "a0-a1:" << std::endl <<a0-a1 << std::endl;
	std::cout << "m:" << std::endl << m << std::endl;
	std::cout << "n:" << n << std::endl;
	std::cout << "o:" << std::endl << o << std::endl;
	std::cout << "p:" << p << std::endl;
	std::cout << "disc:" << std::endl << disc << std::endl;
	*/

	double t_min = 1;
	for (int i = 0; i < c0.rows(); ++i) {
		if (disc[i] >= 0) {
			if (std::abs(A[i]) == 0) {
				const double t = -C[i]/B[i];
				if (t >= 0 && t < t_min) {
					//std::cout << "Collision time updated from t=" << t_min << " to ta=" << ta << std::endl;
					if (verifyCollision(a0, b0, a1, b1, c0.row(i), c1.row(i), t))
						t_min = t;
				}
				continue;
			}
			const double disc_rt = std::sqrt(disc[i]); 
			const double ta = 0.5*(-B[i]-disc_rt)/A[i];
			const double tb = 0.5*(-B[i]+disc_rt)/A[i];
			if (ta >= 0 && ta < t_min) {
				//std::cout << "Collision time updated from t=" << t_min << " to ta=" << ta << std::endl;
				if (verifyCollision(a0, b0, a1, b1, c0.row(i), c1.row(i), ta))
					t_min = ta;
			}
			if (tb >= 0 && tb < t_min) {
				//std::cout << "Collision time updated from t=" << t_min << " to tb=" << tb << std::endl;
				if (verifyCollision(a0, b0, a1, b1, c0.row(i), c1.row(i), tb))
					t_min = tb;
			}
		}
	}
	return t_min;
}

inline double collisionTimeConservative2(
		const RowVector2d a0, const RowVector2d b0, const RowVector2d a1, const RowVector2d b1,
		const VertexMatrix2d& c0, const VertexMatrix2d& c1, double eta
		) {

	using VertArray = Array<double, Dynamic, 2>;

	const RowVector2d l0 = b0-a0;
	const RowVector2d dl = (b1-a1-l0);

	const double l1abs = (b1-a1).norm();
	const double l0abs = (b0-a0).norm();

	const VertArray k0 = c0.rowwise() - a0;
	const VertArray dk = ((c1.rowwise()-a1).array()-k0.array());

	const ArrayXd A = dl[0] * dk.col(1) - dl[1] * dk.col(0);
	const ArrayXd B1 = dl[0] * k0.col(1) - dl[1] * k0.col(0);
	const ArrayXd B2 = l0[0] * dk.col(1) - l0[1] * dk.col(0);
	const double B3 = eta * (l1abs-l0abs);
	const ArrayXd C1 = l0[0] * k0.col(1) - l0[1] * k0.col(0);
	const double C2 = eta * l0abs;

	double t_min = 1;
	for (int i = 0; i < c0.rows(); ++i) {
		const double e = C1[i] -C2 < 0 ? 1 : -1;
			
		const double B = B1[i] + B2[i] + e*B3;
		const double C = C1[i] + e*C2;
		const double disc = B * B - 4 * A[i] * C;
		if (disc >= 0) {
			if (std::abs(A[i]) == 0) {
				const double t = -C/B;
				if (t >= 0 && t < t_min) {
					//std::cout << "Collision time updated from t=" << t_min << " to ta=" << ta << std::endl;
					if (verifyCollisionConservative(a0, b0, a1, b1, c0.row(i), c1.row(i), t, eta))
						t_min = t;
				}
				continue;
			}
			const double disc_rt = std::sqrt(disc); 
			const double ta = 0.5*(-B-disc_rt)/A[i];
			const double tb = 0.5*(-B+disc_rt)/A[i];
			if (ta >= 0 && ta < t_min) {
				//std::cout << "Collision time updated from t=" << t_min << " to ta=" << ta << std::endl;
				if (verifyCollisionConservative(a0, b0, a1, b1, c0.row(i), c1.row(i), ta, eta))
					t_min = ta;
			}
			if (tb >= 0 && tb < t_min) {
				//std::cout << "Collision time updated from t=" << t_min << " to tb=" << tb << std::endl;
				if (verifyCollisionConservative(a0, b0, a1, b1, c0.row(i), c1.row(i), tb, eta))
					t_min = tb;
			}
		}
	}
	return t_min;
}
inline double collisionTimeConservative(
		const RowVector2d a0, const RowVector2d b0, const RowVector2d a1, const RowVector2d b1,
		const VertexMatrix2d& c0, const VertexMatrix2d& c1, double eta
		) {

	using VertArray = Array<double, Dynamic, 2>;

	const RowVector2d l0 = b0-a0;
	const RowVector2d dl = (b1-a1-l0);

	const double l1abs = (b1-a1).norm();
	const double l0abs = (b0-a0).norm();

	const VertArray k0 = c0.rowwise() - a0;
	const VertArray dk = ((c1.rowwise()-a1).array()-k0.array());

	const ArrayXd A = dl[0] * dk.col(1) - dl[1] * dk.col(0);
	const ArrayXd B1 = dl[0] * k0.col(1) - dl[1] * k0.col(0);
	const ArrayXd B2 = l0[0] * dk.col(1) - l0[1] * dk.col(0);
	const double B3 = eta * (l1abs-l0abs);
	const ArrayXd B = B1 + B2 - B3;
	const ArrayXd C = l0[0] * k0.col(1) - l0[1] * k0.col(0) - eta * l0abs;

	const ArrayXd disc = B*B-4*A*C;

	double t_min = 1;
	for (int i = 0; i < c0.rows(); ++i) {
		if (disc[i] >= 0) {
			if (std::abs(A[i]) == 0) {
				const double t = -C[i]/B[i];
				if (t >= 0 && t < t_min) {
					//std::cout << "Collision time updated from t=" << t_min << " to ta=" << ta << std::endl;
					if (verifyCollisionConservative(a0, b0, a1, b1, c0.row(i), c1.row(i), t, eta))
						t_min = t;
				}
				continue;
			}
			const double disc_rt = std::sqrt(disc[i]); 
			const double ta = 0.5*(-B[i]-disc_rt)/A[i];
			const double tb = 0.5*(-B[i]+disc_rt)/A[i];
			if (ta >= 0 && ta < t_min) {
				//std::cout << "Collision time updated from t=" << t_min << " to ta=" << ta << std::endl;
				if (verifyCollisionConservative(a0, b0, a1, b1, c0.row(i), c1.row(i), ta, eta))
					t_min = ta;
			}
			if (tb >= 0 && tb < t_min) {
				//std::cout << "Collision time updated from t=" << t_min << " to tb=" << tb << std::endl;
				if (verifyCollisionConservative(a0, b0, a1, b1, c0.row(i), c1.row(i), tb, eta))
					t_min = tb;
			}
		}
	}
	return t_min;
}
