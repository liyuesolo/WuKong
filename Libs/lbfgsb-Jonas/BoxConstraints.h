#ifndef BOX_CONSTRAINTS_H
#define BOX_CONSTRAINTS_H

#include <Eigen/Core>
#include <vector>
#include <limits>

template<typename Scalar>
class LBFGSCompactB;

namespace BoxConstraints
{

//		boundedBelow && boundedAbove: boundType = 2;
//		boundedBelow && !boundedAbove: boundType = 1;
//		!boundedBelow && boundedAbove: boundType = 3;
//		!boundedBelow && !boundedAbove: boundType = 0;
int computeBoundType(double lb, double ub);

//		boundedBelow && boundedAbove: boundTypes[i] = 2;
//		boundedBelow && !boundedAbove: boundTypes[i] = 1;
//		!boundedBelow && boundedAbove: boundTypes[i] = 3;
//		!boundedBelow && !boundedAbove: boundTypes[i] = 0;
void computeBoundTypes(const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	std::vector<int> &boundTypes);

inline bool isBoundBelow(int boundType)
{
	return boundType == 1 || boundType == 2;
}
inline bool isBoundAbove(int boundType)
{
	return boundType >= 2;
}

void computeIsBound(double epsilon,
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const std::vector<int> &boundTypes,
	const Eigen::VectorXd &x,
	const Eigen::VectorXd &gradient,
	std::vector<bool> &isBound);
//computes isBound, where isBound == 0 means not bound, isBound[i] == 1, means lower bound bound, and isBound[i] == 2 isBound above
void computeIsBound(double epsilon,
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const std::vector<int> &boundTypes,
	const Eigen::VectorXd &x,
	const Eigen::VectorXd &gradient,
	std::vector<int> &isBound);

void projectX(
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const std::vector<int> &boundTypes,
	Eigen::VectorXd &x);

void projectGradient(
	double epsilon,
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const std::vector<int> &boundTypes,
	const Eigen::VectorXd &x,
	const Eigen::VectorXd &gradient,
	Eigen::VectorXd &projGradient);

void projectGradient(
	double epsilon,
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const Eigen::VectorXd &x,
	const Eigen::VectorXd &gradient,
	Eigen::VectorXd &projGradient);


}

#endif