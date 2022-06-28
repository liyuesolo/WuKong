#ifndef LBFGSB_H
#define LBFGSB_H

#include <Eigen/Core>
#include <vector>
#include <limits>

class LBFGSB
{
public:
	enum State {
		INITIAL,
		STARTED,
		FAILED,
		CONVERGED
	};

	LBFGSB();

	void setPrintLevel(int printLevel)
	{
		m_printLevel = printLevel;
	}

	void setHistorySize(int historySize)
	{
		m_historySize = historySize;
	}
	int getHistorySize() const
	{
		return m_historySize;
	}

	void setX(const Eigen::VectorXd &x);
	const Eigen::VectorXd& getX() const { return m_x; }

	void setBounds(const Eigen::VectorXd &lowerBounds, const Eigen::VectorXd &upperBounds);

	void setObjective(std::function<double(const Eigen::VectorXd&x, Eigen::VectorXd &grad)> objectiveFunctor)
	{
		m_objectiveFunctor = objectiveFunctor;
	}

	void setGradientThreshold(double tol)
	{
		m_projectedGradientTolerance = tol;
	}

	double getGradientThreshold() const
	{
		return m_projectedGradientTolerance;
	}

	void setRelativeReductionOfObjectiveFactor(double factor)
	{
		m_accuracyFactor = factor;
	}

	void solve();

	void takeStep();

	void reset()
	{
		m_state = INITIAL;
	}

	State state() const
	{
		return m_state;
	}

	Eigen::VectorXd computeProjectedGradient();
private:
	Eigen::VectorXd	m_x;

	int m_historySize;

	Eigen::VectorXd m_lowerBounds, m_upperBounds;
	std::vector<int> m_boundTypes;

	std::vector<double> wa;
	std::vector<int> iwa;

	double m_objectiveValue;
	Eigen::VectorXd m_gradient;
	std::string m_task;

	std::vector<char> csave;

	bool lsave[4];
	std::vector<int> isave;
	std::vector<double> dsave;
	int m_printLevel;

	State m_state;

	std::function<double(const Eigen::VectorXd&x, Eigen::VectorXd &grad)> m_objectiveFunctor;

	double m_accuracyFactor; // 1.d + 12 for low accuracy; 1.d+7 for moderate accuracy; 1.d+1 for extremely  high accuracy.
	double m_projectedGradientTolerance; // stopping criterion on the maximum entry of the projected gradient
};

#endif