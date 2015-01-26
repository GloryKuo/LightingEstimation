#pragma once
#include "LightingEstimation.h"
#include "nlopt.hpp"


typedef class LightingEstimation_marker :
	public LightingEstimation
{
public:
	static LightingEstimation_marker& getInstance()
	{
		static LightingEstimation_marker instance;
		return instance;
	}
	void setHomoMatrix(cv::Mat H);
	void setInputImg(cv::Mat img);
	bool estimate();
	bool estimate(cv::Mat img, cv::Mat homography);


private:
	LightingEstimation_marker(void);
	~LightingEstimation_marker(void);
	LightingEstimation_marker(LightingEstimation_marker const&);
	void operator=(LightingEstimation_marker const&);

	double static objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data);
	double static constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data);

	cv::Mat _homoMatrix;
	cv::Mat _img;
	nlopt::opt *opt;
} LE_marker;

