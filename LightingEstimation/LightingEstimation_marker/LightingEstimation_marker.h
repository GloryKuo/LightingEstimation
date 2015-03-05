#pragma once
#include "LightingEstimation.h"
#include "nlopt.hpp"

struct objfunc_data
{
	cv::Mat _intensity;
	double _normal[3];
	std::vector<cv::Point3f> _pts_world;   //the vertex in world coordinate;
	std::vector<cv::Point2f> _pts_img;   //the pixel in image coordinate;
};

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
	double estimate();
	double estimate(cv::Mat img, cv::Mat homography);
	void setInitGuess(double ambient, double diffuse, float x, float y, float z);
	void outputData(double output[5]);

private:
	LightingEstimation_marker(void);
	~LightingEstimation_marker(void);
	LightingEstimation_marker(LightingEstimation_marker const&);
	void operator=(LightingEstimation_marker const&);
	void setInitGuess();

	double static objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data);
	double static constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data);

	cv::Mat _homoMatrix;
	cv::Mat _img;
	nlopt::opt *opt;
	bool set_initGuess;

	/* output */
	double intensity_ambient;
	double intensity_diffuse;
	cv::Point3f light_position;

} LE_marker;

