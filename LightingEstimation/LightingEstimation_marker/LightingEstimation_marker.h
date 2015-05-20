#pragma once
#include <opencv2/opencv.hpp>
#include "nlopt/nlopt.hpp"
/* _LE_DEBUG mode is slower */
#define _LE_DEBUG

class objfunc_data
{
public:
	cv::Mat _intensity;
	std::vector<cv::Point3f> _pts_world;   //the vertex in world coordinate;
	std::vector<cv::Point2f> _pts_img;   //the pixel in image coordinate;
};

typedef class LightingEstimation_marker
{
public:
	static LightingEstimation_marker& getInstance()
	{
		static LightingEstimation_marker instance;
		return instance;
	}
	static void computeHomgraphy(double imgpts[4][2], double objpts[4][2], cv::Mat& outputH);

	void setHomoMatrix(std::vector<cv::Mat> Hs);
	void setInputImg(cv::Mat img);
	double estimate();
	double estimate(cv::Mat img, double imgpts[4][2], double objpts[4][2]);
	double estimate(cv::Mat img, cv::Mat homography);
	void setInitGuess(double ambient, double diffuse, double n_x, double n_y, double n_z, double x, double y, double z);
	void outputData(double output[5]);

private:
	LightingEstimation_marker(void);
	~LightingEstimation_marker(void);
	LightingEstimation_marker(LightingEstimation_marker const&);
	void operator=(LightingEstimation_marker const&);
	void setInitGuess();
	cv::Mat makeShading(cv::Mat src);
	cv::Mat& getShading();

	double static objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data);
	double static constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data);

	std::vector<cv::Mat> _homoMatrix;
	cv::Mat _img;
	nlopt::opt *opt;
	bool set_initGuess;
	cv::Mat _shading;
	double marker_vertex[4][2];
	int nMarkers;

	/* output */
	double _ambient;
	double _diffuse;
	double _normal[3];
	cv::Point3d _lightPos;

} LE_marker;

