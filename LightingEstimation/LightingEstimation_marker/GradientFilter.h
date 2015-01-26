#ifndef GRADIENTFILTER_H
#define GRADIENTFILTER_H
#include "opencv2/opencv.hpp"
#include "nlopt.hpp"

class ObjFunc_data
{
public:
	double inputImg_x;
	double pixelWeight_x;
	double clipped_grad_x;
	cv::Mat patch;
	double lambda;
	int itrCount;
};

class GradientFilter
{
public:
	GradientFilter();
	~GradientFilter();
	bool init(cv::Mat img);
	cv::Mat optimize();
private:
	double static objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data);
	double static constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data);
	cv::Mat static getGradient( cv::Mat src );
	cv::Mat static gradClipping( cv::Mat gradient, double tao=-1.0 );
	cv::Mat getPixelWeight( cv::Mat img );

	double lambda;       // smooth部分的權重
	double tao;          // The threshold of the gradient clipping function
	double stopItrCost;
	int stopMaxItrCount;
	double stopPixelVal;

	cv::Size imgSize;
	cv::Mat m_inputImg;
	cv::Mat m_pixelWeights;
	cv::Mat m_clipped_grad;
	nlopt::opt *opt;
	ObjFunc_data data;
};


#endif