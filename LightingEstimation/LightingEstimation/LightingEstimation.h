#include <opencv2\opencv.hpp>
#pragma once

class LightingEstimation
{
public:
	class Line
	{
	public:
		Line(){}
		void recordValidPts(int imgWidth, int imgHeight);
		bool static checkValidPt(cv::Point2i p, cv::Size2i imgSize);
		Line& clone();
		void printLine();

		std::vector<cv::Point2i> pts;
		cv::Point2i _prior;
		double _coef[2];        //直線參數式   x = x0 + coef[0]*t,   y = y0 + coef[1]*t
	};

	LightingEstimation(void);
	~LightingEstimation(void);
	cv::Mat makeShading(cv::Mat src);
	Line& detectBiSymmetry(cv::Mat src, cv::Point2i prior);
	void static generateHypotheses( std::vector<LightingEstimation::Line> &lines, cv::Point2i prior, int imgWidth, int imgHeight);
	void static drawLines( cv::Mat src, cv::Mat &des, std::vector<LightingEstimation::Line> lines);
	cv::Mat& getShading();
private:
	cv::Mat _shading;
};

