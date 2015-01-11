#include <opencv2\opencv.hpp>
#pragma once

class LightingEstimation
{
	class Line
	{
	public:
		Line(){}
		Line(cv::Point2i prior, double coef[2], int imgWidth, int imgHeight){  
			/* To do */
			/* 紀錄在圖片中的所有點 */
		}
		bool static checkValid(cv::Point2i p, cv::Size2i imgSize){
			return p.x>=0 && p.y>=0 && p.x<=imgSize.width && p.y<=imgSize.height;
		}
		Line& clone(){
			Line *l = new Line();
			for(int i=0;i<pt.size();i++){
				l->pt.push_back(pt[i]);
			}
			l->coef[0] = coef[0];
			l->coef[1] = coef[1];
			return *l;
		}

		std::vector<cv::Point2i> pt;
		double coef[2];        //直線參數式   x = x0 + coef[0]*t,   y = y0 + coef[1]*t
	};
public:
	LightingEstimation(void);
	~LightingEstimation(void);
	cv::Mat makeShading(cv::Mat src);
	Line detectBiSymmetry(cv::Mat shading, cv::Point2i prior);
private:
	void generateHypotheses( std::vector<LightingEstimation::Line> &lines, cv::Point2i prior);
};

