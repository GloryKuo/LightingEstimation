#include <opencv2\opencv.hpp>
#include "LightingEstimation.h"
using namespace cv;

int main(void)
{
	Mat src = imread( "../input/input.jpg", 1);
    if( !src.data ){
		std::cout<<"Can not load image!"<<std::endl;
        return -1; 
    }
	imshow("source", src);
	

	LightingEstimation le;
	LightingEstimation::Line l = le.detectBiSymmetry(src, Point2i(228,194));
	l.printLine();

	std::vector<LightingEstimation::Line> lv;
	lv.push_back(l);
	
	Mat show;
	LightingEstimation::drawLines(le.getShading(), show, lv);
	imshow("symmetry axis", show);

    waitKey();
	return 0;
}