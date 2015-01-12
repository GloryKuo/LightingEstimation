#include <opencv2\opencv.hpp>
#include "LightingEstimation.h"
using namespace cv;

int main(void)
{
	/*Mat src = imread( "../input/input20.jpg", 0 );
    if( !src.data ){
		std::cout<<"Can not load image!"<<std::endl;
        return -1; 
    }
	imshow("source", src);

	LightingEstimation demo;
	Mat res = demo.makeShading(src);
	imshow("shading", res);

    waitKey();*/
	LightingEstimation::Line l;
	double coef[2] = {-3.0, 5.0};
	l.recordPt(Point2i(5, 5), coef, 10, 10);
	l.printLine();


	return 0;
}